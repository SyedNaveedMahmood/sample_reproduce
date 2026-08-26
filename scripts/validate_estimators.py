"""Run Task-12 estimator state machines on real unchanged-theta CoOp gradients."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.estimators import EMAEstimator, ExactEstimator, PeriodicEstimator
from sample_fg.full_gradient import (
    FullGradientService,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import RNGSnapshot, capture_rng_state


TASK11_SHA = "6ec3028f217d5ff010952cafb72befaaa8ba2f18"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
EMA_LAMBDA = 0.15


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _rng_equal(left: RNGSnapshot, right: RNGSnapshot) -> bool:
    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(
            torch.equal(a, b)
            for a, b in zip(left.torch_cuda_states, right.torch_cuda_states)
        )
        and len(left.explicit_generators) == len(right.explicit_generators)
        and all(
            a.generator is b.generator
            and a.device == b.device
            and torch.equal(a.state, b.state)
            for a, b in zip(left.explicit_generators, right.explicit_generators)
        )
    )


def _batch_gradient(trainer, model, index, batch) -> tuple[float, GradientState]:
    image, label = trainer.parse_batch_train(batch)
    loss = F.cross_entropy(model(image), label)
    gradients = torch.autograd.grad(
        loss,
        index.parameters,
        create_graph=False,
        retain_graph=False,
    )
    state = GradientState.from_tensors(index, gradients)
    if not state.is_finite():
        raise FloatingPointError("Real mini-batch fixture gradient is nonfinite")
    return float(loss.detach().item()), state


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK11_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-11 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Pinned Dassl SHA changed")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl worktree is dirty")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-12 real integration requires CUDA")

    data_root = Path(args.root).resolve(strict=True)
    manifest_root = Path(args.manifest_root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output = Path(args.output).resolve()
    cfg = build_smoke_cfg(
        REPO_ROOT,
        data_root,
        output.parent / "task12_runtime",
        "base",
    )
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    loaded = load_dataset(data_root, DATASET_SPECS["dtd"])
    source = load_full_gradient_source(
        loaded,
        manifest_root / "dtd" / "shots_4" / "seed_1" / "data_manifest.json",
    )

    normal_iterator = iter(trainer.train_loader_x)
    loss0, g0 = _batch_gradient(trainer, model, index, next(normal_iterator))
    loss1, g1 = _batch_gradient(trainer, model, index, next(normal_iterator))

    exact_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    periodic_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    exact_service = FullGradientService(
        model=model,
        param_index=index,
        loader=exact_loader,
        precision_controller=PrecisionController("fp32"),
        protocol_seed=1,
        dataset="dtd",
        shots=4,
        config_hash="task12-real-estimator-v1",
    )
    periodic_service = FullGradientService(
        model=model,
        param_index=index,
        loader=periodic_loader,
        precision_controller=PrecisionController("fp32"),
        protocol_seed=1,
        dataset="dtd",
        shots=4,
        config_hash="task12-real-estimator-v1",
    )

    for entry, component in zip(index, g0):
        entry.parameter.grad = component.to(dtype=entry.parameter.dtype).clone()
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers_before = {name: b.detach().clone() for name, b in model.named_buffers()}
    grads_before = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    frozen_before = hash_frozen_parameters(model)
    optimizer_before = copy.deepcopy(trainer.optim.state_dict())
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    modes_before = tuple(module.training for module in model.modules())
    rng_before = capture_rng_state((exact_loader.generator, periodic_loader.generator))

    ema = EMAEstimator(index, ema_lambda=EMA_LAMBDA)
    exact = ExactEstimator(index, full_gradient_service=exact_service)
    periodic = PeriodicEstimator(
        index,
        ema_lambda=EMA_LAMBDA,
        refresh_k_steps=2,
        full_gradient_service=periodic_service,
    )
    with count_optimizer_steps(trainer.optim) as counter:
        ema0 = ema.global_direction(batch_grad=g0, optimizer_step=0)
        ema1 = ema.global_direction(batch_grad=g1, optimizer_step=1)
        exact0 = exact.global_direction(batch_grad=g0, optimizer_step=0)
        periodic0 = periodic.global_direction(batch_grad=g0, optimizer_step=0)
        periodic1 = periodic.global_direction(batch_grad=g1, optimizer_step=1)
        periodic2 = periodic.global_direction(batch_grad=g0, optimizer_step=2)
    if counter["count"] != 0:
        raise AssertionError("Estimator validation called optimizer.step")

    torch.testing.assert_close(
        ema0.active_global_estimate.components,
        g0.scale(1.0 - EMA_LAMBDA).components,
        rtol=1e-6,
        atol=1e-7,
    )
    if not torch.equal(
        periodic0.active_global_estimate[0], periodic0.exact_reference[0]
    ) or not torch.equal(
        periodic2.active_global_estimate[0], periodic2.exact_reference[0]
    ):
        raise AssertionError("Real periodic refresh was not a hard exact reset")
    if (periodic0.age_steps, periodic1.age_steps, periodic2.age_steps) != (0, 1, 0):
        raise AssertionError("Real periodic K=2 age sequence differs from 0,1,0")
    if ema.exact_query_count != 0 or exact.exact_query_count != 1:
        raise AssertionError("EMA/exact service call count differs")
    if periodic.exact_query_count != 2:
        raise AssertionError("Periodic exact service call count differs")

    # Validate versioned state payloads on the real ParamIndex without wiring
    # any checkpoint file or trainer resume behavior.
    ema_restored = EMAEstimator(index, ema_lambda=EMA_LAMBDA)
    ema_restored.load_state_dict(copy.deepcopy(ema.state_dict()))
    exact_restored = ExactEstimator(index, full_gradient_service=exact_service)
    exact_restored.load_state_dict(copy.deepcopy(exact.state_dict()))
    periodic_restored = PeriodicEstimator(
        index,
        ema_lambda=EMA_LAMBDA,
        refresh_k_steps=2,
        full_gradient_service=periodic_service,
    )
    periodic_restored.load_state_dict(copy.deepcopy(periodic.state_dict()))
    for left, right in zip(ema.active_state, ema_restored.active_state):
        if not torch.equal(left, right):
            raise AssertionError("Real EMA serialization roundtrip differs")
    for left, right in zip(periodic.active_state, periodic_restored.active_state):
        if not torch.equal(left, right):
            raise AssertionError("Real periodic serialization roundtrip differs")

    rng_after = capture_rng_state((exact_loader.generator, periodic_loader.generator))
    if not _rng_equal(rng_before, rng_after):
        raise AssertionError("Estimator/full-query path changed normal RNG")
    if any(
        not torch.equal(parameters_before[name], parameter)
        for name, parameter in model.named_parameters()
    ):
        raise AssertionError("Estimator path changed model parameters")
    if any(
        not torch.equal(buffers_before[name], buffer)
        for name, buffer in model.named_buffers()
    ):
        raise AssertionError("Estimator path changed model buffers")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Estimator path changed frozen CLIP")
    for name, parameter in model.named_parameters():
        expected = grads_before[name]
        if expected is None and parameter.grad is not None:
            raise AssertionError(f"Estimator path populated .grad for {name}")
        if expected is not None and (
            parameter.grad is None or not torch.equal(expected, parameter.grad)
        ):
            raise AssertionError(f"Estimator path changed .grad for {name}")
    if not _nested_equal(optimizer_before, trainer.optim.state_dict()):
        raise AssertionError("Estimator path changed optimizer")
    if not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError("Estimator path changed scheduler")
    if modes_before != tuple(module.training for module in model.modules()):
        raise AssertionError("Estimator path changed model modes")

    payload = {
        "schema_version": "sample_fg.task12_estimators.v1",
        "status": "PASS",
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "source_count": len(source),
        "param_index_fingerprint": index.fingerprint,
        "logical_optimizer_steps_only": True,
        "actual_optimizer_steps": 0,
        "mini_batch_fixtures": [
            {"logical_step": 0, "loss": loss0, "gradient_norm": float(g0.norm().item())},
            {"logical_step": 1, "loss": loss1, "gradient_norm": float(g1.norm().item())},
        ],
        "ema": {
            "lambda": EMA_LAMBDA,
            "step0_norm": float(ema0.active_global_estimate.norm().item()),
            "step1_norm": float(ema1.active_global_estimate.norm().item()),
            "step0_equals_0_85_g0": True,
            "exact_query_count": ema.exact_query_count,
            "serialization_roundtrip": True,
        },
        "exact": {
            "step0_norm": float(exact0.active_global_estimate.norm().item()),
            "active_equals_exact_reference": all(
                torch.equal(a, b)
                for a, b in zip(
                    exact0.active_global_estimate, exact0.exact_reference
                )
            ),
            "exact_query_count": exact.exact_query_count,
            "purpose": exact0.full_gradient_metadata.seed.purpose,
            "seed_digest": exact0.full_gradient_metadata.seed.sha256,
            "serialization_roundtrip": exact_restored.exact_query_count == 1,
        },
        "periodic_k2": {
            "refreshed": [periodic0.refreshed, periodic1.refreshed, periodic2.refreshed],
            "ages": [periodic0.age_steps, periodic1.age_steps, periodic2.age_steps],
            "last_refresh_steps": [
                periodic0.last_refresh_step,
                periodic1.last_refresh_step,
                periodic2.last_refresh_step,
            ],
            "norms": [
                float(periodic0.active_global_estimate.norm().item()),
                float(periodic1.active_global_estimate.norm().item()),
                float(periodic2.active_global_estimate.norm().item()),
            ],
            "hard_reset_at_steps": [0, 2],
            "no_same_step_ema_mix": True,
            "exact_query_count": periodic.exact_query_count,
            "query_purposes": [
                periodic0.full_gradient_metadata.seed.purpose,
                periodic2.full_gradient_metadata.seed.purpose,
            ],
            "serialization_roundtrip": True,
        },
        "purity": {
            "model_parameters": "unchanged",
            "model_buffers": "unchanged",
            "live_grads": "unchanged",
            "optimizer": "unchanged",
            "scheduler": "unchanged",
            "modes": "unchanged",
            "rng": "unchanged",
        },
        "serialization_scope": "helpers_only_no_checkpoint_integration",
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
