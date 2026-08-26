"""Run Task-11 full-gradient purity checks on the real DTD CoOp path."""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from sample_fg.full_gradient import (
    FullGradientService,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import RNGSnapshot, capture_rng_state, restore_rng_state


TASK10_SHA = "9187f8bfa33bff8df5dabd194be4921e123d2bb1"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


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


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def _normal_batch_result(trainer, model, index, batch, dataset_root: Path):
    image, label = trainer.parse_batch_train(batch)
    loss = F.cross_entropy(model(image), label)
    gradient = torch.autograd.grad(
        loss,
        index.parameters,
        create_graph=False,
        retain_graph=False,
    )
    return {
        "sample_ids": tuple(
            Path(str(value)).resolve(strict=True).relative_to(dataset_root).as_posix()
            for value in batch["impath"]
        ),
        "indices": tuple(int(value) for value in batch["index"].tolist()),
        "image_sha256": _tensor_hash(batch["img"]),
        "label_sha256": _tensor_hash(batch["label"]),
        "loss": float(loss.detach().item()),
        "gradients": tuple(item.detach().to(dtype=torch.float32).clone() for item in gradient),
    }


def _assert_normal_equal(expected, actual):
    for key in ("sample_ids", "indices", "image_sha256", "label_sha256", "loss"):
        if expected[key] != actual[key]:
            raise AssertionError(f"Normal continuation differs for {key}")
    for left, right in zip(expected["gradients"], actual["gradients"]):
        if not torch.equal(left, right):
            raise AssertionError("Normal continuation logical gradient differs")


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
        ["git", "merge-base", "--is-ancestor", TASK10_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-10 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Pinned Dassl SHA changed")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl worktree is dirty")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-11 real integration requires CUDA")

    data_root = Path(args.root).resolve(strict=True)
    manifest_root = Path(args.manifest_root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output = Path(args.output).resolve()
    cfg = build_smoke_cfg(
        REPO_ROOT,
        data_root,
        output.parent / "task11_runtime",
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
    dataset_root = source.dataset_root
    full_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    controller = PrecisionController("fp32")
    service = FullGradientService(
        model=model,
        param_index=index,
        loader=full_loader,
        precision_controller=controller,
        protocol_seed=1,
        dataset="dtd",
        shots=4,
        config_hash="task11-real-dtd-purity-v1",
    )

    normal_start = capture_rng_state()
    control_iterator = iter(trainer.train_loader_x)
    control_a = _normal_batch_result(
        trainer, model, index, next(control_iterator), dataset_root
    )
    control_b = _normal_batch_result(
        trainer, model, index, next(control_iterator), dataset_root
    )

    restore_rng_state(normal_start)
    query_iterator = iter(trainer.train_loader_x)
    query_a = _normal_batch_result(
        trainer, model, index, next(query_iterator), dataset_root
    )
    _assert_normal_equal(control_a, query_a)

    # Preserve a real pre-existing live gradient, rather than only testing None.
    for entry, gradient in zip(index, query_a["gradients"]):
        entry.parameter.grad = gradient.to(dtype=entry.parameter.dtype).clone()
    parameters_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    buffers_before = {name: b.detach().clone() for name, b in model.named_buffers()}
    grads_before = {
        name: None if p.grad is None else p.grad.detach().clone()
        for name, p in model.named_parameters()
    }
    frozen_before = hash_frozen_parameters(model)
    modes_before = tuple(module.training for module in model.modules())
    optimizer_before = copy.deepcopy(trainer.optim.state_dict())
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    precision_before = copy.deepcopy(controller.state_dict())
    precision_phase_before = controller.phase
    query_rng_before = capture_rng_state((full_loader.generator,))

    with count_optimizer_steps(trainer.optim) as counter:
        first = service.compute(optimizer_step=0, purpose="diagnostic")
        middle_rng = capture_rng_state((full_loader.generator,))
        second = service.compute(optimizer_step=0, purpose="diagnostic")
    query_rng_after = capture_rng_state((full_loader.generator,))
    if counter["count"] != 0:
        raise AssertionError("Purity query called optimizer.step")
    if not _rng_equal(query_rng_before, middle_rng) or not _rng_equal(
        query_rng_before, query_rng_after
    ):
        raise AssertionError("Repeated real query changed RNG state")
    for left, right in zip(first.gradient, second.gradient):
        if not torch.equal(left, right):
            raise AssertionError("Repeated real query gradient differs")
    first_meta = first.metadata.as_dict()
    second_meta = second.metadata.as_dict()
    first_meta.pop("elapsed_s")
    second_meta.pop("elapsed_s")
    if first_meta != second_meta:
        raise AssertionError("Repeated deterministic metadata differs")

    query_b = _normal_batch_result(
        trainer, model, index, next(query_iterator), dataset_root
    )
    _assert_normal_equal(control_b, query_b)
    if any(
        not torch.equal(parameters_before[name], parameter)
        for name, parameter in model.named_parameters()
    ):
        raise AssertionError("Real query changed model parameters")
    if any(
        not torch.equal(buffers_before[name], buffer)
        for name, buffer in model.named_buffers()
    ):
        raise AssertionError("Real query changed model buffers")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Real query changed frozen CLIP")
    for name, parameter in model.named_parameters():
        expected = grads_before[name]
        if expected is None and parameter.grad is not None:
            raise AssertionError(f"Real query populated .grad for {name}")
        if expected is not None and (
            parameter.grad is None or not torch.equal(expected, parameter.grad)
        ):
            raise AssertionError(f"Real query changed .grad for {name}")
    if tuple(module.training for module in model.modules()) != modes_before:
        raise AssertionError("Real query changed module modes")
    if not _nested_equal(optimizer_before, trainer.optim.state_dict()):
        raise AssertionError("Real query changed optimizer state")
    if not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError("Real query changed scheduler state")
    if not _nested_equal(precision_before, controller.state_dict()):
        raise AssertionError("Real query changed precision/scaler state")
    if precision_phase_before != controller.phase:
        raise AssertionError("Real query changed precision phase")

    payload = {
        "schema_version": "sample_fg.task11_full_gradient_purity.v1",
        "status": "PASS",
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "source_count": len(source),
        "source_fingerprint": source.fingerprint,
        "gradient_norm": float(first.gradient.norm().item()),
        "mean_loss": first.mean_loss,
        "repeat_query_exact": True,
        "state_categories": {
            "trainable_parameters": "unchanged",
            "frozen_clip_parameters": "unchanged",
            "model_buffers": "unchanged",
            "live_grad_buffers": "unchanged_real_tensor_and_none",
            "optimizer": "unchanged",
            "scheduler": "unchanged",
            "precision_controller": "unchanged",
            "grad_scaler": "not_present_fp32",
            "module_modes": "unchanged",
            "python_numpy_torch_cpu_cuda_rng": "unchanged",
            "dedicated_loader_generator": "unchanged",
            "normal_loader_continuation": "exact",
            "estimator_state": "not_implemented_task12",
            "perturbation_state": "not_implemented_task13",
        },
        "normal_continuation": {
            "batch_a_sample_ids": list(query_a["sample_ids"]),
            "batch_b_sample_ids": list(query_b["sample_ids"]),
            "batch_b_loss": query_b["loss"],
            "batch_b_gradient_norm": float(
                torch.sqrt(
                    sum(torch.sum(item * item) for item in query_b["gradients"])
                ).item()
            ),
            "control_equals_query": True,
        },
        "query": {
            "metadata": first.metadata.as_dict(),
            "optimizer_steps": 0,
            "scheduler_steps": 0,
        },
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
