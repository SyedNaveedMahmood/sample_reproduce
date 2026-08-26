"""Run real K=2 periodic SAMPLe and a K=1 exact-equivalence check."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import build_coop_trainer, build_smoke_cfg, count_optimizer_steps, hash_frozen_parameters, unwrap_model
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.estimators import ExactEstimator, PeriodicEstimator
from sample_fg.full_gradient import FullGradientService, build_full_gradient_loader, load_full_gradient_source
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import capture_rng_state, restore_rng_state
from sample_fg.step_engine import StepEngine


TASK16_SHA = "53a269cedff3d533be1df517789c85e21973414f"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _git(args, root):
    return subprocess.check_output(["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True).strip()


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def _rng_equal(left, right):
    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(torch.equal(a, b) for a, b in zip(left.torch_cuda_states, right.torch_cuda_states))
        and len(left.explicit_generators) == len(right.explicit_generators)
        and all(a.generator is b.generator and torch.equal(a.state, b.state) for a, b in zip(left.explicit_generators, right.explicit_generators))
    )


class _AuditedService:
    def __init__(self, service, index, optimizer, scheduler, generator):
        self.service = service
        self.index = index
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.generator = generator
        self.expected_theta = None
        self.calls = []

    def compute(self, *, optimizer_step, purpose):
        theta = tuple(parameter.detach().clone() for parameter in self.index.parameters)
        if self.expected_theta is None or not all(torch.equal(value, expected) for value, expected in zip(theta, self.expected_theta)):
            raise AssertionError("Periodic refresh query was not at original theta")
        grads = tuple(None if p.grad is None else p.grad.detach().clone() for p in self.index.parameters)
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        rng = capture_rng_state((self.generator,))
        result = self.service.compute(optimizer_step=optimizer_step, purpose=purpose)
        checks = {
            "theta": all(torch.equal(value, entry.parameter) for value, entry in zip(theta, self.index)),
            "live_grads": all((expected is None and entry.parameter.grad is None) or (expected is not None and entry.parameter.grad is not None and torch.equal(expected, entry.parameter.grad)) for expected, entry in zip(grads, self.index)),
            "optimizer": _nested_equal(optimizer_state, self.optimizer.state_dict()),
            "scheduler": _nested_equal(scheduler_state, self.scheduler.state_dict()),
            "rng": _rng_equal(rng, capture_rng_state((self.generator,))),
        }
        if not all(checks.values()):
            raise AssertionError(f"Periodic refresh query purity failed: {checks}")
        self.calls.append({"step": optimizer_step, "purpose": purpose, "checks": checks})
        return result


def _service(model, index, loader, config_hash):
    return FullGradientService(
        model=model,
        param_index=index,
        loader=loader,
        precision_controller=PrecisionController("fp32"),
        protocol_seed=1,
        dataset="dtd",
        shots=4,
        config_hash=config_hash,
    )


def _state_close(left, right, rtol=1e-5, atol=1e-6):
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run(args):
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK16_SHA, "HEAD"], cwd=REPO_ROOT, check=False).returncode:
        raise AssertionError("Accepted Task-16 commit is not an ancestor of HEAD")
    if _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA or _git(["status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-17 real integration requires CUDA")

    output = Path(args.output).resolve()
    data_root = Path(args.root).resolve(strict=True)
    cfg = build_smoke_cfg(REPO_ROOT, data_root, output.parent / "task17_runtime", "base")
    cfg.defrost()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.freeze()
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, Path(args.clip_cache).resolve(strict=True))
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    loaded = load_dataset(data_root, DATASET_SPECS["dtd"])
    source = load_full_gradient_source(loaded, Path(args.manifest_root).resolve(strict=True) / "dtd" / "shots_4" / "seed_1" / "data_manifest.json")
    normal_iterator = iter(trainer.train_loader_x)
    raw_fixed = next(normal_iterator)
    fixed_batch = trainer.parse_batch_train(raw_fixed)

    # Real K=1/exact end-to-end equivalence from identical theta, optimizer,
    # materialized batch, RNG, seed purpose, and full-gradient config.
    initial_prompt = tuple(parameter.detach().clone() for parameter in index.parameters)
    initial_optimizer = copy.deepcopy(trainer.optim.state_dict())
    initial_rng = capture_rng_state()
    exact_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    exact_service = _service(model, index, exact_loader, "task17-k1-equivalence-v1")
    exact_estimator = ExactEstimator(index, full_gradient_service=exact_service)
    exact_engine = StepEngine(param_index=index, optimizer=trainer.optim, precision_controller=PrecisionController("fp32"), rho=0.05, alpha=0.0015)
    exact_record = exact_engine.step_sample(fixed_batch, lambda item: F.cross_entropy(model(item[0]), item[1]), exact_estimator)
    exact_prompt = tuple(parameter.detach().clone() for parameter in index.parameters)

    with torch.no_grad():
        for entry, value in zip(index, initial_prompt):
            entry.parameter.copy_(value)
    trainer.optim.load_state_dict(copy.deepcopy(initial_optimizer))
    trainer.optim.zero_grad(set_to_none=True)
    restore_rng_state(initial_rng)
    periodic_k1_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    periodic_k1_service = _service(model, index, periodic_k1_loader, "task17-k1-equivalence-v1")
    periodic_k1 = PeriodicEstimator(index, ema_lambda=0.15, refresh_k_steps=1, full_gradient_service=periodic_k1_service)
    periodic_k1_engine = StepEngine(param_index=index, optimizer=trainer.optim, precision_controller=PrecisionController("fp32"), rho=0.05, alpha=0.0015)
    periodic_k1_record = periodic_k1_engine.step_sample(fixed_batch, lambda item: F.cross_entropy(model(item[0]), item[1]), periodic_k1)
    periodic_k1_prompt = tuple(parameter.detach().clone() for parameter in index.parameters)
    for left, right in (
        (exact_record.estimator_result.active_global_estimate, periodic_k1_record.estimator_result.active_global_estimate),
        (exact_record.projection.batch_component, periodic_k1_record.projection.batch_component),
        (exact_record.sam_perturbation, periodic_k1_record.sam_perturbation),
        (exact_record.total_displacement, periodic_k1_record.total_displacement),
        (exact_record.perturbed_gradient, periodic_k1_record.perturbed_gradient),
        (exact_record.final_gradient, periodic_k1_record.final_gradient),
    ):
        _state_close(left, right)
    for left, right in zip(exact_prompt, periodic_k1_prompt):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    k1_equivalence = {
        "exact_purpose": exact_record.estimator_result.full_gradient_metadata.seed.purpose,
        "periodic_k1_purpose": periodic_k1_record.estimator_result.full_gradient_metadata.seed.purpose,
        "active_direction": "MATCH",
        "batch_component": "MATCH",
        "sam_perturbation": "MATCH",
        "total_displacement": "MATCH",
        "perturbed_gradient": "MATCH",
        "final_gradient": "MATCH",
        "final_prompt": "MATCH",
    }
    if k1_equivalence["exact_purpose"] != "optimization_exact" or k1_equivalence["periodic_k1_purpose"] != "optimization_exact":
        raise AssertionError("K=1 did not use exact-purpose RNG semantics")

    # Restore once more, then exercise real K=2 at steps 0,1,2.
    with torch.no_grad():
        for entry, value in zip(index, initial_prompt):
            entry.parameter.copy_(value)
    trainer.optim.load_state_dict(copy.deepcopy(initial_optimizer))
    trainer.optim.zero_grad(set_to_none=True)
    restore_rng_state(initial_rng)
    prompt_before = index[0].parameter.detach().clone()
    frozen_before = hash_frozen_parameters(model)
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    k2_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    audited = _AuditedService(_service(model, index, k2_loader, "task17-real-k2-v1"), index, trainer.optim, trainer.sched, k2_loader.generator)
    estimator = PeriodicEstimator(index, ema_lambda=0.15, refresh_k_steps=2, full_gradient_service=audited)
    engine = StepEngine(param_index=index, optimizer=trainer.optim, precision_controller=PrecisionController("fp32"), rho=0.05, alpha=0.0015)
    records = []
    with count_optimizer_steps(trainer.optim) as counter:
        for step in range(3):
            raw_batch = next(normal_iterator)
            materialized = trainer.parse_batch_train(raw_batch)
            identity = (id(materialized), materialized[0].data_ptr(), materialized[1].data_ptr())
            calls = []
            audited.expected_theta = tuple(parameter.detach().clone() for parameter in index.parameters)

            def closure(observed):
                calls.append((id(observed), observed[0].data_ptr(), observed[1].data_ptr()))
                return F.cross_entropy(model(observed[0]), observed[1])

            record = engine.step_sample(materialized, closure, estimator)
            if calls != [identity, identity]:
                raise AssertionError("Periodic SAMPLe did not reuse materialized tensors")
            active = record.estimator_result.active_global_estimate
            residual_dot = float(record.projection.batch_component.dot(active).item())
            denominator = record.batch_component_norm * record.global_direction_norm
            orthogonality = 0.0 if denominator == 0 else abs(residual_dot) / denominator
            values = (
                record.batch_gradient_norm,
                record.global_direction_norm,
                record.projection.xi,
                record.projection.sigma,
                record.batch_component_norm,
                record.total_displacement_norm,
                record.perturbed_gradient_norm,
                record.final_gradient_norm,
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("Periodic SAMPLe produced nonfinite geometry")
            metadata = record.estimator_result.full_gradient_metadata
            records.append({
                "optimizer_step": record.optimizer_step,
                "refreshed": record.estimator_result.refreshed,
                "age_steps": record.estimator_result.age_steps,
                "last_refresh_step": record.estimator_result.last_refresh_step,
                "batch_gradient_norm": record.batch_gradient_norm,
                "global_direction_norm": record.global_direction_norm,
                "xi": record.projection.xi,
                "sigma": record.projection.sigma,
                "batch_component_norm": record.batch_component_norm,
                "batch_component_normalized_orthogonality_residual": orthogonality,
                "total_displacement_norm": record.total_displacement_norm,
                "perturbed_gradient_norm": record.perturbed_gradient_norm,
                "final_gradient_norm": record.final_gradient_norm,
                "cumulative_exact_query_count": record.estimator_result.exact_query_count,
                "full_gradient_sample_count": metadata.sample_count if metadata is not None else 0,
                "full_gradient_micro_batch_count": metadata.micro_batch_count if metadata is not None else 0,
                "full_gradient_elapsed_s": metadata.elapsed_s if metadata is not None else 0.0,
                "same_materialized_tensors": True,
                "restored_before_optimizer": record.restored_before_optimizer,
            })

    expected = {
        "refreshed": [True, False, True],
        "ages": [0, 1, 0],
        "last": [0, 0, 2],
        "queries": [1, 1, 2],
    }
    observed = {
        "refreshed": [item["refreshed"] for item in records],
        "ages": [item["age_steps"] for item in records],
        "last": [item["last_refresh_step"] for item in records],
        "queries": [item["cumulative_exact_query_count"] for item in records],
    }
    if observed != expected or [call["step"] for call in audited.calls] != [0, 2]:
        raise AssertionError(f"K=2 refresh sequence differs: {observed}")
    if [call["purpose"] for call in audited.calls] != ["periodic_refresh", "periodic_refresh"]:
        raise AssertionError("K=2 refresh purpose differs")
    if counter["count"] != 3 or estimator.exact_query_count != 2:
        raise AssertionError("K=2 optimizer/query count differs")
    if torch.equal(index[0].parameter, prompt_before) or hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("K=2 prompt/frozen invariant failed")
    if not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError("Epoch scheduler changed within bounded K=2 steps")

    payload = {
        "schema_version": "sample_fg.task17_sample_periodic.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "method": "sample",
        "estimator": "periodic",
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "backbone": "ViT-B/16",
        "precision": "fp32",
        "batch_size": 2,
        "workers": 0,
        "rho": 0.05,
        "alpha": 0.0015,
        "ema_lambda": 0.15,
        "refresh_k_steps": 2,
        "k1_exact_equivalence": k1_equivalence,
        "records": records,
        "refresh_steps": [0, 2],
        "full_gradient_query_count": estimator.exact_query_count,
        "optimizer_steps": counter["count"],
        "scheduler_steps": 0,
        "prompt_changed": True,
        "frozen_clip_unchanged": True,
        "diagnostics_enabled": False,
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
