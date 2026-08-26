"""Run bounded exact-gradient SAMPLe steps on real CoOp/DTD."""

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
from sample_fg.estimators import ExactEstimator
from sample_fg.full_gradient import FullGradientService, build_full_gradient_loader, load_full_gradient_source
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import capture_rng_state
from sample_fg.step_engine import StepEngine


TASK15_SHA = "ca2b04ae3dafde19427571b4294dce97b12581f1"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _git(args: list[str], root: Path) -> str:
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
    def __init__(self, service, model, index, optimizer, scheduler, generator):
        self.service = service
        self.model = model
        self.index = index
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.generator = generator
        self.expected_theta = None
        self.calls = []

    def compute(self, *, optimizer_step, purpose):
        theta_before = tuple(parameter.detach().clone() for parameter in self.index.parameters)
        if self.expected_theta is None or not all(torch.equal(value, expected) for value, expected in zip(theta_before, self.expected_theta)):
            raise AssertionError("Exact query did not occur at original theta")
        grads_before = tuple(None if p.grad is None else p.grad.detach().clone() for p in self.index.parameters)
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        scheduler_before = copy.deepcopy(self.scheduler.state_dict())
        modes_before = tuple(module.training for module in self.model.modules())
        rng_before = capture_rng_state((self.generator,))
        result = self.service.compute(optimizer_step=optimizer_step, purpose=purpose)
        rng_after = capture_rng_state((self.generator,))
        checks = {
            "theta": all(torch.equal(value, entry.parameter) for value, entry in zip(theta_before, self.index)),
            "live_grads": all(
                (expected is None and entry.parameter.grad is None)
                or (expected is not None and entry.parameter.grad is not None and torch.equal(expected, entry.parameter.grad))
                for expected, entry in zip(grads_before, self.index)
            ),
            "optimizer": _nested_equal(optimizer_before, self.optimizer.state_dict()),
            "scheduler": _nested_equal(scheduler_before, self.scheduler.state_dict()),
            "modes": modes_before == tuple(module.training for module in self.model.modules()),
            "rng": _rng_equal(rng_before, rng_after),
        }
        if not all(checks.values()):
            raise AssertionError(f"Exact query purity failed: {checks}")
        self.calls.append({"step": optimizer_step, "purpose": purpose, "checks": checks})
        return result


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run(args):
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK15_SHA, "HEAD"], cwd=REPO_ROOT, check=False).returncode:
        raise AssertionError("Accepted Task-15 commit is not an ancestor of HEAD")
    if _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA or _git(["status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-16 real integration requires CUDA")

    output = Path(args.output).resolve()
    data_root = Path(args.root).resolve(strict=True)
    cfg = build_smoke_cfg(REPO_ROOT, data_root, output.parent / "task16_runtime", "base")
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
    loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    service = FullGradientService(
        model=model,
        param_index=index,
        loader=loader,
        precision_controller=PrecisionController("fp32"),
        protocol_seed=1,
        dataset="dtd",
        shots=4,
        config_hash="task16-real-exact-step-v1",
    )
    audited = _AuditedService(service, model, index, trainer.optim, trainer.sched, loader.generator)
    estimator = ExactEstimator(index, full_gradient_service=audited)
    engine = StepEngine(
        param_index=index,
        optimizer=trainer.optim,
        precision_controller=PrecisionController("fp32"),
        rho=0.05,
        alpha=0.0015,
    )
    prompt_before = index[0].parameter.detach().clone()
    frozen_before = hash_frozen_parameters(model)
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    records = []
    with count_optimizer_steps(trainer.optim) as counter:
        for batch_index, raw_batch in enumerate(trainer.train_loader_x):
            if len(records) == 2:
                break
            image, label = trainer.parse_batch_train(raw_batch)
            materialized = (image, label)
            identity = (id(materialized), image.data_ptr(), label.data_ptr())
            closure_calls = []
            audited.expected_theta = tuple(parameter.detach().clone() for parameter in index.parameters)
            rng_before = capture_rng_state((loader.generator,))

            def closure(observed):
                closure_calls.append((id(observed), observed[0].data_ptr(), observed[1].data_ptr()))
                return F.cross_entropy(model(observed[0]), observed[1])

            record = engine.step_sample(materialized, closure, estimator)
            rng_after = capture_rng_state((loader.generator,))
            if closure_calls != [identity, identity] or not _rng_equal(rng_before, rng_after):
                raise AssertionError("Exact SAMPLe changed batch/RNG trajectory")
            exact = record.estimator_result.exact_reference
            active = record.estimator_result.active_global_estimate
            self_cosine = float(active.dot(exact).item() / (active.norm().item() * exact.norm().item()))
            residual_dot = float(record.projection.batch_component.dot(active).item())
            denominator = record.batch_component_norm * record.global_direction_norm
            orthogonality = 0.0 if denominator == 0 else abs(residual_dot) / denominator
            metadata = record.estimator_result.full_gradient_metadata
            values = (
                record.batch_gradient_norm,
                record.global_direction_norm,
                self_cosine,
                record.projection.xi,
                record.projection.sigma,
                record.batch_component_norm,
                record.sam_perturbation_norm,
                record.total_displacement_norm,
                record.perturbed_gradient_norm,
                record.final_gradient_norm,
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("Exact SAMPLe produced nonfinite geometry")
            records.append({
                "optimizer_step": record.optimizer_step,
                "batch_index_zero_based": batch_index,
                "loss_current": record.loss_current,
                "batch_gradient_norm": record.batch_gradient_norm,
                "exact_full_gradient_norm": record.global_direction_norm,
                "batch_exact_cosine": record.projection.xi,
                "exact_self_cosine": self_cosine,
                "sigma": record.projection.sigma,
                "projection_coefficient": record.projection.projection_coefficient,
                "batch_component_norm": record.batch_component_norm,
                "batch_component_normalized_orthogonality_residual": orthogonality,
                "sam_perturbation_norm": record.sam_perturbation_norm,
                "total_displacement_norm": record.total_displacement_norm,
                "loss_displaced": record.loss_displaced,
                "perturbed_gradient_norm": record.perturbed_gradient_norm,
                "final_gradient_norm": record.final_gradient_norm,
                "full_gradient_query_count": estimator.exact_query_count,
                "full_gradient_sample_count": metadata.sample_count,
                "full_gradient_micro_batch_count": metadata.micro_batch_count,
                "full_gradient_micro_batch_sizes": list(metadata.observed_micro_batch_sizes),
                "full_gradient_elapsed_s": metadata.elapsed_s,
                "query_at_original_theta": True,
                "query_purity": audited.calls[-1]["checks"],
                "same_materialized_tensors": True,
            })

    if counter["count"] != 2 or estimator.exact_query_count != 2 or len(audited.calls) != 2:
        raise AssertionError("Exact SAMPLe optimizer/query count differs")
    if [call["purpose"] for call in audited.calls] != ["optimization_exact"] * 2:
        raise AssertionError("Exact SAMPLe query purpose differs")
    if torch.equal(index[0].parameter, prompt_before) or hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Prompt/frozen parameter invariant failed")
    if not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError("Epoch scheduler changed within bounded steps")
    if not all(abs(record["exact_self_cosine"] - 1.0) <= 2e-6 for record in records):
        raise AssertionError("Exact estimator self cosine differs from one")

    payload = {
        "schema_version": "sample_fg.task16_sample_exact.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "method": "sample",
        "estimator": "exact",
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "backbone": "ViT-B/16",
        "precision": "fp32",
        "batch_size": 2,
        "workers": 0,
        "full_gradient_micro_batch_size": 32,
        "rho": 0.05,
        "alpha": 0.0015,
        "records": records,
        "optimizer_steps": counter["count"],
        "scheduler_steps": 0,
        "full_gradient_query_count": estimator.exact_query_count,
        "all_queries_at_original_theta": True,
        "exact_reference_reused": True,
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
