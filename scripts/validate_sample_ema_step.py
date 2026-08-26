"""Run a bounded non-scientific SAMPLe-EMA smoke on real CoOp/DTD."""

from __future__ import annotations

import argparse
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
from sample_fg.coop_anchor import (
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.estimators import EMAEstimator
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import capture_rng_state
from sample_fg.step_engine import StepEngine


TASK14_SHA = "ab6da25d2f5c14b26b87e71f35bb8cf14949b8f0"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _git(args: list[str], root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _rng_equal(left, right) -> bool:
    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(torch.equal(a, b) for a, b in zip(left.torch_cuda_states, right.torch_cuda_states))
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK14_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-14 commit is not an ancestor of HEAD")
    if _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA or _git(["status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-15 real integration requires CUDA")

    output = Path(args.output).resolve()
    cfg = build_smoke_cfg(REPO_ROOT, Path(args.root).resolve(strict=True), output.parent / "task15_runtime", "base")
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
    prompt_before = index[0].parameter.detach().clone()
    frozen_before = hash_frozen_parameters(model)
    scheduler_before = trainer.sched.state_dict()
    estimator = EMAEstimator(index, ema_lambda=0.15)
    engine = StepEngine(
        param_index=index,
        optimizer=trainer.optim,
        precision_controller=PrecisionController("fp32"),
        rho=0.05,
        alpha=0.0015,
    )
    records = []
    with count_optimizer_steps(trainer.optim) as counter:
        for batch_index, raw_batch in enumerate(trainer.train_loader_x):
            if len(records) == 3:
                break
            image, label = trainer.parse_batch_train(raw_batch)
            materialized = (image, label)
            identity = (id(materialized), image.data_ptr(), label.data_ptr())
            closure_calls = []
            rng_before = capture_rng_state()

            def closure(observed):
                closure_calls.append((id(observed), observed[0].data_ptr(), observed[1].data_ptr()))
                return F.cross_entropy(model(observed[0]), observed[1])

            record = engine.step_sample(materialized, closure, estimator)
            rng_after = capture_rng_state()
            if closure_calls != [identity, identity]:
                raise AssertionError("SAMPLe did not reuse identical materialized tensors")
            if not _rng_equal(rng_before, rng_after):
                raise AssertionError("SAMPLe-EMA step consumed normal RNG after batch materialization")
            residual_dot = float(record.projection.batch_component.dot(record.estimator_result.active_global_estimate).item())
            denominator = record.batch_component_norm * record.global_direction_norm
            normalized_residual = 0.0 if denominator == 0 else abs(residual_dot) / denominator
            values = (
                record.loss_current,
                record.batch_gradient_norm,
                record.global_direction_norm,
                record.projection.xi,
                record.projection.sigma,
                record.projection.projection_coefficient,
                record.batch_component_norm,
                normalized_residual,
                record.sam_perturbation_norm,
                record.batch_correction_norm,
                record.total_displacement_norm,
                record.loss_displaced,
                record.perturbed_gradient_norm,
                record.final_gradient_norm,
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError("SAMPLe-EMA produced a nonfinite metric")
            records.append(
                {
                    "optimizer_step": record.optimizer_step,
                    "epoch_zero_based": 0,
                    "batch_index_zero_based": batch_index,
                    "loss_current": record.loss_current,
                    "batch_gradient_norm": record.batch_gradient_norm,
                    "global_direction_norm": record.global_direction_norm,
                    "xi": record.projection.xi,
                    "sigma": record.projection.sigma,
                    "projection_coefficient": record.projection.projection_coefficient,
                    "batch_component_norm": record.batch_component_norm,
                    "batch_component_normalized_orthogonality_residual": normalized_residual,
                    "sam_perturbation_norm": record.sam_perturbation_norm,
                    "batch_correction_norm": record.batch_correction_norm,
                    "total_displacement_norm": record.total_displacement_norm,
                    "loss_displaced": record.loss_displaced,
                    "perturbed_gradient_norm": record.perturbed_gradient_norm,
                    "final_gradient_norm": record.final_gradient_norm,
                    "estimator_update_count": record.optimizer_step + 1,
                    "estimator_exact_query_count": estimator.exact_query_count,
                    "restored_before_optimizer": record.restored_before_optimizer,
                    "same_materialized_tensors": True,
                    "normal_rng_unchanged_after_materialization": True,
                }
            )

    if counter["count"] != 3 or engine.optimizer_step != 3 or estimator.last_processed_step != 2:
        raise AssertionError("SAMPLe-EMA logical/optimizer/estimator counts differ")
    if estimator.exact_query_count != 0:
        raise AssertionError("SAMPLe-EMA unexpectedly queried exact service")
    if torch.equal(index[0].parameter, prompt_before) or hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Prompt/frozen parameter invariant failed")
    if trainer.sched.state_dict() != scheduler_before:
        raise AssertionError("Epoch-stepped scheduler changed within bounded batches")
    first_expected = records[0]["batch_gradient_norm"] * 0.85
    if abs(records[0]["global_direction_norm"] - first_expected) > 2e-5:
        raise AssertionError("Step-0 EMA does not equal 0.85*g0")

    payload = {
        "schema_version": "sample_fg.task15_sample_ema.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "method": "sample",
        "estimator": "ema",
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
        "records": records,
        "optimizer_steps": counter["count"],
        "scheduler_steps": 0,
        "estimator_updates": estimator.last_processed_step + 1,
        "exact_query_count": estimator.exact_query_count,
        "prompt_changed": True,
        "frozen_clip_unchanged": True,
        "diagnostics_enabled": False,
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
