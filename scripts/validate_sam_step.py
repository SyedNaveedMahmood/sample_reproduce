"""Run a bounded non-scientific vanilla-SAM smoke on real CoOp/DTD."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.step_engine import StepEngine


TASK13_SHA = "64ddf963ea5404c1bf8202a77f63d3b7b3a9da12"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _git(args: list[str], root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK13_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-13 commit is not an ancestor of HEAD")
    if _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA or _git(["status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-14 real integration requires CUDA")

    output = Path(args.output).resolve()
    cfg = build_smoke_cfg(
        REPO_ROOT,
        Path(args.root).resolve(strict=True),
        output.parent / "task14_runtime",
        "base",
    )
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
    engine = StepEngine(
        param_index=index,
        optimizer=trainer.optim,
        precision_controller=PrecisionController("fp32"),
        rho=0.05,
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

            def closure(observed):
                closure_calls.append(
                    (id(observed), observed[0].data_ptr(), observed[1].data_ptr())
                )
                return F.cross_entropy(model(observed[0]), observed[1])

            lr = float(trainer.get_current_lr())
            record = engine.step_sam(materialized, closure)
            if closure_calls != [identity, identity]:
                raise AssertionError("SAM did not reuse identical materialized tensors")
            if not record.restored_before_optimizer or record.final_gradient_norm != record.perturbed_gradient_norm:
                raise AssertionError("SAM final-gradient/restoration invariant failed")
            records.append(
                {
                    "optimizer_step": record.optimizer_step,
                    "epoch_zero_based": 0,
                    "batch_index_zero_based": batch_index,
                    "loss_current": record.loss_current,
                    "batch_gradient_norm": record.batch_gradient_norm,
                    "sam_perturbation_norm": record.sam_perturbation_norm,
                    "loss_displaced": record.loss_displaced,
                    "perturbed_gradient_norm": record.perturbed_gradient_norm,
                    "final_logical_gradient_norm": record.final_gradient_norm,
                    "lr": lr,
                    "restored_before_optimizer": record.restored_before_optimizer,
                    "same_materialized_tensors": True,
                    "finite": all(
                        state.is_finite()
                        for state in (
                            record.batch_gradient,
                            record.sam_perturbation,
                            record.perturbed_gradient,
                            record.final_gradient,
                        )
                    ),
                }
            )

    if counter["count"] != 3 or engine.optimizer_step != 3:
        raise AssertionError("Bounded SAM smoke did not execute exactly 3 steps")
    if torch.equal(index[0].parameter, prompt_before):
        raise AssertionError("SAM did not update the prompt")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("SAM changed frozen CLIP")
    if trainer.sched.state_dict() != scheduler_before:
        raise AssertionError("SAM changed epoch-stepped scheduler within three batches")
    for record in records:
        if not record["finite"]:
            raise FloatingPointError("SAM produced a nonfinite state")
        if record["sam_perturbation_norm"] <= 0 or abs(record["sam_perturbation_norm"] - 0.05) > 1e-5:
            raise AssertionError("Nondegenerate SAM perturbation norm differs from rho")

    payload = {
        "schema_version": "sample_fg.task14_sam_step.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "method": "sam",
        "estimator": None,
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "backbone": "ViT-B/16",
        "precision": "fp32",
        "batch_size": 2,
        "workers": 0,
        "rho": 0.05,
        "records": records,
        "optimizer_steps": counter["count"],
        "scheduler_steps": 0,
        "scheduler_convention": "pinned_epoch_boundary_not_reached",
        "prompt_changed": True,
        "frozen_clip_unchanged": True,
        "exact_query_count": 0,
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
