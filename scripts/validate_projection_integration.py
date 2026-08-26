"""Validate Task-6 projection with two real gradients at unchanged CoOp state."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    EXPECTED_CLIP_SHA256,
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    hash_frozen_parameters,
    sha256_file,
    unwrap_model,
)
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.projection import DEFAULT_NORM_EPS, project_batch_gradient


COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
TASK4_COOP_SHA = "87ddca7807a0ab4e79f38a3ad8b6b034bd5b19af"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
DTD_SPLIT_SHA = "F26EBECCD2B58E68D70F07A0DC39FE49BF7B69024BAC34396B954DFD87969C38"
DTD_CACHE_SHA = "81EE5B688EC9D80BBE424522A7638FCE5AAE84F07EC40928CBBA2B57B2B142AD"
FIXTURE_DIRECTION_LABEL = "projection integration fixture direction"


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _git_record(path: Path) -> dict[str, Any]:
    status = _run(["git", "status", "--short"], path)
    return {
        "head": _run(["git", "rev-parse", "HEAD"], path),
        "branch": _run(["git", "branch", "--show-current"], path),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _capture_gradient(trainer, param_index, batch):
    trainer.optim.zero_grad(set_to_none=True)
    image, label = trainer.parse_batch_train(batch)
    logits = trainer.model(image)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("Real CoOp projection fixture loss is nonfinite")
    state = GradientState.from_parameter_grads(param_index)
    trainer.optim.zero_grad(set_to_none=True)
    if any(entry.parameter.grad is not None for entry in param_index):
        raise AssertionError("Live gradients were not cleared after capture")
    return state, float(loss.item())


def run(args: argparse.Namespace) -> Path:
    coop_root = REPO_ROOT
    project_root = coop_root.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    data_root = Path(args.root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output_path = Path(args.output).resolve()

    if _run(["git", "branch", "--show-current"], coop_root) != "sample-full-gradient":
        raise AssertionError("Task-6 validation requires branch sample-full-gradient")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK4_COOP_SHA, "HEAD"],
        cwd=coop_root,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-4 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Dassl SHA differs from the pinned baseline")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Dassl worktree is not clean")
    if not torch.cuda.is_available():
        raise RuntimeError("The real Task-6 CoOp integration check requires CUDA")

    checkpoint_path = clip_cache / "ViT-B-16.pt"
    if sha256_file(checkpoint_path).lower() != EXPECTED_CLIP_SHA256:
        raise AssertionError("Task-3 CLIP checkpoint hash changed")
    split_path = data_root / "dtd" / "split_zhou_DescribableTextures.json"
    cache_path = data_root / "dtd" / "split_fewshot" / "shot_4-seed_1.pkl"
    if sha256_file(split_path) != DTD_SPLIT_SHA:
        raise AssertionError("DTD fixed-split hash changed")
    if sha256_file(cache_path) != DTD_CACHE_SHA:
        raise AssertionError("DTD 4-shot seed-1 cache hash changed")

    cfg = build_smoke_cfg(
        coop_root,
        data_root,
        output_path.parent / "task6_runtime",
        "base",
    )
    set_random_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    trainer = build_coop_trainer(cfg, clip_cache)
    model = unwrap_model(trainer.model)
    param_index = ParamIndex.from_model(model)
    if param_index.names != ("prompt_learner.ctx",):
        raise AssertionError(f"Unexpected trainable parameters: {param_index.names}")
    prompt_before = param_index[0].parameter.detach().clone()
    frozen_hash_before = hash_frozen_parameters(model)

    trainer.set_model_mode("train")
    iterator = iter(trainer.train_loader_x)
    with count_optimizer_steps(trainer.optim) as optimizer_counter:
        batch_gradient, first_loss = _capture_gradient(
            trainer, param_index, next(iterator)
        )
        fixture_direction, second_loss = _capture_gradient(
            trainer, param_index, next(iterator)
        )
        result = project_batch_gradient(
            batch_gradient,
            fixture_direction,
            norm_eps=DEFAULT_NORM_EPS,
        )
    if optimizer_counter["count"] != 0:
        raise AssertionError("Projection integration performed an optimizer step")

    residual_dot = float(result.batch_component.dot(fixture_direction).item())
    projected_norm = float(result.projected_component.norm().item())
    residual_norm = float(result.batch_component.norm().item())
    normalization = max(
        result.batch_norm * result.full_direction_norm,
        torch.finfo(torch.float32).tiny,
    )
    normalized_orthogonality = abs(residual_dot) / normalization
    reconstruction_error = float(
        (
            batch_gradient
            - (result.projected_component + result.batch_component)
        ).norm().item()
    )
    scalars = (
        result.batch_norm,
        result.full_direction_norm,
        result.dot_product,
        result.xi,
        result.sigma,
        result.projection_coefficient,
        projected_norm,
        residual_norm,
        residual_dot,
        normalized_orthogonality,
        reconstruction_error,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise FloatingPointError("Projection integration produced nonfinite geometry")
    if normalized_orthogonality > 3e-6:
        raise AssertionError(
            f"Projection residual is not sufficiently orthogonal: "
            f"{normalized_orthogonality}"
        )
    if reconstruction_error > 1e-5 * max(result.batch_norm, 1.0):
        raise AssertionError(f"Projection reconstruction error: {reconstruction_error}")
    if not torch.equal(param_index[0].parameter.detach(), prompt_before):
        raise AssertionError("Prompt changed despite zero optimizer steps")
    frozen_hash_after = hash_frozen_parameters(model)
    if frozen_hash_after != frozen_hash_before:
        raise AssertionError("A frozen parameter changed during projection validation")

    report = {
        "schema_version": "sample_fg.projection_integration.v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "coop_upstream_provenance_sha": COOP_UPSTREAM_SHA,
            "coop_task4_sha": TASK4_COOP_SHA,
            "coop_runtime_git": _git_record(coop_root),
            "dassl_pinned_sha": DASSL_SHA,
            "dassl_runtime_git": _git_record(dassl_root),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "data": {
            "dataset": "DTD",
            "shots": 4,
            "seed": 1,
            "classes": "base",
            "batch_size": int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
            "workers": int(cfg.DATALOADER.NUM_WORKERS),
            "split_sha256": DTD_SPLIT_SHA,
            "fewshot_cache_sha256": DTD_CACHE_SHA,
        },
        "fixture": {
            "batch_gradient_source": "first real mini-batch gradient",
            "active_direction_source": FIXTURE_DIRECTION_LABEL,
            "active_direction_is_full_gradient": False,
            "active_direction_is_estimator": False,
            "same_unchanged_model_parameters": True,
            "first_loss_non_scientific": first_loss,
            "second_loss_non_scientific": second_loss,
        },
        "projection": {
            "norm_eps": DEFAULT_NORM_EPS,
            "degenerate_rule": "norm <= norm_eps",
            "batch_norm": result.batch_norm,
            "full_direction_norm": result.full_direction_norm,
            "dot_product": result.dot_product,
            "xi": result.xi,
            "sigma": result.sigma,
            "projection_coefficient": result.projection_coefficient,
            "projected_component_norm": projected_norm,
            "batch_component_norm": residual_norm,
            "batch_component_dot_full_direction": residual_dot,
            "normalized_orthogonality_residual": normalized_orthogonality,
            "reconstruction_error_norm": reconstruction_error,
            "batch_gradient_degenerate": result.batch_gradient_degenerate,
            "full_direction_degenerate": result.full_direction_degenerate,
            "all_finite": True,
        },
        "immutability": {
            "optimizer_steps": optimizer_counter["count"],
            "prompt_unchanged": True,
            "frozen_sha256_before": frozen_hash_before,
            "frozen_sha256_after": frozen_hash_after,
            "frozen_parameters_unchanged": True,
        },
        "scope": {
            "ema": False,
            "exact_estimator": False,
            "periodic_estimator": False,
            "full_gradient_service": False,
            "sam_optimizer": False,
            "sample_optimizer": False,
            "perturbation": False,
        },
    }
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "fixture_direction": FIXTURE_DIRECTION_LABEL,
                "projection": report["projection"],
                "report": str(output_path),
            },
            indent=2,
        )
    )

    del trainer, model, iterator, batch_gradient, fixture_direction, result
    gc.collect()
    torch.cuda.empty_cache()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Prepared CoOp data root")
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
