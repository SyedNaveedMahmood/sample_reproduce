"""Validate Task-4 gradient capture on the real Task-3 CoOp anchor."""

from __future__ import annotations

import argparse
import gc
import json
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
    sha256_file,
    unwrap_model,
)
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex


COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
TASK3_COOP_SHA = "6de4ed662d8fbe3870e99d8556ed57182694c5eb"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
DTD_SPLIT_SHA = "F26EBECCD2B58E68D70F07A0DC39FE49BF7B69024BAC34396B954DFD87969C38"
DTD_CACHE_SHA = "81EE5B688EC9D80BBE424522A7638FCE5AAE84F07EC40928CBBA2B57B2B142AD"


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


def run(args: argparse.Namespace) -> Path:
    coop_root = REPO_ROOT
    project_root = coop_root.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    data_root = Path(args.root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output_path = Path(args.output).resolve()

    if _run(["git", "branch", "--show-current"], coop_root) != "sample-full-gradient":
        raise AssertionError("Task-4 validation requires branch sample-full-gradient")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK3_COOP_SHA, "HEAD"],
        cwd=coop_root,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-3 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Dassl SHA differs from the pinned baseline")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Dassl worktree is not clean")
    if not torch.cuda.is_available():
        raise RuntimeError("The real Task-4 CoOp integration check requires CUDA")

    checkpoint_path = clip_cache / "ViT-B-16.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Task-3 CLIP checkpoint is missing: {checkpoint_path}")
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash.lower() != EXPECTED_CLIP_SHA256:
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
        output_path.parent / "task4_runtime",
        "base",
    )
    set_random_seed(cfg.SEED)
    torch.backends.cudnn.benchmark = True
    trainer = build_coop_trainer(cfg, clip_cache)
    model = unwrap_model(trainer.model)
    param_index = ParamIndex.from_model(model)
    if param_index.names != ("prompt_learner.ctx",):
        raise AssertionError(f"Unexpected trainable parameters: {param_index.names}")
    if param_index[0].shape != (4, 512):
        raise AssertionError(f"Unexpected CoOp context shape: {param_index[0].shape}")
    if param_index.total_numel != 2048:
        raise AssertionError(f"Unexpected trainable numel: {param_index.total_numel}")
    param_index.assert_matches_model(model)

    trainer.set_model_mode("train")
    trainer.optim.zero_grad(set_to_none=True)
    prompt_before = param_index[0].parameter.detach().clone()
    batch = next(iter(trainer.train_loader_x))
    image, label = trainer.parse_batch_train(batch)
    logits = trainer.model(image)
    loss = F.cross_entropy(logits, label)
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("Real CoOp integration loss is non-finite")

    live_grad_before_capture = param_index[0].parameter.grad.detach().clone()
    state = GradientState.from_parameter_grads(param_index)
    captured_before_clear = state.clone()
    gradient_norm = float(state.norm().item())
    if not state.is_finite() or gradient_norm <= 0:
        raise FloatingPointError("Captured real CoOp gradient is non-finite or zero")
    if any(component.requires_grad or component.grad_fn is not None for component in state):
        raise AssertionError("Captured state retained an autograd graph")
    if not torch.equal(
        state[0], live_grad_before_capture.detach().to(dtype=torch.float32)
    ):
        raise AssertionError("Captured state differs from the live prompt gradient")

    trainer.optim.zero_grad(set_to_none=True)
    if any(entry.parameter.grad is not None for entry in param_index):
        raise AssertionError("Live prompt gradients were not cleared")
    if not all(
        torch.equal(current, saved)
        for current, saved in zip(state, captured_before_clear)
    ):
        raise AssertionError("Captured state changed when live gradients were cleared")
    if not torch.equal(param_index[0].parameter.detach(), prompt_before):
        raise AssertionError("Prompt changed despite zero optimizer steps")

    report = {
        "schema_version": "sample_fg.gradient_state_integration.v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "coop_upstream_provenance_sha": COOP_UPSTREAM_SHA,
            "coop_task3_start_sha": TASK3_COOP_SHA,
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
        "clip": {
            "checkpoint_path": str(checkpoint_path),
            "sha256": checkpoint_hash,
        },
        "param_index": param_index.to_metadata(),
        "gradient_state": {
            "component_count": len(state),
            "component_shapes": [list(component.shape) for component in state],
            "component_dtypes": [str(component.dtype) for component in state],
            "component_devices": [str(component.device) for component in state],
            "total_numel": state.total_numel,
            "raw_tensor_bytes": state.raw_tensor_bytes,
            "norm": gradient_norm,
            "finite": state.is_finite(),
            "detached": all(
                not component.requires_grad and component.grad_fn is None
                for component in state
            ),
            "survived_live_grad_clear_exactly": True,
        },
        "training": {
            "loss": float(loss.item()),
            "optimizer_steps": 0,
            "prompt_unchanged": True,
        },
        "scope": {
            "projection": False,
            "sam": False,
            "sample": False,
            "estimators": False,
            "full_gradient_service": False,
        },
    }
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "fingerprint": param_index.fingerprint,
                "gradient_norm": gradient_norm,
                "raw_tensor_bytes": state.raw_tensor_bytes,
                "report": str(output_path),
            },
            indent=2,
        )
    )

    del trainer, model, batch, image, label, logits, loss
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
