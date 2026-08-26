"""Validate real CoOp logical-gradient capture in fp32, fp16, and AMP."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from torch.cuda.amp import GradScaler
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
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController


COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
TASK6_COOP_SHA = "cbd0d0220c081a32da325a44cb7efa396c4cc254"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
DTD_SPLIT_SHA = "F26EBECCD2B58E68D70F07A0DC39FE49BF7B69024BAC34396B954DFD87969C38"
DTD_CACHE_SHA = "81EE5B688EC9D80BBE424522A7638FCE5AAE84F07EC40928CBBA2B57B2B142AD"
PRECISION_MODES = ("fp32", "fp16", "amp")
MIN_COSINE_WITH_FP32 = 0.99
MAX_RELATIVE_NORM_DIFFERENCE = 0.05
AMP_INTEGRATION_SCALE = 1024.0


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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _release_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _mode_cfg(coop_root, data_root, output_root, mode):
    cfg = build_smoke_cfg(
        coop_root,
        data_root,
        output_root / f"{mode}_runtime",
        "base",
    )
    cfg.defrost()
    cfg.TRAINER.COOP.PREC = mode
    cfg.freeze()
    return cfg


def _capture_mode(coop_root, data_root, output_root, clip_cache, mode):
    cfg = _mode_cfg(coop_root, data_root, output_root, mode)
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, clip_cache)
    model = unwrap_model(trainer.model)
    param_index = ParamIndex.from_model(model)
    if param_index.names != ("prompt_learner.ctx",):
        raise AssertionError(f"Unexpected trainable parameters: {param_index.names}")
    controller = PrecisionController(mode, scaler=trainer.scaler)

    prompt_before = param_index[0].parameter.detach().clone()
    prompt_sha_before = _tensor_sha256(prompt_before)
    frozen_sha_before = hash_frozen_parameters(model)
    scheduler_before = (
        int(trainer.sched.last_epoch),
        int(trainer.sched.successor.last_epoch),
        float(trainer.get_current_lr()),
    )
    batch = next(iter(trainer.train_loader_x))
    batch_identity = {
        "image_paths": list(batch["impath"]),
        "labels": [int(value) for value in batch["label"].tolist()],
        "image_tensor_sha256": _tensor_sha256(batch["img"]),
    }
    image, label = trainer.parse_batch_train(batch)
    trainer.set_model_mode("train")

    amp_default_scale_probe = None
    with count_optimizer_steps(trainer.optim) as optimizer_counter:
        capture = None
        if mode == "amp":
            controller.begin(trainer.optim)
            with controller.autocast_context():
                probe_logits = trainer.model(image)
                probe_loss = F.cross_entropy(probe_logits, label)
            controller.backward(probe_loss)
            probe_gradients_finite = all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all().item())
                for parameter in param_index.parameters
            )
            amp_default_scale_probe = {
                "upstream_default_scale": float(controller.scaler.get_scale()),
                "scaled_live_gradients_finite": probe_gradients_finite,
                "loss_finite": bool(torch.isfinite(probe_loss.detach()).item()),
                "optimizer_steps": optimizer_counter["count"],
            }
            if probe_gradients_finite:
                logits = probe_logits
                loss = probe_loss
                live_gradient = param_index[0].parameter.grad
                live_dtype_before = str(live_gradient.dtype)
                scaled_live_norm = float(live_gradient.detach().float().norm().item())
                capture = controller.capture_gradients(param_index, trainer.optim)
            else:
                trainer.optim.zero_grad(set_to_none=True)
                trainer.scaler = GradScaler(init_scale=AMP_INTEGRATION_SCALE)
                controller = PrecisionController("amp", scaler=trainer.scaler)

        if capture is None:
            controller.begin(trainer.optim)
            with controller.autocast_context():
                logits = trainer.model(image)
                loss = F.cross_entropy(logits, label)
            controller.backward(loss)
            live_gradient = param_index[0].parameter.grad
            live_dtype_before = str(live_gradient.dtype)
            scaled_live_norm = float(live_gradient.detach().float().norm().item())
            capture = controller.capture_gradients(param_index, trainer.optim)
    if optimizer_counter["count"] != 0:
        raise AssertionError(f"{mode} precision capture stepped the optimizer")

    state_snapshot = capture.state.clone()
    trainer.optim.zero_grad(set_to_none=True)
    if param_index[0].parameter.grad is not None:
        raise AssertionError(f"{mode} live gradient was not cleared")
    if not torch.equal(capture.state[0], state_snapshot[0]):
        raise AssertionError(f"{mode} state changed after clearing live gradients")
    if not capture.state.is_finite():
        raise FloatingPointError(f"{mode} logical gradient is nonfinite")
    if any(item.requires_grad or item.grad_fn is not None for item in capture.state):
        raise AssertionError(f"{mode} logical state retained an autograd graph")

    prompt_sha_after = _tensor_sha256(param_index[0].parameter)
    frozen_sha_after = hash_frozen_parameters(model)
    scheduler_after = (
        int(trainer.sched.last_epoch),
        int(trainer.sched.successor.last_epoch),
        float(trainer.get_current_lr()),
    )
    if prompt_sha_after != prompt_sha_before or not torch.equal(
        param_index[0].parameter.detach(), prompt_before
    ):
        raise AssertionError(f"{mode} prompt changed without an optimizer step")
    if frozen_sha_after != frozen_sha_before:
        raise AssertionError(f"{mode} frozen parameter changed")
    if scheduler_after != scheduler_before:
        raise AssertionError(f"{mode} scheduler changed")

    state = capture.state
    record = {
        "mode": mode,
        "model_parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "prompt_parameter_dtype": str(param_index[0].parameter.dtype),
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "loss_value_non_scientific": float(loss.item()),
        "live_gradient_dtype_before_capture": live_dtype_before,
        "live_gradient_dtypes_before_unscale": list(
            capture.live_dtypes_before_unscale
        ),
        "live_gradient_dtypes_after_unscale": list(capture.live_dtypes_after_unscale),
        "scaled_live_gradient_norm_before_unscale": scaled_live_norm
        if mode == "amp"
        else None,
        "gradient_scaling_active": capture.scaling_active,
        "scaler_scale": capture.scale,
        "amp_default_scale_probe": amp_default_scale_probe,
        "authoritative_unscale_performed": capture.authoritative_unscale_performed,
        "gradient_state_dtypes": [str(item.dtype) for item in state],
        "gradient_state_devices": [str(item.device) for item in state],
        "gradient_state_norm": float(state.norm().item()),
        "gradient_state_finite": state.is_finite(),
        "gradient_state_detached_no_graph": all(
            not item.requires_grad and item.grad_fn is None for item in state
        ),
        "gradient_state_survived_grad_clear_exactly": True,
        "prompt_sha256_before": prompt_sha_before,
        "prompt_sha256_after": prompt_sha_after,
        "prompt_unchanged": True,
        "frozen_sha256_before": frozen_sha_before,
        "frozen_sha256_after": frozen_sha_after,
        "frozen_parameters_unchanged": True,
        "optimizer_steps": optimizer_counter["count"],
        "scheduler_unchanged": True,
        "batch_identity": batch_identity,
        "param_index_fingerprint": param_index.fingerprint,
        "precision_phase_after_capture": controller.phase,
    }
    del trainer, model, controller, param_index, batch, image, label, logits, loss
    _release_cuda_cache()
    return state, record


def run(args: argparse.Namespace) -> Path:
    coop_root = REPO_ROOT
    project_root = coop_root.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    data_root = Path(args.root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output_path = Path(args.output).resolve()
    output_root = output_path.parent

    if _run(["git", "branch", "--show-current"], coop_root) != "sample-full-gradient":
        raise AssertionError("Task-7 validation requires branch sample-full-gradient")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK6_COOP_SHA, "HEAD"],
        cwd=coop_root,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-6 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Dassl SHA differs from the pinned baseline")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Dassl worktree is not clean")
    if not torch.cuda.is_available():
        raise RuntimeError("The real Task-7 precision integration requires CUDA")

    checkpoint_path = clip_cache / "ViT-B-16.pt"
    if sha256_file(checkpoint_path).lower() != EXPECTED_CLIP_SHA256:
        raise AssertionError("Task-3 CLIP checkpoint hash changed")
    split_path = data_root / "dtd" / "split_zhou_DescribableTextures.json"
    cache_path = data_root / "dtd" / "split_fewshot" / "shot_4-seed_1.pkl"
    if sha256_file(split_path) != DTD_SPLIT_SHA:
        raise AssertionError("DTD fixed-split hash changed")
    if sha256_file(cache_path) != DTD_CACHE_SHA:
        raise AssertionError("DTD 4-shot seed-1 cache hash changed")

    states = {}
    modes = {}
    for mode in PRECISION_MODES:
        state, record = _capture_mode(
            coop_root, data_root, output_root, clip_cache, mode
        )
        states[mode] = state
        modes[mode] = record

    reference_batch = modes["fp32"]["batch_identity"]
    for mode in PRECISION_MODES[1:]:
        if modes[mode]["batch_identity"] != reference_batch:
            raise AssertionError(f"{mode} did not use the same controlled batch")
        if modes[mode]["param_index_fingerprint"] != modes["fp32"][
            "param_index_fingerprint"
        ]:
            raise AssertionError(f"{mode} ParamIndex differs from FP32")

    fp32_state = states["fp32"]
    fp32_norm = float(fp32_state.norm().item())
    comparisons = {}
    for mode in ("fp16", "amp"):
        candidate = states[mode]
        candidate_norm = float(candidate.norm().item())
        cosine = float(
            candidate.dot(fp32_state).item() / (candidate_norm * fp32_norm)
        )
        cosine = max(-1.0, min(1.0, cosine))
        relative_norm_difference = abs(candidate_norm - fp32_norm) / fp32_norm
        passed = (
            cosine >= MIN_COSINE_WITH_FP32
            and relative_norm_difference <= MAX_RELATIVE_NORM_DIFFERENCE
        )
        comparisons[mode] = {
            "fp32_gradient_norm": fp32_norm,
            "candidate_gradient_norm": candidate_norm,
            "cosine_with_fp32": cosine,
            "relative_norm_difference": relative_norm_difference,
            "minimum_cosine_required": MIN_COSINE_WITH_FP32,
            "maximum_relative_norm_difference_allowed": MAX_RELATIVE_NORM_DIFFERENCE,
            "passed": passed,
        }
        if not passed:
            raise AssertionError(
                f"{mode} gradient is outside predeclared precision tolerance: "
                f"{comparisons[mode]}"
            )

    report = {
        "schema_version": "sample_fg.precision_integration.v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "coop_upstream_provenance_sha": COOP_UPSTREAM_SHA,
            "coop_task6_sha": TASK6_COOP_SHA,
            "coop_runtime_git": _git_record(coop_root),
            "dassl_pinned_sha": DASSL_SHA,
            "dassl_runtime_git": _git_record(dassl_root),
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "data": {
            "dataset": "DTD",
            "shots": 4,
            "seed": 1,
            "classes": "base",
            "batch_size": 2,
            "workers": 0,
            "split_sha256": DTD_SPLIT_SHA,
            "fewshot_cache_sha256": DTD_CACHE_SHA,
            "same_controlled_batch_all_modes": True,
        },
        "precision_policy": {
            "supported_modes_validated": list(PRECISION_MODES),
            "gradient_state_dtype": "torch.float32",
            "first_order": True,
            "create_graph": False,
            "retain_graph": False,
            "amp_unscale": "GradScaler.unscale_(optimizer), exactly once",
            "minimum_cosine_with_fp32": MIN_COSINE_WITH_FP32,
            "maximum_relative_norm_difference": MAX_RELATIVE_NORM_DIFFERENCE,
        },
        "modes": modes,
        "cross_precision": comparisons,
        "scope": {
            "optimizer_steps": 0,
            "rng_isolation": False,
            "full_gradient_service": False,
            "ema": False,
            "exact_estimator": False,
            "periodic_estimator": False,
            "sam": False,
            "sample": False,
            "perturbation": False,
        },
    }
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "mode_summary": {
                    mode: {
                        "prompt_dtype": modes[mode]["prompt_parameter_dtype"],
                        "logits_dtype": modes[mode]["logits_dtype"],
                        "loss_dtype": modes[mode]["loss_dtype"],
                        "live_gradient_dtype": modes[mode][
                            "live_gradient_dtype_before_capture"
                        ],
                        "gradient_state_dtype": modes[mode][
                            "gradient_state_dtypes"
                        ],
                        "gradient_state_norm": modes[mode]["gradient_state_norm"],
                        "scaler_scale": modes[mode]["scaler_scale"],
                    }
                    for mode in PRECISION_MODES
                },
                "cross_precision": comparisons,
                "report": str(output_path),
            },
            indent=2,
        )
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Prepared CoOp data root")
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
