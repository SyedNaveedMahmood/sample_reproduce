"""Run the non-scientific Task-3 CoOp base-to-new anchor smoke."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torchvision

from clip import clip
from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    EXPECTED_CLIP_KEY,
    EXPECTED_CLIP_SHA256,
    audit_prompt_only_training,
    build_coop_trainer,
    build_smoke_cfg,
    evaluate_bounded,
    hash_frozen_parameters,
    run_bounded_coop_steps,
    sha256_file,
    unwrap_model,
)


COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
TASK2_COOP_SHA = "40a7a187d077dc081a4d2722ac98fec845dd81d3"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
DTD_SPLIT_SHA = "F26EBECCD2B58E68D70F07A0DC39FE49BF7B69024BAC34396B954DFD87969C38"
DTD_CACHE_SHA = "81EE5B688EC9D80BBE424522A7638FCE5AAE84F07EC40928CBBA2B57B2B142AD"
MAX_OPTIMIZER_STEPS = 3
EVAL_SAMPLE_LIMIT = 8


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _verify_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    expected_url = clip._MODELS[EXPECTED_CLIP_KEY]
    encoded_hash = expected_url.split("/")[-2]
    if encoded_hash.lower() != EXPECTED_CLIP_SHA256:
        raise AssertionError(
            f"Pinned source encodes {encoded_hash}, expected {EXPECTED_CLIP_SHA256}"
        )
    observed_hash = sha256_file(checkpoint_path)
    if observed_hash.lower() != encoded_hash.lower():
        raise AssertionError(
            f"CLIP checkpoint hash mismatch: {observed_hash} != {encoded_hash}"
        )
    stat = checkpoint_path.stat()
    return {
        "model_key": EXPECTED_CLIP_KEY,
        "source_url": expected_url,
        "local_path": str(checkpoint_path),
        "byte_size": stat.st_size,
        "expected_sha256": encoded_hash.upper(),
        "observed_sha256": observed_hash,
        "hash_match": True,
        "local_mtime_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "modified_or_converted": False,
    }


def _expected_initial_context(checkpoint_path: Path, text: str) -> torch.Tensor:
    jit_model = torch.jit.load(str(checkpoint_path), map_location="cpu").eval()
    tokens = clip.tokenize(text)
    token_count = len(text.replace("_", " ").split(" "))
    with torch.no_grad():
        expected = jit_model.token_embedding(tokens)[0, 1 : 1 + token_count].float()
    del jit_model
    gc.collect()
    return expected


def _release_cuda_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _gpu_driver() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def run(args: argparse.Namespace) -> Path:
    coop_root = Path(__file__).resolve().parents[1]
    project_root = coop_root.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    data_root = Path(args.root).resolve(strict=True)
    output_root = Path(args.output_dir).resolve()
    checkpoint_cache_dir = Path(args.clip_cache).resolve(strict=True)
    checkpoint_path = checkpoint_cache_dir / "ViT-B-16.pt"

    task2_is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK2_COOP_SHA, "HEAD"],
        cwd=coop_root,
        check=False,
    ).returncode == 0
    if not task2_is_ancestor:
        raise AssertionError("Task-3 smoke requires the accepted Task-2 commit as an ancestor")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Dassl SHA differs from the pinned baseline")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Dassl worktree is not clean")
    if not torch.cuda.is_available():
        raise RuntimeError("The real Task-3 integration smoke requires CUDA")

    clip_record = _verify_checkpoint(checkpoint_path)
    split_path = data_root / "dtd" / "split_zhou_DescribableTextures.json"
    cache_path = data_root / "dtd" / "split_fewshot" / "shot_4-seed_1.pkl"
    if sha256_file(split_path) != DTD_SPLIT_SHA:
        raise AssertionError("DTD split hash differs from the Task-2 fixed split")
    if sha256_file(cache_path) != DTD_CACHE_SHA:
        raise AssertionError("DTD 4-shot seed-1 cache differs from Task 2")

    output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = build_smoke_cfg(
        coop_root, data_root, output_root / "base_runtime", "base"
    )
    _write_text(output_root / "resolved_config_base.yaml", base_cfg.dump())

    set_random_seed(base_cfg.SEED)
    torch.backends.cudnn.benchmark = True
    trainer = build_coop_trainer(base_cfg, checkpoint_cache_dir)
    model = unwrap_model(trainer.model)
    prompt = model.prompt_learner.ctx
    nominal_n_ctx = int(base_cfg.TRAINER.COOP.N_CTX)
    effective_n_ctx = int(model.prompt_learner.n_ctx)
    if effective_n_ctx != 4 or tuple(prompt.shape[:1]) != (4,):
        raise AssertionError(
            f"Expected effective context length 4, observed {effective_n_ctx}/{prompt.shape}"
        )
    if base_cfg.TRAINER.COOP.CTX_INIT != "a photo of a":
        raise AssertionError("Unexpected CoOp context initialization string")
    if bool(base_cfg.TRAINER.COOP.CSC):
        raise AssertionError("Task-3 anchor requires unified context")

    expected_ctx = _expected_initial_context(
        checkpoint_path, base_cfg.TRAINER.COOP.CTX_INIT
    )
    initial_ctx_cpu = prompt.detach().cpu().float().clone()
    if not torch.equal(initial_ctx_cpu, expected_ctx):
        raise AssertionError("Runtime context does not equal pinned CLIP token initialization")
    del expected_ctx

    audit = audit_prompt_only_training(trainer.model, trainer.optim)
    if len(trainer.dm.dataset.classnames) != 24:
        raise AssertionError("DTD base dataset does not expose 24 classes")
    if len(trainer.dm.dataset.train_x) != 96:
        raise AssertionError("DTD base 4-shot source does not contain 96 records")
    if len(trainer.train_loader_x) != 48:
        raise AssertionError("Batch-2 DTD smoke loader should contain 48 batches")

    frozen_hash_before = hash_frozen_parameters(trainer.model)
    prompt_norm_before = float(torch.linalg.vector_norm(prompt.detach().float()).item())
    scheduler_before = {
        "wrapper_last_epoch": int(trainer.sched.last_epoch),
        "successor_last_epoch": int(trainer.sched.successor.last_epoch),
        "current_lr": float(trainer.get_current_lr()),
    }
    step_records = run_bounded_coop_steps(trainer, MAX_OPTIMIZER_STEPS)
    scheduler_after = {
        "wrapper_last_epoch": int(trainer.sched.last_epoch),
        "successor_last_epoch": int(trainer.sched.successor.last_epoch),
        "current_lr": float(trainer.get_current_lr()),
    }
    if scheduler_after != scheduler_before:
        raise AssertionError("Scheduler advanced before the inherited epoch boundary")

    learned_ctx_cpu = prompt.detach().cpu().float().clone()
    prompt_norm_after = float(torch.linalg.vector_norm(prompt.detach().float()).item())
    prompt_delta_norm = float(
        torch.linalg.vector_norm(learned_ctx_cpu - initial_ctx_cpu).item()
    )
    if prompt_delta_norm <= 0:
        raise AssertionError("Three real steps did not change the prompt context")
    frozen_hash_after = hash_frozen_parameters(trainer.model)
    if frozen_hash_after != frozen_hash_before:
        raise AssertionError("A retained frozen CLIP parameter changed")

    checkpoint_root = output_root / "checkpoints"
    trainer.save_model(epoch=0, directory=str(checkpoint_root))
    checkpoint_file = (
        checkpoint_root / "prompt_learner" / "model.pth.tar-1"
    )
    checkpoint_record = {
        "path": str(checkpoint_file),
        "sha256": sha256_file(checkpoint_file),
        "byte_size": checkpoint_file.stat().st_size,
        "format": "pinned Dassl TrainerBase.save_model / CoOp prompt_learner",
        "saved_epoch_field": 1,
    }
    del model, prompt, trainer
    _release_cuda_cache()

    # Reconstruct the official BASE path and load via CoOp.load_model.
    set_random_seed(base_cfg.SEED)
    base_reload = build_coop_trainer(base_cfg, checkpoint_cache_dir)
    base_reload.load_model(str(checkpoint_root), epoch=1)
    base_reload_ctx = (
        unwrap_model(base_reload.model).prompt_learner.ctx.detach().cpu().float().clone()
    )
    if not torch.equal(base_reload_ctx, learned_ctx_cpu):
        raise AssertionError("Base checkpoint reload did not exactly restore context")
    base_evaluation = evaluate_bounded(base_reload, EVAL_SAMPLE_LIMIT, 24)
    base_after_eval = (
        unwrap_model(base_reload.model).prompt_learner.ctx.detach().cpu().float()
    )
    if not torch.equal(base_after_eval, learned_ctx_cpu):
        raise AssertionError("Base evaluation changed learned context")
    del base_reload
    _release_cuda_cache()

    # Reconstruct the official NEW path and load the same unified context.
    new_cfg = build_smoke_cfg(
        coop_root, data_root, output_root / "new_runtime", "new"
    )
    _write_text(output_root / "resolved_config_new.yaml", new_cfg.dump())
    set_random_seed(new_cfg.SEED)
    new_trainer = build_coop_trainer(new_cfg, checkpoint_cache_dir)
    if len(new_trainer.dm.dataset.classnames) != 23:
        raise AssertionError("DTD new dataset does not expose 23 classes")
    new_trainer.load_model(str(checkpoint_root), epoch=1)
    new_ctx_before_eval = (
        unwrap_model(new_trainer.model).prompt_learner.ctx.detach().cpu().float().clone()
    )
    if not torch.equal(new_ctx_before_eval, learned_ctx_cpu):
        raise AssertionError("New-class model did not reuse the learned base context")
    new_evaluation = evaluate_bounded(new_trainer, EVAL_SAMPLE_LIMIT, 23)
    new_ctx_after_eval = (
        unwrap_model(new_trainer.model).prompt_learner.ctx.detach().cpu().float()
    )
    if not torch.equal(new_ctx_after_eval, learned_ctx_cpu):
        raise AssertionError("New-class evaluation changed the learned context")
    del new_trainer
    _release_cuda_cache()

    coop_git = _git_record(coop_root)
    dassl_git = _git_record(dassl_root)
    gpu_props = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "sample_fg.coop_anchor_smoke.v1",
        "run": {
            "identity": "non-scientific Stage-0 smoke",
            "smoke": True,
            "allow_scientific_summary": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
            "accuracy_interpretation": "non-scientific execution check only",
        },
        "source": {
            "coop_upstream_provenance_sha": COOP_UPSTREAM_SHA,
            "coop_task2_start_sha": TASK2_COOP_SHA,
            "coop_runtime_git": coop_git,
            "dassl_pinned_sha": DASSL_SHA,
            "dassl_runtime_git": dassl_git,
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_torch_compiled_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_memory_bytes": gpu_props.total_memory,
            "gpu_driver_version": _gpu_driver(),
            "gpu_device_index": 0,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "clip_checkpoint": clip_record,
        "data": {
            "dataset": "DTD",
            "data_root": str(data_root),
            "shots": 4,
            "seed": 1,
            "training_classes": "base",
            "base_class_count": 24,
            "new_class_count": 23,
            "base_selected_train_count": 96,
            "split_path": str(split_path),
            "split_sha256": DTD_SPLIT_SHA,
            "fewshot_cache_path": str(cache_path),
            "fewshot_cache_sha256": DTD_CACHE_SHA,
            "normal_train_sampler": type(base_cfg.DATALOADER.TRAIN_X.SAMPLER).__name__
            if not isinstance(base_cfg.DATALOADER.TRAIN_X.SAMPLER, str)
            else base_cfg.DATALOADER.TRAIN_X.SAMPLER,
            "normal_train_drop_last": True,
            "normal_train_loader_batches": 48,
        },
        "configuration": {
            "resolved_trainer": "CoOp",
            "backbone": base_cfg.MODEL.BACKBONE.NAME,
            "prompt_learner": "CoOp",
            "unified_context": not bool(base_cfg.TRAINER.COOP.CSC),
            "nominal_config_n_ctx": nominal_n_ctx,
            "effective_runtime_n_ctx": effective_n_ctx,
            "context_init": base_cfg.TRAINER.COOP.CTX_INIT,
            "class_token_position": base_cfg.TRAINER.COOP.CLASS_TOKEN_POSITION,
            "precision": base_cfg.TRAINER.COOP.PREC,
            "train_batch_size": base_cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            "test_batch_size": base_cfg.DATALOADER.TEST.BATCH_SIZE,
            "workers": base_cfg.DATALOADER.NUM_WORKERS,
            "optimizer": {
                "name": base_cfg.OPTIM.NAME,
                "base_lr": base_cfg.OPTIM.LR,
                "weight_decay": base_cfg.OPTIM.WEIGHT_DECAY,
                "momentum": base_cfg.OPTIM.MOMENTUM,
                "dampening": base_cfg.OPTIM.SGD_DAMPNING,
                "nesterov": base_cfg.OPTIM.SGD_NESTEROV,
            },
            "scheduler": {
                "name": base_cfg.OPTIM.LR_SCHEDULER,
                "max_epoch": base_cfg.OPTIM.MAX_EPOCH,
                "warmup_epoch": base_cfg.OPTIM.WARMUP_EPOCH,
                "warmup_type": base_cfg.OPTIM.WARMUP_TYPE,
                "warmup_constant_lr": base_cfg.OPTIM.WARMUP_CONS_LR,
                "warmup_recount": base_cfg.OPTIM.WARMUP_RECOUNT,
                "runtime_class": "ConstantWarmupSchedulerCompat",
                "compatibility_adapter": True,
                "step_convention": "once at inherited final batch of each epoch",
                "state_before_smoke": scheduler_before,
                "state_after_smoke": scheduler_after,
            },
        },
        "prompt_and_freezing": {
            **audit,
            "initial_context_matches_pinned_clip_tokens_exactly": True,
            "prompt_norm_before": prompt_norm_before,
            "prompt_norm_after": prompt_norm_after,
            "prompt_delta_norm": prompt_delta_norm,
            "frozen_sha256_before": frozen_hash_before,
            "frozen_sha256_after": frozen_hash_after,
            "frozen_parameters_unchanged": True,
        },
        "training_steps": step_records,
        "checkpoint": {
            **checkpoint_record,
            "base_reload_context_exact": True,
            "new_reload_context_exact": True,
        },
        "evaluation": {
            "base": base_evaluation,
            "new": {
                **new_evaluation,
                "same_learned_base_context_reused": True,
                "retrained": False,
            },
        },
        "scope": {
            "ordinary_coop_forward_backward": True,
            "sam": False,
            "sample": False,
            "param_index": False,
            "gradient_state": False,
            "full_gradient_service": False,
        },
        "status": "PASS",
    }
    report_path = output_root / "smoke_report.json"
    _write_json(report_path, report)
    print(json.dumps({"status": "PASS", "report": str(report_path)}, indent=2))
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Prepared CoOp data root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-cache", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
