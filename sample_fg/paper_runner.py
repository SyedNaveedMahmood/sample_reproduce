"""Scientific DTD/EuroSAT CoOp, SAM, and SAMPLe run lifecycle.

Dry-run performs provenance/configuration validation only. The training entry
point is deliberately separate from the bounded Stage-0 smoke script.  The
public parser in this module remains the fixed R2 paper-reproduction surface;
the extension runner builds validated plans for Exact and Periodic estimators
through the shared lifecycle below.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
import yaml
from torch.nn import functional as F

from dassl.config import get_cfg_default
from dassl.utils import set_random_seed
from train import extend_cfg

from .checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointProgress,
    load_scientific_checkpoint,
    save_scientific_checkpoint,
)
from .coop_anchor import (
    EXPECTED_CLIP_KEY,
    EXPECTED_CLIP_SHA256,
    audit_prompt_only_training,
    build_coop_trainer,
    hash_frozen_parameters,
    unwrap_model,
)
from .data_protocol import COOP_COMMIT, DASSL_COMMIT, DATASET_SPECS, load_dataset
from .diagnostic_schedule import DiagnosticCoordinator, DiagnosticSchedule
from .environment import capture_environment
from .estimators import (
    EMAEstimator,
    ExactEstimator,
    GlobalGradientEstimator,
    PeriodicEstimator,
)
from .full_gradient import (
    FullGradientSource,
    FullGradientService,
    FullGradientSweepMetadata,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from .param_index import ParamIndex
from .perturbation import PromptPerturbation
from .precision import OptimizerStepResult, PrecisionController
from .results import (
    METRICS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    RunAccounting,
    RunArtifacts,
    RunIdentity,
    bind_run_identity,
    load_jsonl,
    resolve_config,
)
from .step_engine import SAMStepRecord, SAMPLeStepRecord, StepEngine


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "sample_fg" / "paper_reproduction.yaml"
SUPPORTED_DATASETS = ("dtd", "eurosat")
SUPPORTED_SEEDS = (1, 2, 3)
SUPPORTED_METHODS = ("coop", "sam", "sample")
PAPER_SHOTS = 16
PAPER_EPOCHS = 200
FULL_GRADIENT_MICRO_BATCH_SIZE = 32
NORM_EPS = 1e-12


class ScientificRunnerError(RuntimeError):
    """Raised before or during an invalid scientific run lifecycle."""


@dataclass(frozen=True)
class MethodSelection:
    method: str
    estimator: str
    method_tag: str
    estimator_tag: str
    rho: float | None
    alpha: float | None
    ema_lambda: float | None
    refresh_k_steps: int | None


@dataclass(frozen=True)
class ScientificPlan:
    dataset: str
    shots: int
    seed: int
    experiment_id: str
    selection: MethodSelection
    data_root: Path
    manifest_root: Path
    manifest_path: Path
    clip_cache: Path
    clip_checkpoint: Path
    output_root: Path
    config_path: Path
    resume_from: Path | None
    recovery_interval_epochs: int
    source: FullGradientSource
    manifest: dict[str, Any]
    resolved_config: dict[str, Any]
    steps_per_epoch: int
    epochs: int
    diagnostic_interval_steps: int | None
    full_gradient_micro_batch_size: int

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.steps_per_epoch


@dataclass(frozen=True)
class CoOpStepRecord:
    method: str
    optimizer_step: int
    loss_current: float
    batch_gradient_norm: float
    final_gradient_norm: float
    optimizer_step_result: OptimizerStepResult


@dataclass
class RuntimeState:
    trainer: Any
    model: torch.nn.Module
    param_index: ParamIndex
    precision: PrecisionController
    perturbation: PromptPerturbation | None
    engine: StepEngine | None
    estimator: GlobalGradientEstimator | None
    full_gradient_loader: Any | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_method(method: str, estimator: str) -> MethodSelection:
    """Resolve the only three methods authorized for the primary matrix."""

    if method not in SUPPORTED_METHODS:
        raise ScientificRunnerError(f"Unsupported method: {method!r}")
    if method in {"coop", "sam"} and estimator != "none":
        raise ScientificRunnerError(f"{method} requires --estimator none")
    if method == "sample" and estimator != "ema":
        raise ScientificRunnerError("sample requires --estimator ema")
    return MethodSelection(
        method=method,
        estimator=estimator,
        method_tag=method,
        estimator_tag=estimator,
        rho=0.05 if method in {"sam", "sample"} else None,
        alpha=0.0015 if method == "sample" else None,
        ema_lambda=0.15 if method == "sample" else None,
        refresh_k_steps=None,
    )


def build_scientific_cfg(
    *,
    dataset: str,
    seed: int,
    data_root: Path,
    output_dir: Path,
    config_path: Path,
    class_subsample: str = "base",
    shots: int = PAPER_SHOTS,
    epochs: int = PAPER_EPOCHS,
):
    """Resolve the pinned CoOp/Dassl configuration for one scientific cell."""

    if dataset not in SUPPORTED_DATASETS:
        raise ScientificRunnerError(f"Unsupported dataset: {dataset!r}")
    if class_subsample not in {"base", "new"}:
        raise ScientificRunnerError(f"Invalid class subsample: {class_subsample!r}")
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"))
    cfg.merge_from_file(
        str(REPO_ROOT / "configs" / "trainers" / "CoOp" / "vit_b16_ctxv1.yaml")
    )
    cfg.merge_from_file(str(config_path))
    cfg.DATASET.ROOT = str(data_root)
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ScientificRunnerError("shots must be a positive integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ScientificRunnerError("epochs must be a positive integer")
    cfg.DATASET.NUM_SHOTS = shots
    cfg.DATASET.SUBSAMPLE_CLASSES = class_subsample
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SEED = seed
    cfg.TRAINER.NAME = "CoOp"
    cfg.OPTIM.MAX_EPOCH = epochs
    cfg.freeze()
    _assert_paper_cfg(cfg, expected_shots=shots, expected_epochs=epochs)
    return cfg


def _assert_paper_cfg(
    cfg,
    *,
    expected_shots: int = PAPER_SHOTS,
    expected_epochs: int = PAPER_EPOCHS,
) -> None:
    observed = {
        "shots": int(cfg.DATASET.NUM_SHOTS),
        "train_batch_size": int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
        "test_batch_size": int(cfg.DATALOADER.TEST.BATCH_SIZE),
        "num_workers": int(cfg.DATALOADER.NUM_WORKERS),
        "backbone": str(cfg.MODEL.BACKBONE.NAME),
        "precision": str(cfg.TRAINER.COOP.PREC),
        "ctx_init": str(cfg.TRAINER.COOP.CTX_INIT),
        "csc": bool(cfg.TRAINER.COOP.CSC),
        "class_token_position": str(cfg.TRAINER.COOP.CLASS_TOKEN_POSITION),
        "optimizer": str(cfg.OPTIM.NAME).lower(),
        "lr": float(cfg.OPTIM.LR),
        "weight_decay": float(cfg.OPTIM.WEIGHT_DECAY),
        "momentum": float(cfg.OPTIM.MOMENTUM),
        "nesterov": bool(cfg.OPTIM.SGD_NESTEROV),
        "epochs": int(cfg.OPTIM.MAX_EPOCH),
        "scheduler": str(cfg.OPTIM.LR_SCHEDULER),
        "warmup_epoch": int(cfg.OPTIM.WARMUP_EPOCH),
        "warmup_type": str(cfg.OPTIM.WARMUP_TYPE),
        "warmup_lr": float(cfg.OPTIM.WARMUP_CONS_LR),
        "final_model": str(cfg.TEST.FINAL_MODEL),
    }
    expected = {
        "shots": expected_shots,
        "train_batch_size": 32,
        "test_batch_size": 100,
        "num_workers": 8,
        "backbone": "ViT-B/16",
        "precision": "fp16",
        "ctx_init": "a photo of a",
        "csc": False,
        "class_token_position": "end",
        "optimizer": "sgd",
        "lr": 0.002,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": False,
        "epochs": expected_epochs,
        "scheduler": "cosine",
        "warmup_epoch": 1,
        "warmup_type": "constant",
        "warmup_lr": 1e-5,
        "final_model": "last_step",
    }
    if observed != expected:
        raise ScientificRunnerError(
            f"Resolved CoOp config differs from the paper protocol: {observed}"
        )


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ScientificRunnerError(f"Cannot read Task-2 manifest: {path}") from error
    if not isinstance(value, dict):
        raise ScientificRunnerError("Task-2 manifest root must be an object")
    return value


def _resolved_scientific_config(
    *,
    cfg,
    selection: MethodSelection,
    source: FullGradientSource,
    manifest: Mapping[str, Any],
    data_root: Path,
    manifest_root: Path,
    output_root: Path,
    clip_checkpoint: Path,
    config_path: Path,
    experiment_id: str,
    recovery_interval_epochs: int,
    diagnostic_interval_steps: int | None = None,
    full_gradient_micro_batch_size: int = FULL_GRADIENT_MICRO_BATCH_SIZE,
    notes: str = "Primary DTD/EuroSAT 16-shot CoOp paper reproduction",
    campaign_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normal = manifest.get("normal_train_loader")
    if not isinstance(normal, Mapping):
        raise ScientificRunnerError("Task-2 manifest lacks normal loader metadata")
    steps_per_epoch = int(normal.get("steps_per_epoch", 0))
    if steps_per_epoch < 1:
        raise ScientificRunnerError("Task-2 manifest has invalid steps_per_epoch")
    interval_steps = recovery_interval_epochs * steps_per_epoch
    diagnostics_enabled = selection.method == "sample"
    if diagnostic_interval_steps is None:
        diagnostic_interval_steps = steps_per_epoch if diagnostics_enabled else None
    if diagnostics_enabled and (
        isinstance(diagnostic_interval_steps, bool)
        or not isinstance(diagnostic_interval_steps, int)
        or diagnostic_interval_steps < 1
    ):
        raise ScientificRunnerError(
            "SAMPLe diagnostic interval must be a positive step count"
        )
    if not diagnostics_enabled and diagnostic_interval_steps is not None:
        raise ScientificRunnerError(
            "Only SAMPLe owns exact-reference diagnostics"
        )
    if (
        isinstance(full_gradient_micro_batch_size, bool)
        or not isinstance(full_gradient_micro_batch_size, int)
        or full_gradient_micro_batch_size < 1
    ):
        raise ScientificRunnerError(
            "full-gradient micro-batch size must be a positive integer"
        )
    payload: dict[str, Any] = {
            "run": {
                "experiment_id": experiment_id,
                "output_root": str(output_root),
                "notes": notes,
                "smoke": False,
            },
            "data": {
                "dataset": source.dataset,
                "root": str(data_root),
                "manifest_root": str(manifest_root),
                "manifest_path": str(source.manifest_path),
                "shots": source.shots,
                "seed": source.seed,
                "class_subsample": "base",
                "split_policy": "official_coop_fixed",
                "require_split_checksum": True,
                "split_sha256": source.official_split_sha256,
                "fewshot_cache_sha256": source.fewshot_cache_sha256,
                "train_batch_size": int(cfg.DATALOADER.TRAIN_X.BATCH_SIZE),
                "test_batch_size": int(cfg.DATALOADER.TEST.BATCH_SIZE),
                "num_workers": int(cfg.DATALOADER.NUM_WORKERS),
                "preserve_upstream_drop_last": True,
                "augmentation_policy": "pinned_coop_train_transform",
                "seed_policy": "coop_legacy_plus_isolated_fullgrad_v1",
                "selected_source_fingerprint": source.fingerprint,
                "selected_count": len(source),
                "normal_steps_per_epoch": steps_per_epoch,
                "normal_samples_consumed_per_epoch": int(
                    normal.get("samples_consumed_per_epoch", 0)
                ),
            },
            "model": {
                "backbone": EXPECTED_CLIP_KEY,
                "prompt_learner": "CoOp",
                "nominal_n_ctx": int(cfg.TRAINER.COOP.N_CTX),
                "effective_n_ctx": 4,
                "ctx_init": str(cfg.TRAINER.COOP.CTX_INIT),
                "class_specific_context": bool(cfg.TRAINER.COOP.CSC),
                "class_token_position": str(cfg.TRAINER.COOP.CLASS_TOKEN_POSITION),
                "freeze_clip": True,
                "checkpoint_path": str(clip_checkpoint),
                "checkpoint_sha256": EXPECTED_CLIP_SHA256,
            },
            "method": {
                "name": selection.method,
                "rho": selection.rho,
                "alpha": selection.alpha,
                "ema_lambda": selection.ema_lambda,
                "norm_eps": NORM_EPS,
                "nonfinite_policy": "abort",
                "first_order_stop_gradient": True,
            },
            "estimator": {
                "mode": selection.estimator,
                "refresh_k_steps": selection.refresh_k_steps,
                "full_gradient_micro_batch_size": full_gradient_micro_batch_size,
                "full_gradient_num_workers": 0,
                "full_gradient_transform_policy": (
                    "train_aug_isolated_conditional_exact_v1"
                ),
                "full_gradient_accum_dtype": "fp32",
            },
            "diagnostics": {
                "enabled": diagnostics_enabled,
                "full_gradient_interval_policy": (
                    "once_per_normal_epoch" if diagnostics_enabled else None
                ),
                "full_gradient_interval_steps": (
                    diagnostic_interval_steps if diagnostics_enabled else None
                ),
                "log_step_interval": 1,
                "eval_interval_epochs": None,
                "store_gradient_vectors": False,
                "write_separate_jsonl": True,
                "purity_assertions": True,
            },
            "optim": {
                "name": str(cfg.OPTIM.NAME).lower(),
                "lr": float(cfg.OPTIM.LR),
                "weight_decay": float(cfg.OPTIM.WEIGHT_DECAY),
                "momentum": float(cfg.OPTIM.MOMENTUM),
                "nesterov": bool(cfg.OPTIM.SGD_NESTEROV),
                "max_epoch": int(cfg.OPTIM.MAX_EPOCH),
                "scheduler": str(cfg.OPTIM.LR_SCHEDULER),
                "warmup_epoch": int(cfg.OPTIM.WARMUP_EPOCH),
                "warmup_type": str(cfg.OPTIM.WARMUP_TYPE),
                "warmup_cons_lr": float(cfg.OPTIM.WARMUP_CONS_LR),
                "scheduler_step_unit": "epoch",
            },
            "runtime": {
                "device": "cuda:0",
                "precision": "coop_fp16",
                "coop_precision": "fp16",
                "gradient_state_dtype": "fp32",
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_sync_timing": True,
            },
            "checkpoint": {
                "enabled": True,
                "recovery_interval_epochs": recovery_interval_epochs,
                "recovery_interval_steps": interval_steps,
                "save_boundary": "completed_epoch_only",
                "normal_loader_resume_policy": (
                    "epoch_boundary_global_rng_restore_workers_8_v1"
                ),
                "save_final": True,
                "save_best": False,
                "resume_from": None,
                "strict_config_match": True,
                "save_rng_state": True,
                "format": CHECKPOINT_SCHEMA_VERSION,
            },
            "logging": {
                "metrics_format": "jsonl",
                "console_level": "human_only",
                "capture_package_freeze": True,
                "capture_git_diff_hash": True,
                "capture_gpu_memory": True,
                "schema_version": "sample_fg.logging.v1",
            },
            "smoke": {
                "max_optimizer_steps": None,
                "limit_train_samples_per_class": None,
                "limit_eval_samples": None,
                "force_num_workers_zero": False,
                "allow_scientific_summary": True,
            },
            "provenance": {
                "coop_upstream_commit": COOP_COMMIT,
                "dassl_commit": DASSL_COMMIT,
                "runner": "train_sample_fg.py",
                "config_source": str(config_path),
            },
        }
    if campaign_metadata is not None:
        payload["campaign"] = dict(campaign_metadata)
        payload["provenance"]["runner"] = "train_sample_fg_extension.py"
    return resolve_config(payload)


def build_scientific_plan(
    *,
    dataset: str,
    shots: int,
    seed: int,
    experiment_id: str,
    selection: MethodSelection,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    output_root: Path,
    config_path: Path,
    recovery_interval_epochs: int,
    epochs: int = PAPER_EPOCHS,
    diagnostic_interval_steps: int | None = None,
    full_gradient_micro_batch_size: int = FULL_GRADIENT_MICRO_BATCH_SIZE,
    resume_from: Path | None = None,
    notes: str = "Scientific SAMPLe experiment",
    campaign_metadata: Mapping[str, Any] | None = None,
) -> ScientificPlan:
    """Build one path-validated scientific plan for a prevalidated cell.

    Campaign membership is deliberately owned by the caller.  This function
    owns the common dataset, manifest, CLIP, CoOp configuration, and artifact
    validation used by both the fixed R2 and extension entry points.
    """

    if dataset not in SUPPORTED_DATASETS:
        raise ScientificRunnerError(f"dataset must be one of {SUPPORTED_DATASETS}")
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ScientificRunnerError("shots must be a positive integer")
    if seed not in SUPPORTED_SEEDS:
        raise ScientificRunnerError(f"seed must be one of {SUPPORTED_SEEDS}")
    if recovery_interval_epochs < 1:
        raise ScientificRunnerError("recovery interval must be a positive epoch count")
    data_root = Path(data_root).resolve(strict=True)
    manifest_root = Path(manifest_root).resolve(strict=True)
    clip_cache = Path(clip_cache).resolve(strict=True)
    output_root = Path(output_root).resolve()
    config_path = Path(config_path).resolve(strict=True)
    clip_checkpoint = (clip_cache / "ViT-B-16.pt").resolve(strict=True)
    checkpoint_hash = _sha256(clip_checkpoint)
    if checkpoint_hash.lower() != EXPECTED_CLIP_SHA256.lower():
        raise ScientificRunnerError(
            f"CLIP checkpoint SHA-256 differs: {checkpoint_hash}"
        )
    manifest_path = (
        manifest_root
        / dataset
        / f"shots_{shots}"
        / f"seed_{seed}"
        / "data_manifest.json"
    ).resolve(strict=True)
    loaded = load_dataset(data_root, DATASET_SPECS[dataset])
    source = load_full_gradient_source(loaded, manifest_path)
    if (source.dataset, source.shots, source.seed) != (
        dataset,
        shots,
        seed,
    ):
        raise ScientificRunnerError("Validated Task-2 source differs from CLI cell")
    manifest = _load_manifest(manifest_path)
    cfg = build_scientific_cfg(
        dataset=dataset,
        seed=seed,
        data_root=data_root,
        output_dir=output_root / "_configuration_probe",
        config_path=config_path,
        shots=shots,
        epochs=epochs,
    )
    resolved = _resolved_scientific_config(
        cfg=cfg,
        selection=selection,
        source=source,
        manifest=manifest,
        data_root=data_root,
        manifest_root=manifest_root,
        output_root=output_root,
        clip_checkpoint=clip_checkpoint,
        config_path=config_path,
        experiment_id=experiment_id,
        recovery_interval_epochs=recovery_interval_epochs,
        diagnostic_interval_steps=diagnostic_interval_steps,
        full_gradient_micro_batch_size=full_gradient_micro_batch_size,
        notes=notes,
        campaign_metadata=campaign_metadata,
    )
    steps = int(manifest["normal_train_loader"]["steps_per_epoch"])
    resolved_resume = Path(resume_from).resolve(strict=True) if resume_from else None
    return ScientificPlan(
        dataset=dataset,
        shots=shots,
        seed=seed,
        experiment_id=experiment_id,
        selection=selection,
        data_root=data_root,
        manifest_root=manifest_root,
        manifest_path=manifest_path,
        clip_cache=clip_cache,
        clip_checkpoint=clip_checkpoint,
        output_root=output_root,
        config_path=config_path,
        resume_from=resolved_resume,
        recovery_interval_epochs=recovery_interval_epochs,
        source=source,
        manifest=manifest,
        resolved_config=resolved,
        steps_per_epoch=steps,
        epochs=epochs,
        diagnostic_interval_steps=resolved["diagnostics"][
            "full_gradient_interval_steps"
        ],
        full_gradient_micro_batch_size=full_gradient_micro_batch_size,
    )


def build_plan(args: argparse.Namespace) -> ScientificPlan:
    if args.shots != PAPER_SHOTS:
        raise ScientificRunnerError("Primary paper runner requires --shots 16")
    selection = resolve_method(args.method, args.estimator)
    return build_scientific_plan(
        dataset=args.dataset,
        shots=args.shots,
        seed=args.seed,
        experiment_id=args.experiment_id,
        selection=selection,
        data_root=Path(args.data_root),
        manifest_root=Path(args.manifest_root),
        clip_cache=Path(args.clip_cache),
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        recovery_interval_epochs=args.recovery_interval_epochs,
        epochs=PAPER_EPOCHS,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        notes="Primary DTD/EuroSAT 16-shot CoOp paper reproduction",
    )


def dry_run_report(plan: ScientificPlan) -> dict[str, Any]:
    total_steps = plan.total_optimizer_steps
    interval = plan.diagnostic_interval_steps
    diagnostic_points = (
        0 if interval is None else ((total_steps - 1) // interval) + 1
    )
    if plan.selection.estimator == "exact":
        optimization_queries = total_steps
    elif plan.selection.estimator == "periodic":
        refresh_k = plan.selection.refresh_k_steps
        if refresh_k is None:
            raise ScientificRunnerError("Periodic dry-run lacks K")
        optimization_queries = ((total_steps - 1) // refresh_k) + 1
    else:
        optimization_queries = 0
    reused_queries = 0
    if interval is not None and plan.selection.estimator == "exact":
        reused_queries = diagnostic_points
    elif interval is not None and plan.selection.estimator == "periodic":
        refresh_k = plan.selection.refresh_k_steps
        assert refresh_k is not None
        reused_queries = sum(
            1
            for step in range(0, total_steps, interval)
            if step % refresh_k == 0
        )
    diagnostic_only_queries = diagnostic_points - reused_queries
    return {
        "status": "DRY_RUN_VALIDATED",
        "dry_run": True,
        "training_started": False,
        "artifacts_created": False,
        "cell": {
            "experiment_id": plan.experiment_id,
            "dataset": plan.dataset,
            "shots": plan.shots,
            "seed": plan.seed,
            "method": plan.selection.method,
            "estimator": plan.selection.estimator,
            "periodic_k_steps": plan.selection.refresh_k_steps,
        },
        "protocol": {
            "epochs": plan.epochs,
            "steps_per_epoch": plan.steps_per_epoch,
            "normal_batches_per_epoch": plan.steps_per_epoch,
            "total_normal_batches": total_steps,
            "total_optimizer_steps": total_steps,
            "diagnostic_interval_steps": plan.resolved_config["diagnostics"][
                "full_gradient_interval_steps"
            ],
            "expected_diagnostic_points": diagnostic_points,
            "expected_periodic_refresh_count": (
                optimization_queries
                if plan.selection.estimator == "periodic"
                else None
            ),
            "expected_exact_sweeps": optimization_queries
            + diagnostic_only_queries,
            "expected_optimization_exact_queries": optimization_queries,
            "expected_diagnostic_only_exact_queries": diagnostic_only_queries,
            "expected_reused_exact_queries": reused_queries,
            "full_gradient_micro_batch_size": plan.full_gradient_micro_batch_size,
            "recovery_interval_epochs": plan.recovery_interval_epochs,
            "recovery_boundary": "completed_epoch_only",
            "optimizer": plan.resolved_config["optim"],
            "precision": plan.resolved_config["runtime"]["precision"],
            "full_gradient_source": {
                "selected_count": len(plan.source),
                "fingerprint": plan.source.fingerprint,
                "manifest": str(plan.manifest_path),
            },
        },
        "validated": {
            "data_root": str(plan.data_root),
            "manifest": str(plan.manifest_path),
            "selected_source_count": len(plan.source),
            "selected_source_fingerprint": plan.source.fingerprint,
            "clip_checkpoint": str(plan.clip_checkpoint),
            "clip_checkpoint_sha256": EXPECTED_CLIP_SHA256,
            "config": str(plan.config_path),
            "config_sha256": plan.resolved_config["config_sha256"],
            "output_root": str(plan.output_root),
        },
    }


def ordinary_coop_step(
    *,
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_closure: Callable[[tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    param_index: ParamIndex,
    optimizer: torch.optim.Optimizer,
    precision: PrecisionController,
    optimizer_step: int,
) -> CoOpStepRecord:
    """Run one ordinary upstream-equivalent CoOp optimizer transition."""

    del model  # The closure owns the exact upstream CustomCLIP forward.
    precision.begin(optimizer)
    with precision.autocast_context():
        loss = loss_closure(batch)
    precision.backward(loss)
    capture = precision.capture_gradients(param_index, optimizer)
    step_result = precision.step(optimizer)
    value = float(loss.detach().item())
    gradient_norm = float(capture.state.norm().item())
    if not math.isfinite(value) or not math.isfinite(gradient_norm):
        raise ScientificRunnerError("CoOp loss or gradient norm is nonfinite")
    return CoOpStepRecord(
        method="coop",
        optimizer_step=optimizer_step,
        loss_current=value,
        batch_gradient_norm=gradient_norm,
        final_gradient_norm=gradient_norm,
        optimizer_step_result=step_result,
    )


def dispatch_training_step(
    *,
    selection: MethodSelection,
    runtime: RuntimeState,
    batch: tuple[torch.Tensor, torch.Tensor],
    loss_closure,
    optimizer_step: int,
    epoch: int,
    batch_index: int,
) -> CoOpStepRecord | SAMStepRecord | SAMPLeStepRecord:
    """Keep CoOp, SAM, and SAMPLe optimizer paths structurally distinct."""

    if selection.method == "coop":
        if runtime.engine is not None or runtime.estimator is not None:
            raise ScientificRunnerError("CoOp must not own a StepEngine/estimator")
        return ordinary_coop_step(
            model=runtime.model,
            batch=batch,
            loss_closure=loss_closure,
            param_index=runtime.param_index,
            optimizer=runtime.trainer.optim,
            precision=runtime.precision,
            optimizer_step=optimizer_step,
        )
    if runtime.engine is None:
        raise ScientificRunnerError("Sharpness-aware method lacks StepEngine")
    if selection.method == "sam":
        if runtime.estimator is not None:
            raise ScientificRunnerError("SAM must not own an estimator")
        return runtime.engine.step_sam(batch, loss_closure)
    if selection.method == "sample":
        if runtime.estimator is None:
            raise ScientificRunnerError(
                f"SAMPLe-{selection.estimator} lacks its estimator"
            )
        if runtime.estimator.mode != selection.estimator:
            raise ScientificRunnerError(
                "Runtime estimator mode differs from the scientific plan"
            )
        return runtime.engine.step_sample(
            batch,
            loss_closure,
            runtime.estimator,
            epoch=epoch,
            batch_index=batch_index,
        )
    raise ScientificRunnerError(f"Unexpected method: {selection.method}")


def advance_epoch_scheduler(trainer) -> None:
    """Advance the inherited scheduler exactly once at an epoch boundary."""

    trainer.update_lr()


def _build_runtime(plan: ScientificPlan, run_dir: Path) -> RuntimeState:
    cfg = build_scientific_cfg(
        dataset=plan.dataset,
        seed=plan.seed,
        data_root=plan.data_root,
        output_dir=run_dir / "runtime" / "base",
        config_path=plan.config_path,
        shots=plan.shots,
        epochs=plan.epochs,
    )
    set_random_seed(plan.seed)
    trainer = build_coop_trainer(cfg, plan.clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    audit_prompt_only_training(model, trainer.optim)
    index = ParamIndex.from_model(model)
    if index.names != ("prompt_learner.ctx",) or index[0].shape != (4, 512):
        raise ScientificRunnerError("Scientific CoOp ParamIndex differs from (4, 512)")
    if len(trainer.train_loader_x) != plan.steps_per_epoch:
        raise ScientificRunnerError("Real normal loader length differs from Task-2 manifest")
    if len(trainer.dm.dataset.train_x) != len(plan.source):
        raise ScientificRunnerError("Real CoOp base training source differs from Task-2 source")
    precision = PrecisionController("fp16")
    perturbation = None
    engine = None
    estimator = None
    full_loader = None
    if plan.selection.method in {"sam", "sample"}:
        perturbation = PromptPerturbation(index)
    if plan.selection.method == "sample":
        full_loader = build_full_gradient_loader(
            cfg,
            plan.source,
            micro_batch_size=plan.full_gradient_micro_batch_size,
            num_workers=0,
        )
        service = FullGradientService(
            model=model,
            param_index=index,
            loader=full_loader,
            precision_controller=PrecisionController("fp16"),
            protocol_seed=plan.seed,
            dataset=plan.dataset,
            shots=plan.shots,
            config_hash=plan.resolved_config["config_sha256"],
        )
        if plan.selection.estimator == "ema":
            if plan.selection.ema_lambda is None:
                raise ScientificRunnerError("EMA runtime lacks an EMA decay")
            estimator = EMAEstimator(
                index, ema_lambda=plan.selection.ema_lambda
            )
        elif plan.selection.estimator == "exact":
            estimator = ExactEstimator(
                index,
                full_gradient_service=service,
            )
        elif plan.selection.estimator == "periodic":
            if plan.selection.refresh_k_steps is None:
                raise ScientificRunnerError("Periodic runtime lacks refresh K")
            estimator = PeriodicEstimator(
                index,
                ema_lambda=0.15,
                refresh_k_steps=plan.selection.refresh_k_steps,
                full_gradient_service=service,
            )
        else:
            raise ScientificRunnerError(
                f"Unsupported SAMPLe estimator: {plan.selection.estimator!r}"
            )
        if plan.diagnostic_interval_steps is None:
            raise ScientificRunnerError("SAMPLe runtime lacks diagnostic cadence")
        coordinator = DiagnosticCoordinator(
            schedule=DiagnosticSchedule(plan.diagnostic_interval_steps),
            full_gradient_service=service,
            norm_eps=NORM_EPS,
        )
        engine = StepEngine(
            param_index=index,
            optimizer=trainer.optim,
            precision_controller=precision,
            rho=0.05,
            alpha=0.0015,
            norm_eps=NORM_EPS,
            perturbation=perturbation,
            diagnostic_coordinator=coordinator,
        )
    elif plan.selection.method == "sam":
        engine = StepEngine(
            param_index=index,
            optimizer=trainer.optim,
            precision_controller=precision,
            rho=0.05,
            alpha=None,
            norm_eps=NORM_EPS,
            perturbation=perturbation,
        )
    return RuntimeState(
        trainer=trainer,
        model=model,
        param_index=index,
        precision=precision,
        perturbation=perturbation,
        engine=engine,
        estimator=estimator,
        full_gradient_loader=full_loader,
    )


def _new_run(plan: ScientificPlan) -> tuple[RunArtifacts, dict[str, Any]]:
    identity = RunIdentity.now(
        dataset=plan.dataset,
        shots=plan.shots,
        method_tag=plan.selection.method_tag,
        estimator_tag=plan.selection.estimator_tag,
        seed=plan.seed,
        config_sha256=plan.resolved_config["config_sha256"],
        experiment_id=plan.experiment_id,
        smoke=False,
        allow_scientific_summary=True,
    )
    config = bind_run_identity(plan.resolved_config, identity)
    environment = capture_environment(
        project_repo=REPO_ROOT,
        coop_upstream_commit=COOP_COMMIT,
        dassl_commit=DASSL_COMMIT,
        precision_mode="coop_fp16",
        clip_backbone=EXPECTED_CLIP_KEY,
        clip_checkpoint_identifier=str(plan.clip_checkpoint),
        clip_checkpoint_sha256=EXPECTED_CLIP_SHA256,
        capture_package_freeze=True,
    )
    artifacts = RunArtifacts(plan.output_root, identity)
    artifacts.create(
        resolved_config=config,
        environment=environment,
        data_manifest_source=plan.manifest_path,
    )
    artifacts.append_log(f"{_utc_now()} scientific run created")
    return artifacts, config


def _resume_run(plan: ScientificPlan) -> tuple[RunArtifacts, dict[str, Any]]:
    checkpoint = plan.resume_from
    if checkpoint is None:
        raise ScientificRunnerError("Resume path is missing")
    run_dir = checkpoint.parent.parent
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise ScientificRunnerError("Resume checkpoint is not inside a Task-20 run")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ScientificRunnerError("Resume config is malformed")
    if config.get("config_sha256") != plan.resolved_config.get("config_sha256"):
        raise ScientificRunnerError("Resume CLI/config differs from original run hash")
    run = config.get("run", {})
    run_id = run.get("run_id")
    if not isinstance(run_id, str) or "_" not in run_id:
        raise ScientificRunnerError("Resume run_id is malformed")
    timestamp = run_id.rsplit("_", 1)[0]
    identity = RunIdentity(
        dataset=plan.dataset,
        shots=plan.shots,
        method_tag=plan.selection.method_tag,
        estimator_tag=plan.selection.estimator_tag,
        seed=plan.seed,
        utc_timestamp=timestamp,
        config_sha256=config["config_sha256"],
        experiment_id=plan.experiment_id,
        smoke=False,
        allow_scientific_summary=True,
    )
    if identity.run_id != run_id:
        raise ScientificRunnerError("Resume run identity does not reproduce run_id")
    artifacts = RunArtifacts(plan.output_root, identity)
    if artifacts.run_dir.resolve() != run_dir.resolve():
        raise ScientificRunnerError("Resume run directory differs from --output-root/cell")
    existing_summary = artifacts.run_dir / "summary.json"
    if existing_summary.is_file():
        summary = json.loads(existing_summary.read_text(encoding="utf-8"))
        if summary.get("status") == "completed":
            raise ScientificRunnerError("Completed runs cannot be resumed")
    return artifacts, config


def _checkpoint_generators(runtime: RuntimeState) -> dict[str, torch.Generator]:
    if runtime.full_gradient_loader is None:
        return {}
    return {"full_gradient": runtime.full_gradient_loader.generator}


def _accounting_from_result_state(payload: Mapping[str, Any]) -> RunAccounting:
    raw = payload.get("accounting", {})
    if not isinstance(raw, Mapping):
        raise ScientificRunnerError("Checkpoint accounting state is malformed")
    counts = raw.get("compute_counts", {})
    if not isinstance(counts, Mapping):
        raise ScientificRunnerError("Checkpoint compute counters are malformed")
    return RunAccounting(
        train_total_s=float(raw.get("train_total_s", 0.0)),
        full_gradient_total_s=float(raw.get("full_gradient_total_s", 0.0)),
        peak_cuda_allocated_bytes=int(raw.get("peak_cuda_allocated_bytes", 0)),
        peak_cuda_reserved_bytes=int(raw.get("peak_cuda_reserved_bytes", 0)),
        compute_counts={str(key): int(value) for key, value in counts.items()},
        total_wall_s=float(raw.get("total_wall_s", 0.0)),
    )


def _atomic_truncate_jsonl(path: Path, record_count: int) -> None:
    records = load_jsonl(path)
    if len(records) < record_count:
        raise ScientificRunnerError(
            f"Artifact has fewer records than its checkpoint: {path}"
        )
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    if temporary.exists():
        raise ScientificRunnerError(f"Stale resume temporary exists: {temporary}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for record in records[:record_count]:
                stream.write(
                    json.dumps(
                        record,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _result_state(
    *,
    accounting: RunAccounting,
    scheduler_steps: int,
    normal_samples_seen: int,
    metric_records: int,
    diagnostic_records: int,
    resume_events: int,
) -> dict[str, Any]:
    return {
        "accounting": accounting.as_dict(),
        "scheduler_steps": scheduler_steps,
        "normal_samples_seen": normal_samples_seen,
        "metric_records": metric_records,
        "diagnostic_records": diagnostic_records,
        "resume_events": resume_events,
        "normal_loader_resume_scope": "completed_epoch_boundary_only_workers_8",
    }


def _save_checkpoint(
    path: Path,
    *,
    plan: ScientificPlan,
    runtime: RuntimeState,
    progress: CheckpointProgress,
    result_state: Mapping[str, Any],
):
    owned_result_state = dict(result_state)
    low_compute = plan.resolved_config.get("low_compute")
    if low_compute is not None:
        if not isinstance(low_compute, Mapping):
            raise ScientificRunnerError("Low-compute checkpoint provenance is malformed")
        owned_result_state["low_compute_fork"] = copy.deepcopy(dict(low_compute))
    return save_scientific_checkpoint(
        path,
        param_index=runtime.param_index,
        optimizer=runtime.trainer.optim,
        scheduler=runtime.trainer.sched,
        precision_controller=runtime.precision,
        step_engine=runtime.engine,
        estimator=runtime.estimator,
        perturbation=runtime.perturbation,
        progress=progress,
        method=plan.selection.method,
        config_sha256=plan.resolved_config["config_sha256"],
        source_fingerprint=plan.source.fingerprint,
        result_state=owned_result_state,
        explicit_generators=_checkpoint_generators(runtime),
    )


def _common_metric(
    artifacts: RunArtifacts,
    plan: ScientificPlan,
    *,
    optimizer_step: int,
    epoch: int,
    batch_index: int,
) -> dict[str, Any]:
    return {
        "run_id": artifacts.identity.run_id,
        "experiment_id": plan.experiment_id,
        "dataset": plan.dataset,
        "shots": plan.shots,
        "seed": plan.seed,
        "method": plan.selection.method,
        "estimator_mode": (
            None if plan.selection.estimator == "none" else plan.selection.estimator
        ),
        "periodic_k_steps": plan.selection.refresh_k_steps,
        "epoch": epoch,
        "batch_index": batch_index,
        "optimizer_step": optimizer_step,
        "wall_time_utc": _utc_now(),
    }


def _append_train_metric(
    artifacts: RunArtifacts,
    plan: ScientificPlan,
    record: CoOpStepRecord | SAMStepRecord | SAMPLeStepRecord,
    *,
    epoch: int,
    batch_index: int,
    learning_rate: float,
    elapsed_s: float,
) -> None:
    metric: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "event_type": "train_step",
        **_common_metric(
            artifacts,
            plan,
            optimizer_step=record.optimizer_step,
            epoch=epoch,
            batch_index=batch_index,
        ),
        "loss/current": record.loss_current,
        "loss/displaced": getattr(record, "loss_displaced", None),
        "loss/sample_objective": getattr(record, "loss_sample_objective", None),
        "optim/learning_rate": learning_rate,
        "optim/nonfinite_event": False,
        "grad/batch_norm": record.batch_gradient_norm,
        "grad/perturbed_norm": getattr(record, "perturbed_gradient_norm", None),
        "grad/update_norm": record.final_gradient_norm,
        "grad/sam_perturb_norm": getattr(record, "sam_perturbation_norm", None),
        "timing/train_step_s": elapsed_s,
    }
    if isinstance(record, SAMPLeStepRecord):
        if record.estimator_result.mode == "ema":
            active_source = "ema_batch_update"
        elif record.estimator_result.mode == "exact":
            active_source = "optimization_exact"
        elif record.estimator_result.refreshed:
            active_source = "periodic_exact_refresh"
        else:
            active_source = "periodic_ema_update"
        metric.update(
            {
                "grad/global_estimate_norm": record.global_direction_norm,
                "grad/batch_component_norm": record.batch_component_norm,
                "grad/batch_correction_norm": record.batch_correction_norm,
                "grad/total_displacement_norm": record.total_displacement_norm,
                "grad/xi": record.projection.xi,
                "grad/sigma": record.projection.sigma,
                "grad/projection_coefficient": record.projection.projection_coefficient,
                "estimator/refreshed": record.estimator_result.refreshed,
                "estimator/age_steps": record.estimator_result.age_steps,
                "estimator/last_refresh_step": (
                    record.estimator_result.last_refresh_step
                ),
                "estimator/mode": record.estimator_result.mode,
                "estimator/active_source": active_source,
                "estimator/periodic_k_steps": plan.selection.refresh_k_steps,
                "estimator/exact_query_count": record.estimator_result.exact_query_count,
            }
        )
    artifacts.append_metric(metric)


def _account_exact_sweep(
    accounting: RunAccounting,
    metadata: FullGradientSweepMetadata,
    *,
    query_kind: str,
) -> None:
    if query_kind not in {"optimization", "diagnostic_only"}:
        raise ScientificRunnerError(f"Invalid exact-query kind: {query_kind!r}")
    accounting.increment("full_gradient_sweeps")
    accounting.increment("full_gradient_forward_microbatches", metadata.forward_calls)
    accounting.increment(
        "full_gradient_backward_microbatches", metadata.autograd_grad_calls
    )
    accounting.increment("full_gradient_samples", metadata.sample_count)
    accounting.increment("exact_sweeps")
    accounting.increment("exact_sweep_forward_batches", metadata.forward_calls)
    accounting.increment(
        "exact_sweep_backward_batches", metadata.autograd_grad_calls
    )
    accounting.increment("exact_sweep_samples", metadata.sample_count)
    accounting.increment(f"{query_kind}_exact_queries")
    accounting.full_gradient_total_s += metadata.elapsed_s


def _update_accounting(
    accounting: RunAccounting,
    record: CoOpStepRecord | SAMStepRecord | SAMPLeStepRecord,
    batch_size: int,
) -> None:
    accounting.increment("optimizer_steps")
    accounting.increment("current_forward_batches")
    accounting.increment("current_backward_batches")
    accounting.increment("current_samples", batch_size)
    if isinstance(record, (SAMStepRecord, SAMPLeStepRecord)):
        accounting.increment("displaced_forward_batches")
        accounting.increment("displaced_backward_batches")
        accounting.increment("displaced_samples", batch_size)
    if isinstance(record, SAMPLeStepRecord):
        optimization_metadata = record.estimator_result.full_gradient_metadata
        if optimization_metadata is not None:
            _account_exact_sweep(
                accounting,
                optimization_metadata,
                query_kind="optimization",
            )
        if record.diagnostic_event is not None:
            event = record.diagnostic_event
            accounting.increment("exact_reference_points")
            if event.reference.exact_service_query_issued:
                _account_exact_sweep(
                    accounting,
                    event.reference.full_gradient_metadata,
                    query_kind="diagnostic_only",
                )
            else:
                accounting.increment("reused_exact_queries")


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _prompt_hash(param_index: ParamIndex) -> str:
    tensor = param_index[0].parameter.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _evaluate_split(trainer) -> tuple[float, int, int, float]:
    _sync_cuda()
    started = time.perf_counter()
    accuracy = float(trainer.test())
    _sync_cuda()
    elapsed = time.perf_counter() - started
    return (
        accuracy,
        len(trainer.test_loader.dataset),
        len(trainer.dm.dataset.classnames),
        elapsed,
    )


def _evaluate_base_new(
    plan: ScientificPlan,
    runtime: RuntimeState,
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    learned = runtime.param_index[0].parameter.detach().cpu().clone()
    base_accuracy, base_samples, base_classes, base_s = _evaluate_split(runtime.trainer)
    if not torch.equal(runtime.param_index[0].parameter.detach().cpu(), learned):
        raise ScientificRunnerError("Base evaluation changed the learned prompt")
    new_cfg = build_scientific_cfg(
        dataset=plan.dataset,
        seed=plan.seed,
        data_root=plan.data_root,
        output_dir=run_dir / "runtime" / "new",
        config_path=plan.config_path,
        class_subsample="new",
        shots=plan.shots,
        epochs=plan.epochs,
    )
    set_random_seed(plan.seed)
    new_trainer = build_coop_trainer(new_cfg, plan.clip_cache)
    new_model = unwrap_model(new_trainer.model)
    new_index = ParamIndex.from_model(new_model)
    if new_index[0].shape != runtime.param_index[0].shape:
        raise ScientificRunnerError("New-class unified prompt shape differs")
    with torch.no_grad():
        new_index[0].parameter.copy_(
            learned.to(
                device=new_index[0].parameter.device,
                dtype=new_index[0].parameter.dtype,
            )
        )
    if not torch.equal(new_index[0].parameter.detach().cpu(), learned):
        raise ScientificRunnerError("New-class model did not receive the learned prompt")
    new_accuracy, new_samples, new_classes, new_s = _evaluate_split(new_trainer)
    if not torch.equal(new_index[0].parameter.detach().cpu(), learned):
        raise ScientificRunnerError("New evaluation changed the learned prompt")
    del new_trainer, new_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        {
            "accuracy_pct": base_accuracy,
            "sample_count": base_samples,
            "class_count": base_classes,
            "elapsed_s": base_s,
        },
        {
            "accuracy_pct": new_accuracy,
            "sample_count": new_samples,
            "class_count": new_classes,
            "elapsed_s": new_s,
        },
    )


def _diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    empty = {
        "num_exact_reference_points": 0,
        "global_estimate_exact_cosine_mean": None,
        "global_estimate_exact_cosine_median": None,
        "global_estimate_exact_relative_l2_mean": None,
        "global_estimate_exact_log_norm_ratio_mean": None,
        "batch_component_exact_abs_cosine_mean": None,
        "batch_component_estimate_exact_cosine_mean": None,
        "batch_component_estimate_exact_relative_l2_mean": None,
        "batch_component_estimate_exact_norm_ratio_mean": None,
        "perturbed_gradient_exact_abs_cosine_mean": None,
        "global_estimate_exact_norm_ratio_mean": None,
        "batch_component_estimator_abs_cosine_mean": None,
        "reference_batch_component_exact_abs_cosine_mean": None,
        "perturbed_gradient_estimator_abs_cosine_mean": None,
        "perturbed_gradient_batch_component_abs_cosine_mean": None,
        "perturbed_gradient_batch_abs_cosine_mean": None,
        "taylor_exploitation_mean": None,
        "taylor_exploration_mean": None,
        "taylor_joint_mean": None,
    }
    if not records:
        return empty
    keys = (
        "grad/global_estimate_exact_cosine",
        "grad/global_estimate_exact_relative_l2",
        "grad/global_estimate_exact_log_norm_ratio",
        "grad/batch_component_exact_cosine",
        "grad/batch_component_estimate_exact_cosine",
        "grad/batch_component_estimate_exact_relative_l2",
        "grad/batch_component_estimate_exact_norm_ratio",
        "grad/perturbed_gradient_exact_cosine",
        "grad/global_estimate_exact_norm_ratio",
        "grad/batch_component_estimator_cosine",
        "grad/reference_batch_component_exact_cosine",
        "grad/perturbed_gradient_estimator_cosine",
        "grad/perturbed_gradient_batch_component_cosine",
        "grad/perturbed_gradient_batch_cosine",
        "taylor/exploitation_term",
        "taylor/exploration_term",
        "taylor/joint_alignment_term",
    )
    values = {
        key: [
            item["metrics"][key]
            for item in records
            if item.get("metrics", {}).get(key) is not None
        ]
        for key in keys
    }
    cosine = sorted(values[keys[0]])
    median = None
    if cosine:
        middle = len(cosine) // 2
        median = (
            cosine[middle]
            if len(cosine) % 2
            else (cosine[middle - 1] + cosine[middle]) / 2
        )
    mean = lambda sequence: sum(sequence) / len(sequence) if sequence else None
    return {
        "num_exact_reference_points": len(records),
        "global_estimate_exact_cosine_mean": mean(values[keys[0]]),
        "global_estimate_exact_cosine_median": median,
        "global_estimate_exact_relative_l2_mean": mean(values[keys[1]]),
        "global_estimate_exact_log_norm_ratio_mean": mean(values[keys[2]]),
        "batch_component_exact_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[3]]]
        ),
        "batch_component_estimate_exact_cosine_mean": mean(values[keys[4]]),
        "batch_component_estimate_exact_relative_l2_mean": mean(values[keys[5]]),
        "batch_component_estimate_exact_norm_ratio_mean": mean(values[keys[6]]),
        "perturbed_gradient_exact_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[7]]]
        ),
        "global_estimate_exact_norm_ratio_mean": mean(values[keys[8]]),
        "batch_component_estimator_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[9]]]
        ),
        "reference_batch_component_exact_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[10]]]
        ),
        "perturbed_gradient_estimator_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[11]]]
        ),
        "perturbed_gradient_batch_component_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[12]]]
        ),
        "perturbed_gradient_batch_abs_cosine_mean": mean(
            [abs(value) for value in values[keys[13]]]
        ),
        "taylor_exploitation_mean": mean(values[keys[14]]),
        "taylor_exploration_mean": mean(values[keys[15]]),
        "taylor_joint_mean": mean(values[keys[16]]),
    }


def _failed_summary(
    artifacts: RunArtifacts,
    accounting: RunAccounting,
    *,
    error: BaseException,
    optimizer_steps: int,
    scheduler_steps: int,
) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_identity": artifacts.identity.as_dict(),
        "status": "failed",
        "smoke": False,
        "allow_scientific_summary": True,
        "failure": {"type": type(error).__name__, "message": str(error)},
        "evaluation": {
            "base_accuracy_pct": None,
            "new_accuracy_pct": None,
            "hm_pct": None,
        },
        "efficiency": accounting.as_dict(),
        "estimator_diagnostics": _diagnostic_summary([]),
        "invariants": {
            "optimizer_steps": optimizer_steps,
            "scheduler_steps": scheduler_steps,
        },
    }


def run_scientific(plan: ScientificPlan) -> Path:
    """Execute one complete, unbounded scientific cell."""

    if not torch.cuda.is_available():
        raise ScientificRunnerError("Scientific execution requires a CUDA GPU")
    run_started = time.perf_counter()
    artifacts, _ = _resume_run(plan) if plan.resume_from else _new_run(plan)
    try:
        runtime = _build_runtime(plan, artifacts.run_dir)
    except BaseException as error:
        artifacts.append_log(
            f"{_utc_now()} run failed during startup: "
            f"{type(error).__name__}: {error}"
        )
        artifacts.write_summary(
            _failed_summary(
                artifacts,
                RunAccounting(),
                error=error,
                optimizer_steps=0,
                scheduler_steps=0,
            )
        )
        raise
    initial_prompt_hash = _prompt_hash(runtime.param_index)
    frozen_before = hash_frozen_parameters(runtime.model)
    accounting = RunAccounting()
    scheduler_steps = 0
    global_step = 0
    normal_samples_seen = 0
    metric_records = 0
    diagnostic_records = 0
    resume_events = 0
    start_epoch = 0
    if plan.resume_from is not None:
        loaded = load_scientific_checkpoint(
            plan.resume_from,
            param_index=runtime.param_index,
            optimizer=runtime.trainer.optim,
            scheduler=runtime.trainer.sched,
            precision_controller=runtime.precision,
            step_engine=runtime.engine,
            estimator=runtime.estimator,
            perturbation=runtime.perturbation,
            expected_method=plan.selection.method,
            expected_config_sha256=plan.resolved_config["config_sha256"],
            expected_source_fingerprint=plan.source.fingerprint,
            explicit_generators=_checkpoint_generators(runtime),
        )
        if loaded.progress.next_batch_index_zero_based != 0:
            raise ScientificRunnerError(
                "The 8-worker scientific runner resumes only at epoch boundaries"
            )
        start_epoch = loaded.progress.epoch_zero_based
        global_step = loaded.progress.next_optimizer_step
        normal_samples_seen = loaded.progress.normal_samples_seen
        if global_step != start_epoch * plan.steps_per_epoch:
            raise ScientificRunnerError("Resume epoch and optimizer-step clocks differ")
        accounting = _accounting_from_result_state(loaded.result_state)
        scheduler_steps = int(loaded.result_state.get("scheduler_steps", -1))
        metric_records = int(loaded.result_state.get("metric_records", -1))
        diagnostic_records = int(loaded.result_state.get("diagnostic_records", -1))
        resume_events = int(loaded.result_state.get("resume_events", 0)) + 1
        if min(scheduler_steps, metric_records, diagnostic_records) < 0:
            raise ScientificRunnerError("Checkpoint result counters are malformed")
        _atomic_truncate_jsonl(artifacts.metrics_path, metric_records)
        _atomic_truncate_jsonl(artifacts.diagnostics_path, diagnostic_records)
        artifacts.append_log(
            f"{_utc_now()} resumed from {plan.resume_from} at epoch {start_epoch}"
        )
    else:
        artifacts.append_log(f"{_utc_now()} training started")

    if start_epoch > plan.epochs:
        raise ScientificRunnerError("Resume epoch exceeds scientific horizon")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    accumulated_train_s = accounting.train_total_s
    accumulated_wall_s = accounting.total_wall_s
    training_started = time.perf_counter()
    final_checkpoint = None
    try:
        for epoch in range(start_epoch, plan.epochs):
            runtime.trainer.set_model_mode("train")
            runtime.trainer.epoch = epoch
            runtime.trainer.num_batches = len(runtime.trainer.train_loader_x)
            epoch_started = time.perf_counter()
            for batch_index, raw_batch in enumerate(runtime.trainer.train_loader_x):
                runtime.trainer.batch_idx = batch_index
                image, label = runtime.trainer.parse_batch_train(raw_batch)
                batch = (image, label)
                learning_rate = float(runtime.trainer.get_current_lr())
                _sync_cuda()
                step_started = time.perf_counter()
                record = dispatch_training_step(
                    selection=plan.selection,
                    runtime=runtime,
                    batch=batch,
                    loss_closure=lambda item: F.cross_entropy(
                        runtime.model(item[0]), item[1]
                    ),
                    optimizer_step=global_step,
                    epoch=epoch,
                    batch_index=batch_index,
                )
                _sync_cuda()
                step_elapsed = time.perf_counter() - step_started
                if record.optimizer_step != global_step:
                    raise ScientificRunnerError("Optimizer step clock diverged")
                _append_train_metric(
                    artifacts,
                    plan,
                    record,
                    epoch=epoch,
                    batch_index=batch_index,
                    learning_rate=learning_rate,
                    elapsed_s=step_elapsed,
                )
                metric_records += 1
                batch_size = int(label.numel())
                normal_samples_seen += batch_size
                _update_accounting(accounting, record, batch_size)
                if isinstance(record, SAMPLeStepRecord) and record.diagnostic_event is not None:
                    diagnostic = {
                        **record.diagnostic_event.as_dict(),
                        **_common_metric(
                            artifacts,
                            plan,
                            optimizer_step=record.optimizer_step,
                            epoch=epoch,
                            batch_index=batch_index,
                        ),
                    }
                    low_compute = plan.resolved_config.get("low_compute")
                    if isinstance(low_compute, Mapping) and plan.experiment_id == "LC2":
                        diagnostic.update(
                            {
                                "intervention_lambda": low_compute["new_lambda"],
                                "source_lambda": low_compute["source_lambda"],
                                "branch_age_epochs": epoch - int(low_compute["source_epoch"]),
                                "branch_age_steps": global_step - int(low_compute["source_optimizer_step"]),
                                "actual_ema_exact_cosine": diagnostic["metrics"].get(
                                    "grad/global_estimate_exact_cosine"
                                ),
                                "actual_ema_exact_relative_l2": diagnostic["metrics"].get(
                                    "grad/global_estimate_exact_relative_l2"
                                ),
                                "gB_exact_fidelity": diagnostic["metrics"].get(
                                    "grad/batch_component_exact_cosine"
                                ),
                                "gB_exact_fidelity_semantics": diagnostic["metrics"].get(
                                    "grad/batch_component_exact_cosine_semantics"
                                ),
                                "gB_est_exact_cosine": diagnostic["metrics"].get(
                                    "grad/batch_component_estimate_exact_cosine"
                                ),
                                "gB_est_exact_relative_l2": diagnostic["metrics"].get(
                                    "grad/batch_component_estimate_exact_relative_l2"
                                ),
                                "gB_est_exact_norm_ratio": diagnostic["metrics"].get(
                                    "grad/batch_component_estimate_exact_norm_ratio"
                                ),
                            }
                        )
                    artifacts.append_diagnostic(diagnostic)
                    diagnostic_records += 1
                global_step += 1
            advance_epoch_scheduler(runtime.trainer)
            scheduler_steps += 1
            elapsed_epoch = time.perf_counter() - epoch_started
            artifacts.append_log(
                f"{_utc_now()} epoch {epoch + 1}/{plan.epochs} completed "
                f"steps={global_step} elapsed_s={elapsed_epoch:.6f}"
            )
            completed_epoch = epoch + 1
            if completed_epoch % plan.recovery_interval_epochs == 0:
                accounting.train_total_s = (
                    accumulated_train_s + time.perf_counter() - training_started
                )
                accounting.total_wall_s = (
                    accumulated_wall_s + time.perf_counter() - run_started
                )
                state = _result_state(
                    accounting=accounting,
                    scheduler_steps=scheduler_steps,
                    normal_samples_seen=normal_samples_seen,
                    metric_records=metric_records,
                    diagnostic_records=diagnostic_records,
                    resume_events=resume_events,
                )
                recovery = (
                    artifacts.run_dir
                    / "checkpoints"
                    / f"recovery_step_{global_step:06d}.pt"
                )
                _save_checkpoint(
                    recovery,
                    plan=plan,
                    runtime=runtime,
                    progress=CheckpointProgress(
                        global_step,
                        completed_epoch,
                        0,
                        normal_samples_seen,
                    ),
                    result_state=state,
                )
        accounting.train_total_s = (
            accumulated_train_s + time.perf_counter() - training_started
        )
        accounting.total_wall_s = (
            accumulated_wall_s + time.perf_counter() - run_started
        )
        accounting.peak_cuda_allocated_bytes = max(
            accounting.peak_cuda_allocated_bytes,
            int(torch.cuda.max_memory_allocated()),
        )
        accounting.peak_cuda_reserved_bytes = max(
            accounting.peak_cuda_reserved_bytes,
            int(torch.cuda.max_memory_reserved()),
        )
        if global_step != plan.total_optimizer_steps:
            raise ScientificRunnerError("Final optimizer-step count differs from plan")
        if scheduler_steps != plan.epochs:
            raise ScientificRunnerError("Scheduler did not advance exactly once per epoch")
        state = _result_state(
            accounting=accounting,
            scheduler_steps=scheduler_steps,
            normal_samples_seen=normal_samples_seen,
            metric_records=metric_records,
            diagnostic_records=diagnostic_records,
            resume_events=resume_events,
        )
        final_checkpoint = _save_checkpoint(
            artifacts.run_dir / "checkpoints" / "final.pt",
            plan=plan,
            runtime=runtime,
            progress=CheckpointProgress(
                global_step,
                plan.epochs,
                0,
                normal_samples_seen,
            ),
            result_state=state,
        )
        prompt_changed = _prompt_hash(runtime.param_index) != initial_prompt_hash
        frozen_unchanged = hash_frozen_parameters(runtime.model) == frozen_before
        if not prompt_changed or not frozen_unchanged:
            raise ScientificRunnerError("Prompt/frozen-CLIP postcondition failed")
        base, new = _evaluate_base_new(plan, runtime, artifacts.run_dir)
        denominator = base["accuracy_pct"] + new["accuracy_pct"]
        harmonic = (
            0.0
            if denominator == 0
            else 2.0 * base["accuracy_pct"] * new["accuracy_pct"] / denominator
        )
        for event_type, value in (("eval_base", base), ("eval_new", new)):
            artifacts.append_metric(
                {
                    "schema_version": METRICS_SCHEMA_VERSION,
                    "event_type": event_type,
                    **_common_metric(
                        artifacts,
                        plan,
                        optimizer_step=global_step,
                        epoch=plan.epochs,
                        batch_index=-1,
                    ),
                    "eval/accuracy_pct": value["accuracy_pct"],
                    "eval/num_samples": value["sample_count"],
                    "eval/class_count": value["class_count"],
                    "timing/eval_s": value["elapsed_s"],
                }
            )
        diagnostics = load_jsonl(artifacts.diagnostics_path)
        diagnostic_overhead = sum(
            float(item["full_gradient"]["elapsed_s"])
            for item in diagnostics
            if item.get("exact_service_query_issued")
        )
        total_wall_time = accumulated_wall_s + time.perf_counter() - run_started
        accounting.total_wall_s = total_wall_time
        summary = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "run_identity": artifacts.identity.as_dict(),
            "status": "completed",
            "smoke": False,
            "allow_scientific_summary": True,
            "evaluation": {
                "base_accuracy_pct": base["accuracy_pct"],
                "new_accuracy_pct": new["accuracy_pct"],
                "hm_pct": harmonic,
                "base_num_samples": base["sample_count"],
                "new_num_samples": new["sample_count"],
                "base_class_count": base["class_count"],
                "new_class_count": new["class_count"],
                "same_unified_context": True,
                "new_retrained": False,
                "final_model_policy": "last_step",
            },
            "efficiency": {
                **accounting.as_dict(),
                "exact_gradient_total_s": accounting.full_gradient_total_s,
                "eval_base_s": base["elapsed_s"],
                "eval_new_s": new["elapsed_s"],
                "evaluation_total_s": base["elapsed_s"] + new["elapsed_s"],
                "diagnostic_overhead_s": diagnostic_overhead,
                "total_wall_time_s": total_wall_time,
            },
            "estimator_diagnostics": _diagnostic_summary(diagnostics),
            "resume": {
                "resume_events": resume_events,
                "last_resume_from": str(plan.resume_from) if plan.resume_from else None,
                "boundary_policy": "completed_epoch_only_workers_8",
            },
            "invariants": {
                "prompt_changed": prompt_changed,
                "frozen_clip_unchanged": frozen_unchanged,
                "optimizer_steps": global_step,
                "scheduler_steps": scheduler_steps,
                "expected_scheduler_steps": plan.epochs,
                "one_optimizer_step_per_batch": True,
                "unperturbed_checkpoint_boundary": True,
            },
            "artifacts": {
                "config": "config.yaml",
                "environment": "environment.json",
                "data_manifest": "data_manifest.json",
                "metrics": "metrics.jsonl",
                "gradient_diagnostics": "gradient_diagnostics.jsonl",
                "checkpoint": "checkpoints/final.pt",
                "checkpoint_metadata": final_checkpoint.as_dict(),
                "log": "logs/run.log",
            },
        }
        low_compute = plan.resolved_config.get("low_compute")
        if isinstance(low_compute, Mapping) and plan.experiment_id == "LC2":
            summary["low_compute_branch"] = {
                **copy.deepcopy(dict(low_compute)),
                "branch_optimizer_steps_executed": global_step
                - int(low_compute["source_optimizer_step"]),
                "branch_scheduler_steps_executed": scheduler_steps
                - int(low_compute["source_epoch"]),
                "interpretation": "one_seed_causal_pilot",
            }
        artifacts.write_summary(summary)
        artifacts.append_log(f"{_utc_now()} scientific run completed")
        return artifacts.run_dir
    except BaseException as error:
        accounting.train_total_s = (
            accumulated_train_s + time.perf_counter() - training_started
        )
        accounting.total_wall_s = (
            accumulated_wall_s + time.perf_counter() - run_started
        )
        artifacts.append_log(
            f"{_utc_now()} run failed: {type(error).__name__}: {error}"
        )
        artifacts.write_summary(
            _failed_summary(
                artifacts,
                accounting,
                error=error,
                optimizer_steps=global_step,
                scheduler_steps=scheduler_steps,
            )
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one scientific DTD/EuroSAT paper-reproduction cell"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--shots", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--estimator", required=True, choices=("none", "ema"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-id", default="R2")
    parser.add_argument("--recovery-interval-epochs", type=int, default=10)
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(dry_run_report(plan), indent=2, sort_keys=True))
        return 0
    run_dir = run_scientific(plan)
    print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
    return 0
