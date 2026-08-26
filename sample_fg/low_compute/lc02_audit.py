"""Read-only correction audit for the completed LC02 diagnostic cadence."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
import yaml

from sample_fg.checkpoint import load_scientific_checkpoint
from sample_fg.coop_anchor import hash_frozen_parameters
from sample_fg.gradient_state import GradientState
from sample_fg.paper_runner import (
    MethodSelection,
    ScientificPlan,
    _build_runtime,
    _checkpoint_generators,
)
from sample_fg.projection import project_batch_gradient
from sample_fg.results import atomic_write_json, load_jsonl

from .budget import ComputeBudget, TransitionGuard
from .checkpoint_probe import sha256_file
from .runner import build_source_scientific_plan


LC02_AUDIT_SCHEMA = "sample_fg.low_compute_lc02_diagnostic_audit.v1"
SOURCE_LAMBDA = 0.15
TARGET_LAMBDA = 11.0 / 13.0
SOURCE_STEP = 2160
TARGET_STEP = 2400
DIAGNOSTIC_INTERVAL = 12
EXPECTED_POINTS = 20
LEGACY_SEMANTICS = (
    "cosine(g_B(active_global_estimate), exact_full_gradient); "
    "construction-orthogonality metric, not projected-component fidelity"
)


class LC02DiagnosticAuditError(RuntimeError):
    """Raised when the immutable LC02 audit cannot be reproduced safely."""


def _mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise LC02DiagnosticAuditError(f"Cannot read audit input: {path}") from error
    if not isinstance(value, dict):
        raise LC02DiagnosticAuditError(f"Audit input is not a mapping: {path}")
    return value


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip().lower()
    if result.returncode != 0 or len(value) != 40:
        raise LC02DiagnosticAuditError("Cannot resolve repository SHA")
    return value


def hash_directory(root: Path) -> dict[str, dict[str, object]]:
    """Return a stable per-file hash inventory without mutating ``root``."""

    resolved = Path(root).resolve(strict=True)
    output: dict[str, dict[str, object]] = {}
    for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
        output[path.relative_to(resolved).as_posix()] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def hash_inventory_fingerprint(inventory: Mapping[str, Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for relative, metadata in sorted(inventory.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(str(metadata["size_bytes"]).encode("ascii"))
        digest.update(str(metadata["sha256"]).encode("ascii"))
    return digest.hexdigest()


def _checkpoint_schedule(run_dir: Path) -> tuple[tuple[int, Path], ...]:
    rows: list[tuple[int, Path]] = [(SOURCE_STEP, run_dir / "checkpoints" / "fork.pt")]
    rows.extend(
        (step, run_dir / "checkpoints" / f"recovery_step_{step:06d}.pt")
        for step in range(SOURCE_STEP + DIAGNOSTIC_INTERVAL, TARGET_STEP, DIAGNOSTIC_INTERVAL)
    )
    if len(rows) != EXPECTED_POINTS:
        raise LC02DiagnosticAuditError("LC02 audit cadence is not exactly 20 points")
    for _, path in rows:
        if not path.is_file():
            raise LC02DiagnosticAuditError(f"Required LC02 checkpoint is missing: {path}")
    return tuple(rows)


def _validate_completed_run(run_dir: Path, source_run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = _mapping(run_dir / "summary.json")
    config = _mapping(run_dir / "config.yaml")
    if summary.get("status") != "completed" or summary.get("smoke") is not False:
        raise LC02DiagnosticAuditError("LC02 audit requires a completed scientific run")
    if summary.get("run_identity", {}).get("experiment_id") != "LC2":
        raise LC02DiagnosticAuditError("Audit target is not LC02")
    if summary.get("run_identity", {}).get("config_sha256") != config.get("config_sha256"):
        raise LC02DiagnosticAuditError("LC02 summary/config hashes differ")
    low_compute = config.get("low_compute", {})
    required = {
        "source_run_dir": str(source_run),
        "source_optimizer_step": SOURCE_STEP,
        "source_epoch": 180,
        "target_epoch": 200,
        "source_lambda": SOURCE_LAMBDA,
        "new_lambda": TARGET_LAMBDA,
        "estimator_state_transplant": "state_preserving_ema_decay_switch_v1",
    }
    if any(low_compute.get(key) != value for key, value in required.items()):
        raise LC02DiagnosticAuditError("LC02 fork provenance differs from the registered intervention")
    final = (run_dir / summary["artifacts"]["checkpoint"]).resolve(strict=True)
    if sha256_file(final) != summary["artifacts"]["checkpoint_metadata"]["sha256"]:
        raise LC02DiagnosticAuditError("LC02 final checkpoint hash differs from summary")
    source_checkpoint = Path(low_compute["source_checkpoint"]).resolve(strict=True)
    if sha256_file(source_checkpoint) != low_compute["source_checkpoint_sha256"]:
        raise LC02DiagnosticAuditError("LC02 source checkpoint hash differs from provenance")
    diagnostics = load_jsonl(run_dir / "gradient_diagnostics.jsonl")
    steps = [int(row.get("optimizer_step", -1)) for row in diagnostics]
    expected_steps = list(range(SOURCE_STEP, TARGET_STEP, DIAGNOSTIC_INTERVAL))
    if steps != expected_steps:
        raise LC02DiagnosticAuditError("Saved LC02 diagnostic cadence differs from 20 epoch starts")
    for row in diagnostics:
        legacy = row.get("metrics", {}).get("grad/batch_component_exact_cosine")
        if row.get("gB_exact_fidelity") != legacy:
            raise LC02DiagnosticAuditError("Saved gB_exact_fidelity is not its serialized alias")
    return summary, config


def build_completed_branch_plan(
    *,
    source_plan: ScientificPlan,
    completed_config: Mapping[str, Any],
    runtime_output_root: Path,
) -> ScientificPlan:
    """Reconstruct the completed branch identity without changing its config hash."""

    selection = MethodSelection(
        method="sample",
        estimator="ema",
        method_tag="sample_coverage",
        estimator_tag="ema",
        rho=0.05,
        alpha=0.0015,
        ema_lambda=TARGET_LAMBDA,
        refresh_k_steps=None,
    )
    if completed_config.get("method", {}).get("rho") != 0.05:
        raise LC02DiagnosticAuditError("Completed LC02 rho differs")
    if completed_config.get("method", {}).get("alpha") != 0.0015:
        raise LC02DiagnosticAuditError("Completed LC02 alpha differs")
    if completed_config.get("method", {}).get("ema_lambda") != TARGET_LAMBDA:
        raise LC02DiagnosticAuditError("Completed LC02 lambda differs")
    return replace(
        source_plan,
        experiment_id="LC2",
        selection=selection,
        output_root=Path(runtime_output_root).resolve(),
        recovery_interval_epochs=1,
        diagnostic_interval_steps=DIAGNOSTIC_INTERVAL,
        resolved_config=copy.deepcopy(dict(completed_config)),
        resume_from=None,
    )


def build_lc02_audit_plan(
    *,
    lc02_run: Path,
    source_run: Path,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    paper_config: Path,
    output_root: Path,
) -> dict[str, Any]:
    lc02_run = Path(lc02_run).resolve(strict=True)
    source_run = Path(source_run).resolve(strict=True)
    summary, config = _validate_completed_run(lc02_run, source_run)
    checkpoints = _checkpoint_schedule(lc02_run)
    source_plan = build_source_scientific_plan(
        source_run=source_run,
        data_root=Path(data_root),
        manifest_root=Path(manifest_root),
        clip_cache=Path(clip_cache),
        paper_config=Path(paper_config),
        runtime_output_root=Path(output_root) / "_runtime_plan",
    )
    build_completed_branch_plan(
        source_plan=source_plan,
        completed_config=config,
        runtime_output_root=Path(output_root) / "_runtime_plan",
    )
    budget = ComputeBudget(
        optimizer_steps=0,
        scheduler_steps=0,
        normal_forward_batches=EXPECTED_POINTS,
        normal_backward_batches=EXPECTED_POINTS,
        exact_forward_batches=EXPECTED_POINTS * DIAGNOSTIC_INTERVAL,
        exact_backward_batches=EXPECTED_POINTS * DIAGNOSTIC_INTERVAL,
        exact_sweeps=EXPECTED_POINTS,
    )
    budget.require_read_only()
    return {
        "schema_version": LC02_AUDIT_SCHEMA,
        "status": "DRY_RUN_VALIDATED",
        "training_started": False,
        "source_artifacts_read_only": True,
        "lc02_run": str(lc02_run),
        "lc02_run_id": summary["run_identity"]["run_id"],
        "source_run": str(source_run),
        "diagnostic_steps": [step for step, _ in checkpoints],
        "checkpoint_sha256": {str(step): sha256_file(path) for step, path in checkpoints},
        "budget": budget.as_dict(),
    }


def _state_comparison(left: GradientState, right: GradientState) -> dict[str, object]:
    left.assert_compatible(right)
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    degenerate = left_norm <= 1e-12 or right_norm <= 1e-12
    cosine = None
    norm_ratio = None
    relative_l2 = None
    if not degenerate:
        cosine = float(left.dot(right).item()) / (left_norm * right_norm)
        cosine = max(-1.0, min(1.0, cosine))
        norm_ratio = left_norm / right_norm
        relative_l2 = float(left.subtract(right).norm().item()) / right_norm
    return {
        "cosine": cosine,
        "relative_l2": relative_l2,
        "norm_ratio": norm_ratio,
        "left_norm": left_norm,
        "right_norm": right_norm,
        "degenerate": degenerate,
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(dict(row), allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_lc02_diagnostic_audit(
    *,
    lc02_run: Path,
    source_run: Path,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    paper_config: Path,
    output_root: Path,
) -> Path:
    """Recompute the 20 corrected metrics without any lifecycle transition."""

    lc02_run = Path(lc02_run).resolve(strict=True)
    source_run = Path(source_run).resolve(strict=True)
    summary, config = _validate_completed_run(lc02_run, source_run)
    checkpoints = _checkpoint_schedule(lc02_run)
    before_lc02 = hash_directory(lc02_run)
    before_source = hash_directory(source_run)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root).resolve() / "lc02_audit" / run_id
    run_dir.mkdir(parents=True)
    source_plan = build_source_scientific_plan(
        source_run=source_run,
        data_root=Path(data_root),
        manifest_root=Path(manifest_root),
        clip_cache=Path(clip_cache),
        paper_config=Path(paper_config),
        runtime_output_root=run_dir / "_runtime_source",
    )
    plan = build_completed_branch_plan(
        source_plan=source_plan,
        completed_config=config,
        runtime_output_root=run_dir / "_runtime",
    )
    old_rows = {
        int(row["optimizer_step"]): row
        for row in load_jsonl(lc02_run / "gradient_diagnostics.jsonl")
    }
    corrected: list[dict[str, object]] = []
    parity_max = {"batch_norm": 0.0, "global_norm": 0.0, "exact_norm": 0.0, "legacy_cosine": 0.0}
    for step, checkpoint in checkpoints:
        # Scientific checkpoint loading intentionally accepts only a fresh,
        # idle runtime.  Construct one runtime per cadence point so restoring
        # a later checkpoint cannot inherit any model, estimator, or loader
        # state from the preceding read-only reconstruction.
        runtime = _build_runtime(plan, run_dir / "_runtime" / f"step_{step:06d}")
        if runtime.engine is None or runtime.estimator is None:
            raise LC02DiagnosticAuditError("LC02 audit runtime lacks SAMPLe state")
        frozen_before = hash_frozen_parameters(runtime.model)
        loaded = load_scientific_checkpoint(
            checkpoint,
            param_index=runtime.param_index,
            optimizer=runtime.trainer.optim,
            scheduler=runtime.trainer.sched,
            precision_controller=runtime.precision,
            step_engine=runtime.engine,
            estimator=runtime.estimator,
            perturbation=runtime.perturbation,
            expected_method="sample",
            expected_config_sha256=config["config_sha256"],
            expected_source_fingerprint=source_plan.source.fingerprint,
            explicit_generators=_checkpoint_generators(runtime),
        )
        if loaded.progress.next_optimizer_step != step or loaded.progress.next_batch_index_zero_based != 0:
            raise LC02DiagnosticAuditError("Audit checkpoint is not the expected epoch boundary")
        prompt_before = tuple(entry.parameter.detach().clone() for entry in runtime.param_index)
        runtime.trainer.set_model_mode("train")
        runtime.trainer.epoch = loaded.progress.epoch_zero_based
        iterator = iter(runtime.trainer.train_loader_x)
        raw_batch = next(iterator)
        del iterator
        image, label = runtime.trainer.parse_batch_train(raw_batch)
        batch = (image, label)
        with TransitionGuard(runtime.trainer.optim, runtime.trainer.sched):
            with runtime.precision.autocast_context():
                loss = F.cross_entropy(runtime.model(batch[0]), batch[1])
            gradients = torch.autograd.grad(
                loss,
                runtime.param_index.parameters,
                create_graph=False,
                retain_graph=False,
                allow_unused=False,
            )
            batch_gradient = GradientState.from_tensors(
                runtime.param_index, gradients
            )
            estimator_result = runtime.estimator.global_direction(
                batch_grad=batch_gradient, optimizer_step=step
            )
            reference = runtime.engine.diagnostic_coordinator.reference_for_step(
                estimator_result,
                optimizer_step=step,
                epoch=loaded.progress.epoch_zero_based,
                batch_index=0,
            )
            if reference is None or not reference.exact_service_query_issued:
                raise LC02DiagnosticAuditError("LC02 audit did not issue its isolated exact query")
            estimated_projection = project_batch_gradient(
                batch_gradient, estimator_result.active_global_estimate
            )
            exact_projection = project_batch_gradient(
                batch_gradient, reference.exact_reference
            )
            projected = _state_comparison(
                estimated_projection.batch_component,
                exact_projection.batch_component,
            )
            legacy = _state_comparison(
                estimated_projection.batch_component,
                reference.exact_reference,
            )
            global_fidelity = _state_comparison(
                estimator_result.active_global_estimate,
                reference.exact_reference,
            )
        if any(
            not torch.equal(entry.parameter.detach(), before)
            for entry, before in zip(runtime.param_index, prompt_before)
        ):
            raise LC02DiagnosticAuditError("Read-only LC02 audit changed the prompt")
        if hash_frozen_parameters(runtime.model) != frozen_before:
            raise LC02DiagnosticAuditError("Read-only LC02 audit changed frozen CLIP")
        old = old_rows[step]
        old_metrics = old["metrics"]
        observed = {
            "batch_norm": float(batch_gradient.norm().item()),
            "global_norm": float(estimator_result.active_global_estimate.norm().item()),
            "exact_norm": float(reference.exact_reference.norm().item()),
            "legacy_cosine": legacy["cosine"],
        }
        expected = {
            "batch_norm": old_metrics["grad/batch_norm"],
            "global_norm": old_metrics["grad/global_estimate_norm"],
            "exact_norm": old_metrics["grad/exact_full_norm"],
            "legacy_cosine": old["gB_exact_fidelity"],
        }
        for key in parity_max:
            difference = abs(float(observed[key]) - float(expected[key]))
            parity_max[key] = max(parity_max[key], difference)
            if difference > 1e-5:
                raise LC02DiagnosticAuditError(
                    f"Recomputed LC02 state differs from saved diagnostic for {step}:{key}"
                )
        corrected.append(
            {
                "schema_version": LC02_AUDIT_SCHEMA,
                "optimizer_step": step,
                "epoch": loaded.progress.epoch_zero_based,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": sha256_file(checkpoint),
                "original_gB_exact_fidelity": old["gB_exact_fidelity"],
                "original_gB_exact_fidelity_semantics": LEGACY_SEMANTICS,
                "gB_est_exact_cosine": projected["cosine"],
                "gB_est_exact_relative_l2": projected["relative_l2"],
                "gB_est_exact_norm_ratio": projected["norm_ratio"],
                "gB_est_norm": projected["left_norm"],
                "gB_exact_norm": projected["right_norm"],
                "gB_degenerate": projected["degenerate"],
                "actual_ema_exact_cosine": global_fidelity["cosine"],
                "actual_ema_exact_relative_l2": global_fidelity["relative_l2"],
                "exact_query_sample_count": reference.full_gradient_metadata.sample_count,
                "exact_query_micro_batch_count": reference.full_gradient_metadata.micro_batch_count,
                "prompt_restored_exactly": True,
                "optimizer_state_unchanged": True,
                "scheduler_state_unchanged": True,
            }
        )
        del runtime
    after_lc02 = hash_directory(lc02_run)
    after_source = hash_directory(source_run)
    if before_lc02 != after_lc02 or before_source != after_source:
        raise LC02DiagnosticAuditError("An immutable source run changed during LC02 audit")
    cosine_values = [float(row["gB_est_exact_cosine"]) for row in corrected]
    relative_values = [float(row["gB_est_exact_relative_l2"]) for row in corrected]
    ratio_values = [float(row["gB_est_exact_norm_ratio"]) for row in corrected]
    _write_jsonl(run_dir / "corrected_diagnostics.jsonl", corrected)
    provenance = {
        "schema_version": LC02_AUDIT_SCHEMA,
        "repository_sha": _git_sha(),
        "original_lc02_run": str(lc02_run),
        "original_lc02_run_id": summary["run_identity"]["run_id"],
        "original_lc02_summary_sha256": sha256_file(lc02_run / "summary.json"),
        "original_lc02_final_checkpoint_sha256": summary["artifacts"]["checkpoint_metadata"]["sha256"],
        "source_r2_run": str(source_run),
        "source_r2_run_id": config["low_compute"]["source_run_id"],
        "source_checkpoint": config["low_compute"]["source_checkpoint"],
        "source_checkpoint_sha256": config["low_compute"]["source_checkpoint_sha256"],
        "manifest_fingerprint": config["low_compute"]["data_manifest_fingerprint"],
        "clip_sha256": config["low_compute"]["clip_sha256"],
        "original_lc02_tree_sha256": hash_inventory_fingerprint(before_lc02),
        "source_r2_tree_sha256": hash_inventory_fingerprint(before_source),
    }
    atomic_write_json(run_dir / "provenance.json", provenance)
    atomic_write_json(
        run_dir / "source_hashes.json",
        {
            "schema_version": LC02_AUDIT_SCHEMA,
            "original_lc02_before": before_lc02,
            "original_lc02_after": after_lc02,
            "source_r2_before": before_source,
            "source_r2_after": after_source,
            "all_unchanged": True,
        },
    )
    atomic_write_json(
        run_dir / "summary.json",
        {
            "schema_version": LC02_AUDIT_SCHEMA,
            "status": "completed",
            "classification": "CATEGORY C — DIAGNOSTIC COMPUTATION BUG ONLY",
            "diagnostic_points_recomputed": len(corrected),
            "corrected_metrics": {
                "gB_est_exact_cosine_mean": statistics.fmean(cosine_values),
                "gB_est_exact_cosine_min": min(cosine_values),
                "gB_est_exact_cosine_max": max(cosine_values),
                "gB_est_exact_relative_l2_mean": statistics.fmean(relative_values),
                "gB_est_exact_norm_ratio_mean": statistics.fmean(ratio_values),
            },
            "saved_diagnostic_parity_max_absolute_difference": parity_max,
            "safety": {
                "optimizer_steps_executed": 0,
                "scheduler_steps_executed": 0,
                "prompt_parameters_changed": False,
                "frozen_clip_changed": False,
                "original_lc02_artifacts_changed": False,
                "source_r2_artifacts_changed": False,
                "training_rerun_required": False,
            },
            "artifacts": {
                "corrected_diagnostics": "corrected_diagnostics.jsonl",
                "provenance": "provenance.json",
                "source_hashes": "source_hashes.json",
            },
        },
    )
    return run_dir
