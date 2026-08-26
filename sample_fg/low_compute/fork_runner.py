"""Explicit, state-preserving LC02 fork of the completed R2 trajectory."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from sample_fg.checkpoint import CheckpointProgress, load_scientific_checkpoint
from sample_fg.coop_anchor import EXPECTED_CLIP_SHA256
from sample_fg.estimators import EMAEstimator
from sample_fg.paper_runner import (
    MethodSelection,
    RunAccounting,
    ScientificPlan,
    _build_runtime,
    _checkpoint_generators,
    _diagnostic_summary,
    _new_run,
    _save_checkpoint,
    run_scientific,
)
from sample_fg.results import atomic_write_json, load_jsonl, resolve_config

from .artifacts import SUMMARY_SCHEMA_VERSION, validate_saved_artifacts
from .budget import _state_equal
from .checkpoint_probe import (
    ProbeCheckpoint,
    ProbeCheckpointError,
    load_probe_checkpoint,
    sha256_file,
    verify_source_immutable,
)
from .planner import load_campaign_config


SOURCE_LAMBDA = 0.15
COVERAGE_LAMBDA = 11.0 / 13.0
INTERVENTION = "state_preserving_ema_decay_switch_v1"
SOURCE_EPOCH = 180
TARGET_EPOCH = 200
STEPS_PER_EPOCH = 12
MAX_OPTIMIZER_STEPS = 240
FORK_SCHEMA_VERSION = "sample_fg.low_compute_fork.v1"


class LowComputeForkError(RuntimeError):
    """Raised before LC02 can create or continue an incompatible branch."""


def _mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise LowComputeForkError(f"Cannot read fork input: {path}") from error
    if not isinstance(value, dict):
        raise LowComputeForkError(f"Fork input is not a mapping: {path}")
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
    return value if result.returncode == 0 and len(value) == 40 else "unavailable"


def source_tree_sha256(root: Path) -> str:
    """Hash every regular source-run artifact without following outside paths."""

    root = Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class GateProof:
    lc01_run_dir: Path
    summary_path: Path
    summary_sha256: str
    source_artifact_sha256: str
    passing_checkpoints: int
    trajectory: tuple[tuple[int, str], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "lc01_run_dir": str(self.lc01_run_dir),
            "summary": str(self.summary_path),
            "summary_sha256": self.summary_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "passing_checkpoints": self.passing_checkpoints,
            "trajectory": [
                {"epoch": epoch, "checkpoint_sha256": value}
                for epoch, value in self.trajectory
            ],
            "accuracy_fields_consumed": False,
            "gate_passed": True,
        }


def verify_lc01_gate(
    summary_path: Path,
    *,
    source_run: Path,
    campaign_config: Path,
) -> GateProof:
    """Verify the completed mechanism-only gate and bind it to its bytes/source."""

    summary_path = Path(summary_path).resolve(strict=True)
    if summary_path.name != "summary.json":
        raise LowComputeForkError("LC01 gate must be the canonical summary.json")
    lc01_root = summary_path.parent
    validate_saved_artifacts(lc01_root)
    summary = _mapping(summary_path)
    source = _mapping(lc01_root / "source.json")
    config = _mapping(lc01_root / "config.yaml")
    campaign = load_campaign_config(campaign_config)
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION or summary.get("status") != "completed":
        raise LowComputeForkError("LC01 summary is not a completed canonical artifact")
    safety = summary.get("safety")
    if not isinstance(safety, dict) or any(
        safety.get(key) != expected
        for key, expected in (
            ("optimizer_steps_executed", 0),
            ("scheduler_steps_executed", 0),
            ("model_parameters_changed", False),
            ("source_artifacts_changed", False),
        )
    ):
        raise LowComputeForkError("LC01 safety proof is incomplete")
    findings = summary.get("primary_findings")
    gate = findings.get("lc02_gate") if isinstance(findings, dict) else None
    if not isinstance(gate, dict) or gate.get("schema_version") != "sample_fg.low_compute_lc02_gate.v1":
        raise LowComputeForkError("LC01 mechanism gate is missing or has the wrong schema")
    if gate.get("gate_passed") is not True:
        raise LowComputeForkError("LC01 mechanism gate did not pass")
    if gate.get("accuracy_fields_consumed") is not False:
        raise LowComputeForkError("LC01 gate is accuracy-dependent")
    passing = gate.get("passing_checkpoint_count")
    evaluated = gate.get("evaluated_checkpoint_count")
    if passing != 5 or evaluated != 5:
        raise LowComputeForkError("LC01 gate does not cover the five registered checkpoints")
    resolved_source = Path(source.get("source_run", "")).resolve()
    if resolved_source != Path(source_run).resolve(strict=True):
        raise LowComputeForkError("LC01 gate is bound to a different R2 source run")
    expected_lambda = float(campaign["lc02"]["dtd_lambda_cov"])
    if expected_lambda != COVERAGE_LAMBDA or config.get("lc01", {}).get("coverage_lambda") != COVERAGE_LAMBDA:
        raise LowComputeForkError("LC01/manifest coverage lambda differs from 11/13")
    checkpoints = source.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 5:
        raise LowComputeForkError("LC01 source trajectory is incomplete")
    trajectory = []
    for item in checkpoints:
        if not isinstance(item, dict) or not isinstance(item.get("epoch"), int):
            raise LowComputeForkError("LC01 checkpoint identity is malformed")
        path = Path(item.get("path", "")).resolve(strict=True)
        observed = sha256_file(path)
        if observed != item.get("sha256"):
            raise LowComputeForkError("LC01 checkpoint hash is stale")
        trajectory.append((item["epoch"], observed))
    if tuple(epoch for epoch, _ in trajectory) != (20, 60, 100, 140, 200):
        raise LowComputeForkError("LC01 trajectory differs from the registered checkpoints")
    evidence_epochs = sorted(int(item["checkpoint"]) for item in gate.get("evidence", []))
    if evidence_epochs != [20, 60, 100, 140, 200]:
        raise LowComputeForkError("LC01 gate evidence differs from its source trajectory")
    return GateProof(
        lc01_run_dir=lc01_root,
        summary_path=summary_path,
        summary_sha256=sha256_file(summary_path),
        source_artifact_sha256=sha256_file(lc01_root / "source.json"),
        passing_checkpoints=passing,
        trajectory=tuple(trajectory),
    )


@dataclass(frozen=True)
class ForkSpecification:
    source_run_dir: Path
    source_checkpoint: Path
    source_checkpoint_sha256: str
    source_config_sha256: str
    source_epoch: int
    source_optimizer_step: int
    source_lambda: float
    target_lambda: float
    intervention: str
    target_epoch: int
    remaining_epochs: int
    steps_per_epoch: int
    max_optimizer_steps: int

    @property
    def optimizer_steps(self) -> int:
        return self.remaining_epochs * self.steps_per_epoch

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_run_dir": str(self.source_run_dir),
            "source_checkpoint": str(self.source_checkpoint),
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_config_sha256": self.source_config_sha256,
            "source_epoch": self.source_epoch,
            "source_optimizer_step": self.source_optimizer_step,
            "source_lambda": self.source_lambda,
            "new_lambda": self.target_lambda,
            "intervention": self.intervention,
            "target_epoch": self.target_epoch,
            "remaining_epochs": self.remaining_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "optimizer_steps": self.optimizer_steps,
            "max_optimizer_steps": self.max_optimizer_steps,
        }


@dataclass(frozen=True)
class LowComputeForkPlan:
    specification: ForkSpecification
    gate: GateProof
    source_probe: ProbeCheckpoint
    source_plan: ScientificPlan
    branch_plan: ScientificPlan
    source_tree_sha256_before: str
    baseline_final_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "DRY_RUN_VALIDATED",
            "dry_run": True,
            "execute": False,
            "training_started": False,
            "task": "lc02",
            "experiment_id": "LC2",
            "intervention": "coverage_aware_ema_decay",
            "fork": self.specification.as_dict(),
            "lc01_gate": self.gate.as_dict(),
            "branch_config_sha256": self.branch_plan.resolved_config["config_sha256"],
            "baseline_final_sha256": self.baseline_final_sha256,
            "source_tree_sha256": self.source_tree_sha256_before,
            "source_immutable": True,
            "optimizer_steps_executed": 0,
            "scheduler_steps_executed": 0,
        }


def _validate_source_final(source_run: Path, source_config_sha256: str) -> str:
    summary = _mapping(source_run / "summary.json")
    if summary.get("status") != "completed" or summary.get("smoke") is not False:
        raise LowComputeForkError("LC02 baseline must be a completed non-smoke R2 run")
    identity = summary.get("run_identity", {})
    if identity.get("config_sha256") != source_config_sha256 or identity.get("experiment_id") != "R2":
        raise LowComputeForkError("R2 final summary identity differs from the fork source")
    metadata = summary.get("artifacts", {}).get("checkpoint_metadata", {})
    final_path = (source_run / summary.get("artifacts", {}).get("checkpoint", "")).resolve(strict=True)
    observed = sha256_file(final_path)
    if metadata.get("sha256") != observed:
        raise LowComputeForkError("R2 final checkpoint hash differs from its summary")
    return observed


def _branch_config(
    source_plan: ScientificPlan,
    *,
    specification: ForkSpecification,
    gate: GateProof,
    output_root: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(source_plan.resolved_config)
    payload.pop("config_sha256", None)
    payload.pop("schema_version", None)
    payload["run"].update(
        {
            "experiment_id": "LC2",
            "output_root": str(output_root),
            "notes": "LC02 one-seed coverage-aware EMA causal branch",
        }
    )
    payload["method"]["ema_lambda"] = COVERAGE_LAMBDA
    payload["diagnostics"]["full_gradient_interval_policy"] = "once_per_branch_epoch"
    payload["diagnostics"]["full_gradient_interval_steps"] = STEPS_PER_EPOCH
    payload["checkpoint"]["recovery_interval_epochs"] = 1
    payload["checkpoint"]["recovery_interval_steps"] = STEPS_PER_EPOCH
    payload["checkpoint"]["resume_from"] = None
    payload["provenance"]["runner"] = "scripts/run_low_compute_branch.py"
    source_environment = _mapping(specification.source_run_dir / "environment.json")
    source_summary = _mapping(specification.source_run_dir / "summary.json")
    payload["low_compute"] = {
        "schema_version": FORK_SCHEMA_VERSION,
        "experiment_id": "LC2",
        "intervention": "coverage_aware_ema_decay",
        "estimator_state_transplant": INTERVENTION,
        "source_run_id": source_summary["run_identity"]["run_id"],
        "source_run_dir": str(specification.source_run_dir),
        "source_checkpoint": str(specification.source_checkpoint),
        "source_checkpoint_sha256": specification.source_checkpoint_sha256,
        "source_epoch": specification.source_epoch,
        "source_optimizer_step": specification.source_optimizer_step,
        "source_config_sha256": specification.source_config_sha256,
        "source_lambda": SOURCE_LAMBDA,
        "new_lambda": COVERAGE_LAMBDA,
        "lambda_derivation": "(B-1)/(B+1)",
        "B": STEPS_PER_EPOCH,
        "target_epoch": TARGET_EPOCH,
        "remaining_epochs": specification.remaining_epochs,
        "max_optimizer_steps": MAX_OPTIMIZER_STEPS,
        "lc01_gate": gate.as_dict(),
        "source_git_sha": source_environment.get("git", {}).get("project_commit"),
        "current_git_sha": _git_sha(),
        "clip_sha256": EXPECTED_CLIP_SHA256,
        "data_manifest_fingerprint": source_plan.source.fingerprint,
    }
    return resolve_config(payload)


def build_low_compute_fork_plan(
    *,
    source_run: Path,
    source_checkpoint: Path,
    lc01_summary: Path,
    campaign_config: Path,
    source_plan: ScientificPlan,
    output_root: Path,
    resume_from: Path | None = None,
) -> LowComputeForkPlan:
    """Build the only registered LC02 intervention; never restore or step."""

    campaign = load_campaign_config(campaign_config)
    lc02 = campaign.get("lc02", {})
    if (
        lc02.get("estimator_state_transplant") != INTERVENTION
        or lc02.get("dtd_lambda_cov") != COVERAGE_LAMBDA
        or lc02.get("max_optimizer_steps") != MAX_OPTIMIZER_STEPS
    ):
        raise LowComputeForkError("Campaign does not register the LC02 intervention")
    source_run = Path(source_run).resolve(strict=True)
    if source_plan.dataset != "dtd" or source_plan.shots != 16 or source_plan.seed != 1:
        raise LowComputeForkError("LC02 source must be DTD/16-shot/seed-1")
    if source_plan.selection.method != "sample" or source_plan.selection.estimator != "ema":
        raise LowComputeForkError("LC02 source must be SAMPLe-EMA")
    if source_plan.selection.ema_lambda != SOURCE_LAMBDA:
        raise LowComputeForkError("LC02 source lambda must be 0.15")
    if source_plan.steps_per_epoch != STEPS_PER_EPOCH or source_plan.epochs != TARGET_EPOCH:
        raise LowComputeForkError("LC02 source clocks differ from DTD R2")
    probe = load_probe_checkpoint(source_run, source_checkpoint)
    if probe.epoch_zero_based != SOURCE_EPOCH or probe.next_optimizer_step != SOURCE_EPOCH * STEPS_PER_EPOCH:
        raise LowComputeForkError("Primary LC02 fork must be the exact epoch-180 boundary")
    remaining = TARGET_EPOCH - probe.epoch_zero_based
    requested = remaining * STEPS_PER_EPOCH
    if remaining != 20 or requested != MAX_OPTIMIZER_STEPS:
        raise LowComputeForkError("LC02 primary branch must resolve exactly 20 epochs/240 steps")
    if requested > MAX_OPTIMIZER_STEPS or probe.epoch_zero_based < SOURCE_EPOCH:
        raise LowComputeForkError("LC02 fork exceeds its optimizer-step permit")
    gate = verify_lc01_gate(
        lc01_summary, source_run=source_run, campaign_config=campaign_config
    )
    specification = ForkSpecification(
        source_run_dir=source_run,
        source_checkpoint=probe.checkpoint_path,
        source_checkpoint_sha256=probe.checkpoint_sha256,
        source_config_sha256=probe.source_config_sha256,
        source_epoch=probe.epoch_zero_based,
        source_optimizer_step=probe.next_optimizer_step,
        source_lambda=SOURCE_LAMBDA,
        target_lambda=COVERAGE_LAMBDA,
        intervention=INTERVENTION,
        target_epoch=TARGET_EPOCH,
        remaining_epochs=remaining,
        steps_per_epoch=STEPS_PER_EPOCH,
        max_optimizer_steps=MAX_OPTIMIZER_STEPS,
    )
    branch_root = Path(output_root).resolve() / "lc02"
    target_selection = MethodSelection(
        method="sample",
        estimator="ema",
        method_tag="sample_coverage",
        estimator_tag="ema",
        rho=0.05,
        alpha=0.0015,
        ema_lambda=COVERAGE_LAMBDA,
        refresh_k_steps=None,
    )
    resolved = _branch_config(
        source_plan,
        specification=specification,
        gate=gate,
        output_root=branch_root,
    )
    branch_plan = replace(
        source_plan,
        experiment_id="LC2",
        selection=target_selection,
        output_root=branch_root,
        recovery_interval_epochs=1,
        diagnostic_interval_steps=STEPS_PER_EPOCH,
        resolved_config=resolved,
        resume_from=Path(resume_from).resolve(strict=True) if resume_from else None,
    )
    baseline_final = _validate_source_final(source_run, probe.source_config_sha256)
    tree_hash = source_tree_sha256(source_run)
    plan = LowComputeForkPlan(
        specification=specification,
        gate=gate,
        source_probe=probe,
        source_plan=source_plan,
        branch_plan=branch_plan,
        source_tree_sha256_before=tree_hash,
        baseline_final_sha256=baseline_final,
    )
    if resume_from is not None:
        validate_branch_resume_checkpoint(plan, branch_plan.resume_from)
    return plan


def transplant_ema_state_preserving_direction(
    source: EMAEstimator,
    *,
    target_lambda: float,
) -> EMAEstimator:
    """Create a new EMA estimator while changing only its registered decay."""

    if not isinstance(source, EMAEstimator) or source.ema_lambda != SOURCE_LAMBDA:
        raise LowComputeForkError("Fork source estimator is not paper-lambda EMA")
    if target_lambda != COVERAGE_LAMBDA:
        raise LowComputeForkError("Only the registered 11/13 target lambda is allowed")
    source_payload = source.state_dict()
    target = EMAEstimator(source.param_index, ema_lambda=target_lambda)
    transplanted = copy.deepcopy(source_payload)
    transplanted["ema_lambda"] = target_lambda
    target.load_state_dict(transplanted)
    before = source.active_state
    after = target.active_state
    if any(not torch.equal(a, b) for a, b in zip(before, after)):
        raise LowComputeForkError("EMA active direction changed during transplant")
    if target.last_processed_step != source.last_processed_step or target.exact_query_count != source.exact_query_count:
        raise LowComputeForkError("EMA clocks changed during transplant")
    return target


def restore_fork_runtime_transactionally(plan: LowComputeForkPlan, runtime):
    """Strictly restore source identity first, then perform the one-field fork."""

    loaded = load_scientific_checkpoint(
        plan.specification.source_checkpoint,
        param_index=runtime.param_index,
        optimizer=runtime.trainer.optim,
        scheduler=runtime.trainer.sched,
        precision_controller=runtime.precision,
        step_engine=runtime.engine,
        estimator=runtime.estimator,
        perturbation=runtime.perturbation,
        expected_method="sample",
        expected_config_sha256=plan.source_probe.source_config_sha256,
        expected_source_fingerprint=plan.source_probe.source_fingerprint,
        explicit_generators=_checkpoint_generators(runtime),
    )
    if loaded.progress != CheckpointProgress(
        plan.specification.source_optimizer_step,
        plan.specification.source_epoch,
        0,
        loaded.progress.normal_samples_seen,
    ):
        raise LowComputeForkError("Restored source progress differs from the fork boundary")
    if not isinstance(runtime.estimator, EMAEstimator):
        raise LowComputeForkError("Restored fork runtime lacks EMA")
    source_state = runtime.estimator.state_dict()
    replacement = transplant_ema_state_preserving_direction(
        runtime.estimator, target_lambda=plan.specification.target_lambda
    )
    runtime.estimator = replacement
    target_state = replacement.state_dict()
    for key in set(source_state) - {"ema_lambda"}:
        if not _state_equal(source_state[key], target_state[key]):
            raise LowComputeForkError(f"Fork changed unauthorized estimator field: {key}")
    verify_source_immutable(plan.source_probe)
    return loaded


def validate_branch_resume_checkpoint(
    plan: LowComputeForkPlan, checkpoint: Path | None
) -> dict[str, Any]:
    if checkpoint is None:
        raise LowComputeForkError("Branch resume checkpoint is missing")
    checkpoint = Path(checkpoint).resolve(strict=True)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise LowComputeForkError("Cannot load LC02 branch checkpoint") from error
    if not isinstance(payload, dict):
        raise LowComputeForkError("LC02 branch checkpoint is malformed")
    if payload.get("config_sha256") != plan.branch_plan.resolved_config["config_sha256"]:
        raise LowComputeForkError("LC02 branch checkpoint config differs")
    estimator = payload.get("estimator_state")
    if not isinstance(estimator, dict) or estimator.get("ema_lambda") != COVERAGE_LAMBDA:
        raise LowComputeForkError("LC02 branch checkpoint lost its target lambda")
    provenance = payload.get("result_state", {}).get("low_compute_fork")
    expected = plan.branch_plan.resolved_config["low_compute"]
    if provenance != expected:
        raise LowComputeForkError("LC02 branch checkpoint fork provenance differs")
    progress = CheckpointProgress.from_dict(payload.get("progress"))
    consumed = progress.next_optimizer_step - SOURCE_EPOCH * STEPS_PER_EPOCH
    if progress.epoch_zero_based > TARGET_EPOCH or consumed < 0 or consumed > MAX_OPTIMIZER_STEPS:
        raise LowComputeForkError("LC02 branch checkpoint exceeds its authorized horizon")
    return payload


def run_low_compute_fork(plan: LowComputeForkPlan) -> Path:
    """Execute LC02 only after explicit CLI authorization."""

    before = source_tree_sha256(plan.specification.source_run_dir)
    if before != plan.source_tree_sha256_before:
        raise LowComputeForkError("R2 source changed after LC02 planning")
    if plan.branch_plan.resume_from is not None:
        validate_branch_resume_checkpoint(plan, plan.branch_plan.resume_from)
        result = run_scientific(plan.branch_plan)
    else:
        artifacts, _ = _new_run(plan.branch_plan)
        runtime = _build_runtime(plan.source_plan, artifacts.run_dir / "fork_restore")
        loaded = restore_fork_runtime_transactionally(plan, runtime)
        fork_state = {
            "accounting": RunAccounting().as_dict(),
            "scheduler_steps": plan.specification.source_epoch,
            "normal_samples_seen": loaded.progress.normal_samples_seen,
            "metric_records": 0,
            "diagnostic_records": 0,
            "resume_events": 0,
            "normal_loader_resume_scope": "completed_epoch_boundary_only_workers_8",
        }
        fork_checkpoint = artifacts.run_dir / "checkpoints" / "fork.pt"
        _save_checkpoint(
            fork_checkpoint,
            plan=plan.branch_plan,
            runtime=runtime,
            progress=loaded.progress,
            result_state=fork_state,
        )
        validate_branch_resume_checkpoint(plan, fork_checkpoint)
        del runtime
        executable = replace(plan.branch_plan, resume_from=fork_checkpoint)
        result = run_scientific(executable)
    _finalize_branch_summary(plan, result)
    after = source_tree_sha256(plan.specification.source_run_dir)
    if after != before:
        raise LowComputeForkError("R2 source artifacts changed during LC02")
    return result


def _finalize_branch_summary(plan: LowComputeForkPlan, run_dir: Path) -> None:
    """Add the immutable R2 counterfactual and branch-only deltas."""

    branch_path = Path(run_dir) / "summary.json"
    branch = _mapping(branch_path)
    baseline = _mapping(plan.specification.source_run_dir / "summary.json")
    branch_eval = branch.get("evaluation", {})
    baseline_eval = baseline.get("evaluation", {})
    keys = ("base_accuracy_pct", "new_accuracy_pct", "hm_pct")
    if any(not isinstance(mapping.get(key), (int, float)) for mapping in (branch_eval, baseline_eval) for key in keys):
        raise LowComputeForkError("LC02 branch/baseline evaluation is incomplete")
    source_diagnostics = [
        row
        for row in load_jsonl(plan.specification.source_run_dir / "gradient_diagnostics.jsonl")
        if int(row.get("optimizer_step", -1)) >= plan.specification.source_optimizer_step
    ]
    source_mechanism = _diagnostic_summary(source_diagnostics)
    branch_mechanism = branch.get("estimator_diagnostics", {})
    mechanism_keys = (
        "global_estimate_exact_cosine_mean",
        "global_estimate_exact_relative_l2_mean",
        "batch_component_estimate_exact_cosine_mean",
        "batch_component_estimate_exact_relative_l2_mean",
        "batch_component_estimate_exact_norm_ratio_mean",
    )
    branch["counterfactual_baseline"] = {
        "source_run_dir": str(plan.specification.source_run_dir),
        "source_final_checkpoint_sha256": plan.baseline_final_sha256,
        "evaluation": {key: baseline_eval[key] for key in keys},
        "branch_minus_baseline": {
            key: float(branch_eval[key]) - float(baseline_eval[key]) for key in keys
        },
        "mechanism_source_window": {
            "absolute_optimizer_step_start": plan.specification.source_optimizer_step,
            "absolute_optimizer_step_stop": TARGET_EPOCH * STEPS_PER_EPOCH,
            "source": {key: source_mechanism.get(key) for key in mechanism_keys},
            "branch": {key: branch_mechanism.get(key) for key in mechanism_keys},
            "branch_minus_source": {
                key: (
                    None
                    if source_mechanism.get(key) is None or branch_mechanism.get(key) is None
                    else float(branch_mechanism[key]) - float(source_mechanism[key])
                )
                for key in mechanism_keys
            },
        },
        "interpretation": "one_seed_causal_pilot_not_population_evidence",
    }
    atomic_write_json(branch_path, branch)
