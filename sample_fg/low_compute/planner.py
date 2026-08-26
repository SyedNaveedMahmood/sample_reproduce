"""Canonical LC01+LC04 checkpoint selection and dry-run budget planning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from sample_fg.checkpoint import CHECKPOINT_SCHEMA_VERSION

from .budget import ComputeBudget
from .checkpoint_probe import ProbeCheckpointError, sha256_file


TARGET_EPOCHS = (20, 60, 100, 140, 200)


@dataclass(frozen=True)
class PlannedCheckpoint:
    path: Path
    sha256: str
    epoch: int
    optimizer_step: int
    materialization_replicates: int


@dataclass(frozen=True)
class IntegratedProbePlan:
    source_run: Path
    checkpoints: tuple[PlannedCheckpoint, ...]
    lambda_grid: tuple[float, ...]
    order_trials: int
    analysis_seed: int
    radii: tuple[float, ...]
    selected_examples: int
    fixed_batches_per_replicate: int
    stationary_replay_epochs: int
    budget: ComputeBudget

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "DRY_RUN_VALIDATED",
            "dry_run": True,
            "training_started": False,
            "artifacts_created": False,
            "task": "lc01_lc04",
            "source_run": str(self.source_run),
            "checkpoints": [
                {
                    "path": str(item.path), "sha256": item.sha256,
                    "epoch": item.epoch,
                    "optimizer_step": item.optimizer_step,
                    "materialization_replicates": item.materialization_replicates,
                }
                for item in self.checkpoints
            ],
            "gradient_source": {
                "primary": "fixed_materialized_training_transform",
                "optional_reference": "independent_full_gradient_service_transform",
                "selected_examples": self.selected_examples,
                "fixed_batches_per_replicate": self.fixed_batches_per_replicate,
            },
            "lambda_grid": list(self.lambda_grid),
            "paper_lambda": 0.15,
            "coverage_lambda": 0.8461538461538461,
            "order_trials": self.order_trials,
            "analysis_seed": self.analysis_seed,
            "stationary_replay_epochs": self.stationary_replay_epochs,
            "function_space": {
                "checkpoints": [item.epoch for item in self.checkpoints],
                "radii": list(self.radii),
                "probe_set": "canonical DTD Base+New evaluation source",
                "eval_image_feature_cache_required": True,
                "backward_batches": 0,
            },
            "budget": self.budget.as_dict(),
        }


def load_campaign_config(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ProbeCheckpointError(f"Cannot read low-compute config: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != "sample_fg.low_compute_config.v1":
        raise ProbeCheckpointError("Unsupported low-compute config schema")
    return value


def _header(path: Path) -> tuple[int, int]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ProbeCheckpointError(f"Cannot inspect checkpoint: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ProbeCheckpointError(f"Unsupported scientific checkpoint: {path}")
    if payload.get("boundary") != "after_logical_optimizer_step_unperturbed_v1":
        raise ProbeCheckpointError(f"Checkpoint is not an unperturbed step boundary: {path}")
    if payload.get("method") != "sample" or payload.get("estimator_mode") != "ema":
        raise ProbeCheckpointError(f"Checkpoint is not SAMPLe-EMA: {path}")
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        raise ProbeCheckpointError(f"Checkpoint progress is malformed: {path}")
    epoch = progress.get("epoch_zero_based")
    step = progress.get("next_optimizer_step")
    batch_index = progress.get("next_batch_index_zero_based")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (epoch, step)):
        raise ProbeCheckpointError(f"Checkpoint progress counters are invalid: {path}")
    if batch_index != 0:
        raise ProbeCheckpointError(f"Checkpoint is not at an epoch boundary: {path}")
    return epoch, step


def select_checkpoints(source_run: Path, targets: Sequence[int] = TARGET_EPOCHS) -> tuple[PlannedCheckpoint, ...]:
    root = Path(source_run).resolve(strict=True)
    candidates: list[tuple[Path, int, int]] = []
    for path in sorted((root / "checkpoints").glob("*.pt")):
        epoch, step = _header(path)
        candidates.append((path.resolve(), epoch, step))
    if not candidates:
        raise ProbeCheckpointError("Source run contains no scientific checkpoints")
    selected = []
    used: set[Path] = set()
    for target in targets:
        path, epoch, step = min(candidates, key=lambda item: (abs(item[1] - target), item[1], str(item[0])))
        if path in used:
            raise ProbeCheckpointError(
                f"Checkpoint policy cannot resolve five distinct targets; duplicate near epoch {target}"
            )
        used.add(path)
        selected.append(
            PlannedCheckpoint(
                path=path,
                sha256=sha256_file(path),
                epoch=epoch,
                optimizer_step=step,
                materialization_replicates=3 if target == max(targets) else 1,
            )
        )
    return tuple(selected)


def build_integrated_plan(
    *,
    source_run: Path,
    config_path: Path,
    order_trials: int | None = None,
    analysis_seed: int = 10401,
) -> IntegratedProbePlan:
    config = load_campaign_config(config_path)
    common = config["common"]
    lc01 = config["lc01"]
    lc04 = config["lc04"]
    trials = int(lc01["order_trials"] if order_trials is None else order_trials)
    if trials != int(lc01["order_trials"]):
        raise ProbeCheckpointError("Scientific LC01 requires all 512 order trials")
    checkpoints = select_checkpoints(source_run, tuple(common["checkpoint_target_epochs"]))
    batches = int(common["selected_examples"]) // int(common["train_batch_size"])
    bank_backward = sum(item.materialization_replicates for item in checkpoints) * batches
    exact_sweeps = int(lc01["optional_independent_exact_sweeps"])
    exact_backward = exact_sweeps * batches
    budget = ComputeBudget(
        optimizer_steps=0,
        scheduler_steps=0,
        normal_forward_batches=bank_backward,
        normal_backward_batches=bank_backward,
        exact_forward_batches=exact_backward,
        exact_backward_batches=exact_backward,
        # DTD has 960 Base and 920 New test images; pinned test batch is 100.
        image_encoder_forward_batches=bank_backward + exact_backward + 20,
        # LC04 primary replicate only: 5 checkpoints * 2 h * 2 directions *
        # 2 signs * 2 Base/New text encoders = 80 calls.
        text_encoder_forward_calls=bank_backward + exact_backward + 80,
        exact_sweeps=exact_sweeps,
    )
    if budget.backward_batches > int(lc01["max_backward_batches"]):
        raise ProbeCheckpointError("Resolved LC01 work exceeds its backward-batch permit")
    budget.require_read_only()
    return IntegratedProbePlan(
        source_run=Path(source_run).resolve(strict=True),
        checkpoints=checkpoints,
        lambda_grid=tuple(float(value) for value in lc01["lambda_grid"]),
        order_trials=trials,
        analysis_seed=int(analysis_seed),
        radii=tuple(float(value) for value in lc04["finite_difference_h"]),
        selected_examples=int(common["selected_examples"]),
        fixed_batches_per_replicate=batches,
        stationary_replay_epochs=int(lc01["stationary_replay_epochs"]),
        budget=budget,
    )
