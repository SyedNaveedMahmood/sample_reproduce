"""LC03 zero-step fidelity/geometry/generalization trajectory probe."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed

from sample_fg.coop_anchor import (
    audit_prompt_only_training,
    build_coop_trainer,
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.diagnostics import compute_gradient_diagnostics
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.paper_runner import _build_runtime, build_scientific_cfg
from sample_fg.projection import project_batch_gradient, safe_unit
from sample_fg.results import atomic_write_json

from .budget import ComputeBudget, TransitionGuard
from .checkpoint_probe import (
    load_probe_checkpoint,
    sha256_file,
    verify_source_immutable,
)
from .feature_cache import FeatureCacheKey, load_feature_cache
from .gradient_bank import gradient_sha256
from .math import gradient_metrics
from .replay import projection_displacement_metrics
from .runner import _device_state, _gradient_bank, _text_features


LC03_SCHEMA_VERSION = "sample_fg.low_compute_lc03.v1"
TRAJECTORY_SCHEMA_VERSION = "sample_fg.low_compute_lc03_trajectory.v1"
TAYLOR_SCHEMA_VERSION = "sample_fg.low_compute_lc03_taylor.v1"
SUMMARY_SCHEMA_VERSION = "sample_fg.low_compute_lc03_summary.v1"
MAX_DISPLACED_BACKWARDS = 60
PARITY_TOLERANCE_PCT = 1e-9


class TrajectoryProbeError(RuntimeError):
    """Raised before an LC03 result can use incompatible source material."""


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise TrajectoryProbeError(f"Cannot read LC03 input: {path}") from error
    if not isinstance(value, dict):
        raise TrajectoryProbeError(f"LC03 input is not a mapping: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except (OSError, UnicodeError, ValueError) as error:
        raise TrajectoryProbeError(f"Cannot read LC03 JSONL input: {path}") from error
    return rows


def _tree_sha256(root: Path) -> str:
    root = Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    epoch: int
    path: Path
    sha256: str
    feature_cache: Path
    feature_cache_sha256: str
    exact_gradient_sha256: str
    expected_batch_hashes: tuple[str, ...]
    lc01_fidelity: Mapping[str, Any]
    lc01_projection_summary: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "path": str(self.path),
            "sha256": self.sha256,
            "feature_cache": str(self.feature_cache),
            "feature_cache_sha256": self.feature_cache_sha256,
            "exact_gradient_sha256": self.exact_gradient_sha256,
            "batch_count": len(self.expected_batch_hashes),
            "lc01_fidelity_reused": True,
            "lc01_projection_summary_reused": True,
        }


@dataclass(frozen=True)
class TrajectoryPlan:
    lc01_run: Path
    source_run: Path
    checkpoints: tuple[TrajectoryCheckpoint, ...]
    missing_checkpoints: tuple[int, ...]
    source_config_sha256: str
    source_fingerprint: str
    clip_sha256: str
    eval_cache: Path
    eval_cache_sha256: str
    base_class_count: int
    new_class_count: int
    base_samples: int
    new_samples: int
    baseline_evaluation: Mapping[str, float]
    lc01_tree_sha256_before: str
    source_tree_sha256_before: str
    budget: ComputeBudget
    gradient_bank_reuse: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "DRY_RUN_VALIDATED",
            "dry_run": True,
            "execute": False,
            "training_started": False,
            "task": "lc03",
            "observational": True,
            "lc01_run": str(self.lc01_run),
            "source_run": str(self.source_run),
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": [item.as_dict() for item in self.checkpoints],
            "missing_checkpoints": list(self.missing_checkpoints),
            "checkpoint_identity_source": "LC01 source.json hashes",
            "gradient_bank_reuse": self.gradient_bank_reuse,
            "gradient_bank_regeneration_backward_batches": len(self.checkpoints) * 12,
            "lc01_tree_sha256": self.lc01_tree_sha256_before,
            "source_tree_sha256": self.source_tree_sha256_before,
            "budget": self.budget.as_dict(),
            "additional_displaced_backward_batches": self.budget.normal_backward_batches,
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "source_immutable": True,
        }


def build_trajectory_plan(
    *,
    lc01_run: Path,
    source_run: Path | None = None,
    campaign_config: Path | None = None,
) -> TrajectoryPlan:
    """Consume LC01-owned checkpoint/cache identities without selecting anew."""

    lc01_run = Path(lc01_run).resolve(strict=True)
    summary = _mapping(lc01_run / "summary.json")
    source = _mapping(lc01_run / "source.json")
    config_path = lc01_run / "config.yaml"
    try:
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise TrajectoryProbeError("Cannot read LC01 config") from error
    if not isinstance(config, dict) or config.get("schema_version") != "sample_fg.low_compute_config.v1":
        raise TrajectoryProbeError("LC01 config schema differs")
    if campaign_config is not None:
        from .planner import load_campaign_config

        campaign = load_campaign_config(campaign_config)
        lc03 = campaign.get("lc03", {})
        if (
            lc03.get("reuse_lc01_gradient_banks") is not True
            or lc03.get("max_additional_displaced_backward_batches")
            != MAX_DISPLACED_BACKWARDS
            or lc03.get("optimizer_steps") != 0
            or lc03.get("scheduler_steps") != 0
            or lc03.get("significance_testing") is not False
        ):
            raise TrajectoryProbeError("Campaign does not register the LC03 zero-step probe")
    safety = summary.get("safety")
    if summary.get("status") != "completed" or not isinstance(safety, dict):
        raise TrajectoryProbeError("LC03 requires a completed LC01 run")
    if safety.get("optimizer_steps_executed") != 0 or safety.get("source_artifacts_changed") is not False:
        raise TrajectoryProbeError("LC01 safety proof is incompatible with LC03")
    resolved_source = Path(source.get("source_run", "")).resolve(strict=True)
    if source_run is not None and resolved_source != Path(source_run).resolve(strict=True):
        raise TrajectoryProbeError("Requested R2 source differs from LC01")
    source_run = resolved_source
    source_summary = _mapping(source_run / "summary.json")
    if source_summary.get("status") != "completed" or source_summary.get("run_identity", {}).get("experiment_id") != "R2":
        raise TrajectoryProbeError("LC03 source is not a completed R2 run")
    if source_summary.get("run_identity", {}).get("config_sha256") != source.get("source_config_sha256"):
        raise TrajectoryProbeError("LC01 source config identity is stale")
    baseline = source_summary.get("evaluation")
    if not isinstance(baseline, dict) or baseline.get("new_retrained") is not False:
        raise TrajectoryProbeError("R2 Base/New evaluation contract is missing")

    index = _mapping(lc01_run / "lc01" / "gradient_bank_index.json")
    if index.get("schema_version") != "sample_fg.low_compute_gradient_bank_index.v1":
        raise TrajectoryProbeError("LC01 gradient-bank index schema differs")
    banks = index.get("banks")
    if not isinstance(banks, list):
        raise TrajectoryProbeError("LC01 gradient-bank index is malformed")
    primary = {}
    for bank in banks:
        if isinstance(bank, dict) and bank.get("materialization_replicate") == 0:
            primary[(bank.get("epoch"), bank.get("checkpoint_sha256"))] = bank
    fidelity_rows = {
        (row.get("epoch"), row.get("checkpoint_sha256")): row
        for row in _jsonl(lc01_run / "lc01" / "checkpoint_fidelity.jsonl")
        if row.get("materialization_replicate") == 0
    }
    geometry_payload = _mapping(lc01_run / "lc01" / "geometry_summary.json")
    if geometry_payload.get("schema_version") != "sample_fg.low_compute_geometry_summary.v1":
        raise TrajectoryProbeError("LC01 geometry-summary schema differs")
    geometry_rows = {
        (row.get("epoch"), row.get("checkpoint_sha256")): row
        for row in geometry_payload.get("rows", [])
        if row.get("materialization_replicate") == 0
        and row.get("lambda") == 0.15
        and row.get("replay_mode") == "stationary_20_epochs"
    }
    source_checkpoints = source.get("checkpoints")
    if not isinstance(source_checkpoints, list):
        raise TrajectoryProbeError("LC01 checkpoint trajectory is malformed")
    clip_sha = _mapping(source_run / "environment.json").get("clip", {}).get("checkpoint_sha256")
    if not isinstance(clip_sha, str):
        raise TrajectoryProbeError("LC01 source CLIP hash is missing")
    selected = []
    missing = []
    for item in source_checkpoints:
        if not isinstance(item, dict) or not isinstance(item.get("epoch"), int):
            raise TrajectoryProbeError("LC01 checkpoint row is malformed")
        epoch = item["epoch"]
        path = Path(item.get("path", ""))
        if not path.is_file():
            missing.append(epoch)
            continue
        checkpoint_sha = sha256_file(path)
        if checkpoint_sha != item.get("sha256"):
            raise TrajectoryProbeError("LC01 checkpoint hash changed")
        bank = primary.get((epoch, checkpoint_sha))
        if bank is None:
            raise TrajectoryProbeError("LC01 primary gradient bank does not match trajectory")
        fidelity = fidelity_rows.get((epoch, checkpoint_sha))
        geometry = geometry_rows.get((epoch, checkpoint_sha))
        if fidelity is None or geometry is None:
            raise TrajectoryProbeError("LC01 fidelity/projection summaries do not match trajectory")
        key = FeatureCacheKey(
            dataset="dtd",
            split="fixed_materialized_train",
            clip_sha256=clip_sha,
            transform_signature="pinned_coop_train_transform_per_sample_seed_v1",
            checkpoint_sha256=checkpoint_sha,
            replicate=0,
        )
        cache_path = lc01_run / "cache" / f"train_{key.digest}.pt"
        features, labels, sample_ids = load_feature_cache(cache_path, expected_key=key)
        if len(features) != 384 or len(labels) != 384 or len(sample_ids) != 384:
            raise TrajectoryProbeError("LC01 feature cache does not contain the fixed DTD source")
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("content_sha256") != bank.get("materialized_feature_cache_sha256"):
            raise TrajectoryProbeError("LC01 feature-cache hash differs from bank index")
        batch_rows = bank.get("batches")
        if not isinstance(batch_rows, list) or len(batch_rows) != 12:
            raise TrajectoryProbeError("LC01 primary gradient bank is not twelve batches")
        selected.append(
            TrajectoryCheckpoint(
                epoch=epoch,
                path=path.resolve(),
                sha256=checkpoint_sha,
                feature_cache=cache_path.resolve(),
                feature_cache_sha256=bank["materialized_feature_cache_sha256"],
                exact_gradient_sha256=bank["exact_gradient_sha256"],
                expected_batch_hashes=tuple(row["gradient_sha256"] for row in batch_rows),
                lc01_fidelity=dict(fidelity),
                lc01_projection_summary=dict(geometry),
            )
        )
    if not 1 <= len(selected) <= 5:
        raise TrajectoryProbeError("LC03 requires between one and five existing LC01 checkpoints")
    if tuple(item.epoch for item in selected) != tuple(
        epoch for epoch in (20, 60, 100, 140, 200) if epoch not in missing
    ):
        raise TrajectoryProbeError("LC03 checkpoint order differs from LC01")
    # LC01 v1 stores hashes/metadata but not GradientState tensors. This check
    # proves why exact reconstruction from the immutable feature cache is needed.
    tensor_bank_candidates = tuple((lc01_run / "cache").glob("*gradient_bank*.pt"))
    reuse_mode = (
        "serialized_lc01_gradient_state_bank"
        if tensor_bank_candidates
        else "verified_reconstruction_from_lc01_materialized_feature_cache"
    )
    eval_key = FeatureCacheKey(
        dataset="dtd",
        split="base_new_test",
        clip_sha256=clip_sha,
        transform_signature="canonical_evaluation_transform",
    )
    eval_cache = lc01_run / "cache" / f"eval_{eval_key.digest}.pt"
    eval_features, eval_labels, _ = load_feature_cache(eval_cache, expected_key=eval_key)
    eval_index = _mapping(lc01_run / "cache" / "eval_feature_index.json")
    eval_payload = torch.load(eval_cache, map_location="cpu", weights_only=False)
    if eval_payload.get("content_sha256") != eval_index.get("sha256"):
        raise TrajectoryProbeError("LC01 evaluation cache hash differs")
    base_samples = int(baseline["base_num_samples"])
    new_samples = int(baseline["new_num_samples"])
    if len(eval_features) != base_samples + new_samples or len(eval_labels) != len(eval_features):
        raise TrajectoryProbeError("LC01 evaluation cache size differs from R2")
    budget = ComputeBudget(
        optimizer_steps=0,
        scheduler_steps=0,
        normal_forward_batches=len(selected) * 12,
        normal_backward_batches=len(selected) * 12,
        image_encoder_forward_batches=0,
        text_encoder_forward_calls=len(selected) * 2 + len(selected) * 24,
    )
    budget.require_read_only()
    if budget.normal_backward_batches > MAX_DISPLACED_BACKWARDS:
        raise TrajectoryProbeError("LC03 displaced-backward budget exceeds 60")
    return TrajectoryPlan(
        lc01_run=lc01_run,
        source_run=source_run,
        checkpoints=tuple(selected),
        missing_checkpoints=tuple(missing),
        source_config_sha256=source["source_config_sha256"],
        source_fingerprint=source["selected_source_fingerprint"],
        clip_sha256=clip_sha,
        eval_cache=eval_cache.resolve(),
        eval_cache_sha256=eval_index["sha256"],
        base_class_count=int(eval_index["base_class_count"]),
        new_class_count=int(eval_index["new_class_count"]),
        base_samples=base_samples,
        new_samples=new_samples,
        baseline_evaluation={
            "base_accuracy_pct": float(baseline["base_accuracy_pct"]),
            "new_accuracy_pct": float(baseline["new_accuracy_pct"]),
            "hm_pct": float(baseline["hm_pct"]),
        },
        lc01_tree_sha256_before=_tree_sha256(lc01_run),
        source_tree_sha256_before=_tree_sha256(source_run),
        budget=budget,
        gradient_bank_reuse=reuse_mode,
    )


def exploration_summary(exploitation: float, exploration: float, epsilon: float = 1e-12) -> dict[str, Any]:
    if any(not math.isfinite(value) for value in (exploitation, exploration, epsilon)) or epsilon <= 0:
        raise TrajectoryProbeError("Taylor sign summary requires finite values and positive epsilon")
    ratio = abs(exploration) / (abs(exploration) + abs(exploitation) + epsilon)
    if abs(exploitation) <= epsilon and abs(exploration) <= epsilon:
        category = "near_zero"
    elif exploitation <= 0 and exploration <= 0:
        category = "both_descent_favoring"
    elif exploitation <= 0 < exploration:
        category = "exploration_opposes_exploitation"
    elif exploration <= 0 < exploitation:
        category = "exploitation_opposes_exploration"
    else:
        # Both positive terms oppose local descent; retain a finite, exhaustive
        # category without inventing a fifth registered label.
        category = "near_zero"
    return {"R_explore": ratio, "sign_category": category}


def _mean(values: Iterable[float | None]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return statistics.fmean(data) if data else None


def _median(values: Iterable[float | None]) -> float | None:
    data = [float(value) for value in values if value is not None]
    return statistics.median(data) if data else None


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        value = (cursor + end - 1) / 2.0 + 1.0
        for position in order[cursor:end]:
            ranks[position] = value
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    a = statistics.fmean(left)
    b = statistics.fmean(right)
    numerator = sum((x - a) * (y - b) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - a) ** 2 for x in left) * sum((y - b) ** 2 for y in right)
    )
    return None if denominator == 0 else numerator / denominator


def descriptive_association(
    rows: Sequence[Mapping[str, Any]], predictor: str, outcome: str
) -> dict[str, Any]:
    pairs = [
        (float(row[predictor]), float(row[outcome]))
        for row in rows
        if isinstance(row.get(predictor), (int, float))
        and isinstance(row.get(outcome), (int, float))
    ]
    left = [item[0] for item in pairs]
    right = [item[1] for item in pairs]
    pearson = _pearson(left, right)
    spearman = _pearson(_rank(left), _rank(right)) if len(left) >= 2 else None
    loo_pearson = []
    loo_spearman = []
    if len(left) >= 3:
        for index in range(len(left)):
            a = left[:index] + left[index + 1 :]
            b = right[:index] + right[index + 1 :]
            p = _pearson(a, b)
            s = _pearson(_rank(a), _rank(b))
            if p is not None:
                loo_pearson.append(p)
            if s is not None:
                loo_spearman.append(s)
    return {
        "predictor": predictor,
        "outcome": outcome,
        "checkpoint_count": len(pairs),
        "pearson": pearson,
        "spearman": spearman,
        "leave_one_out_pearson_range": [min(loo_pearson), max(loo_pearson)] if loo_pearson else None,
        "leave_one_out_spearman_range": [min(loo_spearman), max(loo_spearman)] if loo_spearman else None,
        "interpretation": "descriptive_observational_no_significance_test",
    }


def _evaluate_cached(
    runtime,
    new_model,
    new_index: ParamIndex,
    plan: TrajectoryPlan,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    runtime.model.eval()
    new_model.eval()
    learned = runtime.param_index[0].parameter.detach().clone()
    with torch.no_grad():
        new_index[0].parameter.copy_(learned.to(new_index[0].parameter))
        base_text = _text_features(runtime.model)
        new_text = _text_features(new_model)
        base_features = features[: plan.base_samples].to(base_text)
        new_features = features[plan.base_samples :].to(new_text)
        base_labels = labels[: plan.base_samples].to(base_text.device)
        new_labels = labels[plan.base_samples :].to(new_text.device) - plan.base_class_count
        base_logits = runtime.model.logit_scale.exp() * base_features @ base_text.t()
        new_logits = new_model.logit_scale.exp() * new_features @ new_text.t()
        base_correct = int((base_logits.argmax(1) == base_labels).sum().item())
        new_correct = int((new_logits.argmax(1) == new_labels).sum().item())
    if not torch.equal(runtime.param_index[0].parameter, learned):
        raise TrajectoryProbeError("Base evaluation changed the prompt")
    if not torch.equal(new_index[0].parameter, learned.to(new_index[0].parameter)):
        raise TrajectoryProbeError("New evaluation changed the unified prompt")
    base = 100.0 * base_correct / plan.base_samples
    new = 100.0 * new_correct / plan.new_samples
    hm = 0.0 if base + new == 0 else 2.0 * base * new / (base + new)
    return {
        "base_accuracy_pct": base,
        "new_accuracy_pct": new,
        "hm_pct": hm,
        "base_samples": plan.base_samples,
        "new_samples": plan.new_samples,
        "base_correct": base_correct,
        "new_correct": new_correct,
        "new_retrained": False,
        "same_unified_context": True,
    }


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")


def _materialized(cache: Path, checkpoint: TrajectoryCheckpoint, clip_sha: str):
    key = FeatureCacheKey(
        dataset="dtd",
        split="fixed_materialized_train",
        clip_sha256=clip_sha,
        transform_signature="pinned_coop_train_transform_per_sample_seed_v1",
        checkpoint_sha256=checkpoint.sha256,
        replicate=0,
    )
    features, labels, sample_ids = load_feature_cache(cache, expected_key=key)
    return tuple(
        {"feature": feature, "label": int(label), "sample_id": sample_id}
        for feature, label, sample_id in zip(features, labels, sample_ids)
    )


def _displaced_gradient(runtime, features: torch.Tensor, labels: torch.Tensor, displacement: GradientState) -> GradientState:
    parameters = runtime.param_index.parameters
    before = tuple(entry.parameter.detach().clone() for entry in runtime.param_index)
    with runtime.perturbation.displaced(displacement):
        text = _text_features(runtime.model)
        logits = runtime.model.logit_scale.exp() * features.to(text) @ text.t()
        loss = F.cross_entropy(logits, labels.to(text.device))
        gradients = torch.autograd.grad(
            loss, parameters, create_graph=False, retain_graph=False, allow_unused=False
        )
        result = GradientState.from_tensors(runtime.param_index, gradients)
    if any(not torch.equal(entry.parameter, value) for entry, value in zip(runtime.param_index, before)):
        raise TrajectoryProbeError("LC03 displaced prompt was not restored bitwise")
    if any(parameter.grad is not None for parameter in parameters):
        raise TrajectoryProbeError("LC03 displaced query contaminated live gradients")
    return result


def run_trajectory_probe(
    plan: TrajectoryPlan,
    *,
    scientific_plan,
    output_root: Path,
) -> Path:
    """Execute the explicitly authorized LC03 read-only probe (zero steps)."""

    plan.budget.require_read_only()
    if _tree_sha256(plan.source_run) != plan.source_tree_sha256_before or _tree_sha256(plan.lc01_run) != plan.lc01_tree_sha256_before:
        raise TrajectoryProbeError("LC03 source artifacts changed after planning")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        Path(output_root).resolve()
        / "lc03"
        / "dtd"
        / "shots_16"
        / "sample_ema"
        / "seed_1"
        / timestamp
    )
    run_dir.mkdir(parents=True)
    trajectory_path = run_dir / "checkpoint_trajectory.jsonl"
    taylor_path = run_dir / "taylor_probe.jsonl"
    trajectory_path.write_bytes(b"")
    taylor_path.write_bytes(b"")
    atomic_write_json(
        run_dir / "source.json",
        {
            "schema_version": LC03_SCHEMA_VERSION,
            "lc01_run": str(plan.lc01_run),
            "lc01_tree_sha256": plan.lc01_tree_sha256_before,
            "source_run": str(plan.source_run),
            "source_tree_sha256": plan.source_tree_sha256_before,
            "checkpoints": [item.as_dict() for item in plan.checkpoints],
            "missing_checkpoints": list(plan.missing_checkpoints),
        },
    )
    atomic_write_json(run_dir / "compute_budget.json", plan.budget.as_dict())

    runtime = _build_runtime(scientific_plan, run_dir / "runtime")
    new_cfg = build_scientific_cfg(
        dataset="dtd",
        seed=1,
        data_root=scientific_plan.data_root,
        output_dir=run_dir / "runtime" / "new",
        config_path=scientific_plan.config_path,
        class_subsample="new",
        shots=16,
        epochs=200,
    )
    set_random_seed(1)
    new_trainer = build_coop_trainer(new_cfg, scientific_plan.clip_cache)
    new_model = unwrap_model(new_trainer.model)
    audit_prompt_only_training(new_model, new_trainer.optim)
    new_index = ParamIndex.from_model(new_model)
    eval_key = FeatureCacheKey(
        dataset="dtd",
        split="base_new_test",
        clip_sha256=plan.clip_sha256,
        transform_signature="canonical_evaluation_transform",
    )
    eval_features, eval_labels, _ = load_feature_cache(plan.eval_cache, expected_key=eval_key)
    frozen_base = hash_frozen_parameters(runtime.model)
    frozen_new = hash_frozen_parameters(new_model)
    trajectory_rows = []
    displaced_count = 0
    regeneration_backwards = 0
    final_parity = False
    started = time.perf_counter()
    with (
        TransitionGuard(runtime.trainer.optim, runtime.trainer.sched),
        TransitionGuard(new_trainer.optim, new_trainer.sched),
    ):
        for selected in plan.checkpoints:
            probe = load_probe_checkpoint(plan.source_run, selected.path)
            if probe.checkpoint_sha256 != selected.sha256:
                raise TrajectoryProbeError("LC03 source checkpoint changed")
            probe.install_prompt(runtime.param_index)
            probe.install_prompt(new_index)
            prompt_before = tuple(entry.parameter.detach().clone() for entry in runtime.param_index)
            evaluation = _evaluate_cached(
                runtime, new_model, new_index, plan, eval_features, eval_labels
            )
            materialized = _materialized(
                selected.feature_cache, selected, plan.clip_sha256
            )
            bank = _gradient_bank(runtime, materialized, 0)
            regeneration_backwards += len(bank.batches)
            observed_hashes = tuple(item.gradient_sha256 for item in bank.batches)
            if observed_hashes != selected.expected_batch_hashes:
                raise TrajectoryProbeError("LC01 reconstructed batch-gradient hashes differ")
            if gradient_sha256(bank.exact) != selected.exact_gradient_sha256:
                raise TrajectoryProbeError("LC01 reconstructed exact-gradient hash differs")
            # LC01 computed and serialized these scalar comparisons from its
            # owned CPU GradientStates.  Validate in that same arithmetic
            # domain before making device-local copies for the displaced
            # probe; CUDA reductions can differ in the last few digits even
            # when every serialized tensor hash is identical.
            actual_cpu = probe.actual_ema(runtime.param_index)
            fidelity = gradient_metrics(actual_cpu, bank.exact)
            for key in ("cosine", "angle_degrees", "relative_l2", "norm_ratio", "log_norm_ratio"):
                expected = selected.lc01_fidelity.get(key)
                observed = fidelity.get(key)
                if expected is None or observed is None:
                    if expected is not observed:
                        raise TrajectoryProbeError("LC01 fidelity degeneracy differs on reconstruction")
                elif abs(float(expected) - float(observed)) > 1e-6:
                    raise TrajectoryProbeError(
                        "LC01 fidelity metric differs on reconstruction: "
                        f"{key} expected={expected!r} observed={observed!r}"
                    )
            actual = _device_state(actual_cpu, runtime.param_index)
            exact = _device_state(bank.exact, runtime.param_index)
            per_checkpoint = []
            geometry_rows = []
            for batch, start in zip(bank.batches, range(0, len(materialized), 32)):
                if displaced_count >= MAX_DISPLACED_BACKWARDS:
                    raise TrajectoryProbeError("LC03 displaced-backward permit exhausted")
                records = materialized[start : start + batch.sample_count]
                features = torch.stack([row["feature"] for row in records]).to(
                    runtime.param_index[0].parameter.device
                )
                labels = torch.tensor(
                    [row["label"] for row in records],
                    device=features.device,
                    dtype=torch.long,
                )
                gradient = _device_state(batch.gradient, runtime.param_index)
                projection = project_batch_gradient(gradient, actual)
                displacement = safe_unit(gradient).unit.scale(0.05).subtract(
                    projection.batch_component.scale(0.0015)
                )
                perturbed = _displaced_gradient(
                    runtime, features, labels, displacement
                )
                displaced_count += 1
                diagnostics = compute_gradient_diagnostics(
                    batch_gradient=gradient,
                    active_global_estimate=actual,
                    projection=projection,
                    perturbed_gradient=perturbed,
                    alpha=0.0015,
                    exact_full_gradient=exact,
                ).as_dict()
                normalized = exploration_summary(
                    diagnostics["taylor/exploitation_term"],
                    diagnostics["taylor/exploration_term"],
                )
                geometry = projection_displacement_metrics(
                    gradient, actual, exact, rho=0.05, alpha=0.0015
                )
                row = {
                    "schema_version": TAYLOR_SCHEMA_VERSION,
                    "checkpoint_sha256": selected.sha256,
                    "epoch": selected.epoch,
                    "training_fraction": selected.epoch / 200.0,
                    "batch_index": batch.batch_index,
                    "batch_gradient_sha256": batch.gradient_sha256,
                    "sample_count": batch.sample_count,
                    "diagnostics_source": "sample_fg.diagnostics.compute_gradient_diagnostics",
                    **diagnostics,
                    **normalized,
                    "gB_exact_fidelity": geometry["gB"],
                    "displacement_fidelity": geometry["delta"],
                    "prompt_restored_bitwise": True,
                }
                _append(taylor_path, row)
                per_checkpoint.append(row)
                geometry_rows.append(geometry)
            if any(not torch.equal(entry.parameter, value) for entry, value in zip(runtime.param_index, prompt_before)):
                raise TrajectoryProbeError("LC03 changed checkpoint prompt")
            g_b_cosines = [item["gB"]["cosine"] for item in geometry_rows]
            g_b_l2 = [item["gB"]["relative_l2"] for item in geometry_rows]
            g_b_norm = [item["gB"]["norm_ratio"] for item in geometry_rows]
            delta_cosines = [item["delta"]["cosine"] for item in geometry_rows]
            row = {
                "schema_version": TRAJECTORY_SCHEMA_VERSION,
                "checkpoint_sha256": selected.sha256,
                "epoch": selected.epoch,
                "training_fraction": selected.epoch / 200.0,
                "materialization_replicate": 0,
                "ema_exact_cosine": fidelity["cosine"],
                "ema_exact_angle_degrees": fidelity["angle_degrees"],
                "ema_exact_relative_l2": fidelity["relative_l2"],
                "ema_exact_norm_ratio": fidelity["norm_ratio"],
                "ema_exact_log_norm_ratio": fidelity["log_norm_ratio"],
                "lc01_fidelity_reused": True,
                "lc01_stationary_paper_lambda_projection": {
                    key: selected.lc01_projection_summary.get(key)
                    for key in (
                        "exact_cosine", "relative_l2", "norm_ratio",
                        "gB_exact_cosine_mean", "gB_exact_cosine_min",
                        "delta_exact_cosine_mean",
                    )
                },
                "gB_exact_cosine_mean": _mean(g_b_cosines),
                "gB_exact_cosine_median": _median(g_b_cosines),
                "gB_exact_cosine_min": min(value for value in g_b_cosines if value is not None),
                "gB_exact_relative_l2_mean": _mean(g_b_l2),
                "gB_exact_norm_ratio_mean": _mean(g_b_norm),
                "displacement_exact_cosine_mean": _mean(delta_cosines),
                "exploration_fraction_mean": _mean(item["R_explore"] for item in per_checkpoint),
                "taylor_exploitation_mean": _mean(item["taylor/exploitation_term"] for item in per_checkpoint),
                "taylor_exploration_mean": _mean(item["taylor/exploration_term"] for item in per_checkpoint),
                "taylor_joint_alignment_mean": _mean(item["taylor/joint_alignment_term"] for item in per_checkpoint),
                **evaluation,
            }
            if selected.epoch == 200:
                final_parity = all(
                    abs(row[key] - plan.baseline_evaluation[key]) <= PARITY_TOLERANCE_PCT
                    for key in ("base_accuracy_pct", "new_accuracy_pct", "hm_pct")
                )
                if not final_parity:
                    raise TrajectoryProbeError("Final LC03 Base/New/HM evaluation differs from R2")
            _append(trajectory_path, row)
            trajectory_rows.append(row)
            verify_source_immutable(probe)
    if any(item.epoch == 200 for item in plan.checkpoints) and not final_parity:
        raise TrajectoryProbeError("LC03 final parity was not demonstrated")
    if displaced_count != len(plan.checkpoints) * 12 or displaced_count > MAX_DISPLACED_BACKWARDS:
        raise TrajectoryProbeError("LC03 displaced-backward accounting differs")
    if hash_frozen_parameters(runtime.model) != frozen_base or hash_frozen_parameters(new_model) != frozen_new:
        raise TrajectoryProbeError("Frozen CLIP changed during LC03")
    if _tree_sha256(plan.source_run) != plan.source_tree_sha256_before or _tree_sha256(plan.lc01_run) != plan.lc01_tree_sha256_before:
        raise TrajectoryProbeError("LC03 changed source artifacts")
    associations = [
        descriptive_association(trajectory_rows, predictor, "new_accuracy_pct")
        for predictor in (
            "ema_exact_cosine",
            "ema_exact_relative_l2",
            "gB_exact_cosine_mean",
            "exploration_fraction_mean",
        )
    ]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "completed",
        "title": "Fidelity -> Geometry -> Generalization Trajectory",
        "interpretation": "observational_descriptive_lc02_is_causal_experiment",
        "checkpoint_count": len(trajectory_rows),
        "missing_checkpoints": list(plan.missing_checkpoints),
        "final_r2_parity": {
            "required": True,
            "final_checkpoint_available": any(item.epoch == 200 for item in plan.checkpoints),
            "passed": final_parity,
            "tolerance_pct": PARITY_TOLERANCE_PCT,
        },
        "associations": associations,
        "significance_testing": False,
        "p_values_generated": False,
        "safety": {
            "optimizer_steps_executed": 0,
            "scheduler_steps_executed": 0,
            "additional_displaced_backward_batches": displaced_count,
            "gradient_bank_regeneration_backward_batches": regeneration_backwards,
            "gradient_bank_reuse": plan.gradient_bank_reuse,
            "new_class_retraining": False,
            "source_artifacts_changed": False,
            "frozen_clip_changed": False,
        },
        "elapsed_s": time.perf_counter() - started,
        "artifacts": {
            "checkpoint_trajectory": "checkpoint_trajectory.jsonl",
            "taylor_probe": "taylor_probe.jsonl",
            "source": "source.json",
            "compute_budget": "compute_budget.json",
        },
    }
    atomic_write_json(run_dir / "trajectory_summary.json", summary)
    if final_parity:
        render_trajectory_artifacts(run_dir)
    return run_dir


def render_trajectory_artifacts(run_dir: Path) -> tuple[Path, ...]:
    """Regenerate all LC03 tables/plots strictly from saved scalar artifacts."""

    root = Path(run_dir).resolve(strict=True)
    rows = _jsonl(root / "checkpoint_trajectory.jsonl")
    summary = _mapping(root / "trajectory_summary.json")
    parity = summary.get("final_r2_parity", {})
    if parity.get("required") and parity.get("passed") is not True:
        raise TrajectoryProbeError("Trajectory plots require demonstrated final R2 parity")
    if not rows:
        raise TrajectoryProbeError("Trajectory artifact is empty")
    table = root / "trajectory_table.csv"
    fields = [
        "epoch", "training_fraction", "ema_exact_cosine",
        "ema_exact_relative_l2", "gB_exact_cosine_mean",
        "exploration_fraction_mean", "base_accuracy_pct", "new_accuracy_pct", "hm_pct",
    ]
    with table.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "sample_fg_lc03_v1"
        import matplotlib.pyplot as plt
    except Exception as error:
        raise TrajectoryProbeError("matplotlib is required to render LC03 plots") from error
    plots = (
        ("training_fraction_vs_ema_exact_cosine", "training_fraction", ("ema_exact_cosine",), "Training fraction", "EMA / exact cosine"),
        ("training_fraction_vs_new_hm", "training_fraction", ("new_accuracy_pct", "hm_pct"), "Training fraction", "Accuracy (%)"),
        ("ema_exact_cosine_vs_new_accuracy", "ema_exact_cosine", ("new_accuracy_pct",), "EMA / exact cosine", "New accuracy (%)"),
        ("gB_exact_fidelity_vs_new_accuracy", "gB_exact_cosine_mean", ("new_accuracy_pct",), "Mean g_B exact cosine", "New accuracy (%)"),
        ("exploration_fraction_vs_new_accuracy", "exploration_fraction_mean", ("new_accuracy_pct",), "Mean exploration fraction", "New accuracy (%)"),
    )
    outputs = [table]
    for stem, x_key, y_keys, x_label, y_label in plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        for y_key in y_keys:
            axis.plot(
                [row[x_key] for row in rows],
                [row[y_key] for row in rows],
                marker="o",
                label=y_key.replace("_", " "),
            )
        for row in rows:
            axis.annotate(str(row["epoch"]), (row[x_key], row[y_keys[0]]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.25)
        if len(y_keys) > 1:
            axis.legend(frameon=False)
        figure.tight_layout()
        for suffix in ("png", "svg"):
            destination = root / f"{stem}.{suffix}"
            figure.savefig(destination, dpi=160, metadata={"Date": None})
            outputs.append(destination)
        plt.close(figure)
    return tuple(outputs)
