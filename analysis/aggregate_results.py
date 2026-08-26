"""Aggregate validated run artifacts without parsing human console output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.environment import ENVIRONMENT_SCHEMA_VERSION
from sample_fg.results import (
    METRICS_SCHEMA_VERSION,
    RESOLVED_CONFIG_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    atomic_write_json,
    load_jsonl,
    resolve_config,
)


AGGREGATION_SCHEMA_VERSION = "sample_fg.aggregation.v1"
AGGREGATION_REPORT_SCHEMA_VERSION = "sample_fg.aggregation_report.v1"
METHOD_ORDER = {
    "coop:none": 0,
    "sam:none": 1,
    "sample:ema": 2,
    "sample:exact": 3,
}
DIAGNOSTIC_FIELDS = (
    "grad/exact_reference_available",
    "grad/batch_norm",
    "grad/global_estimate_norm",
    "grad/exact_full_norm",
    "grad/batch_component_norm",
    "grad/reference_batch_component_norm",
    "grad/perturbed_norm",
    "grad/xi",
    "grad/sigma",
    "grad/projection_coefficient",
    "grad/batch_gradient_degenerate",
    "grad/global_direction_degenerate",
    "grad/exact_full_direction_degenerate",
    "grad/batch_component_degenerate",
    "grad/reference_batch_component_degenerate",
    "grad/perturbed_gradient_degenerate",
    "grad/global_estimate_exact_cosine",
    "grad/global_estimate_exact_norm_ratio",
    "grad/global_estimate_exact_log_norm_ratio",
    "grad/global_estimate_exact_relative_l2",
    "grad/batch_component_estimator_cosine",
    "grad/batch_component_exact_cosine",
    "grad/reference_batch_component_exact_cosine",
    "grad/perturbed_gradient_estimator_cosine",
    "grad/perturbed_gradient_exact_cosine",
    "grad/perturbed_gradient_batch_component_cosine",
    "grad/perturbed_gradient_batch_cosine",
    "taylor/exploitation_dot_unweighted",
    "taylor/exploitation_term",
    "taylor/exploration_dot_unweighted",
    "taylor/exploration_term",
    "taylor/joint_alignment_term",
    "raw/dot_batch_global",
    "raw/dot_batch_exact",
    "raw/dot_global_exact",
    "raw/dot_batch_component_global",
    "raw/dot_batch_component_exact",
    "raw/dot_perturbed_batch",
    "raw/dot_perturbed_global",
    "raw/dot_perturbed_exact",
    "raw/dot_perturbed_batch_component",
)
EFFICIENCY_FIELDS = (
    "train_total_s",
    "full_gradient_total_s",
    "diagnostic_overhead_s",
    "eval_base_s",
    "eval_new_s",
    "evaluation_total_s",
    "exact_gradient_total_s",
    "total_wall_time_s",
    "peak_cuda_allocated_bytes",
    "peak_cuda_reserved_bytes",
)
COMPUTE_FIELDS = (
    "optimizer_steps",
    "current_forward_batches",
    "current_backward_batches",
    "displaced_forward_batches",
    "displaced_backward_batches",
    "full_gradient_sweeps",
    "current_samples",
    "displaced_samples",
    "full_gradient_samples",
    "full_gradient_forward_microbatches",
    "full_gradient_backward_microbatches",
    "exact_sweeps",
    "exact_sweep_samples",
    "exact_sweep_forward_batches",
    "exact_sweep_backward_batches",
    "optimization_exact_queries",
    "diagnostic_only_exact_queries",
    "reused_exact_queries",
)


class AggregationError(RuntimeError):
    """Raised when artifacts cannot be compared without ambiguity."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _get(mapping: Mapping[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _method_key(method: str, estimator: str | None, periodic_k: int | None) -> str:
    if method == "sample" and estimator == "periodic":
        return f"sample:periodic_k{periodic_k}"
    return f"{method}:{estimator or 'none'}"


def _method_sort(row: Mapping[str, Any]) -> tuple[Any, ...]:
    key = _method_key(row["method"], row.get("estimator_mode"), row.get("periodic_k_steps"))
    order = METHOD_ORDER.get(key, 4 if key.startswith("sample:periodic_k") else 99)
    return (
        str(row.get("experiment_id")), str(row.get("dataset")), int(row.get("shots", 0)),
        order, int(row.get("periodic_k_steps") or 0), str(row.get("precision")),
        int(row.get("seed", 0)), str(row.get("run_id")),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AggregationError(f"Invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise AggregationError(f"JSON artifact is not an object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise AggregationError(f"Invalid YAML artifact: {path}") from error
    if not isinstance(value, dict):
        raise AggregationError(f"YAML artifact is not an object: {path}")
    return value


def _protocol_signature(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    payload = {
        "data": {
            key: _get(config, "data", key)
            for key in (
                "dataset", "shots", "split_policy", "train_batch_size", "test_batch_size",
                "num_workers", "preserve_upstream_drop_last", "augmentation_policy",
                "selected_source_fingerprint", "selected_count",
            )
        },
        "model": {
            key: _get(config, "model", key)
            for key in (
                "backbone", "prompt_learner", "effective_n_ctx", "ctx_init",
                "class_specific_context", "class_token_position", "freeze_clip",
                "checkpoint_sha256",
            )
        },
        "optim": {
            key: _get(config, "optim", key)
            for key in (
                "name", "lr", "weight_decay", "momentum", "nesterov", "max_epoch",
                "scheduler", "warmup_epoch", "warmup_type", "warmup_cons_lr",
                "scheduler_step_unit",
            )
        },
        "runtime": {
            "precision": _get(config, "runtime", "precision"),
            "gradient_state_dtype": _get(config, "runtime", "gradient_state_dtype"),
        },
        "split_sha256": _get(manifest, "official_split", "sha256"),
        "source_count": _get(manifest, "complete_selected_source", "count"),
    }
    return _sha(payload), payload


def _validate_hm(evaluation: Mapping[str, Any], run_dir: Path) -> tuple[float, float, float]:
    try:
        base = float(evaluation["base_accuracy_pct"])
        new = float(evaluation["new_accuracy_pct"])
        logged = float(evaluation["hm_pct"])
    except (KeyError, TypeError, ValueError) as error:
        raise AggregationError(f"Run lacks numeric Base/New/HM: {run_dir}") from error
    if not all(math.isfinite(value) for value in (base, new, logged)):
        raise AggregationError(f"Run has nonfinite Base/New/HM: {run_dir}")
    expected = 0.0 if base + new == 0 else 2.0 * base * new / (base + new)
    if not math.isclose(logged, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AggregationError(f"Logged HM differs from recomputed per-run HM: {run_dir}")
    return base, new, expected


def load_run(run_dir: Path) -> dict[str, Any]:
    """Load and validate one complete Task-20 run directory."""

    run_dir = Path(run_dir).resolve()
    required = (
        "config.yaml", "environment.json", "data_manifest.json", "metrics.jsonl",
        "gradient_diagnostics.jsonl", "summary.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise AggregationError(f"Incomplete run {run_dir}: missing {missing}")
    config = _load_yaml(run_dir / "config.yaml")
    environment = _load_json(run_dir / "environment.json")
    manifest = _load_json(run_dir / "data_manifest.json")
    summary = _load_json(run_dir / "summary.json")
    try:
        metrics = load_jsonl(run_dir / "metrics.jsonl")
        diagnostics = load_jsonl(run_dir / "gradient_diagnostics.jsonl")
    except Exception as error:
        raise AggregationError(f"Malformed JSONL in {run_dir}") from error
    if config.get("schema_version") != RESOLVED_CONFIG_SCHEMA_VERSION:
        raise AggregationError(f"Unsupported config schema in {run_dir}")
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise AggregationError(f"Unsupported summary schema in {run_dir}")
    if environment.get("schema_version") != ENVIRONMENT_SCHEMA_VERSION:
        raise AggregationError(f"Unsupported environment schema in {run_dir}")
    if manifest.get("schema_version") != "sample_fg.data_manifest.v1":
        raise AggregationError(f"Unsupported data manifest schema in {run_dir}")
    if any(record.get("schema_version") != METRICS_SCHEMA_VERSION for record in metrics):
        raise AggregationError(f"Unsupported metric schema in {run_dir}")
    if any(not isinstance(record.get("schema_version"), str) for record in diagnostics):
        raise AggregationError(f"Unsupported diagnostic schema in {run_dir}")
    if resolve_config(config).get("config_sha256") != config.get("config_sha256"):
        raise AggregationError(f"Config hash mismatch in {run_dir}")
    identity = summary.get("run_identity")
    if not isinstance(identity, dict) or identity.get("run_id") != _get(config, "run", "run_id"):
        raise AggregationError(f"Run identity mismatch in {run_dir}")
    if summary.get("status") != "completed":
        raise AggregationError(f"Run status is not completed: {run_dir}")
    smoke = bool(summary.get("smoke"))
    allowed = summary.get("allow_scientific_summary") is True
    if smoke != bool(_get(config, "run", "smoke")):
        raise AggregationError(f"Smoke flag mismatch in {run_dir}")
    if bool(_get(config, "smoke", "allow_scientific_summary")) != allowed:
        raise AggregationError(f"Scientific gate mismatch in {run_dir}")
    if smoke and allowed:
        raise AggregationError(f"Smoke run permits scientific summary: {run_dir}")
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, dict):
        raise AggregationError(f"Missing evaluation in {run_dir}")
    base, new, hm = _validate_hm(evaluation, run_dir)
    efficiency = summary.get("efficiency")
    if not isinstance(efficiency, dict):
        raise AggregationError(f"Missing efficiency in {run_dir}")
    counts = efficiency.get("compute_counts", {})
    if not isinstance(counts, dict):
        raise AggregationError(f"Invalid compute counters in {run_dir}")
    method = str(_get(config, "method", "name"))
    estimator = _get(config, "estimator", "mode")
    estimator = None if estimator in {None, "none"} else str(estimator)
    periodic_k = _get(config, "estimator", "refresh_k_steps") if estimator == "periodic" else None
    protocol_hash, protocol = _protocol_signature(config, manifest)
    row: dict[str, Any] = {
        "run_id": identity["run_id"], "experiment_id": identity.get("experiment_id"),
        "run_dir": str(run_dir), "config_sha256": config["config_sha256"],
        "dataset": str(_get(config, "data", "dataset")),
        "shots": int(_get(config, "data", "shots")),
        "seed": int(_get(config, "data", "seed")), "method": method,
        "estimator_mode": estimator, "periodic_k_steps": periodic_k,
        "method_key": _method_key(method, estimator, periodic_k),
        "precision": str(_get(config, "runtime", "precision")),
        "smoke": smoke, "allow_scientific_summary": allowed,
        "base_accuracy_pct": base, "new_accuracy_pct": new, "hm_pct": hm,
        "protocol_signature": protocol_hash,
        "source_fingerprint": _get(config, "data", "selected_source_fingerprint"),
        "split_sha256": _get(manifest, "official_split", "sha256"),
        "gpu_name": _get(environment, "gpu", "name"),
        "artifact_reused": False,
        "source_run_id": identity["run_id"],
        "reuse_source_experiment_id": None,
    }
    diagnostic_summary = summary.get("estimator_diagnostics", {})
    if not isinstance(diagnostic_summary, dict):
        raise AggregationError(f"Invalid estimator_diagnostics in {run_dir}")
    for field in (
        "num_exact_reference_points",
        "global_estimate_exact_cosine_mean",
        "global_estimate_exact_relative_l2_mean",
        "global_estimate_exact_log_norm_ratio_mean",
        "batch_component_exact_abs_cosine_mean",
        "perturbed_gradient_exact_abs_cosine_mean",
        "global_estimate_exact_norm_ratio_mean",
        "batch_component_estimator_abs_cosine_mean",
        "reference_batch_component_exact_abs_cosine_mean",
        "perturbed_gradient_estimator_abs_cosine_mean",
        "perturbed_gradient_batch_component_abs_cosine_mean",
        "perturbed_gradient_batch_abs_cosine_mean",
        "taylor_exploitation_mean",
        "taylor_exploration_mean",
        "taylor_joint_mean",
    ):
        row[field] = diagnostic_summary.get(field)
    for field in EFFICIENCY_FIELDS:
        value = efficiency.get(field)
        row[field] = value
    for field in COMPUTE_FIELDS:
        row[field] = int(counts.get(field, 0))
    if row["exact_gradient_total_s"] is None:
        row["exact_gradient_total_s"] = row["full_gradient_total_s"]
    if row["exact_sweeps"] == 0:
        row["exact_sweeps"] = row["full_gradient_sweeps"]
    if row["exact_sweep_samples"] == 0:
        row["exact_sweep_samples"] = row["full_gradient_samples"]
    if row["exact_sweep_forward_batches"] == 0:
        row["exact_sweep_forward_batches"] = row[
            "full_gradient_forward_microbatches"
        ]
    if row["exact_sweep_backward_batches"] == 0:
        row["exact_sweep_backward_batches"] = row[
            "full_gradient_backward_microbatches"
        ]
    return {
        "row": row, "config": config, "environment": environment,
        "manifest": manifest, "summary": summary, "metrics": metrics,
        "diagnostics": diagnostics, "protocol": protocol,
    }


def discover_runs(input_root: Path, *, strict: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(input_root).resolve(strict=True)
    loaded: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    summaries = sorted(root.rglob("summary.json"), key=lambda path: path.as_posix())
    candidate_dirs = {path.parent for path in summaries}
    # Also make incomplete run-like directories visible to strict discovery.
    candidate_dirs.update(path.parent for path in root.rglob("config.yaml"))
    for run_dir in sorted(candidate_dirs, key=lambda path: path.as_posix()):
        try:
            loaded.append(load_run(run_dir))
        except AggregationError as error:
            record = {"run_dir": str(run_dir), "reason": str(error)}
            excluded.append(record)
            if strict:
                raise
    if not candidate_dirs:
        raise AggregationError(f"No structured run artifacts found below {root}")
    return loaded, excluded


def _canonicalize_attempts(runs: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_ids: set[str] = set()
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        row = run["row"]
        if row["run_id"] in run_ids:
            raise AggregationError(f"Duplicate run_id: {row['run_id']}")
        run_ids.add(row["run_id"])
        key = (
            row["experiment_id"], row["dataset"], row["shots"], row["method"],
            row["estimator_mode"], row["periodic_k_steps"], row["precision"], row["seed"],
        )
        groups[key].append(run)
    canonical: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda value: tuple("" if item is None else str(item) for item in value)):
        attempts = sorted(groups[key], key=lambda item: item["row"]["run_id"])
        hashes = {item["row"]["config_sha256"] for item in attempts}
        if len(hashes) != 1:
            raise AggregationError(f"Conflicting configs for planned cell {key}")
        canonical.append(attempts[0])
        for item in attempts[1:]:
            duplicates.append(
                {
                    "run_id": item["row"]["run_id"],
                    "canonical_run_id": attempts[0]["row"]["run_id"],
                    "reason": "same-cell same-config rerun; selected earliest run_id, never performance",
                }
            )
    return canonical, duplicates


def _reuse_r2_ema_for_e2(
    runs: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add analysis-only E2 aliases without copying or editing R2 artifacts."""

    expected = {(dataset, seed) for dataset in ("dtd", "eurosat") for seed in (1, 2, 3)}
    existing_e2 = {
        (run["row"]["dataset"], run["row"]["seed"])
        for run in runs
        if run["row"]["experiment_id"] == "E2"
        and run["row"]["method_key"] == "sample:ema"
        and run["row"]["shots"] == 16
    }
    if existing_e2:
        raise AggregationError(
            "E2 contains newly trained EMA cells; protocol requires immutable R2 reuse"
        )
    reusable = {
        (run["row"]["dataset"], run["row"]["seed"]): run
        for run in runs
        if run["row"]["experiment_id"] == "R2"
        and run["row"]["method_key"] == "sample:ema"
        and run["row"]["shots"] == 16
        and run["row"]["dataset"] in {"dtd", "eurosat"}
        and run["row"]["seed"] in {1, 2, 3}
    }
    missing = sorted(expected - set(reusable))
    if missing:
        raise AggregationError(
            f"Task 28 R2 EMA reuse is incomplete; missing cells: {missing}"
        )
    aliases: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for key in sorted(expected):
        source = reusable[key]
        alias = dict(source)
        row = dict(source["row"])
        source_run_id = row["run_id"]
        row.update(
            {
                "run_id": f"reuse_E2_{source_run_id}",
                "experiment_id": "E2",
                "artifact_reused": True,
                "source_run_id": source_run_id,
                "reuse_source_experiment_id": "R2",
            }
        )
        alias["row"] = row
        aliases.append(alias)
        records.append(
            {
                "experiment_id": "E2",
                "dataset": row["dataset"],
                "shots": row["shots"],
                "seed": row["seed"],
                "method_key": row["method_key"],
                "source_experiment_id": "R2",
                "source_run_id": source_run_id,
                "source_run_dir": row["run_dir"],
                "config_sha256": row["config_sha256"],
            }
        )
    return list(runs) + aliases, records


def _validate_comparison_compatibility(runs: Sequence[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        row = run["row"]
        key = (row["experiment_id"], row["dataset"], row["shots"], row["precision"], row["seed"])
        groups[key].append(run)
    for key, values in groups.items():
        signatures = {value["row"]["protocol_signature"] for value in values}
        if len(signatures) != 1:
            detail = {value["row"]["run_id"]: value["protocol"] for value in values}
            raise AggregationError(f"Incompatible fixed protocol in comparison cell {key}: {detail}")
        sample_values = [value for value in values if value["row"]["method"] == "sample"]
        rho_alpha = {
            (_get(value["config"], "method", "rho"), _get(value["config"], "method", "alpha"))
            for value in sample_values
        }
        if len(rho_alpha) > 1:
            raise AggregationError(f"SAMPLe rho/alpha mismatch in comparison cell {key}")
        ema_lambdas = {
            _get(value["config"], "method", "ema_lambda")
            for value in sample_values
            if value["row"]["estimator_mode"] in {"ema", "periodic"}
        }
        if len(ema_lambdas) > 1:
            raise AggregationError(f"SAMPLe EMA lambda mismatch in comparison cell {key}")


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "n": len(values),
    }


def _summary_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["experiment_id"], row["dataset"], row["shots"], row["method"],
            row["estimator_mode"], row["periodic_k_steps"], row["precision"],
        )
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, values in groups.items():
        values = sorted(values, key=lambda row: row["seed"])
        item = {
            "experiment_id": key[0], "dataset": key[1], "shots": key[2],
            "method": key[3], "estimator_mode": key[4],
            "periodic_k_steps": key[5], "precision": key[6],
            "method_key": values[0]["method_key"], "seeds": [row["seed"] for row in values],
            "artifact_reused": all(row.get("artifact_reused") is True for row in values),
            "source_run_ids": [row.get("source_run_id") for row in values],
            "reuse_source_experiment_ids": sorted(
                {
                    row.get("reuse_source_experiment_id")
                    for row in values
                    if row.get("reuse_source_experiment_id") is not None
                }
            ),
        }
        for label, field in (
            ("base", "base_accuracy_pct"), ("new", "new_accuracy_pct"),
            ("hm", "hm_pct"), ("train_time", "train_total_s"),
            ("peak_allocated", "peak_cuda_allocated_bytes"),
            ("estimator_exact_cosine", "global_estimate_exact_cosine_mean"),
            ("estimator_exact_relative_l2", "global_estimate_exact_relative_l2_mean"),
            ("estimator_exact_log_norm_ratio", "global_estimate_exact_log_norm_ratio_mean"),
            ("batch_component_exact_abs_cosine", "batch_component_exact_abs_cosine_mean"),
            ("perturbed_gradient_exact_abs_cosine", "perturbed_gradient_exact_abs_cosine_mean"),
            ("global_estimate_exact_norm_ratio", "global_estimate_exact_norm_ratio_mean"),
            ("batch_component_estimator_abs_cosine", "batch_component_estimator_abs_cosine_mean"),
            ("reference_batch_component_exact_abs_cosine", "reference_batch_component_exact_abs_cosine_mean"),
            ("perturbed_gradient_estimator_abs_cosine", "perturbed_gradient_estimator_abs_cosine_mean"),
            ("perturbed_gradient_batch_component_abs_cosine", "perturbed_gradient_batch_component_abs_cosine_mean"),
            ("perturbed_gradient_batch_abs_cosine", "perturbed_gradient_batch_abs_cosine_mean"),
            ("taylor_exploitation", "taylor_exploitation_mean"),
            ("taylor_exploration", "taylor_exploration_mean"),
            ("taylor_joint", "taylor_joint_mean"),
        ):
            stats = _stats([float(row[field]) for row in values if row.get(field) is not None])
            item[f"{label}_mean"] = stats["mean"]
            item[f"{label}_std"] = stats["std"]
            item[f"{label}_n"] = stats["n"]
        output.append(item)
    return sorted(output, key=_method_sort)


def _paired_rows(rows: Sequence[dict[str, Any]], baseline_method_key: str) -> list[dict[str, Any]]:
    lookup = {
        (row["experiment_id"], row["dataset"], row["shots"], row["precision"], row["method_key"], row["seed"]): row
        for row in rows
    }
    candidates: dict[tuple[Any, ...], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        if row["method_key"] == baseline_method_key:
            continue
        baseline = lookup.get((
            row["experiment_id"], row["dataset"], row["shots"], row["precision"], baseline_method_key, row["seed"]
        ))
        if baseline is None:
            continue
        key = (
            row["experiment_id"], row["dataset"], row["shots"], row["precision"],
            row["method"], row["estimator_mode"], row["periodic_k_steps"], row["method_key"],
        )
        candidates[key].append((row, baseline))
    output: list[dict[str, Any]] = []
    for key, pairs in candidates.items():
        pairs.sort(key=lambda pair: pair[0]["seed"])
        item = {
            "experiment_id": key[0], "dataset": key[1], "shots": key[2],
            "precision": key[3], "method": key[4], "estimator_mode": key[5],
            "periodic_k_steps": key[6], "method_key": key[7],
            "baseline_method_key": baseline_method_key,
            "direction": "candidate_minus_baseline",
            "paired_seeds": [pair[0]["seed"] for pair in pairs],
        }
        for label, field in (("base", "base_accuracy_pct"), ("new", "new_accuracy_pct"), ("hm", "hm_pct")):
            deltas = [candidate[field] - baseline[field] for candidate, baseline in pairs]
            stats = _stats(deltas)
            item[f"{label}_delta_mean"] = stats["mean"]
            item[f"{label}_delta_std"] = stats["std"]
            item[f"{label}_paired_n"] = stats["n"]
        output.append(item)
    return sorted(output, key=_method_sort)


def _diagnostic_rows(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        identity = run["row"]
        metrics_by_step = {
            record.get("optimizer_step"): record
            for record in run["metrics"]
            if record.get("event_type") == "train_step"
        }
        for record in run["diagnostics"]:
            metrics = record.get("metrics", {})
            step_metric = metrics_by_step.get(record.get("optimizer_step"), {})
            row = {
                key: identity[key]
                for key in (
                    "run_id", "experiment_id", "dataset", "shots", "seed", "method",
                    "estimator_mode", "periodic_k_steps", "precision", "smoke",
                    "artifact_reused", "source_run_id",
                    "reuse_source_experiment_id",
                )
            }
            row.update(
                {
                    "optimizer_step": record.get("optimizer_step"),
                    "epoch": record.get("epoch"), "batch_index": record.get("batch_index"),
                    "estimator_refreshed": record.get("estimator_refresh"),
                    "estimator_age_steps": step_metric.get("estimator/age_steps"),
                    "estimator_last_refresh_step": step_metric.get(
                        "estimator/last_refresh_step"
                    ),
                    "estimator_active_source": step_metric.get(
                        "estimator/active_source"
                    ),
                    "exact_reference_source": record.get("exact_reference_source"),
                    "exact_service_query_issued": record.get("exact_service_query_issued"),
                    "exact_reference_reused": record.get("exact_reference_reused"),
                    "exact_reference_purpose": _get(
                        record, "exact_reference_auxiliary_seed", "purpose"
                    ),
                    "full_gradient_elapsed_s": _get(record, "full_gradient", "elapsed_s"),
                    "full_gradient_sample_count": _get(record, "full_gradient", "sample_count"),
                }
            )
            for field in DIAGNOSTIC_FIELDS:
                row[field] = metrics.get(field)
            output.append(row)
    return sorted(output, key=lambda row: (_method_sort(row), int(row.get("optimizer_step") or 0)))


def _efficiency_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "run_id", "experiment_id", "dataset", "shots", "seed", "method",
        "estimator_mode", "periodic_k_steps", "precision", "smoke", "gpu_name",
        "artifact_reused", "source_run_id", "reuse_source_experiment_id",
    ) + EFFICIENCY_FIELDS + COMPUTE_FIELDS
    baseline = {
        (row["experiment_id"], row["dataset"], row["shots"], row["precision"], row["seed"]): row
        for row in rows
        if row["method_key"] == "sample:ema"
    }
    output: list[dict[str, Any]] = []
    for source in rows:
        row = {key: source.get(key) for key in keys}
        match = baseline.get((source["experiment_id"], source["dataset"], source["shots"], source["precision"], source["seed"]))
        if match is None:
            row["train_time_overhead_vs_sample_ema_pct"] = None
            row["overhead_pair_status"] = "no_matched_sample_ema"
        elif match.get("gpu_name") != source.get("gpu_name"):
            row["train_time_overhead_vs_sample_ema_pct"] = None
            row["overhead_pair_status"] = "hardware_mismatch"
        elif not match.get("train_total_s"):
            row["train_time_overhead_vs_sample_ema_pct"] = None
            row["overhead_pair_status"] = "zero_or_missing_baseline_time"
        else:
            row["train_time_overhead_vs_sample_ema_pct"] = 100.0 * (
                float(source["train_total_s"]) / float(match["train_total_s"]) - 1.0
            )
            row["overhead_pair_status"] = "matched_same_hardware"
        output.append(row)
    return sorted(output, key=_method_sort)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _write_dataset(output_dir: Path, stem: str, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {"schema_version": AGGREGATION_SCHEMA_VERSION, "rows": list(rows)}
    atomic_write_json(output_dir / f"{stem}.json", payload)
    _write_csv(output_dir / f"{stem}.csv", rows)


def aggregate(
    input_root: Path,
    output_dir: Path,
    *,
    mode: str = "scientific",
    strict: bool = True,
    baseline_method_key: str = "coop:none",
    reuse_r2_ema_for_e2: bool = False,
) -> dict[str, Any]:
    if mode not in {"scientific", "smoke"}:
        raise AggregationError("mode must be scientific or smoke")
    loaded, invalid = discover_runs(input_root, strict=strict)
    canonical, duplicates = _canonicalize_attempts(loaded)
    if mode == "scientific":
        eligible = [run for run in canonical if not run["row"]["smoke"] and run["row"]["allow_scientific_summary"]]
        gated = [run for run in canonical if run not in eligible]
        gate_reason = "excluded_by_scientific_gate"
    else:
        eligible = [run for run in canonical if run["row"]["smoke"] and not run["row"]["allow_scientific_summary"]]
        gated = [run for run in canonical if run not in eligible]
        gate_reason = "excluded_by_smoke_validation_gate"
    reuse_records: list[dict[str, Any]] = []
    if reuse_r2_ema_for_e2:
        eligible, reuse_records = _reuse_r2_ema_for_e2(eligible)
    _validate_comparison_compatibility(eligible)
    rows = sorted([run["row"] for run in eligible], key=_method_sort)
    summaries = _summary_rows(rows)
    paired = _paired_rows(rows, baseline_method_key)
    diagnostics = _diagnostic_rows(eligible)
    efficiency = _efficiency_rows(rows)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, values in (
        ("runs_long", rows), ("summary_by_cell", summaries),
        ("paired_differences", paired), ("diagnostics_long", diagnostics),
        ("efficiency", efficiency),
    ):
        _write_dataset(output_dir, stem, values)
    report = {
        "schema_version": AGGREGATION_REPORT_SCHEMA_VERSION,
        "mode": mode, "input_root": str(Path(input_root).resolve()),
        "scientific_default": mode == "scientific",
        "discovered_runs": len(loaded), "canonical_completed_runs": len(canonical),
        "eligible_runs": len(rows), "scientific_rows": sum(not row["smoke"] for row in rows),
        "smoke_rows": sum(row["smoke"] for row in rows),
        "duplicate_attempts": duplicates,
        "reused_artifacts": reuse_records,
        "excluded_runs": invalid + [
            {"run_id": run["row"]["run_id"], "reason": gate_reason} for run in gated
        ],
        "outputs": {
            stem: {"json": f"{stem}.json", "csv": f"{stem}.csv", "rows": len(values)}
            for stem, values in (
                ("runs_long", rows), ("summary_by_cell", summaries),
                ("paired_differences", paired), ("diagnostics_long", diagnostics),
                ("efficiency", efficiency),
            )
        },
        "statistics": {
            "mean": "arithmetic seed mean", "std": "sample standard deviation ddof=1",
            "n1_std": None, "hm": "recomputed per run before seed aggregation",
            "paired_direction": "candidate_minus_baseline", "baseline_method_key": baseline_method_key,
        },
        "deterministic_order": "manifest method order, then K, precision, seed, run_id",
        "console_log_parsing": False,
    }
    atomic_write_json(output_dir / "aggregation_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--baseline-method-key", default="coop:none")
    parser.add_argument("--reuse-r2-ema-for-e2", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = aggregate(
        Path(arguments.input_root), Path(arguments.output_dir), mode=arguments.mode,
        strict=not arguments.allow_invalid,
        baseline_method_key=arguments.baseline_method_key,
        reuse_r2_ema_for_e2=arguments.reuse_r2_ema_for_e2,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
