"""Render deterministic research tables from Task-23 aggregate artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


AGGREGATION_SCHEMA_VERSION = "sample_fg.aggregation.v1"
AGGREGATION_REPORT_SCHEMA_VERSION = "sample_fg.aggregation_report.v1"
TABLE_MANIFEST_SCHEMA_VERSION = "sample_fg.analysis_tables.v1"
SMOKE_LABEL = "SMOKE VALIDATION — NON-SCIENTIFIC"
METHOD_ORDER = {
    "coop:none": 0,
    "sam:none": 1,
    "sample:ema": 2,
    "sample:exact": 3,
}


class AnalysisInputError(RuntimeError):
    """Raised when aggregate inputs cannot support an honest rendering."""


def method_key(row: Mapping[str, Any]) -> str:
    method = str(row.get("method"))
    estimator = row.get("estimator_mode") or "none"
    if method == "sample" and estimator == "periodic":
        return f"sample:periodic_k{row.get('periodic_k_steps')}"
    return f"{method}:{estimator}"


def method_label(row: Mapping[str, Any]) -> str:
    key = method_key(row)
    if key == "coop:none":
        label = "CoOp"
    elif key == "sam:none":
        label = "SAM"
    elif key == "sample:ema":
        label = "SAMPLe-EMA"
    elif key == "sample:exact":
        label = "SAMPLe-Exact"
    elif key.startswith("sample:periodic_k"):
        label = f"SAMPLe-Periodic (K={row.get('periodic_k_steps')})"
    else:
        label = key
    precision = row.get("precision")
    if precision and precision != "fp32":
        label += f" [{precision}]"
    return label


def method_sort(row: Mapping[str, Any]) -> tuple[Any, ...]:
    key = method_key(row)
    order = METHOD_ORDER.get(key, 4 if key.startswith("sample:periodic_k") else 99)
    return (
        str(row.get("experiment_id")), str(row.get("dataset")),
        int(row.get("shots") or 0), order, int(row.get("periodic_k_steps") or 0),
        str(row.get("precision") or ""), int(row.get("seed") or 0),
        str(row.get("run_id") or ""),
    )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AnalysisInputError(f"Invalid JSON input: {path}") from error
    if not isinstance(value, dict):
        raise AnalysisInputError(f"JSON input must be an object: {path}")
    return value


def load_rows(input_dir: Path, stem: str) -> list[dict[str, Any]]:
    path = input_dir / f"{stem}.json"
    payload = _load_object(path)
    if payload.get("schema_version") != AGGREGATION_SCHEMA_VERSION:
        raise AnalysisInputError(f"Unsupported aggregate schema: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise AnalysisInputError(f"Aggregate rows must be a list of objects: {path}")
    return rows


def load_aggregate_bundle(input_dir: Path, mode: str) -> dict[str, Any]:
    input_dir = Path(input_dir).resolve(strict=True)
    report = _load_object(input_dir / "aggregation_report.json")
    if report.get("schema_version") != AGGREGATION_REPORT_SCHEMA_VERSION:
        raise AnalysisInputError("Unsupported aggregation report schema")
    if mode not in {"scientific", "smoke"}:
        raise AnalysisInputError("mode must be scientific or smoke")
    if report.get("mode") != mode:
        raise AnalysisInputError(
            f"Rendering mode {mode!r} does not match aggregation mode {report.get('mode')!r}"
        )
    runs = load_rows(input_dir, "runs_long")
    if not runs:
        raise AnalysisInputError(f"No eligible {mode} runs in aggregate input")
    if mode == "scientific":
        if report.get("smoke_rows") != 0 or any(bool(row.get("smoke")) for row in runs):
            raise AnalysisInputError("Scientific rendering rejects smoke rows")
    else:
        if any(not bool(row.get("smoke")) for row in runs):
            raise AnalysisInputError("Smoke rendering rejects scientific rows")
    return {
        "input_dir": input_dir,
        "report": report,
        "runs": runs,
        "summary": load_rows(input_dir, "summary_by_cell"),
        "paired": load_rows(input_dir, "paired_differences"),
        "diagnostics": load_rows(input_dir, "diagnostics_long"),
        "efficiency": load_rows(input_dir, "efficiency"),
    }


def _number(value: Any) -> str:
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value):
        raise AnalysisInputError("Nonfinite value reached table rendering")
    return f"{value:.3f}"


def format_stat(mean: Any, std: Any, n: int) -> str:
    if mean is None or n == 0:
        return "— (n=0)"
    if n == 1:
        return f"{_number(mean)} (n=1; std unavailable)"
    if std is None:
        raise AnalysisInputError("Multi-seed statistic is missing sample standard deviation")
    return f"{_number(mean)} ± {_number(std)} (n={n})"


def _metric_columns(row: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label in ("base", "new", "hm"):
        mean, std, n = row.get(f"{label}_mean"), row.get(f"{label}_std"), int(row.get(f"{label}_n") or 0)
        output[f"{label}_mean"] = mean
        output[f"{label}_std"] = std
        output[f"{label}_n"] = n
        output[f"{label}_mean_std"] = format_stat(mean, std, n)
    return output


def _summary_table(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in sorted(rows, key=method_sort):
        row = {
            "experiment_id": source.get("experiment_id"),
            "dataset": source.get("dataset"), "shots": source.get("shots"),
            "method": method_label(source), "method_key": method_key(source),
            "estimator": source.get("estimator_mode"), "K": source.get("periodic_k_steps"),
            "precision": source.get("precision"),
        }
        row.update(_metric_columns(source))
        output.append(row)
    return output


def _stats(values: Sequence[float]) -> tuple[float | None, float | None, int]:
    if not values:
        return None, None, 0
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else None, len(values)


def _efficiency_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("experiment_id"), row.get("dataset"), row.get("shots"), method_key(row), row.get("precision"))].append(row)
    output: list[dict[str, Any]] = []
    for _, values in groups.items():
        source = sorted(values, key=lambda row: int(row.get("seed") or 0))[0]
        item = {
            "experiment_id": source.get("experiment_id"), "dataset": source.get("dataset"),
            "shots": source.get("shots"), "method": method_label(source),
            "method_key": method_key(source), "precision": source.get("precision"),
            "hardware": source.get("gpu_name"),
        }
        for label, field in (
            ("train_time_s", "train_total_s"),
            ("overhead_vs_ema_pct", "train_time_overhead_vs_sample_ema_pct"),
            ("full_gradient_time_s", "exact_gradient_total_s"),
            ("total_wall_time_s", "total_wall_time_s"),
            ("peak_allocated_bytes", "peak_cuda_allocated_bytes"),
            ("peak_reserved_bytes", "peak_cuda_reserved_bytes"),
            ("exact_sweeps", "exact_sweeps"),
            ("exact_samples", "exact_sweep_samples"),
        ):
            numbers = [float(row[field]) for row in values if row.get(field) is not None]
            mean, std, n = _stats(numbers)
            item[f"{label}_mean"] = mean
            item[f"{label}_std"] = std
            item[f"{label}_n"] = n
            item[f"{label}_mean_std"] = format_stat(mean, std, n)
        statuses = sorted({str(row.get("overhead_pair_status")) for row in values})
        item["overhead_pair_status"] = ",".join(statuses)
        output.append(item)
    return sorted(output, key=method_sort)


def _periodic_k_table(
    summary: Sequence[dict[str, Any]],
    efficiency: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render the pre-confirmatory K evidence without accuracy columns."""

    efficiency_lookup = {
        (
            row.get("experiment_id"),
            row.get("dataset"),
            row.get("shots"),
            method_key(row),
            row.get("precision"),
        ): row
        for row in _efficiency_table(efficiency)
    }
    output: list[dict[str, Any]] = []
    for source in sorted(summary, key=method_sort):
        if (
            source.get("experiment_id") != "E1"
            or source.get("estimator_mode") != "periodic"
        ):
            continue
        evidence = efficiency_lookup.get(
            (
                source.get("experiment_id"),
                source.get("dataset"),
                source.get("shots"),
                method_key(source),
                source.get("precision"),
            ),
            {},
        )
        row = {
            "experiment_id": source.get("experiment_id"),
            "dataset": source.get("dataset"),
            "shots": source.get("shots"),
            "K": source.get("periodic_k_steps"),
            "precision": source.get("precision"),
        }
        for metric in (
            "estimator_exact_cosine",
            "estimator_exact_relative_l2",
            "estimator_exact_log_norm_ratio",
        ):
            row[f"{metric}_mean_std"] = format_stat(
                source.get(f"{metric}_mean"),
                source.get(f"{metric}_std"),
                int(source.get(f"{metric}_n") or 0),
            )
        for metric in (
            "train_time_s_mean_std",
            "full_gradient_time_s_mean_std",
            "total_wall_time_s_mean_std",
            "exact_sweeps_mean_std",
            "exact_samples_mean_std",
        ):
            row[metric] = evidence.get(metric, "— (n=0)")
        output.append(row)
    return output


def _mechanism_table(summary: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "batch_component_estimator_abs_cosine",
        "batch_component_exact_abs_cosine",
        "reference_batch_component_exact_abs_cosine",
        "perturbed_gradient_estimator_abs_cosine",
        "perturbed_gradient_exact_abs_cosine",
        "perturbed_gradient_batch_component_abs_cosine",
        "perturbed_gradient_batch_abs_cosine",
        "taylor_exploitation",
        "taylor_exploration",
        "taylor_joint",
    )
    output: list[dict[str, Any]] = []
    for source in sorted(summary, key=method_sort):
        if source.get("method") != "sample":
            continue
        row = {
            "experiment_id": source.get("experiment_id"),
            "dataset": source.get("dataset"),
            "shots": source.get("shots"),
            "method": method_label(source),
            "precision": source.get("precision"),
        }
        for metric in metrics:
            row[f"{metric}_mean_std"] = format_stat(
                source.get(f"{metric}_mean"),
                source.get(f"{metric}_std"),
                int(source.get(f"{metric}_n") or 0),
            )
        output.append(row)
    return output


def _raw_seed_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in sorted(rows, key=method_sort):
        output.append(
            {
                "experiment_id": source.get("experiment_id"), "dataset": source.get("dataset"),
                "shots": source.get("shots"), "method": method_label(source),
                "method_key": method_key(source), "precision": source.get("precision"),
                "seed": source.get("seed"), "Base": source.get("base_accuracy_pct"),
                "New": source.get("new_accuracy_pct"), "HM": source.get("hm_pct"),
                "run_id": source.get("run_id"),
                "artifact_reused": source.get("artifact_reused"),
                "source_run_id": source.get("source_run_id"),
                "reuse_source_experiment_id": source.get(
                    "reuse_source_experiment_id"
                ),
            }
        )
    return output


def _paired_table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for source in sorted(rows, key=method_sort):
        item = {
            "experiment_id": source.get("experiment_id"), "dataset": source.get("dataset"),
            "shots": source.get("shots"), "candidate": method_label(source),
            "candidate_method_key": method_key(source),
            "baseline": source.get("baseline_method_key"),
            "direction": source.get("direction"),
            "paired_seeds": source.get("paired_seeds"),
        }
        for metric in ("base", "new", "hm"):
            mean, std, n = (
                source.get(f"{metric}_delta_mean"), source.get(f"{metric}_delta_std"),
                int(source.get(f"{metric}_paired_n") or 0),
            )
            item[f"{metric}_delta_mean"] = mean
            item[f"{metric}_delta_std"] = std
            item[f"{metric}_paired_n"] = n
            item[f"{metric}_delta_mean_std"] = format_stat(mean, std, n)
        output.append(item)
    return output


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _escape_md(value: Any) -> str:
    return str("" if value is None else value).replace("|", "\\|").replace("\n", " ")


def _escape_tex(value: Any) -> str:
    text = str("" if value is None else value)
    for source, target in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("±", "$\\pm$")):
        text = text.replace(source, target)
    return text


def _write_human(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, latex: bool) -> None:
    if latex:
        lines = ["\\begin{tabular}{" + "l" * len(fields) + "}", " \\toprule", " & ".join(_escape_tex(field) for field in fields) + " \\\\", " \\midrule"]
        lines.extend(" & ".join(_escape_tex(row.get(field)) for field in fields) + " \\\\" for row in rows)
        lines.extend([" \\bottomrule", "\\end{tabular}"])
    else:
        lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
        lines.extend("| " + " | ".join(_escape_md(row.get(field)) for field in fields) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _emit_table(output_dir: Path, name: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], label: str | None) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decorated = list(rows)
    if label:
        decorated = [{"analysis_status": label, **row} for row in decorated]
        fields = ("analysis_status", *fields)
    _write_csv(output_dir / f"{name}.csv", decorated, fields)
    _write_human(output_dir / f"{name}.md", decorated, fields, latex=False)
    _write_human(output_dir / f"{name}.tex", decorated, fields, latex=True)
    return {"rows": len(decorated), "files": [f"{name}.csv", f"{name}.md", f"{name}.tex"]}


def make_tables(input_dir: Path, output_dir: Path, *, mode: str = "scientific") -> dict[str, Any]:
    bundle = load_aggregate_bundle(Path(input_dir), mode)
    output_dir = Path(output_dir).resolve()
    label = SMOKE_LABEL if mode == "smoke" else None
    summary = bundle["summary"]
    tables: dict[str, Any] = {}
    common = ("experiment_id", "dataset", "shots", "method", "precision", "base_mean_std", "base_n", "new_mean_std", "new_n", "hm_mean_std", "hm_n")
    tables["reproduction_summary"] = _emit_table(
        output_dir, "reproduction_summary",
        _summary_table(row for row in summary if method_key(row) in {"coop:none", "sam:none", "sample:ema"}), common, label,
    )
    extension_fields = (
        *common, "estimator", "K", "estimator_exact_cosine_mean_std",
        "estimator_exact_relative_l2_mean_std", "estimator_exact_log_norm_ratio_mean_std",
        "batch_component_exact_abs_cosine_mean_std",
        "perturbed_gradient_exact_abs_cosine_mean_std", "train_time_s_mean_std",
        "overhead_vs_ema_pct_mean_std", "full_gradient_time_s_mean_std",
        "total_wall_time_s_mean_std",
        "peak_allocated_bytes_mean_std", "peak_reserved_bytes_mean_std",
        "exact_sweeps_mean_std", "exact_samples_mean_std",
    )
    extension_rows = _summary_table(row for row in summary if str(row.get("method")) == "sample")
    efficiency_lookup = {
        (row["experiment_id"], row["dataset"], row["shots"], row["method_key"], row["precision"]): row
        for row in _efficiency_table(bundle["efficiency"])
    }
    for rendered, source in zip(extension_rows, sorted((row for row in summary if str(row.get("method")) == "sample"), key=method_sort)):
        for prefix in (
            "estimator_exact_cosine", "estimator_exact_relative_l2",
            "estimator_exact_log_norm_ratio", "batch_component_exact_abs_cosine",
            "perturbed_gradient_exact_abs_cosine",
        ):
            rendered[f"{prefix}_mean_std"] = format_stat(source.get(f"{prefix}_mean"), source.get(f"{prefix}_std"), int(source.get(f"{prefix}_n") or 0))
        efficiency = efficiency_lookup.get(
            (source.get("experiment_id"), source.get("dataset"), source.get("shots"), method_key(source), source.get("precision")),
            {},
        )
        for field in (
            "train_time_s_mean_std", "overhead_vs_ema_pct_mean_std",
            "full_gradient_time_s_mean_std", "peak_allocated_bytes_mean_std",
            "total_wall_time_s_mean_std",
            "peak_reserved_bytes_mean_std", "exact_sweeps_mean_std",
            "exact_samples_mean_std",
        ):
            rendered[field] = efficiency.get(field, "— (n=0)")
    tables["extension_summary"] = _emit_table(output_dir, "extension_summary", extension_rows, extension_fields, label)
    tables["few_shot_summary"] = _emit_table(output_dir, "few_shot_summary", _summary_table(summary), common, label)
    raw_fields = ("experiment_id", "dataset", "shots", "method", "precision", "seed", "Base", "New", "HM", "run_id", "artifact_reused", "source_run_id", "reuse_source_experiment_id")
    tables["raw_seed_results"] = _emit_table(output_dir, "raw_seed_results", _raw_seed_table(bundle["runs"]), raw_fields, label)
    efficiency_fields = ("experiment_id", "dataset", "shots", "method", "precision", "hardware", "train_time_s_mean_std", "total_wall_time_s_mean_std", "overhead_vs_ema_pct_mean_std", "overhead_pair_status", "peak_allocated_bytes_mean_std", "peak_reserved_bytes_mean_std", "exact_sweeps_mean_std", "exact_samples_mean_std")
    tables["efficiency_summary"] = _emit_table(output_dir, "efficiency_summary", _efficiency_table(bundle["efficiency"]), efficiency_fields, label)
    periodic_fields = (
        "experiment_id", "dataset", "shots", "K", "precision",
        "estimator_exact_cosine_mean_std",
        "estimator_exact_relative_l2_mean_std",
        "estimator_exact_log_norm_ratio_mean_std",
        "train_time_s_mean_std", "full_gradient_time_s_mean_std",
        "total_wall_time_s_mean_std", "exact_sweeps_mean_std",
        "exact_samples_mean_std",
    )
    tables["periodic_k_selection"] = _emit_table(
        output_dir,
        "periodic_k_selection",
        _periodic_k_table(summary, bundle["efficiency"]),
        periodic_fields,
        label,
    )
    mechanism_fields = (
        "experiment_id", "dataset", "shots", "method", "precision",
        "batch_component_estimator_abs_cosine_mean_std",
        "batch_component_exact_abs_cosine_mean_std",
        "reference_batch_component_exact_abs_cosine_mean_std",
        "perturbed_gradient_estimator_abs_cosine_mean_std",
        "perturbed_gradient_exact_abs_cosine_mean_std",
        "perturbed_gradient_batch_component_abs_cosine_mean_std",
        "perturbed_gradient_batch_abs_cosine_mean_std",
        "taylor_exploitation_mean_std", "taylor_exploration_mean_std",
        "taylor_joint_mean_std",
    )
    tables["mechanism_summary"] = _emit_table(
        output_dir,
        "mechanism_summary",
        _mechanism_table(summary),
        mechanism_fields,
        label,
    )
    paired_fields = ("experiment_id", "dataset", "shots", "candidate", "baseline", "direction", "paired_seeds", "base_delta_mean_std", "new_delta_mean_std", "hm_delta_mean_std")
    tables["paired_differences"] = _emit_table(output_dir, "paired_differences", _paired_table(bundle["paired"]), paired_fields, label)
    manifest = {
        "schema_version": TABLE_MANIFEST_SCHEMA_VERSION,
        "mode": mode, "scientific": mode == "scientific", "smoke": mode == "smoke",
        "display_label": label, "source_aggregation": str(bundle["input_dir"]),
        "statistics": {"center": "arithmetic mean", "error": "sample standard deviation (ddof=1)", "n1": "std unavailable", "paired_direction": "candidate - baseline", "significance_stars": False},
        "ordering": "experiment manifest method order; never performance",
        "tables": tables,
    }
    (output_dir / "analysis_tables_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(make_tables(Path(arguments.input_dir), Path(arguments.output_dir), mode=arguments.mode), indent=2, sort_keys=True, ensure_ascii=False))
