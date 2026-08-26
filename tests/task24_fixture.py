"""Deterministic three-seed Task-23 aggregate fixture for Task-24 rendering."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


METHODS = (
    ("coop", None, None),
    ("sam", None, None),
    ("sample", "ema", None),
    ("sample", "exact", None),
    ("sample", "periodic", 4),
)


def _key(method: str, estimator: str | None, k: int | None) -> str:
    return f"sample:periodic_k{k}" if estimator == "periodic" else f"{method}:{estimator or 'none'}"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_fixture(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, summaries, paired, diagnostics, efficiency = [], [], [], [], []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for dataset_index, dataset in enumerate(("dtd", "eurosat")):
        for shots in (4, 8, 16):
            for method_index, (method, estimator, k) in enumerate(METHODS):
                group = []
                for seed in (1, 2, 3):
                    base = 60.0 + dataset_index * 2 + shots * 0.25 + method_index + seed
                    new = 40.0 + dataset_index * 3 + shots * 0.20 + method_index * 1.5 + seed * 0.5
                    hm = 2 * base * new / (base + new)
                    fidelity_error = None if method != "sample" else max(0.0, 0.60 - method_index * 0.12 + seed * 0.01)
                    row = {
                        "run_id": f"F0__{dataset}__{shots}__{_key(method, estimator, k)}__seed{seed}",
                        "experiment_id": "F0", "dataset": dataset, "shots": shots,
                        "seed": seed, "method": method, "estimator_mode": estimator,
                        "periodic_k_steps": k, "method_key": _key(method, estimator, k),
                        "precision": "fp32", "smoke": False,
                        "allow_scientific_summary": True, "base_accuracy_pct": base,
                        "new_accuracy_pct": new, "hm_pct": hm,
                        "global_estimate_exact_relative_l2_mean": fidelity_error,
                        "global_estimate_exact_cosine_mean": None if method != "sample" else 1.0 - fidelity_error,
                        "global_estimate_exact_log_norm_ratio_mean": None if method != "sample" else fidelity_error * 0.2,
                        "batch_component_exact_abs_cosine_mean": None if method != "sample" else 0.02 + seed * 0.001,
                        "perturbed_gradient_exact_abs_cosine_mean": None if method != "sample" else 0.03 + seed * 0.001,
                        "global_estimate_exact_norm_ratio_mean": None if method != "sample" else 1.2,
                        "batch_component_estimator_abs_cosine_mean": None if method != "sample" else 0.01,
                        "reference_batch_component_exact_abs_cosine_mean": None if method != "sample" else 0.005,
                        "perturbed_gradient_estimator_abs_cosine_mean": None if method != "sample" else 0.2,
                        "perturbed_gradient_batch_component_abs_cosine_mean": None if method != "sample" else 0.3,
                        "perturbed_gradient_batch_abs_cosine_mean": None if method != "sample" else 0.4,
                        "taylor_exploitation_mean": None if method != "sample" else -0.2,
                        "taylor_exploration_mean": None if method != "sample" else 0.19,
                        "taylor_joint_mean": None if method != "sample" else -0.01,
                        "train_total_s": 100.0 + 12 * method_index + seed,
                        "peak_cuda_allocated_bytes": 1_000_000_000 + 10_000_000 * method_index,
                    }
                    runs.append(row); group.append(row)
                    overhead = None if method_index < 2 else 100.0 * (row["train_total_s"] / (100.0 + 12 * 2 + seed) - 1.0)
                    efficiency.append(
                        {
                            **{key: row.get(key) for key in ("run_id", "experiment_id", "dataset", "shots", "seed", "method", "estimator_mode", "periodic_k_steps", "precision", "smoke")},
                            "gpu_name": "synthetic-gpu", "train_total_s": row["train_total_s"],
                            "full_gradient_total_s": 0.0 if method_index < 3 else 10.0 * (method_index - 2),
                            "exact_gradient_total_s": 0.0 if method_index < 3 else 10.0 * (method_index - 2),
                            "total_wall_time_s": row["train_total_s"] + 4.0,
                            "diagnostic_overhead_s": 1.0 if method == "sample" else 0.0,
                            "eval_base_s": 2.0, "eval_new_s": 2.0, "evaluation_total_s": 4.0,
                            "peak_cuda_allocated_bytes": row["peak_cuda_allocated_bytes"],
                            "peak_cuda_reserved_bytes": row["peak_cuda_allocated_bytes"] + 100_000_000,
                            "optimizer_steps": 10, "full_gradient_sweeps": 0 if method_index < 3 else 10,
                            "full_gradient_samples": 0 if method_index < 3 else 10 * shots,
                            "exact_sweeps": 0 if method_index < 3 else 10,
                            "exact_sweep_samples": 0 if method_index < 3 else 10 * shots,
                            "train_time_overhead_vs_sample_ema_pct": overhead,
                            "overhead_pair_status": "no_matched_sample_ema" if overhead is None else "matched_same_hardware",
                        }
                    )
                    if method == "sample":
                        for step in (0, 1, 2):
                            refreshed = estimator == "periodic" and step % 2 == 0
                            diagnostics.append(
                                {
                                    **{key: row.get(key) for key in ("run_id", "experiment_id", "dataset", "shots", "seed", "method", "estimator_mode", "periodic_k_steps", "precision", "smoke")},
                                    "optimizer_step": step, "epoch": 0, "batch_index": step,
                                    "estimator_refreshed": refreshed,
                                    "estimator_age_steps": step % 2 if estimator == "periodic" else None,
                                    "grad/global_estimate_exact_cosine": 0.45 + method_index * 0.1 + step * 0.03,
                                    "grad/global_estimate_exact_norm_ratio": 1.4 - method_index * 0.1 + step * 0.02,
                                    "grad/global_estimate_exact_log_norm_ratio": 0.2 - method_index * 0.03 + step * 0.01,
                                    "grad/global_estimate_exact_relative_l2": 0.7 - method_index * 0.1 + step * 0.02,
                                    "grad/batch_component_exact_cosine": -0.20 + step * 0.1,
                                    "grad/batch_component_estimator_cosine": 0.01,
                                    "grad/reference_batch_component_exact_cosine": 0.005,
                                    "grad/perturbed_gradient_estimator_cosine": 0.2,
                                    "grad/perturbed_gradient_exact_cosine": 0.12 - step * 0.03,
                                    "grad/perturbed_gradient_batch_component_cosine": 0.3 + step * 0.04,
                                    "grad/perturbed_gradient_batch_cosine": 0.4,
                                    "taylor/exploitation_term": -0.2, "taylor/exploration_term": 0.19,
                                    "taylor/joint_alignment_term": -0.01,
                                }
                            )
                groups[(dataset, shots, method, estimator, k)] = group
    for (dataset, shots, method, estimator, k), values in groups.items():
        item = {
            "experiment_id": "F0", "dataset": dataset, "shots": shots,
            "method": method, "estimator_mode": estimator, "periodic_k_steps": k,
            "precision": "fp32", "method_key": _key(method, estimator, k),
            "seeds": [1, 2, 3],
        }
        for label, field in (("base", "base_accuracy_pct"), ("new", "new_accuracy_pct"), ("hm", "hm_pct"), ("train_time", "train_total_s"), ("peak_allocated", "peak_cuda_allocated_bytes"), ("estimator_exact_cosine", "global_estimate_exact_cosine_mean"), ("estimator_exact_relative_l2", "global_estimate_exact_relative_l2_mean"), ("estimator_exact_log_norm_ratio", "global_estimate_exact_log_norm_ratio_mean"), ("batch_component_exact_abs_cosine", "batch_component_exact_abs_cosine_mean"), ("perturbed_gradient_exact_abs_cosine", "perturbed_gradient_exact_abs_cosine_mean"), ("global_estimate_exact_norm_ratio", "global_estimate_exact_norm_ratio_mean"), ("batch_component_estimator_abs_cosine", "batch_component_estimator_abs_cosine_mean"), ("reference_batch_component_exact_abs_cosine", "reference_batch_component_exact_abs_cosine_mean"), ("perturbed_gradient_estimator_abs_cosine", "perturbed_gradient_estimator_abs_cosine_mean"), ("perturbed_gradient_batch_component_abs_cosine", "perturbed_gradient_batch_component_abs_cosine_mean"), ("perturbed_gradient_batch_abs_cosine", "perturbed_gradient_batch_abs_cosine_mean"), ("taylor_exploitation", "taylor_exploitation_mean"), ("taylor_exploration", "taylor_exploration_mean"), ("taylor_joint", "taylor_joint_mean")):
            numbers = [float(row[field]) for row in values if row.get(field) is not None]
            item[f"{label}_mean"] = statistics.fmean(numbers) if numbers else None
            item[f"{label}_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
            item[f"{label}_n"] = len(numbers)
        summaries.append(item)
    lookup = {(row["dataset"], row["shots"], row["method_key"], row["seed"]): row for row in runs}
    for row in runs:
        if row["method_key"] == "coop:none" or row["seed"] != 1:
            continue
        values = [lookup[(row["dataset"], row["shots"], row["method_key"], seed)] for seed in (1, 2, 3)]
        baselines = [lookup[(row["dataset"], row["shots"], "coop:none", seed)] for seed in (1, 2, 3)]
        item = {
            "experiment_id": "F0", "dataset": row["dataset"], "shots": row["shots"],
            "precision": "fp32", "method": row["method"], "estimator_mode": row["estimator_mode"],
            "periodic_k_steps": row["periodic_k_steps"], "method_key": row["method_key"],
            "baseline_method_key": "coop:none", "direction": "candidate_minus_baseline",
            "paired_seeds": [1, 2, 3],
        }
        for label, field in (("base", "base_accuracy_pct"), ("new", "new_accuracy_pct"), ("hm", "hm_pct")):
            deltas = [candidate[field] - baseline[field] for candidate, baseline in zip(values, baselines)]
            item[f"{label}_delta_mean"] = statistics.fmean(deltas)
            item[f"{label}_delta_std"] = statistics.stdev(deltas)
            item[f"{label}_paired_n"] = 3
        paired.append(item)
    for stem, rows in (("runs_long", runs), ("summary_by_cell", summaries), ("paired_differences", paired), ("diagnostics_long", diagnostics), ("efficiency", efficiency)):
        _write(output_dir / f"{stem}.json", {"schema_version": "sample_fg.aggregation.v1", "rows": rows})
    report = {
        "schema_version": "sample_fg.aggregation_report.v1", "mode": "scientific",
        "scientific_default": True, "eligible_runs": len(runs),
        "scientific_rows": len(runs), "smoke_rows": 0,
        "fixture": "synthetic known-value three-seed analysis fixture",
        "statistics": {"std": "sample standard deviation ddof=1", "paired_direction": "candidate_minus_baseline"},
    }
    _write(output_dir / "aggregation_report.json", report)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(write_fixture(Path(args.output_dir)))
