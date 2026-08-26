"""Regenerate the LC01+LC04 primary table from saved scalar artifacts only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


class LowComputeAnalysisError(RuntimeError):
    pass


TABLE_FIELDS = (
    "checkpoint_epoch", "checkpoint_sha256", "lambda", "effective_sample_size",
    "ema_exact_cosine", "ema_exact_relative_l2", "ema_exact_norm_ratio",
    "gB_exact_ema_cosine", "delta_exact_ema_cosine", "order_sensitivity_sd",
    "actual_checkpoint_parameter_cosine",
    "text_embedding_agreement", "logit_agreement", "gradient_query_gpu_time_s",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LowComputeAnalysisError(f"Artifact root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise LowComputeAnalysisError(f"Non-object JSONL row {line_number}: {path}")
        rows.append(value)
    return rows


def build_primary_rows(run_dir: Path) -> list[dict[str, Any]]:
    root = Path(run_dir).resolve(strict=True)
    replay = _load(root / "lc01" / "replay_summary.json").get("rows", [])
    geometry = _load(root / "lc01" / "geometry_summary.json").get("rows", [])
    function = _jsonl(root / "lc04" / "function_space_fidelity.jsonl")
    accounting = _load(root / "lc01" / "compute_accounting.json")
    geometry_index = {
        (row["checkpoint_sha256"], row["materialization_replicate"], row["lambda"]): row
        for row in geometry
    }
    function_index = {
        (row["checkpoint_sha256"], row["materialization_replicate"]): row
        for row in function if row.get("radius") == 0.005
    }
    rows = []
    for row in replay:
        if row.get("materialization_replicate") != 0:
            continue
        key = (row["checkpoint_sha256"], 0, row["lambda"])
        geo = geometry_index.get(key)
        fun = function_index.get((row["checkpoint_sha256"], 0))
        if geo is None or fun is None:
            raise LowComputeAnalysisError(f"Incomplete LC01/LC04 join for {key}")
        canonical = row["canonical_order"]
        functions = fun["function_space"]
        rows.append(
            {
                "checkpoint_epoch": row["epoch"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "lambda": row["lambda"],
                "effective_sample_size": row["effective_sample_size"],
                "ema_exact_cosine": canonical["cosine"],
                "ema_exact_relative_l2": canonical["relative_l2"],
                "ema_exact_norm_ratio": canonical["norm_ratio"],
                "gB_exact_ema_cosine": geo["gB_exact_cosine_mean"],
                "delta_exact_ema_cosine": geo["delta_exact_cosine_mean"],
                "order_sensitivity_sd": row["order_cosine"]["sd"],
                "actual_checkpoint_parameter_cosine": fun["parameter_space"]["cosine"],
                "text_embedding_agreement": functions["text_all"]["cosine"],
                "logit_agreement": functions["logits_all"]["cosine"],
                "gradient_query_gpu_time_s": accounting.get("exact_gradient_gpu_wall_s"),
            }
        )
    return sorted(rows, key=lambda row: (row["checkpoint_epoch"], row["lambda"]))


def write_primary_table(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    rows = build_primary_rows(run_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "global_gradient_fidelity_primary_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    md_path = destination / "global_gradient_fidelity_primary_table.md"
    lines = [
        "| " + " | ".join(TABLE_FIELDS) + " |",
        "|" + "|".join("---" for _ in TABLE_FIELDS) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]) for field in TABLE_FIELDS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    csv_path, md_path = write_primary_table(Path(args.run_dir), Path(args.output_dir))
    print(json.dumps({"csv": str(csv_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
