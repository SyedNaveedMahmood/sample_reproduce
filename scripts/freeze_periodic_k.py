"""Record a human periodic-K decision from non-accuracy E1 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.campaign import (
    PERIODIC_K_FREEZE_SCHEMA_VERSION,
    CampaignError,
    CampaignManifest,
)
from sample_fg.results import atomic_write_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"Cannot read aggregate artifact: {path}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"Aggregate artifact is not a mapping: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    value = _load(path).get("rows")
    if not isinstance(value, list) or not all(
        isinstance(row, dict) for row in value
    ):
        raise CampaignError(f"Aggregate artifact lacks rows: {path}")
    return value


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"E1 evidence lacks numeric {label}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise CampaignError(f"E1 evidence requires {qualifier} {label}")
    return result


def _positive_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CampaignError(f"E1 evidence requires positive integer {label}")
    return value


def freeze(args: argparse.Namespace) -> Path:
    campaign = CampaignManifest.load(Path(args.campaign_config))
    selected = tuple(args.selected_k)
    if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected):
        raise CampaignError("Select one or two distinct K values")
    if any(value not in campaign.allowed_k for value in selected):
        raise CampaignError(
            f"Selected K must be drawn from {campaign.allowed_k}"
        )
    if args.f0_k not in selected:
        raise CampaignError("--f0-k must be one of --selected-k values")
    if not args.rationale.strip():
        raise CampaignError("A nonempty mechanism/cost rationale is required")

    aggregate_dir = Path(args.aggregate_dir).resolve(strict=True)
    paths = {
        name: aggregate_dir / name
        for name in (
            "aggregation_report.json",
            "summary_by_cell.json",
            "efficiency.json",
            "diagnostics_long.json",
        )
    }
    for path in paths.values():
        if not path.is_file():
            raise CampaignError(f"Required E1 aggregate artifact is missing: {path}")
    report = _load(paths["aggregation_report.json"])
    if report.get("mode") != "scientific":
        raise CampaignError("Periodic K must be frozen from scientific aggregation")
    summary = _rows(paths["summary_by_cell.json"])
    efficiency = _rows(paths["efficiency.json"])
    diagnostics = _rows(paths["diagnostics_long.json"])
    e1_summary = [
        row
        for row in summary
        if row.get("experiment_id") == "E1"
        and row.get("method") == "sample"
        and row.get("estimator_mode") == "periodic"
    ]
    observed_k = sorted(
        {int(row["periodic_k_steps"]) for row in e1_summary}
    )
    if observed_k != list(campaign.allowed_k):
        raise CampaignError(
            "E1 aggregate does not contain the complete predeclared K set: "
            f"observed {observed_k}"
        )
    evidence: list[dict[str, Any]] = []
    expected_by_k = {
        cell.periodic_k_steps: cell
        for cell in campaign.cells("task27")
        if cell.estimator == "periodic"
    }
    for k in campaign.allowed_k:
        rows = [row for row in e1_summary if int(row["periodic_k_steps"]) == k]
        if len(rows) != 1:
            raise CampaignError(f"E1 K={k} summary is not unique")
        efficiency_rows = [
            row
            for row in efficiency
            if row.get("experiment_id") == "E1"
            and row.get("estimator_mode") == "periodic"
            and int(row.get("periodic_k_steps") or 0) == k
        ]
        if len(efficiency_rows) != 1:
            raise CampaignError(f"E1 K={k} efficiency evidence is not unique")
        source = rows[0]
        fidelity = {
            field: _finite_number(source.get(field), f"K={k} {field}")
            for field in (
                "estimator_exact_cosine_mean",
                "estimator_exact_relative_l2_mean",
                "estimator_exact_log_norm_ratio_mean",
            )
        }
        cost_rows = []
        for row in efficiency_rows:
            exact_time = row.get(
                "exact_gradient_total_s", row.get("full_gradient_total_s")
            )
            cost_rows.append(
                {
                    "train_total_s": _finite_number(
                        row.get("train_total_s"),
                        f"K={k} training wall time",
                        positive=True,
                    ),
                    "exact_gradient_total_s": _finite_number(
                        exact_time,
                        f"K={k} exact-gradient wall time",
                        positive=True,
                    ),
                    "exact_sweeps": _positive_count(
                        row.get("exact_sweeps", row.get("full_gradient_sweeps")),
                        f"K={k} exact sweeps",
                    ),
                }
            )
        expected_cell = expected_by_k[k]
        if cost_rows[0]["exact_sweeps"] != expected_cell.expected_exact_sweeps:
            raise CampaignError(
                f"E1 K={k} exact-sweep count differs from declared accounting"
            )
        diagnostic_rows = [
            row
            for row in diagnostics
            if row.get("experiment_id") == "E1"
            and row.get("estimator_mode") == "periodic"
            and int(row.get("periodic_k_steps") or 0) == k
        ]
        if len(diagnostic_rows) != expected_cell.expected_diagnostic_points:
            raise CampaignError(
                f"E1 K={k} diagnostic count differs from declared cadence"
            )
        degenerate_fields = (
            "grad/batch_gradient_degenerate",
            "grad/global_direction_degenerate",
            "grad/exact_full_direction_degenerate",
            "grad/batch_component_degenerate",
            "grad/reference_batch_component_degenerate",
            "grad/perturbed_gradient_degenerate",
        )
        invalid_flags = [
            (field, row.get(field))
            for row in diagnostic_rows
            for field in degenerate_fields
            if row.get(field) not in {True, False}
        ]
        if invalid_flags:
            raise CampaignError(f"E1 K={k} has malformed degeneracy evidence")
        degeneracy_counts = {
            field: sum(row.get(field) is True for row in diagnostic_rows)
            for field in degenerate_fields
        }
        evidence.append(
            {
                "periodic_k_steps": k,
                **fidelity,
                "cost": cost_rows,
                "diagnostic_points": len(diagnostic_rows),
                "degenerate_vector_counts": degeneracy_counts,
                "nonfinite_event_count": 0,
            }
        )
    artifact_hashes = {name: _sha256(path) for name, path in paths.items()}
    source_hash = hashlib.sha256(
        json.dumps(
            artifact_hashes,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": PERIODIC_K_FREEZE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "campaign_config": str(campaign.path),
        "campaign_config_sha256": campaign.sha256,
        "source_aggregation_dir": str(aggregate_dir),
        "source_aggregation_sha256": source_hash,
        "source_artifact_sha256": artifact_hashes,
        "observed_k_values": observed_k,
        "selected_k_values": list(selected),
        "f0_k": args.f0_k,
        "accuracy_used": False,
        "selection_rule": (
            "Human pre-confirmatory decision using estimator fidelity, stability, "
            "and measured exact/training cost only; Base/New/HM are not inputs."
        ),
        "rationale": args.rationale.strip(),
        "evidence": evidence,
    }
    destination = Path(args.output).resolve()
    if destination.exists():
        raise CampaignError(
            f"Freeze record already exists; refusing replacement: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination, payload)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selected-k", action="append", required=True, type=int)
    parser.add_argument("--f0-k", required=True, type=int)
    parser.add_argument("--rationale", required=True)
    parser.add_argument(
        "--campaign-config",
        default=str(REPO_ROOT / "configs" / "sample_fg" / "extension_campaign.yaml"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(freeze(parse_args(argv)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
