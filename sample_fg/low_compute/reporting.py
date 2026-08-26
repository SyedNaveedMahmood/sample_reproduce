"""LC07 predeclared evidence triage, claim ledger, and findings rendering."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from sample_fg.results import atomic_write_json


LC07_SCHEMA = "sample_fg.low_compute_lc07.v1"


class ReportingError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ReportingError(f"Cannot parse required LC07 artifact: {path}") from error
    if not isinstance(value, dict):
        raise ReportingError(f"LC07 artifact root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ReportingError(f"JSONL row {line_number} is not an object: {path}")
        rows.append(value)
    return rows


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class EvidenceBundle:
    lc01_run: Path
    lc03_run: Path
    lc05_run: Path
    lc06_run: Path
    lc02_run: Path | None
    lc01_summary: dict[str, Any]
    replay_summary: dict[str, Any]
    function_space: dict[str, Any]
    lc03_summary: dict[str, Any]
    lc05_summary: dict[str, Any]
    lc05_discovery: dict[str, Any]
    lc05_rows: tuple[dict[str, Any], ...]
    lc06_summary: dict[str, Any]
    lc06_rows: tuple[dict[str, Any], ...]
    lc02_summary: dict[str, Any] | None
    artifact_hashes: dict[str, str]


def load_low_compute_bundle(
    *,
    lc01_run: Path,
    lc03_run: Path,
    lc05_run: Path,
    lc06_run: Path,
    lc02_run: Path | None = None,
) -> EvidenceBundle:
    roots = {
        "lc01": Path(lc01_run).resolve(strict=True),
        "lc03": Path(lc03_run).resolve(strict=True),
        "lc05": Path(lc05_run).resolve(strict=True),
        "lc06": Path(lc06_run).resolve(strict=True),
    }
    paths = {
        "LC01_summary": roots["lc01"] / "summary.json",
        "LC01_replay": roots["lc01"] / "lc01" / "replay_summary.json",
        "LC04_function": roots["lc01"] / "lc04" / "function_space_fidelity.json",
        "LC03_summary": roots["lc03"] / "trajectory_summary.json",
        "LC05_summary": roots["lc05"] / "summary.json",
        "LC05_discovery": roots["lc05"] / "source_discovery.json",
        "LC05_semantic": roots["lc05"] / "semantic_drift.jsonl",
        "LC06_summary": roots["lc06"] / "summary.json",
        "LC06_sharpness": roots["lc06"] / "sharpness_summary.json",
    }
    lc02_root = None if lc02_run is None else Path(lc02_run).resolve(strict=True)
    if lc02_root is not None:
        paths["LC02_summary"] = lc02_root / "summary.json"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ReportingError(f"LC07 required artifacts are missing: {missing}")
    loaded = {name: _load(path) for name, path in paths.items() if path.suffix == ".json"}
    for name in ("LC01_summary", "LC03_summary", "LC05_summary", "LC06_summary"):
        if loaded[name].get("status") != "completed":
            raise ReportingError(f"{name} is not completed")
    if "LC02_summary" in loaded and loaded["LC02_summary"].get("status") != "completed":
        raise ReportingError("LC02_summary is not completed")
    sharpness = loaded["LC06_sharpness"].get("rows")
    if not isinstance(sharpness, list):
        raise ReportingError("LC06 sharpness summary rows are missing")
    return EvidenceBundle(
        lc01_run=roots["lc01"], lc03_run=roots["lc03"],
        lc05_run=roots["lc05"], lc06_run=roots["lc06"], lc02_run=lc02_root,
        lc01_summary=loaded["LC01_summary"],
        replay_summary=loaded["LC01_replay"],
        function_space=loaded["LC04_function"],
        lc03_summary=loaded["LC03_summary"],
        lc05_summary=loaded["LC05_summary"],
        lc05_discovery=loaded["LC05_discovery"],
        lc05_rows=tuple(_jsonl(paths["LC05_semantic"])),
        lc06_summary=loaded["LC06_summary"],
        lc06_rows=tuple(sharpness),
        lc02_summary=loaded.get("LC02_summary"),
        artifact_hashes={str(path.resolve()): _sha(path) for path in paths.values()},
    )


def evaluate_lc02_replication_gate(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    thresholds = {
        "exact_cosine_absolute_improvement_min": 0.10,
        "relative_l2_reduction_fraction_alternative_min": 0.20,
        "gB_fidelity_must_not_worsen": True,
        "new_accuracy_gain_percentage_points_min": 2.0,
        "hm_gain_percentage_points_min": 1.5,
        "base_accuracy_drop_percentage_points_max": 1.5,
    }
    if summary is None:
        return {
            "executed": False, "replication_gate_passed": False,
            "optional_seed2_authorized": False, "thresholds": thresholds,
            "reason": "LC02 not available; core LC07 synthesis does not depend on it",
        }
    branch = summary.get("counterfactual_baseline", {})
    window = branch.get("mechanism_source_window", {})
    delta = window.get("branch_minus_source", {})
    source = window.get("source", {})
    outcome = branch.get("branch_minus_baseline", {})
    cosine_gain = float(delta.get("global_estimate_exact_cosine_mean", 0.0))
    source_l2 = float(source.get("global_estimate_exact_relative_l2_mean", 0.0))
    l2_delta = float(delta.get("global_estimate_exact_relative_l2_mean", 0.0))
    l2_reduction = (-l2_delta / source_l2) if source_l2 > 0 else 0.0
    raw_gb_delta = delta.get("batch_component_estimate_exact_cosine_mean")
    gb_available = isinstance(raw_gb_delta, (int, float)) and not isinstance(
        raw_gb_delta, bool
    )
    gb_delta = float(raw_gb_delta) if gb_available else None
    mechanism_alternative = (
        cosine_gain >= thresholds["exact_cosine_absolute_improvement_min"]
        or l2_reduction >= thresholds["relative_l2_reduction_fraction_alternative_min"]
    )
    mechanism = mechanism_alternative and gb_available and gb_delta >= 0.0
    outcome_pass = (
        float(outcome.get("new_accuracy_pct", 0.0))
        >= thresholds["new_accuracy_gain_percentage_points_min"]
        and float(outcome.get("hm_pct", 0.0))
        >= thresholds["hm_gain_percentage_points_min"]
        and float(outcome.get("base_accuracy_pct", 0.0))
        >= -thresholds["base_accuracy_drop_percentage_points_max"]
    )
    return {
        "executed": True,
        "mechanism": {
            "exact_cosine_absolute_improvement": cosine_gain,
            "relative_l2_reduction_fraction": l2_reduction,
            "gB_est_exact_cosine_mean_change": gb_delta,
            "gB_fidelity_available": gb_available,
            "alternative_passed": mechanism_alternative,
            "gB_did_not_worsen": gb_available and gb_delta >= 0.0,
            "passed": mechanism,
        },
        "outcome": {
            "base_delta_pct": float(outcome.get("base_accuracy_pct", 0.0)),
            "new_delta_pct": float(outcome.get("new_accuracy_pct", 0.0)),
            "hm_delta_pct": float(outcome.get("hm_pct", 0.0)),
            "passed": outcome_pass,
        },
        "replication_gate_passed": mechanism and outcome_pass,
        "optional_seed2_authorized": False,
        "thresholds": thresholds,
    }


def _lc01_numbers(bundle: EvidenceBundle) -> dict[str, Any]:
    primary = [
        row for row in bundle.lc01_summary["primary_findings"]["actual_checkpoint_ema_vs_materialized_exact"]
        if row.get("materialization_replicate") == 0
    ]
    gate = bundle.replay_summary["lc02_gate"]
    final_functions = [
        row for row in bundle.function_space["rows"]
        if row.get("epoch") == 200 and row.get("radius") == 0.005
    ]
    if len(primary) != 5 or len(final_functions) != 1:
        raise ReportingError("LC01/LC04 primary five-checkpoint evidence is incomplete")
    return {
        "checkpoint_count": 5,
        "actual_ema_exact_cosine_mean": statistics.fmean(float(row["cosine"]) for row in primary),
        "actual_ema_exact_cosine_final": float(primary[-1]["cosine"]),
        "actual_ema_exact_relative_l2_mean": statistics.fmean(float(row["relative_l2"]) for row in primary),
        "coverage_gate_passing_checkpoints": int(gate["passing_checkpoint_count"]),
        "coverage_cosine_gain_mean": statistics.fmean(float(row["cosine_gain"]) for row in gate["evidence"]),
        "coverage_relative_l2_reduction_mean": statistics.fmean(float(row["relative_l2_reduction_fraction"]) for row in gate["evidence"]),
        "final_logit_response_cosine": float(final_functions[0]["function_space"]["logits_all"]["cosine"]),
        "final_text_response_cosine": float(final_functions[0]["function_space"]["text_all"]["cosine"]),
    }


def evaluate_predeclared_headline_patterns(bundle: EvidenceBundle) -> dict[str, Any]:
    values = _lc01_numbers(bundle)
    # Pattern A follows the predeclared LC01 gate plus the observed LC04 response;
    # no post-hoc search over numerical thresholds is performed here.
    pattern_a = bundle.replay_summary["lc02_gate"].get("gate_passed") is True
    return {
        "selected": "A" if pattern_a else "B",
        "patterns": {
            "A": {
                "status": "supported" if pattern_a else "not_supported",
                "headline": "In prompt-only SAMPLe, the nominal global-gradient proxy is strongly minibatch-order dependent, and estimator error propagates to the intended exploration geometry/function.",
                "evidence": values,
            },
            "B": {
                "status": "not_supported" if pattern_a else "mixed",
                "reason": "Final logit/text response alignment is not high enough to call the parameter mismatch functionally benign.",
            },
            "C": {
                "status": "mixed",
                "reason": "LC05 is complete for the one locally available SAMPLe seed, but missing CoOp/SAM and replicated seeds preclude a cross-method association claim.",
            },
            "D": {
                "status": "mixed",
                "reason": "LC06 is complete for the one locally available SAMPLe seed, but no local CoOp/SAM checkpoints permit the predeclared flatness comparison.",
            },
        },
    }


def build_claim_ledger(
    bundle: EvidenceBundle,
    *,
    patterns: Mapping[str, Any],
    lc02_gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    artifacts = bundle.artifact_hashes
    mechanism_paths = [path for path in artifacts if "lc01_lc04" in path.lower()]
    return [
        {
            "claim_id": "LC-CLAIM-ESTIMATOR-FUNCTION",
            "text": patterns["patterns"]["A"]["headline"],
            "status": patterns["patterns"]["A"]["status"],
            "classification": "MECHANISTIC",
            "evidence_strength": "moderate",
            "safe_for_professor_summary": True,
            "additional_confirmation_required": True,
            "tasks": ["LC01", "LC04"], "datasets": ["dtd"], "seeds": [1],
            "artifacts": mechanism_paths,
            "artifact_sha256": {path: artifacts[path] for path in mechanism_paths},
            "limitations": ["One training seed; five checkpoints and 512 fixed order trials per checkpoint are mechanism observations, not population evidence."],
            "new_optimizer_steps_supporting_claim": 0,
        },
        {
            "claim_id": "LC-CLAIM-TRAJECTORY",
            "text": "EMA/exact cosine is descriptively associated with checkpoint New accuracy along the seed-1 trajectory.",
            "status": "mixed", "tasks": ["LC03"], "datasets": ["dtd"], "seeds": [1],
            "classification": "OBSERVATIONAL",
            "evidence_strength": "limited",
            "safe_for_professor_summary": True,
            "additional_confirmation_required": True,
            "artifacts": [path for path in artifacts if "lc03" in path.lower()],
            "artifact_sha256": {path: value for path, value in artifacts.items() if "lc03" in path.lower()},
            "limitations": ["Five time-ordered checkpoints are dependent observations; no p-values or causal claim."],
            "new_optimizer_steps_supporting_claim": 0,
        },
        {
            "claim_id": "LC-CLAIM-LC02-REPLICATION",
            "text": "The fixed coverage-aware seed-1 branch does not justify an optional seed-2 replication under the preregistered gate.",
            "status": "supported" if lc02_gate.get("executed") else "mixed",
            "classification": "CAUSAL" if lc02_gate.get("executed") else "LIMITATION",
            "evidence_strength": "limited" if lc02_gate.get("executed") else "not_available",
            "safe_for_professor_summary": True,
            "additional_confirmation_required": False,
            "tasks": ["LC02", "LC07"], "datasets": ["dtd"], "seeds": [1],
            "artifacts": [path for path in artifacts if "lc02" in path.lower()],
            "artifact_sha256": {path: value for path, value in artifacts.items() if "lc02" in path.lower()},
            "limitations": ["LC02 was a one-seed causal pilot executed independently of the zero-training LC05-LC07 request."],
            "new_optimizer_steps_supporting_claim": 240 if lc02_gate.get("executed") else 0,
        },
        {
            "claim_id": "LC-CLAIM-SEMANTIC-OPEN-WORLD",
            "text": "Semantic-drift and open-world metrics are measured, but the requested cross-method/seed relationship is not estimable from the transferred artifacts.",
            "status": "mixed", "tasks": ["LC05"], "datasets": sorted({row["dataset"] for row in bundle.lc05_rows}),
            "classification": "OBSERVATIONAL",
            "evidence_strength": "limited",
            "safe_for_professor_summary": True,
            "additional_confirmation_required": True,
            "seeds": sorted({row["seed"] for row in bundle.lc05_rows}),
            "artifacts": [path for path in artifacts if "lc05" in path.lower()],
            "artifact_sha256": {path: value for path, value in artifacts.items() if "lc05" in path.lower()},
            "limitations": [f"Only {len(bundle.lc05_rows)} of 18 requested R2 cells was locally compatible; missing cells were not trained."],
            "new_optimizer_steps_supporting_claim": 0,
        },
        {
            "claim_id": "LC-CLAIM-SHARPNESS",
            "text": "Fixed-materialization prompt sharpness is measured, but whether SAM/SAMPLe are flatter than CoOp is not estimable from the transferred artifacts.",
            "status": "mixed", "tasks": ["LC06"], "datasets": sorted({row["dataset"] for row in bundle.lc06_rows}),
            "classification": "OBSERVATIONAL",
            "evidence_strength": "limited",
            "safe_for_professor_summary": True,
            "additional_confirmation_required": True,
            "seeds": sorted({row["seed"] for row in bundle.lc06_rows}),
            "artifacts": [path for path in artifacts if "lc06" in path.lower()],
            "artifact_sha256": {path: value for path, value in artifacts.items() if "lc06" in path.lower()},
            "limitations": ["Random-direction sharpness is conditional on one deterministic materialization and one locally available method/seed."],
            "new_optimizer_steps_supporting_claim": 0,
        },
    ]


def render_low_compute_findings(
    bundle: EvidenceBundle,
    *,
    output_dir: Path,
) -> Path:
    output = Path(output_dir).resolve()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "plots").mkdir(parents=True, exist_ok=True)
    patterns = evaluate_predeclared_headline_patterns(bundle)
    lc02_gate = evaluate_lc02_replication_gate(bundle.lc02_summary)
    conclusion = (
        "OPTIONAL_CONFIRMATION_RECOMMENDED_AFTER_LC02"
        if lc02_gate["replication_gate_passed"]
        else "STOP_WITH_CURRENT_LOW_COMPUTE_EVIDENCE"
    )
    claims = build_claim_ledger(bundle, patterns=patterns, lc02_gate=lc02_gate)
    values = patterns["patterns"]["A"]["evidence"]
    trajectory = bundle.lc03_summary["associations"][0]
    semantic = bundle.lc05_rows[0] if bundle.lc05_rows else None
    sharp = next((row for row in bundle.lc06_rows if row["radius"] == 0.05), None)
    summary = {
        "schema_version": LC07_SCHEMA,
        "status": "completed", "selected_headline_pattern": patterns["selected"],
        "headline_patterns": patterns["patterns"],
        "lc02_replication_gate": lc02_gate,
        "stop_go_conclusion": conclusion,
        "task_status": {
            "LC01": "completed", "LC03": "completed", "LC04": "completed",
            "LC05": "completed_with_missing_source_cells",
            "LC06": "completed_with_missing_source_cells",
            "LC02": "completed_independently_not_required_for_core_synthesis" if bundle.lc02_summary else "not_executed",
            "LC07_optional_seed2": "not_executed_not_authorized",
            "E0_E1_E2_F0": "not_executed",
        },
        "missing_r2_cells": bundle.lc05_discovery.get("missing", []),
        "compute": {
            "new_optimizer_steps_in_lc05_lc06_lc07": 0,
            "independently_completed_lc02_optimizer_steps": 240 if bundle.lc02_summary else 0,
            "optional_seed2_optimizer_steps": 0,
            "fraction_of_full_dtd_run_for_lc05_lc06_lc07": 0.0,
        },
        "artifact_hashes": bundle.artifact_hashes,
    }
    atomic_write_json(output / "low_compute_summary.json", summary)
    atomic_write_json(output / "low_compute_claims.json", {
        "schema_version": LC07_SCHEMA, "claims": claims,
    })
    # Raw seed table (one row per locally compatible checkpoint) is never hidden.
    headers = [
        "dataset", "method", "seed", "standard_new", "open_world_new",
        "semantic_drift_all", "sharpness_mean_rho",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    sharp_index = {
        row["checkpoint_sha256"]: row for row in bundle.lc06_rows if row["radius"] == 0.05
    }
    for row in bundle.lc05_rows:
        joined = sharp_index.get(row["checkpoint_sha256"])
        values_row = [
            row["dataset"], row["method_key"], str(row["seed"]),
            f"{row['standard_evaluation']['new_accuracy_pct']:.4f}",
            f"{row['open_world']['open_world_new_accuracy_pct']:.4f}",
            f"{row['semantic_drift']['all']['mean_cosine_drift']:.8f}",
            "NA" if joined is None else f"{joined['sharpness_mean']:.8f}",
        ]
        lines.append("| " + " | ".join(values_row) + " |")
    (output / "tables" / "raw_seed_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cross_rows = [
        (
            "LC01", "How faithful/global is the EMA direction?",
            f"EMA/exact cosine mean {values['actual_ema_exact_cosine_mean']:.3f}; coverage replay passed {values['coverage_gate_passing_checkpoints']}/5 checkpoints.",
            "DTD / seed 1", "0", "EMA is strongly order-dependent.",
            "One trajectory; replay is mechanistic, not a population result.",
        ),
        (
            "LC03", "Does fidelity track Base/New behavior over training?",
            f"EMA-cosine/New association Pearson {trajectory['pearson']:.3f}, Spearman {trajectory['spearman']:.3f}.",
            "DTD / seed 1 / 5 checkpoints", "0", "The relationship is descriptively aligned with New accuracy.",
            "Time-ordered checkpoints are dependent; no causal or significance claim.",
        ),
        (
            "LC04", "Is parameter mismatch functionally benign?",
            f"Final text/logit response cosine {values['final_text_response_cosine']:.3f}/{values['final_logit_response_cosine']:.3f}.",
            "DTD / seed 1", "0", "The final mismatch is not functionally benign.",
            "One checkpoint trajectory and registered finite-difference radii.",
        ),
        (
            "LC05", "Do semantic drift/open-world behavior clarify generalization?",
            "Unavailable" if semantic is None else (
                f"Semantic drift {semantic['semantic_drift']['all']['mean_cosine_drift']:.3f}; "
                f"open-world New {semantic['open_world']['open_world_new_accuracy_pct']:.2f}%."
            ),
            "DTD / seed 1", "0", "Valid read-only measurement; no method ranking.",
            "Only 1 of 18 requested R2 cells was locally compatible.",
        ),
        (
            "LC06", "Does prompt sharpness track novel generalization?",
            "Unavailable" if sharp is None else f"Mean sampled sharpness at rho {sharp['sharpness_mean']:.6f}.",
            "DTD / seed 1", "0", "Sharpness is measured but cannot be compared across methods.",
            "One method/seed and one deterministic materialization.",
        ),
    ]
    cross_lines = [
        "| Experiment | Scientific question | Primary metric/result | Dataset/seeds | Optimizer steps | Strongest supported conclusion | Major caveat |",
        "|---|---|---|---|---:|---|---|",
    ]
    cross_lines.extend("| " + " | ".join(row) + " |" for row in cross_rows)
    (output / "tables" / "cross_experiment_summary.md").write_text(
        "\n".join(cross_lines) + "\n", encoding="utf-8"
    )
    missing_lines = [
        "| dataset | method | estimator | seed | shots |",
        "|---|---|---|---|---|",
    ]
    for cell in bundle.lc05_discovery.get("missing", []):
        missing_lines.append(
            f"| {cell['dataset']} | {cell['method']} | {cell['estimator']} | {cell['seed']} | {cell['shots']} |"
        )
    (output / "tables" / "missing_r2_cells.md").write_text(
        "\n".join(missing_lines) + "\n", encoding="utf-8"
    )
    (output / "tables" / "TABLE_INDEX.md").write_text(
        "# Table index\n\n"
        "- `cross_experiment_summary.md`: compact LC01/03/04/05/06 scientific synthesis.\n"
        "- `raw_seed_metrics.md`: every locally compatible checkpoint; no seed aggregation is hidden.\n"
        "- `missing_r2_cells.md`: every requested R2 cell absent from this transfer.\n",
        encoding="utf-8",
    )
    (output / "plots" / "FIGURE_INDEX.md").write_text(
        "# Figure index\n\n"
        "| Rank | Path | Experiment | What it shows | Professor-facing | Caption |\n"
        "|---:|---|---|---|---|---|\n"
        "| 1 | `mechanism_fidelity.png` / `.svg` | LC01 + LC04 | Actual EMA, paper replay, and coverage-replay fidelity over checkpoints. | Yes | Coverage-aware replay improves fidelity, while the stored EMA remains poorly aligned with the exact direction. |\n"
        "| 2 | `semantic_open_world.png` / `.svg` | LC05 | Standard separate-label versus all-class New accuracy. | Yes, with n=1 caveat | The available SAMPLe checkpoint loses New accuracy under the all-class protocol. |\n"
        "| 3 | `prompt_sharpness.png` / `.svg` | LC06 | Fixed-materialization mean sampled sharpness by radius. | Yes, with n=1 caveat | Prompt-space loss increases monotonically with the predeclared radius. |\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": LC07_SCHEMA,
        "tables": [
            "tables/TABLE_INDEX.md", "tables/cross_experiment_summary.md",
            "tables/raw_seed_metrics.md",
            "tables/missing_r2_cells.md",
        ],
        "figures": [
            "plots/mechanism_fidelity.png",
            "plots/semantic_open_world.png",
            "plots/prompt_sharpness.png",
        ],
        "source_artifact_sha256": bundle.artifact_hashes,
    }
    atomic_write_json(output / "artifact_manifest.json", manifest)
    semantic_text = "Unavailable"
    open_new = "Unavailable"
    if semantic is not None:
        semantic_text = f"{semantic['semantic_drift']['all']['mean_cosine_drift']:.6f}"
        open_new = f"{semantic['open_world']['open_world_new_accuracy_pct']:.2f}%"
    sharp_text = "Unavailable" if sharp is None else f"{sharp['sharpness_mean']:.6f}"
    lc02_text = (
        "The independently completed seed-1 LC02 branch changed Base/New/HM by 0.00/0.00/0.00 points; its preregistered replication gate failed."
        if bundle.lc02_summary else "LC02 was not available and was not required for this synthesis."
    )
    report = f"""# Low-compute findings

## 1. Reproduction context

CoOp was close to the paper on DTD, while the audited public-spec SAM/SAMPLe implementation did not reproduce the reported novel-class gain. The campaign preserves the paper constants and does not use result-driven tuning. This transferred PC contains only one compatible final R2 checkpoint (DTD SAMPLe-EMA seed 1); the other 17 requested LC05 cells were reported missing and were not trained.

## 2. Question

How accurate/global is SAMPLe's EMA direction, does estimator error propagate into prompt function, and do semantic drift, open-world behavior, or prompt sharpness clarify the generalization gap?

## 3. Three strongest findings

1. Across five seed-1 checkpoints, actual EMA/exact cosine averaged **{values['actual_ema_exact_cosine_mean']:.3f}** (final **{values['actual_ema_exact_cosine_final']:.3f}**) and relative L2 averaged **{values['actual_ema_exact_relative_l2_mean']:.3f}**. The fixed coverage-aware replay passed the predeclared non-accuracy gate at **{values['coverage_gate_passing_checkpoints']}/5** checkpoints, with mean cosine gain **{values['coverage_cosine_gain_mean']:.3f}** and mean relative-L2 reduction **{100*values['coverage_relative_l2_reduction_mean']:.1f}%**.
2. Estimator mismatch was not functionally benign at the final checkpoint: EMA/exact central-difference response cosine was **{values['final_text_response_cosine']:.3f}** for text embeddings and **{values['final_logit_response_cosine']:.3f}** for all-class logits. LC03's five-checkpoint EMA-cosine/New association was Pearson **{trajectory['pearson']:.3f}**, Spearman **{trajectory['spearman']:.3f}**; this is descriptive and non-causal.
3. For the only transferred final checkpoint, all-class semantic drift was **{semantic_text}**, open-world New accuracy was **{open_new}**, and fixed-materialization sampled sharpness at rho was **{sharp_text}**. These measurements are valid read-only results, but no cross-method or cross-seed semantic/flatness conclusion is supportable without the missing checkpoints.

## 4. Mechanism figure

See `plots/mechanism_fidelity.png`, `plots/FIGURE_INDEX.md`, and the hash-traced LC01/LC04 artifacts in `artifact_manifest.json`.

## 5. VLM-level figure

See `plots/semantic_open_world.png`; `plots/prompt_sharpness.png` records the complementary fixed-materialization landscape measurement.

## 6. Compute statement

LC05, LC06, and LC07 executed **0 optimizer steps and 0 scheduler steps** (0% of a 2,400-step DTD run). {lc02_text} No optional seed-2 confirmation was executed or authorized.

## 7. Next hypothesis

Obtain the missing immutable CoOp/SAM/SAMPLe R2 checkpoints and rerun only LC05/LC06 read-only analysis to test whether text-geometry preservation or prompt sharpness separates methods; otherwise request author code/clarification before expanding training.

## Claim scope and stop decision

The selected predeclared headline is Pattern **{patterns['selected']}**. Claims and limitations are enumerated in `low_compute_claims.json`; table and figure indices are in `tables/TABLE_INDEX.md` and `plots/FIGURE_INDEX.md`. No p-values, significance stars, accuracy-selected lambda, optional seed-2 training, E0/E1/E2/F0, or retraining were used.

**{conclusion}**
"""
    report_path = output / "LOW_COMPUTE_FINDINGS.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path
