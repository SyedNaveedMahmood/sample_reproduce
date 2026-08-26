from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sample_fg.low_compute.reporting import (
    evaluate_lc02_replication_gate,
    load_low_compute_bundle,
    render_low_compute_findings,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _lc02(*, cosine=.11, l2_delta=-1.0, gb=.01, base=-1.0, new=2.1, hm=1.6):
    return {
        "counterfactual_baseline": {
            "mechanism_source_window": {
                "source": {"global_estimate_exact_relative_l2_mean": 4.0},
                "branch_minus_source": {
                    "global_estimate_exact_cosine_mean": cosine,
                    "global_estimate_exact_relative_l2_mean": l2_delta,
                    "batch_component_estimate_exact_cosine_mean": gb,
                },
            },
            "branch_minus_baseline": {
                "base_accuracy_pct": base, "new_accuracy_pct": new,
                "hm_pct": hm,
            },
        }
    }


class LC07GateTests(unittest.TestCase):
    def test_campaign_plot_cli_is_directly_launchable(self):
        result = subprocess.run(
            [sys.executable, "analysis/plot_low_compute_campaign.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_lc02_never_authorizes_confirmation(self):
        result = evaluate_lc02_replication_gate(None)
        self.assertFalse(result["executed"])
        self.assertFalse(result["replication_gate_passed"])
        self.assertFalse(result["optional_seed2_authorized"])

    def test_fixed_threshold_gate_requires_mechanism_gb_and_outcome(self):
        passing = evaluate_lc02_replication_gate(_lc02())
        self.assertTrue(passing["replication_gate_passed"])
        self.assertFalse(passing["optional_seed2_authorized"])
        gb_worse = evaluate_lc02_replication_gate(_lc02(gb=-.001))
        self.assertFalse(gb_worse["mechanism"]["passed"])
        outcome_miss = evaluate_lc02_replication_gate(_lc02(new=1.99))
        self.assertFalse(outcome_miss["outcome"]["passed"])


class LC07ArtifactTests(unittest.TestCase):
    def test_bundle_claim_ledger_and_report_are_hash_traced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lc01 = root / "lc01_lc04"; lc03 = root / "lc03"
            lc05 = root / "lc05"; lc06 = root / "lc06"; output = root / "analysis"
            primary = [
                {"materialization_replicate": 0, "cosine": .2, "relative_l2": 2.0, "epoch": epoch}
                for epoch in (20, 60, 100, 140, 200)
            ]
            _write(lc01 / "summary.json", {
                "status": "completed", "primary_findings": {
                    "actual_checkpoint_ema_vs_materialized_exact": primary
                },
            })
            _write(lc01 / "lc01" / "replay_summary.json", {
                "lc02_gate": {
                    "gate_passed": True, "passing_checkpoint_count": 5,
                    "evidence": [
                        {"cosine_gain": .5, "relative_l2_reduction_fraction": .8}
                        for _ in range(5)
                    ],
                }, "rows": [],
            })
            _write(lc01 / "lc04" / "function_space_fidelity.json", {
                "rows": [{
                    "epoch": 200, "radius": .005,
                    "function_space": {
                        "logits_all": {"cosine": .15}, "text_all": {"cosine": .30}
                    },
                }]
            })
            _write(lc03 / "trajectory_summary.json", {
                "status": "completed", "associations": [{"pearson": .8, "spearman": .7}]
            })
            checkpoint = "a" * 64
            semantic = {
                "dataset": "dtd", "method_key": "sample_ema", "seed": 1,
                "checkpoint_sha256": checkpoint,
                "standard_evaluation": {"base_accuracy_pct": 80., "new_accuracy_pct": 46., "hm_pct": 58.},
                "open_world": {"open_world_new_accuracy_pct": 35.},
                "semantic_drift": {"all": {"mean_cosine_drift": .02}},
            }
            _write(lc05 / "summary.json", {
                "status": "completed", "missing_checkpoint_count": 17
            })
            _write(lc05 / "source_discovery.json", {
                "compatible": [],
                "missing": [{
                    "dataset": "dtd", "method": "coop", "estimator": "none",
                    "seed": 2, "shots": 16,
                }],
                "excluded": [],
            })
            (lc05 / "semantic_drift.jsonl").write_text(json.dumps(semantic) + "\n", encoding="utf-8")
            _write(lc06 / "summary.json", {
                "status": "completed", "missing_checkpoint_count": 17
            })
            sharpness = [
                {
                    "dataset": "dtd", "method_key": "sample_ema", "seed": 1,
                    "checkpoint_sha256": checkpoint, "radius": radius,
                    "sharpness_mean": radius / 10,
                }
                for radius in (.0125, .025, .05)
            ]
            _write(lc06 / "sharpness_summary.json", {"rows": sharpness})
            bundle = load_low_compute_bundle(
                lc01_run=lc01, lc03_run=lc03, lc05_run=lc05, lc06_run=lc06
            )
            report = render_low_compute_findings(bundle, output_dir=output)
            self.assertTrue(report.is_file())
            summary = json.loads((output / "low_compute_summary.json").read_text())
            self.assertEqual(
                summary["stop_go_conclusion"],
                "STOP_WITH_CURRENT_LOW_COMPUTE_EVIDENCE",
            )
            claims = json.loads((output / "low_compute_claims.json").read_text())["claims"]
            self.assertEqual(len(claims), 5)
            self.assertTrue(all("artifact_sha256" in claim for claim in claims))
            for claim in claims:
                self.assertIn(
                    claim["classification"],
                    {"OBSERVATIONAL", "CAUSAL", "MECHANISTIC", "REPRODUCTION", "LIMITATION"},
                )
                self.assertIn("evidence_strength", claim)
                self.assertIsInstance(claim["safe_for_professor_summary"], bool)
                self.assertIsInstance(claim["additional_confirmation_required"], bool)
            self.assertIn(
                "STOP_WITH_CURRENT_LOW_COMPUTE_EVIDENCE", report.read_text()
            )
            self.assertTrue((output / "tables" / "missing_r2_cells.md").is_file())
            cross = output / "tables" / "cross_experiment_summary.md"
            self.assertTrue(cross.is_file())
            self.assertIn("Scientific question", cross.read_text())
            figure_index = output / "plots" / "FIGURE_INDEX.md"
            self.assertTrue(figure_index.is_file())
            figure_text = figure_index.read_text()
            self.assertIn("Professor-facing", figure_text)
            self.assertIn("| 1 |", figure_text)


if __name__ == "__main__":
    unittest.main()
