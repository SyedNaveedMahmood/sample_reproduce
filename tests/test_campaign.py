from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.freeze_periodic_k import freeze
from sample_fg.campaign import (
    CampaignError,
    CampaignManifest,
    load_periodic_k_freeze,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = REPO_ROOT / "configs" / "sample_fg" / "extension_campaign.yaml"


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.campaign = CampaignManifest.load(CAMPAIGN_PATH)

    def test_declared_task_matrix_and_k_set_are_deterministic(self):
        self.assertEqual(self.campaign.allowed_k, (2, 4, 8, 16))
        self.assertEqual(len(self.campaign.cells("task25")), 18)
        self.assertEqual(len(self.campaign.cells("task26")), 3)
        self.assertEqual(len(self.campaign.cells("task27")), 4)
        self.assertEqual(
            len(self.campaign.cells("task28", frozen_k_values=(4,))), 18
        )
        self.assertEqual(
            len(self.campaign.cells("task28", frozen_k_values=(4, 16))), 24
        )
        task28_ema = [
            cell
            for cell in self.campaign.cells("task28", frozen_k_values=(4,))
            if cell.estimator == "ema"
        ]
        self.assertEqual(len(task28_ema), 6)
        self.assertTrue(
            all(cell.reuse_experiment_id == "R2" for cell in task28_ema)
        )

    def test_expected_step_and_refresh_counts_follow_zero_based_protocol(self):
        periodic = next(
            cell
            for cell in self.campaign.cells("task27")
            if cell.periodic_k_steps == 4
        )
        self.assertEqual(periodic.selected_count, 384)
        self.assertEqual(periodic.steps_per_epoch, 12)
        self.assertEqual(periodic.total_optimizer_steps, 2400)
        self.assertEqual(periodic.expected_periodic_refresh_count, 600)
        self.assertEqual(periodic.expected_diagnostic_points, 200)
        self.assertEqual(periodic.expected_reused_exact_queries, 200)
        self.assertEqual(periodic.expected_exact_sweeps, 600)

    def test_task28_requires_a_real_freeze_for_cell_validation(self):
        with self.assertRaisesRegex(CampaignError, "requires a periodic-K freeze"):
            self.campaign.cells("task28")

    def test_freeze_rejects_accuracy_use_and_out_of_grid_k(self):
        base = {
            "schema_version": "sample_fg.periodic_k_freeze.v1",
            "campaign_config_sha256": self.campaign.sha256,
            "selected_k_values": [4],
            "f0_k": 4,
            "accuracy_used": False,
            "source_aggregation_sha256": "a" * 64,
            "rationale": "fidelity and cost",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "freeze.json"
            path.write_text(json.dumps(base), encoding="utf-8")
            loaded = load_periodic_k_freeze(path, campaign=self.campaign)
            self.assertEqual(loaded.selected_k_values, (4,))
            for change in (
                {"accuracy_used": True},
                {"selected_k_values": [3], "f0_k": 3},
            ):
                payload = {**base, **change}
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(CampaignError):
                    load_periodic_k_freeze(path, campaign=self.campaign)

    def test_freeze_script_requires_complete_non_accuracy_e1_evidence(self):
        summary_rows = [
            {
                "experiment_id": "E1",
                "method": "sample",
                "estimator_mode": "periodic",
                "periodic_k_steps": k,
                "estimator_exact_cosine_mean": 0.9,
                "estimator_exact_relative_l2_mean": 0.1,
                "estimator_exact_log_norm_ratio_mean": 0.0,
            }
            for k in self.campaign.allowed_k
        ]
        efficiency_rows = [
            {
                "experiment_id": "E1",
                "estimator_mode": "periodic",
                "periodic_k_steps": k,
                "train_total_s": 10.0,
                "full_gradient_total_s": 2.0,
                "exact_sweeps": next(
                    cell.expected_exact_sweeps
                    for cell in self.campaign.cells("task27")
                    if cell.periodic_k_steps == k
                ),
            }
            for k in self.campaign.allowed_k
        ]
        diagnostic_rows = [
            {
                "experiment_id": "E1",
                "estimator_mode": "periodic",
                "periodic_k_steps": k,
                "grad/batch_gradient_degenerate": False,
                "grad/global_direction_degenerate": False,
                "grad/exact_full_direction_degenerate": False,
                "grad/batch_component_degenerate": False,
                "grad/reference_batch_component_degenerate": False,
                "grad/perturbed_gradient_degenerate": False,
            }
            for k in self.campaign.allowed_k
            for _ in range(200)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregate = root / "aggregate"
            aggregate.mkdir()
            for name, rows in (
                ("summary_by_cell.json", summary_rows),
                ("efficiency.json", efficiency_rows),
                ("diagnostics_long.json", diagnostic_rows),
            ):
                (aggregate / name).write_text(
                    json.dumps({"rows": rows}), encoding="utf-8"
                )
            (aggregate / "aggregation_report.json").write_text(
                json.dumps({"mode": "scientific"}), encoding="utf-8"
            )
            destination = root / "periodic_k_freeze.json"
            freeze(
                Namespace(
                    campaign_config=str(CAMPAIGN_PATH),
                    aggregate_dir=str(aggregate),
                    output=str(destination),
                    selected_k=[4, 16],
                    f0_k=4,
                    rationale="K=4 and K=16 bracket fidelity and measured cost.",
                )
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))
            self.assertFalse(payload["accuracy_used"])
            self.assertEqual(payload["observed_k_values"], [2, 4, 8, 16])
            self.assertEqual(payload["selected_k_values"], [4, 16])
            self.assertEqual(payload["evidence"][0]["nonfinite_event_count"], 0)
            self.assertNotIn("base_mean", json.dumps(payload).lower())
            with self.assertRaisesRegex(CampaignError, "refusing replacement"):
                freeze(
                    Namespace(
                        campaign_config=str(CAMPAIGN_PATH),
                        aggregate_dir=str(aggregate),
                        output=str(destination),
                        selected_k=[4],
                        f0_k=4,
                        rationale="fidelity and cost",
                    )
                )


if __name__ == "__main__":
    unittest.main()
