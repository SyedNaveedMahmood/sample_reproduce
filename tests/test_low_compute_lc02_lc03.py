from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from sample_fg.diagnostics import compute_gradient_diagnostics
from sample_fg.estimators import EMAEstimator
from sample_fg.gradient_state import GradientState
from sample_fg.low_compute.fork_runner import (
    COVERAGE_LAMBDA,
    INTERVENTION,
    LowComputeForkError,
    transplant_ema_state_preserving_direction,
    validate_branch_resume_checkpoint,
)
from sample_fg.low_compute.lc02_audit import hash_directory
from sample_fg.low_compute.trajectory import (
    descriptive_association,
    exploration_summary,
    render_trajectory_artifacts,
)
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.projection import project_batch_gradient
from sample_fg.results import atomic_write_json


class _Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = nn.Module()
        self.prompt_learner.ctx = nn.Parameter(
            torch.tensor([0.25, -0.5], dtype=torch.float32)
        )


def _state(index: ParamIndex, values) -> GradientState:
    return GradientState.from_tensors(
        index, (torch.tensor(values, dtype=torch.float32),)
    )


class LC02ForkUnitTests(unittest.TestCase):
    def setUp(self):
        self.model = _Prompt()
        self.index = ParamIndex.from_model(self.model)

    def test_audit_hash_inventory_is_read_only_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "summary.json"
            artifact.write_bytes(b"immutable")
            before = hash_directory(root)
            self.assertEqual(before, hash_directory(root))
            self.assertEqual(artifact.read_bytes(), b"immutable")
            artifact.write_bytes(b"changed")
            self.assertNotEqual(before, hash_directory(root))

    def test_registered_transplant_preserves_active_state_and_clocks(self):
        source = EMAEstimator(self.index, ema_lambda=0.15)
        source.global_direction(batch_grad=_state(self.index, [1.0, -2.0]), optimizer_step=0)
        before = source.active_state
        target = transplant_ema_state_preserving_direction(
            source, target_lambda=COVERAGE_LAMBDA
        )
        self.assertEqual(source.ema_lambda, 0.15)
        self.assertEqual(target.ema_lambda, 11.0 / 13.0)
        self.assertEqual(target.last_processed_step, source.last_processed_step)
        self.assertEqual(target.exact_query_count, 0)
        self.assertTrue(
            all(torch.equal(a, b) for a, b in zip(before, target.active_state))
        )

    def test_first_divergence_matches_independent_ema_recurrence(self):
        source = EMAEstimator(self.index, ema_lambda=0.15)
        first = _state(self.index, [1.0, -2.0])
        second = _state(self.index, [-0.5, 4.0])
        source.global_direction(batch_grad=first, optimizer_step=0)
        target = transplant_ema_state_preserving_direction(
            source, target_lambda=COVERAGE_LAMBDA
        )
        fork_value = target.active_state.clone()
        observed = target.global_direction(
            batch_grad=second, optimizer_step=1
        ).active_global_estimate
        expected = fork_value.clone().affine_(
            second, COVERAGE_LAMBDA, 1.0 - COVERAGE_LAMBDA
        )
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(observed, expected)))
        baseline = source.global_direction(
            batch_grad=second, optimizer_step=1
        ).active_global_estimate
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(observed, baseline)))

    def test_unregistered_target_lambda_is_rejected(self):
        source = EMAEstimator(self.index, ema_lambda=0.15)
        with self.assertRaises(LowComputeForkError):
            transplant_ema_state_preserving_direction(source, target_lambda=0.9)

    def test_non_paper_source_lambda_is_rejected(self):
        source = EMAEstimator(self.index, ema_lambda=COVERAGE_LAMBDA)
        with self.assertRaises(LowComputeForkError):
            transplant_ema_state_preserving_direction(
                source, target_lambda=COVERAGE_LAMBDA
            )

    def _branch_fixture(self, path: Path, *, step: int = 2160):
        provenance = {
            "schema_version": "sample_fg.low_compute_fork.v1",
            "intervention": "coverage_aware_ema_decay",
            "estimator_state_transplant": INTERVENTION,
            "max_optimizer_steps": 240,
        }
        payload = {
            "config_sha256": "branch-config",
            "estimator_state": {"ema_lambda": COVERAGE_LAMBDA},
            "result_state": {"low_compute_fork": provenance},
            "progress": {
                "next_optimizer_step": step,
                "epoch_zero_based": step // 12,
                "next_batch_index_zero_based": 0,
                "normal_samples_seen": step * 32,
            },
        }
        torch.save(payload, path)
        plan = SimpleNamespace(
            branch_plan=SimpleNamespace(
                resolved_config={
                    "config_sha256": "branch-config",
                    "low_compute": provenance,
                }
            )
        )
        return plan

    def test_branch_resume_preserves_fork_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.pt"
            plan = self._branch_fixture(path, step=2280)
            payload = validate_branch_resume_checkpoint(plan, path)
            self.assertEqual(
                payload["result_state"]["low_compute_fork"],
                plan.branch_plan.resolved_config["low_compute"],
            )

    def test_branch_resume_cannot_increase_authorized_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.pt"
            plan = self._branch_fixture(path, step=2412)
            with self.assertRaises(LowComputeForkError):
                validate_branch_resume_checkpoint(plan, path)

    def test_branch_resume_rejects_wrong_lambda(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "branch.pt"
            plan = self._branch_fixture(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["estimator_state"]["ema_lambda"] = 0.15
            torch.save(payload, path)
            with self.assertRaises(LowComputeForkError):
                validate_branch_resume_checkpoint(plan, path)


class LC03AnalysisUnitTests(unittest.TestCase):
    def setUp(self):
        self.model = _Prompt()
        self.index = ParamIndex.from_model(self.model)

    def test_taylor_diagnostic_matches_independent_raw_calculation(self):
        batch = _state(self.index, [2.0, 1.0])
        estimate = _state(self.index, [1.0, 3.0])
        exact = _state(self.index, [-1.0, 2.0])
        perturbed = _state(self.index, [0.5, -0.25])
        projection = project_batch_gradient(batch, estimate)
        metrics = compute_gradient_diagnostics(
            batch_gradient=batch,
            active_global_estimate=estimate,
            projection=projection,
            perturbed_gradient=perturbed,
            alpha=0.0015,
            exact_full_gradient=exact,
        )
        exploitation = -0.0015 * float(perturbed.dot(batch).item())
        exploration = (
            0.0015
            * projection.xi
            * projection.sigma
            * float(perturbed.dot(estimate).item())
        )
        self.assertAlmostEqual(metrics["taylor/exploitation_term"], exploitation)
        self.assertAlmostEqual(metrics["taylor/exploration_term"], exploration)
        self.assertAlmostEqual(
            metrics["taylor/joint_alignment_term"], exploitation + exploration
        )

    def test_prompt_perturbation_is_restored_bitwise_without_step(self):
        before = self.index[0].parameter.detach().clone()
        with PromptPerturbation(self.index).displaced(
            _state(self.index, [0.01, -0.02])
        ):
            self.assertFalse(torch.equal(self.index[0].parameter, before))
        self.assertTrue(torch.equal(self.index[0].parameter, before))

    def test_exploration_ratio_and_registered_sign_categories(self):
        observed = exploration_summary(-3.0, 1.0)
        self.assertAlmostEqual(observed["R_explore"], 0.25, places=10)
        self.assertEqual(
            observed["sign_category"], "exploration_opposes_exploitation"
        )
        self.assertEqual(
            exploration_summary(-1.0, -2.0)["sign_category"],
            "both_descent_favoring",
        )
        self.assertEqual(
            exploration_summary(0.0, 0.0)["sign_category"], "near_zero"
        )

    def test_association_is_descriptive_and_records_count_without_p_values(self):
        rows = [
            {"metric": 0.1, "new_accuracy_pct": 40.0},
            {"metric": 0.2, "new_accuracy_pct": 45.0},
            {"metric": 0.3, "new_accuracy_pct": 50.0},
        ]
        result = descriptive_association(rows, "metric", "new_accuracy_pct")
        self.assertEqual(result["checkpoint_count"], 3)
        self.assertAlmostEqual(result["pearson"], 1.0)
        self.assertAlmostEqual(result["spearman"], 1.0)
        self.assertFalse(any("p_value" in key for key in result))
        self.assertIn("observational", result["interpretation"])

    def test_association_handles_one_checkpoint(self):
        result = descriptive_association(
            [{"metric": 0.1, "new_accuracy_pct": 40.0}],
            "metric",
            "new_accuracy_pct",
        )
        self.assertEqual(result["checkpoint_count"], 1)
        self.assertIsNone(result["pearson"])
        self.assertIsNone(result["spearman"])

    def test_saved_only_plot_and_table_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "epoch": epoch,
                    "training_fraction": epoch / 200,
                    "ema_exact_cosine": 0.1 + index * 0.1,
                    "ema_exact_relative_l2": 2.0 - index * 0.1,
                    "gB_exact_cosine_mean": 0.7 + index * 0.02,
                    "exploration_fraction_mean": 0.3 + index * 0.01,
                    "base_accuracy_pct": 75.0 + index,
                    "new_accuracy_pct": 40.0 + index,
                    "hm_pct": 52.0 + index,
                }
                for index, epoch in enumerate((20, 60, 100, 140, 200))
            ]
            with (root / "checkpoint_trajectory.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            atomic_write_json(
                root / "trajectory_summary.json",
                {
                    "schema_version": "sample_fg.low_compute_lc03_summary.v1",
                    "final_r2_parity": {"required": True, "passed": True},
                    "p_values_generated": False,
                },
            )
            outputs = render_trajectory_artifacts(root)
            self.assertEqual(len(outputs), 11)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertIn("epoch", (root / "trajectory_table.csv").read_text())


if __name__ == "__main__":
    unittest.main()
