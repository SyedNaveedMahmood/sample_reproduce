from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from sample_fg.low_compute.probe_runtime import fixed_feature_loss_fn
from sample_fg.low_compute.campaign_sources import discover_r2_sources
from sample_fg.low_compute.semantic import (
    compute_neighbor_preservation,
    compute_semantic_drift,
    compute_topology_distortion,
    evaluate_open_world_logits,
    pearson_spearman,
)
from sample_fg.low_compute.sharpness import (
    RADII,
    parameter_sha256,
    probe_structured_direction,
    probe_symmetric_loss_sharpness,
    sample_prompt_directions,
    summarize_sharpness,
)
from sample_fg.param_index import ParamIndex
from sample_fg.gradient_state import GradientState
from tests.test_low_compute import make_source_run


class _PromptLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Parameter(
            torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
        )

    def forward(self):
        return self.ctx


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = _PromptLearner()
        self.register_buffer("logit_scale", torch.tensor(0.0))
        self.register_buffer("tokenized_prompts", torch.zeros(2, 1, dtype=torch.long))

    @staticmethod
    def text_encoder(prompts, _tokens):
        return prompts / prompts.norm(dim=1, keepdim=True)


class SemanticMetricTests(unittest.TestCase):
    def setUp(self):
        self.reference = torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]],
            dtype=torch.float32,
        )

    def test_clone_has_zero_drift_and_perfect_neighbors(self):
        learned = self.reference.clone()
        drift = compute_semantic_drift(
            self.reference, learned, base_class_count=2
        )
        self.assertLess(abs(drift["all"]["mean_cosine_drift"]), 1e-7)
        self.assertAlmostEqual(drift["all"]["normalized_frobenius_drift"], 0.0)
        topology = compute_topology_distortion(
            self.reference, learned, base_class_count=2
        )
        self.assertTrue(all(value == 0.0 for value in topology.values()))
        neighbors = compute_neighbor_preservation(
            self.reference, learned, base_class_count=2
        )
        self.assertEqual(neighbors["k"], 3)
        self.assertEqual(neighbors["all"]["mean_jaccard"], 1.0)
        self.assertEqual(neighbors["all"]["top1_preservation_fraction"], 1.0)

    def test_topology_all_metric_matches_explicit_off_diagonal(self):
        learned = self.reference.clone()
        learned[1] = torch.tensor([-0.8, 0.2])
        observed = compute_topology_distortion(
            self.reference, learned, base_class_count=2
        )["all_off_diagonal"]
        ref = self.reference / self.reference.norm(dim=1, keepdim=True)
        value = learned / learned.norm(dim=1, keepdim=True)
        mask = ~torch.eye(4, dtype=torch.bool)
        expected = torch.linalg.vector_norm(
            (value @ value.t())[mask] - (ref @ ref.t())[mask]
        ) / torch.linalg.vector_norm((ref @ ref.t())[mask])
        self.assertAlmostEqual(observed, float(expected), places=6)

    def test_open_world_uses_all_classes_and_group_confusion(self):
        text = torch.eye(2)
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
        labels = torch.tensor([0, 1, 1])
        result = evaluate_open_world_logits(
            image_features=images, labels=labels, text_features=text,
            base_class_count=1, logit_scale=1.0,
        )
        self.assertEqual(result["open_world_base_accuracy_pct"], 100.0)
        self.assertEqual(result["open_world_new_accuracy_pct"], 50.0)
        self.assertEqual(result["new_to_base_group_confusion_pct"], 50.0)

    def test_correlations_are_descriptive_and_tie_aware(self):
        result = pearson_spearman([(1, 3), (2, 2), (3, 1)])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["pearson"], -1.0)
        self.assertAlmostEqual(result["spearman"], -1.0)
        self.assertIsNone(pearson_spearman([(1, 2)])["pearson"])


class SharpnessMetricTests(unittest.TestCase):
    def setUp(self):
        self.model = _TextModel()
        self.index = ParamIndex.from_model(self.model)
        self.features = torch.eye(2)
        self.labels = torch.tensor([0, 1])
        self.loss_fn = fixed_feature_loss_fn(
            self.model, features=self.features, labels=self.labels
        )

    def test_cached_feature_loss_matches_manual_loss(self):
        text = self.model.text_encoder(
            self.model.prompt_learner(), self.model.tokenized_prompts
        )
        expected = F.cross_entropy(self.features @ text.t(), self.labels)
        self.assertTrue(torch.allclose(self.loss_fn(), expected))

    def test_zero_perturbation_reproduces_reference_loss(self):
        from sample_fg.perturbation import PromptPerturbation
        before = self.loss_fn().detach().clone()
        with PromptPerturbation(self.index).displaced(GradientState.zeros(self.index)):
            observed = self.loss_fn().detach().clone()
        self.assertTrue(torch.equal(before, observed))

    def test_directions_are_deterministic_global_unit_and_checkpoint_keyed(self):
        first = sample_prompt_directions(self.index, checkpoint_sha256="12" * 32)
        second = sample_prompt_directions(self.index, checkpoint_sha256="12" * 32)
        other = sample_prompt_directions(self.index, checkpoint_sha256="34" * 32)
        self.assertEqual(len(first), 32)
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left[0], right[0]))
            self.assertAlmostEqual(float(left.norm()), 1.0, places=6)
        self.assertTrue(any(not torch.equal(a[0], b[0]) for a, b in zip(first, other)))

    def test_full_probe_restores_prompt_and_uses_all_32_for_quantiles(self):
        directions = sample_prompt_directions(
            self.index, checkpoint_sha256="56" * 32
        )
        before = parameter_sha256(self.index)
        rows = probe_symmetric_loss_sharpness(
            param_index=self.index, loss_fn=self.loss_fn, directions=directions
        )
        self.assertEqual(len(rows), 32 * 3)
        self.assertEqual(parameter_sha256(self.index), before)
        for row in rows:
            self.assertLess(
                abs(row["live_displacement_norm_plus"] - row["radius"]), 1e-6
            )
        selected = [row for row in rows if row["radius"] == RADII[0]]
        summary = summarize_sharpness(
            selected, baseline_loss=float(self.loss_fn().detach()), radius=RADII[0]
        )
        self.assertEqual(summary["direction_count"], 32)
        self.assertIn("sharpness_p95", summary)

    def test_direction_sign_swap_and_structured_restoration(self):
        direction = sample_prompt_directions(
            self.index, checkpoint_sha256="78" * 32
        )[0]
        plus = probe_structured_direction(
            name="u", param_index=self.index, loss_fn=self.loss_fn,
            direction=direction,
        )
        minus = probe_structured_direction(
            name="minus_u", param_index=self.index, loss_fn=self.loss_fn,
            direction=direction.scale(-1.0),
        )
        for left, right in zip(plus, minus):
            self.assertAlmostEqual(left["delta_loss_plus"], right["delta_loss_minus"])
            self.assertAlmostEqual(left["delta_loss_minus"], right["delta_loss_plus"])

    def test_fp16_logical_radius_is_exact_and_live_quantization_is_observable(self):
        model = nn.Module()
        model.prompt_learner = nn.Module()
        model.prompt_learner.ctx = nn.Parameter(
            torch.linspace(-0.03, 0.03, 4 * 512, dtype=torch.float16).reshape(4, 512)
        )
        index = ParamIndex.from_model(model)
        direction = sample_prompt_directions(index, checkpoint_sha256="90" * 32)[0]
        before = parameter_sha256(index)
        displacement = direction.scale(RADII[0])
        self.assertLess(abs(float(displacement.norm()) - RADII[0]), 1e-7)
        from sample_fg.perturbation import PromptPerturbation
        with PromptPerturbation(index).displaced(displacement) as snapshot:
            live = torch.sqrt(sum(
                torch.sum((entry.parameter.float() - original.float()) ** 2)
                for entry, original in zip(index, snapshot)
            ))
        self.assertGreater(abs(float(live.detach()) - RADII[0]), 0.0)
        self.assertEqual(parameter_sha256(index), before)


class SourceDiscoveryTests(unittest.TestCase):
    @staticmethod
    def _complete_summary(run: Path) -> None:
        (run / "summary.json").write_text(json.dumps({
            "status": "completed",
            "run_identity": {
                "dataset": "dtd", "shots": 16, "seed": 1,
                "method_tag": "sample", "estimator_tag": "ema",
            },
            "artifacts": {"checkpoint": "checkpoints/final.pt"},
        }), encoding="utf-8")

    def test_discovers_by_structured_identity_and_reports_missing_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = make_source_run(root)
            self._complete_summary(run)
            report = discover_r2_sources(root)
            self.assertEqual(len(report.compatible), 1)
            self.assertEqual(report.compatible[0].key.method_key, "sample_ema")
            self.assertEqual(len(report.missing), 17)

    def test_duplicate_compatible_cell_is_excluded_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, _ = make_source_run(root)
            self._complete_summary(run)
            shutil.copytree(run, root / "duplicate")
            report = discover_r2_sources(root)
            self.assertEqual(len(report.compatible), 0)
            self.assertEqual(len(report.missing), 18)
            self.assertEqual(len(report.excluded), 2)


if __name__ == "__main__":
    unittest.main()
