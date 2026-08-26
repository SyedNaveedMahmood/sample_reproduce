from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from analysis.aggregate_low_compute import build_primary_rows, write_primary_table
from analysis.plot_low_compute import plot_saved_artifacts
from sample_fg.checkpoint import CHECKPOINT_SCHEMA_VERSION
from sample_fg.coop_anchor import EXPECTED_CLIP_KEY, EXPECTED_CLIP_SHA256
from sample_fg.gradient_state import GradientState
from sample_fg.low_compute.artifacts import (
    CONFIG_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    LowComputeArtifactError,
    LowComputeArtifacts,
    validate_saved_artifacts,
)
from sample_fg.low_compute.budget import ComputeBudget, ComputeBudgetError, TransitionGuard
from sample_fg.low_compute.checkpoint_probe import (
    ProbeCheckpointError,
    load_probe_checkpoint,
    sha256_file,
    verify_source_immutable,
)
from sample_fg.low_compute.functional_probe import (
    FunctionalProbeError,
    _comparison_rows,
    _directional_response,
    _objects,
    compare_functional_directions,
)
from sample_fg.low_compute.feature_cache import (
    FeatureCacheError,
    FeatureCacheKey,
    load_feature_cache,
    materialization_seed_clock,
    save_feature_cache,
)
from sample_fg.low_compute.gradient_bank import build_gradient_bank, weighted_exact
from sample_fg.low_compute.math import flatten_state
from sample_fg.low_compute.planner import build_integrated_plan
from sample_fg.low_compute.replay import (
    effective_sample_size,
    ema_replay,
    evaluate_lc02_gate,
    history_length,
    permutation_trials,
    projection_displacement_metrics,
    stationary_ema_replay,
)
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.projection import project_batch_gradient
from sample_fg.results import resolve_config


REPO_ROOT = Path(__file__).resolve().parents[1]


class ToyPrompt(nn.Module):
    def __init__(self, values=(0.2, -0.1)):
        super().__init__()
        self.prompt_learner = nn.Module()
        self.prompt_learner.ctx = nn.Parameter(torch.tensor(values, dtype=torch.float32))


def state(index: ParamIndex, values) -> GradientState:
    return GradientState.from_tensors(index, (torch.tensor(values, dtype=torch.float32),))


def scientific_checkpoint_payload(index: ParamIndex, config_hash: str = "config-hash") -> dict:
    active = state(index, (0.3, -0.2)).state_dict()
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_utc": "2026-08-17T00:00:00Z",
        "boundary": "after_logical_optimizer_step_unperturbed_v1",
        "method": "sample",
        "estimator_mode": "ema",
        "config_sha256": config_hash,
        "source_fingerprint": "source-fingerprint",
        "param_index": index.to_metadata(),
        "trainable_model_state": {
            "prompt_learner.ctx": index[0].parameter.detach().clone()
        },
        "optimizer_state": {},
        "scheduler_state": {},
        "precision_state": {},
        "step_engine_state": {},
        "estimator_state": {
            "schema_version": "sample_fg.estimator_state.v1",
            "mode": "ema",
            "param_index_fingerprint_schema": index.fingerprint_schema,
            "param_index_fingerprint": index.fingerprint,
            "last_processed_step": 239,
            "exact_query_count": 0,
            "ema_lambda": 0.15,
            "active_state": active,
        },
        "progress": {
            "next_optimizer_step": 240,
            "epoch_zero_based": 20,
            "next_batch_index_zero_based": 0,
            "normal_samples_seen": 7680,
        },
        "rng_state": {},
        "normal_loader_state": None,
        "result_state": {},
        "gradient_buffer_policy": "not_serialized_safe_step_boundary",
        "legacy_compatibility": {},
    }


def make_source_run(root: Path, *, epochs=(20, 60, 100, 140, 200)) -> tuple[Path, ParamIndex]:
    run = root / "source"
    (run / "checkpoints").mkdir(parents=True)
    config = {
        "data": {
            "dataset": "dtd", "shots": 16, "seed": 1,
            "selected_source_fingerprint": "source-fingerprint",
            "selected_count": 384, "train_batch_size": 32,
        },
        "run": {"experiment_id": "R2", "smoke": False},
        "smoke": {"allow_scientific_summary": True},
        "method": {
            "name": "sample", "rho": 0.05, "alpha": 0.0015,
            "ema_lambda": 0.15,
        },
        "estimator": {"mode": "ema"},
        "model": {
            "backbone": EXPECTED_CLIP_KEY,
            "checkpoint_sha256": EXPECTED_CLIP_SHA256,
        },
    }
    config = resolve_config(config)
    import yaml
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "data_manifest.json").write_text(
        json.dumps({"schema_version": "sample_fg.data_manifest.v1"}), encoding="utf-8"
    )
    (run / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    model = ToyPrompt()
    index = ParamIndex.from_model(model)
    for epoch in epochs:
        payload = scientific_checkpoint_payload(index, config["config_sha256"])
        payload["progress"]["epoch_zero_based"] = epoch
        payload["progress"]["next_optimizer_step"] = epoch * 12
        payload["estimator_state"]["last_processed_step"] = epoch * 12 - 1
        name = "final.pt" if epoch == 200 else f"recovery_step_{epoch * 12:06d}.pt"
        torch.save(payload, run / "checkpoints" / name)
    return run, index


class ReplayAndBankTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyPrompt()
        self.index = ParamIndex.from_model(self.model)
        self.gradients = (state(self.index, (1.0, 0.0)), state(self.index, (0.0, 2.0)))

    def test_replay_matches_direct_recurrence_and_lambda_zero(self):
        observed = flatten_state(ema_replay(self.gradients, 0.15))
        expected = 0.15 * (0.85 * torch.tensor([1.0, 0.0])) + 0.85 * torch.tensor([0.0, 2.0])
        self.assertTrue(torch.allclose(observed, expected))
        self.assertTrue(torch.equal(flatten_state(ema_replay(self.gradients, 0.0)), torch.tensor([0.0, 2.0])))
        self.assertAlmostEqual(1.0 - 0.15**3, 0.996625)

    def test_replay_accumulator_uses_stored_gradient_device(self):
        model = ToyPrompt().to(device="meta")
        index = ParamIndex.from_model(model)
        gradients = (
            state(index, (1.0, 0.0)),
            state(index, (0.0, 2.0)),
        )
        self.assertEqual(index[0].parameter.device, torch.device("meta"))
        self.assertEqual(gradients[0].devices, (torch.device("cpu"),))

        replayed = ema_replay(gradients, 0.15)
        stationary, _ = stationary_ema_replay(
            gradients, 0.15, epochs=2, seed=7
        )

        self.assertEqual(replayed.devices, (torch.device("cpu"),))
        self.assertEqual(stationary.devices, (torch.device("cpu"),))
        expected = (
            0.15 * (0.85 * torch.tensor([1.0, 0.0]))
            + 0.85 * torch.tensor([0.0, 2.0])
        )
        self.assertTrue(torch.allclose(flatten_state(replayed), expected))

    def test_ess_history_and_deterministic_stationary_replay(self):
        self.assertAlmostEqual(effective_sample_size(0.6), 4.0)
        self.assertEqual(history_length(0.0, 0.99), 1)
        first = stationary_ema_replay(self.gradients, 0.6, epochs=20, seed=7)
        second = stationary_ema_replay(self.gradients, 0.6, epochs=20, seed=7)
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(flatten_state(first[0]), flatten_state(second[0])))

    def test_exact_is_order_invariant_but_ema_is_order_sensitive(self):
        def batch(index, ids, count, gradient):
            from sample_fg.low_compute.gradient_bank import GradientBatch, gradient_sha256
            return GradientBatch(index, ids, count, 0.0, gradient, gradient_sha256(gradient))
        batches = (
            batch(0, ("a",), 1, self.gradients[0]),
            batch(1, ("b", "c", "d"), 3, self.gradients[1]),
        )
        exact = weighted_exact(batches)
        reverse = weighted_exact(tuple(reversed(batches)))
        self.assertTrue(torch.equal(flatten_state(exact), flatten_state(reverse)))
        forward_ema = ema_replay(self.gradients, 0.6)
        reverse_ema = ema_replay(self.gradients, 0.6, order=(1, 0))
        self.assertFalse(torch.equal(flatten_state(forward_ema), flatten_state(reverse_ema)))

    def test_permutation_replay_is_deterministic(self):
        exact = state(self.index, (0.5, 1.0))
        first = permutation_trials(self.gradients, exact, ema_lambda=0.15, trial_count=20, seed=9)
        second = permutation_trials(self.gradients, exact, ema_lambda=0.15, trial_count=20, seed=9)
        self.assertEqual(first, second)

    def test_toy_gradient_bank_matches_full_mean_loss_gradient(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
        y = torch.tensor([1.0, -1.0, 0.5])
        batches = []
        for indices in ((0, 1), (2,)):
            ids = tuple(str(i) for i in indices)
            def closure(indices=indices):
                prediction = x[list(indices)] @ self.model.prompt_learner.ctx
                return ((prediction - y[list(indices)]) ** 2).mean()
            batches.append((ids, len(ids), closure))
        bank = build_gradient_bank(
            param_index=self.index, materialized_batches=batches, materialization_replicate=0
        )
        full = ((x @ self.model.prompt_learner.ctx - y) ** 2).mean()
        direct = torch.autograd.grad(full, self.index.parameters)[0]
        self.assertTrue(torch.allclose(flatten_state(bank.exact), direct, atol=1e-6))
        self.assertIsNone(self.index[0].parameter.grad)

    def test_projection_and_displacement_use_canonical_geometry(self):
        estimate = state(self.index, (1.0, 1.0))
        exact = state(self.index, (0.5, 1.5))
        observed = projection_displacement_metrics(self.gradients[0], estimate, exact, rho=0.05, alpha=0.0015)
        canonical_est = project_batch_gradient(self.gradients[0], estimate)
        canonical_exact = project_batch_gradient(self.gradients[0], exact)
        dot = float(canonical_est.batch_component.dot(canonical_exact.batch_component).item())
        norms = float(canonical_est.batch_component.norm().item() * canonical_exact.batch_component.norm().item())
        self.assertAlmostEqual(observed["gB"]["cosine"], dot / norms, places=6)

    def test_gate_rejects_accuracy_fields(self):
        with self.assertRaises(Exception):
            evaluate_lc02_gate([{"checkpoint": 1, "lambda": 0.15, "hm": 50.0}])


class CheckpointBudgetArtifactTests(unittest.TestCase):
    def test_probe_checkpoint_is_read_only_and_missing_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, index = make_source_run(Path(temporary), epochs=(20,))
            checkpoint = next((run / "checkpoints").glob("*.pt"))
            before = {path: sha256_file(path) for path in run.rglob("*") if path.is_file()}
            probe = load_probe_checkpoint(run, checkpoint)
            probe.install_prompt(index)
            self.assertTrue(torch.equal(flatten_state(probe.actual_ema(index)), torch.tensor([0.3, -0.2])))
            verify_source_immutable(probe)
            after = {path: sha256_file(path) for path in run.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            with self.assertRaises((FileNotFoundError, ProbeCheckpointError)):
                load_probe_checkpoint(run, run / "checkpoints" / "missing.pt")

    def test_planner_resolves_budget_96_and_zero_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, _ = make_source_run(Path(temporary))
            plan = build_integrated_plan(
                source_run=run,
                config_path=REPO_ROOT / "configs" / "sample_fg" / "low_compute_campaign.yaml",
            )
            self.assertEqual([item.epoch for item in plan.checkpoints], [20, 60, 100, 140, 200])
            self.assertEqual(plan.budget.backward_batches, 96)
            self.assertEqual(plan.budget.optimizer_steps, 0)
            self.assertEqual(plan.budget.scheduler_steps, 0)
            self.assertEqual(plan.order_trials, 512)
            self.assertEqual(plan.radii, (0.0025, 0.005))
            report = plan.as_dict()
            self.assertEqual(report["status"], "DRY_RUN_VALIDATED")
            self.assertEqual(report["gradient_source"]["selected_examples"], 384)
            self.assertEqual(report["budget"]["exact_sweeps"], 1)
            self.assertEqual(report["budget"]["normal_backward_batches"], 84)
            self.assertEqual(report["budget"]["exact_backward_batches"], 12)
            self.assertEqual(report["budget"]["optimizer_steps"], 0)
            self.assertEqual(report["budget"]["image_encoder_forward_batches"], 116)
            self.assertEqual(report["budget"]["text_encoder_forward_calls"], 176)

    def test_budget_overflow_and_transition_mutation_fail(self):
        requested = ComputeBudget(normal_backward_batches=97)
        permit = ComputeBudget(normal_backward_batches=96)
        with self.assertRaises(ComputeBudgetError):
            requested.assert_within(permit)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        with self.assertRaises(ComputeBudgetError):
            with TransitionGuard(optimizer):
                optimizer.param_groups[0]["lr"] = 0.2

    def test_strict_artifacts_and_summary_safety(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "probe"
            artifacts = LowComputeArtifacts(run)
            artifacts.create(
                config={"schema_version": CONFIG_SCHEMA_VERSION},
                environment={"fixture": True},
                source={"schema_version": SOURCE_SCHEMA_VERSION},
                budget=ComputeBudget().as_dict(),
            )
            artifacts.append_metric(
                {"schema_version": METRICS_SCHEMA_VERSION, "task": "lc01", "value": 1.0}
            )
            with self.assertRaises(LowComputeArtifactError):
                artifacts.append_metric(
                    {"schema_version": METRICS_SCHEMA_VERSION, "task": "lc01", "value": math.nan}
                )
            artifacts.write_summary(
                {
                    "schema_version": SUMMARY_SCHEMA_VERSION,
                    "safety": {
                        "optimizer_steps_executed": 0,
                        "scheduler_steps_executed": 0,
                        "model_parameters_changed": False,
                    },
                }
            )
            validate_saved_artifacts(run)

    def test_feature_cache_key_order_invariance_and_corruption_guard(self):
        key = FeatureCacheKey("dtd", "train", "clip", "transform", "checkpoint", 0)
        changed = FeatureCacheKey("dtd", "train", "clip", "transform", "checkpoint-2", 0)
        self.assertNotEqual(key.digest, changed.digest)
        seeds = {
            sample_id: materialization_seed_clock("checkpoint", 0, sample_id)
            for sample_id in ("a", "b", "c")
        }
        reverse = {
            sample_id: materialization_seed_clock("checkpoint", 0, sample_id)
            for sample_id in reversed(("a", "b", "c"))
        }
        self.assertEqual(seeds, reverse)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.pt"
            features = torch.arange(6, dtype=torch.float32).reshape(3, 2)
            labels = torch.tensor([0, 1, 2])
            save_feature_cache(
                path, key=key, features=features, labels=labels,
                sample_ids=("a", "b", "c"),
            )
            loaded = load_feature_cache(path, expected_key=key)
            self.assertTrue(torch.equal(loaded[0], features))
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["features"][0, 0] += 1
            torch.save(payload, path)
            with self.assertRaises(FeatureCacheError):
                load_feature_cache(path, expected_key=key)


class FunctionalProbeTests(unittest.TestCase):
    def setUp(self):
        self.model = ToyPrompt()
        self.index = ParamIndex.from_model(self.model)
        self.images = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.2, 0.8]])
        self.labels = torch.tensor([0, 1, 2, 3])

    def text(self):
        p = self.model.prompt_learner.ctx
        fixed = torch.tensor([[1.0, 0.2], [0.3, 1.0], [-0.7, 0.6], [0.5, -0.8]])
        return fixed + torch.stack((p, p.flip(0), -p, torch.stack((p[1], -p[0]))))

    def test_identical_directions_restore_prompt_and_emit_both_radii(self):
        direction = state(self.index, (1.0, 0.4))
        before = self.model.prompt_learner.ctx.detach().clone()
        rows = compare_functional_directions(
            param_index=self.index,
            ema_direction=direction,
            exact_direction=direction.clone(),
            radii=(0.0025, 0.005),
            text_feature_fn=self.text,
            eval_image_features=self.images,
            eval_labels=self.labels,
            base_class_count=2,
            logit_scale=2.0,
        )
        self.assertEqual([row["radius"] for row in rows], [0.0025, 0.005])
        self.assertGreater(rows[0]["function_space"]["text_all"]["cosine"], 0.99999)
        self.assertTrue(torch.equal(before, self.model.prompt_learner.ctx))

    def test_central_difference_is_odd_and_orthogonal_directions_differ(self):
        first = state(self.index, (1.0, 0.0))
        second = state(self.index, (0.0, 1.0))
        perturbation = PromptPerturbation(self.index)
        kwargs = dict(
            radius=0.005,
            perturbation=perturbation,
            text_feature_fn=self.text,
            eval_image_features=self.images,
            eval_labels=self.labels,
            logit_scale=2.0,
        )
        response = _directional_response(direction=first, **kwargs)
        negative = _directional_response(direction=first.scale(-1.0), **kwargs)
        self.assertTrue(torch.allclose(response["text"], -negative["text"], atol=2e-5))
        p = self.model.prompt_learner.ctx.detach()
        fixed = torch.tensor([[1.0, 0.2], [0.3, 1.0], [-0.7, 0.6], [0.5, -0.8]])
        matrices = torch.stack(
            (
                torch.eye(2),
                torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
                -torch.eye(2),
                torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
            )
        )
        raw = fixed + torch.stack((p, p.flip(0), -p, torch.stack((p[1], -p[0]))))
        analytic = []
        unit = torch.tensor([1.0, 0.0])
        for value, matrix in zip(raw, matrices):
            norm = torch.linalg.vector_norm(value)
            normalized = value / norm
            tangent = matrix @ unit
            analytic.append((tangent - normalized * torch.dot(normalized, tangent)) / norm)
        self.assertTrue(torch.allclose(response["text"], torch.stack(analytic), atol=3e-4))
        rows = compare_functional_directions(
            param_index=self.index, ema_direction=first, exact_direction=second,
            radii=(0.0025, 0.005), text_feature_fn=self.text,
            eval_image_features=self.images, eval_labels=self.labels,
            base_class_count=2, logit_scale=2.0,
        )
        self.assertLess(rows[0]["parameter_space"]["cosine"], 0.01)

    def test_cached_logit_formula_matches_direct(self):
        text = self.text()
        text = text / text.norm(dim=-1, keepdim=True)
        objects = _objects(
            text, eval_image_features=self.images, eval_labels=self.labels, logit_scale=2.0
        )
        normalized_images = self.images / self.images.norm(dim=-1, keepdim=True)
        self.assertTrue(torch.allclose(objects["logits"], 2.0 * normalized_images @ text.t()))

    def test_base_new_indexing_uses_the_declared_contiguous_boundary(self):
        ema = {
            "text": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
            "topology": torch.eye(4),
            "logits": torch.ones(4, 4),
            "margins": torch.ones(4),
        }
        exact = {
            "text": torch.tensor([[1.0], [2.0], [-3.0], [-4.0]]),
            "topology": torch.eye(4),
            "logits": torch.ones(4, 4),
            "margins": torch.ones(4),
        }
        compared = _comparison_rows(
            ema, exact, base_class_count=2, eval_labels=torch.tensor([0, 1, 2, 3])
        )
        self.assertAlmostEqual(compared["text_base"]["cosine"], 1.0)
        self.assertAlmostEqual(compared["text_new"]["cosine"], -1.0)

    def test_radii_cannot_be_silently_selected(self):
        direction = state(self.index, (1.0, 0.0))
        with self.assertRaises(FunctionalProbeError):
            compare_functional_directions(
                param_index=self.index, ema_direction=direction, exact_direction=direction,
                radii=(0.005,), text_feature_fn=self.text,
                eval_image_features=self.images, eval_labels=self.labels,
                base_class_count=2, logit_scale=2.0,
            )


class SavedAnalysisTests(unittest.TestCase):
    def test_plot_cli_can_be_launched_as_a_script(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "analysis" / "plot_low_compute.py"),
                "--help",
            ],
            cwd=REPO_ROOT.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--run-dir", completed.stdout)

    def test_table_and_plots_regenerate_from_saved_scalars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            (root / "lc01").mkdir(parents=True)
            (root / "lc04").mkdir()
            replay = {
                "rows": [
                    {
                        "checkpoint_sha256": "a" * 64, "epoch": 100,
                        "materialization_replicate": 0, "lambda": 0.15,
                        "effective_sample_size": effective_sample_size(0.15),
                        "canonical_order": {"cosine": 0.4, "relative_l2": 0.8, "norm_ratio": 1.2},
                        "order_cosine": {"sd": 0.03},
                    }
                ]
            }
            geometry = {
                "rows": [
                    {
                        "checkpoint_sha256": "a" * 64, "epoch": 100,
                        "materialization_replicate": 0, "lambda": 0.15,
                        "gB_exact_cosine_mean": 0.9,
                        "delta_exact_cosine_mean": 0.95,
                    }
                ]
            }
            (root / "lc01" / "replay_summary.json").write_text(json.dumps(replay), encoding="utf-8")
            (root / "lc01" / "geometry_summary.json").write_text(json.dumps(geometry), encoding="utf-8")
            (root / "lc01" / "compute_accounting.json").write_text(
                json.dumps({"exact_gradient_gpu_wall_s": 1.5}), encoding="utf-8"
            )
            import yaml
            (root / "config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "paper_constants": {"ema_lambda": 0.15},
                        "lc01": {"coverage_lambda": 0.8461538461538461},
                    }
                ),
                encoding="utf-8",
            )
            function = {
                "checkpoint_sha256": "a" * 64, "epoch": 100,
                "materialization_replicate": 0, "radius": 0.005,
                "parameter_space": {"cosine": 0.35},
                "function_space": {
                    "text_all": {"cosine": 0.8}, "logits_all": {"cosine": 0.85}
                },
            }
            (root / "lc04" / "function_space_fidelity.jsonl").write_text(
                json.dumps(function) + "\n", encoding="utf-8"
            )
            rows = build_primary_rows(root)
            self.assertEqual(len(rows), 1)
            output = Path(temporary) / "analysis"
            table_paths = write_primary_table(root, output)
            plot_paths = plot_saved_artifacts(root, output)
            self.assertTrue(all(path.is_file() for path in (*table_paths, *plot_paths)))


if __name__ == "__main__":
    unittest.main()
