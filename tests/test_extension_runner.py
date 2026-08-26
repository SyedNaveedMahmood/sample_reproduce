from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from sample_fg.campaign import CampaignManifest
from sample_fg.extension_runner import (
    DEFAULT_CAMPAIGN,
    build_extension_plan,
    build_parser,
    main,
    resolve_extension_method,
)
from sample_fg.full_gradient import FullGradientSweepMetadata
from sample_fg.paper_runner import (
    ScientificRunnerError,
    _update_accounting,
    dispatch_training_step,
    dry_run_report,
)
from sample_fg.results import RunAccounting, RunArtifacts, RunIdentity
from sample_fg.rng import derive_auxiliary_seed
from sample_fg.step_engine import SAMPLeStepRecord


def _metadata(purpose: str) -> FullGradientSweepMetadata:
    return FullGradientSweepMetadata(
        sample_count=8,
        micro_batch_count=2,
        configured_micro_batch_size=4,
        observed_micro_batch_sizes=(4, 4),
        forward_calls=2,
        autograd_grad_calls=2,
        mean_loss=1.0,
        elapsed_s=0.25,
        precision_mode="fp32",
        param_index_fingerprint="p",
        source_fingerprint="s",
        seed=derive_auxiliary_seed(
            protocol_seed=1,
            dataset="dtd",
            shots=16,
            config_hash="config",
            optimizer_step=0,
            purpose=purpose,
        ),
    )


def _record(*, diagnostic_issued: bool) -> SAMPLeStepRecord:
    optimization = _metadata("optimization_exact")
    event_metadata = _metadata("diagnostic") if diagnostic_issued else optimization
    estimator_result = SimpleNamespace(
        full_gradient_metadata=optimization,
    )
    event = SimpleNamespace(
        reference=SimpleNamespace(
            exact_service_query_issued=diagnostic_issued,
            full_gradient_metadata=event_metadata,
        )
    )
    return SAMPLeStepRecord(
        method="sample",
        optimizer_step=0,
        loss_current=1.0,
        loss_displaced=1.0,
        loss_sample_objective=2.0,
        batch_gradient=mock.Mock(),
        estimator_result=estimator_result,
        projection=mock.Mock(),
        sam_perturbation=mock.Mock(),
        batch_correction=mock.Mock(),
        total_displacement=mock.Mock(),
        perturbed_gradient=mock.Mock(),
        final_gradient=mock.Mock(),
        diagnostic_event=event,
        batch_gradient_norm=1.0,
        global_direction_norm=1.0,
        batch_component_norm=1.0,
        sam_perturbation_norm=0.05,
        batch_correction_norm=0.01,
        total_displacement_norm=0.05,
        perturbed_gradient_norm=1.0,
        final_gradient_norm=2.0,
        restored_before_optimizer=True,
        same_batch_object_reused=True,
        optimizer_step_result=mock.Mock(),
    )


class ExtensionRunnerTests(unittest.TestCase):
    def test_extension_method_validation_and_periodic_identity(self):
        campaign = CampaignManifest.load(DEFAULT_CAMPAIGN)
        exact = resolve_extension_method(
            "sample", "exact", None, allowed_k=campaign.allowed_k
        )
        periodic = resolve_extension_method(
            "sample", "periodic", 4, allowed_k=campaign.allowed_k
        )
        self.assertEqual(exact.estimator_tag, "exact")
        self.assertEqual(periodic.estimator_tag, "periodic-k4")
        self.assertEqual(periodic.refresh_k_steps, 4)
        for method, estimator, k in (
            ("coop", "exact", None),
            ("sample", "ema", 4),
            ("sample", "periodic", None),
            ("sample", "periodic", 3),
        ):
            with self.subTest(method=method, estimator=estimator, k=k):
                with self.assertRaises(ScientificRunnerError):
                    resolve_extension_method(
                        method, estimator, k, allowed_k=campaign.allowed_k
                    )

    def test_periodic_canonical_identity_and_path_include_task_estimator_and_k(self):
        identity = RunIdentity(
            dataset="dtd",
            shots=16,
            method_tag="sample",
            estimator_tag="periodic-k4",
            seed=1,
            utc_timestamp="20260817T000000000000Z",
            config_sha256="a" * 64,
            experiment_id="E1",
            smoke=False,
            allow_scientific_summary=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            relative = RunArtifacts(Path(temporary), identity).run_dir.relative_to(
                temporary
            )
        self.assertEqual(identity.experiment_id, "E1")
        self.assertEqual(
            relative.parts[:5],
            ("dtd", "shots_16", "sample", "periodic-k4", "seed_1"),
        )

    def test_exact_and_periodic_scientific_dispatch_use_shared_sample_step(self):
        campaign = CampaignManifest.load(DEFAULT_CAMPAIGN)
        for estimator, k in (("exact", None), ("periodic", 4)):
            with self.subTest(estimator=estimator):
                selection = resolve_extension_method(
                    "sample", estimator, k, allowed_k=campaign.allowed_k
                )
                engine = mock.Mock()
                engine.step_sample.return_value = object()
                active_estimator = mock.Mock()
                active_estimator.mode = estimator
                runtime = SimpleNamespace(
                    engine=engine,
                    estimator=active_estimator,
                )
                result = dispatch_training_step(
                    selection=selection,
                    runtime=runtime,
                    batch=object(),
                    loss_closure=mock.Mock(),
                    optimizer_step=0,
                    epoch=0,
                    batch_index=0,
                )
                self.assertIs(result, engine.step_sample.return_value)
                engine.step_sample.assert_called_once_with(
                    mock.ANY,
                    mock.ANY,
                    active_estimator,
                    epoch=0,
                    batch_index=0,
                )

    def test_task28_gates_every_cell_on_freeze(self):
        args = build_parser().parse_args(
            [
                "--task", "task28", "--dataset", "dtd", "--shots", "16",
                "--seed", "1", "--method", "sample", "--estimator", "ema",
                "--data-root", ".", "--manifest-root", ".", "--clip-cache", ".",
                "--output-root", ".",
            ]
        )
        with self.assertRaisesRegex(ScientificRunnerError, "gated"):
            build_extension_plan(args)

    def test_dry_run_main_executes_zero_training_steps(self):
        fake_plan = SimpleNamespace(
            resolved_config={"campaign": {"task": "task26"}},
            dataset="dtd",
            shots=16,
            seed=1,
            selection=SimpleNamespace(
                method="sample",
                estimator="exact",
                refresh_k_steps=None,
                method_tag="sample",
                estimator_tag="exact",
            ),
        )
        with mock.patch(
            "sample_fg.extension_runner.build_extension_plan", return_value=fake_plan
        ), mock.patch(
            "sample_fg.extension_runner.extension_dry_run_report",
            return_value={"dry_run": True, "optimizer_steps_executed": 0},
        ), mock.patch("sample_fg.extension_runner.run_scientific") as training:
            code = main(
                [
                    "--task", "task26", "--dataset", "dtd", "--shots", "16",
                    "--seed", "1", "--method", "sample", "--estimator", "exact",
                    "--data-root", ".", "--manifest-root", ".", "--clip-cache", ".",
                    "--output-root", ".", "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        training.assert_not_called()

    def test_task28_reused_ema_cannot_be_retrained_directly(self):
        fake_plan = SimpleNamespace(
            resolved_config={
                "campaign": {"task": "task28", "reuse_experiment_id": "R2"}
            }
        )
        with mock.patch(
            "sample_fg.extension_runner.build_extension_plan", return_value=fake_plan
        ), mock.patch(
            "sample_fg.extension_runner.run_scientific"
        ) as training:
            with self.assertRaisesRegex(ScientificRunnerError, "retraining"):
                main(
                    [
                        "--task", "task28", "--dataset", "dtd", "--shots", "16",
                        "--seed", "1", "--method", "sample", "--estimator", "ema",
                        "--data-root", ".", "--manifest-root", ".", "--clip-cache", ".",
                        "--output-root", ".",
                    ]
                )
        training.assert_not_called()

    def test_exact_sweep_accounting_does_not_double_count_reuse(self):
        reused = RunAccounting()
        _update_accounting(reused, _record(diagnostic_issued=False), 4)
        self.assertEqual(reused.compute_counts["exact_sweeps"], 1)
        self.assertEqual(reused.compute_counts["optimization_exact_queries"], 1)
        self.assertEqual(reused.compute_counts["reused_exact_queries"], 1)
        self.assertNotIn("diagnostic_only_exact_queries", reused.compute_counts)
        self.assertEqual(reused.full_gradient_total_s, 0.25)

        independent = RunAccounting()
        _update_accounting(independent, _record(diagnostic_issued=True), 4)
        self.assertEqual(independent.compute_counts["exact_sweeps"], 2)
        self.assertEqual(
            independent.compute_counts["diagnostic_only_exact_queries"], 1
        )
        self.assertEqual(independent.full_gradient_total_s, 0.5)

    def test_dry_run_reports_exact_and_periodic_expected_sweeps(self):
        class _Source(list):
            fingerprint = "source"

        def plan(estimator: str, k: int | None):
            return SimpleNamespace(
                experiment_id="X", dataset="dtd", shots=16, seed=1,
                selection=SimpleNamespace(
                    method="sample", estimator=estimator, refresh_k_steps=k
                ),
                epochs=2, steps_per_epoch=4, total_optimizer_steps=8,
                diagnostic_interval_steps=4,
                full_gradient_micro_batch_size=32,
                recovery_interval_epochs=1,
                resolved_config={
                    "diagnostics": {"full_gradient_interval_steps": 4},
                    "optim": {"name": "sgd", "scheduler": "cosine"},
                    "runtime": {"precision": "coop_fp16"},
                    "config_sha256": "a" * 64,
                },
                source=_Source([None] * 384),
                manifest_path=Path("manifest.json"), data_root=Path("data"),
                clip_checkpoint=Path("clip.pt"), config_path=Path("config.yaml"),
                output_root=Path("runs"),
            )

        exact = dry_run_report(plan("exact", None))
        periodic = dry_run_report(plan("periodic", 2))
        self.assertEqual(exact["protocol"]["expected_exact_sweeps"], 8)
        self.assertEqual(exact["protocol"]["expected_reused_exact_queries"], 2)
        self.assertEqual(periodic["protocol"]["expected_exact_sweeps"], 4)
        self.assertEqual(
            periodic["protocol"]["expected_periodic_refresh_count"], 4
        )


if __name__ == "__main__":
    unittest.main()
