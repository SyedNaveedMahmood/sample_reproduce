"""Focused non-training tests for the scientific paper runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sample_fg.checkpoint import (
    CheckpointProgress,
    load_scientific_checkpoint,
    save_scientific_checkpoint,
)
from sample_fg.param_index import ParamIndex
from sample_fg.paper_runner import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    ScientificRunnerError,
    _build_runtime,
    _resolved_scientific_config,
    advance_epoch_scheduler,
    build_scientific_cfg,
    dispatch_training_step,
    main,
    ordinary_coop_step,
    resolve_method,
    run_scientific,
)
from sample_fg.perturbation import PromptPerturbation
from sample_fg.precision import PrecisionController
from sample_fg.step_engine import StepEngine


class _Prompt(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Parameter(torch.tensor([0.25, -0.5]))

    def forward(self, value):
        return (self.ctx * value).sum()


class PaperRunnerTests(unittest.TestCase):
    def test_paper_cfg_resolves_fixed_protocol(self):
        cfg = build_scientific_cfg(
            dataset="dtd",
            seed=1,
            data_root=REPO_ROOT,
            output_dir=REPO_ROOT / "unused",
            config_path=DEFAULT_CONFIG,
        )
        self.assertEqual(cfg.DATASET.NUM_SHOTS, 16)
        self.assertEqual(cfg.DATALOADER.TRAIN_X.BATCH_SIZE, 32)
        self.assertEqual(cfg.DATALOADER.TEST.BATCH_SIZE, 100)
        self.assertEqual(cfg.DATALOADER.NUM_WORKERS, 8)
        self.assertEqual(cfg.TRAINER.COOP.PREC, "fp16")
        self.assertEqual(cfg.TRAINER.COOP.CTX_INIT, "a photo of a")
        self.assertEqual(cfg.OPTIM.MAX_EPOCH, 200)
        self.assertEqual(cfg.OPTIM.WARMUP_CONS_LR, 1e-5)
        self.assertEqual(cfg.TEST.FINAL_MODEL, "last_step")

    def test_primary_method_matrix_is_strict(self):
        coop = resolve_method("coop", "none")
        sam = resolve_method("sam", "none")
        sample = resolve_method("sample", "ema")
        self.assertIsNone(coop.rho)
        self.assertEqual(sam.rho, 0.05)
        self.assertEqual((sample.rho, sample.alpha, sample.ema_lambda), (0.05, 0.0015, 0.15))
        for method, estimator in (("coop", "ema"), ("sam", "ema"), ("sample", "none")):
            with self.subTest(method=method, estimator=estimator):
                with self.assertRaises(ScientificRunnerError):
                    resolve_method(method, estimator)

    def test_resolved_config_is_scientific_and_method_specific(self):
        cfg = build_scientific_cfg(
            dataset="eurosat",
            seed=2,
            data_root=REPO_ROOT,
            output_dir=REPO_ROOT / "unused",
            config_path=DEFAULT_CONFIG,
        )
        source = type(
            "Source",
            (),
            {
                "dataset": "eurosat",
                "shots": 16,
                "seed": 2,
                "manifest_path": REPO_ROOT / "manifest.json",
                "official_split_sha256": "split",
                "fewshot_cache_sha256": "cache",
                "fingerprint": "source",
                "__len__": lambda self: 80,
            },
        )()
        manifest = {
            "normal_train_loader": {
                "steps_per_epoch": 2,
                "samples_consumed_per_epoch": 64,
            }
        }
        config = _resolved_scientific_config(
            cfg=cfg,
            selection=resolve_method("sample", "ema"),
            source=source,
            manifest=manifest,
            data_root=REPO_ROOT,
            manifest_root=REPO_ROOT,
            output_root=REPO_ROOT / "runs",
            clip_checkpoint=REPO_ROOT / "ViT-B-16.pt",
            config_path=DEFAULT_CONFIG,
            experiment_id="R2",
            recovery_interval_epochs=10,
        )
        self.assertFalse(config["run"]["smoke"])
        self.assertTrue(config["smoke"]["allow_scientific_summary"])
        self.assertEqual(config["method"]["name"], "sample")
        self.assertEqual(config["estimator"]["mode"], "ema")
        self.assertEqual(config["diagnostics"]["full_gradient_interval_steps"], 2)
        self.assertEqual(config["checkpoint"]["recovery_interval_steps"], 20)
        self.assertEqual(config["runtime"]["precision"], "coop_fp16")

    def test_dispatch_keeps_three_optimizer_paths_distinct(self):
        sentinel = object()
        runtime = SimpleNamespace(
            engine=None,
            estimator=None,
            model=mock.Mock(),
            param_index=mock.Mock(),
            trainer=SimpleNamespace(optim=mock.Mock()),
            precision=mock.Mock(),
        )
        with mock.patch(
            "sample_fg.paper_runner.ordinary_coop_step", return_value=sentinel
        ) as ordinary:
            observed = dispatch_training_step(
                selection=resolve_method("coop", "none"),
                runtime=runtime,
                batch=mock.Mock(),
                loss_closure=mock.Mock(),
                optimizer_step=0,
                epoch=0,
                batch_index=0,
            )
        self.assertIs(observed, sentinel)
        ordinary.assert_called_once()

        engine = mock.Mock()
        engine.step_sam.return_value = sentinel
        runtime.engine = engine
        observed = dispatch_training_step(
            selection=resolve_method("sam", "none"),
            runtime=runtime,
            batch=mock.Mock(),
            loss_closure=mock.Mock(),
            optimizer_step=0,
            epoch=0,
            batch_index=0,
        )
        self.assertIs(observed, sentinel)
        engine.step_sam.assert_called_once()
        engine.step_sample.assert_not_called()

        runtime.estimator = mock.Mock()
        runtime.estimator.mode = "ema"
        engine.step_sample.return_value = sentinel
        observed = dispatch_training_step(
            selection=resolve_method("sample", "ema"),
            runtime=runtime,
            batch=mock.Mock(),
            loss_closure=mock.Mock(),
            optimizer_step=0,
            epoch=0,
            batch_index=0,
        )
        self.assertIs(observed, sentinel)
        engine.step_sample.assert_called_once()

    def test_scheduler_helper_advances_once_per_call(self):
        trainer = mock.Mock()
        advance_epoch_scheduler(trainer)
        trainer.update_lr.assert_called_once_with()

    def test_dry_run_main_never_calls_training(self):
        fake_plan = object()
        with mock.patch("sample_fg.paper_runner.build_plan", return_value=fake_plan), mock.patch(
            "sample_fg.paper_runner.dry_run_report", return_value={"dry_run": True}
        ), mock.patch("sample_fg.paper_runner.run_scientific") as train:
            code = main(
                [
                    "--dataset", "dtd", "--shots", "16", "--seed", "1",
                    "--method", "coop", "--estimator", "none",
                    "--data-root", ".", "--manifest-root", ".",
                    "--clip-cache", ".", "--output-root", ".", "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        train.assert_not_called()

    def test_sample_runtime_uses_full_gradient_service_public_signature(self):
        source = [object(), object()]
        plan = SimpleNamespace(
            dataset="dtd",
            seed=1,
            shots=16,
            data_root=REPO_ROOT,
            config_path=DEFAULT_CONFIG,
            clip_cache=REPO_ROOT,
            selection=resolve_method("sample", "ema"),
            steps_per_epoch=2,
            epochs=200,
            diagnostic_interval_steps=2,
            full_gradient_micro_batch_size=32,
            source=source,
            resolved_config={"config_sha256": "config"},
        )
        trainer = mock.MagicMock()
        trainer.model = mock.Mock(spec=nn.Module)
        trainer.train_loader_x = [object(), object()]
        trainer.dm.dataset.train_x = list(source)
        index = mock.MagicMock()
        index.names = ("prompt_learner.ctx",)
        index.__getitem__.return_value.shape = (4, 512)
        full_loader = mock.Mock()
        with mock.patch(
            "sample_fg.paper_runner.build_scientific_cfg",
            return_value=mock.Mock(),
        ), mock.patch("sample_fg.paper_runner.set_random_seed"), mock.patch(
            "sample_fg.paper_runner.build_coop_trainer", return_value=trainer
        ), mock.patch(
            "sample_fg.paper_runner.unwrap_model", return_value=trainer.model
        ), mock.patch(
            "sample_fg.paper_runner.audit_prompt_only_training"
        ), mock.patch(
            "sample_fg.paper_runner.ParamIndex.from_model", return_value=index
        ), mock.patch(
            "sample_fg.paper_runner.PrecisionController"
        ), mock.patch(
            "sample_fg.paper_runner.PromptPerturbation"
        ), mock.patch(
            "sample_fg.paper_runner.build_full_gradient_loader",
            return_value=full_loader,
        ), mock.patch(
            "sample_fg.paper_runner.FullGradientService"
        ) as service, mock.patch(
            "sample_fg.paper_runner.EMAEstimator"
        ), mock.patch(
            "sample_fg.paper_runner.DiagnosticCoordinator"
        ), mock.patch(
            "sample_fg.paper_runner.StepEngine"
        ):
            runtime = _build_runtime(plan, REPO_ROOT / "unused")

        self.assertIs(runtime.full_gradient_loader, full_loader)
        self.assertNotIn("source", service.call_args.kwargs)
        self.assertEqual(service.call_args.kwargs["dataset"], "dtd")
        self.assertEqual(service.call_args.kwargs["shots"], 16)

    def test_startup_failure_is_persisted_before_training(self):
        artifacts = mock.MagicMock()
        artifacts.run_dir = REPO_ROOT / "unused"
        plan = SimpleNamespace(resume_from=None)
        with mock.patch(
            "sample_fg.paper_runner.torch.cuda.is_available",
            return_value=True,
        ), mock.patch(
            "sample_fg.paper_runner._new_run", return_value=(artifacts, {})
        ), mock.patch(
            "sample_fg.paper_runner._build_runtime",
            side_effect=TypeError("startup broke"),
        ):
            with self.assertRaisesRegex(TypeError, "startup broke"):
                run_scientific(plan)

        self.assertIn(
            "run failed during startup", artifacts.append_log.call_args.args[0]
        )
        summary = artifacts.write_summary.call_args.args[0]
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(
            summary["failure"],
            {"type": "TypeError", "message": "startup broke"},
        )
        self.assertEqual(summary["invariants"]["optimizer_steps"], 0)

    def test_task21_checkpoint_supports_coop_without_fake_estimator(self):
        first = _Prompt()
        index = ParamIndex.from_model(first)
        optimizer = torch.optim.SGD(index.parameters, lr=0.02, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        precision = PrecisionController("fp32")
        ordinary_coop_step(
            model=first,
            batch=(torch.tensor([0.5, -0.25]), torch.tensor(0.0)),
            loss_closure=lambda batch: (first(batch[0]) - batch[1]).square(),
            param_index=index,
            optimizer=optimizer,
            precision=precision,
            optimizer_step=0,
        )
        scheduler.step()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coop.pt"
            save_scientific_checkpoint(
                path,
                param_index=index,
                optimizer=optimizer,
                scheduler=scheduler,
                precision_controller=precision,
                step_engine=None,
                estimator=None,
                perturbation=None,
                progress=CheckpointProgress(1, 1, 0, 1),
                method="coop",
                config_sha256="config",
                source_fingerprint="source",
                result_state={"metric_records": 1},
            )
            resumed = _Prompt()
            resumed_index = ParamIndex.from_model(resumed)
            resumed_optimizer = torch.optim.SGD(
                resumed_index.parameters, lr=0.02, momentum=0.9
            )
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1
            )
            result = load_scientific_checkpoint(
                path,
                param_index=resumed_index,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                precision_controller=PrecisionController("fp32"),
                step_engine=None,
                estimator=None,
                perturbation=None,
                expected_method="coop",
                expected_config_sha256="config",
                expected_source_fingerprint="source",
            )
        self.assertEqual(result.metadata.estimator_mode, "none")
        self.assertEqual(result.progress.epoch_zero_based, 1)
        self.assertTrue(torch.equal(first.ctx, resumed.ctx))

    def test_task21_checkpoint_supports_sam_without_fake_estimator(self):
        first = _Prompt()
        index = ParamIndex.from_model(first)
        optimizer = torch.optim.SGD(index.parameters, lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        precision = PrecisionController("fp32")
        perturbation = PromptPerturbation(index)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=precision,
            rho=0.05,
            perturbation=perturbation,
        )
        batch = (torch.tensor([0.5, -0.25]), torch.tensor(0.0))
        engine.step_sam(batch, lambda item: (first(item[0]) - item[1]).square())
        scheduler.step()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sam.pt"
            save_scientific_checkpoint(
                path,
                param_index=index,
                optimizer=optimizer,
                scheduler=scheduler,
                precision_controller=precision,
                step_engine=engine,
                estimator=None,
                perturbation=perturbation,
                progress=CheckpointProgress(1, 1, 0, 1),
                method="sam",
                config_sha256="config",
                source_fingerprint="source",
                result_state={},
            )
            resumed = _Prompt()
            resumed_index = ParamIndex.from_model(resumed)
            resumed_optimizer = torch.optim.SGD(resumed_index.parameters, lr=0.02)
            resumed_scheduler = torch.optim.lr_scheduler.StepLR(
                resumed_optimizer, step_size=1
            )
            resumed_precision = PrecisionController("fp32")
            resumed_perturbation = PromptPerturbation(resumed_index)
            resumed_engine = StepEngine(
                param_index=resumed_index,
                optimizer=resumed_optimizer,
                precision_controller=resumed_precision,
                rho=0.05,
                perturbation=resumed_perturbation,
            )
            result = load_scientific_checkpoint(
                path,
                param_index=resumed_index,
                optimizer=resumed_optimizer,
                scheduler=resumed_scheduler,
                precision_controller=resumed_precision,
                step_engine=resumed_engine,
                estimator=None,
                perturbation=resumed_perturbation,
                expected_method="sam",
                expected_config_sha256="config",
                expected_source_fingerprint="source",
            )
        self.assertEqual(result.metadata.estimator_mode, "none")
        self.assertEqual(resumed_engine.optimizer_step, 1)


if __name__ == "__main__":
    unittest.main()
