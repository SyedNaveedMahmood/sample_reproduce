"""Task-21 scientific checkpoint/resume tests."""

from __future__ import annotations

import copy
import math
import random
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sample_fg.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointProgress,
    load_scientific_checkpoint,
    save_scientific_checkpoint,
)
from sample_fg.estimators import EMAEstimator, ExactEstimator, PeriodicEstimator
from sample_fg.full_gradient import FullGradientResult, FullGradientSweepMetadata
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.precision import PrecisionController
from sample_fg.rng import capture_rng_state, derive_auxiliary_seed
from sample_fg.step_engine import StepEngine


class _Prompt(nn.Module):
    def __init__(self, name: str = "ctx", size: int = 2):
        super().__init__()
        values = torch.linspace(0.25, -0.5, size)
        self.register_parameter(name, nn.Parameter(values))
        self._name = name

    @property
    def parameter(self):
        return getattr(self, self._name)

    def forward(self, value):
        return (self.parameter * value).sum()


class _ExactService:
    def __init__(self, index: ParamIndex):
        self.index = index
        self.calls: list[tuple[int, str]] = []

    def compute(self, *, optimizer_step, purpose):
        self.calls.append((optimizer_step, purpose))
        value = torch.tensor(
            [0.25 + optimizer_step * 0.01, -0.15 + optimizer_step * 0.005],
            device=self.index[0].parameter.device,
        )
        gradient = GradientState.from_tensors(self.index, (value,))
        seed = derive_auxiliary_seed(
            protocol_seed=1,
            dataset="fixture",
            shots=1,
            config_hash="checkpoint-fixture-v1",
            optimizer_step=optimizer_step,
            purpose=purpose,
        )
        metadata = FullGradientSweepMetadata(
            sample_count=2,
            micro_batch_count=1,
            configured_micro_batch_size=2,
            observed_micro_batch_sizes=(2,),
            forward_calls=1,
            autograd_grad_calls=1,
            mean_loss=0.5,
            elapsed_s=0.0,
            precision_mode="fp32",
            param_index_fingerprint=self.index.fingerprint,
            source_fingerprint="synthetic-source-v1",
            seed=seed,
        )
        return FullGradientResult(gradient=gradient, metadata=metadata)


@dataclass
class _Runtime:
    model: _Prompt
    index: ParamIndex
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.StepLR
    precision: PrecisionController
    perturbation: PromptPerturbation
    engine: StepEngine
    estimator: object
    generator: torch.Generator


def _runtime(
    mode: str, *, name: str = "ctx", size: int = 2,
    ema_lambda=0.15, k=2,
) -> _Runtime:
    model = _Prompt(name, size)
    index = ParamIndex.from_model(model)
    optimizer = torch.optim.SGD(index.parameters, lr=0.02, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    precision = PrecisionController("fp32")
    perturbation = PromptPerturbation(index)
    engine = StepEngine(
        param_index=index,
        optimizer=optimizer,
        precision_controller=precision,
        rho=0.05,
        alpha=0.0015,
        perturbation=perturbation,
    )
    service = _ExactService(index)
    if mode == "ema":
        estimator = EMAEstimator(index, ema_lambda=ema_lambda)
    elif mode == "exact":
        estimator = ExactEstimator(index, full_gradient_service=service)
    elif mode == "periodic":
        estimator = PeriodicEstimator(
            index,
            ema_lambda=ema_lambda,
            refresh_k_steps=k,
            full_gradient_service=service,
        )
    else:
        raise AssertionError(mode)
    generator = torch.Generator().manual_seed(90210)
    return _Runtime(
        model, index, optimizer, scheduler, precision, perturbation,
        engine, estimator, generator,
    )


def _seed_normal() -> None:
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)


def _one_step(runtime: _Runtime) -> dict[str, object]:
    x = torch.tensor(
        [
            random.random() + float(np.random.rand()),
            float(torch.rand(())) + float(torch.rand((), generator=runtime.generator)),
        ]
    )
    target = torch.tensor(0.125)

    def closure(batch):
        prediction = runtime.model(batch[0])
        return (prediction - batch[1]).square()

    record = runtime.engine.step_sample((x, target), closure, runtime.estimator)
    runtime.scheduler.step()
    return {
        "step": record.optimizer_step,
        "loss": record.loss_current,
        "grad_norm": record.batch_gradient_norm,
        "refresh": record.estimator_result.refreshed,
        "query_count": record.estimator_result.exact_query_count,
    }


def _one_fixed_step(runtime: _Runtime, offset: float) -> None:
    x = torch.tensor([0.5 + offset, -0.25])
    target = torch.tensor(0.125)
    runtime.engine.step_sample(
        (x, target),
        lambda batch: (runtime.model(batch[0]) - batch[1]).square(),
        runtime.estimator,
    )
    runtime.scheduler.step()


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _save(
    path: Path,
    runtime: _Runtime,
    step: int,
    *,
    explicit_generators=None,
    **kwargs,
):
    return save_scientific_checkpoint(
        path,
        param_index=runtime.index,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        precision_controller=runtime.precision,
        step_engine=runtime.engine,
        estimator=runtime.estimator,
        perturbation=runtime.perturbation,
        progress=CheckpointProgress(step, 0, step, step),
        method="sample",
        config_sha256="config-v1",
        source_fingerprint="source-v1",
        result_state={"optimizer_steps": step, "records_written": step},
        explicit_generators=(
            {"normal": runtime.generator}
            if explicit_generators is None
            else explicit_generators
        ),
        **kwargs,
    )


def _load(path: Path, runtime: _Runtime, *, explicit_generators=None, **kwargs):
    return load_scientific_checkpoint(
        path,
        param_index=runtime.index,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        precision_controller=runtime.precision,
        step_engine=runtime.engine,
        estimator=runtime.estimator,
        perturbation=runtime.perturbation,
        expected_method="sample",
        expected_config_sha256="config-v1",
        expected_source_fingerprint="source-v1",
        explicit_generators=(
            {"normal": runtime.generator}
            if explicit_generators is None
            else explicit_generators
        ),
        **kwargs,
    )


class _StochasticDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return (
            index,
            random.random(),
            float(np.random.rand()),
            float(torch.rand(())),
        )


class CheckpointTests(unittest.TestCase):
    def _split_run(self, mode: str, split: int):
        _seed_normal()
        control = _runtime(mode)
        control_records = [_one_step(control) for _ in range(6)]
        control_rng = capture_rng_state((control.generator,))
        control_final = {
            "parameter": control.model.parameter.detach().clone(),
            "optimizer": copy.deepcopy(control.optimizer.state_dict()),
            "scheduler": copy.deepcopy(control.scheduler.state_dict()),
            "precision": copy.deepcopy(control.precision.state_dict()),
            "estimator": copy.deepcopy(control.estimator.state_dict()),
            "engine": copy.deepcopy(control.engine.state_dict()),
            "rng": control_rng,
        }

        _seed_normal()
        first = _runtime(mode)
        first_records = [_one_step(first) for _ in range(split)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recovery.pt"
            metadata = _save(path, first, split)
            self.assertGreater(metadata.byte_size, 0)
            resumed = _runtime(mode)
            result = _load(path, resumed)
            self.assertEqual(result.progress.next_optimizer_step, split)
            self.assertEqual(result.result_state["optimizer_steps"], split)
            resumed_records = [_one_step(resumed) for _ in range(split, 6)]
        observed_records = first_records + resumed_records
        self.assertEqual(control_records, observed_records)
        self.assertTrue(torch.equal(control_final["parameter"], resumed.model.parameter))
        self.assertTrue(_nested_equal(control_final["optimizer"], resumed.optimizer.state_dict()))
        self.assertTrue(_nested_equal(control_final["scheduler"], resumed.scheduler.state_dict()))
        self.assertTrue(_nested_equal(control_final["precision"], resumed.precision.state_dict()))
        self.assertTrue(_nested_equal(control_final["estimator"], resumed.estimator.state_dict()))
        self.assertTrue(_nested_equal(control_final["engine"], resumed.engine.state_dict()))
        resumed_rng = capture_rng_state((resumed.generator,))
        self.assertTrue(_nested_equal(control_final["rng"].python_state, resumed_rng.python_state))
        self.assertTrue(np.array_equal(control_final["rng"].numpy_state[1], resumed_rng.numpy_state[1]))
        self.assertTrue(torch.equal(control_final["rng"].torch_cpu_state, resumed_rng.torch_cpu_state))
        self.assertTrue(torch.equal(control_final["rng"].explicit_generators[0].state, resumed_rng.explicit_generators[0].state))
        return observed_records

    def test_uninterrupted_equals_resumed_for_all_estimators(self):
        for mode in ("ema", "exact", "periodic"):
            with self.subTest(mode=mode):
                records = self._split_run(mode, 3)
                self.assertEqual([item["step"] for item in records], list(range(6)))

    def test_periodic_resume_around_both_k2_boundaries(self):
        after_refresh = self._split_run("periodic", 1)
        after_nonrefresh = self._split_run("periodic", 2)
        expected_refresh = [True, False, True, False, True, False]
        self.assertEqual([item["refresh"] for item in after_refresh], expected_refresh)
        self.assertEqual([item["refresh"] for item in after_nonrefresh], expected_refresh)
        self.assertEqual([item["query_count"] for item in after_refresh], [1, 1, 2, 2, 3, 3])

    def test_method_config_source_and_paramindex_mismatches_fail(self):
        _seed_normal()
        runtime = _runtime("ema")
        _one_step(runtime)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            _save(path, runtime, 1)
            fresh = _runtime("ema")
            base = dict(
                path=path,
                param_index=fresh.index,
                optimizer=fresh.optimizer,
                scheduler=fresh.scheduler,
                precision_controller=fresh.precision,
                step_engine=fresh.engine,
                estimator=fresh.estimator,
                perturbation=fresh.perturbation,
                expected_method="sample",
                expected_config_sha256="config-v1",
                expected_source_fingerprint="source-v1",
                explicit_generators={"normal": fresh.generator},
            )
            for key, value in (
                ("expected_method", "sam"),
                ("expected_config_sha256", "config-v2"),
                ("expected_source_fingerprint", "source-v2"),
            ):
                changed = dict(base)
                changed[key] = value
                with self.subTest(key=key), self.assertRaises(CheckpointCompatibilityError):
                    load_scientific_checkpoint(**changed)
            wrong_name = _runtime("ema", name="other")
            with self.assertRaises(CheckpointCompatibilityError):
                _load(path, wrong_name)
            wrong_shape = _runtime("ema", size=3)
            with self.assertRaises(CheckpointCompatibilityError):
                _load(path, wrong_shape)
            wrong_mode = _runtime("exact")
            with self.assertRaises(CheckpointCompatibilityError):
                _load(path, wrong_mode)
            wrong_lambda = _runtime("ema", ema_lambda=0.25)
            with self.assertRaises(CheckpointCompatibilityError):
                _load(path, wrong_lambda)

    def test_periodic_k_mismatch_fails_before_restore(self):
        _seed_normal()
        runtime = _runtime("periodic", k=2)
        _one_step(runtime)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            _save(path, runtime, 1)
            wrong = _runtime("periodic", k=3)
            initial = wrong.model.parameter.detach().clone()
            with self.assertRaises(CheckpointCompatibilityError):
                _load(path, wrong)
            self.assertTrue(torch.equal(initial, wrong.model.parameter))
            self.assertEqual(wrong.engine.optimizer_step, 0)

    def test_no_perturbed_checkpoint_and_no_gradient_buffers(self):
        _seed_normal()
        runtime = _runtime("ema")
        _one_step(runtime)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            zero = GradientState.zeros(runtime.index)
            with runtime.perturbation.displaced(zero):
                with self.assertRaises(Exception):
                    _save(path, runtime, 1)
            _save(path, runtime, 1)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["gradient_buffer_policy"], "not_serialized_safe_step_boundary")
            self.assertNotIn("grad", payload["trainable_model_state"])
            fresh = _runtime("ema")
            fresh.model.parameter.grad = torch.ones_like(fresh.model.parameter)
            _load(path, fresh)
            self.assertIsNone(fresh.model.parameter.grad)

    def test_worker0_loader_replay_reproduces_next_batch(self):
        _seed_normal()
        runtime = _runtime("ema")
        loader_generator = torch.Generator().manual_seed(4567)
        loader = DataLoader(
            _StochasticDataset(),
            batch_size=2,
            shuffle=True,
            num_workers=0,
            generator=loader_generator,
        )
        epoch_start = capture_rng_state((loader_generator,))
        iterator = iter(loader)
        next(iterator)
        next(iterator)
        _one_fixed_step(runtime, 0.0)
        _one_fixed_step(runtime, 0.1)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            _save(
                path,
                runtime,
                2,
                normal_loader_epoch_start_rng=epoch_start,
                normal_loader_length=len(loader),
                explicit_generators={"loader": loader_generator},
            )
            expected_next = next(iterator)
            fresh = _runtime("ema")
            resumed_loader_generator = torch.Generator().manual_seed(9999)
            resumed_loader = DataLoader(
                _StochasticDataset(),
                batch_size=2,
                shuffle=True,
                num_workers=0,
                generator=resumed_loader_generator,
            )
            result = _load(
                path,
                fresh,
                explicit_generators={"loader": resumed_loader_generator},
            )
            resumed_iterator = result.resume_worker0_loader(
                resumed_loader,
                explicit_generators={"loader": resumed_loader_generator},
            )
            actual_next = next(resumed_iterator)
            self.assertTrue(_nested_equal(expected_next, actual_next))

    def test_explicit_generator_names_are_validated(self):
        _seed_normal()
        runtime = _runtime("ema")
        _one_step(runtime)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            _save(path, runtime, 1)
            fresh = _runtime("ema")
            with self.assertRaises(CheckpointCompatibilityError):
                load_scientific_checkpoint(
                    path,
                    param_index=fresh.index,
                    optimizer=fresh.optimizer,
                    scheduler=fresh.scheduler,
                    precision_controller=fresh.precision,
                    step_engine=fresh.engine,
                    estimator=fresh.estimator,
                    perturbation=fresh.perturbation,
                    expected_method="sample",
                    expected_config_sha256="config-v1",
                    expected_source_fingerprint="source-v1",
                    explicit_generators={"renamed": fresh.generator},
                )

    def test_legacy_checkpoint_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            torch.save({"state_dict": {"ctx": torch.ones(2)}}, path)
            fresh = _runtime("ema")
            with self.assertRaisesRegex(CheckpointCompatibilityError, "Legacy/upstream"):
                _load(path, fresh)

    def test_atomic_failure_preserves_existing_checkpoint(self):
        _seed_normal()
        runtime = _runtime("ema")
        _one_step(runtime)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            path.write_bytes(b"known-good")
            with mock.patch("sample_fg.checkpoint.os.replace", side_effect=OSError("fixture")):
                with self.assertRaises(OSError):
                    _save(path, runtime, 1)
            self.assertEqual(path.read_bytes(), b"known-good")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_payload_has_versioned_complete_state(self):
        _seed_normal()
        runtime = _runtime("periodic")
        _one_step(runtime)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.pt"
            metadata = _save(path, runtime, 1)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["schema_version"], CHECKPOINT_SCHEMA_VERSION)
            for key in (
                "trainable_model_state", "optimizer_state", "scheduler_state",
                "precision_state", "step_engine_state", "estimator_state",
                "progress", "rng_state", "param_index", "config_sha256",
                "source_fingerprint", "result_state",
            ):
                self.assertIn(key, payload)
            self.assertRegex(metadata.sha256, r"^[0-9a-f]{64}$")
            self.assertTrue(math.isfinite(float(payload["estimator_state"]["ema_lambda"])))


if __name__ == "__main__":
    unittest.main()
