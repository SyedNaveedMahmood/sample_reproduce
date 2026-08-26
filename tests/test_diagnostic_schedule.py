import copy
import unittest

import torch
from torch.nn import functional as F

from sample_fg.diagnostic_schedule import (
    EXACT_ESTIMATOR_REUSE,
    INDEPENDENT_DIAGNOSTIC_QUERY,
    PERIODIC_REFRESH_REUSE,
    DiagnosticCoordinator,
    DiagnosticSchedule,
    DiagnosticScheduleError,
)
from sample_fg.estimators import EMAEstimator, ExactEstimator, PeriodicEstimator
from sample_fg.full_gradient import (
    FullGradientResult,
    FullGradientSweepMetadata,
)
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import derive_auxiliary_seed
from sample_fg.step_engine import StepEngine


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2]))

    def forward(self, x):
        return x @ self.weight


def _state(index, values=(1.5, -0.75)):
    return GradientState.from_tensors(
        index, [torch.tensor(values, dtype=torch.float32)]
    )


def _batch():
    return (
        torch.tensor([[1.0, 2.0], [-0.5, 1.5]], dtype=torch.float32),
        torch.tensor([0.3, -0.8], dtype=torch.float32),
    )


def _loss(model, batch):
    return F.mse_loss(model(batch[0]), batch[1])


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class _CountingService:
    def __init__(self, index, values=(1.5, -0.75)):
        self.index = index
        self.values = values
        self.calls = []

    def compute(self, *, optimizer_step, purpose):
        self.calls.append((optimizer_step, purpose))
        seed = derive_auxiliary_seed(
            protocol_seed=1,
            dataset="fixture",
            shots=4,
            config_hash="diagnostic-schedule-test",
            optimizer_step=optimizer_step,
            purpose=purpose,
        )
        metadata = FullGradientSweepMetadata(
            sample_count=4,
            micro_batch_count=2,
            configured_micro_batch_size=2,
            observed_micro_batch_sizes=(2, 2),
            forward_calls=2,
            autograd_grad_calls=2,
            mean_loss=0.5,
            elapsed_s=0.25,
            precision_mode="fp32",
            param_index_fingerprint=self.index.fingerprint,
            source_fingerprint="fixture-source",
            seed=seed,
        )
        return FullGradientResult(_state(self.index, self.values), metadata)


class DiagnosticSchedulingTest(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.index = ParamIndex.from_model(self.model)
        self.batch_grad = _state(self.index, (2.0, 1.0))

    def coordinator(self, service, interval=1):
        return DiagnosticCoordinator(
            schedule=DiagnosticSchedule(interval),
            full_gradient_service=service,
        )

    def test_concrete_schedule_uses_zero_based_optimizer_steps(self):
        schedule = DiagnosticSchedule(3)
        self.assertEqual(
            [schedule.is_due(step) for step in range(8)],
            [True, False, False, True, False, False, True, False],
        )
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises((DiagnosticScheduleError, TypeError)):
                DiagnosticSchedule(invalid)

    def test_ema_no_diagnostic_zero_calls_and_diagnostic_one_read_only_call(self):
        service = _CountingService(self.index)
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        result = estimator.global_direction(
            batch_grad=self.batch_grad, optimizer_step=0
        )
        self.assertEqual(service.calls, [])
        before = copy.deepcopy(estimator.state_dict())
        reference = self.coordinator(service).reference_for_step(
            result, optimizer_step=0
        )
        self.assertEqual(service.calls, [(0, "diagnostic")])
        self.assertEqual(reference.source, INDEPENDENT_DIAGNOSTIC_QUERY)
        self.assertTrue(reference.exact_service_query_issued)
        self.assertTrue(_nested_equal(before, estimator.state_dict()))

    def test_exact_diagnostic_reuses_single_optimization_query(self):
        service = _CountingService(self.index)
        estimator = ExactEstimator(
            self.index, full_gradient_service=service
        )
        result = estimator.global_direction(
            batch_grad=self.batch_grad, optimizer_step=0
        )
        reference = self.coordinator(service).reference_for_step(
            result, optimizer_step=0
        )
        self.assertEqual(service.calls, [(0, "optimization_exact")])
        self.assertEqual(reference.source, EXACT_ESTIMATOR_REUSE)
        self.assertFalse(reference.exact_service_query_issued)
        self.assertEqual(
            reference.full_gradient_metadata.seed.purpose,
            "optimization_exact",
        )

    def test_periodic_refresh_reuse_and_nonrefresh_independent_query(self):
        service = _CountingService(self.index)
        estimator = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=2,
            full_gradient_service=service,
        )
        coordinator = self.coordinator(service)
        refresh = estimator.global_direction(
            batch_grad=self.batch_grad, optimizer_step=0
        )
        reused = coordinator.reference_for_step(refresh, optimizer_step=0)
        self.assertEqual(service.calls, [(0, "periodic_refresh")])
        self.assertEqual(reused.source, PERIODIC_REFRESH_REUSE)
        self.assertFalse(reused.exact_service_query_issued)

        nonrefresh = estimator.global_direction(
            batch_grad=_state(self.index, (-0.25, 2.0)), optimizer_step=1
        )
        before = copy.deepcopy(estimator.state_dict())
        independent = coordinator.reference_for_step(
            nonrefresh, optimizer_step=1
        )
        self.assertEqual(
            service.calls,
            [(0, "periodic_refresh"), (1, "diagnostic")],
        )
        self.assertEqual(independent.source, INDEPENDENT_DIAGNOSTIC_QUERY)
        self.assertTrue(independent.exact_service_query_issued)
        self.assertTrue(_nested_equal(before, estimator.state_dict()))
        self.assertEqual(estimator.last_refresh_step, 0)
        self.assertEqual(estimator.age_steps, 1)

    def test_periodic_step_two_refresh_and_diagnostic_still_one_call(self):
        service = _CountingService(self.index)
        estimator = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=2,
            full_gradient_service=service,
        )
        coordinator = self.coordinator(service)
        estimator.global_direction(batch_grad=self.batch_grad, optimizer_step=0)
        estimator.global_direction(batch_grad=self.batch_grad, optimizer_step=1)
        result = estimator.global_direction(
            batch_grad=self.batch_grad, optimizer_step=2
        )
        reference = coordinator.reference_for_step(result, optimizer_step=2)
        self.assertEqual(
            service.calls,
            [(0, "periodic_refresh"), (2, "periodic_refresh")],
        )
        self.assertEqual(reference.source, PERIODIC_REFRESH_REUSE)

    def test_step_mismatch_and_missing_reuse_metadata_fail(self):
        service = _CountingService(self.index)
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        result = estimator.global_direction(
            batch_grad=self.batch_grad, optimizer_step=0
        )
        with self.assertRaises(DiagnosticScheduleError):
            self.coordinator(service).reference_for_step(
                result, optimizer_step=1
            )

    def test_paired_ema_trajectory_is_identical_with_diagnostic_on(self):
        control_model = _Model()
        diagnostic_model = copy.deepcopy(control_model)
        control_index = ParamIndex.from_model(control_model)
        diagnostic_index = ParamIndex.from_model(diagnostic_model)
        control_optimizer = torch.optim.SGD(
            control_index.parameters, lr=0.03, momentum=0.8
        )
        diagnostic_optimizer = torch.optim.SGD(
            diagnostic_index.parameters, lr=0.03, momentum=0.8
        )
        control_estimator = EMAEstimator(control_index, ema_lambda=0.15)
        diagnostic_estimator = EMAEstimator(diagnostic_index, ema_lambda=0.15)
        service = _CountingService(diagnostic_index)
        control_engine = StepEngine(
            param_index=control_index,
            optimizer=control_optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        diagnostic_engine = StepEngine(
            param_index=diagnostic_index,
            optimizer=diagnostic_optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
            diagnostic_coordinator=self.coordinator(service),
        )
        for step in range(2):
            control = control_engine.step_sample(
                _batch(), lambda item: _loss(control_model, item), control_estimator
            )
            diagnostic = diagnostic_engine.step_sample(
                _batch(),
                lambda item: _loss(diagnostic_model, item),
                diagnostic_estimator,
                epoch=0,
                batch_index=step,
            )
            self.assertIsNone(control.diagnostic_event)
            self.assertIsNotNone(diagnostic.diagnostic_event)
            torch.testing.assert_close(
                control.final_gradient.components,
                diagnostic.final_gradient.components,
                rtol=0,
                atol=0,
            )
        self.assertTrue(torch.equal(control_model.weight, diagnostic_model.weight))
        self.assertTrue(
            _nested_equal(control_optimizer.state_dict(), diagnostic_optimizer.state_dict())
        )
        self.assertTrue(
            _nested_equal(control_estimator.state_dict(), diagnostic_estimator.state_dict())
        )
        self.assertEqual(service.calls, [(0, "diagnostic"), (1, "diagnostic")])

    def test_event_contains_query_provenance_and_complete_metrics(self):
        service = _CountingService(self.index)
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        engine = StepEngine(
            param_index=self.index,
            optimizer=torch.optim.SGD(self.index.parameters, lr=0.01),
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
            diagnostic_coordinator=self.coordinator(service),
        )
        record = engine.step_sample(
            _batch(),
            lambda item: _loss(self.model, item),
            estimator,
            epoch=3,
            batch_index=4,
        )
        payload = record.diagnostic_event.as_dict()
        self.assertEqual(payload["optimizer_step"], 0)
        self.assertEqual(payload["epoch"], 3)
        self.assertEqual(payload["batch_index"], 4)
        self.assertEqual(
            payload["exact_reference_source"], INDEPENDENT_DIAGNOSTIC_QUERY
        )
        self.assertTrue(payload["exact_service_query_issued"])
        self.assertEqual(
            payload["exact_reference_auxiliary_seed"]["purpose"], "diagnostic"
        )
        self.assertIn("grad/global_estimate_exact_cosine", payload["metrics"])


if __name__ == "__main__":
    unittest.main()
