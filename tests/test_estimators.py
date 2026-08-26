import copy
import unittest

import torch
from torch import nn

from sample_fg.estimators import (
    EMAEstimator,
    ESTIMATOR_STATE_SCHEMA_VERSION,
    ExactEstimator,
    EstimatorError,
    EstimatorNumericalError,
    EstimatorStateError,
    PeriodicEstimator,
)
from sample_fg.full_gradient import (
    FullGradientResult,
    FullGradientSweepMetadata,
)
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.rng import derive_auxiliary_seed


class _TwoParameterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.second = nn.Parameter(torch.zeros(1, 2, dtype=torch.float32))


def _state(index, values):
    first, second = values
    return GradientState.from_tensors(
        index,
        (
            torch.tensor(first, dtype=torch.float32),
            torch.tensor([second], dtype=torch.float32),
        ),
    )


def _flat(state):
    return torch.cat([component.reshape(-1) for component in state])


class _FakeFullGradientService:
    def __init__(self, index, outputs):
        self.index = index
        self.outputs = {step: state.clone() for step, state in outputs.items()}
        self.calls = []

    def compute(self, *, optimizer_step, purpose):
        self.calls.append((optimizer_step, purpose))
        gradient = self.outputs[optimizer_step].clone()
        seed = derive_auxiliary_seed(
            protocol_seed=1,
            dataset="fixture",
            shots=4,
            config_hash="estimator-test-v1",
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
            mean_loss=1.0 + optimizer_step,
            elapsed_s=0.01,
            precision_mode="fp32",
            param_index_fingerprint=self.index.fingerprint,
            source_fingerprint="fixture-source",
            seed=seed,
        )
        return FullGradientResult(gradient=gradient, metadata=metadata)


class EstimatorStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.model = _TwoParameterModel()
        self.index = ParamIndex.from_model(self.model)
        self.gradients = [
            _state(self.index, ([1.0, 2.0], [3.0, 4.0])),
            _state(self.index, ([2.0, -1.0], [0.5, 3.0])),
            _state(self.index, ([-1.0, 0.25], [4.0, -2.0])),
            _state(self.index, ([3.0, 1.0], [-1.0, 2.0])),
            _state(self.index, ([0.5, 0.5], [0.5, 0.5])),
            _state(self.index, ([4.0, -3.0], [2.0, 1.0])),
            _state(self.index, ([-2.0, -1.0], [1.0, 3.0])),
            _state(self.index, ([1.5, 2.5], [-0.5, 0.75])),
        ]
        self.full = {
            step: _state(
                self.index,
                ([100.0 + step, 200.0 + step], [300.0 + step, 400.0 + step]),
            )
            for step in range(8)
        }

    def test_ema_zero_init_recurrence_no_bias_correction_and_no_service(self):
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        expected = GradientState.zeros(self.index)
        for step in range(3):
            expected.affine_(
                self.gradients[step], self_weight=0.15, other_weight=0.85
            )
            result = estimator.global_direction(
                batch_grad=self.gradients[step], optimizer_step=step
            )
            torch.testing.assert_close(
                _flat(result.active_global_estimate), _flat(expected), rtol=0, atol=0
            )
            self.assertFalse(result.refreshed)
            self.assertIsNone(result.exact_reference)
            self.assertEqual(result.exact_query_count, 0)
        torch.testing.assert_close(
            _flat(estimator.active_state), 0.85 * _flat(self.gradients[0]) * 0.15**2
            + 0.85 * _flat(self.gradients[1]) * 0.15
            + 0.85 * _flat(self.gradients[2]),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_exact_queries_once_per_logical_step_and_does_not_mix_batch(self):
        service = _FakeFullGradientService(self.index, self.full)
        estimator = ExactEstimator(self.index, full_gradient_service=service)
        for step in range(3):
            result = estimator.global_direction(
                batch_grad=self.gradients[step], optimizer_step=step
            )
            self.assertTrue(result.refreshed)
            self.assertIsNone(result.age_steps)
            self.assertEqual(result.last_refresh_step, step)
            self.assertEqual(result.exact_query_count, step + 1)
            self.assertTrue(torch.equal(_flat(result.active_global_estimate), _flat(self.full[step])))
            self.assertTrue(torch.equal(_flat(result.exact_reference), _flat(self.full[step])))
            self.assertFalse(torch.equal(_flat(result.active_global_estimate), _flat(self.gradients[step])))
        self.assertEqual(
            service.calls,
            [(0, "optimization_exact"), (1, "optimization_exact"), (2, "optimization_exact")],
        )

    def test_periodic_k3_refresh_ages_hard_reset_and_between_refresh_ema(self):
        service = _FakeFullGradientService(self.index, self.full)
        estimator = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=3,
            full_gradient_service=service,
        )
        expected = None
        expected_ages = (0, 1, 2, 0, 1, 2, 0, 1)
        for step in range(8):
            if step % 3 == 0:
                expected = self.full[step].clone()
            else:
                expected.affine_(
                    self.gradients[step], self_weight=0.15, other_weight=0.85
                )
            result = estimator.global_direction(
                batch_grad=self.gradients[step], optimizer_step=step
            )
            self.assertEqual(result.refreshed, step % 3 == 0)
            self.assertEqual(result.age_steps, expected_ages[step])
            self.assertEqual(result.last_refresh_step, (step // 3) * 3)
            torch.testing.assert_close(
                _flat(result.active_global_estimate), _flat(expected), rtol=0, atol=0
            )
            if step % 3 == 0:
                # A refresh equals exact state, with no same-step EMA mix.
                self.assertTrue(
                    torch.equal(_flat(result.active_global_estimate), _flat(self.full[step]))
                )
        self.assertEqual(
            service.calls,
            [(0, "periodic_refresh"), (3, "periodic_refresh"), (6, "periodic_refresh")],
        )

    def test_k_validation_and_k1_is_true_exact_alias(self):
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(EstimatorError):
                PeriodicEstimator(
                    self.index,
                    ema_lambda=0.15,
                    refresh_k_steps=invalid,
                    full_gradient_service=_FakeFullGradientService(self.index, self.full),
                )

        exact_service = _FakeFullGradientService(self.index, self.full)
        periodic_service = _FakeFullGradientService(self.index, self.full)
        exact = ExactEstimator(self.index, full_gradient_service=exact_service)
        periodic = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=1,
            full_gradient_service=periodic_service,
        )
        for step in range(4):
            exact_result = exact.global_direction(
                batch_grad=self.gradients[step], optimizer_step=step
            )
            periodic_result = periodic.global_direction(
                batch_grad=self.gradients[step], optimizer_step=step
            )
            self.assertTrue(periodic_result.refreshed)
            self.assertEqual(periodic_result.age_steps, 0)
            self.assertTrue(
                torch.equal(
                    _flat(exact_result.active_global_estimate),
                    _flat(periodic_result.active_global_estimate),
                )
            )
        self.assertEqual(exact_service.calls, periodic_service.calls)
        self.assertTrue(all(purpose == "optimization_exact" for _, purpose in periodic_service.calls))

    def test_zero_based_sequential_step_contract(self):
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        with self.assertRaises(EstimatorStateError):
            estimator.global_direction(batch_grad=self.gradients[0], optimizer_step=1)
        estimator.global_direction(batch_grad=self.gradients[0], optimizer_step=0)
        for invalid in (0, 2):
            with self.assertRaises(EstimatorStateError):
                estimator.global_direction(
                    batch_grad=self.gradients[1], optimizer_step=invalid
                )

    def test_nonfinite_and_param_index_mismatch_are_rejected(self):
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        nonfinite = self.gradients[0].clone()
        nonfinite[0][0] = float("nan")
        with self.assertRaises(EstimatorNumericalError):
            estimator.global_direction(batch_grad=nonfinite, optimizer_step=0)

        other_model = nn.Module()
        other_model.register_parameter("different", nn.Parameter(torch.zeros(4)))
        other_index = ParamIndex.from_model(other_model)
        other_state = GradientState.from_tensors(other_index, (torch.zeros(4),))
        with self.assertRaises(EstimatorStateError):
            estimator.global_direction(batch_grad=other_state, optimizer_step=0)

    def test_estimator_owns_state_and_results_do_not_alias(self):
        input_state = self.gradients[0].clone()
        estimator = EMAEstimator(self.index, ema_lambda=0.15)
        result = estimator.global_direction(batch_grad=input_state, optimizer_step=0)
        internal_before = estimator.active_state
        input_state[0].add_(999.0)
        result.active_global_estimate[0].sub_(777.0)
        self.assertTrue(torch.equal(_flat(estimator.active_state), _flat(internal_before)))

        service = _FakeFullGradientService(self.index, self.full)
        periodic = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=3,
            full_gradient_service=service,
        )
        periodic_result = periodic.global_direction(
            batch_grad=self.gradients[0], optimizer_step=0
        )
        periodic_before = periodic.active_state
        periodic_result.active_global_estimate[0].add_(123.0)
        periodic_result.exact_reference[0].sub_(456.0)
        self.assertTrue(torch.equal(_flat(periodic.active_state), _flat(periodic_before)))

    def test_versioned_serialization_roundtrip_and_resume(self):
        ema = EMAEstimator(self.index, ema_lambda=0.15)
        for step in range(2):
            ema.global_direction(batch_grad=self.gradients[step], optimizer_step=step)
        payload = ema.state_dict()
        self.assertEqual(payload["schema_version"], ESTIMATOR_STATE_SCHEMA_VERSION)
        restored_ema = EMAEstimator(self.index, ema_lambda=0.15)
        restored_ema.load_state_dict(copy.deepcopy(payload))
        original_next = ema.global_direction(batch_grad=self.gradients[2], optimizer_step=2)
        restored_next = restored_ema.global_direction(batch_grad=self.gradients[2], optimizer_step=2)
        self.assertTrue(torch.equal(_flat(original_next.active_global_estimate), _flat(restored_next.active_global_estimate)))

        service_a = _FakeFullGradientService(self.index, self.full)
        periodic = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=3,
            full_gradient_service=service_a,
        )
        for step in range(3):
            periodic.global_direction(batch_grad=self.gradients[step], optimizer_step=step)
        periodic_payload = periodic.state_dict()
        service_b = _FakeFullGradientService(self.index, self.full)
        restored_periodic = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=3,
            full_gradient_service=service_b,
        )
        restored_periodic.load_state_dict(copy.deepcopy(periodic_payload))
        next_a = periodic.global_direction(batch_grad=self.gradients[3], optimizer_step=3)
        next_b = restored_periodic.global_direction(batch_grad=self.gradients[3], optimizer_step=3)
        self.assertTrue(next_a.refreshed and next_b.refreshed)
        self.assertEqual(next_a.age_steps, next_b.age_steps)
        self.assertTrue(torch.equal(_flat(next_a.active_global_estimate), _flat(next_b.active_global_estimate)))

        exact_service_a = _FakeFullGradientService(self.index, self.full)
        exact = ExactEstimator(self.index, full_gradient_service=exact_service_a)
        exact.global_direction(batch_grad=self.gradients[0], optimizer_step=0)
        exact_payload = exact.state_dict()
        exact_service_b = _FakeFullGradientService(self.index, self.full)
        exact_restored = ExactEstimator(self.index, full_gradient_service=exact_service_b)
        exact_restored.load_state_dict(copy.deepcopy(exact_payload))
        exact_restored.global_direction(batch_grad=self.gradients[1], optimizer_step=1)
        self.assertEqual(exact_restored.exact_query_count, 2)

    def test_serialization_rejects_config_fingerprint_and_counter_mismatch(self):
        ema = EMAEstimator(self.index, ema_lambda=0.15)
        payload = ema.state_dict()
        wrong_lambda = EMAEstimator(self.index, ema_lambda=0.2)
        with self.assertRaises(EstimatorStateError):
            wrong_lambda.load_state_dict(copy.deepcopy(payload))

        service = _FakeFullGradientService(self.index, self.full)
        periodic = PeriodicEstimator(
            self.index,
            ema_lambda=0.15,
            refresh_k_steps=3,
            full_gradient_service=service,
        )
        periodic.global_direction(batch_grad=self.gradients[0], optimizer_step=0)
        periodic_payload = periodic.state_dict()
        periodic_payload["exact_query_count"] = 99
        with self.assertRaises(EstimatorStateError):
            PeriodicEstimator(
                self.index,
                ema_lambda=0.15,
                refresh_k_steps=3,
                full_gradient_service=service,
            ).load_state_dict(periodic_payload)

        fingerprint_payload = ema.state_dict()
        fingerprint_payload["param_index_fingerprint"] = "0" * 64
        with self.assertRaises(EstimatorStateError):
            ema.load_state_dict(fingerprint_payload)

    def test_ema_lambda_validation(self):
        for invalid in (-0.1, 1.0, float("inf"), True, "0.15"):
            with self.assertRaises(EstimatorError):
                EMAEstimator(self.index, ema_lambda=invalid)


if __name__ == "__main__":
    unittest.main()
