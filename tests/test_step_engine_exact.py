import unittest

import torch
from torch.nn import functional as F

from sample_fg.estimators import EstimatorResult, ExactEstimator, GlobalGradientEstimator
from sample_fg.full_gradient import FullGradientResult, FullGradientSweepMetadata
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import derive_auxiliary_seed
from sample_fg.step_engine import StepEngine


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.2]))

    def forward(self, x):
        return x @ self.weight


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0

    def step(self, closure=None):
        self.step_calls += 1
        return super().step(closure=closure)


def _batch():
    return (
        torch.tensor([[1.0, 2.0], [-0.5, 1.5]]),
        torch.tensor([0.3, -0.8]),
    )


def _loss(model, batch):
    return F.mse_loss(model(batch[0]), batch[1])


def _metadata(index, step, purpose):
    return FullGradientSweepMetadata(
        sample_count=4,
        micro_batch_count=2,
        configured_micro_batch_size=2,
        observed_micro_batch_sizes=(2, 2),
        forward_calls=2,
        autograd_grad_calls=2,
        mean_loss=1.0,
        elapsed_s=0.01,
        precision_mode="fp32",
        param_index_fingerprint=index.fingerprint,
        source_fingerprint="fixture-source",
        seed=derive_auxiliary_seed(
            protocol_seed=1,
            dataset="fixture",
            shots=2,
            config_hash="fixture-config",
            optimizer_step=step,
            purpose=purpose,
        ),
    )


class _RecordingExactService:
    def __init__(self, model, index, direction):
        self.model = model
        self.index = index
        self.direction = direction
        self.calls = []
        self.expected_theta = None

    def compute(self, *, optimizer_step, purpose):
        theta = tuple(parameter.detach().clone() for parameter in self.index.parameters)
        grads = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in self.index.parameters
        )
        if self.expected_theta is None or not all(
            torch.equal(value, expected)
            for value, expected in zip(theta, self.expected_theta)
        ):
            raise AssertionError("Exact service was queried away from original theta")
        self.calls.append(
            {
                "optimizer_step": optimizer_step,
                "purpose": purpose,
                "theta": theta,
                "grads": grads,
            }
        )
        return FullGradientResult(
            gradient=self.direction.clone(),
            metadata=_metadata(self.index, optimizer_step, purpose),
        )


class _FixedEstimator(GlobalGradientEstimator):
    def __init__(self, index, direction, mode):
        super().__init__(index)
        self.direction = direction.clone()
        self.mode = mode

    def global_direction(self, *, batch_grad, optimizer_step):
        self._validate_call(batch_grad, optimizer_step)
        self._last_processed_step = optimizer_step
        return EstimatorResult(
            active_global_estimate=self.direction.clone(),
            mode=self.mode,
            optimizer_step=optimizer_step,
            refreshed=self.mode == "exact-style",
            age_steps=None,
            last_refresh_step=None,
            exact_reference=None,
            full_gradient_metadata=None,
            exact_query_count=0,
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, payload):
        if payload != {}:
            raise ValueError("fixture payload differs")


class SAMPLeExactStepTest(unittest.TestCase):
    def test_exact_query_once_at_unperturbed_theta_before_displacement(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.02)
        direction = GradientState.from_tensors(index, (torch.tensor([1.5, -0.75]),))
        service = _RecordingExactService(model, index, direction)
        estimator = ExactEstimator(index, full_gradient_service=service)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        service.expected_theta = tuple(p.detach().clone() for p in index.parameters)
        record = engine.step_sample(_batch(), lambda item: _loss(model, item), estimator)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(service.calls[0]["optimizer_step"], 0)
        self.assertEqual(service.calls[0]["purpose"], "optimization_exact")
        self.assertTrue(all(grad is not None for grad in service.calls[0]["grads"]))
        torch.testing.assert_close(record.estimator_result.active_global_estimate.components, direction.components)
        torch.testing.assert_close(record.estimator_result.exact_reference.components, direction.components)
        self.assertEqual(estimator.exact_query_count, 1)
        self.assertEqual(optimizer.step_calls, 1)

    def test_exact_does_not_ema_mix_batch_gradient(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.02)
        direction = GradientState.from_tensors(index, (torch.tensor([-2.0, 0.25]),))
        service = _RecordingExactService(model, index, direction)
        estimator = ExactEstimator(index, full_gradient_service=service)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        service.expected_theta = tuple(p.detach().clone() for p in index.parameters)
        record = engine.step_sample(_batch(), lambda item: _loss(model, item), estimator)
        self.assertTrue(torch.equal(record.estimator_result.active_global_estimate[0], direction[0]))
        mixed = direction.scale(0.15).add(record.batch_gradient.scale(0.85))
        self.assertFalse(torch.allclose(record.estimator_result.active_global_estimate[0], mixed[0]))

    def test_same_direction_proves_shared_downstream_engine(self):
        left_model = _TinyModel()
        right_model = _TinyModel()
        left_index = ParamIndex.from_model(left_model)
        right_index = ParamIndex.from_model(right_model)
        left_optimizer = _CountingSGD(left_index.parameters, lr=0.02)
        right_optimizer = _CountingSGD(right_index.parameters, lr=0.02)
        left_engine = StepEngine(
            param_index=left_index,
            optimizer=left_optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        right_engine = StepEngine(
            param_index=right_index,
            optimizer=right_optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        left_direction = GradientState.from_tensors(left_index, (torch.tensor([1.0, 2.0]),))
        right_direction = GradientState.from_tensors(right_index, (torch.tensor([1.0, 2.0]),))
        left = left_engine.step_sample(
            _batch(),
            lambda item: _loss(left_model, item),
            _FixedEstimator(left_index, left_direction, "ema-style"),
        )
        right = right_engine.step_sample(
            _batch(),
            lambda item: _loss(right_model, item),
            _FixedEstimator(right_index, right_direction, "exact-style"),
        )
        for left_state, right_state in (
            (left.projection.batch_component, right.projection.batch_component),
            (left.sam_perturbation, right.sam_perturbation),
            (left.total_displacement, right.total_displacement),
            (left.perturbed_gradient, right.perturbed_gradient),
            (left.final_gradient, right.final_gradient),
        ):
            torch.testing.assert_close(left_state.components, right_state.components)
        torch.testing.assert_close(left_model.weight, right_model.weight)


if __name__ == "__main__":
    unittest.main()
