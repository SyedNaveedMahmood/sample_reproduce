import copy
import unittest

import torch
from torch.nn import functional as F

from sample_fg.estimators import EMAEstimator
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.projection import project_batch_gradient, safe_unit
from sample_fg.step_engine import StepEngine, StepEngineError


class _TinyModel(torch.nn.Module):
    def __init__(self, value=(0.4, -0.2)):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.frozen = torch.nn.Parameter(torch.tensor([7.0]), requires_grad=False)

    def forward(self, x):
        return x @ self.weight


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.parameter_values_at_step = []

    def step(self, closure=None):
        self.step_calls += 1
        self.parameter_values_at_step.append(
            tuple(p.detach().clone() for group in self.param_groups for p in group["params"])
        )
        return super().step(closure=closure)


class _CountingEMA(EMAEstimator):
    def __init__(self, index, ema_lambda):
        super().__init__(index, ema_lambda=ema_lambda)
        self.calls = []

    def global_direction(self, *, batch_grad, optimizer_step):
        self.calls.append((optimizer_step, batch_grad.clone()))
        return super().global_direction(
            batch_grad=batch_grad, optimizer_step=optimizer_step
        )


def _batch():
    return (
        torch.tensor([[1.0, 2.0], [-0.5, 1.5]], dtype=torch.float32),
        torch.tensor([0.3, -0.8], dtype=torch.float32),
    )


def _loss(model, batch):
    return F.mse_loss(model(batch[0]), batch[1])


def _manual_sample_step(model, optimizer, batch, prior_ema, *, rho, alpha, ema_lambda):
    original = model.weight.detach().clone()
    current_loss = _loss(model, batch)
    (g,) = torch.autograd.grad(current_loss, (model.weight,))
    ema = ema_lambda * prior_ema + (1.0 - ema_lambda) * g.detach()
    coefficient = torch.dot(g, ema) / torch.dot(ema, ema)
    batch_component = g - coefficient * ema
    epsilon = rho * g / torch.linalg.vector_norm(g)
    delta = epsilon - alpha * batch_component
    with torch.no_grad():
        model.weight.copy_(original + delta)
    displaced_loss = _loss(model, batch)
    (p,) = torch.autograd.grad(displaced_loss, (model.weight,))
    with torch.no_grad():
        model.weight.copy_(original)
    final = g.detach() + p.detach()
    model.weight.grad = final.clone()
    optimizer.step()
    return {
        "g": g.detach(),
        "ema": ema,
        "batch_component": batch_component.detach(),
        "epsilon": epsilon.detach(),
        "delta": delta.detach(),
        "p": p.detach(),
        "final": final,
    }


class SAMPLeEMAStepTest(unittest.TestCase):
    def _engine(self, model, optimizer, *, alpha=0.0015):
        index = ParamIndex.from_model(model)
        return index, StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=alpha,
        )

    def test_end_to_end_manual_reference_and_final_sum_without_half(self):
        model = _TinyModel()
        reference = copy.deepcopy(model)
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.03, momentum=0.8, weight_decay=0.02)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03, momentum=0.8, weight_decay=0.02)
        optimizer.state[model.weight]["momentum_buffer"] = torch.tensor([0.1, -0.2])
        reference_optimizer.state[reference.weight]["momentum_buffer"] = torch.tensor([0.1, -0.2])
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        estimator = _CountingEMA(index, 0.15)
        reference_ema = torch.zeros_like(reference.weight)
        batch = _batch()
        original = model.weight.detach().clone()

        record = engine.step_sample(batch, lambda item: _loss(model, item), estimator)
        manual = _manual_sample_step(
            reference,
            reference_optimizer,
            batch,
            reference_ema,
            rho=0.05,
            alpha=0.0015,
            ema_lambda=0.15,
        )
        torch.testing.assert_close(record.batch_gradient[0], manual["g"])
        torch.testing.assert_close(record.estimator_result.active_global_estimate[0], manual["ema"])
        torch.testing.assert_close(record.projection.batch_component[0], manual["batch_component"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(record.sam_perturbation[0], manual["epsilon"])
        torch.testing.assert_close(record.total_displacement[0], manual["delta"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(record.perturbed_gradient[0], manual["p"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(record.final_gradient[0], manual["final"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(model.weight, reference.weight, atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(record.final_gradient[0], manual["final"] / 2.0))
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(torch.equal(optimizer.parameter_values_at_step[0][0], original))
        self.assertEqual(len(estimator.calls), 1)
        self.assertEqual(estimator.exact_query_count, 0)

    def test_ema_updates_once_before_projection_and_never_with_p(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        estimator = _CountingEMA(index, 0.15)
        first = engine.step_sample(_batch(), lambda item: _loss(model, item), estimator)
        second = engine.step_sample(_batch(), lambda item: _loss(model, item), estimator)
        self.assertEqual([step for step, _ in estimator.calls], [0, 1])
        expected0 = first.batch_gradient.scale(0.85)
        expected1 = expected0.scale(0.15).add(second.batch_gradient.scale(0.85))
        torch.testing.assert_close(first.estimator_result.active_global_estimate.components, expected0.components)
        torch.testing.assert_close(second.estimator_result.active_global_estimate.components, expected1.components)
        self.assertEqual(optimizer.step_calls, 2)

    def test_displacement_is_epsilon_minus_alpha_batch_component(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.2,
        )
        estimator = _CountingEMA(index, 0.15)
        # Prime EMA with a different gradient so step 1 has a nonzero residual.
        engine.step_sample(_batch(), lambda item: _loss(model, item), estimator)
        other_batch = (
            torch.tensor([[2.0, -1.0], [1.0, 0.25]]),
            torch.tensor([0.5, -0.1]),
        )
        record = engine.step_sample(other_batch, lambda item: _loss(model, item), estimator)
        expected_correction = record.projection.batch_component.scale(0.2)
        expected_delta = record.sam_perturbation.subtract(expected_correction)
        torch.testing.assert_close(record.batch_correction.components, expected_correction.components)
        torch.testing.assert_close(record.total_displacement.components, expected_delta.components)
        plus = record.sam_perturbation.add(expected_correction)
        self.assertFalse(torch.allclose(record.total_displacement[0], plus[0]))

    def test_same_materialized_batch_and_frozen_parameter(self):
        model = _TinyModel()
        frozen = model.frozen.detach().clone()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        estimator = EMAEstimator(index, ema_lambda=0.15)
        batch = _batch()
        calls = []

        def closure(observed):
            calls.append((id(observed), observed[0].data_ptr(), observed[1].data_ptr()))
            return _loss(model, observed)

        record = engine.step_sample(batch, closure, estimator)
        self.assertEqual(calls[0], calls[1])
        self.assertTrue(record.same_batch_object_reused)
        self.assertTrue(record.restored_before_optimizer)
        self.assertTrue(torch.equal(model.frozen, frozen))

    def test_alpha_required_and_states_remain_detached_fp32(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        estimator = EMAEstimator(index, ema_lambda=0.15)
        with self.assertRaises(StepEngineError):
            engine.step_sample(_batch(), lambda item: _loss(model, item), estimator)

        configured = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        record = configured.step_sample(_batch(), lambda item: _loss(model, item), estimator)
        states = (
            record.batch_gradient,
            record.estimator_result.active_global_estimate,
            record.projection.batch_component,
            record.sam_perturbation,
            record.batch_correction,
            record.total_displacement,
            record.perturbed_gradient,
            record.final_gradient,
        )
        for state in states:
            self.assertTrue(all(component.dtype == torch.float32 for component in state))
            self.assertTrue(all(not component.requires_grad and component.grad_fn is None for component in state))

    def test_degenerate_inputs_follow_existing_primitives(self):
        model = _TinyModel(value=(0.0, 0.0))
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        estimator = EMAEstimator(index, ema_lambda=0.15)
        batch = (torch.zeros(2, 2), torch.zeros(2))
        record = engine.step_sample(batch, lambda item: _loss(model, item), estimator)
        self.assertTrue(record.projection.batch_gradient_degenerate)
        self.assertTrue(record.projection.full_direction_degenerate)
        self.assertEqual(record.sam_perturbation_norm, 0.0)
        self.assertEqual(record.total_displacement_norm, 0.0)
        self.assertTrue(record.final_gradient.is_finite())


if __name__ == "__main__":
    unittest.main()
