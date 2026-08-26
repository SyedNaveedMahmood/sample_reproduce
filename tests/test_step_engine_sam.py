import copy
import unittest

import torch
from torch.nn import functional as F

from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController, PrecisionStateError
from sample_fg.step_engine import StepEngine


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


def _batch():
    return (
        torch.tensor([[1.0, 2.0], [-0.5, 1.5]], dtype=torch.float32),
        torch.tensor([0.3, -0.8], dtype=torch.float32),
    )


def _loss(model, batch):
    x, target = batch
    return F.mse_loss(model(x), target)


def _reference_step(model, optimizer, batch, rho):
    original = model.weight.detach().clone()
    loss = _loss(model, batch)
    (g,) = torch.autograd.grad(loss, (model.weight,))
    epsilon = rho * g / torch.linalg.vector_norm(g)
    with torch.no_grad():
        model.weight.copy_(original + epsilon)
    displaced_loss = _loss(model, batch)
    (p,) = torch.autograd.grad(displaced_loss, (model.weight,))
    with torch.no_grad():
        model.weight.copy_(original)
    model.weight.grad = p.detach().clone()
    optimizer.step()
    return g.detach(), epsilon.detach(), p.detach()


class VanillaSAMStepTest(unittest.TestCase):
    def test_manual_reference_with_momentum_weight_decay_and_final_p_only(self):
        model = _TinyModel()
        reference = copy.deepcopy(model)
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(
            index.parameters,
            lr=0.07,
            momentum=0.9,
            weight_decay=0.04,
        )
        reference_optimizer = torch.optim.SGD(
            reference.parameters(),
            lr=0.07,
            momentum=0.9,
            weight_decay=0.04,
        )
        optimizer.state[model.weight]["momentum_buffer"] = torch.tensor([0.2, -0.1])
        reference_optimizer.state[reference.weight]["momentum_buffer"] = torch.tensor([0.2, -0.1])
        batch = _batch()
        original = model.weight.detach().clone()

        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        record = engine.step_sam(batch, lambda materialized: _loss(model, materialized))
        g_ref, epsilon_ref, p_ref = _reference_step(
            reference, reference_optimizer, batch, 0.05
        )

        torch.testing.assert_close(record.batch_gradient[0], g_ref)
        torch.testing.assert_close(record.sam_perturbation[0], epsilon_ref)
        torch.testing.assert_close(record.perturbed_gradient[0], p_ref)
        torch.testing.assert_close(record.final_gradient[0], p_ref)
        torch.testing.assert_close(model.weight, reference.weight)
        self.assertFalse(torch.allclose(record.final_gradient[0], record.batch_gradient[0] + p_ref))
        self.assertEqual(optimizer.step_calls, 1)
        self.assertTrue(torch.equal(optimizer.parameter_values_at_step[0][0], original))
        self.assertTrue(record.restored_before_optimizer)
        self.assertEqual(engine.optimizer_step, 1)

    def test_same_materialized_batch_object_and_storage_are_reused(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.1)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        batch = _batch()
        observations = []

        def closure(observed):
            observations.append((id(observed), observed[0].data_ptr(), observed[1].data_ptr()))
            return _loss(model, observed)

        record = engine.step_sam(batch, closure)
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0], observations[1])
        self.assertTrue(record.same_batch_object_reused)

    def test_degenerate_gradient_uses_zero_epsilon_and_still_one_step(self):
        model = _TinyModel(value=(0.0, 0.0))
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.1)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        batch = (torch.zeros(2, 2), torch.zeros(2))
        record = engine.step_sam(batch, lambda item: _loss(model, item))
        self.assertTrue(record.batch_gradient_degenerate)
        self.assertEqual(record.sam_perturbation_norm, 0.0)
        self.assertEqual(optimizer.step_calls, 1)

    def test_displaced_exception_restores_and_prevents_optimizer_step(self):
        model = _TinyModel()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.1)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        original = model.weight.detach().clone()
        calls = 0

        def closure(batch):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected displaced forward failure")
            return _loss(model, batch)

        with self.assertRaisesRegex(RuntimeError, "injected displaced forward failure"):
            engine.step_sam(_batch(), closure)
        self.assertTrue(torch.equal(model.weight, original))
        self.assertEqual(optimizer.step_calls, 0)
        self.assertFalse(engine.perturbation.active)

    def test_frozen_parameter_is_unchanged(self):
        model = _TinyModel()
        frozen = model.frozen.detach().clone()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.1)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        engine.step_sam(_batch(), lambda item: _loss(model, item))
        self.assertTrue(torch.equal(model.frozen, frozen))
        self.assertIsNone(model.frozen.grad)

    def test_fp16_logical_states_are_fp32(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        model = _TinyModel().cuda().half()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp16"),
            rho=0.05,
        )
        batch = tuple(item.cuda().half() for item in _batch())
        record = engine.step_sam(batch, lambda item: _loss(model, item))
        for state in (
            record.batch_gradient,
            record.sam_perturbation,
            record.perturbed_gradient,
            record.final_gradient,
        ):
            self.assertTrue(all(component.dtype == torch.float32 for component in state))
            self.assertTrue(all(component.grad_fn is None for component in state))

    def test_amp_multi_capture_is_rejected_without_scaler_state_hack(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        model = _TinyModel().cuda()
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(index.parameters, lr=0.01)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("amp"),
            rho=0.05,
        )
        batch = tuple(item.cuda() for item in _batch())
        with self.assertRaises(PrecisionStateError):
            engine.step_sam(batch, lambda item: _loss(model, item))
        self.assertEqual(optimizer.step_calls, 0)


if __name__ == "__main__":
    unittest.main()
