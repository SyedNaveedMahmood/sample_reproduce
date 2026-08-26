"""Independent numerical checks of the paper's SAM and SAMPLe equations.

The reference calculations deliberately use raw tensors/autograd. They do not
import the production projection or gradient-state arithmetic helpers.
"""

import copy
import unittest

import torch
from torch.nn import functional as F

from sample_fg.estimators import EMAEstimator
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.step_engine import StepEngine


class _TwoParameterRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.35, -0.25]))
        self.bias = torch.nn.Parameter(torch.tensor(0.12))

    def forward(self, x):
        return x @ self.weight + self.bias


class _CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_calls = 0
        self.parameters_at_step = []

    def step(self, closure=None):
        self.step_calls += 1
        self.parameters_at_step.append(_flat_parameters(self.param_groups))
        return super().step(closure=closure)


def _batches():
    return (
        (
            torch.tensor([[1.2, -0.3], [-0.4, 1.7], [0.8, 0.5]]),
            torch.tensor([0.2, -0.7, 0.9]),
        ),
        (
            torch.tensor([[-1.1, 0.4], [0.3, 1.4], [1.5, -0.8]]),
            torch.tensor([-0.1, 0.6, 0.3]),
        ),
    )


def _loss(model, batch):
    return F.mse_loss(model(batch[0]), batch[1])


def _parameters(model):
    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def _flat_tensors(tensors):
    return torch.cat(tuple(tensor.detach().reshape(-1) for tensor in tensors))


def _flat_parameters(param_groups):
    return _flat_tensors(
        parameter
        for group in param_groups
        for parameter in group["params"]
    )


def _flat_state(state):
    return _flat_tensors(state.components)


def _components_like(vector, parameters):
    components = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        components.append(vector[offset : offset + count].reshape_as(parameter))
        offset += count
    if offset != vector.numel():
        raise AssertionError("Reference vector does not match parameter layout")
    return tuple(components)


def _assign_parameters(parameters, originals, displacement):
    with torch.no_grad():
        for parameter, original, component in zip(
            parameters, originals, _components_like(displacement, parameters)
        ):
            parameter.copy_(original + component)


def _restore_parameters(parameters, originals):
    with torch.no_grad():
        for parameter, original in zip(parameters, originals):
            parameter.copy_(original)


def _install_flat_gradient(parameters, gradient):
    for parameter, component in zip(
        parameters, _components_like(gradient, parameters)
    ):
        parameter.grad = component.detach().clone()


def _raw_gradient(model, batch):
    parameters = _parameters(model)
    gradients = torch.autograd.grad(_loss(model, batch), parameters)
    return _flat_tensors(gradients)


def _manual_sam_step(model, optimizer, batch, *, rho):
    parameters = _parameters(model)
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    theta_before = _flat_tensors(originals)
    g = _raw_gradient(model, batch)
    epsilon = rho * g / torch.linalg.vector_norm(g)
    _assign_parameters(parameters, originals, epsilon)
    p = _raw_gradient(model, batch)
    _restore_parameters(parameters, originals)
    _install_flat_gradient(parameters, p)
    optimizer.step()
    return {
        "g": g,
        "epsilon": epsilon,
        "p": p,
        "final": p,
        "parameter_delta": _flat_tensors(parameters) - theta_before,
    }


def _manual_sample_step(
    model,
    optimizer,
    batch,
    prior_ema,
    *,
    rho,
    alpha,
    ema_lambda,
):
    parameters = _parameters(model)
    originals = tuple(parameter.detach().clone() for parameter in parameters)
    theta_before = _flat_tensors(originals)
    g = _raw_gradient(model, batch)
    global_estimate = ema_lambda * prior_ema + (1.0 - ema_lambda) * g
    g_norm = torch.linalg.vector_norm(g)
    global_norm = torch.linalg.vector_norm(global_estimate)
    dot = torch.dot(g, global_estimate)
    xi = dot / (g_norm * global_norm)
    sigma = g_norm / global_norm
    coefficient = dot / torch.dot(global_estimate, global_estimate)
    g_b = g - sigma * xi * global_estimate
    epsilon = rho * g / g_norm
    delta = epsilon - alpha * g_b
    _assign_parameters(parameters, originals, delta)
    p = _raw_gradient(model, batch)
    _restore_parameters(parameters, originals)
    final = g + p
    _install_flat_gradient(parameters, final)
    optimizer.step()
    return {
        "g": g,
        "global_estimate": global_estimate,
        "xi": xi,
        "sigma": sigma,
        "coefficient": coefficient,
        "g_b": g_b,
        "epsilon": epsilon,
        "delta": delta,
        "p": p,
        "final": final,
        "parameter_delta": _flat_tensors(parameters) - theta_before,
    }


def _seed_momentum(optimizer, model):
    buffers = (torch.tensor([0.08, -0.03]), torch.tensor(0.05))
    for parameter, buffer in zip(_parameters(model), buffers):
        optimizer.state[parameter]["momentum_buffer"] = buffer.clone()


class PaperEquationReferenceTest(unittest.TestCase):
    def test_sam_matches_raw_autograd_reference_and_parameter_delta(self):
        model = _TwoParameterRegressor()
        reference = copy.deepcopy(model)
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(
            index.parameters, lr=0.04, momentum=0.8, weight_decay=0.03
        )
        reference_optimizer = torch.optim.SGD(
            _parameters(reference), lr=0.04, momentum=0.8, weight_decay=0.03
        )
        _seed_momentum(optimizer, model)
        _seed_momentum(reference_optimizer, reference)
        theta_before = _flat_tensors(_parameters(model))

        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
        )
        record = engine.step_sam(
            _batches()[0], lambda batch: _loss(model, batch)
        )
        expected = _manual_sam_step(
            reference, reference_optimizer, _batches()[0], rho=0.05
        )

        torch.testing.assert_close(_flat_state(record.batch_gradient), expected["g"])
        torch.testing.assert_close(
            _flat_state(record.sam_perturbation), expected["epsilon"]
        )
        torch.testing.assert_close(_flat_state(record.perturbed_gradient), expected["p"])
        torch.testing.assert_close(_flat_state(record.final_gradient), expected["final"])
        torch.testing.assert_close(
            _flat_tensors(_parameters(model)) - theta_before,
            expected["parameter_delta"],
        )
        torch.testing.assert_close(
            _flat_tensors(_parameters(model)), _flat_tensors(_parameters(reference))
        )
        torch.testing.assert_close(optimizer.parameters_at_step[0], theta_before)
        self.assertEqual(optimizer.step_calls, 1)

    def test_sample_ema_matches_two_step_raw_reference_including_geometry(self):
        model = _TwoParameterRegressor()
        reference = copy.deepcopy(model)
        index = ParamIndex.from_model(model)
        optimizer = _CountingSGD(
            index.parameters, lr=0.025, momentum=0.85, weight_decay=0.02
        )
        reference_optimizer = torch.optim.SGD(
            _parameters(reference), lr=0.025, momentum=0.85, weight_decay=0.02
        )
        _seed_momentum(optimizer, model)
        _seed_momentum(reference_optimizer, reference)
        engine = StepEngine(
            param_index=index,
            optimizer=optimizer,
            precision_controller=PrecisionController("fp32"),
            rho=0.05,
            alpha=0.0015,
        )
        estimator = EMAEstimator(index, ema_lambda=0.15)
        manual_ema = torch.zeros(sum(parameter.numel() for parameter in _parameters(reference)))

        for step, batch in enumerate(_batches()):
            with self.subTest(step=step):
                theta_before = _flat_tensors(_parameters(model))
                record = engine.step_sample(
                    batch, lambda materialized: _loss(model, materialized), estimator
                )
                expected = _manual_sample_step(
                    reference,
                    reference_optimizer,
                    batch,
                    manual_ema,
                    rho=0.05,
                    alpha=0.0015,
                    ema_lambda=0.15,
                )
                manual_ema = expected["global_estimate"].clone()

                torch.testing.assert_close(_flat_state(record.batch_gradient), expected["g"])
                torch.testing.assert_close(
                    _flat_state(record.estimator_result.active_global_estimate),
                    expected["global_estimate"],
                )
                self.assertAlmostEqual(record.projection.xi, expected["xi"].item(), places=6)
                self.assertAlmostEqual(
                    record.projection.sigma, expected["sigma"].item(), places=6
                )
                self.assertAlmostEqual(
                    record.projection.projection_coefficient,
                    expected["coefficient"].item(),
                    places=6,
                )
                torch.testing.assert_close(
                    _flat_state(record.projection.batch_component),
                    expected["g_b"],
                    atol=1e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    _flat_state(record.sam_perturbation), expected["epsilon"]
                )
                torch.testing.assert_close(
                    _flat_state(record.total_displacement),
                    expected["delta"],
                    atol=1e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    _flat_state(record.perturbed_gradient),
                    expected["p"],
                    atol=1e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    _flat_state(record.final_gradient),
                    expected["final"],
                    atol=1e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    _flat_tensors(_parameters(model)) - theta_before,
                    expected["parameter_delta"],
                    atol=1e-6,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    _flat_tensors(_parameters(model)),
                    _flat_tensors(_parameters(reference)),
                    atol=1e-6,
                    rtol=1e-5,
                )

        self.assertEqual(optimizer.step_calls, 2)
        self.assertGreater(torch.linalg.vector_norm(expected["g_b"]).item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
