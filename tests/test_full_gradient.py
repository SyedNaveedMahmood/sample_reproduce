import gc
import random
import unittest
import weakref
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from sample_fg.full_gradient import (
    FullGradientNumericalError,
    FullGradientService,
)
from sample_fg.param_index import ParamIndex, ParamIndexMismatchError
from sample_fg.precision import PrecisionController


class _TensorDataset(Dataset):
    def __init__(self, inputs, labels, *, stochastic=False):
        self.inputs = inputs
        self.labels = labels
        self.stochastic = stochastic

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        image = self.inputs[index].clone()
        if self.stochastic:
            noise = random.random() + float(np.random.random())
            noise += float(torch.rand(()).item())
            image = image + noise * 0.01
        return {
            "img": image,
            "label": self.labels[index].clone(),
            "sample_id": f"sample/{index}.dat",
        }


class _TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[0.25, -0.35], [0.5, 0.1], [-0.2, 0.4]],
                dtype=torch.float32,
            )
        )
        self.register_parameter(
            "frozen_bias",
            nn.Parameter(torch.tensor([0.1, -0.1]), requires_grad=False),
        )

    def forward(self, image):
        return image @ self.weight + self.frozen_bias


class _FiniteLossNonfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.shape = tensor.shape
        return tensor.sum() * 0.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full(ctx.shape, float("inf"), device=grad_output.device)


def _loader(dataset, batch_size):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(777)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=0,
        drop_last=False,
        generator=generator,
    )


def _fixture():
    inputs = torch.tensor(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [0.5, 0.5, 1.0],
            [1.5, -0.5, 0.25],
            [-0.25, 0.75, 1.25],
            [2.0, 0.5, -1.0],
            [-1.0, 1.0, 0.75],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1], dtype=torch.long)
    return inputs, labels


def _service(model, dataset, batch_size, *, config_hash="fixture-v1"):
    index = ParamIndex.from_model(model)
    return FullGradientService(
        model=model,
        param_index=index,
        loader=_loader(dataset, batch_size),
        precision_controller=PrecisionController("fp32"),
        protocol_seed=1,
        dataset="fixture",
        shots=1,
        config_hash=config_hash,
    )


class FullGradientServiceTest(unittest.TestCase):
    def test_exact_sample_mean_matches_direct_reference_with_short_final(self):
        inputs, labels = _fixture()
        dataset = _TensorDataset(inputs, labels)
        model = _TinyClassifier()
        service = _service(model, dataset, 3)

        direct_loss = F.cross_entropy(model(inputs), labels)
        direct_gradient, = torch.autograd.grad(direct_loss, (model.weight,))
        result = service.compute(optimizer_step=0, purpose="estimator_refresh")

        torch.testing.assert_close(
            result.gradient[0], direct_gradient, rtol=1e-5, atol=1e-7
        )
        self.assertAlmostEqual(result.mean_loss, float(direct_loss.item()), places=7)
        self.assertEqual(result.metadata.observed_micro_batch_sizes, (3, 3, 1))
        self.assertEqual(result.metadata.sample_count, 7)
        self.assertEqual(result.metadata.forward_calls, 3)
        self.assertEqual(result.metadata.autograd_grad_calls, 3)

    def test_unweighted_microbatch_average_is_distinguishably_wrong(self):
        inputs, labels = _fixture()
        model = _TinyClassifier()
        result = _service(model, _TensorDataset(inputs, labels), 3).compute(
            optimizer_step=0, purpose="estimator_refresh"
        )

        gradients = []
        for start in (0, 3, 6):
            stop = min(start + 3, len(inputs))
            loss = F.cross_entropy(model(inputs[start:stop]), labels[start:stop])
            gradient, = torch.autograd.grad(loss, (model.weight,))
            gradients.append(gradient)
        wrong = sum(gradients) / len(gradients)
        self.assertGreater(
            float(torch.linalg.vector_norm(result.gradient[0] - wrong).item()),
            1e-3,
        )

    def test_microbatch_invariance(self):
        inputs, labels = _fixture()
        model = _TinyClassifier()
        states = []
        losses = []
        for batch_size in (1, 3, 7):
            result = _service(
                model,
                _TensorDataset(inputs, labels, stochastic=True),
                batch_size,
            ).compute(optimizer_step=4, purpose="diagnostic")
            states.append(result.gradient)
            losses.append(result.mean_loss)
        for state in states[1:]:
            torch.testing.assert_close(state[0], states[0][0], rtol=1e-5, atol=1e-7)
        for loss in losses[1:]:
            self.assertAlmostEqual(loss, losses[0], places=7)

    def test_uses_first_order_autograd_grad_and_releases_loss_graphs(self):
        inputs, labels = _fixture()
        model = _TinyClassifier()
        loss_refs = []

        def loss_fn(candidate, batch):
            loss = F.cross_entropy(candidate(batch["img"]), batch["label"])
            loss_refs.append(weakref.ref(loss))
            return loss

        service = FullGradientService(
            model=model,
            param_index=ParamIndex.from_model(model),
            loader=_loader(_TensorDataset(inputs, labels), 3),
            precision_controller=PrecisionController("fp32"),
            protocol_seed=1,
            dataset="fixture",
            shots=1,
            config_hash="fixture-v1",
            mean_loss_fn=loss_fn,
        )
        original_grad = torch.autograd.grad
        calls = []

        def recording_grad(*args, **kwargs):
            calls.append(dict(kwargs))
            return original_grad(*args, **kwargs)

        with mock.patch("torch.autograd.grad", side_effect=recording_grad):
            result = service.compute(optimizer_step=0, purpose="diagnostic")
        gc.collect()
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(call["create_graph"] is False for call in calls))
        self.assertTrue(all(call["retain_graph"] is False for call in calls))
        self.assertTrue(all(reference() is None for reference in loss_refs))
        self.assertTrue(result.gradient.is_finite())

    def test_output_is_owned_fp32_detached_and_live_grads_are_untouched(self):
        inputs, labels = _fixture()
        model = _TinyClassifier()
        live_grad = torch.full_like(model.weight, 7.0)
        model.weight.grad = live_grad.clone()
        prompt_before = model.weight.detach().clone()
        frozen_before = model.frozen_bias.detach().clone()
        service = _service(model, _TensorDataset(inputs, labels), 4)
        result = service.compute(optimizer_step=0, purpose="diagnostic")

        self.assertEqual(result.gradient[0].dtype, torch.float32)
        self.assertFalse(result.gradient[0].requires_grad)
        self.assertIsNone(result.gradient[0].grad_fn)
        self.assertTrue(torch.equal(model.weight.grad, live_grad))
        self.assertTrue(torch.equal(model.weight.detach(), prompt_before))
        self.assertTrue(torch.equal(model.frozen_bias.detach(), frozen_before))
        model.weight.grad.zero_()
        self.assertGreater(float(result.gradient.norm().item()), 0.0)

    def test_nonfinite_loss_and_gradient_fail_fast(self):
        inputs, labels = _fixture()
        model = _TinyClassifier()

        def nan_loss(candidate, batch):
            return candidate(batch["img"]).sum() * torch.tensor(float("nan"))

        service = FullGradientService(
            model=model,
            param_index=ParamIndex.from_model(model),
            loader=_loader(_TensorDataset(inputs, labels), 3),
            precision_controller=PrecisionController("fp32"),
            protocol_seed=1,
            dataset="fixture",
            shots=1,
            config_hash="fixture-v1",
            mean_loss_fn=nan_loss,
        )
        with self.assertRaises(FullGradientNumericalError):
            service.compute(optimizer_step=0, purpose="diagnostic")

        def finite_loss_bad_gradient(candidate, batch):
            ordinary = F.cross_entropy(candidate(batch["img"]), batch["label"])
            return ordinary + _FiniteLossNonfiniteGradient.apply(candidate.weight)

        gradient_service = FullGradientService(
            model=model,
            param_index=ParamIndex.from_model(model),
            loader=_loader(_TensorDataset(inputs, labels), 3),
            precision_controller=PrecisionController("fp32"),
            protocol_seed=1,
            dataset="fixture",
            shots=1,
            config_hash="fixture-v1",
            mean_loss_fn=finite_loss_bad_gradient,
        )
        with self.assertRaises(FullGradientNumericalError):
            gradient_service.compute(optimizer_step=0, purpose="diagnostic")

    def test_param_index_must_reference_the_queried_model(self):
        inputs, labels = _fixture()
        indexed_model = _TinyClassifier()
        queried_model = _TinyClassifier()
        service = FullGradientService(
            model=queried_model,
            param_index=ParamIndex.from_model(indexed_model),
            loader=_loader(_TensorDataset(inputs, labels), 3),
            precision_controller=PrecisionController("fp32"),
            protocol_seed=1,
            dataset="fixture",
            shots=1,
            config_hash="fixture-v1",
        )
        with self.assertRaises(ParamIndexMismatchError):
            service.compute(optimizer_step=0, purpose="diagnostic")

    def test_isolated_rng_reproducibility_and_outside_continuation(self):
        inputs, labels = _fixture()
        model = _TinyClassifier()
        random.seed(55)
        np.random.seed(55)
        torch.manual_seed(55)
        expected_python_state = random.getstate()
        expected_numpy_state = np.random.get_state()
        expected_torch_state = torch.get_rng_state().clone()

        first = _service(
            model, _TensorDataset(inputs, labels, stochastic=True), 3
        ).compute(optimizer_step=2, purpose="diagnostic")
        self.assertEqual(random.getstate(), expected_python_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1], expected_numpy_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))
        second = _service(
            model, _TensorDataset(inputs, labels, stochastic=True), 3
        ).compute(optimizer_step=2, purpose="diagnostic")
        torch.testing.assert_close(first.gradient[0], second.gradient[0], rtol=0, atol=0)
        self.assertEqual(first.metadata.seed.sha256, second.metadata.seed.sha256)


if __name__ == "__main__":
    unittest.main()
