import random
import unittest

import numpy as np
import torch

from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import (
    ParameterSnapshot,
    PerturbationError,
    PerturbationNumericalError,
    PerturbationStateError,
    PromptPerturbation,
    temporary_prompt_perturbation,
)


class _TwoParameterModel(torch.nn.Module):
    def __init__(self, *, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.first = torch.nn.Parameter(
            torch.tensor([1.0, -2.0], dtype=dtype, device=device)
        )
        self.second = torch.nn.Parameter(
            torch.tensor([[3.0]], dtype=dtype, device=device)
        )
        self.register_parameter(
            "frozen",
            torch.nn.Parameter(
                torch.tensor([9.0], dtype=dtype, device=device),
                requires_grad=False,
            ),
        )


class _DifferentModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.nn.Parameter(torch.ones(3))


def _state(index, first=(0.25, -0.5), second=(1.5,)):
    return GradientState.from_tensors(
        index,
        (
            torch.tensor(first, dtype=torch.float32, device=index[0].device),
            torch.tensor(second, dtype=torch.float32, device=index[1].device).reshape(1, 1),
        ),
    )


class PerturbationTest(unittest.TestCase):
    def setUp(self):
        self.model = _TwoParameterModel()
        self.index = ParamIndex.from_model(self.model)
        self.displacement = _state(self.index)

    def test_snapshot_is_owned_exact_dtype_shape_and_independent(self):
        snapshot = ParameterSnapshot.capture(self.index)
        self.assertEqual(tuple(value.dtype for value in snapshot), (torch.float32,) * 2)
        self.assertEqual(tuple(tuple(value.shape) for value in snapshot), ((2,), (1, 1)))
        self.model.first.data.add_(10.0)
        torch.testing.assert_close(snapshot[0], torch.tensor([1.0, -2.0]))
        self.assertNotEqual(snapshot[0].data_ptr(), self.model.first.data_ptr())

    def test_multicomponent_inside_value_identity_and_exact_restoration(self):
        controller = PromptPerturbation(self.index)
        identities = tuple(id(parameter) for parameter in self.index.parameters)
        original = tuple(parameter.detach().clone() for parameter in self.index.parameters)
        frozen = self.model.frozen.detach().clone()
        for parameter in self.index.parameters:
            parameter.grad = torch.full_like(parameter, 7.0)
        grads = tuple(parameter.grad.clone() for parameter in self.index.parameters)

        with controller.displaced(self.displacement) as snapshot:
            self.assertTrue(controller.active)
            for entry, before, delta in zip(self.index, snapshot, self.displacement):
                torch.testing.assert_close(
                    entry.parameter,
                    before + delta.to(dtype=entry.parameter.dtype),
                    rtol=0,
                    atol=0,
                )
            with self.assertRaises(PerturbationStateError):
                with controller.displaced(self.displacement):
                    pass

        self.assertFalse(controller.active)
        self.assertEqual(identities, tuple(id(p) for p in self.index.parameters))
        for parameter, before in zip(self.index.parameters, original):
            self.assertTrue(torch.equal(parameter, before))
        for parameter, before in zip(self.index.parameters, grads):
            self.assertTrue(torch.equal(parameter.grad, before))
        self.assertTrue(torch.equal(self.model.frozen, frozen))

    def test_restore_uses_snapshot_not_inverse_arithmetic(self):
        originals = tuple(parameter.detach().clone() for parameter in self.index.parameters)
        with temporary_prompt_perturbation(self.index, self.displacement):
            with torch.no_grad():
                for parameter in self.index.parameters:
                    parameter.fill_(123.0)
        for parameter, original in zip(self.index.parameters, originals):
            self.assertTrue(torch.equal(parameter, original))

    def test_exception_restores_and_original_exception_propagates(self):
        originals = tuple(parameter.detach().clone() for parameter in self.index.parameters)
        with self.assertRaisesRegex(RuntimeError, "injected displaced failure"):
            with PromptPerturbation(self.index).displaced(self.displacement):
                raise RuntimeError("injected displaced failure")
        for parameter, original in zip(self.index.parameters, originals):
            self.assertTrue(torch.equal(parameter, original))

    def test_sequential_reentry_is_allowed_after_restoration(self):
        controller = PromptPerturbation(self.index)
        original = self.model.first.detach().clone()
        for _ in range(2):
            with controller.displaced(self.displacement):
                self.assertTrue(controller.active)
            self.assertFalse(controller.active)
            self.assertTrue(torch.equal(self.model.first, original))

    def test_rng_is_observationally_unchanged(self):
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        with PromptPerturbation(self.index).displaced(self.displacement):
            pass
        observed = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(expected[0], observed[0])
        self.assertEqual(expected[1], observed[1])
        self.assertTrue(torch.equal(expected[2], observed[2]))

    def test_malformed_incompatible_and_nonfinite_displacements_rejected(self):
        incompatible_index = ParamIndex.from_model(_DifferentModel())
        incompatible = GradientState.from_tensors(
            incompatible_index, (torch.ones(3),)
        )
        with self.assertRaises(PerturbationError):
            with PromptPerturbation(self.index).displaced(incompatible):
                pass

        bad = self.displacement.clone()
        bad[0][0] = float("nan")
        with self.assertRaises(PerturbationNumericalError):
            with PromptPerturbation(self.index).displaced(bad):
                pass

        malformed = self.displacement.clone()
        malformed[0].resize_(3)
        with self.assertRaises(ValueError):
            with PromptPerturbation(self.index).displaced(malformed):
                pass

    def test_fp16_application_and_bitwise_restore(self):
        model = _TwoParameterModel(dtype=torch.float16)
        index = ParamIndex.from_model(model)
        displacement = _state(index)
        originals = tuple(parameter.detach().clone() for parameter in index.parameters)
        with PromptPerturbation(index).displaced(displacement):
            self.assertTrue(any(not torch.equal(p, b) for p, b in zip(index.parameters, originals)))
        for parameter, original in zip(index.parameters, originals):
            self.assertTrue(torch.equal(parameter, original))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_device_preserved_and_cross_device_rejected(self):
        model = _TwoParameterModel(device="cuda")
        index = ParamIndex.from_model(model)
        displacement = _state(index)
        originals = tuple(parameter.detach().clone() for parameter in index.parameters)
        with PromptPerturbation(index).displaced(displacement):
            self.assertTrue(all(parameter.device.type == "cuda" for parameter in index.parameters))
        for parameter, original in zip(index.parameters, originals):
            self.assertTrue(torch.equal(parameter, original))

        cpu_same_structure = _state(self.index)
        with self.assertRaises(PerturbationError):
            with PromptPerturbation(index).displaced(cpu_same_structure):
                pass


if __name__ == "__main__":
    unittest.main()
