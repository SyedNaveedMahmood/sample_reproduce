import tempfile
import unittest
from pathlib import Path

import torch

from sample_fg.gradient_state import (
    GradientState,
    GradientStateError,
    GradientStateMismatchError,
    MissingGradientError,
)
from sample_fg.param_index import ParamIndex


class _TwoParameterModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype))
        self.vector = torch.nn.Parameter(torch.zeros(4, dtype=dtype))
        self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class _IncompatibleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.nn.Parameter(torch.zeros(3, 2))
        self.vector = torch.nn.Parameter(torch.zeros(4))


def _flatten(state):
    return torch.cat([component.reshape(-1) for component in state.components])


class GradientStateTest(unittest.TestCase):
    def setUp(self):
        self.model = _TwoParameterModel()
        self.index = ParamIndex.from_model(self.model)
        generator = torch.Generator().manual_seed(7)
        self.left_tensors = [
            torch.randn(entry.shape, generator=generator) for entry in self.index
        ]
        self.right_tensors = [
            torch.randn(entry.shape, generator=generator) for entry in self.index
        ]
        self.left = GradientState.from_tensors(self.index, self.left_tensors)
        self.right = GradientState.from_tensors(self.index, self.right_tensors)

    def test_structure_validation_rejects_count_shape_type_and_sparse(self):
        with self.assertRaises(GradientStateError):
            GradientState.from_tensors(self.index, self.left_tensors[:1])
        wrong_shape = [torch.zeros(3, 2), torch.zeros(4)]
        with self.assertRaises(GradientStateError):
            GradientState.from_tensors(self.index, wrong_shape)
        with self.assertRaises(GradientStateError):
            GradientState.from_tensors(self.index, ["not-a-tensor", torch.zeros(4)])
        sparse = torch.zeros(2, 3).to_sparse()
        with self.assertRaises(GradientStateError):
            GradientState.from_tensors(self.index, [sparse, torch.zeros(4)])

    def test_explicit_zero_state_is_fp32_detached_and_shape_preserving(self):
        state = GradientState.zeros(self.index)
        self.assertEqual(len(state), 2)
        self.assertEqual([tuple(item.shape) for item in state], [(2, 3), (4,)])
        self.assertTrue(all(item.dtype == torch.float32 for item in state))
        self.assertTrue(all(not item.requires_grad for item in state))
        self.assertTrue(all(item.grad_fn is None for item in state))
        self.assertEqual(state.total_numel, 10)
        self.assertEqual(state.raw_tensor_bytes, 40)

    def test_fp16_grad_capture_owns_storage_and_survives_grad_clear(self):
        model = _TwoParameterModel(dtype=torch.float16)
        index = ParamIndex.from_model(model)
        model.matrix.grad = torch.full_like(model.matrix, 2.0)
        model.vector.grad = torch.full_like(model.vector, 3.0)
        matrix_grad_before = model.matrix.grad.clone()
        vector_grad_before = model.vector.grad.clone()

        state = GradientState.from_parameter_grads(index)
        self.assertTrue(torch.equal(model.matrix.grad, matrix_grad_before))
        self.assertTrue(torch.equal(model.vector.grad, vector_grad_before))
        self.assertTrue(all(component.dtype == torch.float32 for component in state))
        snapshot = state.clone()

        model.matrix.grad.zero_()
        model.vector.grad = None
        self.assertTrue(torch.equal(state[0], snapshot[0]))
        self.assertTrue(torch.equal(state[1], snapshot[1]))
        state[0].add_(5.0)
        self.assertTrue(torch.equal(model.matrix.grad, torch.zeros_like(model.matrix)))

    def test_input_tensor_clone_and_result_ownership(self):
        state = GradientState.from_tensors(self.index, self.left_tensors)
        before = state.clone()
        self.left_tensors[0].add_(100.0)
        self.assertTrue(torch.equal(state[0], before[0]))

        clone = state.clone()
        clone[0].mul_(0.0)
        self.assertFalse(torch.equal(clone[0], state[0]))

        result = state + self.right
        result[0].add_(10.0)
        self.assertFalse(torch.equal(result[0], state[0]))
        self.assertFalse(torch.equal(result[0], self.right[0]))

    def test_arithmetic_dot_and_norm_match_flat_reference(self):
        flat_left = _flatten(self.left)
        flat_right = _flatten(self.right)

        self.assertTrue(
            torch.allclose(_flatten(self.left + self.right), flat_left + flat_right)
        )
        self.assertTrue(
            torch.allclose(_flatten(self.left - self.right), flat_left - flat_right)
        )
        self.assertTrue(torch.allclose(_flatten(self.left * 2.5), flat_left * 2.5))
        self.assertTrue(torch.allclose(_flatten(2.5 * self.left), flat_left * 2.5))
        self.assertTrue(
            torch.allclose(
                self.left.dot(self.right),
                torch.dot(flat_left, flat_right),
                rtol=1e-5,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                self.left.squared_norm(),
                torch.dot(flat_left, flat_left),
                rtol=1e-5,
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                self.left.norm(),
                torch.linalg.vector_norm(flat_left),
                rtol=1e-5,
                atol=1e-6,
            )
        )

    def test_accumulation_matches_reference_and_does_not_touch_grads(self):
        self.model.matrix.grad = torch.ones_like(self.model.matrix)
        self.model.vector.grad = torch.ones_like(self.model.vector)
        grad_snapshots = [parameter.grad.clone() for parameter in self.index.parameters]
        target = self.left.clone()
        reference = _flatten(self.left) + 0.25 * _flatten(self.right)
        returned = target.accumulate_(self.right, weight=0.25)
        self.assertIs(returned, target)
        self.assertTrue(torch.allclose(_flatten(target), reference))
        for parameter, before in zip(self.index.parameters, grad_snapshots):
            self.assertTrue(torch.equal(parameter.grad, before))

    def test_affine_update_matches_flat_reference(self):
        target = self.left.clone()
        reference = 0.15 * _flatten(self.left) + 0.85 * _flatten(self.right)
        returned = target.affine_(self.right, self_weight=0.15, other_weight=0.85)
        self.assertIs(returned, target)
        self.assertTrue(
            torch.allclose(_flatten(target), reference, rtol=1e-6, atol=1e-7)
        )

    def test_named_serialization_roundtrip_preserves_structure_and_values(self):
        payload = self.left.state_dict()
        self.assertEqual(
            tuple(payload["components"]), ("matrix", "vector")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gradient_state.pt"
            torch.save(payload, path)
            loaded = torch.load(path)
        restored = GradientState.from_state_dict(self.index, loaded)
        self.assertEqual(restored.param_index.fingerprint, self.index.fingerprint)
        for observed, expected in zip(restored, self.left):
            self.assertTrue(torch.equal(observed, expected))

        payload["components"]["matrix"].zero_()
        self.assertFalse(torch.equal(payload["components"]["matrix"], self.left[0]))

    def test_serialization_rejects_wrong_fingerprint_or_names(self):
        wrong_fingerprint = self.left.state_dict()
        wrong_fingerprint["param_index_fingerprint"] = "0" * 64
        with self.assertRaises(GradientStateMismatchError):
            GradientState.from_state_dict(self.index, wrong_fingerprint)

        wrong_names = self.left.state_dict()
        wrong_names["components"]["unexpected"] = wrong_names["components"].pop(
            "matrix"
        )
        with self.assertRaises(GradientStateMismatchError):
            GradientState.from_state_dict(self.index, wrong_names)

    def test_finite_nan_and_infinity_detection(self):
        self.assertTrue(self.left.is_finite())
        nan_state = self.left.clone()
        nan_state[0][0, 0] = float("nan")
        self.assertFalse(nan_state.is_finite())
        pos_inf = self.left.clone()
        pos_inf[1][0] = float("inf")
        self.assertFalse(pos_inf.is_finite())
        neg_inf = self.left.clone()
        neg_inf[1][0] = float("-inf")
        self.assertFalse(neg_inf.is_finite())

    def test_incompatible_index_is_rejected(self):
        incompatible_model = _IncompatibleModel()
        incompatible_index = ParamIndex.from_model(incompatible_model)
        incompatible = GradientState.zeros(incompatible_index)
        with self.assertRaises(GradientStateMismatchError):
            self.left.add(incompatible)
        with self.assertRaises(GradientStateMismatchError):
            self.left.dot(incompatible)

    def test_missing_gradient_fails_instead_of_becoming_zero(self):
        self.model.matrix.grad = torch.ones_like(self.model.matrix)
        self.model.vector.grad = None
        with self.assertRaises(MissingGradientError):
            GradientState.from_parameter_grads(self.index)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tiny_cuda_state_and_cross_device_rejection(self):
        cuda_model = _TwoParameterModel().cuda()
        cuda_index = ParamIndex.from_model(cuda_model)
        cuda_state = GradientState.from_tensors(
            cuda_index,
            [torch.ones(entry.shape, device="cuda") for entry in cuda_index],
        )
        doubled = cuda_state + cuda_state
        self.assertEqual(doubled.devices, (torch.device("cuda:0"),) * 2)
        self.assertAlmostEqual(doubled.norm().item(), (40.0) ** 0.5, places=5)

        cpu_same_structure = GradientState.zeros(
            ParamIndex.from_model(_TwoParameterModel())
        )
        with self.assertRaises(GradientStateMismatchError):
            cuda_state.add(cpu_same_structure)


if __name__ == "__main__":
    unittest.main()
