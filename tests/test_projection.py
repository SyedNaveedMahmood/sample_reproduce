import math
import unittest

import torch

from sample_fg.gradient_state import GradientState, GradientStateMismatchError
from sample_fg.param_index import ParamIndex
from sample_fg.projection import (
    DEFAULT_NORM_EPS,
    ProjectionError,
    ProjectionNumericalError,
    project_batch_gradient,
    safe_unit,
)


class _VectorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2))
        self.second = torch.nn.Parameter(torch.zeros(1))


class _IncompatibleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(3))


def _state(index, values, device=None):
    flat = torch.tensor(values, dtype=torch.float32, device=device)
    return GradientState.from_tensors(index, (flat[:2], flat[2:]))


def _flat(state, dtype=torch.float32):
    return torch.cat([component.reshape(-1) for component in state]).to(dtype=dtype)


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.model = _VectorModel()
        self.index = ParamIndex.from_model(self.model)

    def assertStateClose(self, observed, expected, rtol=1e-5, atol=1e-6):
        torch.testing.assert_close(
            _flat(observed), _flat(expected), rtol=rtol, atol=atol
        )

    def assertIdentities(self, g, full, rtol=1e-5, atol=1e-6):
        result = project_batch_gradient(g, full)
        reference_projection = full.scale(result.projection_coefficient)
        self.assertStateClose(result.projected_component, reference_projection, rtol, atol)
        self.assertStateClose(
            result.projected_component + result.batch_component, g, rtol, atol
        )
        scale = max(result.batch_norm * result.full_direction_norm, 1.0)
        self.assertLessEqual(
            abs(result.batch_component.dot(full).item()),
            3e-6 * scale,
        )
        expected_coefficient = result.dot_product / full.squared_norm().item()
        self.assertAlmostEqual(
            result.projection_coefficient, expected_coefficient, places=6
        )
        paper_projection = full.scale(result.sigma * result.xi)
        self.assertStateClose(result.projected_component, paper_projection, rtol, atol)
        return result

    def test_parallel_and_antiparallel_signs(self):
        full = _state(self.index, [1.0, -2.0, 3.0])
        parallel = self.assertIdentities(
            _state(self.index, [2.0, -4.0, 6.0]), full
        )
        self.assertAlmostEqual(parallel.xi, 1.0, places=6)
        self.assertLess(parallel.batch_component.norm().item(), 1e-6)

        antiparallel = self.assertIdentities(
            _state(self.index, [-2.0, 4.0, -6.0]), full
        )
        self.assertAlmostEqual(antiparallel.xi, -1.0, places=6)
        self.assertLess(antiparallel.batch_component.norm().item(), 1e-6)

    def test_orthogonal_acute_and_obtuse_cases(self):
        full = _state(self.index, [1.0, 0.0, 0.0])
        orthogonal_g = _state(self.index, [0.0, 2.0, -1.0])
        orthogonal = self.assertIdentities(orthogonal_g, full)
        self.assertAlmostEqual(orthogonal.xi, 0.0, places=7)
        self.assertAlmostEqual(orthogonal.projection_coefficient, 0.0, places=7)
        self.assertStateClose(orthogonal.batch_component, orthogonal_g)

        acute = self.assertIdentities(_state(self.index, [2.0, 3.0, 1.0]), full)
        obtuse = self.assertIdentities(_state(self.index, [-2.0, 3.0, 1.0]), full)
        self.assertGreater(acute.xi, 0.0)
        self.assertLess(obtuse.xi, 0.0)

    def test_multicomponent_against_flattened_fp64_reference(self):
        g = _state(self.index, [3.25, -4.5, 1.75])
        full = _state(self.index, [-2.0, 0.5, 3.0])
        result = self.assertIdentities(g, full)
        g64 = _flat(g, torch.float64)
        full64 = _flat(full, torch.float64)
        dot64 = torch.dot(g64, full64).item()
        g_norm64 = torch.linalg.vector_norm(g64).item()
        full_norm64 = torch.linalg.vector_norm(full64).item()
        coefficient64 = dot64 / torch.dot(full64, full64).item()
        xi64 = dot64 / (g_norm64 * full_norm64)
        sigma64 = g_norm64 / full_norm64

        self.assertAlmostEqual(result.dot_product, dot64, places=5)
        self.assertAlmostEqual(result.projection_coefficient, coefficient64, places=6)
        self.assertAlmostEqual(result.xi, xi64, places=6)
        self.assertAlmostEqual(result.sigma, sigma64, places=6)
        projection64 = coefficient64 * full64
        residual64 = g64 - projection64
        torch.testing.assert_close(
            _flat(result.projected_component, torch.float64),
            projection64,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            _flat(result.batch_component, torch.float64),
            residual64,
            rtol=1e-6,
            atol=1e-6,
        )

    def test_xi_sigma_and_coefficient_are_distinct(self):
        result = project_batch_gradient(
            _state(self.index, [3.0, 4.0, 0.0]),
            _state(self.index, [2.0, 0.0, 0.0]),
        )
        self.assertAlmostEqual(result.xi, 0.6, places=6)
        self.assertAlmostEqual(result.sigma, 2.5, places=6)
        self.assertAlmostEqual(result.projection_coefficient, 1.5, places=6)
        self.assertEqual(len({result.xi, result.sigma, result.projection_coefficient}), 3)

    def test_zero_batch_full_and_both_degeneracy(self):
        zero = _state(self.index, [0.0, 0.0, 0.0])
        valid = _state(self.index, [1.0, 2.0, 3.0])
        cases = [
            (zero, valid, True, False),
            (valid, zero, False, True),
            (zero, zero, True, True),
        ]
        for g, full, batch_flag, full_flag in cases:
            with self.subTest(batch=batch_flag, full=full_flag):
                result = project_batch_gradient(g, full)
                self.assertEqual(result.batch_gradient_degenerate, batch_flag)
                self.assertEqual(result.full_direction_degenerate, full_flag)
                self.assertEqual(result.xi, 0.0)
                self.assertEqual(result.sigma, 0.0)
                self.assertEqual(result.projection_coefficient, 0.0)
                self.assertEqual(result.projected_component.norm().item(), 0.0)
                self.assertStateClose(result.batch_component, g)

    def test_norm_threshold_below_equal_and_above_uses_less_equal_rule(self):
        threshold = float(torch.tensor(DEFAULT_NORM_EPS, dtype=torch.float32).item())
        center = torch.tensor(threshold, dtype=torch.float32)
        below = float(torch.nextafter(center, torch.tensor(0.0)).item())
        above = float(torch.nextafter(center, torch.tensor(float("inf"))).item())
        full = _state(self.index, [1.0, 0.0, 0.0])

        below_result = project_batch_gradient(
            _state(self.index, [below, 0.0, 0.0]), full, norm_eps=threshold
        )
        equal_result = project_batch_gradient(
            _state(self.index, [threshold, 0.0, 0.0]), full, norm_eps=threshold
        )
        above_result = project_batch_gradient(
            _state(self.index, [above, 0.0, 0.0]), full, norm_eps=threshold
        )
        default_equal = project_batch_gradient(
            _state(self.index, [DEFAULT_NORM_EPS, 0.0, 0.0]), full
        )
        self.assertTrue(below_result.batch_gradient_degenerate)
        self.assertTrue(equal_result.batch_gradient_degenerate)
        self.assertFalse(above_result.batch_gradient_degenerate)
        self.assertTrue(default_equal.batch_gradient_degenerate)
        valid_g = _state(self.index, [1.0, 0.0, 0.0])
        self.assertTrue(
            project_batch_gradient(
                valid_g,
                _state(self.index, [below, 0.0, 0.0]),
                norm_eps=threshold,
            ).full_direction_degenerate
        )
        self.assertTrue(
            project_batch_gradient(
                valid_g,
                _state(self.index, [threshold, 0.0, 0.0]),
                norm_eps=threshold,
            ).full_direction_degenerate
        )
        self.assertFalse(
            project_batch_gradient(
                valid_g,
                _state(self.index, [above, 0.0, 0.0]),
                norm_eps=threshold,
            ).full_direction_degenerate
        )
        self.assertTrue(
            safe_unit(
                _state(self.index, [below, 0.0, 0.0]), norm_eps=threshold
            ).degenerate
        )
        self.assertTrue(
            safe_unit(
                _state(self.index, [threshold, 0.0, 0.0]), norm_eps=threshold
            ).degenerate
        )
        self.assertFalse(
            safe_unit(
                _state(self.index, [above, 0.0, 0.0]), norm_eps=threshold
            ).degenerate
        )

    def test_extreme_finite_scale_disparities(self):
        cases = [
            ([1e10, -2e10, 3e10], [1e-5, 2e-5, -1e-5]),
            ([1e-5, -2e-5, 3e-5], [1e10, 2e10, -1e10]),
        ]
        for g_values, full_values in cases:
            with self.subTest(g=g_values, full=full_values):
                g = _state(self.index, g_values)
                full = _state(self.index, full_values)
                result = self.assertIdentities(g, full, rtol=3e-5, atol=1e-5)
                scalars = (
                    result.batch_norm,
                    result.full_direction_norm,
                    result.dot_product,
                    result.xi,
                    result.sigma,
                    result.projection_coefficient,
                )
                self.assertTrue(all(math.isfinite(value) for value in scalars))
                self.assertTrue(result.projected_component.is_finite())
                self.assertTrue(result.batch_component.is_finite())

    def test_nonfinite_inputs_and_invalid_threshold_fail_loudly(self):
        valid = _state(self.index, [1.0, 2.0, 3.0])
        for value in (float("nan"), float("inf"), float("-inf")):
            bad_g = _state(self.index, [value, 0.0, 0.0])
            bad_full = _state(self.index, [value, 0.0, 0.0])
            with self.assertRaises(ProjectionNumericalError):
                project_batch_gradient(bad_g, valid)
            with self.assertRaises(ProjectionNumericalError):
                project_batch_gradient(valid, bad_full)
        for invalid in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ProjectionError):
                project_batch_gradient(valid, valid, norm_eps=invalid)
        finite_but_overflowing = _state(self.index, [3e38, 3e38, 3e38])
        self.assertTrue(finite_but_overflowing.is_finite())
        with self.assertRaises(ProjectionNumericalError):
            project_batch_gradient(finite_but_overflowing, valid)

    def test_fingerprint_and_malformed_state_rejection(self):
        valid = _state(self.index, [1.0, 2.0, 3.0])
        incompatible_index = ParamIndex.from_model(_IncompatibleModel())
        incompatible = GradientState.from_tensors(
            incompatible_index, [torch.ones(3)]
        )
        with self.assertRaises(GradientStateMismatchError):
            project_batch_gradient(valid, incompatible)

        malformed = valid.clone()
        malformed[0].resize_(3)
        with self.assertRaises(ValueError):
            project_batch_gradient(malformed, valid)

    def test_outputs_are_owned_detached_and_do_not_alias_inputs(self):
        g = _state(self.index, [3.0, 4.0, 1.0])
        full = _state(self.index, [2.0, -1.0, 2.0])
        g_before = g.clone()
        full_before = full.clone()
        result = project_batch_gradient(g, full)
        for state in (result.projected_component, result.batch_component):
            self.assertTrue(all(not item.requires_grad for item in state))
            self.assertTrue(all(item.grad_fn is None for item in state))
        result.projected_component[0].add_(10.0)
        result.batch_component[1].mul_(0.0)
        self.assertStateClose(g, g_before)
        self.assertStateClose(full, full_before)

    def test_safe_unit_valid_and_degenerate_without_perturbation(self):
        state = _state(self.index, [3.0, 4.0, 0.0])
        state_before = state.clone()
        result = safe_unit(state)
        self.assertFalse(result.degenerate)
        self.assertAlmostEqual(result.norm, 5.0, places=6)
        self.assertAlmostEqual(result.unit.norm().item(), 1.0, places=6)
        self.assertTrue(
            all(
                not item.requires_grad and item.grad_fn is None
                for item in result.unit
            )
        )
        result.unit[0].add_(10.0)
        self.assertStateClose(state, state_before)

        zero = _state(self.index, [0.0, 0.0, 0.0])
        zero_result = safe_unit(zero)
        self.assertTrue(zero_result.degenerate)
        self.assertEqual(zero_result.unit.norm().item(), 0.0)
        with self.assertRaises(ProjectionNumericalError):
            safe_unit(_state(self.index, [float("nan"), 0.0, 0.0]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_tiny_cuda_projection_and_device_mismatch(self):
        cuda_model = _VectorModel().cuda()
        cuda_index = ParamIndex.from_model(cuda_model)
        g = _state(cuda_index, [3.0, 4.0, 1.0], device="cuda")
        full = _state(cuda_index, [2.0, -1.0, 2.0], device="cuda")
        result = self.assertIdentities(g, full)
        self.assertTrue(all(item.device.type == "cuda" for item in result.batch_component))
        self.assertTrue(all(item.dtype == torch.float32 for item in result.batch_component))
        self.assertFalse(safe_unit(g).degenerate)

        cpu_same_structure = _state(self.index, [2.0, -1.0, 2.0])
        with self.assertRaises(GradientStateMismatchError):
            project_batch_gradient(g, cpu_same_structure)


if __name__ == "__main__":
    unittest.main()
