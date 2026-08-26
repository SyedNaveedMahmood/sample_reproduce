import copy
import math
import unittest
from unittest import mock

import torch

from sample_fg.diagnostics import (
    DIAGNOSTIC_SCHEMA_VERSION,
    LEGACY_BATCH_COMPONENT_EXACT_COSINE_SEMANTICS,
    DiagnosticError,
    DiagnosticNumericalError,
    compute_gradient_diagnostics,
)
from sample_fg.gradient_state import GradientState, GradientStateMismatchError
from sample_fg.param_index import ParamIndex
from sample_fg.projection import project_batch_gradient


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2))
        self.b = torch.nn.Parameter(torch.zeros(2))


class _Other(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.nn.Parameter(torch.zeros(4))


def _state(index, values):
    flat = torch.tensor(values, dtype=torch.float32)
    return GradientState.from_tensors(index, (flat[:2], flat[2:]))


def _flat(state, dtype=torch.float64):
    return torch.cat([component.reshape(-1) for component in state]).to(dtype)


class GradientDiagnosticsTest(unittest.TestCase):
    def setUp(self):
        self.model = _Model()
        self.index = ParamIndex.from_model(self.model)

    def metrics(self, g, active, p, exact=None, alpha=0.2):
        projection = project_batch_gradient(g, active)
        return compute_gradient_diagnostics(
            batch_gradient=g,
            active_global_estimate=active,
            projection=projection,
            perturbed_gradient=p,
            exact_full_gradient=exact,
            alpha=alpha,
        )

    def test_full_schema_and_exact_equal_estimator_reference(self):
        g = _state(self.index, [2.0, 3.0, -1.0, 4.0])
        exact = _state(self.index, [1.0, -2.0, 0.5, 3.0])
        p = _state(self.index, [-1.0, 1.0, 2.0, 0.5])
        result = self.metrics(g, exact, p, exact)
        self.assertEqual(result["schema_version"], DIAGNOSTIC_SCHEMA_VERSION)
        self.assertAlmostEqual(result["grad/global_estimate_exact_cosine"], 1.0, places=6)
        self.assertAlmostEqual(result["grad/global_estimate_exact_norm_ratio"], 1.0, places=6)
        self.assertAlmostEqual(result["grad/global_estimate_exact_log_norm_ratio"], 0.0, places=7)
        self.assertAlmostEqual(result["grad/global_estimate_exact_relative_l2"], 0.0, places=7)
        self.assertNotEqual(
            "grad/batch_component_estimator_cosine",
            "grad/perturbed_gradient_estimator_cosine",
        )
        required = {
            "grad/batch_component_estimator_cosine",
            "grad/batch_component_exact_cosine",
            "grad/batch_component_estimate_exact_cosine",
            "grad/batch_component_estimate_exact_relative_l2",
            "grad/batch_component_estimate_exact_norm_ratio",
            "grad/reference_batch_component_exact_cosine",
            "grad/perturbed_gradient_estimator_cosine",
            "grad/perturbed_gradient_exact_cosine",
            "grad/perturbed_gradient_batch_component_cosine",
            "grad/perturbed_gradient_batch_cosine",
            "taylor/exploitation_term",
            "taylor/exploration_term",
            "taylor/joint_alignment_term",
        }
        self.assertTrue(required.issubset(result))

    def test_fp64_reference_for_inaccurate_estimator_and_raw_values(self):
        g = _state(self.index, [3.0, -1.0, 4.0, 2.0])
        active = _state(self.index, [1.0, 2.0, -1.0, 0.5])
        exact = _state(self.index, [-2.0, 0.5, 3.0, 1.0])
        p = _state(self.index, [0.25, -4.0, 2.0, 3.0])
        result = self.metrics(g, active, p, exact)
        vectors = [_flat(value) for value in (g, active, exact, p)]
        g64, active64, exact64, p64 = vectors
        cosine = torch.dot(active64, exact64) / (
            torch.linalg.vector_norm(active64) * torch.linalg.vector_norm(exact64)
        )
        self.assertAlmostEqual(result["grad/global_estimate_exact_cosine"], cosine.item(), places=6)
        self.assertAlmostEqual(result["raw/dot_batch_global"], torch.dot(g64, active64).item(), places=5)
        self.assertAlmostEqual(result["raw/dot_batch_exact"], torch.dot(g64, exact64).item(), places=5)
        self.assertAlmostEqual(result["raw/dot_global_exact"], torch.dot(active64, exact64).item(), places=5)
        self.assertAlmostEqual(result["raw/dot_perturbed_exact"], torch.dot(p64, exact64).item(), places=5)

    def test_orthogonal_and_antialigned_estimator_exact_signs(self):
        g = _state(self.index, [1.0, 2.0, 3.0, 4.0])
        p = _state(self.index, [1.0, 0.0, 0.0, 0.0])
        exact = _state(self.index, [1.0, 0.0, 0.0, 0.0])
        orthogonal = self.metrics(
            g, _state(self.index, [0.0, 1.0, 0.0, 0.0]), p, exact
        )
        anti = self.metrics(g, exact.scale(-1.0), p, exact)
        self.assertAlmostEqual(orthogonal["grad/global_estimate_exact_cosine"], 0.0, places=7)
        self.assertAlmostEqual(anti["grad/global_estimate_exact_cosine"], -1.0, places=7)

    def test_distinct_construction_and_perturbed_alignments(self):
        g = _state(self.index, [2.0, 3.0, 0.0, 0.0])
        active = _state(self.index, [1.0, 0.0, 0.0, 0.0])
        projection = project_batch_gradient(g, active)
        along_batch_component = self.metrics(g, active, projection.batch_component, active)
        along_global = self.metrics(g, active, active, active)
        anti_global = self.metrics(g, active, active.scale(-1.0), active)
        self.assertAlmostEqual(along_batch_component["grad/perturbed_gradient_batch_component_cosine"], 1.0, places=6)
        self.assertAlmostEqual(along_batch_component["grad/perturbed_gradient_estimator_cosine"], 0.0, places=6)
        self.assertAlmostEqual(along_global["grad/perturbed_gradient_estimator_cosine"], 1.0, places=6)
        self.assertAlmostEqual(anti_global["grad/perturbed_gradient_estimator_cosine"], -1.0, places=6)

    def test_reference_projection_and_estimator_leakage_are_separate(self):
        g = _state(self.index, [2.0, 3.0, 4.0, 1.0])
        active = _state(self.index, [1.0, 0.0, 0.0, 0.0])
        exact = _state(self.index, [0.0, 1.0, 1.0, 0.0])
        p = _state(self.index, [1.0, 1.0, 0.0, 0.0])
        result = self.metrics(g, active, p, exact)
        self.assertLess(abs(result["grad/reference_batch_component_exact_cosine"]), 1e-6)
        self.assertGreater(abs(result["grad/batch_component_exact_cosine"]), 0.1)
        self.assertLess(abs(result["grad/batch_component_estimator_cosine"]), 1e-6)

    def test_projected_component_fidelity_matches_independent_reference(self):
        g = _state(self.index, [2.0, 3.0, 4.0, 1.0])
        active = _state(self.index, [1.0, 0.0, 0.0, 0.0])
        exact = _state(self.index, [0.0, 1.0, 1.0, 0.0])
        p = _state(self.index, [1.0, 1.0, 0.0, 0.0])
        result = self.metrics(g, active, p, exact)

        g64, active64, exact64 = (_flat(value) for value in (g, active, exact))
        estimated_component = g64 - (
            torch.dot(g64, active64) / torch.dot(active64, active64)
        ) * active64
        exact_component = g64 - (
            torch.dot(g64, exact64) / torch.dot(exact64, exact64)
        ) * exact64
        expected_cosine = torch.dot(estimated_component, exact_component) / (
            torch.linalg.vector_norm(estimated_component)
            * torch.linalg.vector_norm(exact_component)
        )
        expected_relative_l2 = torch.linalg.vector_norm(
            estimated_component - exact_component
        ) / torch.linalg.vector_norm(exact_component)
        expected_norm_ratio = torch.linalg.vector_norm(
            estimated_component
        ) / torch.linalg.vector_norm(exact_component)
        self.assertAlmostEqual(
            result["grad/batch_component_estimate_exact_cosine"],
            expected_cosine.item(),
            places=6,
        )
        self.assertAlmostEqual(
            result["grad/batch_component_estimate_exact_relative_l2"],
            expected_relative_l2.item(),
            places=6,
        )
        self.assertAlmostEqual(
            result["grad/batch_component_estimate_exact_norm_ratio"],
            expected_norm_ratio.item(),
            places=6,
        )
        self.assertGreater(
            abs(
                result["grad/batch_component_estimate_exact_cosine"]
                - result["grad/batch_component_exact_cosine"]
            ),
            0.1,
        )
        self.assertEqual(
            result["grad/batch_component_exact_cosine_semantics"],
            LEGACY_BATCH_COMPONENT_EXACT_COSINE_SEMANTICS,
        )

    def test_projected_component_fidelity_reuses_canonical_projection(self):
        g = _state(self.index, [2.0, 3.0, 4.0, 1.0])
        active = _state(self.index, [1.0, 0.0, 0.0, 0.0])
        exact = _state(self.index, [0.0, 1.0, 1.0, 0.0])
        p = _state(self.index, [1.0, 1.0, 0.0, 0.0])
        projection = project_batch_gradient(g, active)
        with mock.patch(
            "sample_fg.diagnostics.project_batch_gradient",
            wraps=project_batch_gradient,
        ) as canonical:
            compute_gradient_diagnostics(
                batch_gradient=g,
                active_global_estimate=active,
                projection=projection,
                perturbed_gradient=p,
                exact_full_gradient=exact,
                alpha=0.2,
            )
        self.assertEqual(canonical.call_count, 2)
        self.assertIs(canonical.call_args_list[-1].args[1], exact)

    def test_taylor_identity_and_signs(self):
        alpha = 0.0015
        g = _state(self.index, [2.0, -1.0, 3.0, 4.0])
        active = _state(self.index, [1.0, 2.0, -0.5, 1.0])
        p = _state(self.index, [-1.0, 0.25, 2.0, -3.0])
        projection = project_batch_gradient(g, active)
        result = self.metrics(g, active, p, active, alpha=alpha)
        reference = -alpha * float(p.dot(projection.batch_component).item())
        self.assertAlmostEqual(result["taylor/joint_alignment_term"], reference, places=6)
        self.assertAlmostEqual(result["taylor/exploitation_term"], -alpha * float(p.dot(g).item()), places=7)
        self.assertAlmostEqual(
            result["taylor/exploration_term"],
            alpha * projection.xi * projection.sigma * float(p.dot(active).item()),
            places=7,
        )

    def test_degenerate_cosines_are_null_with_flags(self):
        zero = _state(self.index, [0.0, 0.0, 0.0, 0.0])
        valid = _state(self.index, [1.0, 2.0, 3.0, 4.0])
        result = self.metrics(zero, valid, zero, zero)
        self.assertTrue(result["grad/batch_gradient_degenerate"])
        self.assertTrue(result["grad/exact_full_direction_degenerate"])
        self.assertTrue(result["grad/perturbed_gradient_degenerate"])
        self.assertIsNone(result["grad/global_estimate_exact_cosine"])
        self.assertIsNone(result["grad/global_estimate_exact_norm_ratio"])
        self.assertIsNone(result["grad/batch_component_estimator_cosine"])
        self.assertIsNone(result["grad/perturbed_gradient_exact_cosine"])
        self.assertEqual(result["taylor/exploitation_term"], 0.0)
        self.assertEqual(result["taylor/exploration_term"], 0.0)

    def test_optional_exact_reference_has_explicit_null_fields(self):
        g = _state(self.index, [1.0, 2.0, 3.0, 4.0])
        active = _state(self.index, [2.0, -1.0, 0.5, 1.0])
        p = _state(self.index, [0.5, 1.0, -2.0, 3.0])
        result = self.metrics(g, active, p)
        self.assertFalse(result["grad/exact_reference_available"])
        for key in (
            "grad/exact_full_norm",
            "grad/global_estimate_exact_cosine",
            "grad/batch_component_exact_cosine",
            "grad/batch_component_estimate_exact_cosine",
            "grad/batch_component_estimate_exact_relative_l2",
            "grad/batch_component_estimate_exact_norm_ratio",
            "grad/perturbed_gradient_exact_cosine",
            "raw/dot_batch_exact",
        ):
            self.assertIsNone(result[key])

    def test_nonfinite_incompatible_and_mismatched_projection_fail(self):
        valid = _state(self.index, [1.0, 2.0, 3.0, 4.0])
        bad = _state(self.index, [float("nan"), 0.0, 0.0, 0.0])
        with self.assertRaises(DiagnosticNumericalError):
            compute_gradient_diagnostics(
                batch_gradient=bad,
                active_global_estimate=valid,
                projection=project_batch_gradient(valid, valid),
                perturbed_gradient=valid,
                exact_full_gradient=valid,
                alpha=0.1,
            )
        other_index = ParamIndex.from_model(_Other())
        other = GradientState.from_tensors(other_index, [torch.ones(4)])
        with self.assertRaises(GradientStateMismatchError):
            compute_gradient_diagnostics(
                batch_gradient=valid,
                active_global_estimate=other,
                projection=project_batch_gradient(other, other),
                perturbed_gradient=valid,
                alpha=0.1,
            )
        projection = project_batch_gradient(valid, valid)
        projection.batch_component[0].add_(1.0)
        with self.assertRaises(DiagnosticError):
            compute_gradient_diagnostics(
                batch_gradient=valid,
                active_global_estimate=valid,
                projection=projection,
                perturbed_gradient=valid,
                alpha=0.1,
            )

    def test_purity_scalar_ownership_and_json_ready_values(self):
        g = _state(self.index, [1.0, 2.0, 3.0, 4.0])
        active = _state(self.index, [2.0, 0.5, -1.0, 3.0])
        exact = _state(self.index, [-1.0, 2.0, 0.5, 4.0])
        p = _state(self.index, [3.0, -2.0, 1.0, 0.25])
        before = [state.clone() for state in (g, active, exact, p)]
        result = self.metrics(g, active, p, exact)
        for observed, expected in zip((g, active, exact, p), before):
            for left, right in zip(observed, expected):
                self.assertTrue(torch.equal(left, right))
        copied = result.as_dict()
        copied["grad/batch_norm"] = -1.0
        self.assertGreater(result["grad/batch_norm"], 0.0)
        self.assertTrue(
            all(value is None or isinstance(value, (str, bool, int, float)) for value in result.values())
        )
        self.assertTrue(all(not isinstance(value, torch.Tensor) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
