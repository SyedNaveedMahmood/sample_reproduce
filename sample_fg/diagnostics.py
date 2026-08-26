"""Pure scalar diagnostics for SAMPLe gradient geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Iterator, Mapping

import torch

from .gradient_state import GradientState
from .projection import (
    DEFAULT_NORM_EPS,
    ProjectionResult,
    project_batch_gradient,
)


DIAGNOSTIC_SCHEMA_VERSION = "sample_fg.gradient_diagnostics.v1"
LEGACY_BATCH_COMPONENT_EXACT_COSINE_SEMANTICS = (
    "cosine(g_B(active_global_estimate), exact_full_gradient); "
    "construction-orthogonality metric, not projected-component fidelity"
)


class DiagnosticError(ValueError):
    """Raised when diagnostic inputs do not describe one coherent geometry."""


class DiagnosticNumericalError(FloatingPointError):
    """Raised when an input or derived scientific diagnostic is nonfinite."""


@dataclass(frozen=True)
class DiagnosticMetrics(Mapping[str, object]):
    """Immutable, JSON-ready scalar diagnostic record.

    The record intentionally owns no :class:`GradientState` or model object.
    Undefined directional metrics are represented by ``None`` alongside the
    applicable degeneracy flags.
    """

    _values: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def as_dict(self) -> dict[str, object]:
        return dict(self._values)


def _finite_real(value: Real, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise DiagnosticError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return result


def _require_state(state: GradientState, name: str) -> None:
    if not isinstance(state, GradientState):
        raise TypeError(f"{name} must be a GradientState")
    state.assert_valid()
    if not state.is_finite():
        raise DiagnosticNumericalError(f"{name} contains NaN or Inf")


def _scalar(tensor: torch.Tensor, name: str) -> float:
    value = float(tensor.item())
    if not math.isfinite(value):
        raise DiagnosticNumericalError(f"Derived diagnostic is nonfinite: {name}")
    return value


def _norm(state: GradientState, name: str) -> float:
    return _scalar(state.norm(), name)


def _dot(left: GradientState, right: GradientState, name: str) -> float:
    return _scalar(left.dot(right), name)


def _cosine(
    left: GradientState,
    right: GradientState,
    *,
    left_norm: float,
    right_norm: float,
    norm_eps: float,
    name: str,
) -> float | None:
    if left_norm <= norm_eps or right_norm <= norm_eps:
        return None
    value = _dot(left, right, name) / (left_norm * right_norm)
    if not math.isfinite(value):
        raise DiagnosticNumericalError(f"Derived cosine is nonfinite: {name}")
    # Reporting-only roundoff clamp, matching projection.py. It never changes
    # projection, displacement, or another state.
    return max(-1.0, min(1.0, value))


def _states_equal(left: GradientState, right: GradientState) -> bool:
    left.assert_compatible(right)
    return all(torch.equal(a, b) for a, b in zip(left, right))


def _validate_projection(
    batch_gradient: GradientState,
    active_global_estimate: GradientState,
    projection: ProjectionResult,
    norm_eps: float,
) -> None:
    if not isinstance(projection, ProjectionResult):
        raise TypeError("projection must be a ProjectionResult")
    expected = project_batch_gradient(
        batch_gradient, active_global_estimate, norm_eps=norm_eps
    )
    scalar_names = (
        "batch_norm",
        "full_direction_norm",
        "dot_product",
        "xi",
        "sigma",
        "projection_coefficient",
    )
    if any(getattr(projection, name) != getattr(expected, name) for name in scalar_names):
        raise DiagnosticError("Projection scalars do not match the supplied states")
    if (
        projection.batch_gradient_degenerate != expected.batch_gradient_degenerate
        or projection.full_direction_degenerate != expected.full_direction_degenerate
        or not _states_equal(projection.projected_component, expected.projected_component)
        or not _states_equal(projection.batch_component, expected.batch_component)
    ):
        raise DiagnosticError("Projection components do not match the supplied states")


def compute_gradient_diagnostics(
    *,
    batch_gradient: GradientState,
    active_global_estimate: GradientState,
    projection: ProjectionResult,
    perturbed_gradient: GradientState,
    alpha: Real,
    exact_full_gradient: GradientState | None = None,
    norm_eps: Real = DEFAULT_NORM_EPS,
    log_ratio_eps: Real = DEFAULT_NORM_EPS,
) -> DiagnosticMetrics:
    """Compute the complete Task-18 scalar metric set without side effects."""

    alpha_value = _finite_real(alpha, "alpha", nonnegative=True)
    threshold = _finite_real(norm_eps, "norm_eps", nonnegative=True)
    ratio_epsilon = _finite_real(
        log_ratio_eps, "log_ratio_eps", nonnegative=True
    )
    if ratio_epsilon == 0.0:
        raise DiagnosticError("log_ratio_eps must be positive")

    for state, name in (
        (batch_gradient, "batch_gradient"),
        (active_global_estimate, "active_global_estimate"),
        (perturbed_gradient, "perturbed_gradient"),
    ):
        _require_state(state, name)
    batch_gradient.assert_compatible(active_global_estimate)
    batch_gradient.assert_compatible(perturbed_gradient)
    _validate_projection(
        batch_gradient, active_global_estimate, projection, threshold
    )

    batch_component = projection.batch_component
    batch_norm = projection.batch_norm
    global_norm = projection.full_direction_norm
    batch_component_norm = _norm(batch_component, "batch-component norm")
    perturbed_norm = _norm(perturbed_gradient, "perturbed-gradient norm")
    batch_component_degenerate = batch_component_norm <= threshold
    perturbed_degenerate = perturbed_norm <= threshold

    dot_batch_global = projection.dot_product
    dot_batch_component_global = _dot(
        batch_component,
        active_global_estimate,
        "batch-component/global dot",
    )
    dot_perturbed_batch = _dot(
        perturbed_gradient, batch_gradient, "perturbed/batch dot"
    )
    dot_perturbed_global = _dot(
        perturbed_gradient,
        active_global_estimate,
        "perturbed/global dot",
    )
    dot_perturbed_batch_component = _dot(
        perturbed_gradient,
        batch_component,
        "perturbed/batch-component dot",
    )

    batch_component_estimator_cosine = _cosine(
        batch_component,
        active_global_estimate,
        left_norm=batch_component_norm,
        right_norm=global_norm,
        norm_eps=threshold,
        name="batch-component/estimator cosine",
    )
    perturbed_gradient_estimator_cosine = _cosine(
        perturbed_gradient,
        active_global_estimate,
        left_norm=perturbed_norm,
        right_norm=global_norm,
        norm_eps=threshold,
        name="perturbed-gradient/estimator cosine",
    )
    perturbed_gradient_batch_component_cosine = _cosine(
        perturbed_gradient,
        batch_component,
        left_norm=perturbed_norm,
        right_norm=batch_component_norm,
        norm_eps=threshold,
        name="perturbed-gradient/batch-component cosine",
    )
    perturbed_gradient_batch_cosine = _cosine(
        perturbed_gradient,
        batch_gradient,
        left_norm=perturbed_norm,
        right_norm=batch_norm,
        norm_eps=threshold,
        name="perturbed-gradient/batch cosine",
    )

    exact_available = exact_full_gradient is not None
    exact_norm: float | None = None
    exact_degenerate: bool | None = None
    reference_component_norm: float | None = None
    reference_component_degenerate: bool | None = None
    global_exact_cosine: float | None = None
    global_exact_norm_ratio: float | None = None
    global_exact_log_norm_ratio: float | None = None
    global_exact_relative_l2: float | None = None
    batch_component_exact_cosine: float | None = None
    batch_component_estimate_exact_cosine: float | None = None
    batch_component_estimate_exact_relative_l2: float | None = None
    batch_component_estimate_exact_norm_ratio: float | None = None
    reference_batch_component_exact_cosine: float | None = None
    perturbed_gradient_exact_cosine: float | None = None
    dot_batch_exact: float | None = None
    dot_global_exact: float | None = None
    dot_batch_component_exact: float | None = None
    dot_batch_components: float | None = None
    dot_perturbed_exact: float | None = None

    if exact_full_gradient is not None:
        _require_state(exact_full_gradient, "exact_full_gradient")
        batch_gradient.assert_compatible(exact_full_gradient)
        exact_norm = _norm(exact_full_gradient, "exact-full norm")
        exact_degenerate = exact_norm <= threshold
        reference_projection = project_batch_gradient(
            batch_gradient, exact_full_gradient, norm_eps=threshold
        )
        reference_component = reference_projection.batch_component
        reference_component_norm = _norm(
            reference_component, "reference batch-component norm"
        )
        reference_component_degenerate = reference_component_norm <= threshold
        dot_batch_exact = _dot(
            batch_gradient, exact_full_gradient, "batch/exact dot"
        )
        dot_global_exact = _dot(
            active_global_estimate, exact_full_gradient, "global/exact dot"
        )
        dot_batch_component_exact = _dot(
            batch_component, exact_full_gradient, "batch-component/exact dot"
        )
        dot_perturbed_exact = _dot(
            perturbed_gradient, exact_full_gradient, "perturbed/exact dot"
        )
        global_exact_cosine = _cosine(
            active_global_estimate,
            exact_full_gradient,
            left_norm=global_norm,
            right_norm=exact_norm,
            norm_eps=threshold,
            name="global-estimate/exact cosine",
        )
        if not exact_degenerate:
            global_exact_norm_ratio = global_norm / exact_norm
        global_exact_log_norm_ratio = math.log(
            (global_norm + ratio_epsilon) / (exact_norm + ratio_epsilon)
        )
        difference_norm = _norm(
            active_global_estimate.subtract(exact_full_gradient),
            "global-estimate/exact difference norm",
        )
        global_exact_relative_l2 = difference_norm / max(exact_norm, threshold)
        batch_component_exact_cosine = _cosine(
            batch_component,
            exact_full_gradient,
            left_norm=batch_component_norm,
            right_norm=exact_norm,
            norm_eps=threshold,
            name="batch-component/exact cosine",
        )
        dot_batch_components = _dot(
            batch_component,
            reference_component,
            "estimated/reference batch-component dot",
        )
        batch_component_estimate_exact_cosine = _cosine(
            batch_component,
            reference_component,
            left_norm=batch_component_norm,
            right_norm=reference_component_norm,
            norm_eps=threshold,
            name="estimated/reference batch-component cosine",
        )
        if not reference_component_degenerate:
            batch_component_estimate_exact_norm_ratio = (
                batch_component_norm / reference_component_norm
            )
            component_difference_norm = _norm(
                batch_component.subtract(reference_component),
                "estimated/reference batch-component difference norm",
            )
            batch_component_estimate_exact_relative_l2 = (
                component_difference_norm / reference_component_norm
            )
        reference_batch_component_exact_cosine = _cosine(
            reference_component,
            exact_full_gradient,
            left_norm=reference_component_norm,
            right_norm=exact_norm,
            norm_eps=threshold,
            name="reference-batch-component/exact cosine",
        )
        perturbed_gradient_exact_cosine = _cosine(
            perturbed_gradient,
            exact_full_gradient,
            left_norm=perturbed_norm,
            right_norm=exact_norm,
            norm_eps=threshold,
            name="perturbed-gradient/exact cosine",
        )

    exploitation_term = -alpha_value * dot_perturbed_batch
    exploration_term = (
        alpha_value
        * projection.xi
        * projection.sigma
        * dot_perturbed_global
    )
    joint_alignment_term = exploitation_term + exploration_term

    values: dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "grad/exact_reference_available": exact_available,
        "grad/batch_norm": batch_norm,
        "grad/global_estimate_norm": global_norm,
        "grad/exact_full_norm": exact_norm,
        "grad/batch_component_norm": batch_component_norm,
        "grad/reference_batch_component_norm": reference_component_norm,
        "grad/perturbed_norm": perturbed_norm,
        "grad/xi": projection.xi,
        "grad/sigma": projection.sigma,
        "grad/projection_coefficient": projection.projection_coefficient,
        "grad/batch_gradient_degenerate": projection.batch_gradient_degenerate,
        "grad/global_direction_degenerate": projection.full_direction_degenerate,
        "grad/exact_full_direction_degenerate": exact_degenerate,
        "grad/batch_component_degenerate": batch_component_degenerate,
        "grad/reference_batch_component_degenerate": reference_component_degenerate,
        "grad/perturbed_gradient_degenerate": perturbed_degenerate,
        "grad/global_estimate_exact_cosine": global_exact_cosine,
        "grad/global_estimate_exact_norm_ratio": global_exact_norm_ratio,
        "grad/global_estimate_exact_log_norm_ratio": global_exact_log_norm_ratio,
        "grad/global_estimate_exact_relative_l2": global_exact_relative_l2,
        "grad/batch_component_estimator_cosine": batch_component_estimator_cosine,
        "grad/batch_component_exact_cosine": batch_component_exact_cosine,
        "grad/batch_component_exact_cosine_semantics": LEGACY_BATCH_COMPONENT_EXACT_COSINE_SEMANTICS,
        "grad/batch_component_estimate_exact_cosine": batch_component_estimate_exact_cosine,
        "grad/batch_component_estimate_exact_relative_l2": batch_component_estimate_exact_relative_l2,
        "grad/batch_component_estimate_exact_norm_ratio": batch_component_estimate_exact_norm_ratio,
        "grad/reference_batch_component_exact_cosine": reference_batch_component_exact_cosine,
        "grad/perturbed_gradient_estimator_cosine": perturbed_gradient_estimator_cosine,
        "grad/perturbed_gradient_exact_cosine": perturbed_gradient_exact_cosine,
        "grad/perturbed_gradient_batch_component_cosine": perturbed_gradient_batch_component_cosine,
        "grad/perturbed_gradient_batch_cosine": perturbed_gradient_batch_cosine,
        "taylor/exploitation_dot_unweighted": dot_perturbed_batch,
        "taylor/exploitation_term": exploitation_term,
        "taylor/exploration_dot_unweighted": dot_perturbed_global,
        "taylor/exploration_term": exploration_term,
        "taylor/joint_alignment_term": joint_alignment_term,
        "raw/dot_batch_global": dot_batch_global,
        "raw/dot_batch_exact": dot_batch_exact,
        "raw/dot_global_exact": dot_global_exact,
        "raw/dot_batch_component_global": dot_batch_component_global,
        "raw/dot_batch_component_exact": dot_batch_component_exact,
        "raw/dot_batch_components": dot_batch_components,
        "raw/dot_perturbed_batch": dot_perturbed_batch,
        "raw/dot_perturbed_global": dot_perturbed_global,
        "raw/dot_perturbed_exact": dot_perturbed_exact,
        "raw/dot_perturbed_batch_component": dot_perturbed_batch_component,
    }
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise DiagnosticNumericalError(f"Nonfinite output diagnostic: {key}")
    return DiagnosticMetrics(values)
