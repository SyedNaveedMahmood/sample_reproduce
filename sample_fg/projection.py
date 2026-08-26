"""Pure FP32 projection and safe-direction geometry for SAMPLe equations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from .gradient_state import GradientState


DEFAULT_NORM_EPS = 1e-12


class ProjectionError(ValueError):
    """Raised when projection configuration or state is invalid."""


class ProjectionNumericalError(FloatingPointError):
    """Raised when projection receives or produces nonfinite numerical state."""


@dataclass(frozen=True)
class ProjectionResult:
    """Eq. 8 scalars and the two owned vector components it defines."""

    batch_norm: float
    full_direction_norm: float
    dot_product: float
    xi: float
    sigma: float
    projection_coefficient: float
    projected_component: GradientState
    batch_component: GradientState
    batch_gradient_degenerate: bool
    full_direction_degenerate: bool


@dataclass(frozen=True)
class UnitVectorResult:
    """A safe unit direction and the norm/branch used to construct it."""

    unit: GradientState
    norm: float
    degenerate: bool


def _validated_norm_eps(norm_eps: Real) -> float:
    if isinstance(norm_eps, bool) or not isinstance(norm_eps, Real):
        raise TypeError("norm_eps must be a nonnegative finite real number")
    threshold = float(norm_eps)
    if not math.isfinite(threshold) or threshold < 0:
        raise ProjectionError("norm_eps must be a nonnegative finite value")
    return threshold


def _require_finite_state(state: GradientState, role: str) -> None:
    state.assert_valid()
    if not state.is_finite():
        raise ProjectionNumericalError(f"{role} contains NaN or Inf")


def _finite_scalar(value, role: str) -> float:
    scalar = float(value.item())
    if not math.isfinite(scalar):
        raise ProjectionNumericalError(f"Nonfinite derived scalar: {role}")
    return scalar


def _require_finite_output(state: GradientState, role: str) -> None:
    if not state.is_finite():
        raise ProjectionNumericalError(f"Projection produced nonfinite {role}")


def project_batch_gradient(
    batch_gradient: GradientState,
    full_direction: GradientState,
    norm_eps: Real = DEFAULT_NORM_EPS,
) -> ProjectionResult:
    """Project ``g`` onto ``F`` and return the Eq. 8 residual ``g_B``.

    Projection geometry uses ``c=(g dot F)/(F dot F)``. ``xi`` and
    ``sigma`` are derived independently for paper variables/diagnostics and
    never drive the projected tensors.
    """

    threshold = _validated_norm_eps(norm_eps)
    batch_gradient.assert_compatible(full_direction)
    _require_finite_state(batch_gradient, "batch gradient")
    _require_finite_state(full_direction, "full direction")

    batch_norm = _finite_scalar(batch_gradient.norm(), "batch norm")
    full_norm = _finite_scalar(full_direction.norm(), "full-direction norm")
    dot_product = _finite_scalar(
        batch_gradient.dot(full_direction), "batch/full dot product"
    )
    batch_degenerate = batch_norm <= threshold
    full_degenerate = full_norm <= threshold

    if batch_degenerate or full_degenerate:
        projected = GradientState.zeros(full_direction.param_index)
        residual = batch_gradient.clone()
        return ProjectionResult(
            batch_norm=batch_norm,
            full_direction_norm=full_norm,
            dot_product=dot_product,
            xi=0.0,
            sigma=0.0,
            projection_coefficient=0.0,
            projected_component=projected,
            batch_component=residual,
            batch_gradient_degenerate=batch_degenerate,
            full_direction_degenerate=full_degenerate,
        )

    full_squared_norm = _finite_scalar(
        full_direction.squared_norm(), "full-direction squared norm"
    )
    coefficient = dot_product / full_squared_norm
    sigma = batch_norm / full_norm
    xi_unclamped = dot_product / (batch_norm * full_norm)
    if not all(math.isfinite(value) for value in (coefficient, sigma, xi_unclamped)):
        raise ProjectionNumericalError("Projection produced a nonfinite scalar")

    # This clamp affects only the reported cosine, never projection geometry.
    xi = max(-1.0, min(1.0, xi_unclamped))
    projected = full_direction.scale(coefficient)
    residual = batch_gradient.subtract(projected)
    _require_finite_output(projected, "projected component")
    _require_finite_output(residual, "batch component")
    return ProjectionResult(
        batch_norm=batch_norm,
        full_direction_norm=full_norm,
        dot_product=dot_product,
        xi=xi,
        sigma=sigma,
        projection_coefficient=coefficient,
        projected_component=projected,
        batch_component=residual,
        batch_gradient_degenerate=False,
        full_direction_degenerate=False,
    )


def safe_unit(
    state: GradientState,
    norm_eps: Real = DEFAULT_NORM_EPS,
) -> UnitVectorResult:
    """Return ``state / ||state||`` or explicit zero at/below threshold."""

    threshold = _validated_norm_eps(norm_eps)
    _require_finite_state(state, "unit-vector input")
    norm = _finite_scalar(state.norm(), "unit-vector norm")
    if norm <= threshold:
        return UnitVectorResult(
            unit=GradientState.zeros(state.param_index),
            norm=norm,
            degenerate=True,
        )
    unit = state.scale(1.0 / norm)
    _require_finite_output(unit, "unit vector")
    return UnitVectorResult(unit=unit, norm=norm, degenerate=False)
