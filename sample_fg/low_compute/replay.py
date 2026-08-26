"""CPU-only EMA replay and LC01 projection/displacement comparisons."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from sample_fg.gradient_state import GradientState
from sample_fg.projection import project_batch_gradient, safe_unit

from .math import gradient_metrics


class ReplayError(ValueError):
    pass


def _validate_lambda(ema_lambda: float) -> float:
    value = float(ema_lambda)
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ReplayError("EMA lambda must be finite and in [0, 1)")
    return value


def effective_sample_size(ema_lambda: float) -> float:
    value = _validate_lambda(ema_lambda)
    return (1.0 + value) / (1.0 - value)


def history_length(ema_lambda: float, coverage: float) -> int:
    value = _validate_lambda(ema_lambda)
    target = float(coverage)
    if not 0.0 < target < 1.0:
        raise ReplayError("Coverage must be in (0, 1)")
    if value == 0.0:
        return 1
    return int(math.ceil(math.log(1.0 - target) / math.log(value)))


def ema_replay(
    gradients: Sequence[GradientState],
    ema_lambda: float,
    *,
    order: Sequence[int] | None = None,
) -> GradientState:
    """Replay the paper recurrence without owning a model or optimizer."""

    if not gradients:
        raise ReplayError("At least one stored gradient is required")
    value = _validate_lambda(ema_lambda)
    indices = tuple(range(len(gradients))) if order is None else tuple(order)
    if sorted(indices) != list(range(len(gradients))):
        raise ReplayError("Replay order must be one permutation of all gradients")
    # Stored probe gradients deliberately live on CPU even though their
    # ParamIndex still references the live CUDA prompt.  Initializing through
    # GradientState.zeros() would therefore put the accumulator on CUDA and
    # make the first affine update cross-device.  Derive the zero from the
    # stored state so replay follows the bank's owned device.
    state = gradients[0].scale(0.0)
    for index in indices:
        gradients[0].assert_compatible(gradients[index])
        state.affine_(gradients[index], value, 1.0 - value)
    return state


def stationary_ema_replay(
    gradients: Sequence[GradientState],
    ema_lambda: float,
    *,
    epochs: int,
    seed: int,
) -> tuple[GradientState, tuple[tuple[int, ...], ...]]:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ReplayError("Synthetic replay epochs must be positive")
    if not gradients:
        raise ReplayError("At least one stored gradient is required")
    value = _validate_lambda(ema_lambda)
    local = random.Random(seed)
    state = gradients[0].scale(0.0)
    orders: list[tuple[int, ...]] = []
    for _ in range(epochs):
        order = list(range(len(gradients)))
        local.shuffle(order)
        orders.append(tuple(order))
        for index in order:
            state.affine_(gradients[index], value, 1.0 - value)
    return state, tuple(orders)


@dataclass(frozen=True)
class OrderTrial:
    trial: int
    order: tuple[int, ...]
    last_batch_id: int
    metrics: dict[str, object]


def permutation_trials(
    gradients: Sequence[GradientState],
    exact: GradientState,
    *,
    ema_lambda: float,
    trial_count: int,
    seed: int,
) -> tuple[OrderTrial, ...]:
    if isinstance(trial_count, bool) or not isinstance(trial_count, int) or trial_count < 1:
        raise ReplayError("trial_count must be positive")
    local = random.Random(seed)
    rows = []
    for trial in range(trial_count):
        order = list(range(len(gradients)))
        local.shuffle(order)
        estimate = ema_replay(gradients, ema_lambda, order=order)
        rows.append(
            OrderTrial(
                trial=trial,
                order=tuple(order),
                last_batch_id=order[-1],
                metrics=gradient_metrics(estimate, exact),
            )
        )
    return tuple(rows)


def projection_displacement_metrics(
    batch_gradient: GradientState,
    estimate: GradientState,
    exact: GradientState,
    *,
    rho: float,
    alpha: float,
) -> dict[str, object]:
    """Compare geometry using the one canonical projection implementation."""

    projected_estimate = project_batch_gradient(batch_gradient, estimate)
    projected_exact = project_batch_gradient(batch_gradient, exact)
    unit = safe_unit(batch_gradient)
    epsilon = unit.unit.scale(float(rho))
    delta_estimate = epsilon.subtract(projected_estimate.batch_component.scale(float(alpha)))
    delta_exact = epsilon.subtract(projected_exact.batch_component.scale(float(alpha)))
    return {
        "gB": gradient_metrics(
            projected_estimate.batch_component, projected_exact.batch_component
        ),
        "delta": gradient_metrics(delta_estimate, delta_exact),
        "projection_coefficient_difference": (
            projected_estimate.projection_coefficient
            - projected_exact.projection_coefficient
        ),
        "xi_difference": projected_estimate.xi - projected_exact.xi,
        "sigma_difference": projected_estimate.sigma - projected_exact.sigma,
        "estimator_orthogonality_residual": abs(
            float(projected_estimate.batch_component.dot(estimate).item())
        ),
        "exact_orthogonality_residual": abs(
            float(projected_exact.batch_component.dot(exact).item())
        ),
        "estimate_projection_degenerate": (
            projected_estimate.batch_gradient_degenerate
            or projected_estimate.full_direction_degenerate
        ),
        "exact_projection_degenerate": (
            projected_exact.batch_gradient_degenerate
            or projected_exact.full_direction_degenerate
        ),
    }


def evaluate_lc02_gate(
    checkpoint_rows: Sequence[dict[str, float | int | None]],
    *,
    paper_lambda: float = 0.15,
    coverage_lambda: float = 0.8461538461538461,
) -> dict[str, object]:
    """Evaluate the predeclared mechanism-only gate; accuracy is rejected."""

    forbidden = {"accuracy", "base", "new", "hm", "accuracy_pct"}
    if any(forbidden.intersection(row) for row in checkpoint_rows):
        raise ReplayError("LC02 gate cannot consume accuracy fields")
    by_checkpoint: dict[object, dict[float, dict[str, float | int | None]]] = {}
    for row in checkpoint_rows:
        checkpoint = row.get("checkpoint")
        value = row.get("lambda")
        if checkpoint is None or not isinstance(value, (float, int)):
            raise ReplayError("Gate rows require checkpoint and lambda")
        by_checkpoint.setdefault(checkpoint, {})[float(value)] = row
    evidence = []
    for checkpoint, rows in sorted(by_checkpoint.items(), key=lambda item: str(item[0])):
        if paper_lambda not in rows or coverage_lambda not in rows:
            continue
        paper = rows[paper_lambda]
        coverage = rows[coverage_lambda]
        required = (
            "median_exact_cosine", "median_relative_l2",
            "median_gB_exact_cosine", "degenerate_projection_rate",
        )
        if any(not isinstance(paper.get(key), (int, float)) or not isinstance(coverage.get(key), (int, float)) for key in required):
            continue
        cosine_gain = float(coverage["median_exact_cosine"]) - float(paper["median_exact_cosine"])
        baseline_l2 = float(paper["median_relative_l2"])
        l2_reduction = (
            (baseline_l2 - float(coverage["median_relative_l2"])) / baseline_l2
            if baseline_l2 > 0 else 0.0
        )
        geometry_gain = float(coverage["median_gB_exact_cosine"]) - float(paper["median_gB_exact_cosine"])
        degeneracy_pp = 100.0 * (
            float(coverage["degenerate_projection_rate"])
            - float(paper["degenerate_projection_rate"])
        )
        fidelity = cosine_gain >= 0.10 and l2_reduction >= 0.20
        geometry = geometry_gain >= 0.10 and degeneracy_pp <= 5.0
        evidence.append(
            {
                "checkpoint": checkpoint,
                "cosine_gain": cosine_gain,
                "relative_l2_reduction_fraction": l2_reduction,
                "gB_cosine_gain": geometry_gain,
                "degeneracy_rate_increase_percentage_points": degeneracy_pp,
                "checkpoint_pass": fidelity or geometry,
            }
        )
    passing = sum(bool(row["checkpoint_pass"]) for row in evidence)
    return {
        "schema_version": "sample_fg.low_compute_lc02_gate.v1",
        "accuracy_fields_consumed": False,
        "required_checkpoint_count": 4,
        "evaluated_checkpoint_count": len(evidence),
        "passing_checkpoint_count": passing,
        "gate_passed": len(evidence) >= 4 and passing >= 4,
        "evidence": evidence,
    }
