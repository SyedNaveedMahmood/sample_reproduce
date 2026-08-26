"""Pure LC06 fixed-materialization prompt-sharpness primitives."""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Callable, Mapping, Sequence

import torch

from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.projection import safe_unit


RADII = (0.0125, 0.025, 0.05)
NUM_RANDOM_DIRECTIONS = 32


class SharpnessProbeError(RuntimeError):
    pass


def parameter_sha256(param_index: ParamIndex) -> str:
    digest = hashlib.sha256()
    for entry in param_index:
        value = entry.parameter.detach().cpu().contiguous()
        digest.update(entry.name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def sample_prompt_directions(
    param_index: ParamIndex,
    *,
    checkpoint_sha256: str,
    count: int = NUM_RANDOM_DIRECTIONS,
) -> tuple[GradientState, ...]:
    """Generate checkpoint-keyed Rademacher directions with global unit norm."""

    if count != NUM_RANDOM_DIRECTIONS:
        raise SharpnessProbeError("LC06 requires exactly 32 random directions")
    try:
        seed = int(checkpoint_sha256[:16], 16) % (2**63 - 1)
    except (TypeError, ValueError) as error:
        raise SharpnessProbeError("Checkpoint SHA-256 is malformed") from error
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = []
    for _ in range(count):
        components = []
        for entry in param_index:
            sampled = torch.randint(
                0, 2, entry.shape, generator=generator, dtype=torch.int64
            ).mul_(2).sub_(1).to(dtype=torch.float32, device=entry.parameter.device)
            components.append(sampled)
        state = GradientState.from_tensors(param_index, components)
        unit = safe_unit(state)
        if unit.degenerate:
            raise SharpnessProbeError("Random prompt direction is degenerate")
        directions.append(unit.unit)
    return tuple(directions)


def _quantile(values: Sequence[float], q: float) -> float:
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q).item())


def summarize_sharpness(
    rows: Sequence[Mapping[str, float]],
    *,
    baseline_loss: float,
    radius: float,
    eps: float = 1e-12,
) -> dict[str, Any]:
    if len(rows) != NUM_RANDOM_DIRECTIONS:
        raise SharpnessProbeError("Sharpness quantiles require all 32 directions")
    values = [float(row["sharpness"]) for row in rows]
    signed = [
        float(row[name])
        for row in rows
        for name in ("delta_loss_plus", "delta_loss_minus")
    ]
    asymmetry = [
        abs(float(row["delta_loss_plus"]) - float(row["delta_loss_minus"]))
        for row in rows
    ]
    summary = {
        "direction_count": len(rows),
        "radius": float(radius),
        "reference_loss": float(baseline_loss),
        "sharpness_mean": statistics.fmean(values),
        "sharpness_median": statistics.median(values),
        "sharpness_p90": _quantile(values, 0.90),
        "sharpness_p95": _quantile(values, 0.95),
        "sharpness_max": max(values),
        "fraction_negative_loss_change": statistics.fmean(
            float(value < 0.0) for value in signed
        ),
        "plus_minus_asymmetry": statistics.fmean(asymmetry),
    }
    denominator = abs(float(baseline_loss)) + eps
    summary["normalized"] = {
        "by_abs_reference_loss": {
            key: float(summary[key]) / denominator
            for key in (
                "sharpness_mean", "sharpness_median", "sharpness_p90",
                "sharpness_p95", "sharpness_max",
            )
        },
        "by_radius": {
            key: float(summary[key]) / float(radius)
            for key in (
                "sharpness_mean", "sharpness_median", "sharpness_p90",
                "sharpness_p95", "sharpness_max",
            )
        },
        "by_radius_squared": {
            key: float(summary[key]) / (float(radius) ** 2)
            for key in (
                "sharpness_mean", "sharpness_median", "sharpness_p90",
                "sharpness_p95", "sharpness_max",
            )
        },
    }
    return summary


def probe_symmetric_loss_sharpness(
    *,
    param_index: ParamIndex,
    loss_fn: Callable[[], torch.Tensor],
    directions: Sequence[GradientState],
    radii: Sequence[float] = RADII,
) -> tuple[dict[str, Any], ...]:
    """Evaluate fixed-feature loss under symmetric prompt perturbations."""

    if tuple(float(value) for value in radii) != RADII:
        raise SharpnessProbeError("LC06 radii are fixed at rho fractions")
    if len(directions) != NUM_RANDOM_DIRECTIONS:
        raise SharpnessProbeError("LC06 requires exactly 32 random directions")
    before = parameter_sha256(param_index)
    with torch.no_grad():
        baseline = float(loss_fn().detach().cpu().item())
    if not math.isfinite(baseline):
        raise SharpnessProbeError("Reference loss is nonfinite")
    perturbation = PromptPerturbation(param_index)
    rows = []
    for direction_index, direction in enumerate(directions):
        unit = safe_unit(direction)
        if unit.degenerate:
            raise SharpnessProbeError("Prompt direction is degenerate")
        for radius in radii:
            values = {}
            measured = {}
            logical = {}
            for sign, label in ((1.0, "plus"), (-1.0, "minus")):
                displacement = unit.unit.scale(sign * float(radius))
                logical[label] = float(displacement.norm().detach().cpu().item())
                with perturbation.displaced(displacement) as snapshot:
                    squared = sum(
                        torch.sum(
                            (
                                entry.parameter.detach().to(dtype=torch.float32)
                                - original.to(dtype=torch.float32)
                            ) ** 2
                        )
                        for entry, original in zip(param_index, snapshot)
                    )
                    measured[label] = float(torch.sqrt(squared).detach().cpu().item())
                    with torch.no_grad():
                        values[label] = float(loss_fn().detach().cpu().item())
                if parameter_sha256(param_index) != before:
                    raise SharpnessProbeError("Prompt did not restore bitwise")
            delta_plus = values["plus"] - baseline
            delta_minus = values["minus"] - baseline
            rows.append(
                {
                    "direction_index": direction_index,
                    "radius": float(radius),
                    "loss_plus": values["plus"],
                    "loss_minus": values["minus"],
                    "delta_loss_plus": delta_plus,
                    "delta_loss_minus": delta_minus,
                    "sharpness": max(delta_plus, delta_minus),
                    "logical_displacement_norm_plus": logical["plus"],
                    "logical_displacement_norm_minus": logical["minus"],
                    "live_displacement_norm_plus": measured["plus"],
                    "live_displacement_norm_minus": measured["minus"],
                }
            )
    return tuple(rows)


def probe_structured_direction(
    *,
    name: str,
    param_index: ParamIndex,
    loss_fn: Callable[[], torch.Tensor],
    direction: GradientState,
    radii: Sequence[float] = RADII,
) -> tuple[dict[str, Any], ...]:
    unit = safe_unit(direction)
    if unit.degenerate:
        return tuple(
            {"direction": name, "radius": float(radius), "degenerate": True}
            for radius in radii
        )
    before = parameter_sha256(param_index)
    with torch.no_grad():
        baseline = float(loss_fn().detach().cpu().item())
    perturbation = PromptPerturbation(param_index)
    rows = []
    for radius in radii:
        losses = {}
        live_norms = {}
        logical_norms = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            displacement = unit.unit.scale(sign * float(radius))
            logical_norms[label] = float(displacement.norm().detach().cpu().item())
            with perturbation.displaced(displacement) as snapshot:
                squared = sum(
                    torch.sum(
                        (
                            entry.parameter.detach().to(dtype=torch.float32)
                            - original.to(dtype=torch.float32)
                        ) ** 2
                    )
                    for entry, original in zip(param_index, snapshot)
                )
                live_norms[label] = float(torch.sqrt(squared).detach().cpu().item())
                with torch.no_grad():
                    losses[label] = float(loss_fn().detach().cpu().item())
            if parameter_sha256(param_index) != before:
                raise SharpnessProbeError("Prompt did not restore after structured probe")
        rows.append(
            {
                "direction": name,
                "radius": float(radius),
                "degenerate": False,
                "delta_loss_plus": losses["plus"] - baseline,
                "delta_loss_minus": losses["minus"] - baseline,
                "sharpness": max(losses["plus"] - baseline, losses["minus"] - baseline),
                "logical_displacement_norm_plus": logical_norms["plus"],
                "logical_displacement_norm_minus": logical_norms["minus"],
                "live_displacement_norm_plus": live_norms["plus"],
                "live_displacement_norm_minus": live_norms["minus"],
            }
        )
    return tuple(rows)


def exact_materialized_gradient(
    *,
    param_index: ParamIndex,
    loss_fn: Callable[[], torch.Tensor],
) -> tuple[float, GradientState]:
    """Run the optional single prompt backward without an optimizer transition."""

    for entry in param_index:
        entry.parameter.grad = None
    loss = loss_fn()
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
        raise SharpnessProbeError("Materialized reference loss is invalid")
    loss.backward()
    gradient = GradientState.from_parameter_grads(param_index)
    for entry in param_index:
        entry.parameter.grad = None
    return float(loss.detach().cpu().item()), gradient
