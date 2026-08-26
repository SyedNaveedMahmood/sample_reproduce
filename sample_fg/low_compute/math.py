"""Strict scalar comparisons shared by LC01 and LC04."""

from __future__ import annotations

import math
from typing import Any

import torch

from sample_fg.gradient_state import GradientState


NORM_EPS = 1e-12


def _tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    a = left.detach().to(dtype=torch.float32).reshape(-1)
    b = right.detach().to(dtype=torch.float32).reshape(-1)
    if a.numel() != b.numel():
        raise ValueError("Comparison tensors must contain the same number of values")
    na = float(torch.linalg.vector_norm(a).item())
    nb = float(torch.linalg.vector_norm(b).item())
    degenerate_left = na <= NORM_EPS
    degenerate_right = nb <= NORM_EPS
    if degenerate_left or degenerate_right:
        return {
            "cosine": None,
            "angle_degrees": None,
            "relative_l2": None if nb <= NORM_EPS else float(torch.linalg.vector_norm(a - b).item()) / nb,
            "norm_ratio": None if nb <= NORM_EPS else na / nb,
            "log_norm_ratio": None,
            "left_norm": na,
            "right_norm": nb,
            "left_degenerate": degenerate_left,
            "right_degenerate": degenerate_right,
        }
    cosine = float(torch.dot(a, b).item()) / (na * nb)
    cosine = max(-1.0, min(1.0, cosine))
    ratio = na / nb
    return {
        "cosine": cosine,
        "angle_degrees": math.degrees(math.acos(cosine)),
        "relative_l2": float(torch.linalg.vector_norm(a - b).item()) / nb,
        "norm_ratio": ratio,
        "log_norm_ratio": math.log(ratio),
        "left_norm": na,
        "right_norm": nb,
        "left_degenerate": False,
        "right_degenerate": False,
    }


def tensor_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    return _tensor_metrics(left, right)


def gradient_metrics(left: GradientState, right: GradientState) -> dict[str, Any]:
    left.assert_compatible(right)
    return _tensor_metrics(flatten_state(left), flatten_state(right))


def flatten_state(state: GradientState) -> torch.Tensor:
    state.assert_valid()
    return torch.cat(tuple(component.reshape(-1) for component in state))
