"""LC04 central-difference probes over frozen CLIP-level functions."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Mapping, Sequence

import torch

from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.projection import safe_unit

from .math import gradient_metrics, tensor_metrics


class FunctionalProbeError(RuntimeError):
    pass


def _parameter_hash(param_index: ParamIndex) -> str:
    digest = hashlib.sha256()
    for entry in param_index:
        tensor = entry.parameter.detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _normalized_text(text_feature_fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    value = text_feature_fn()
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise FunctionalProbeError("text_feature_fn must return [classes, features]")
    return value / value.norm(dim=-1, keepdim=True)


def _objects(
    text: torch.Tensor,
    *,
    eval_image_features: torch.Tensor,
    eval_labels: torch.Tensor,
    logit_scale: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    images = eval_image_features / eval_image_features.norm(dim=-1, keepdim=True)
    logits = torch.as_tensor(logit_scale, device=text.device, dtype=text.dtype) * images.to(text) @ text.t()
    labels = eval_labels.to(device=logits.device, dtype=torch.long)
    if labels.ndim != 1 or len(labels) != logits.shape[0]:
        raise FunctionalProbeError("Evaluation labels differ from image features")
    if bool((labels < 0).any().item()) or bool((labels >= logits.shape[1]).any().item()):
        raise FunctionalProbeError("Evaluation labels are outside the class vocabulary")
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], float("-inf"))
    margins = true - other.max(dim=1).values
    return {
        "text": text,
        "topology": text @ text.t(),
        "logits": logits,
        "margins": margins,
    }


def _directional_response(
    *,
    direction: GradientState,
    radius: float,
    perturbation: PromptPerturbation,
    text_feature_fn: Callable[[], torch.Tensor],
    eval_image_features: torch.Tensor,
    eval_labels: torch.Tensor,
    logit_scale: torch.Tensor | float,
) -> dict[str, torch.Tensor]:
    unit = safe_unit(direction)
    if unit.degenerate:
        raise FunctionalProbeError("Cannot probe a degenerate direction")
    displacement = unit.unit.scale(float(radius))
    with perturbation.displaced(displacement):
        plus = _objects(
            _normalized_text(text_feature_fn),
            eval_image_features=eval_image_features,
            eval_labels=eval_labels,
            logit_scale=logit_scale,
        )
    with perturbation.displaced(displacement.scale(-1.0)):
        minus = _objects(
            _normalized_text(text_feature_fn),
            eval_image_features=eval_image_features,
            eval_labels=eval_labels,
            logit_scale=logit_scale,
        )
    return {key: (plus[key] - minus[key]) / (2.0 * radius) for key in plus}


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _comparison_rows(
    ema: Mapping[str, torch.Tensor],
    exact: Mapping[str, torch.Tensor],
    *,
    base_class_count: int,
    eval_labels: torch.Tensor,
) -> dict[str, Any]:
    classes = ema["text"].shape[0]
    if not 0 < base_class_count < classes:
        raise FunctionalProbeError("Base/New class boundary is invalid")
    base = slice(0, base_class_count)
    new = slice(base_class_count, classes)
    labels = eval_labels.to(dtype=torch.long, device=ema["logits"].device)
    base_rows = labels < base_class_count
    new_rows = ~base_rows
    result: dict[str, Any] = {
        "text_all": tensor_metrics(ema["text"], exact["text"]),
        "text_base": tensor_metrics(ema["text"][base], exact["text"][base]),
        "text_new": tensor_metrics(ema["text"][new], exact["text"][new]),
        "topology_off_diagonal": tensor_metrics(
            _off_diagonal(ema["topology"]), _off_diagonal(exact["topology"])
        ),
        "topology_base_base": tensor_metrics(ema["topology"][base, base], exact["topology"][base, base]),
        "topology_new_new": tensor_metrics(ema["topology"][new, new], exact["topology"][new, new]),
        "topology_base_new": tensor_metrics(ema["topology"][base, new], exact["topology"][base, new]),
        "logits_all": tensor_metrics(ema["logits"], exact["logits"]),
        "logits_base_rows": tensor_metrics(ema["logits"][base_rows], exact["logits"][base_rows]),
        "logits_new_rows": tensor_metrics(ema["logits"][new_rows], exact["logits"][new_rows]),
        "logits_base_columns": tensor_metrics(ema["logits"][:, base], exact["logits"][:, base]),
        "logits_new_columns": tensor_metrics(ema["logits"][:, new], exact["logits"][:, new]),
    }
    for name, mask in (("base", base_rows), ("new", new_rows)):
        left = ema["margins"][mask]
        right = exact["margins"][mask]
        metrics = tensor_metrics(left, right)
        metrics["sign_agreement"] = float((torch.sign(left) == torch.sign(right)).float().mean().item())
        metrics["mean_absolute_difference"] = float(torch.mean(torch.abs(left - right)).item())
        result[f"margins_{name}"] = metrics
    return result


def compare_functional_directions(
    *,
    param_index: ParamIndex,
    ema_direction: GradientState,
    exact_direction: GradientState,
    radii: Sequence[float],
    text_feature_fn: Callable[[], torch.Tensor],
    eval_image_features: torch.Tensor,
    eval_labels: torch.Tensor,
    base_class_count: int,
    logit_scale: torch.Tensor | float,
) -> tuple[dict[str, Any], ...]:
    """Compare LC04 objects and prove bitwise prompt restoration."""

    if tuple(float(value) for value in radii) != (0.0025, 0.005):
        raise FunctionalProbeError("LC04 requires both predeclared radii")
    before = _parameter_hash(param_index)
    perturbation = PromptPerturbation(param_index)
    parameter = gradient_metrics(ema_direction, exact_direction)
    rows = []
    for radius in radii:
        ema = _directional_response(
            direction=ema_direction,
            radius=float(radius),
            perturbation=perturbation,
            text_feature_fn=text_feature_fn,
            eval_image_features=eval_image_features,
            eval_labels=eval_labels,
            logit_scale=logit_scale,
        )
        exact = _directional_response(
            direction=exact_direction,
            radius=float(radius),
            perturbation=perturbation,
            text_feature_fn=text_feature_fn,
            eval_image_features=eval_image_features,
            eval_labels=eval_labels,
            logit_scale=logit_scale,
        )
        rows.append(
            {
                "radius": float(radius),
                "parameter_space": parameter,
                "function_space": _comparison_rows(
                    ema, exact,
                    base_class_count=base_class_count,
                    eval_labels=eval_labels,
                ),
            }
        )
        if _parameter_hash(param_index) != before:
            raise FunctionalProbeError("Prompt was not restored after LC04 probe")
    return tuple(rows)
