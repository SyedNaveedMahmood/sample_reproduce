"""Pure LC05 semantic-drift and open-world evaluation primitives."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import torch


class SemanticProbeError(RuntimeError):
    pass


def _normalized(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[0] < 2:
        raise SemanticProbeError("Text features must be a [classes, features] tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise SemanticProbeError("Text features contain nonfinite values")
    norm = value.norm(dim=1, keepdim=True)
    if bool((norm <= 0).any().item()):
        raise SemanticProbeError("Text features contain a zero row")
    return value / norm


def _group_rows(classes: int, base_class_count: int) -> dict[str, torch.Tensor]:
    if not 0 < base_class_count < classes:
        raise SemanticProbeError("Base/New class boundary is invalid")
    indexes = torch.arange(classes)
    return {
        "base": indexes < base_class_count,
        "new": indexes >= base_class_count,
        "all": torch.ones(classes, dtype=torch.bool),
    }


def compute_semantic_drift(
    reference: torch.Tensor,
    learned: torch.Tensor,
    *,
    base_class_count: int,
) -> dict[str, Any]:
    """Compute predeclared cosine and normalized-Frobenius text drift."""

    reference = _normalized(reference).to(dtype=torch.float64)
    learned = _normalized(learned).to(dtype=torch.float64)
    if reference.shape != learned.shape:
        raise SemanticProbeError("Reference and learned text shapes differ")
    cosine_drift = 1.0 - torch.sum(reference * learned, dim=1).clamp(-1.0, 1.0)
    groups = _group_rows(len(reference), base_class_count)
    result: dict[str, Any] = {"per_class_cosine_drift": cosine_drift.tolist()}
    for name, mask in groups.items():
        ref = reference[mask]
        value = learned[mask]
        result[name] = {
            "class_count": int(mask.sum().item()),
            "mean_cosine_drift": float(cosine_drift[mask].mean().item()),
            "normalized_frobenius_drift": float(
                torch.linalg.vector_norm(value - ref).item()
                / torch.linalg.vector_norm(ref).item()
            ),
        }
    return result


def _relative_topology(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(right).item()
    numerator = torch.linalg.vector_norm(left - right).item()
    return float(numerator / denominator) if denominator > 0 else 0.0


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def compute_topology_distortion(
    reference: torch.Tensor,
    learned: torch.Tensor,
    *,
    base_class_count: int,
) -> dict[str, float]:
    """Compute off-diagonal all/block relative topology distortion."""

    reference = _normalized(reference).to(dtype=torch.float64)
    learned = _normalized(learned).to(dtype=torch.float64)
    if reference.shape != learned.shape:
        raise SemanticProbeError("Reference and learned text shapes differ")
    ref = reference @ reference.t()
    value = learned @ learned.t()
    base = slice(0, base_class_count)
    new = slice(base_class_count, len(reference))
    return {
        "all_off_diagonal": _relative_topology(
            _off_diagonal(value), _off_diagonal(ref)
        ),
        "base_base_off_diagonal": _relative_topology(
            _off_diagonal(value[base, base]), _off_diagonal(ref[base, base])
        ),
        "new_new_off_diagonal": _relative_topology(
            _off_diagonal(value[new, new]), _off_diagonal(ref[new, new])
        ),
        "base_new": _relative_topology(value[base, new], ref[base, new]),
    }


def compute_neighbor_preservation(
    reference: torch.Tensor,
    learned: torch.Tensor,
    *,
    base_class_count: int,
    k: int | None = None,
) -> dict[str, Any]:
    """Compare class neighborhoods in the common all-class vocabulary."""

    reference = _normalized(reference).to(dtype=torch.float64)
    learned = _normalized(learned).to(dtype=torch.float64)
    if reference.shape != learned.shape:
        raise SemanticProbeError("Reference and learned text shapes differ")
    classes = len(reference)
    expected_k = min(3, classes - 1)
    if k is None:
        k = expected_k
    if k != expected_k:
        raise SemanticProbeError(f"LC05 requires k=min(3,C-1)={expected_k}")
    diagonal = torch.arange(classes)
    ref_similarity = reference @ reference.t()
    learned_similarity = learned @ learned.t()
    ref_similarity[diagonal, diagonal] = -torch.inf
    learned_similarity[diagonal, diagonal] = -torch.inf
    ref_neighbors = torch.topk(ref_similarity, k=k, dim=1).indices
    learned_neighbors = torch.topk(learned_similarity, k=k, dim=1).indices
    jaccard = []
    top1 = []
    for row in range(classes):
        left = set(int(value) for value in ref_neighbors[row].tolist())
        right = set(int(value) for value in learned_neighbors[row].tolist())
        jaccard.append(len(left & right) / len(left | right))
        top1.append(int(ref_neighbors[row, 0]) == int(learned_neighbors[row, 0]))
    groups = _group_rows(classes, base_class_count)
    result: dict[str, Any] = {"k": k}
    for name, mask in groups.items():
        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        result[name] = {
            "mean_jaccard": statistics.fmean(jaccard[index] for index in indices),
            "top1_preservation_fraction": statistics.fmean(
                float(top1[index]) for index in indices
            ),
        }
    return result


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item() * 100.0)


def evaluate_open_world_logits(
    *,
    image_features: torch.Tensor,
    labels: torch.Tensor,
    text_features: torch.Tensor,
    base_class_count: int,
    logit_scale: float | torch.Tensor,
) -> dict[str, float]:
    """Evaluate Base/New samples in one all-class label space."""

    images = _normalized(image_features).to(dtype=torch.float64)
    text = _normalized(text_features).to(dtype=torch.float64)
    labels = labels.detach().cpu().to(dtype=torch.long)
    if len(images) != len(labels):
        raise SemanticProbeError("Evaluation images and labels differ")
    if bool((labels < 0).any().item()) or bool((labels >= len(text)).any().item()):
        raise SemanticProbeError("Evaluation label is outside the all-class vocabulary")
    scale = float(torch.as_tensor(logit_scale).detach().cpu().item())
    logits = scale * images @ text.t()
    predictions = logits.argmax(dim=1)
    base_rows = labels < base_class_count
    new_rows = ~base_rows
    base_accuracy = _accuracy(logits[base_rows], labels[base_rows])
    new_accuracy = _accuracy(logits[new_rows], labels[new_rows])
    hm = 0.0 if base_accuracy + new_accuracy == 0 else (
        2.0 * base_accuracy * new_accuracy / (base_accuracy + new_accuracy)
    )
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, labels[:, None], -torch.inf)
    margin = true - other.max(dim=1).values
    base_cross = (predictions[base_rows] >= base_class_count).float().mean() * 100.0
    new_cross = (predictions[new_rows] < base_class_count).float().mean() * 100.0
    return {
        "open_world_base_accuracy_pct": base_accuracy,
        "open_world_new_accuracy_pct": new_accuracy,
        "open_world_hm_pct": hm,
        "open_world_overall_accuracy_pct": _accuracy(logits, labels),
        "base_to_new_group_confusion_pct": float(base_cross.item()),
        "new_to_base_group_confusion_pct": float(new_cross.item()),
        "mean_ground_truth_margin_base": float(margin[base_rows].mean().item()),
        "mean_ground_truth_margin_new": float(margin[new_rows].mean().item()),
    }


def evaluate_standard_logits(
    *,
    base_image_features: torch.Tensor,
    base_labels: torch.Tensor,
    base_text_features: torch.Tensor,
    new_image_features: torch.Tensor,
    new_labels: torch.Tensor,
    new_text_features: torch.Tensor,
    logit_scale: float | torch.Tensor,
) -> dict[str, float]:
    scale = float(torch.as_tensor(logit_scale).detach().cpu().item())
    base_logits = scale * _normalized(base_image_features).to(dtype=torch.float64) @ _normalized(base_text_features).to(dtype=torch.float64).t()
    new_logits = scale * _normalized(new_image_features).to(dtype=torch.float64) @ _normalized(new_text_features).to(dtype=torch.float64).t()
    base = _accuracy(base_logits, base_labels.to(dtype=torch.long).cpu())
    new = _accuracy(new_logits, new_labels.to(dtype=torch.long).cpu())
    hm = 0.0 if base + new == 0 else 2.0 * base * new / (base + new)
    return {"base_accuracy_pct": base, "new_accuracy_pct": new, "hm_pct": hm}


def descriptive(values: Iterable[float]) -> dict[str, float | int | None]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(data),
        "mean": statistics.fmean(data) if data else None,
        "sample_sd": statistics.stdev(data) if len(data) > 1 else None,
    }


def pearson_spearman(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    """Return descriptive Pearson/Spearman coefficients without p-values."""

    clean = [(float(x), float(y)) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 2:
        return {"n": len(clean), "pearson": None, "spearman": None}
    x = torch.tensor([item[0] for item in clean], dtype=torch.float64)
    y = torch.tensor([item[1] for item in clean], dtype=torch.float64)

    def correlation(a: torch.Tensor, b: torch.Tensor) -> float | None:
        a = a - a.mean(); b = b - b.mean()
        denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
        return None if denominator.item() == 0 else float(torch.dot(a, b).item() / denominator.item())

    def ranks(value: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(value, stable=True)
        result = torch.empty_like(value)
        position = 0
        while position < len(value):
            end = position + 1
            while end < len(value) and value[order[end]] == value[order[position]]:
                end += 1
            result[order[position:end]] = (position + end - 1) / 2.0
            position = end
        return result

    return {"n": len(clean), "pearson": correlation(x, y), "spearman": correlation(ranks(x), ranks(y))}
