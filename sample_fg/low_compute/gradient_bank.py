"""Fixed-source prompt-gradient banks built with canonical GradientState."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch

from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex


class GradientBankError(RuntimeError):
    pass


def gradient_sha256(state: GradientState) -> str:
    digest = hashlib.sha256()
    for component in state:
        tensor = component.detach().cpu().contiguous()
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class GradientBatch:
    batch_index: int
    sample_ids: tuple[str, ...]
    sample_count: int
    mean_loss: float
    gradient: GradientState
    gradient_sha256: str


@dataclass(frozen=True)
class GradientBank:
    batches: tuple[GradientBatch, ...]
    exact: GradientState
    total_samples: int
    materialization_replicate: int


def weighted_exact(batches: Sequence[GradientBatch]) -> GradientState:
    if not batches:
        raise GradientBankError("At least one gradient batch is required")
    total = sum(batch.sample_count for batch in batches)
    if total < 1:
        raise GradientBankError("Gradient bank is empty")
    result = GradientState.zeros(batches[0].gradient.param_index)
    for batch in batches:
        result.add_(batch.gradient, alpha=batch.sample_count / total)
    return result


def build_gradient_bank(
    *,
    param_index: ParamIndex,
    materialized_batches: Iterable[tuple[Sequence[str], int, Callable[[], torch.Tensor]]],
    materialization_replicate: int,
) -> GradientBank:
    """Capture detached prompt gradients via ``autograd.grad``, never ``.grad``.

    Each input item is ``(sample_ids, sample_count, mean_loss_closure)``.  The
    closure must use already materialized inputs and return the canonical mean
    cross-entropy loss for that batch.
    """

    captured = []
    parameters = param_index.parameters
    for batch_index, (sample_ids, sample_count, loss_closure) in enumerate(materialized_batches):
        if sample_count != len(sample_ids) or sample_count < 1:
            raise GradientBankError("Batch sample metadata is inconsistent")
        if any(parameter.grad is not None for parameter in parameters):
            raise GradientBankError("Gradient bank requires empty live .grad buffers")
        loss = loss_closure()
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise GradientBankError("Loss closure must return one scalar tensor")
        gradients = torch.autograd.grad(
            loss,
            parameters,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        state = GradientState.from_tensors(param_index, gradients)
        if any(parameter.grad is not None for parameter in parameters):
            raise GradientBankError("Gradient query contaminated live .grad buffers")
        captured.append(
            GradientBatch(
                batch_index=batch_index,
                sample_ids=tuple(sample_ids),
                sample_count=sample_count,
                mean_loss=float(loss.detach().item()),
                gradient=state,
                gradient_sha256=gradient_sha256(state),
            )
        )
    result = tuple(captured)
    return GradientBank(
        batches=result,
        exact=weighted_exact(result),
        total_samples=sum(item.sample_count for item in result),
        materialization_replicate=materialization_replicate,
    )
