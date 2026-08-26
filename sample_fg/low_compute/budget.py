"""Fail-closed work accounting for low-compute probes."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import torch


BUDGET_SCHEMA_VERSION = "sample_fg.low_compute_budget.v1"


class ComputeBudgetError(RuntimeError):
    """Raised before a probe can exceed its checked-in permit."""


@dataclass
class ComputeBudget:
    optimizer_steps: int = 0
    scheduler_steps: int = 0
    normal_forward_batches: int = 0
    normal_backward_batches: int = 0
    exact_forward_batches: int = 0
    exact_backward_batches: int = 0
    image_encoder_forward_batches: int = 0
    text_encoder_forward_calls: int = 0
    exact_sweeps: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ComputeBudgetError(f"{name} must be a nonnegative integer")

    @property
    def backward_batches(self) -> int:
        return self.normal_backward_batches + self.exact_backward_batches

    def require_read_only(self) -> None:
        if self.optimizer_steps != 0 or self.scheduler_steps != 0:
            raise ComputeBudgetError(
                "Frozen probes require zero optimizer and scheduler steps"
            )

    def assert_within(self, permit: "ComputeBudget") -> None:
        self.require_read_only()
        permit.require_read_only()
        for name, value in asdict(self).items():
            maximum = getattr(permit, name)
            if value > maximum:
                raise ComputeBudgetError(
                    f"Budget overflow for {name}: requested {value}, permit {maximum}"
                )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "schema_version": BUDGET_SCHEMA_VERSION,
                "backward_batches": self.backward_batches,
            }
        )
        return payload


class TransitionGuard:
    """Snapshot optimizer/scheduler state and forbid lifecycle transitions."""

    def __init__(self, optimizer: Any = None, scheduler: Any = None) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer_steps = 0
        self.scheduler_steps = 0
        self._optimizer_state = None
        self._scheduler_state = None

    def __enter__(self) -> "TransitionGuard":
        if self.optimizer is not None:
            self._optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        if self.scheduler is not None:
            self._scheduler_state = copy.deepcopy(self.scheduler.state_dict())
        return self

    def optimizer_step(self, *_args, **_kwargs) -> None:
        self.optimizer_steps += 1
        raise ComputeBudgetError("optimizer.step() is forbidden in a frozen probe")

    def scheduler_step(self, *_args, **_kwargs) -> None:
        self.scheduler_steps += 1
        raise ComputeBudgetError("scheduler.step() is forbidden in a frozen probe")

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.optimizer_steps or self.scheduler_steps:
            raise ComputeBudgetError("A forbidden lifecycle transition was attempted")
        if self.optimizer is not None and not _state_equal(
            self.optimizer.state_dict(), self._optimizer_state
        ):
            raise ComputeBudgetError("Optimizer state changed during a frozen probe")
        if self.scheduler is not None and not _state_equal(
            self.scheduler.state_dict(), self._scheduler_state
        ):
            raise ComputeBudgetError("Scheduler state changed during a frozen probe")
        return False


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict) and isinstance(right, dict)
            and set(left) == set(right)
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            type(left) is type(right) and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right))
        )
    return left == right
