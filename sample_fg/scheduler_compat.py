"""Narrow PyTorch compatibility for pinned Dassl warmup schedulers.

Pinned Dassl passes the removed ``verbose`` positional argument to
``_LRScheduler``.  These wrappers preserve its epoch-level LR formulas while
using the current constructor and a primitive-only checkpoint payload.
"""

from __future__ import annotations

from typing import Any

from torch.optim.lr_scheduler import _LRScheduler

from dassl.optim.lr_scheduler import build_lr_scheduler as _build_pinned_scheduler


class _WarmupSchedulerCompat(_LRScheduler):
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch: int,
        last_epoch: int = -1,
    ) -> None:
        self.successor = successor
        self.warmup_epoch = warmup_epoch
        # PyTorch 2.11 removed the legacy ``verbose`` argument.
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if self.last_epoch >= self.warmup_epoch:
            self.successor.step(epoch)
            self._last_lr = self.successor.get_last_lr()
        else:
            super().step(epoch)

    def state_dict(self) -> dict[str, Any]:
        """Serialize without embedding a scheduler Python object.

        The pinned wrapper inherits ``LRScheduler.state_dict()``, which places
        ``successor`` itself in the payload.  PyTorch 2.11's safe-by-default
        loader rejects that object.  Storing the successor's state instead
        preserves resume semantics and remains safe-loader compatible.
        """

        state = {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"optimizer", "successor"}
        }
        state["_successor_state_dict"] = self.successor.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        state = dict(state_dict)
        successor_state = state.pop("_successor_state_dict")
        self.__dict__.update(state)
        self.successor.load_state_dict(successor_state)


class ConstantWarmupSchedulerCompat(_WarmupSchedulerCompat):
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch: int,
        cons_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.cons_lr = cons_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        return [self.cons_lr for _ in self.base_lrs]


class LinearWarmupSchedulerCompat(_WarmupSchedulerCompat):
    def __init__(
        self,
        optimizer,
        successor,
        warmup_epoch: int,
        min_lr: float,
        last_epoch: int = -1,
    ) -> None:
        self.min_lr = min_lr
        super().__init__(optimizer, successor, warmup_epoch, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epoch:
            return self.successor.get_last_lr()
        if self.last_epoch == 0:
            return [self.min_lr for _ in self.base_lrs]
        return [
            lr * self.last_epoch / self.warmup_epoch for lr in self.base_lrs
        ]


def build_lr_scheduler_compat(optimizer, optim_cfg):
    """Build the pinned scheduler, adapting only its warmup wrapper API."""

    if optim_cfg.WARMUP_EPOCH <= 0:
        return _build_pinned_scheduler(optimizer, optim_cfg)

    successor_cfg = optim_cfg.clone()
    successor_cfg.defrost()
    successor_cfg.WARMUP_EPOCH = -1
    successor_cfg.freeze()
    successor = _build_pinned_scheduler(optimizer, successor_cfg)

    if not optim_cfg.WARMUP_RECOUNT:
        successor.last_epoch = optim_cfg.WARMUP_EPOCH

    if optim_cfg.WARMUP_TYPE == "constant":
        return ConstantWarmupSchedulerCompat(
            optimizer,
            successor,
            optim_cfg.WARMUP_EPOCH,
            optim_cfg.WARMUP_CONS_LR,
        )
    if optim_cfg.WARMUP_TYPE == "linear":
        return LinearWarmupSchedulerCompat(
            optimizer,
            successor,
            optim_cfg.WARMUP_EPOCH,
            optim_cfg.WARMUP_MIN_LR,
        )
    raise ValueError(f"Unknown warmup type: {optim_cfg.WARMUP_TYPE}")
