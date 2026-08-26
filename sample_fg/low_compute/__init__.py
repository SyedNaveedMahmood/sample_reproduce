"""Frozen-checkpoint, zero-optimizer-step mechanism probes.

This namespace is intentionally additive.  It does not expose a training
loop and it does not alter the R2 or extension runners.
"""

from .budget import ComputeBudget, ComputeBudgetError, TransitionGuard
from .checkpoint_probe import ProbeCheckpoint, load_probe_checkpoint
from .functional_probe import compare_functional_directions
from .gradient_bank import GradientBank, GradientBatch, build_gradient_bank
from .replay import ema_replay, stationary_ema_replay

__all__ = [
    "ComputeBudget",
    "ComputeBudgetError",
    "GradientBank",
    "GradientBatch",
    "ProbeCheckpoint",
    "TransitionGuard",
    "build_gradient_bank",
    "compare_functional_directions",
    "ema_replay",
    "load_probe_checkpoint",
    "stationary_ema_replay",
]
