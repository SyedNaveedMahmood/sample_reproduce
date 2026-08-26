"""Precision-safe first-order backward and logical-gradient boundaries."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import ContextManager

import torch
from torch.cuda.amp import GradScaler, autocast

from .gradient_state import GradientState
from .param_index import ParamIndex


PRECISION_STATE_SCHEMA = "sample_fg.precision_controller.v1"
SUPPORTED_PRECISION_MODES = ("fp16", "fp32", "amp")


class PrecisionError(ValueError):
    """Raised for invalid precision configuration or structural state."""


class PrecisionStateError(RuntimeError):
    """Raised when precision operations are requested out of sequence."""


class PrecisionNumericalError(FloatingPointError):
    """Raised when loss or logical gradients contain NaN/Inf."""


class _Phase(Enum):
    IDLE = "idle"
    READY = "ready"
    BACKWARD_COMPLETE = "backward_complete"
    CAPTURED = "captured"
    INSTALLED = "installed"
    STEPPED = "stepped"


@dataclass(frozen=True)
class GradientCapture:
    """Owned logical gradient plus non-owning scalar/dtype metadata."""

    state: GradientState
    precision_mode: str
    scaling_active: bool
    scale: float | None
    authoritative_unscale_performed: bool
    live_dtypes_before_unscale: tuple[str, ...]
    live_dtypes_after_unscale: tuple[str, ...]
    live_devices: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerStepResult:
    """Precision transition metadata for one delegated optimizer step."""

    precision_mode: str
    scaling_active: bool
    scale_before: float | None
    scale_after: float | None
    scaler_step_skipped: bool


class PrecisionController:
    """Own precision mechanics without owning an optimization algorithm."""

    def __init__(self, mode: str, scaler: GradScaler | None = None):
        if mode not in SUPPORTED_PRECISION_MODES:
            raise PrecisionError(
                f"Unsupported precision mode {mode!r}; "
                f"expected one of {SUPPORTED_PRECISION_MODES}"
            )
        if mode == "amp":
            self.scaler = scaler if scaler is not None else GradScaler()
        elif scaler is not None:
            raise PrecisionError("A GradScaler is valid only for AMP mode")
        else:
            self.scaler = None
        self.mode = mode
        self._phase = _Phase.IDLE
        self._optimizer_id: int | None = None
        self._captured_index: ParamIndex | None = None

    @property
    def phase(self) -> str:
        return self._phase.value

    @property
    def scaling_active(self) -> bool:
        return bool(
            self.mode == "amp"
            and self.scaler is not None
            and self.scaler.is_enabled()
        )

    def autocast_context(self) -> ContextManager:
        """Match pinned CoOp: autocast only for the explicit AMP mode."""

        if self.mode == "amp":
            return autocast()
        return nullcontext()

    def begin(self, optimizer: torch.optim.Optimizer) -> None:
        """Start one precision cycle and clear live gradients once."""

        if self._phase not in {_Phase.IDLE, _Phase.STEPPED}:
            raise PrecisionStateError(
                f"Cannot begin a precision cycle from phase {self.phase!r}"
            )
        optimizer.zero_grad(set_to_none=True)
        self._optimizer_id = id(optimizer)
        self._captured_index = None
        self._phase = _Phase.READY

    def backward(self, loss: torch.Tensor) -> None:
        """Run one ordinary first-order backward, scaled only in AMP mode."""

        if self._phase is not _Phase.READY:
            raise PrecisionStateError(
                f"Backward requires phase 'ready', observed {self.phase!r}"
            )
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise PrecisionError("loss must be a scalar tensor")
        if not loss.is_floating_point():
            raise PrecisionError("loss must have a floating dtype")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise PrecisionNumericalError("Loss is NaN or Inf")

        backward_loss = self.scaler.scale(loss) if self.mode == "amp" else loss
        torch.autograd.backward(
            backward_loss,
            retain_graph=False,
            create_graph=False,
        )
        self._phase = _Phase.BACKWARD_COMPLETE

    def begin_additional_backward(
        self, optimizer: torch.optim.Optimizer
    ) -> None:
        """Clear live gradients for another first-order evaluation.

        SAM/SAMPLe need two logical gradients but still perform one optimizer
        transition. Native FP32/fp16 can restart the backward phase directly.
        AMP requires a dedicated multi-backward scaler design because PyTorch
        forbids calling ``unscale_`` twice before ``update``; it is rejected
        here instead of resetting scaler internals or manually dividing.
        """

        self._require_optimizer(optimizer)
        if self._phase is not _Phase.CAPTURED:
            raise PrecisionStateError(
                "An additional backward requires one captured logical gradient; "
                f"observed phase {self.phase!r}"
            )
        if self.mode == "amp":
            raise PrecisionStateError(
                "Multiple logical backward captures in one AMP optimizer cycle "
                "are not supported by this precision controller"
            )
        optimizer.zero_grad(set_to_none=True)
        self._captured_index = None
        self._phase = _Phase.READY

    def capture_gradients(
        self,
        param_index: ParamIndex,
        optimizer: torch.optim.Optimizer,
    ) -> GradientCapture:
        """Authoritatively unscale once, then own a detached FP32 state."""

        self._require_optimizer(optimizer)
        if self._phase is not _Phase.BACKWARD_COMPLETE:
            raise PrecisionStateError(
                "Gradient capture requires exactly one completed backward and "
                f"no prior capture; observed phase {self.phase!r}"
            )
        self._assert_optimizer_matches_index(optimizer, param_index)
        before_dtypes, devices = self._live_gradient_metadata(param_index)
        scale = None
        unscaled = False
        if self.mode == "amp":
            scale = float(self.scaler.get_scale())
            self.scaler.unscale_(optimizer)
            unscaled = True
        after_dtypes, after_devices = self._live_gradient_metadata(param_index)
        if devices != after_devices:
            raise PrecisionStateError("Live gradient device changed during unscale")

        state = GradientState.from_parameter_grads(param_index)
        if not state.is_finite():
            raise PrecisionNumericalError(
                "Logical gradient contains NaN or Inf after authoritative unscale"
            )
        self._captured_index = param_index
        self._phase = _Phase.CAPTURED
        return GradientCapture(
            state=state,
            precision_mode=self.mode,
            scaling_active=self.scaling_active,
            scale=scale,
            authoritative_unscale_performed=unscaled,
            live_dtypes_before_unscale=before_dtypes,
            live_dtypes_after_unscale=after_dtypes,
            live_devices=devices,
        )

    def install_logical_gradients(
        self,
        param_index: ParamIndex,
        state: GradientState,
    ) -> tuple[str, ...]:
        """Install one finite logical state in live parameter gradient dtype."""

        if self._phase is not _Phase.CAPTURED or self._captured_index is None:
            raise PrecisionStateError(
                f"Gradient installation requires phase 'captured', observed {self.phase!r}"
            )
        self._captured_index.assert_compatible(param_index)
        param_index.assert_compatible(state.param_index)
        state.assert_valid()
        if not state.is_finite():
            raise PrecisionNumericalError("Cannot install a nonfinite logical gradient")

        converted = []
        for entry, component in zip(param_index, state):
            if component.device != entry.parameter.device:
                raise PrecisionError(
                    f"Device mismatch for {entry.name!r}: "
                    f"{component.device} != {entry.parameter.device}"
                )
            value = component.to(dtype=entry.parameter.dtype)
            if not bool(torch.isfinite(value).all().item()):
                raise PrecisionNumericalError(
                    f"Logical gradient becomes nonfinite in live dtype: {entry.name!r}"
                )
            converted.append(value)

        with torch.no_grad():
            for entry, value in zip(param_index, converted):
                if entry.parameter.grad is None:
                    entry.parameter.grad = value.clone()
                else:
                    entry.parameter.grad.copy_(value)
        self._phase = _Phase.INSTALLED
        return tuple(str(entry.parameter.grad.dtype) for entry in param_index)

    def step(self, optimizer: torch.optim.Optimizer) -> OptimizerStepResult:
        """Delegate exactly one optimizer step and one AMP scaler update."""

        self._require_optimizer(optimizer)
        if self._phase not in {_Phase.CAPTURED, _Phase.INSTALLED}:
            raise PrecisionStateError(
                f"Optimizer step requires captured/installed gradients, "
                f"observed {self.phase!r}"
            )
        self._require_finite_live_gradients()

        if self.mode == "amp":
            scale_before = float(self.scaler.get_scale())
            self.scaler.step(optimizer)
            self.scaler.update()
            scale_after = float(self.scaler.get_scale())
            skipped = self.scaling_active and scale_after < scale_before
        else:
            optimizer.step()
            scale_before = None
            scale_after = None
            skipped = False
        self._phase = _Phase.STEPPED
        return OptimizerStepResult(
            precision_mode=self.mode,
            scaling_active=self.scaling_active,
            scale_before=scale_before,
            scale_after=scale_after,
            scaler_step_skipped=skipped,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": PRECISION_STATE_SCHEMA,
            "mode": self.mode,
            "scaler_state_dict": self.scaler.state_dict()
            if self.scaler is not None
            else None,
        }

    def load_state_dict(self, payload: object) -> None:
        if self._phase is not _Phase.IDLE:
            raise PrecisionStateError("Precision state can be loaded only while idle")
        if not isinstance(payload, dict):
            raise PrecisionError("Precision state payload must be a dictionary")
        if payload.get("schema_version") != PRECISION_STATE_SCHEMA:
            raise PrecisionError("Unsupported precision state schema")
        if payload.get("mode") != self.mode:
            raise PrecisionError("Precision mode differs from serialized state")
        scaler_state = payload.get("scaler_state_dict")
        if self.mode == "amp":
            if not isinstance(scaler_state, dict):
                raise PrecisionError("AMP precision state requires scaler state")
            self.scaler.load_state_dict(scaler_state)
        elif scaler_state is not None:
            raise PrecisionError("Non-AMP precision state cannot contain a scaler")

    def _require_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if self._optimizer_id is None or id(optimizer) != self._optimizer_id:
            raise PrecisionStateError("Optimizer differs from the active precision cycle")

    @staticmethod
    def _assert_optimizer_matches_index(
        optimizer: torch.optim.Optimizer,
        param_index: ParamIndex,
    ) -> None:
        optimizer_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise PrecisionError("Optimizer contains a duplicate parameter")
        if set(optimizer_ids) != {id(parameter) for parameter in param_index.parameters}:
            raise PrecisionError("Optimizer parameters differ from ParamIndex")

    @staticmethod
    def _live_gradient_metadata(
        param_index: ParamIndex,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        dtypes = []
        devices = []
        for entry in param_index:
            gradient = entry.parameter.grad
            if gradient is None:
                raise PrecisionStateError(
                    f"Required live gradient is missing: {entry.name!r}"
                )
            dtypes.append(str(gradient.dtype))
            devices.append(str(gradient.device))
        return tuple(dtypes), tuple(devices)

    def _require_finite_live_gradients(self) -> None:
        if self._captured_index is None:
            raise PrecisionStateError("No captured ParamIndex is available")
        for entry in self._captured_index:
            gradient = entry.parameter.grad
            if gradient is None:
                raise PrecisionStateError(
                    f"Required live gradient is missing before step: {entry.name!r}"
                )
            if not bool(torch.isfinite(gradient).all().item()):
                raise PrecisionNumericalError(
                    f"Live gradient is nonfinite before step: {entry.name!r}"
                )
