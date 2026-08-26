"""First-order optimizer-step sequencing for CoOp sharpness-aware methods."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable, Generic, TypeVar

import torch

from .diagnostic_schedule import DiagnosticCoordinator, DiagnosticEvent
from .estimators import EstimatorResult, GlobalGradientEstimator
from .gradient_state import GradientState
from .param_index import ParamIndex
from .perturbation import PromptPerturbation
from .precision import OptimizerStepResult, PrecisionController
from .projection import (
    DEFAULT_NORM_EPS,
    ProjectionResult,
    project_batch_gradient,
    safe_unit,
)


BatchT = TypeVar("BatchT")
STEP_ENGINE_STATE_SCHEMA_VERSION = "sample_fg.step_engine_state.v1"


class StepEngineError(ValueError):
    """Raised for invalid optimizer-step configuration or sequencing."""


class StepEngineNumericalError(FloatingPointError):
    """Raised when an algorithm produces a nonfinite numerical state."""


@dataclass(frozen=True)
class SAMStepRecord:
    """Ephemeral numerical evidence from one vanilla-SAM logical step."""

    method: str
    optimizer_step: int
    loss_current: float
    loss_displaced: float
    batch_gradient: GradientState
    sam_perturbation: GradientState
    perturbed_gradient: GradientState
    final_gradient: GradientState
    batch_gradient_norm: float
    sam_perturbation_norm: float
    perturbed_gradient_norm: float
    final_gradient_norm: float
    batch_gradient_degenerate: bool
    restored_before_optimizer: bool
    same_batch_object_reused: bool
    optimizer_step_result: OptimizerStepResult


@dataclass(frozen=True)
class SAMPLeStepRecord:
    """Ephemeral numerical evidence from one first-order SAMPLe step."""

    method: str
    optimizer_step: int
    loss_current: float
    loss_displaced: float
    loss_sample_objective: float
    batch_gradient: GradientState
    estimator_result: EstimatorResult
    projection: ProjectionResult
    sam_perturbation: GradientState
    batch_correction: GradientState
    total_displacement: GradientState
    perturbed_gradient: GradientState
    final_gradient: GradientState
    diagnostic_event: DiagnosticEvent | None
    batch_gradient_norm: float
    global_direction_norm: float
    batch_component_norm: float
    sam_perturbation_norm: float
    batch_correction_norm: float
    total_displacement_norm: float
    perturbed_gradient_norm: float
    final_gradient_norm: float
    restored_before_optimizer: bool
    same_batch_object_reused: bool
    optimizer_step_result: OptimizerStepResult


class StepEngine(Generic[BatchT]):
    """Own logical sharpness-aware step order, not data or scheduling policy."""

    def __init__(
        self,
        *,
        param_index: ParamIndex,
        optimizer: torch.optim.Optimizer,
        precision_controller: PrecisionController,
        rho: Real,
        alpha: Real | None = None,
        norm_eps: Real = DEFAULT_NORM_EPS,
        perturbation: PromptPerturbation | None = None,
        diagnostic_coordinator: DiagnosticCoordinator | None = None,
    ) -> None:
        if not isinstance(param_index, ParamIndex):
            raise TypeError("param_index must be a ParamIndex")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        if not isinstance(precision_controller, PrecisionController):
            raise TypeError("precision_controller must be a PrecisionController")
        if isinstance(rho, bool) or not isinstance(rho, Real):
            raise TypeError("rho must be a finite nonnegative real number")
        self.rho = float(rho)
        if not math.isfinite(self.rho) or self.rho < 0:
            raise StepEngineError("rho must be a finite nonnegative value")
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, bool) or not isinstance(alpha, Real):
                raise TypeError("alpha must be a finite nonnegative real number")
            self.alpha = float(alpha)
            if not math.isfinite(self.alpha) or self.alpha < 0:
                raise StepEngineError("alpha must be a finite nonnegative value")
        if isinstance(norm_eps, bool) or not isinstance(norm_eps, Real):
            raise TypeError("norm_eps must be a finite nonnegative real number")
        self.norm_eps = float(norm_eps)
        if not math.isfinite(self.norm_eps) or self.norm_eps < 0:
            raise StepEngineError("norm_eps must be a finite nonnegative value")
        self.param_index = param_index
        self.optimizer = optimizer
        self.precision = precision_controller
        self.perturbation = (
            perturbation
            if perturbation is not None
            else PromptPerturbation(param_index)
        )
        if self.perturbation.param_index is not param_index:
            raise StepEngineError(
                "Perturbation controller must use the authoritative ParamIndex"
            )
        if diagnostic_coordinator is not None and not isinstance(
            diagnostic_coordinator, DiagnosticCoordinator
        ):
            raise TypeError(
                "diagnostic_coordinator must be a DiagnosticCoordinator"
            )
        self.diagnostic_coordinator = diagnostic_coordinator
        self._optimizer_step = 0

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    def state_dict(self) -> dict[str, object]:
        """Serialize the logical-step clock and immutable numerical contract."""

        self.perturbation.assert_inactive()
        return {
            "schema_version": STEP_ENGINE_STATE_SCHEMA_VERSION,
            "param_index_fingerprint_schema": self.param_index.fingerprint_schema,
            "param_index_fingerprint": self.param_index.fingerprint,
            "optimizer_step": self._optimizer_step,
            "rho": self.rho,
            "alpha": self.alpha,
            "norm_eps": self.norm_eps,
        }

    def load_state_dict(self, payload: object) -> None:
        """Restore only into a fresh, inactive engine before its first cycle."""

        self.perturbation.assert_inactive()
        if self._optimizer_step != 0 or self.precision.phase != "idle":
            raise StepEngineError(
                "Step-engine state can be loaded only into a fresh idle engine"
            )
        if not isinstance(payload, dict):
            raise StepEngineError("Step-engine payload must be a dictionary")
        expected = {
            "schema_version",
            "param_index_fingerprint_schema",
            "param_index_fingerprint",
            "optimizer_step",
            "rho",
            "alpha",
            "norm_eps",
        }
        if set(payload) != expected:
            raise StepEngineError("Step-engine payload fields differ from schema")
        if payload.get("schema_version") != STEP_ENGINE_STATE_SCHEMA_VERSION:
            raise StepEngineError("Unsupported step-engine state schema")
        if (
            payload.get("param_index_fingerprint_schema")
            != self.param_index.fingerprint_schema
            or payload.get("param_index_fingerprint") != self.param_index.fingerprint
        ):
            raise StepEngineError("Step-engine ParamIndex fingerprint differs")
        if payload.get("rho") != self.rho or payload.get("alpha") != self.alpha:
            raise StepEngineError("Step-engine method hyperparameters differ")
        if payload.get("norm_eps") != self.norm_eps:
            raise StepEngineError("Step-engine numerical threshold differs")
        optimizer_step = payload.get("optimizer_step")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
        ):
            raise StepEngineError("Serialized optimizer step is invalid")
        self._optimizer_step = optimizer_step

    def _loss_and_gradient(
        self,
        batch: BatchT,
        loss_closure: Callable[[BatchT], torch.Tensor],
    ) -> tuple[float, GradientState]:
        with self.precision.autocast_context():
            loss = loss_closure(batch)
        self.precision.backward(loss)
        capture = self.precision.capture_gradients(
            self.param_index, self.optimizer
        )
        value = float(loss.detach().item())
        if not math.isfinite(value) or not capture.state.is_finite():
            raise StepEngineNumericalError("Loss or logical gradient is nonfinite")
        return value, capture.state

    def _displaced_gradient(
        self,
        *,
        batch: BatchT,
        loss_closure: Callable[[BatchT], torch.Tensor],
        displacement: GradientState,
        parameter_values: tuple[torch.Tensor, ...],
    ) -> tuple[float, GradientState]:
        """Run the shared same-batch displaced pass and prove restoration."""

        self.precision.begin_additional_backward(self.optimizer)
        with self.perturbation.displaced(displacement):
            loss_displaced, perturbed_gradient = self._loss_and_gradient(
                batch, loss_closure
            )
        restored = all(
            torch.equal(entry.parameter, expected)
            for entry, expected in zip(self.param_index, parameter_values)
        )
        if not restored:
            raise StepEngineError(
                "Prompt parameters were not restored before optimizer.step"
            )
        return loss_displaced, perturbed_gradient

    def step_sam(
        self,
        batch: BatchT,
        loss_closure: Callable[[BatchT], torch.Tensor],
    ) -> SAMStepRecord:
        """Execute one vanilla-SAM step with final logical gradient ``p``."""

        if not callable(loss_closure):
            raise TypeError("loss_closure must be callable")
        self.perturbation.assert_inactive()
        parameter_values = tuple(
            entry.parameter.detach().clone() for entry in self.param_index
        )
        batch_identity = id(batch)

        self.precision.begin(self.optimizer)
        loss_current, batch_gradient = self._loss_and_gradient(
            batch, loss_closure
        )
        unit = safe_unit(batch_gradient, norm_eps=self.norm_eps)
        displacement = unit.unit.scale(self.rho)
        if not displacement.is_finite():
            raise StepEngineNumericalError("SAM perturbation is nonfinite")

        loss_displaced, perturbed_gradient = self._displaced_gradient(
            batch=batch,
            loss_closure=loss_closure,
            displacement=displacement,
            parameter_values=parameter_values,
        )

        # Vanilla SAM Eq. 6 uses the displaced-point gradient only. The
        # current-point gradient determines epsilon and is not accumulated.
        final_gradient = perturbed_gradient.clone()
        self.precision.install_logical_gradients(
            self.param_index, final_gradient
        )
        step_result = self.precision.step(self.optimizer)
        record = SAMStepRecord(
            method="sam",
            optimizer_step=self._optimizer_step,
            loss_current=loss_current,
            loss_displaced=loss_displaced,
            batch_gradient=batch_gradient,
            sam_perturbation=displacement,
            perturbed_gradient=perturbed_gradient,
            final_gradient=final_gradient,
            batch_gradient_norm=float(batch_gradient.norm().item()),
            sam_perturbation_norm=float(displacement.norm().item()),
            perturbed_gradient_norm=float(perturbed_gradient.norm().item()),
            final_gradient_norm=float(final_gradient.norm().item()),
            batch_gradient_degenerate=unit.degenerate,
            restored_before_optimizer=True,
            same_batch_object_reused=id(batch) == batch_identity,
            optimizer_step_result=step_result,
        )
        self._optimizer_step += 1
        return record

    def step_sample(
        self,
        batch: BatchT,
        loss_closure: Callable[[BatchT], torch.Tensor],
        estimator: GlobalGradientEstimator,
        *,
        epoch: int | None = None,
        batch_index: int | None = None,
    ) -> SAMPLeStepRecord:
        """Execute one shared SAMPLe step using the supplied direction source."""

        if self.alpha is None:
            raise StepEngineError("SAMPLe requires a configured alpha")
        if not callable(loss_closure):
            raise TypeError("loss_closure must be callable")
        if not isinstance(estimator, GlobalGradientEstimator):
            raise TypeError("estimator must implement GlobalGradientEstimator")
        self.param_index.assert_compatible(estimator.param_index)
        self.perturbation.assert_inactive()
        parameter_values = tuple(
            entry.parameter.detach().clone() for entry in self.param_index
        )
        batch_identity = id(batch)

        self.precision.begin(self.optimizer)
        loss_current, batch_gradient = self._loss_and_gradient(
            batch, loss_closure
        )
        estimator_result = estimator.global_direction(
            batch_grad=batch_gradient,
            optimizer_step=self._optimizer_step,
        )
        diagnostic_reference = (
            self.diagnostic_coordinator.reference_for_step(
                estimator_result,
                optimizer_step=self._optimizer_step,
                epoch=epoch,
                batch_index=batch_index,
            )
            if self.diagnostic_coordinator is not None
            else None
        )
        global_direction = estimator_result.active_global_estimate
        projection = project_batch_gradient(
            batch_gradient,
            global_direction,
            norm_eps=self.norm_eps,
        )
        unit = safe_unit(batch_gradient, norm_eps=self.norm_eps)
        sam_perturbation = unit.unit.scale(self.rho)
        batch_correction = projection.batch_component.scale(self.alpha)
        total_displacement = sam_perturbation.subtract(batch_correction)
        for label, state in (
            ("SAM perturbation", sam_perturbation),
            ("batch correction", batch_correction),
            ("total displacement", total_displacement),
        ):
            if not state.is_finite():
                raise StepEngineNumericalError(f"{label} is nonfinite")

        loss_displaced, perturbed_gradient = self._displaced_gradient(
            batch=batch,
            loss_closure=loss_closure,
            displacement=total_displacement,
            parameter_values=parameter_values,
        )
        # Eq. 10 is a sum of current and displaced losses. Under the accepted
        # first-order stop-gradient policy, the final logical gradient is the
        # sum g+p, with no division by two.
        final_gradient = batch_gradient.add(perturbed_gradient)
        if not final_gradient.is_finite():
            raise StepEngineNumericalError("SAMPLe final gradient is nonfinite")
        diagnostic_event = (
            self.diagnostic_coordinator.finalize(
                diagnostic_reference,
                batch_gradient=batch_gradient,
                active_global_estimate=global_direction,
                projection=projection,
                perturbed_gradient=perturbed_gradient,
                alpha=self.alpha,
            )
            if self.diagnostic_coordinator is not None
            else None
        )
        self.precision.install_logical_gradients(
            self.param_index, final_gradient
        )
        step_result = self.precision.step(self.optimizer)
        record = SAMPLeStepRecord(
            method="sample",
            optimizer_step=self._optimizer_step,
            loss_current=loss_current,
            loss_displaced=loss_displaced,
            loss_sample_objective=loss_current + loss_displaced,
            batch_gradient=batch_gradient,
            estimator_result=estimator_result,
            projection=projection,
            sam_perturbation=sam_perturbation,
            batch_correction=batch_correction,
            total_displacement=total_displacement,
            perturbed_gradient=perturbed_gradient,
            final_gradient=final_gradient,
            diagnostic_event=diagnostic_event,
            batch_gradient_norm=float(batch_gradient.norm().item()),
            global_direction_norm=float(global_direction.norm().item()),
            batch_component_norm=float(projection.batch_component.norm().item()),
            sam_perturbation_norm=float(sam_perturbation.norm().item()),
            batch_correction_norm=float(batch_correction.norm().item()),
            total_displacement_norm=float(total_displacement.norm().item()),
            perturbed_gradient_norm=float(perturbed_gradient.norm().item()),
            final_gradient_norm=float(final_gradient.norm().item()),
            restored_before_optimizer=True,
            same_batch_object_reused=id(batch) == batch_identity,
            optimizer_step_result=step_result,
        )
        self._optimizer_step += 1
        return record
