"""Deterministic exact-reference scheduling and single-query reuse."""

from __future__ import annotations

from dataclasses import dataclass

from .diagnostics import DiagnosticMetrics, compute_gradient_diagnostics
from .estimators import EstimatorResult
from .full_gradient import (
    FullGradientResult,
    FullGradientService,
    FullGradientSweepMetadata,
)
from .gradient_state import GradientState
from .projection import DEFAULT_NORM_EPS, ProjectionResult


DIAGNOSTIC_EVENT_SCHEMA_VERSION = "sample_fg.diagnostic_event.v1"
EXACT_ESTIMATOR_REUSE = "exact_estimator_reuse"
PERIODIC_REFRESH_REUSE = "periodic_refresh_reuse"
INDEPENDENT_DIAGNOSTIC_QUERY = "independent_diagnostic_query"


class DiagnosticScheduleError(ValueError):
    """Raised for invalid cadence or exact-reference provenance."""


@dataclass(frozen=True)
class DiagnosticSchedule:
    """Concrete optimizer-step cadence; callers resolve symbolic epochs first."""

    interval_steps: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.interval_steps, bool)
            or not isinstance(self.interval_steps, int)
            or self.interval_steps < 1
        ):
            raise DiagnosticScheduleError(
                "diagnostic interval_steps must be a positive integer"
            )

    def is_due(self, optimizer_step: int) -> bool:
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
        ):
            raise DiagnosticScheduleError(
                "optimizer_step must be a zero-based nonnegative integer"
            )
        return optimizer_step % self.interval_steps == 0


@dataclass(frozen=True)
class DiagnosticReference:
    """Owned exact reference selected at unperturbed theta."""

    optimizer_step: int
    estimator_mode: str
    estimator_refreshed: bool
    source: str
    exact_service_query_issued: bool
    exact_reference: GradientState
    full_gradient_metadata: FullGradientSweepMetadata
    epoch: int | None
    batch_index: int | None


@dataclass(frozen=True)
class DiagnosticEvent:
    """One complete scheduled diagnostic with query provenance and metrics."""

    reference: DiagnosticReference
    metrics: DiagnosticMetrics

    def as_dict(self) -> dict[str, object]:
        metadata = self.reference.full_gradient_metadata
        return {
            "schema_version": DIAGNOSTIC_EVENT_SCHEMA_VERSION,
            "diagnostic_scheduled": True,
            "optimizer_step": self.reference.optimizer_step,
            "epoch": self.reference.epoch,
            "batch_index": self.reference.batch_index,
            "estimator_mode": self.reference.estimator_mode,
            "estimator_refresh": self.reference.estimator_refreshed,
            "exact_reference_source": self.reference.source,
            "exact_service_query_issued": self.reference.exact_service_query_issued,
            "exact_reference_reused": not self.reference.exact_service_query_issued,
            "exact_reference_auxiliary_seed": metadata.seed.as_dict(),
            "full_gradient": metadata.as_dict(),
            "metrics": self.metrics.as_dict(),
        }


class DiagnosticCoordinator:
    """Obtain at most one exact reference and finalize pure metrics."""

    def __init__(
        self,
        *,
        schedule: DiagnosticSchedule,
        full_gradient_service: FullGradientService,
        norm_eps: float = DEFAULT_NORM_EPS,
    ) -> None:
        if not isinstance(schedule, DiagnosticSchedule):
            raise TypeError("schedule must be a DiagnosticSchedule")
        if full_gradient_service is None or not callable(
            getattr(full_gradient_service, "compute", None)
        ):
            raise TypeError("full_gradient_service must expose compute()")
        self.schedule = schedule
        self.full_gradient_service = full_gradient_service
        self.norm_eps = norm_eps

    def reference_for_step(
        self,
        estimator_result: EstimatorResult,
        *,
        optimizer_step: int,
        epoch: int | None = None,
        batch_index: int | None = None,
    ) -> DiagnosticReference | None:
        if not isinstance(estimator_result, EstimatorResult):
            raise TypeError("estimator_result must be an EstimatorResult")
        if estimator_result.optimizer_step != optimizer_step:
            raise DiagnosticScheduleError(
                "Estimator result and diagnostic optimizer steps differ"
            )
        if not self.schedule.is_due(optimizer_step):
            return None

        exact_reference = estimator_result.exact_reference
        metadata = estimator_result.full_gradient_metadata
        if exact_reference is not None:
            if metadata is None:
                raise DiagnosticScheduleError(
                    "Reusable exact reference lacks sweep metadata"
                )
            if estimator_result.mode == "exact":
                source = EXACT_ESTIMATOR_REUSE
            elif estimator_result.mode == "periodic" and estimator_result.refreshed:
                source = PERIODIC_REFRESH_REUSE
            else:
                raise DiagnosticScheduleError(
                    "Unexpected estimator-owned exact reference"
                )
            issued = False
            reference = exact_reference.clone()
        else:
            queried = self.full_gradient_service.compute(
                optimizer_step=optimizer_step, purpose="diagnostic"
            )
            if not isinstance(queried, FullGradientResult):
                raise DiagnosticScheduleError(
                    "Diagnostic full-gradient service returned an invalid result"
                )
            estimator_result.active_global_estimate.assert_compatible(
                queried.gradient
            )
            if not queried.gradient.is_finite():
                raise DiagnosticScheduleError(
                    "Independent diagnostic exact gradient is nonfinite"
                )
            source = INDEPENDENT_DIAGNOSTIC_QUERY
            issued = True
            reference = queried.gradient.clone()
            metadata = queried.metadata

        return DiagnosticReference(
            optimizer_step=optimizer_step,
            estimator_mode=estimator_result.mode,
            estimator_refreshed=estimator_result.refreshed,
            source=source,
            exact_service_query_issued=issued,
            exact_reference=reference,
            full_gradient_metadata=metadata,
            epoch=epoch,
            batch_index=batch_index,
        )

    def finalize(
        self,
        reference: DiagnosticReference | None,
        *,
        batch_gradient: GradientState,
        active_global_estimate: GradientState,
        projection: ProjectionResult,
        perturbed_gradient: GradientState,
        alpha: float,
    ) -> DiagnosticEvent | None:
        if reference is None:
            return None
        metrics = compute_gradient_diagnostics(
            batch_gradient=batch_gradient,
            active_global_estimate=active_global_estimate,
            projection=projection,
            perturbed_gradient=perturbed_gradient,
            exact_full_gradient=reference.exact_reference,
            alpha=alpha,
            norm_eps=self.norm_eps,
        )
        return DiagnosticEvent(reference=reference, metrics=metrics)
