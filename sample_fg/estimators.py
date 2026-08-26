"""Estimator-only state machines for SAMPLe global-gradient directions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .full_gradient import (
    FullGradientResult,
    FullGradientService,
    FullGradientSweepMetadata,
)
from .gradient_state import GradientState
from .param_index import ParamIndex, ParamIndexMismatchError


ESTIMATOR_STATE_SCHEMA_VERSION = "sample_fg.estimator_state.v1"


class EstimatorError(ValueError):
    """Raised for invalid estimator configuration or inputs."""


class EstimatorStateError(RuntimeError):
    """Raised for invalid step transitions or serialized state."""


class EstimatorNumericalError(FloatingPointError):
    """Raised when an estimator receives or produces nonfinite state."""


@dataclass(frozen=True)
class EstimatorResult:
    """Owned active direction and transition metadata for one logical step."""

    active_global_estimate: GradientState
    mode: str
    optimizer_step: int
    refreshed: bool
    age_steps: int | None
    last_refresh_step: int | None
    exact_reference: GradientState | None
    full_gradient_metadata: FullGradientSweepMetadata | None
    exact_query_count: int

    def as_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "optimizer_step": self.optimizer_step,
            "refreshed": self.refreshed,
            "age_steps": self.age_steps,
            "last_refresh_step": self.last_refresh_step,
            "exact_query_count": self.exact_query_count,
            "active_global_norm": float(self.active_global_estimate.norm().item()),
            "active_global_finite": self.active_global_estimate.is_finite(),
            "exact_reference_available": self.exact_reference is not None,
            "full_gradient": (
                self.full_gradient_metadata.as_dict()
                if self.full_gradient_metadata is not None
                else None
            ),
        }


class GlobalGradientEstimator(ABC):
    """Common zero-based, sequential estimator contract."""

    mode: str

    def __init__(self, param_index: ParamIndex):
        if not isinstance(param_index, ParamIndex):
            raise TypeError("param_index must be a ParamIndex")
        self.param_index = param_index
        self._last_processed_step: int | None = None
        self._exact_query_count = 0

    @property
    def last_processed_step(self) -> int | None:
        return self._last_processed_step

    @property
    def exact_query_count(self) -> int:
        return self._exact_query_count

    @property
    def active_state(self) -> GradientState | None:
        return None

    @abstractmethod
    def global_direction(
        self, *, batch_grad: GradientState, optimizer_step: int
    ) -> EstimatorResult:
        """Consume the current unperturbed mini-batch gradient."""

    @abstractmethod
    def state_dict(self) -> dict[str, object]:
        """Return an owned, versioned estimator payload."""

    @abstractmethod
    def load_state_dict(self, payload: object) -> None:
        """Restore a validated payload without checkpoint-file integration."""

    def _validate_call(
        self, batch_grad: GradientState, optimizer_step: int
    ) -> None:
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
        ):
            raise EstimatorStateError(
                "optimizer_step must be a zero-based nonnegative integer"
            )
        expected = 0 if self._last_processed_step is None else self._last_processed_step + 1
        if optimizer_step != expected:
            raise EstimatorStateError(
                f"Expected logical optimizer step {expected}, observed {optimizer_step}"
            )
        self._validate_gradient_state(batch_grad, label="batch_grad")

    def _validate_gradient_state(
        self, state: GradientState, *, label: str
    ) -> None:
        if not isinstance(state, GradientState):
            raise EstimatorStateError(f"{label} must be a GradientState")
        try:
            self.param_index.assert_compatible(state.param_index)
        except ParamIndexMismatchError as error:
            raise EstimatorStateError(f"{label} ParamIndex mismatch") from error
        state.assert_valid()
        for entry, component in zip(self.param_index, state):
            if component.device != entry.parameter.device:
                raise EstimatorStateError(
                    f"{label} device differs for {entry.name!r}: "
                    f"{component.device} != {entry.parameter.device}"
                )
        if not state.is_finite():
            raise EstimatorNumericalError(f"{label} contains NaN or Inf")

    def _query_exact(
        self,
        service: FullGradientService,
        *,
        optimizer_step: int,
        purpose: str,
    ) -> FullGradientResult:
        result = service.compute(optimizer_step=optimizer_step, purpose=purpose)
        if not isinstance(result, FullGradientResult):
            raise EstimatorStateError(
                "FullGradientService returned an incompatible result"
            )
        self._validate_gradient_state(result.gradient, label="exact full gradient")
        return result

    def _common_state_dict(self) -> dict[str, object]:
        return {
            "schema_version": ESTIMATOR_STATE_SCHEMA_VERSION,
            "mode": self.mode,
            "param_index_fingerprint_schema": self.param_index.fingerprint_schema,
            "param_index_fingerprint": self.param_index.fingerprint,
            "last_processed_step": self._last_processed_step,
            "exact_query_count": self._exact_query_count,
        }

    def _load_common(
        self, payload: object, *, expected_keys: set[str]
    ) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise EstimatorStateError("Estimator payload must be a dictionary")
        if set(payload) != expected_keys:
            raise EstimatorStateError("Estimator payload fields differ from schema")
        if payload.get("schema_version") != ESTIMATOR_STATE_SCHEMA_VERSION:
            raise EstimatorStateError("Unsupported estimator state schema")
        if payload.get("mode") != self.mode:
            raise EstimatorStateError("Serialized estimator mode differs")
        if (
            payload.get("param_index_fingerprint_schema")
            != self.param_index.fingerprint_schema
            or payload.get("param_index_fingerprint") != self.param_index.fingerprint
        ):
            raise EstimatorStateError("Serialized ParamIndex fingerprint differs")
        last_step = payload.get("last_processed_step")
        if last_step is not None and (
            isinstance(last_step, bool)
            or not isinstance(last_step, int)
            or last_step < 0
        ):
            raise EstimatorStateError("Invalid serialized last_processed_step")
        query_count = payload.get("exact_query_count")
        if (
            isinstance(query_count, bool)
            or not isinstance(query_count, int)
            or query_count < 0
        ):
            raise EstimatorStateError("Invalid serialized exact_query_count")
        return payload


class EMAEstimator(GlobalGradientEstimator):
    """Paper Eq. 9 with zero initialization and no bias correction."""

    mode = "ema"

    def __init__(self, param_index: ParamIndex, *, ema_lambda: float):
        super().__init__(param_index)
        self.ema_lambda = _validate_ema_lambda(ema_lambda)
        self._state = GradientState.zeros(param_index)

    @property
    def active_state(self) -> GradientState:
        return self._state.clone()

    def global_direction(
        self, *, batch_grad: GradientState, optimizer_step: int
    ) -> EstimatorResult:
        self._validate_call(batch_grad, optimizer_step)
        candidate = self._state.clone().affine_(
            batch_grad,
            self_weight=self.ema_lambda,
            other_weight=1.0 - self.ema_lambda,
        )
        self._validate_gradient_state(candidate, label="EMA candidate")
        self._state = candidate.clone()
        self._last_processed_step = optimizer_step
        return EstimatorResult(
            active_global_estimate=candidate.clone(),
            mode=self.mode,
            optimizer_step=optimizer_step,
            refreshed=False,
            age_steps=None,
            last_refresh_step=None,
            exact_reference=None,
            full_gradient_metadata=None,
            exact_query_count=0,
        )

    def state_dict(self) -> dict[str, object]:
        payload = self._common_state_dict()
        payload.update(
            {
                "ema_lambda": self.ema_lambda,
                "active_state": self._state.state_dict(),
            }
        )
        return payload

    def load_state_dict(self, payload: object) -> None:
        expected = {
            "schema_version",
            "mode",
            "param_index_fingerprint_schema",
            "param_index_fingerprint",
            "last_processed_step",
            "exact_query_count",
            "ema_lambda",
            "active_state",
        }
        loaded = self._load_common(payload, expected_keys=expected)
        if loaded["ema_lambda"] != self.ema_lambda:
            raise EstimatorStateError("Serialized EMA lambda differs")
        if loaded["exact_query_count"] != 0:
            raise EstimatorStateError("EMA cannot contain exact query count")
        state = GradientState.from_state_dict(
            self.param_index, loaded["active_state"]
        )
        self._validate_gradient_state(state, label="serialized EMA state")
        self._state = state
        self._last_processed_step = loaded["last_processed_step"]
        self._exact_query_count = 0


class ExactEstimator(GlobalGradientEstimator):
    """Use one current-theta exact full-gradient query at every logical step."""

    mode = "exact"

    def __init__(
        self, param_index: ParamIndex, *, full_gradient_service: FullGradientService
    ):
        super().__init__(param_index)
        self.full_gradient_service = full_gradient_service

    def global_direction(
        self, *, batch_grad: GradientState, optimizer_step: int
    ) -> EstimatorResult:
        self._validate_call(batch_grad, optimizer_step)
        exact = self._query_exact(
            self.full_gradient_service,
            optimizer_step=optimizer_step,
            purpose="optimization_exact",
        )
        active = exact.gradient.clone()
        reference = exact.gradient.clone()
        self._exact_query_count += 1
        self._last_processed_step = optimizer_step
        return EstimatorResult(
            active_global_estimate=active,
            mode=self.mode,
            optimizer_step=optimizer_step,
            refreshed=True,
            age_steps=None,
            last_refresh_step=optimizer_step,
            exact_reference=reference,
            full_gradient_metadata=exact.metadata,
            exact_query_count=self._exact_query_count,
        )

    def state_dict(self) -> dict[str, object]:
        return self._common_state_dict()

    def load_state_dict(self, payload: object) -> None:
        expected = {
            "schema_version",
            "mode",
            "param_index_fingerprint_schema",
            "param_index_fingerprint",
            "last_processed_step",
            "exact_query_count",
        }
        loaded = self._load_common(payload, expected_keys=expected)
        last_step = loaded["last_processed_step"]
        expected_queries = 0 if last_step is None else last_step + 1
        if loaded["exact_query_count"] != expected_queries:
            raise EstimatorStateError("Exact query counter differs from step history")
        self._last_processed_step = last_step
        self._exact_query_count = loaded["exact_query_count"]


class PeriodicEstimator(GlobalGradientEstimator):
    """Hard exact reset at ``t % K == 0``; EMA only between refreshes."""

    mode = "periodic"

    def __init__(
        self,
        param_index: ParamIndex,
        *,
        ema_lambda: float,
        refresh_k_steps: int,
        full_gradient_service: FullGradientService,
    ):
        super().__init__(param_index)
        self.ema_lambda = _validate_ema_lambda(ema_lambda)
        if (
            isinstance(refresh_k_steps, bool)
            or not isinstance(refresh_k_steps, int)
            or refresh_k_steps < 1
        ):
            raise EstimatorError("refresh_k_steps must be a positive integer")
        self.refresh_k_steps = refresh_k_steps
        self.full_gradient_service = full_gradient_service
        self._state: GradientState | None = None
        self._last_refresh_step: int | None = None

    @property
    def active_state(self) -> GradientState | None:
        return self._state.clone() if self._state is not None else None

    @property
    def last_refresh_step(self) -> int | None:
        return self._last_refresh_step

    @property
    def age_steps(self) -> int | None:
        if self._last_processed_step is None or self._last_refresh_step is None:
            return None
        return self._last_processed_step - self._last_refresh_step

    def global_direction(
        self, *, batch_grad: GradientState, optimizer_step: int
    ) -> EstimatorResult:
        self._validate_call(batch_grad, optimizer_step)
        refresh = optimizer_step % self.refresh_k_steps == 0
        exact: FullGradientResult | None = None
        if refresh:
            # K=1 is a true alias of exact, including its purpose-derived RNG
            # realization. K>1 uses the distinct periodic-refresh substream.
            purpose = (
                "optimization_exact"
                if self.refresh_k_steps == 1
                else "periodic_refresh"
            )
            exact = self._query_exact(
                self.full_gradient_service,
                optimizer_step=optimizer_step,
                purpose=purpose,
            )
            candidate = exact.gradient.clone()
            last_refresh = optimizer_step
        else:
            if self._state is None or self._last_refresh_step is None:
                raise EstimatorStateError(
                    "Periodic non-refresh step has no prior exact state"
                )
            candidate = self._state.clone().affine_(
                batch_grad,
                self_weight=self.ema_lambda,
                other_weight=1.0 - self.ema_lambda,
            )
            last_refresh = self._last_refresh_step
        self._validate_gradient_state(candidate, label="periodic candidate")

        self._state = candidate.clone()
        self._last_refresh_step = last_refresh
        if exact is not None:
            self._exact_query_count += 1
        self._last_processed_step = optimizer_step
        age = optimizer_step - last_refresh
        return EstimatorResult(
            active_global_estimate=candidate.clone(),
            mode=self.mode,
            optimizer_step=optimizer_step,
            refreshed=refresh,
            age_steps=age,
            last_refresh_step=last_refresh,
            exact_reference=exact.gradient.clone() if exact is not None else None,
            full_gradient_metadata=exact.metadata if exact is not None else None,
            exact_query_count=self._exact_query_count,
        )

    def state_dict(self) -> dict[str, object]:
        payload = self._common_state_dict()
        payload.update(
            {
                "ema_lambda": self.ema_lambda,
                "refresh_k_steps": self.refresh_k_steps,
                "last_refresh_step": self._last_refresh_step,
                "active_state": (
                    self._state.state_dict() if self._state is not None else None
                ),
            }
        )
        return payload

    def load_state_dict(self, payload: object) -> None:
        expected = {
            "schema_version",
            "mode",
            "param_index_fingerprint_schema",
            "param_index_fingerprint",
            "last_processed_step",
            "exact_query_count",
            "ema_lambda",
            "refresh_k_steps",
            "last_refresh_step",
            "active_state",
        }
        loaded = self._load_common(payload, expected_keys=expected)
        if loaded["ema_lambda"] != self.ema_lambda:
            raise EstimatorStateError("Serialized EMA lambda differs")
        if loaded["refresh_k_steps"] != self.refresh_k_steps:
            raise EstimatorStateError("Serialized periodic K differs")
        last_step = loaded["last_processed_step"]
        last_refresh = loaded["last_refresh_step"]
        query_count = loaded["exact_query_count"]
        if last_step is None:
            if last_refresh is not None or query_count != 0 or loaded["active_state"] is not None:
                raise EstimatorStateError("Uninitialized periodic payload is inconsistent")
            state = None
        else:
            expected_refresh = (last_step // self.refresh_k_steps) * self.refresh_k_steps
            expected_queries = last_step // self.refresh_k_steps + 1
            if last_refresh != expected_refresh:
                raise EstimatorStateError("Periodic last refresh differs from modulo-K history")
            if query_count != expected_queries:
                raise EstimatorStateError("Periodic query counter differs from modulo-K history")
            if loaded["active_state"] is None:
                raise EstimatorStateError("Initialized periodic payload lacks active state")
            state = GradientState.from_state_dict(
                self.param_index, loaded["active_state"]
            )
            self._validate_gradient_state(state, label="serialized periodic state")

        self._state = state
        self._last_processed_step = last_step
        self._last_refresh_step = last_refresh
        self._exact_query_count = query_count


def _validate_ema_lambda(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EstimatorError("ema_lambda must be a real number")
    converted = float(value)
    if not 0.0 <= converted < 1.0:
        raise EstimatorError("ema_lambda must satisfy 0 <= lambda < 1")
    return converted
