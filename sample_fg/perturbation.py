"""Exception-safe snapshot, displacement, and restoration of prompt parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, Sequence

import torch

from .gradient_state import GradientState
from .param_index import ParamIndex, ParamIndexMismatchError


class PerturbationError(ValueError):
    """Raised when a snapshot or displacement is structurally unsafe."""


class PerturbationStateError(RuntimeError):
    """Raised when temporary parameter mutation is requested out of sequence."""


class PerturbationNumericalError(FloatingPointError):
    """Raised when a displacement is nonfinite or overflows live storage."""


class ParameterSnapshot(Sequence[torch.Tensor]):
    """Owned exact-dtype values for one authoritative :class:`ParamIndex`."""

    def __init__(
        self,
        param_index: ParamIndex,
        values: Iterable[torch.Tensor],
    ) -> None:
        if not isinstance(param_index, ParamIndex):
            raise TypeError("param_index must be a ParamIndex")
        source = tuple(values)
        if len(source) != len(param_index):
            raise PerturbationError(
                f"Expected {len(param_index)} snapshot values, observed {len(source)}"
            )
        owned: list[torch.Tensor] = []
        for entry, value in zip(param_index, source):
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise PerturbationError(
                    f"Snapshot value for {entry.name!r} must be a dense tensor"
                )
            if tuple(value.shape) != entry.shape:
                raise PerturbationError(
                    f"Snapshot shape differs for {entry.name!r}: "
                    f"{tuple(value.shape)} != {entry.shape}"
                )
            if value.dtype != entry.parameter.dtype:
                raise PerturbationError(
                    f"Snapshot dtype differs for {entry.name!r}: "
                    f"{value.dtype} != {entry.parameter.dtype}"
                )
            if value.device != entry.parameter.device:
                raise PerturbationError(
                    f"Snapshot device differs for {entry.name!r}: "
                    f"{value.device} != {entry.parameter.device}"
                )
            owned.append(value.detach().clone())
        self._param_index = param_index
        self._values = tuple(owned)

    @classmethod
    def capture(cls, param_index: ParamIndex) -> "ParameterSnapshot":
        """Copy the live trainable values without changing object identity."""

        return cls(param_index, (entry.parameter.detach() for entry in param_index))

    @property
    def param_index(self) -> ParamIndex:
        return self._param_index

    @property
    def values(self) -> tuple[torch.Tensor, ...]:
        return self._values

    def clone(self) -> "ParameterSnapshot":
        return type(self)(self.param_index, self.values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class PromptPerturbation:
    """Own one guarded temporary-mutation lifecycle for a ``ParamIndex``."""

    def __init__(self, param_index: ParamIndex) -> None:
        if not isinstance(param_index, ParamIndex):
            raise TypeError("param_index must be a ParamIndex")
        self.param_index = param_index
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    def assert_inactive(self) -> None:
        if self._active:
            raise PerturbationStateError("A prompt perturbation is already active")

    def snapshot(self) -> ParameterSnapshot:
        self.assert_inactive()
        return ParameterSnapshot.capture(self.param_index)

    def _validate_displacement(self, displacement: GradientState) -> None:
        if not isinstance(displacement, GradientState):
            raise PerturbationError("displacement must be a GradientState")
        try:
            self.param_index.assert_compatible(displacement.param_index)
        except ParamIndexMismatchError as error:
            raise PerturbationError("Displacement ParamIndex differs") from error
        displacement.assert_valid()
        if not displacement.is_finite():
            raise PerturbationNumericalError("Displacement contains NaN or Inf")
        for entry, component in zip(self.param_index, displacement):
            if component.device != entry.parameter.device:
                raise PerturbationError(
                    f"Displacement device differs for {entry.name!r}: "
                    f"{component.device} != {entry.parameter.device}"
                )

    def _displaced_values(
        self,
        snapshot: ParameterSnapshot,
        displacement: GradientState,
    ) -> tuple[torch.Tensor, ...]:
        values: list[torch.Tensor] = []
        for entry, original, component in zip(
            self.param_index, snapshot, displacement
        ):
            converted = component.to(dtype=entry.parameter.dtype)
            candidate = original + converted
            if not bool(torch.isfinite(candidate).all().item()):
                raise PerturbationNumericalError(
                    f"Displacement becomes nonfinite in live dtype: {entry.name!r}"
                )
            values.append(candidate)
        return tuple(values)

    def _restore(self, snapshot: ParameterSnapshot) -> None:
        if snapshot.param_index is not self.param_index:
            raise PerturbationError(
                "Snapshot does not belong to this authoritative ParamIndex"
            )
        with torch.no_grad():
            for entry, value in zip(self.param_index, snapshot):
                entry.parameter.copy_(value)

    @contextmanager
    def displaced(
        self, displacement: GradientState
    ) -> Iterator[ParameterSnapshot]:
        """Apply ``snapshot + displacement`` and always restore the snapshot.

        A single controller cannot be nested while active. It can be reused
        after the prior context has restored successfully.
        """

        self.assert_inactive()
        self._validate_displacement(displacement)
        snapshot = ParameterSnapshot.capture(self.param_index)
        displaced_values = self._displaced_values(snapshot, displacement)
        self._active = True
        try:
            with torch.no_grad():
                for entry, value in zip(self.param_index, displaced_values):
                    entry.parameter.copy_(value)
            yield snapshot
        finally:
            # Copy the original values back. Never try to invert the mutation
            # arithmetically: body code may have changed parameters and mixed
            # precision addition need not be exactly reversible.
            self._restore(snapshot)
            self._active = False


@contextmanager
def temporary_prompt_perturbation(
    param_index: ParamIndex,
    displacement: GradientState,
) -> Iterator[ParameterSnapshot]:
    """One-shot convenience wrapper around :class:`PromptPerturbation`."""

    controller = PromptPerturbation(param_index)
    with controller.displaced(displacement) as snapshot:
        yield snapshot
