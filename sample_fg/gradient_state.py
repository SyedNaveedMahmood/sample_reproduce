"""Detached FP32 vectors structured by a :class:`ParamIndex`."""

from __future__ import annotations

from numbers import Real
from typing import Iterable, Iterator, Sequence

import torch

from .param_index import ParamIndex, ParamIndexMismatchError


SERIALIZATION_SCHEMA = "sample_fg.gradient_state.v1"


class GradientStateError(ValueError):
    """Raised when a numerical gradient state is invalid or unsafe."""


class GradientStateMismatchError(GradientStateError):
    """Raised when states have incompatible structures or devices."""


class MissingGradientError(GradientStateError):
    """Raised when a required trainable parameter has no live gradient."""


def _validate_dense_tensor(component: object, name: str) -> torch.Tensor:
    if not isinstance(component, torch.Tensor):
        raise GradientStateError(f"Component {name!r} is not a tensor")
    if component.layout != torch.strided:
        raise GradientStateError(f"Sparse/non-strided component is unsupported: {name!r}")
    if not component.is_floating_point():
        raise GradientStateError(f"Component {name!r} must have a floating dtype")
    return component


class GradientState(Sequence[torch.Tensor]):
    """An owned, shape-preserving FP32 vector with no autograd history."""

    def __init__(self, param_index: ParamIndex, components: Iterable[torch.Tensor]):
        if not isinstance(param_index, ParamIndex):
            raise TypeError("param_index must be a ParamIndex")
        source = tuple(components)
        if len(source) != len(param_index):
            raise GradientStateError(
                f"Expected {len(param_index)} components, observed {len(source)}"
            )

        owned = []
        for entry, component in zip(param_index, source):
            tensor = _validate_dense_tensor(component, entry.name)
            if tuple(tensor.shape) != entry.shape:
                raise GradientStateError(
                    f"Wrong shape for {entry.name!r}: "
                    f"{tuple(tensor.shape)} != {entry.shape}"
                )
            owned.append(tensor.detach().to(dtype=torch.float32).clone())

        self._param_index = param_index
        self._components = tuple(owned)
        self.assert_valid()

    @classmethod
    def _from_owned(
        cls, param_index: ParamIndex, components: Iterable[torch.Tensor]
    ) -> "GradientState":
        """Internal no-copy path for fresh tensors created by state operations."""

        instance = cls.__new__(cls)
        instance._param_index = param_index
        instance._components = tuple(components)
        instance.assert_valid()
        return instance

    @classmethod
    def from_tensors(
        cls, param_index: ParamIndex, tensors: Iterable[torch.Tensor]
    ) -> "GradientState":
        return cls(param_index, tensors)

    @classmethod
    def from_parameter_grads(cls, param_index: ParamIndex) -> "GradientState":
        gradients = []
        for entry in param_index:
            gradient = entry.parameter.grad
            if gradient is None:
                raise MissingGradientError(
                    f"Required trainable gradient is missing: {entry.name!r}"
                )
            _validate_dense_tensor(gradient, entry.name)
            if tuple(gradient.shape) != entry.shape:
                raise GradientStateError(
                    f"Live gradient shape changed for {entry.name!r}: "
                    f"{tuple(gradient.shape)} != {entry.shape}"
                )
            gradients.append(gradient)
        return cls(param_index, gradients)

    @classmethod
    def zeros(cls, param_index: ParamIndex) -> "GradientState":
        components = (
            torch.zeros(
                entry.shape,
                dtype=torch.float32,
                device=entry.parameter.device,
            )
            for entry in param_index
        )
        return cls._from_owned(param_index, components)

    @classmethod
    def from_state_dict(
        cls, param_index: ParamIndex, payload: object
    ) -> "GradientState":
        """Restore canonical named tensors without relying on mapping order."""

        if not isinstance(payload, dict):
            raise GradientStateError("GradientState payload must be a dictionary")
        if payload.get("schema_version") != SERIALIZATION_SCHEMA:
            raise GradientStateError("Unsupported GradientState serialization schema")
        if payload.get("param_index_fingerprint_schema") != param_index.fingerprint_schema:
            raise GradientStateMismatchError("ParamIndex fingerprint schema differs")
        if payload.get("param_index_fingerprint") != param_index.fingerprint:
            raise GradientStateMismatchError("ParamIndex fingerprint differs")
        named = payload.get("components")
        if not isinstance(named, dict):
            raise GradientStateError("Serialized components must be a name/tensor mapping")
        if not all(isinstance(name, str) for name in named):
            raise GradientStateError("Serialized component names must be strings")
        expected_names = set(param_index.names)
        if set(named) != expected_names:
            missing = sorted(expected_names - set(named))
            unexpected = sorted(set(named) - expected_names)
            raise GradientStateMismatchError(
                f"Serialized component names differ; missing={missing}, "
                f"unexpected={unexpected}"
            )
        return cls.from_tensors(
            param_index, (named[entry.name] for entry in param_index)
        )

    @property
    def param_index(self) -> ParamIndex:
        return self._param_index

    @property
    def components(self) -> tuple[torch.Tensor, ...]:
        return self._components

    @property
    def total_numel(self) -> int:
        return sum(component.numel() for component in self._components)

    @property
    def raw_tensor_bytes(self) -> int:
        return sum(
            component.numel() * component.element_size()
            for component in self._components
        )

    @property
    def devices(self) -> tuple[torch.device, ...]:
        return tuple(component.device for component in self._components)

    def assert_valid(self) -> None:
        if len(self._components) != len(self._param_index):
            raise GradientStateError("GradientState component count changed")
        for entry, component in zip(self._param_index, self._components):
            _validate_dense_tensor(component, entry.name)
            if tuple(component.shape) != entry.shape:
                raise GradientStateError(
                    f"GradientState shape changed for {entry.name!r}"
                )
            if component.dtype != torch.float32:
                raise GradientStateError(
                    f"GradientState component is not FP32: {entry.name!r}"
                )
            if component.requires_grad or component.grad_fn is not None:
                raise GradientStateError(
                    f"GradientState retained autograd history: {entry.name!r}"
                )

    def assert_compatible(self, other: "GradientState") -> None:
        self.assert_valid()
        if not isinstance(other, GradientState):
            raise GradientStateMismatchError("Expected another GradientState")
        other.assert_valid()
        try:
            self.param_index.assert_compatible(other.param_index)
        except ParamIndexMismatchError as error:
            raise GradientStateMismatchError(str(error)) from error
        for entry, own, candidate in zip(
            self.param_index, self.components, other.components
        ):
            if own.device != candidate.device:
                raise GradientStateMismatchError(
                    f"Device mismatch for {entry.name!r}: "
                    f"{own.device} != {candidate.device}"
                )

    def clone(self) -> "GradientState":
        self.assert_valid()
        return type(self)._from_owned(
            self.param_index, (component.clone() for component in self.components)
        )

    def copy(self) -> "GradientState":
        return self.clone()

    def state_dict(self) -> dict[str, object]:
        """Return an owned, device-preserving payload keyed by canonical names."""

        self.assert_valid()
        return {
            "schema_version": SERIALIZATION_SCHEMA,
            "param_index_fingerprint_schema": self.param_index.fingerprint_schema,
            "param_index_fingerprint": self.param_index.fingerprint,
            "components": {
                entry.name: component.clone()
                for entry, component in zip(self.param_index, self.components)
            },
        }

    def add(self, other: "GradientState") -> "GradientState":
        self.assert_compatible(other)
        return type(self)._from_owned(
            self.param_index,
            (left + right for left, right in zip(self.components, other.components)),
        )

    def subtract(self, other: "GradientState") -> "GradientState":
        self.assert_compatible(other)
        return type(self)._from_owned(
            self.param_index,
            (left - right for left, right in zip(self.components, other.components)),
        )

    def scale(self, scalar: Real) -> "GradientState":
        self.assert_valid()
        if not isinstance(scalar, Real):
            raise TypeError("GradientState scalar must be a real number")
        factor = float(scalar)
        return type(self)._from_owned(
            self.param_index,
            (component * factor for component in self.components),
        )

    def add_(self, other: "GradientState", alpha: Real = 1.0) -> "GradientState":
        self.assert_compatible(other)
        if not isinstance(alpha, Real):
            raise TypeError("GradientState accumulation weight must be real")
        weight = float(alpha)
        with torch.no_grad():
            for target, source in zip(self.components, other.components):
                target.add_(source, alpha=weight)
        return self

    def accumulate_(
        self, other: "GradientState", weight: Real = 1.0
    ) -> "GradientState":
        return self.add_(other, alpha=weight)

    def affine_(
        self,
        other: "GradientState",
        self_weight: Real,
        other_weight: Real,
    ) -> "GradientState":
        """Apply a generic component-wise weighted sum in place."""

        self.assert_compatible(other)
        if not isinstance(self_weight, Real) or not isinstance(other_weight, Real):
            raise TypeError("GradientState affine weights must be real")
        own_factor = float(self_weight)
        other_factor = float(other_weight)
        with torch.no_grad():
            for target, source in zip(self.components, other.components):
                target.mul_(own_factor).add_(source, alpha=other_factor)
        return self

    def dot(self, other: "GradientState") -> torch.Tensor:
        self.assert_compatible(other)
        total = None
        for left, right in zip(self.components, other.components):
            term = torch.sum(left * right)
            if total is None:
                total = term
            elif total.device != term.device:
                raise GradientStateMismatchError(
                    "Dot product cannot reduce components across devices without "
                    "an explicit transfer"
                )
            else:
                total = total + term
        if total is None:  # ParamIndex disallows this, retained as a safety guard.
            raise GradientStateError("Cannot reduce an empty GradientState")
        return total

    def squared_norm(self) -> torch.Tensor:
        return self.dot(self)

    def norm(self) -> torch.Tensor:
        return torch.sqrt(self.squared_norm())

    def l2_norm(self) -> torch.Tensor:
        return self.norm()

    def is_finite(self) -> bool:
        self.assert_valid()
        return all(bool(torch.isfinite(component).all().item()) for component in self)

    def __add__(self, other: "GradientState") -> "GradientState":
        return self.add(other)

    def __sub__(self, other: "GradientState") -> "GradientState":
        return self.subtract(other)

    def __mul__(self, scalar: Real) -> "GradientState":
        return self.scale(scalar)

    def __rmul__(self, scalar: Real) -> "GradientState":
        return self.scale(scalar)

    def __getitem__(self, index):
        return self._components[index]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def __repr__(self) -> str:
        return (
            f"GradientState(components={len(self)}, total_numel={self.total_numel}, "
            f"devices={tuple(str(device) for device in self.devices)!r})"
        )
