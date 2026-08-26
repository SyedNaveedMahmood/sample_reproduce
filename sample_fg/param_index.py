"""Canonical indexing for the model's trainable parameter subspace."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch
from torch import nn


FINGERPRINT_SCHEMA = "sample_fg.param_index.name_shape.v1"


class ParamIndexError(ValueError):
    """Raised when a trainable-parameter index is ambiguous or invalid."""


class ParamIndexMismatchError(ParamIndexError):
    """Raised when two trainable-parameter structures do not correspond."""


@dataclass(frozen=True)
class ParamEntry:
    """Metadata and stable live reference for one canonical parameter."""

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype: torch.dtype
    device: torch.device
    parameter: nn.Parameter = field(repr=False, compare=False)

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "dtype": str(self.dtype),
            "device": str(self.device),
        }


class ParamIndex(Sequence[ParamEntry]):
    """One immutable ordering of all and only runtime-trainable parameters."""

    def __init__(self, entries: Sequence[ParamEntry]):
        self._entries = tuple(entries)
        if not self._entries:
            raise ParamIndexError("ParamIndex requires at least one trainable parameter")

        names: set[str] = set()
        identities: set[int] = set()
        for entry in self._entries:
            if not entry.name:
                raise ParamIndexError("Trainable parameter names must be nonempty")
            if entry.name in names:
                raise ParamIndexError(
                    f"Duplicate trainable parameter name: {entry.name!r}"
                )
            identity = id(entry.parameter)
            if identity in identities:
                raise ParamIndexError(
                    f"Trainable parameter has multiple canonical names: {entry.name!r}"
                )
            if not entry.parameter.requires_grad:
                raise ParamIndexError(
                    f"Indexed parameter is not trainable: {entry.name!r}"
                )
            if tuple(entry.parameter.shape) != entry.shape:
                raise ParamIndexError(
                    f"Stale shape metadata for {entry.name!r}: "
                    f"{entry.shape} != {tuple(entry.parameter.shape)}"
                )
            if entry.parameter.numel() != entry.numel:
                raise ParamIndexError(f"Stale numel metadata for {entry.name!r}")
            if entry.parameter.dtype != entry.dtype:
                raise ParamIndexError(f"Stale dtype metadata for {entry.name!r}")
            if entry.parameter.device != entry.device:
                raise ParamIndexError(f"Stale device metadata for {entry.name!r}")
            names.add(entry.name)
            identities.add(identity)

        self._total_numel = sum(entry.numel for entry in self._entries)
        self._fingerprint_payload = {
            "schema_version": FINGERPRINT_SCHEMA,
            "parameters": [
                {"name": entry.name, "shape": list(entry.shape)}
                for entry in self._entries
            ],
        }
        encoded = json.dumps(
            self._fingerprint_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._fingerprint = hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_model(cls, model: nn.Module) -> "ParamIndex":
        """Capture PyTorch registration traversal order exactly once."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        try:
            named_parameters = model.named_parameters(remove_duplicate=False)
        except TypeError as error:  # pragma: no cover - pinned PyTorch supports this.
            raise RuntimeError(
                "This PyTorch version cannot expose duplicate parameter aliases safely"
            ) from error

        entries = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            entries.append(
                ParamEntry(
                    name=name,
                    shape=tuple(parameter.shape),
                    numel=parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                    parameter=parameter,
                )
            )
        return cls(entries)

    @property
    def fingerprint_schema(self) -> str:
        return FINGERPRINT_SCHEMA

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "schema_version": self._fingerprint_payload["schema_version"],
            "parameters": [
                {"name": item["name"], "shape": list(item["shape"])}
                for item in self._fingerprint_payload["parameters"]
            ],
        }

    @property
    def total_numel(self) -> int:
        return self._total_numel

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self._entries)

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(entry.parameter for entry in self._entries)

    def to_metadata(self) -> dict[str, object]:
        return {
            "fingerprint_schema": self.fingerprint_schema,
            "fingerprint": self.fingerprint,
            "total_numel": self.total_numel,
            "parameters": [entry.to_metadata() for entry in self._entries],
        }

    def assert_compatible(self, other: "ParamIndex") -> None:
        if not isinstance(other, ParamIndex):
            raise ParamIndexMismatchError("Expected another ParamIndex")
        own_structure = tuple((entry.name, entry.shape) for entry in self)
        other_structure = tuple((entry.name, entry.shape) for entry in other)
        if self.fingerprint != other.fingerprint or own_structure != other_structure:
            raise ParamIndexMismatchError(
                "ParamIndex fingerprints or canonical name/shape sequences differ"
            )

    def assert_matches_model(self, model: nn.Module) -> None:
        """Explicitly detect a changed trainable set after index construction."""

        current = type(self).from_model(model)
        self.assert_compatible(current)
        for expected, observed in zip(self, current):
            if expected.parameter is not observed.parameter:
                raise ParamIndexMismatchError(
                    f"Live parameter identity changed for {expected.name!r}"
                )

    def __getitem__(self, index):
        return self._entries[index]

    def __iter__(self) -> Iterator[ParamEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return (
            f"ParamIndex(names={self.names!r}, total_numel={self.total_numel}, "
            f"fingerprint={self.fingerprint!r})"
        )
