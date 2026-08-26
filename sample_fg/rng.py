"""Deterministic, exception-safe RNG isolation for future auxiliary sweeps."""

from __future__ import annotations

import copy
import hashlib
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import torch


RNG_SEED_SCHEMA_VERSION = "sample_fg.fullgrad_rng.v1"
_SEED_DELIMITER = "|"
_UINT32_MODULUS = 1 << 32


class RNGError(ValueError):
    """Raised for invalid seed metadata or generator configuration."""


class RNGIsolationError(RuntimeError):
    """Raised when an RNG snapshot cannot be restored exactly."""


@dataclass(frozen=True)
class DerivedSeed:
    """Stable auxiliary-seed derivation and per-library reductions."""

    schema_version: str
    protocol_seed: int
    dataset: str
    shots: int
    config_hash: str
    optimizer_step: int
    purpose: str
    canonical_preimage: str
    sha256: str
    raw_uint64: int
    python_seed: int
    numpy_seed: int
    torch_seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_seed": self.protocol_seed,
            "dataset": self.dataset,
            "shots": self.shots,
            "config_hash": self.config_hash,
            "optimizer_step": self.optimizer_step,
            "purpose": self.purpose,
            "canonical_preimage": self.canonical_preimage,
            "sha256": self.sha256,
            "raw_uint64": self.raw_uint64,
            "python_seed": self.python_seed,
            "numpy_seed": self.numpy_seed,
            "torch_seed": self.torch_seed,
        }


@dataclass(frozen=True)
class GeneratorSnapshot:
    """Owned state for one caller-owned explicit torch.Generator."""

    generator: torch.Generator
    device: str
    state: torch.Tensor


@dataclass(frozen=True)
class RNGSnapshot:
    """Owned global and explicit-generator states for exact restoration."""

    python_state: object
    numpy_state: tuple[object, ...]
    torch_cpu_state: torch.Tensor
    cuda_was_initialized: bool
    torch_cuda_states: tuple[torch.Tensor, ...]
    explicit_generators: tuple[GeneratorSnapshot, ...]


def derive_auxiliary_seed(
    *,
    protocol_seed: int,
    dataset: str,
    shots: int,
    config_hash: str,
    optimizer_step: int,
    purpose: str,
) -> DerivedSeed:
    """Derive a portable auxiliary seed from the normative project fields."""

    _validate_integer("protocol_seed", protocol_seed, minimum=0)
    _validate_integer("shots", shots, minimum=1)
    _validate_integer("optimizer_step", optimizer_step, minimum=0)
    string_fields = {
        "dataset": dataset,
        "config_hash": config_hash,
        "purpose": purpose,
    }
    for name, value in string_fields.items():
        _validate_seed_string(name, value)

    canonical_preimage = _SEED_DELIMITER.join(
        (
            RNG_SEED_SCHEMA_VERSION,
            str(protocol_seed),
            dataset,
            str(shots),
            config_hash,
            str(optimizer_step),
            purpose,
        )
    )
    digest = hashlib.sha256(canonical_preimage.encode("utf-8")).digest()
    raw_uint64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return DerivedSeed(
        schema_version=RNG_SEED_SCHEMA_VERSION,
        protocol_seed=protocol_seed,
        dataset=dataset,
        shots=shots,
        config_hash=config_hash,
        optimizer_step=optimizer_step,
        purpose=purpose,
        canonical_preimage=canonical_preimage,
        sha256=digest.hex(),
        raw_uint64=raw_uint64,
        python_seed=raw_uint64,
        numpy_seed=raw_uint64 % _UINT32_MODULUS,
        torch_seed=raw_uint64,
    )


def capture_rng_state(
    explicit_generators: Iterable[torch.Generator] = (),
) -> RNGSnapshot:
    """Own all restorable parent-process RNG state without reseeding it."""

    generators = _normalize_generators(explicit_generators)
    python_state = copy.deepcopy(random.getstate())
    numpy_state = np.random.get_state()
    owned_numpy_state = (
        numpy_state[0],
        numpy_state[1].copy(),
        int(numpy_state[2]),
        int(numpy_state[3]),
        float(numpy_state[4]),
    )
    torch_cpu_state = torch.get_rng_state().clone()

    cuda_was_initialized = bool(
        torch.cuda.is_available() and torch.cuda.is_initialized()
    )
    cuda_states = ()
    if cuda_was_initialized:
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())

    generator_states = tuple(
        GeneratorSnapshot(
            generator=generator,
            device=str(generator.device),
            state=generator.get_state().clone(),
        )
        for generator in generators
    )
    return RNGSnapshot(
        python_state=python_state,
        numpy_state=owned_numpy_state,
        torch_cpu_state=torch_cpu_state,
        cuda_was_initialized=cuda_was_initialized,
        torch_cuda_states=cuda_states,
        explicit_generators=generator_states,
    )


def restore_rng_state(snapshot: RNGSnapshot) -> None:
    """Restore a snapshot in the documented generator-to-global order."""

    if not isinstance(snapshot, RNGSnapshot):
        raise RNGError("snapshot must be an RNGSnapshot")

    for item in snapshot.explicit_generators:
        if str(item.generator.device) != item.device:
            raise RNGIsolationError(
                "An explicit generator's device changed after snapshot capture"
            )
        item.generator.set_state(item.state.clone())

    if snapshot.cuda_was_initialized:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            raise RNGIsolationError("Captured CUDA RNG state is no longer available")
        if len(snapshot.torch_cuda_states) != torch.cuda.device_count():
            raise RNGIsolationError("Visible CUDA device count changed after capture")
        torch.cuda.set_rng_state_all(
            [state.clone() for state in snapshot.torch_cuda_states]
        )
    elif torch.cuda.is_available() and torch.cuda.is_initialized():
        raise RNGIsolationError(
            "CUDA became initialized inside a context that had no CUDA snapshot"
        )

    torch.set_rng_state(snapshot.torch_cpu_state.clone())
    np.random.set_state(_clone_numpy_state(snapshot.numpy_state))
    random.setstate(copy.deepcopy(snapshot.python_state))


@contextmanager
def isolated_rng(
    *,
    protocol_seed: int,
    dataset: str,
    shots: int,
    config_hash: str,
    optimizer_step: int,
    purpose: str,
    explicit_generators: Iterable[torch.Generator] = (),
) -> Iterator[DerivedSeed]:
    """Run stochastic auxiliary work without advancing the caller's streams."""

    generators = _normalize_generators(explicit_generators)
    snapshot = capture_rng_state(generators)
    try:
        derived = derive_auxiliary_seed(
            protocol_seed=protocol_seed,
            dataset=dataset,
            shots=shots,
            config_hash=config_hash,
            optimizer_step=optimizer_step,
            purpose=purpose,
        )
        _seed_isolated_domains(derived, snapshot)
        yield derived
    finally:
        restore_rng_state(snapshot)


def _seed_isolated_domains(seed: DerivedSeed, snapshot: RNGSnapshot) -> None:
    random.seed(seed.python_seed)
    np.random.seed(seed.numpy_seed)

    # Seed CPU without torch.manual_seed(), whose implementation also queues a
    # CUDA seed and could mutate an uninitialized CUDA domain we did not capture.
    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed.torch_seed)
    torch.set_rng_state(cpu_generator.get_state())

    if snapshot.cuda_was_initialized:
        torch.cuda.manual_seed_all(seed.torch_seed)
    for item in snapshot.explicit_generators:
        item.generator.manual_seed(seed.torch_seed)


def _normalize_generators(
    explicit_generators: Iterable[torch.Generator],
) -> tuple[torch.Generator, ...]:
    try:
        generators = tuple(explicit_generators)
    except TypeError as error:
        raise RNGError("explicit_generators must be an iterable") from error
    if any(not isinstance(generator, torch.Generator) for generator in generators):
        raise RNGError("Every explicit generator must be a torch.Generator")
    identities = [id(generator) for generator in generators]
    if len(identities) != len(set(identities)):
        raise RNGError("An explicit generator was supplied more than once")
    return generators


def _clone_numpy_state(state: tuple[object, ...]) -> tuple[object, ...]:
    if len(state) != 5 or not isinstance(state[1], np.ndarray):
        raise RNGIsolationError("Malformed NumPy RNG snapshot")
    return (
        state[0],
        state[1].copy(),
        int(state[2]),
        int(state[3]),
        float(state[4]),
    )


def _validate_integer(name: str, value: object, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RNGError(f"{name} must be an integer >= {minimum}")


def _validate_seed_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise RNGError(f"{name} must be a nonempty string")
    if _SEED_DELIMITER in value or "\n" in value or "\r" in value:
        raise RNGError(
            f"{name} contains a reserved seed-preimage delimiter/control character"
        )
