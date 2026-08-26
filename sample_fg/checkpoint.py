"""Atomic estimator-aware scientific checkpoints at logical step boundaries."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch

from .estimators import GlobalGradientEstimator
from .param_index import ParamIndex
from .perturbation import PromptPerturbation
from .precision import PrecisionController
from .rng import GeneratorSnapshot, RNGSnapshot, capture_rng_state, restore_rng_state
from .step_engine import StepEngine


CHECKPOINT_SCHEMA_VERSION = "sample_fg.scientific_checkpoint.v1"
CHECKPOINT_METADATA_SCHEMA_VERSION = "sample_fg.checkpoint_metadata.v1"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint boundary or payload is unsafe."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised before scientifically incompatible state can be restored."""


@dataclass(frozen=True)
class CheckpointProgress:
    """Unambiguous next-step and worker-0 loader position."""

    next_optimizer_step: int
    epoch_zero_based: int
    next_batch_index_zero_based: int
    normal_samples_seen: int

    def __post_init__(self) -> None:
        for name, value in (
            ("next_optimizer_step", self.next_optimizer_step),
            ("epoch_zero_based", self.epoch_zero_based),
            ("next_batch_index_zero_based", self.next_batch_index_zero_based),
            ("normal_samples_seen", self.normal_samples_seen),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CheckpointError(f"{name} must be a nonnegative integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "next_optimizer_step": self.next_optimizer_step,
            "epoch_zero_based": self.epoch_zero_based,
            "next_batch_index_zero_based": self.next_batch_index_zero_based,
            "normal_samples_seen": self.normal_samples_seen,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "CheckpointProgress":
        if not isinstance(payload, dict) or set(payload) != {
            "next_optimizer_step",
            "epoch_zero_based",
            "next_batch_index_zero_based",
            "normal_samples_seen",
        }:
            raise CheckpointError("Checkpoint progress fields differ from schema")
        return cls(**payload)


@dataclass(frozen=True)
class CheckpointMetadata:
    path: Path
    byte_size: int
    sha256: str
    next_optimizer_step: int
    epoch_zero_based: int
    next_batch_index_zero_based: int
    method: str
    estimator_mode: str
    schema_version: str = CHECKPOINT_METADATA_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path": str(self.path),
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "next_optimizer_step": self.next_optimizer_step,
            "epoch_zero_based": self.epoch_zero_based,
            "next_batch_index_zero_based": self.next_batch_index_zero_based,
            "method": self.method,
            "estimator_mode": self.estimator_mode,
        }


@dataclass(frozen=True)
class NormalLoaderReplayState:
    """Worker-0 epoch replay information; no worker-process state is claimed."""

    batches_consumed: int
    loader_length: int
    epoch_start_rng_state: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "workers": 0,
            "batches_consumed": self.batches_consumed,
            "loader_length": self.loader_length,
            "epoch_start_rng_state": copy.deepcopy(self.epoch_start_rng_state),
            "resume_policy": "restore_epoch_start_rng_then_replay_consumed_batches_v1",
        }


@dataclass(frozen=True)
class CheckpointLoadResult:
    metadata: CheckpointMetadata
    progress: CheckpointProgress
    result_state: dict[str, Any]
    normal_loader_state: NormalLoaderReplayState | None

    def epoch_start_rng_snapshot(
        self,
        *,
        explicit_generators: Mapping[str, torch.Generator] | None = None,
    ) -> RNGSnapshot:
        """Bind serialized epoch-start state to reconstructed generators."""

        if self.normal_loader_state is None:
            raise CheckpointError("Checkpoint has no worker-0 loader replay state")
        generators = _normalize_named_generators(explicit_generators)
        return _rng_snapshot_from_payload(
            self.normal_loader_state.epoch_start_rng_state, generators
        )

    def resume_worker0_loader(
        self,
        loader,
        *,
        explicit_generators: Mapping[str, torch.Generator] | None = None,
    ) -> Iterator[Any]:
        """Rebuild the current iterator and prove its replay reaches saved RNG."""

        state = self.normal_loader_state
        if state is None:
            raise CheckpointError("Checkpoint has no worker-0 loader replay state")
        if getattr(loader, "num_workers", None) != 0:
            raise CheckpointError("Exact loader replay is supported only for workers=0")
        if len(loader) != state.loader_length:
            raise CheckpointCompatibilityError("Normal loader length differs on resume")
        generators = _normalize_named_generators(explicit_generators)
        expected = capture_rng_state(generators.values())
        try:
            _restore_rng_payload(state.epoch_start_rng_state, generators)
            iterator = iter(loader)
            for _ in range(state.batches_consumed):
                try:
                    next(iterator)
                except StopIteration as error:
                    raise CheckpointCompatibilityError(
                        "Normal loader ended before saved replay position"
                    ) from error
            observed = capture_rng_state(generators.values())
            if not _rng_snapshots_equal(observed, expected):
                raise CheckpointCompatibilityError(
                    "Worker-0 loader replay did not reproduce saved normal RNG state"
                )
            return iterator
        except Exception:
            restore_rng_state(expected)
            raise


def save_scientific_checkpoint(
    path: Path,
    *,
    param_index: ParamIndex,
    optimizer: torch.optim.Optimizer,
    scheduler,
    precision_controller: PrecisionController,
    step_engine: StepEngine | None,
    estimator: GlobalGradientEstimator | None,
    perturbation: PromptPerturbation | None,
    progress: CheckpointProgress,
    method: str,
    config_sha256: str,
    source_fingerprint: str,
    result_state: Mapping[str, Any],
    explicit_generators: Mapping[str, torch.Generator] | None = None,
    normal_loader_epoch_start_rng: RNGSnapshot | None = None,
    normal_loader_length: int | None = None,
) -> CheckpointMetadata:
    """Write one atomic checkpoint at an unperturbed completed-step boundary."""

    _validate_runtime_objects(
        param_index, optimizer, precision_controller, step_engine, estimator, perturbation
    )
    _validate_method_runtime(method, step_engine, estimator, perturbation)
    if perturbation is not None:
        perturbation.assert_inactive()
    if precision_controller.phase not in {"idle", "stepped"}:
        raise CheckpointError(
            "Checkpoint is allowed only before a cycle or after optimizer transition"
        )
    if (
        step_engine is not None
        and progress.next_optimizer_step != step_engine.optimizer_step
    ):
        raise CheckpointError("Progress and step-engine next optimizer steps differ")
    expected_last = None if progress.next_optimizer_step == 0 else progress.next_optimizer_step - 1
    if estimator is not None and estimator.last_processed_step != expected_last:
        raise CheckpointError("Estimator and optimizer-step boundary are inconsistent")
    _validate_identity(method, config_sha256, source_fingerprint)
    generators = _normalize_named_generators(explicit_generators)
    current_rng = capture_rng_state(generators.values())
    loader_payload = None
    if normal_loader_epoch_start_rng is not None:
        if normal_loader_length is None or normal_loader_length < 1:
            raise CheckpointError("Normal loader length is required for epoch replay")
        if progress.next_batch_index_zero_based > normal_loader_length:
            raise CheckpointError("Saved batch position exceeds normal loader length")
        loader_payload = NormalLoaderReplayState(
            batches_consumed=progress.next_batch_index_zero_based,
            loader_length=normal_loader_length,
            epoch_start_rng_state=_serialize_rng_snapshot(
                normal_loader_epoch_start_rng, generators
            ),
        ).as_dict()
    elif normal_loader_length is not None:
        raise CheckpointError("Normal loader length supplied without epoch RNG state")

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "boundary": "after_logical_optimizer_step_unperturbed_v1",
        "method": method,
        "estimator_mode": _estimator_mode(estimator),
        "config_sha256": config_sha256,
        "source_fingerprint": source_fingerprint,
        "param_index": param_index.to_metadata(),
        "trainable_model_state": {
            entry.name: entry.parameter.detach().cpu().clone() for entry in param_index
        },
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": copy.deepcopy(scheduler.state_dict()),
        "precision_state": copy.deepcopy(precision_controller.state_dict()),
        "step_engine_state": (
            copy.deepcopy(step_engine.state_dict())
            if step_engine is not None
            else None
        ),
        "estimator_state": (
            copy.deepcopy(estimator.state_dict())
            if estimator is not None
            else None
        ),
        "progress": progress.as_dict(),
        "rng_state": _serialize_rng_snapshot(current_rng, generators),
        "normal_loader_state": loader_payload,
        "result_state": _owned_json(result_state),
        "gradient_buffer_policy": "not_serialized_safe_step_boundary",
        "legacy_compatibility": {
            "upstream_task3_checkpoint_loadable_as_scientific_resume": False,
            "reason": "legacy checkpoint lacks estimator/RNG/config/source state",
        },
    }
    destination = Path(path).resolve()
    _atomic_torch_save(destination, payload)
    return _metadata_for(
        destination,
        progress=progress,
        method=method,
        estimator_mode=_estimator_mode(estimator),
    )


def load_scientific_checkpoint(
    path: Path,
    *,
    param_index: ParamIndex,
    optimizer: torch.optim.Optimizer,
    scheduler,
    precision_controller: PrecisionController,
    step_engine: StepEngine | None,
    estimator: GlobalGradientEstimator | None,
    perturbation: PromptPerturbation | None,
    expected_method: str,
    expected_config_sha256: str,
    expected_source_fingerprint: str,
    explicit_generators: Mapping[str, torch.Generator] | None = None,
) -> CheckpointLoadResult:
    """Validate first, then transactionally restore a fresh runtime."""

    _validate_runtime_objects(
        param_index, optimizer, precision_controller, step_engine, estimator, perturbation
    )
    if perturbation is not None:
        perturbation.assert_inactive()
    if precision_controller.phase != "idle" or (
        step_engine is not None and step_engine.optimizer_step != 0
    ):
        raise CheckpointError("Scientific resume requires a fresh idle runtime")
    if estimator is not None and estimator.last_processed_step is not None:
        raise CheckpointError("Scientific resume requires a fresh estimator")
    _validate_identity(expected_method, expected_config_sha256, expected_source_fingerprint)
    source_path = Path(path).resolve(strict=True)
    try:
        payload = torch.load(source_path, map_location=None, weights_only=False)
    except Exception as error:
        raise CheckpointError(f"Checkpoint could not be loaded: {source_path}") from error
    progress, normal_loader = _validate_checkpoint_payload(
        payload,
        param_index=param_index,
        precision_controller=precision_controller,
        step_engine=step_engine,
        estimator=estimator,
        expected_method=expected_method,
        expected_config_sha256=expected_config_sha256,
        expected_source_fingerprint=expected_source_fingerprint,
    )
    _validate_method_runtime(expected_method, step_engine, estimator, perturbation)
    generators = _normalize_named_generators(explicit_generators)
    _validate_rng_payload(payload["rng_state"], generators)
    if normal_loader is not None:
        _validate_rng_payload(normal_loader.epoch_start_rng_state, generators)

    parameter_before = tuple(entry.parameter.detach().clone() for entry in param_index)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    scheduler_before = copy.deepcopy(scheduler.state_dict())
    precision_before = copy.deepcopy(precision_controller.state_dict())
    estimator_before = (
        copy.deepcopy(estimator.state_dict()) if estimator is not None else None
    )
    rng_before = capture_rng_state(generators.values())
    try:
        _restore_rng_payload(payload["rng_state"], generators)
        with torch.no_grad():
            for entry in param_index:
                entry.parameter.copy_(
                    payload["trainable_model_state"][entry.name].to(entry.parameter.device)
                )
                entry.parameter.grad = None
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
        scheduler.load_state_dict(copy.deepcopy(payload["scheduler_state"]))
        precision_controller.load_state_dict(copy.deepcopy(payload["precision_state"]))
        if estimator is not None:
            estimator.load_state_dict(copy.deepcopy(payload["estimator_state"]))
        if step_engine is not None:
            step_engine.load_state_dict(copy.deepcopy(payload["step_engine_state"]))
    except Exception:
        restore_rng_state(rng_before)
        with torch.no_grad():
            for entry, value in zip(param_index, parameter_before):
                entry.parameter.copy_(value)
                entry.parameter.grad = None
        optimizer.load_state_dict(optimizer_before)
        scheduler.load_state_dict(scheduler_before)
        precision_controller.load_state_dict(precision_before)
        if estimator is not None:
            estimator.load_state_dict(estimator_before)
        raise

    metadata = _metadata_for(
        source_path,
        progress=progress,
        method=expected_method,
        estimator_mode=_estimator_mode(estimator),
    )
    return CheckpointLoadResult(
        metadata=metadata,
        progress=progress,
        result_state=copy.deepcopy(payload["result_state"]),
        normal_loader_state=normal_loader,
    )


def _validate_runtime_objects(
    param_index,
    optimizer,
    precision_controller,
    step_engine,
    estimator,
    perturbation,
) -> None:
    if not isinstance(param_index, ParamIndex):
        raise TypeError("param_index must be a ParamIndex")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(precision_controller, PrecisionController):
        raise TypeError("precision_controller must be a PrecisionController")
    if step_engine is not None and not isinstance(step_engine, StepEngine):
        raise TypeError("step_engine must be a StepEngine or None")
    if estimator is not None and not isinstance(estimator, GlobalGradientEstimator):
        raise TypeError("estimator must be a GlobalGradientEstimator or None")
    if perturbation is not None and not isinstance(perturbation, PromptPerturbation):
        raise TypeError("perturbation must be a PromptPerturbation or None")
    owned_indices = [
        value.param_index
        for value in (step_engine, estimator, perturbation)
        if value is not None
    ]
    for owned in owned_indices:
        param_index.assert_compatible(owned)
    if step_engine is not None and (
        step_engine.optimizer is not optimizer
        or step_engine.precision is not precision_controller
    ):
        raise CheckpointError("Step engine does not own the supplied optimizer/precision")
    if step_engine is not None and step_engine.perturbation is not perturbation:
        raise CheckpointError("Step engine does not own the supplied perturbation controller")


def _validate_method_runtime(
    method: str,
    step_engine: StepEngine | None,
    estimator: GlobalGradientEstimator | None,
    perturbation: PromptPerturbation | None,
) -> None:
    """Require only the algorithm state that the selected method actually owns."""

    if method == "coop":
        expected = (False, False, False)
    elif method == "sam":
        expected = (True, False, True)
    elif method == "sample":
        expected = (True, True, True)
    else:
        raise CheckpointError(f"Unsupported scientific checkpoint method: {method!r}")
    observed = (
        step_engine is not None,
        estimator is not None,
        perturbation is not None,
    )
    if observed != expected:
        raise CheckpointError(
            "Checkpoint runtime objects do not match the selected method: "
            f"method={method!r}, engine/estimator/perturbation={observed}"
        )


def _estimator_mode(estimator: GlobalGradientEstimator | None) -> str:
    return estimator.mode if estimator is not None else "none"


def _validate_identity(method: str, config_sha256: str, source_fingerprint: str) -> None:
    if not isinstance(method, str) or not method:
        raise CheckpointError("method must be a nonempty string")
    if not isinstance(config_sha256, str) or not config_sha256:
        raise CheckpointError("config_sha256 must be a nonempty stable string")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise CheckpointError("source_fingerprint must be a nonempty stable string")


def _validate_checkpoint_payload(
    payload: object,
    *,
    param_index: ParamIndex,
    precision_controller: PrecisionController,
    step_engine: StepEngine | None,
    estimator: GlobalGradientEstimator | None,
    expected_method: str,
    expected_config_sha256: str,
    expected_source_fingerprint: str,
) -> tuple[CheckpointProgress, NormalLoaderReplayState | None]:
    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a dictionary")
    expected_keys = {
        "schema_version", "created_utc", "boundary", "method", "estimator_mode",
        "config_sha256", "source_fingerprint", "param_index", "trainable_model_state",
        "optimizer_state", "scheduler_state", "precision_state", "step_engine_state",
        "estimator_state", "progress", "rng_state", "normal_loader_state", "result_state",
        "gradient_buffer_policy", "legacy_compatibility",
    }
    if set(payload) != expected_keys:
        if "schema_version" not in payload:
            raise CheckpointCompatibilityError(
                "Legacy/upstream checkpoint lacks mandatory scientific resume state"
            )
        raise CheckpointError("Checkpoint fields differ from schema")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError("Unsupported scientific checkpoint schema")
    if payload["boundary"] != "after_logical_optimizer_step_unperturbed_v1":
        raise CheckpointCompatibilityError("Checkpoint boundary is unsupported")
    for observed, expected, label in (
        (payload["method"], expected_method, "method"),
        (payload["estimator_mode"], _estimator_mode(estimator), "estimator mode"),
        (payload["config_sha256"], expected_config_sha256, "config hash"),
        (payload["source_fingerprint"], expected_source_fingerprint, "source fingerprint"),
    ):
        if observed != expected:
            raise CheckpointCompatibilityError(f"Checkpoint {label} differs")
    index_payload = payload["param_index"]
    if not isinstance(index_payload, dict):
        raise CheckpointError("Checkpoint ParamIndex metadata is malformed")
    if (
        index_payload.get("fingerprint_schema") != param_index.fingerprint_schema
        or index_payload.get("fingerprint") != param_index.fingerprint
    ):
        raise CheckpointCompatibilityError("Checkpoint ParamIndex fingerprint differs")
    expected_names = set(param_index.names)
    model_state = payload["trainable_model_state"]
    if not isinstance(model_state, dict) or set(model_state) != expected_names:
        raise CheckpointCompatibilityError("Checkpoint trainable parameter names differ")
    for entry in param_index:
        value = model_state[entry.name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != entry.shape:
            raise CheckpointCompatibilityError(
                f"Checkpoint trainable tensor differs for {entry.name!r}"
            )
        if value.dtype != entry.parameter.dtype or not bool(torch.isfinite(value).all().item()):
            raise CheckpointCompatibilityError(
                f"Checkpoint trainable dtype/nonfinite state differs for {entry.name!r}"
            )
    for key in ("optimizer_state", "scheduler_state", "precision_state"):
        if not isinstance(payload[key], dict):
            raise CheckpointError(f"Checkpoint {key} is malformed")
    if payload["precision_state"].get("mode") != precision_controller.mode:
        raise CheckpointCompatibilityError("Checkpoint precision mode differs")
    engine_state = payload["step_engine_state"]
    if step_engine is None:
        if engine_state is not None:
            raise CheckpointCompatibilityError(
                "Checkpoint unexpectedly contains step-engine state"
            )
    else:
        if not isinstance(engine_state, dict):
            raise CheckpointError("Checkpoint step_engine_state is malformed")
        if (
            engine_state.get("rho") != step_engine.rho
            or engine_state.get("alpha") != step_engine.alpha
            or engine_state.get("norm_eps") != step_engine.norm_eps
        ):
            raise CheckpointCompatibilityError("Checkpoint step-engine contract differs")
    estimator_state = payload["estimator_state"]
    if estimator is None:
        if estimator_state is not None:
            raise CheckpointCompatibilityError(
                "Checkpoint unexpectedly contains estimator state"
            )
    else:
        if not isinstance(estimator_state, dict):
            raise CheckpointError("Checkpoint estimator_state is malformed")
        if estimator_state.get("mode") != estimator.mode:
            raise CheckpointCompatibilityError("Checkpoint estimator state mode differs")
        if hasattr(estimator, "ema_lambda") and estimator_state.get("ema_lambda") != estimator.ema_lambda:
            raise CheckpointCompatibilityError("Checkpoint estimator EMA lambda differs")
        if (
            hasattr(estimator, "refresh_k_steps")
            and estimator_state.get("refresh_k_steps") != estimator.refresh_k_steps
        ):
            raise CheckpointCompatibilityError("Checkpoint periodic K differs")
    progress = CheckpointProgress.from_dict(payload["progress"])
    if (
        engine_state is not None
        and engine_state.get("optimizer_step") != progress.next_optimizer_step
    ):
        raise CheckpointError("Checkpoint step clock and progress differ")
    expected_last = None if progress.next_optimizer_step == 0 else progress.next_optimizer_step - 1
    if (
        estimator_state is not None
        and estimator_state.get("last_processed_step") != expected_last
    ):
        raise CheckpointError("Checkpoint estimator and progress clocks differ")
    _owned_json(payload["result_state"])
    normal_payload = payload["normal_loader_state"]
    normal_state = None
    if normal_payload is not None:
        if not isinstance(normal_payload, dict) or set(normal_payload) != {
            "workers", "batches_consumed", "loader_length", "epoch_start_rng_state", "resume_policy"
        }:
            raise CheckpointError("Normal loader replay state differs from schema")
        if normal_payload["workers"] != 0 or normal_payload["resume_policy"] != "restore_epoch_start_rng_then_replay_consumed_batches_v1":
            raise CheckpointCompatibilityError("Normal loader resume policy differs")
        consumed = normal_payload["batches_consumed"]
        length = normal_payload["loader_length"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (consumed, length)) or consumed < 0 or length < 1 or consumed > length:
            raise CheckpointError("Normal loader replay counters are invalid")
        normal_state = NormalLoaderReplayState(consumed, length, normal_payload["epoch_start_rng_state"])
    return progress, normal_state


def _normalize_named_generators(
    generators: Mapping[str, torch.Generator] | None,
) -> dict[str, torch.Generator]:
    if generators is None:
        return {}
    if not isinstance(generators, Mapping):
        raise CheckpointError("explicit_generators must be a name/generator mapping")
    result: dict[str, torch.Generator] = {}
    identities: set[int] = set()
    for name, generator in generators.items():
        if not isinstance(name, str) or not name:
            raise CheckpointError("Explicit generator names must be nonempty strings")
        if not isinstance(generator, torch.Generator):
            raise CheckpointError("Explicit generator values must be torch.Generator")
        if id(generator) in identities:
            raise CheckpointError("Explicit generator object has multiple names")
        result[name] = generator
        identities.add(id(generator))
    return dict(sorted(result.items()))


def _serialize_rng_snapshot(
    snapshot: RNGSnapshot,
    generators: Mapping[str, torch.Generator],
) -> dict[str, object]:
    if not isinstance(snapshot, RNGSnapshot):
        raise CheckpointError("RNG snapshot is malformed")
    name_by_identity = {id(generator): name for name, generator in generators.items()}
    observed = {id(item.generator) for item in snapshot.explicit_generators}
    if observed != set(name_by_identity):
        raise CheckpointError("RNG snapshot explicit generators differ from checkpoint mapping")
    return {
        "python_state": copy.deepcopy(snapshot.python_state),
        "numpy_state": (
            snapshot.numpy_state[0], snapshot.numpy_state[1].copy(),
            snapshot.numpy_state[2], snapshot.numpy_state[3], snapshot.numpy_state[4],
        ),
        "torch_cpu_state": snapshot.torch_cpu_state.clone(),
        "cuda_was_initialized": snapshot.cuda_was_initialized,
        "torch_cuda_states": tuple(state.clone() for state in snapshot.torch_cuda_states),
        "explicit_generators": {
            name_by_identity[id(item.generator)]: {
                "device": item.device,
                "state": item.state.clone(),
            }
            for item in snapshot.explicit_generators
        },
    }


def _validate_rng_payload(
    payload: object, generators: Mapping[str, torch.Generator]
) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "python_state", "numpy_state", "torch_cpu_state", "cuda_was_initialized",
        "torch_cuda_states", "explicit_generators",
    }:
        raise CheckpointError("Checkpoint RNG fields differ from schema")
    if not isinstance(payload["torch_cpu_state"], torch.Tensor):
        raise CheckpointError("Checkpoint CPU RNG state is malformed")
    if not isinstance(payload["cuda_was_initialized"], bool):
        raise CheckpointError("Checkpoint CUDA initialization flag is malformed")
    if not isinstance(payload["torch_cuda_states"], tuple) or not all(
        isinstance(item, torch.Tensor) for item in payload["torch_cuda_states"]
    ):
        raise CheckpointError("Checkpoint CUDA RNG states are malformed")
    explicit = payload["explicit_generators"]
    if not isinstance(explicit, dict) or set(explicit) != set(generators):
        raise CheckpointCompatibilityError("Checkpoint explicit generator names differ")
    for name, generator in generators.items():
        item = explicit[name]
        if not isinstance(item, dict) or set(item) != {"device", "state"}:
            raise CheckpointError("Checkpoint explicit generator entry is malformed")
        if item["device"] != str(generator.device):
            raise CheckpointCompatibilityError(
                f"Checkpoint explicit generator device differs for {name!r}"
            )
        if not isinstance(item["state"], torch.Tensor):
            raise CheckpointError("Checkpoint explicit generator state is malformed")
    numpy_state = payload["numpy_state"]
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5 or not isinstance(numpy_state[1], np.ndarray):
        raise CheckpointError("Checkpoint NumPy RNG state is malformed")


def _restore_rng_payload(
    payload: Mapping[str, object], generators: Mapping[str, torch.Generator]
) -> None:
    restore_rng_state(_rng_snapshot_from_payload(payload, generators))


def _rng_snapshot_from_payload(
    payload: Mapping[str, object], generators: Mapping[str, torch.Generator]
) -> RNGSnapshot:
    _validate_rng_payload(payload, generators)
    explicit = tuple(
        GeneratorSnapshot(
            generator=generator,
            device=str(generator.device),
            state=payload["explicit_generators"][name]["state"].clone(),
        )
        for name, generator in generators.items()
    )
    return RNGSnapshot(
        python_state=copy.deepcopy(payload["python_state"]),
        numpy_state=(
            payload["numpy_state"][0], payload["numpy_state"][1].copy(),
            payload["numpy_state"][2], payload["numpy_state"][3], payload["numpy_state"][4],
        ),
        torch_cpu_state=payload["torch_cpu_state"].clone(),
        cuda_was_initialized=payload["cuda_was_initialized"],
        torch_cuda_states=tuple(state.clone() for state in payload["torch_cuda_states"]),
        explicit_generators=explicit,
    )


def _rng_snapshots_equal(left: RNGSnapshot, right: RNGSnapshot) -> bool:
    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(torch.equal(a, b) for a, b in zip(left.torch_cuda_states, right.torch_cuda_states))
        and len(left.explicit_generators) == len(right.explicit_generators)
        and all(
            a.device == b.device and torch.equal(a.state, b.state)
            for a, b in zip(left.explicit_generators, right.explicit_generators)
        )
    )


def _owned_json(value: Any, path: str = "result_state") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CheckpointError(f"Nonfinite result scalar at {path}")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise CheckpointError(f"Non-string result key at {path}")
            result[key] = _owned_json(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_owned_json(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise CheckpointError(f"Unsupported result state at {path}: {type(value).__name__}")


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_for(
    path: Path,
    *,
    progress: CheckpointProgress,
    method: str,
    estimator_mode: str,
) -> CheckpointMetadata:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return CheckpointMetadata(
        path=path,
        byte_size=path.stat().st_size,
        sha256=digest.hexdigest(),
        next_optimizer_step=progress.next_optimizer_step,
        epoch_zero_based=progress.epoch_zero_based,
        next_batch_index_zero_based=progress.next_batch_index_zero_based,
        method=method,
        estimator_mode=estimator_mode,
    )
