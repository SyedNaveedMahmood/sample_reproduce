"""Read-only scientific-checkpoint validation for LC probes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch
import yaml

from sample_fg.checkpoint import CHECKPOINT_SCHEMA_VERSION, CheckpointProgress
from sample_fg.coop_anchor import EXPECTED_CLIP_KEY, EXPECTED_CLIP_SHA256
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.results import resolve_config


class ProbeCheckpointError(RuntimeError):
    """Raised without installing state when a probe source is ambiguous."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ProbeCheckpointError(f"Cannot parse source artifact: {path}") from error
    if not isinstance(value, dict):
        raise ProbeCheckpointError(f"Source artifact is not a mapping: {path}")
    return value


@dataclass(frozen=True)
class ProbeCheckpoint:
    source_run_dir: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    source_config_sha256: str
    source_fingerprint: str
    dataset: str
    shots: int
    seed: int
    method: str
    estimator: str
    epoch_zero_based: int
    next_optimizer_step: int
    param_index_metadata: Mapping[str, Any]
    trainable_state: Mapping[str, torch.Tensor]
    estimator_state: Mapping[str, object] | None
    source_artifact_sha256: Mapping[str, str]

    def actual_ema(self, param_index: ParamIndex) -> GradientState:
        if self.estimator_state is None:
            raise ProbeCheckpointError("Checkpoint has no serialized estimator state")
        if self.estimator_state.get("mode") != "ema":
            raise ProbeCheckpointError("Checkpoint estimator is not EMA")
        return GradientState.from_state_dict(
            param_index, self.estimator_state.get("active_state")
        )

    def install_prompt(self, param_index: ParamIndex) -> None:
        metadata = self.param_index_metadata
        if (
            metadata.get("fingerprint_schema") != param_index.fingerprint_schema
            or metadata.get("fingerprint") != param_index.fingerprint
        ):
            raise ProbeCheckpointError("Runtime ParamIndex differs from checkpoint")
        if set(self.trainable_state) != set(param_index.names):
            raise ProbeCheckpointError("Checkpoint trainable names differ from runtime")
        with torch.no_grad():
            for entry in param_index:
                value = self.trainable_state[entry.name]
                if tuple(value.shape) != entry.shape or value.dtype != entry.parameter.dtype:
                    raise ProbeCheckpointError(
                        f"Checkpoint tensor differs for {entry.name!r}"
                    )
                entry.parameter.copy_(value.to(device=entry.parameter.device))
                entry.parameter.grad = None


def load_probe_checkpoint(
    source_run_dir: Path,
    checkpoint_path: Path,
    *,
    expected_dataset: str = "dtd",
    expected_shots: int = 16,
    expected_seed: int = 1,
    expected_method: str = "sample",
    expected_estimator: str = "ema",
) -> ProbeCheckpoint:
    """Validate and own state without restoring optimizer/scheduler/RNG."""

    run_dir = Path(source_run_dir).resolve(strict=True)
    checkpoint = Path(checkpoint_path).resolve(strict=True)
    try:
        checkpoint.relative_to(run_dir)
    except ValueError as error:
        raise ProbeCheckpointError("Checkpoint must be inside its source run") from error
    config_path = (run_dir / "config.yaml").resolve(strict=True)
    manifest_path = (run_dir / "data_manifest.json").resolve(strict=True)
    summary_path = (run_dir / "summary.json").resolve(strict=True)
    config = _load_mapping(config_path)
    manifest = _load_mapping(manifest_path)
    summary = _load_mapping(summary_path)
    if summary.get("status") != "completed":
        raise ProbeCheckpointError("LC01 requires a completed source run")
    if manifest.get("schema_version") != "sample_fg.data_manifest.v1":
        raise ProbeCheckpointError("Source data manifest schema differs")

    data = config.get("data")
    method = config.get("method")
    estimator = config.get("estimator")
    model = config.get("model")
    run = config.get("run")
    smoke = config.get("smoke")
    if not all(isinstance(item, dict) for item in (data, method, estimator, model, run, smoke)):
        raise ProbeCheckpointError("Source config lacks scientific identity mappings")
    if run.get("experiment_id") != "R2" or run.get("smoke") is not False:
        raise ProbeCheckpointError("LC01 requires a non-smoke R2 source run")
    if smoke.get("allow_scientific_summary") is not True:
        raise ProbeCheckpointError("Source run is not permitted for scientific summary")
    identity = (
        data.get("dataset"), data.get("shots"), data.get("seed"),
        method.get("name"), estimator.get("mode"),
    )
    expected = (
        expected_dataset, expected_shots, expected_seed,
        expected_method, expected_estimator,
    )
    if identity != expected:
        raise ProbeCheckpointError(
            f"Source run identity differs: observed={identity}, expected={expected}"
        )
    expected_method_fields = {
        ("coop", "none"): (None, None, None),
        ("sam", "none"): (0.05, None, None),
        ("sample", "ema"): (0.05, 0.0015, 0.15),
    }
    method_key = (expected_method, expected_estimator)
    if method_key not in expected_method_fields:
        raise ProbeCheckpointError(f"Unsupported R2 method identity: {method_key}")
    observed_constants = (
        method.get("rho"), method.get("alpha"), method.get("ema_lambda")
    )
    if observed_constants != expected_method_fields[method_key]:
        raise ProbeCheckpointError("Source run does not use the paper method constants")
    expected_selected_count = {"dtd": 384, "eurosat": 80}.get(expected_dataset)
    if (
        expected_selected_count is None
        or data.get("selected_count") != expected_selected_count
        or data.get("train_batch_size") != 32
    ):
        raise ProbeCheckpointError("Source run does not use the primary selected source")
    if (
        model.get("backbone") != EXPECTED_CLIP_KEY
        or str(model.get("checkpoint_sha256", "")).lower()
        != EXPECTED_CLIP_SHA256.lower()
    ):
        raise ProbeCheckpointError("Source CLIP identity differs from pinned ViT-B/16")
    config_hash = config.get("config_sha256")
    source_fingerprint = data.get("selected_source_fingerprint")
    if not isinstance(config_hash, str) or not config_hash:
        raise ProbeCheckpointError("Source config hash is missing")
    if resolve_config(config).get("config_sha256") != config_hash:
        raise ProbeCheckpointError("Source config content does not match its hash")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ProbeCheckpointError("Selected-source fingerprint is missing")

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ProbeCheckpointError(f"Cannot load checkpoint: {checkpoint}") from error
    required = {
        "schema_version", "boundary", "method", "estimator_mode",
        "config_sha256", "source_fingerprint", "param_index",
        "trainable_model_state", "optimizer_state", "scheduler_state",
        "precision_state", "step_engine_state", "estimator_state", "progress",
        "rng_state", "normal_loader_state", "result_state",
        "gradient_buffer_policy", "legacy_compatibility", "created_utc",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ProbeCheckpointError("Checkpoint fields differ from scientific schema")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ProbeCheckpointError("Unsupported checkpoint schema")
    if payload.get("boundary") != "after_logical_optimizer_step_unperturbed_v1":
        raise ProbeCheckpointError("Checkpoint is not at an unperturbed step boundary")
    for observed, wanted, label in (
        (payload.get("method"), expected_method, "method"),
        (payload.get("estimator_mode"), expected_estimator, "estimator"),
        (payload.get("config_sha256"), config_hash, "config hash"),
        (payload.get("source_fingerprint"), source_fingerprint, "source fingerprint"),
    ):
        if observed != wanted:
            raise ProbeCheckpointError(f"Checkpoint {label} differs from source run")
    index = payload.get("param_index")
    trainable = payload.get("trainable_model_state")
    estimator_payload = payload.get("estimator_state")
    if not isinstance(index, dict) or not isinstance(trainable, dict):
        raise ProbeCheckpointError("Checkpoint trainable state is malformed")
    if set(trainable) != {"prompt_learner.ctx"}:
        raise ProbeCheckpointError("Probe source must train prompt_learner.ctx only")
    for value in trainable.values():
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ProbeCheckpointError("Checkpoint prompt tensor is malformed")
        if not bool(torch.isfinite(value).all().item()):
            raise ProbeCheckpointError("Checkpoint prompt tensor is nonfinite")
    if expected_estimator == "ema":
        if not isinstance(estimator_payload, dict) or estimator_payload.get("mode") != "ema":
            raise ProbeCheckpointError("Checkpoint EMA state is missing or malformed")
        if estimator_payload.get("ema_lambda") != 0.15:
            raise ProbeCheckpointError("Checkpoint does not use the paper EMA lambda")
    elif estimator_payload is not None:
        raise ProbeCheckpointError("CoOp/SAM checkpoint unexpectedly stores estimator state")
    progress = CheckpointProgress.from_dict(payload.get("progress"))
    artifacts = {
        "config.yaml": sha256_file(config_path),
        "data_manifest.json": sha256_file(manifest_path),
        "summary.json": sha256_file(summary_path),
        checkpoint.relative_to(run_dir).as_posix(): sha256_file(checkpoint),
    }
    return ProbeCheckpoint(
        source_run_dir=run_dir,
        checkpoint_path=checkpoint,
        checkpoint_sha256=artifacts[checkpoint.relative_to(run_dir).as_posix()],
        source_config_sha256=config_hash,
        source_fingerprint=source_fingerprint,
        dataset=expected_dataset,
        shots=expected_shots,
        seed=expected_seed,
        method=expected_method,
        estimator=expected_estimator,
        epoch_zero_based=progress.epoch_zero_based,
        next_optimizer_step=progress.next_optimizer_step,
        param_index_metadata=MappingProxyType(copy.deepcopy(index)),
        trainable_state=MappingProxyType(
            {name: value.detach().clone() for name, value in trainable.items()}
        ),
        estimator_state=MappingProxyType(copy.deepcopy(estimator_payload)),
        source_artifact_sha256=MappingProxyType(artifacts),
    )


def verify_source_immutable(probe: ProbeCheckpoint) -> None:
    for relative, expected in probe.source_artifact_sha256.items():
        path = probe.source_run_dir / relative
        if sha256_file(path) != expected:
            raise ProbeCheckpointError(f"Source artifact changed during probe: {relative}")
