"""Strict, atomic machine-readable run artifact writers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml


RESOLVED_CONFIG_SCHEMA_VERSION = "sample_fg.resolved_config.v1"
METRICS_SCHEMA_VERSION = "sample_fg.metrics.v1"
SUMMARY_SCHEMA_VERSION = "sample_fg.summary.v1"
RUN_ARTIFACT_SCHEMA_VERSION = "sample_fg.run_artifacts.v1"
_SAFE_TAG = re.compile(r"^[A-Za-z0-9_.-]+$")


class RunArtifactError(RuntimeError):
    """Raised before an artifact can become ambiguous or partially corrupt."""


def _owned_json(value: Any, path: str = "root") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunArtifactError(f"Nonfinite JSON scalar at {path}")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise RunArtifactError(f"Non-string JSON key at {path}")
            result[key] = _owned_json(child, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_owned_json(child, f"{path}[{index}]") for index, child in enumerate(value)]
    raise RunArtifactError(f"Unsupported JSON value at {path}: {type(value).__name__}")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    owned = _owned_json(value)
    return json.dumps(
        owned,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _config_digest(payload: Mapping[str, Any]) -> str:
    preimage = _owned_json(payload)
    preimage.pop("config_sha256", None)
    run = preimage.get("run")
    if isinstance(run, dict):
        run.pop("run_id", None)
    return hashlib.sha256(_canonical_bytes(preimage)).hexdigest()


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Own a resolved config and add its versioned deterministic hash."""

    payload = _owned_json(config)
    if not isinstance(payload, dict):
        raise RunArtifactError("Resolved config must be a mapping")
    payload["schema_version"] = RESOLVED_CONFIG_SCHEMA_VERSION
    payload.pop("config_sha256", None)
    payload["config_sha256"] = _config_digest(payload)
    return payload


def bind_run_identity(
    resolved_config: Mapping[str, Any], identity: "RunIdentity"
) -> dict[str, Any]:
    """Bind the derived run ID without making it part of its own hash preimage."""

    payload = _owned_json(resolved_config)
    if payload.get("config_sha256") != identity.config_sha256:
        raise RunArtifactError("Run identity and resolved config hashes differ")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise RunArtifactError("Resolved config requires a run mapping")
    run["run_id"] = identity.run_id
    if _config_digest(payload) != identity.config_sha256:
        raise RunArtifactError("Resolved config changed after hashing")
    return payload


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(
        _owned_json(payload),
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(Path(path), content)


def atomic_write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    content = yaml.safe_dump(
        _owned_json(payload),
        allow_unicode=True,
        sort_keys=True,
    ).encode("utf-8")
    _atomic_bytes(Path(path), content)


def _validate_tag(name: str, value: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_TAG.fullmatch(value):
        raise RunArtifactError(f"Unsafe or empty {name}: {value!r}")
    return value


@dataclass(frozen=True)
class RunIdentity:
    dataset: str
    shots: int
    method_tag: str
    estimator_tag: str
    seed: int
    utc_timestamp: str
    config_sha256: str
    experiment_id: str
    smoke: bool
    allow_scientific_summary: bool

    def __post_init__(self) -> None:
        for name in ("dataset", "method_tag", "estimator_tag", "utc_timestamp", "experiment_id"):
            _validate_tag(name, getattr(self, name))
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots < 1:
            raise RunArtifactError("shots must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise RunArtifactError("seed must be an integer")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_sha256):
            raise RunArtifactError("config_sha256 must be lowercase SHA-256")
        if self.smoke and self.allow_scientific_summary:
            raise RunArtifactError("Smoke runs cannot allow scientific summary")

    @property
    def config_hash8(self) -> str:
        return self.config_sha256[:8]

    @property
    def run_id(self) -> str:
        return f"{self.utc_timestamp}_{self.config_hash8}"

    @classmethod
    def now(
        cls,
        *,
        dataset: str,
        shots: int,
        method_tag: str,
        estimator_tag: str,
        seed: int,
        config_sha256: str,
        experiment_id: str,
        smoke: bool,
        allow_scientific_summary: bool,
    ) -> "RunIdentity":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return cls(
            dataset=dataset,
            shots=shots,
            method_tag=method_tag,
            estimator_tag=estimator_tag,
            seed=seed,
            utc_timestamp=timestamp,
            config_sha256=config_sha256,
            experiment_id=experiment_id,
            smoke=smoke,
            allow_scientific_summary=allow_scientific_summary,
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run_id"] = self.run_id
        return payload


@dataclass
class RunAccounting:
    train_total_s: float = 0.0
    full_gradient_total_s: float = 0.0
    peak_cuda_allocated_bytes: int = 0
    peak_cuda_reserved_bytes: int = 0
    compute_counts: dict[str, int] = field(default_factory=dict)
    total_wall_s: float = 0.0

    def increment(self, name: str, amount: int = 1) -> None:
        _validate_tag("counter name", name)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise RunArtifactError("Counter increments must be nonnegative integers")
        self.compute_counts[name] = self.compute_counts.get(name, 0) + amount

    def as_dict(self) -> dict[str, Any]:
        return _owned_json(asdict(self))


class RunArtifacts:
    """One self-contained canonical run directory."""

    def __init__(self, root: Path, identity: RunIdentity) -> None:
        self.root = Path(root)
        self.identity = identity
        self.run_dir = (
            self.root
            / identity.dataset
            / f"shots_{identity.shots}"
            / identity.method_tag
            / identity.estimator_tag
            / f"seed_{identity.seed}"
            / identity.run_id
        )
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.diagnostics_path = self.run_dir / "gradient_diagnostics.jsonl"

    def create(
        self,
        *,
        resolved_config: Mapping[str, Any],
        environment: Mapping[str, Any],
        data_manifest_source: Path,
    ) -> Path:
        if self.run_dir.exists():
            raise RunArtifactError(f"Run directory already exists: {self.run_dir}")
        config = _owned_json(resolved_config)
        if config.get("schema_version") != RESOLVED_CONFIG_SCHEMA_VERSION:
            raise RunArtifactError("Config is not resolved with the supported schema")
        if config.get("config_sha256") != self.identity.config_sha256:
            raise RunArtifactError("Run identity and resolved config hashes differ")
        if _config_digest(config) != self.identity.config_sha256:
            raise RunArtifactError("Resolved config content does not match its hash")
        if config.get("run", {}).get("run_id") != self.identity.run_id:
            raise RunArtifactError("Resolved config run_id is not fully bound")
        manifest_source = Path(data_manifest_source).resolve(strict=True)
        manifest_bytes = manifest_source.read_bytes()
        try:
            _owned_json(json.loads(manifest_bytes))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RunArtifactError("Data manifest is not valid strict JSON") from error

        self.run_dir.mkdir(parents=True)
        (self.run_dir / "checkpoints").mkdir()
        (self.run_dir / "logs").mkdir()
        atomic_write_yaml(self.run_dir / "config.yaml", config)
        atomic_write_json(self.run_dir / "environment.json", environment)
        _atomic_bytes(self.run_dir / "data_manifest.json", manifest_bytes)
        _atomic_bytes(self.metrics_path, b"")
        _atomic_bytes(self.diagnostics_path, b"")
        _atomic_bytes(self.run_dir / "logs" / "run.log", b"")
        return self.run_dir

    def _append_jsonl(self, path: Path, record: Mapping[str, Any]) -> None:
        if not self.run_dir.is_dir():
            raise RunArtifactError("Run artifacts have not been created")
        payload = _owned_json(record)
        if payload.get("run_id") != self.identity.run_id:
            raise RunArtifactError("Record run_id does not match its run directory")
        line = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n"
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def append_metric(self, record: Mapping[str, Any]) -> None:
        if record.get("schema_version") != METRICS_SCHEMA_VERSION:
            raise RunArtifactError("Metric schema_version is missing or unsupported")
        if not isinstance(record.get("event_type"), str):
            raise RunArtifactError("Metric event_type is required")
        self._append_jsonl(self.metrics_path, record)

    def append_diagnostic(self, record: Mapping[str, Any]) -> None:
        if not isinstance(record.get("schema_version"), str):
            raise RunArtifactError("Diagnostic schema_version is required")
        if record.get("optimizer_step") is None:
            raise RunArtifactError("Diagnostic optimizer_step is required")
        self._append_jsonl(self.diagnostics_path, record)

    def append_log(self, message: str) -> None:
        if not isinstance(message, str) or "\x00" in message:
            raise RunArtifactError("Log message must be text without NUL")
        with (self.run_dir / "logs" / "run.log").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(message.rstrip("\n") + "\n")

    def write_summary(self, summary: Mapping[str, Any]) -> Path:
        payload = _owned_json(summary)
        if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
            raise RunArtifactError("Summary schema_version is missing or unsupported")
        if payload.get("run_identity") != self.identity.as_dict():
            raise RunArtifactError("Summary run identity differs from directory identity")
        if payload.get("status") not in {"completed", "aborted", "failed"}:
            raise RunArtifactError("Summary status is invalid")
        if self.identity.smoke and payload.get("allow_scientific_summary") is not False:
            raise RunArtifactError("Smoke summary must explicitly prohibit scientific use")
        path = self.run_dir / "summary.json"
        atomic_write_json(path, payload)
        return path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunArtifactError(f"Invalid JSONL line {line_number} in {path}") from error
        if not isinstance(value, dict):
            raise RunArtifactError(f"JSONL line {line_number} is not an object")
        records.append(value)
    return records


def copy_artifact(source: Path, destination: Path) -> None:
    """Atomic byte-for-byte copy for an existing authoritative artifact."""

    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
