"""Strict, self-contained artifact family for the integrated LC01+LC04 probe."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from sample_fg.results import atomic_write_json, atomic_write_yaml


CONFIG_SCHEMA_VERSION = "sample_fg.low_compute_config.v1"
SOURCE_SCHEMA_VERSION = "sample_fg.low_compute_source.v1"
METRICS_SCHEMA_VERSION = "sample_fg.low_compute_metrics.v1"
SUMMARY_SCHEMA_VERSION = "sample_fg.low_compute_summary.v1"


class LowComputeArtifactError(RuntimeError):
    pass


def _strict(value: Any, path: str = "root") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LowComputeArtifactError(f"Nonfinite scalar at {path}")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise LowComputeArtifactError(f"Non-string JSON key at {path}")
        return {key: _strict(child, f"{path}.{key}") for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(child, f"{path}[]") for child in value]
    raise LowComputeArtifactError(f"Unsupported artifact value at {path}")


@dataclass
class LowComputeArtifacts:
    run_dir: Path

    def create(
        self,
        *,
        config: Mapping[str, Any],
        environment: Mapping[str, Any],
        source: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> None:
        self.run_dir = Path(self.run_dir)
        if self.run_dir.exists():
            raise LowComputeArtifactError(f"Probe directory already exists: {self.run_dir}")
        if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
            raise LowComputeArtifactError("Unsupported low-compute config schema")
        if source.get("schema_version") != SOURCE_SCHEMA_VERSION:
            raise LowComputeArtifactError("Unsupported low-compute source schema")
        if budget.get("schema_version") != "sample_fg.low_compute_budget.v1":
            raise LowComputeArtifactError("Unsupported low-compute budget schema")
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "logs").mkdir()
        (self.run_dir / "cache").mkdir()
        (self.run_dir / "lc01").mkdir()
        (self.run_dir / "lc04").mkdir()
        atomic_write_yaml(self.run_dir / "config.yaml", _strict(config))
        atomic_write_json(self.run_dir / "environment.json", _strict(environment))
        atomic_write_json(self.run_dir / "source.json", _strict(source))
        atomic_write_json(self.run_dir / "compute_budget.json", _strict(budget))
        (self.run_dir / "metrics.jsonl").write_bytes(b"")
        (self.run_dir / "logs" / "run.log").write_bytes(b"")

    def append_metric(self, record: Mapping[str, Any]) -> None:
        payload = _strict(record)
        if payload.get("schema_version") != METRICS_SCHEMA_VERSION:
            raise LowComputeArtifactError("Unsupported low-compute metric schema")
        if payload.get("task") not in {"lc01", "lc04"}:
            raise LowComputeArtifactError("Metric task must be lc01 or lc04")
        line = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())

    def write_summary(self, summary: Mapping[str, Any]) -> None:
        payload = _strict(summary)
        if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
            raise LowComputeArtifactError("Unsupported low-compute summary schema")
        safety = payload.get("safety")
        if not isinstance(safety, dict):
            raise LowComputeArtifactError("Summary safety mapping is required")
        if safety.get("optimizer_steps_executed") != 0:
            raise LowComputeArtifactError("Frozen-probe summary must report zero optimizer steps")
        if safety.get("scheduler_steps_executed") != 0:
            raise LowComputeArtifactError("Frozen-probe summary must report zero scheduler steps")
        if safety.get("model_parameters_changed") is not False:
            raise LowComputeArtifactError("Frozen-probe summary must prove model immutability")
        atomic_write_json(self.run_dir / "summary.json", payload)

    def write_lc01(self, name: str, payload: Mapping[str, Any]) -> None:
        allowed = {
            "gradient_bank_index.json", "replay_summary.json",
            "geometry_summary.json", "compute_accounting.json",
        }
        if name not in allowed:
            raise LowComputeArtifactError(f"Unexpected LC01 JSON artifact: {name}")
        atomic_write_json(self.run_dir / "lc01" / name, _strict(payload))

    def write_lc04(self, payload: Mapping[str, Any]) -> None:
        atomic_write_json(
            self.run_dir / "lc04" / "function_space_fidelity.json", _strict(payload)
        )


def validate_saved_artifacts(run_dir: Path) -> None:
    root = Path(run_dir).resolve(strict=True)
    required = {
        "config.yaml", "environment.json", "source.json", "compute_budget.json",
        "metrics.jsonl", "summary.json", "logs/run.log",
    }
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
    }
    missing = required - observed
    if missing:
        raise LowComputeArtifactError(f"Low-compute artifact set is incomplete: {sorted(missing)}")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise LowComputeArtifactError("Saved summary schema differs")
