"""Discovery and validation of immutable R2 checkpoint inputs for LC05/LC06."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

from sample_fg.paper_runner import (
    DEFAULT_CONFIG as DEFAULT_PAPER_CONFIG,
    build_scientific_plan,
    resolve_method,
)

from .checkpoint_probe import (
    ProbeCheckpoint,
    ProbeCheckpointError,
    load_probe_checkpoint,
)
from .runner import _portable_protocol


METHODS = (("coop", "none"), ("sam", "none"), ("sample", "ema"))
SEEDS = (1, 2, 3)
DATASETS = ("dtd", "eurosat")


@dataclass(frozen=True, order=True)
class R2CellKey:
    dataset: str
    method: str
    estimator: str
    seed: int
    shots: int = 16

    @property
    def method_key(self) -> str:
        return f"{self.method}_{self.estimator}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "shots": self.shots,
            "method": self.method,
            "estimator": self.estimator,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class R2Source:
    key: R2CellKey
    run_dir: Path
    checkpoint: ProbeCheckpoint
    source_config: dict[str, Any]
    source_summary: dict[str, Any]


@dataclass(frozen=True)
class DiscoveryReport:
    compatible: tuple[R2Source, ...]
    missing: tuple[R2CellKey, ...]
    excluded: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "compatible": [
                {
                    **item.key.as_dict(),
                    "run_dir": str(item.run_dir),
                    "checkpoint": str(item.checkpoint.checkpoint_path),
                    "checkpoint_sha256": item.checkpoint.checkpoint_sha256,
                }
                for item in self.compatible
            ],
            "missing": [item.as_dict() for item in self.missing],
            "excluded": list(self.excluded),
        }


def _mapping(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise ProbeCheckpointError(f"Cannot parse R2 artifact: {path}") from error
    if not isinstance(value, dict):
        raise ProbeCheckpointError(f"R2 artifact root is not a mapping: {path}")
    return value


def expected_cells(datasets: Iterable[str] = DATASETS) -> tuple[R2CellKey, ...]:
    return tuple(
        R2CellKey(dataset, method, estimator, seed)
        for dataset in datasets
        for method, estimator in METHODS
        for seed in SEEDS
    )


def _candidate_identity(config: dict[str, Any]) -> R2CellKey:
    try:
        return R2CellKey(
            dataset=str(config["data"]["dataset"]),
            shots=int(config["data"]["shots"]),
            seed=int(config["data"]["seed"]),
            method=str(config["method"]["name"]),
            estimator=str(config["estimator"]["mode"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProbeCheckpointError("R2 config identity is malformed") from error


def discover_r2_sources(
    results_root: Path,
    *,
    datasets: Iterable[str] = DATASETS,
) -> DiscoveryReport:
    """Discover, validate, and de-duplicate completed scientific R2 cells."""

    root = Path(results_root).resolve(strict=True)
    wanted = set(expected_cells(tuple(datasets)))
    found: dict[R2CellKey, R2Source] = {}
    conflicted: set[R2CellKey] = set()
    excluded: list[dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary.json")):
        run_dir = summary_path.parent.resolve()
        try:
            summary = _mapping(summary_path)
            if summary.get("status") != "completed":
                raise ProbeCheckpointError("summary status is not completed")
            config = _mapping(run_dir / "config.yaml")
            key = _candidate_identity(config)
            if key not in wanted:
                continue
            identity = summary.get("run_identity")
            if not isinstance(identity, dict):
                raise ProbeCheckpointError("summary run_identity is missing")
            expected_summary = (
                key.dataset, key.shots, key.seed, key.method, key.estimator
            )
            observed_summary = (
                identity.get("dataset"), identity.get("shots"), identity.get("seed"),
                identity.get("method_tag"), identity.get("estimator_tag"),
            )
            if observed_summary != expected_summary:
                raise ProbeCheckpointError("summary and config identities differ")
            relative_checkpoint = summary.get("artifacts", {}).get("checkpoint")
            if not isinstance(relative_checkpoint, str) or not relative_checkpoint:
                raise ProbeCheckpointError("summary final-checkpoint path is missing")
            probe = load_probe_checkpoint(
                run_dir,
                run_dir / relative_checkpoint,
                expected_dataset=key.dataset,
                expected_shots=key.shots,
                expected_seed=key.seed,
                expected_method=key.method,
                expected_estimator=key.estimator,
            )
            if key in conflicted:
                raise ProbeCheckpointError("cell already excluded due to duplicate compatible runs")
            if key in found:
                first = found.pop(key)
                conflicted.add(key)
                excluded.append(
                    {
                        "run_dir": str(first.run_dir),
                        "reason": f"duplicate compatible completed cell; other={run_dir}",
                    }
                )
                raise ProbeCheckpointError(
                    f"duplicate compatible completed cell; first={first.run_dir}"
                )
            found[key] = R2Source(key, run_dir, probe, config, summary)
        except (OSError, ProbeCheckpointError) as error:
            excluded.append({"run_dir": str(run_dir), "reason": str(error)})
    compatible = tuple(found[key] for key in sorted(found))
    missing = tuple(sorted(wanted - set(found)))
    return DiscoveryReport(compatible, missing, tuple(excluded))


def build_r2_scientific_plan(
    source: R2Source,
    *,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    runtime_output_root: Path,
    paper_config: Path = DEFAULT_PAPER_CONFIG,
):
    """Reconstruct one R2 protocol against machine-local resources."""

    config = source.source_config
    run = config["run"]
    checkpoint = config["checkpoint"]
    diagnostics = config["diagnostics"]
    estimator = config["estimator"]
    selection = resolve_method(source.key.method, source.key.estimator)
    plan = build_scientific_plan(
        dataset=source.key.dataset,
        shots=source.key.shots,
        seed=source.key.seed,
        experiment_id=str(run.get("experiment_id", "R2")),
        selection=selection,
        data_root=Path(data_root),
        manifest_root=Path(manifest_root),
        clip_cache=Path(clip_cache),
        output_root=Path(runtime_output_root),
        config_path=Path(paper_config),
        recovery_interval_epochs=int(checkpoint.get("recovery_interval_epochs", 10)),
        epochs=int(config["optim"]["max_epoch"]),
        diagnostic_interval_steps=diagnostics.get("full_gradient_interval_steps"),
        full_gradient_micro_batch_size=int(estimator["full_gradient_micro_batch_size"]),
        notes=str(run.get("notes", "Primary DTD/EuroSAT 16-shot CoOp paper reproduction")),
    )
    if _portable_protocol(plan.resolved_config) != _portable_protocol(config):
        raise ProbeCheckpointError(
            f"Reconstructed protocol differs for {source.key.as_dict()}"
        )
    return replace(plan, output_root=Path(runtime_output_root).resolve())
