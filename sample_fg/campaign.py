"""Declarative Task 25--28 campaign matrix and periodic-K freeze gates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


CAMPAIGN_SCHEMA_VERSION = "sample_fg.extension_campaign.v1"
PERIODIC_K_FREEZE_SCHEMA_VERSION = "sample_fg.periodic_k_freeze.v1"
TASK_KEYS = ("task25", "task26", "task27", "task28")


class CampaignError(RuntimeError):
    """Raised when a planned scientific cell or freeze record is ambiguous."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CampaignError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CampaignError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class CampaignCell:
    task: str
    title: str
    experiment_id: str
    dataset: str
    shots: int
    seed: int
    method: str
    estimator: str
    periodic_k_steps: int | None
    reuse_experiment_id: str | None
    epochs: int
    train_batch_size: int
    base_classes: int
    full_gradient_micro_batch_size: int
    diagnostic_cadence: str

    @property
    def selected_count(self) -> int:
        return self.base_classes * self.shots

    @property
    def steps_per_epoch(self) -> int:
        # Pinned Dassl drops the short final batch only when the selected source
        # is at least one configured batch.  This is the protocol recorded by
        # the Task-2 manifests and the R2 runner.
        if self.selected_count < self.train_batch_size:
            return 1
        return self.selected_count // self.train_batch_size

    @property
    def samples_consumed_per_epoch(self) -> int:
        if self.selected_count < self.train_batch_size:
            return self.selected_count
        return self.steps_per_epoch * self.train_batch_size

    @property
    def total_optimizer_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def expected_periodic_refresh_count(self) -> int | None:
        if self.estimator != "periodic" or self.periodic_k_steps is None:
            return None
        return ((self.total_optimizer_steps - 1) // self.periodic_k_steps) + 1

    @property
    def expected_diagnostic_points(self) -> int:
        return self.epochs if self.method == "sample" else 0

    @property
    def expected_optimization_exact_queries(self) -> int:
        if self.estimator == "exact":
            return self.total_optimizer_steps
        return self.expected_periodic_refresh_count or 0

    @property
    def expected_reused_exact_queries(self) -> int:
        if self.estimator == "exact":
            return self.expected_diagnostic_points
        if self.estimator != "periodic" or self.periodic_k_steps is None:
            return 0
        return sum(
            1
            for step in range(0, self.total_optimizer_steps, self.steps_per_epoch)
            if step % self.periodic_k_steps == 0
        )

    @property
    def expected_diagnostic_only_exact_queries(self) -> int:
        return self.expected_diagnostic_points - self.expected_reused_exact_queries

    @property
    def expected_exact_sweeps(self) -> int:
        return (
            self.expected_optimization_exact_queries
            + self.expected_diagnostic_only_exact_queries
        )

    @property
    def estimator_tag(self) -> str:
        if self.estimator == "periodic":
            return f"periodic-k{self.periodic_k_steps}"
        return self.estimator

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "title": self.title,
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "shots": self.shots,
            "seed": self.seed,
            "method": self.method,
            "estimator": self.estimator,
            "estimator_tag": self.estimator_tag,
            "periodic_k_steps": self.periodic_k_steps,
            "reuse_experiment_id": self.reuse_experiment_id,
            "execution_mode": (
                "reuse_existing_artifact"
                if self.reuse_experiment_id is not None
                else "new_scientific_run"
            ),
            "epochs": self.epochs,
            "train_batch_size": self.train_batch_size,
            "selected_count": self.selected_count,
            "steps_per_epoch": self.steps_per_epoch,
            "samples_consumed_per_epoch": self.samples_consumed_per_epoch,
            "total_optimizer_steps": self.total_optimizer_steps,
            "expected_periodic_refresh_count": (
                self.expected_periodic_refresh_count
            ),
            "expected_diagnostic_points": self.expected_diagnostic_points,
            "expected_optimization_exact_queries": (
                self.expected_optimization_exact_queries
            ),
            "expected_diagnostic_only_exact_queries": (
                self.expected_diagnostic_only_exact_queries
            ),
            "expected_reused_exact_queries": self.expected_reused_exact_queries,
            "expected_exact_sweeps": self.expected_exact_sweeps,
            "full_gradient_micro_batch_size": (
                self.full_gradient_micro_batch_size
            ),
            "diagnostic_cadence": self.diagnostic_cadence,
        }


@dataclass(frozen=True)
class PeriodicKFreeze:
    path: Path
    campaign_config_sha256: str
    selected_k_values: tuple[int, ...]
    f0_k: int
    accuracy_used: bool
    source_aggregation_sha256: str
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "campaign_config_sha256": self.campaign_config_sha256,
            "selected_k_values": list(self.selected_k_values),
            "f0_k": self.f0_k,
            "accuracy_used": self.accuracy_used,
            "source_aggregation_sha256": self.source_aggregation_sha256,
            "rationale": self.rationale,
        }


class CampaignManifest:
    """Validated, deterministic expansion of the checked-in campaign YAML."""

    def __init__(self, path: Path, payload: Mapping[str, Any]) -> None:
        self.path = Path(path).resolve(strict=True)
        self.sha256 = _sha256(self.path)
        if payload.get("schema_version") != CAMPAIGN_SCHEMA_VERSION:
            raise CampaignError("Unsupported extension campaign schema")
        self.payload = dict(payload)
        self.protocol = _mapping(payload.get("protocol"), "protocol")
        self.selection = _mapping(
            payload.get("periodic_k_selection"), "periodic_k_selection"
        )
        self.tasks = _mapping(payload.get("tasks"), "tasks")
        if tuple(self.tasks) != TASK_KEYS:
            raise CampaignError(
                f"Campaign tasks must be ordered exactly as {TASK_KEYS}"
            )
        allowed = self.selection.get("allowed_k")
        if not isinstance(allowed, list) or not allowed:
            raise CampaignError("periodic_k_selection.allowed_k must be a list")
        self.allowed_k = tuple(
            _positive_int(value, "allowed periodic K") for value in allowed
        )
        if len(set(self.allowed_k)) != len(self.allowed_k):
            raise CampaignError("Allowed periodic K values contain duplicates")
        if self.selection.get("accuracy_selects_k") is not False:
            raise CampaignError("Campaign must prohibit accuracy-based K selection")

    @classmethod
    def load(cls, path: Path) -> "CampaignManifest":
        path = Path(path).resolve(strict=True)
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError) as error:
            raise CampaignError(f"Cannot read campaign manifest: {path}") from error
        return cls(path, _mapping(payload, "campaign root"))

    def task_title(self, task: str) -> str:
        return str(_mapping(self.tasks.get(task), task).get("title"))

    def cells(
        self,
        task: str,
        *,
        frozen_k_values: Iterable[int] | None = None,
        allow_unfrozen: bool = False,
    ) -> tuple[CampaignCell, ...]:
        if task not in TASK_KEYS:
            raise CampaignError(f"Unknown campaign task: {task!r}")
        definition = _mapping(self.tasks[task], task)
        protocol_datasets = _mapping(self.protocol.get("datasets"), "datasets")
        epochs = _positive_int(self.protocol.get("epochs"), "protocol epochs")
        train_batch = _positive_int(
            self.protocol.get("train_batch_size"), "train batch size"
        )
        full_batch = _positive_int(
            self.protocol.get("full_gradient_micro_batch_size"),
            "full-gradient micro-batch size",
        )
        cadence = str(self.protocol.get("diagnostic_cadence"))
        frozen = tuple(frozen_k_values or ())
        for value in frozen:
            if value not in self.allowed_k:
                raise CampaignError(f"Frozen K is outside allowed set: {value}")
        output: list[CampaignCell] = []
        variants = definition.get("variants")
        if not isinstance(variants, list) or not variants:
            raise CampaignError(f"{task}.variants must be a nonempty list")
        for dataset in definition.get("datasets", ()):
            dataset_protocol = _mapping(
                protocol_datasets.get(dataset), f"protocol dataset {dataset}"
            )
            base_classes = _positive_int(
                dataset_protocol.get("base_classes"), f"{dataset} base classes"
            )
            for shots in definition.get("shots", ()):
                shots = _positive_int(shots, "shots")
                for seed in definition.get("seeds", ()):
                    seed = _positive_int(seed, "seed")
                    for raw_variant in variants:
                        variant = _mapping(raw_variant, f"{task} variant")
                        estimator = str(variant.get("estimator"))
                        reuse_experiment_id = variant.get("reuse_experiment_id")
                        if reuse_experiment_id is not None and (
                            not isinstance(reuse_experiment_id, str)
                            or not reuse_experiment_id
                        ):
                            raise CampaignError(
                                f"{task} reuse_experiment_id must be a nonempty string"
                            )
                        raw_k = variant.get("periodic_k_steps")
                        if raw_k == "frozen":
                            if not frozen:
                                if allow_unfrozen:
                                    k_values: tuple[int | None, ...] = (None,)
                                else:
                                    raise CampaignError(
                                        f"{task} requires a periodic-K freeze"
                                    )
                            else:
                                k_values = frozen
                        else:
                            k_values = (raw_k,)
                        for periodic_k in k_values:
                            if estimator == "periodic":
                                if periodic_k is not None:
                                    periodic_k = _positive_int(
                                        periodic_k, "periodic K"
                                    )
                                    if periodic_k not in self.allowed_k:
                                        raise CampaignError(
                                            f"Periodic K {periodic_k} is not predeclared"
                                        )
                            elif periodic_k is not None:
                                raise CampaignError(
                                    f"Non-periodic {estimator} variant owns K"
                                )
                            output.append(
                                CampaignCell(
                                    task=task,
                                    title=str(definition.get("title")),
                                    experiment_id=str(
                                        definition.get("experiment_id")
                                    ),
                                    dataset=str(dataset),
                                    shots=shots,
                                    seed=seed,
                                    method=str(variant.get("method")),
                                    estimator=estimator,
                                    periodic_k_steps=periodic_k,
                                    reuse_experiment_id=reuse_experiment_id,
                                    epochs=epochs,
                                    train_batch_size=train_batch,
                                    base_classes=base_classes,
                                    full_gradient_micro_batch_size=full_batch,
                                    diagnostic_cadence=cadence,
                                )
                            )
        return tuple(output)

    def validate_cell(
        self,
        *,
        task: str,
        dataset: str,
        shots: int,
        seed: int,
        method: str,
        estimator: str,
        periodic_k_steps: int | None,
        frozen_k_values: Iterable[int] | None = None,
    ) -> CampaignCell:
        candidates = self.cells(task, frozen_k_values=frozen_k_values)
        key = (dataset, shots, seed, method, estimator, periodic_k_steps)
        matches = [
            cell
            for cell in candidates
            if (
                cell.dataset,
                cell.shots,
                cell.seed,
                cell.method,
                cell.estimator,
                cell.periodic_k_steps,
            )
            == key
        ]
        if len(matches) != 1:
            raise CampaignError(
                f"Cell is not declared exactly once in {task}: {key}"
            )
        return matches[0]


def load_periodic_k_freeze(
    path: Path,
    *,
    campaign: CampaignManifest,
) -> PeriodicKFreeze:
    path = Path(path).resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"Cannot read periodic-K freeze: {path}") from error
    root = _mapping(payload, "periodic-K freeze")
    if root.get("schema_version") != PERIODIC_K_FREEZE_SCHEMA_VERSION:
        raise CampaignError("Unsupported periodic-K freeze schema")
    if root.get("campaign_config_sha256") != campaign.sha256:
        raise CampaignError("Periodic-K freeze targets a different campaign config")
    selected = root.get("selected_k_values")
    if not isinstance(selected, list) or not 1 <= len(selected) <= 2:
        raise CampaignError("Freeze must retain one or two periodic K values")
    selected_k = tuple(_positive_int(value, "selected K") for value in selected)
    if len(set(selected_k)) != len(selected_k):
        raise CampaignError("Freeze selected K values contain duplicates")
    if any(value not in campaign.allowed_k for value in selected_k):
        raise CampaignError("Freeze selected a K outside the predeclared set")
    if root.get("accuracy_used") is not False:
        raise CampaignError("Accuracy must not be used to freeze periodic K")
    f0_k = _positive_int(root.get("f0_k"), "F0 K")
    if f0_k not in selected_k:
        raise CampaignError("F0 K must be one of the retained K values")
    source_hash = root.get("source_aggregation_sha256")
    rationale = root.get("rationale")
    if not isinstance(source_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", source_hash
    ):
        raise CampaignError("Freeze lacks the source aggregation SHA-256")
    if not isinstance(rationale, str) or not rationale.strip():
        raise CampaignError("Freeze requires a nonempty rationale")
    return PeriodicKFreeze(
        path=path,
        campaign_config_sha256=campaign.sha256,
        selected_k_values=selected_k,
        f0_k=f0_k,
        accuracy_used=False,
        source_aggregation_sha256=source_hash,
        rationale=rationale.strip(),
    )
