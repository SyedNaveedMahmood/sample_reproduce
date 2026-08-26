"""Complete selected-record data path and exact dataset-mean gradient query."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from dassl.data.data_manager import DatasetWrapper
from dassl.data.datasets import Datum
from dassl.data.transforms import build_transform
from datasets.oxford_pets import OxfordPets

from .data_protocol import (
    COOP_COMMIT,
    DASSL_COMMIT,
    SCHEMA_VERSION as DATA_MANIFEST_SCHEMA_VERSION,
    LoadedDataset,
    sha256_file,
    stable_sample_id,
)
from .gradient_state import GradientState
from .param_index import ParamIndex
from .precision import PrecisionController
from .rng import DerivedSeed, isolated_rng


FULL_GRADIENT_SOURCE_SCHEMA_VERSION = "sample_fg.full_gradient_source.v1"
FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION = (
    "sample_fg.full_gradient_source.sequence.v1"
)
FULL_GRADIENT_SOURCE_ORDERING = "original_label_then_sample_id_v1"
FULL_GRADIENT_RESULT_SCHEMA_VERSION = "sample_fg.full_gradient_result.v1"


class FullGradientDataError(RuntimeError):
    """Raised when the dedicated source or loader violates its protocol."""


class FullGradientServiceError(RuntimeError):
    """Raised when an exact-gradient query violates its state contract."""


class FullGradientNumericalError(FloatingPointError):
    """Raised when an exact-gradient loss or gradient is nonfinite."""


@dataclass(frozen=True)
class FullGradientSweepMetadata:
    """Scalar/accounting evidence for one exact conditional sweep."""

    sample_count: int
    micro_batch_count: int
    configured_micro_batch_size: int
    observed_micro_batch_sizes: tuple[int, ...]
    forward_calls: int
    autograd_grad_calls: int
    mean_loss: float
    elapsed_s: float
    precision_mode: str
    param_index_fingerprint: str
    source_fingerprint: str | None
    seed: DerivedSeed
    schema_version: str = FULL_GRADIENT_RESULT_SCHEMA_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sample_count": self.sample_count,
            "micro_batch_count": self.micro_batch_count,
            "configured_micro_batch_size": self.configured_micro_batch_size,
            "observed_micro_batch_sizes": list(self.observed_micro_batch_sizes),
            "forward_calls": self.forward_calls,
            "autograd_grad_calls": self.autograd_grad_calls,
            "mean_loss": self.mean_loss,
            "elapsed_s": self.elapsed_s,
            "precision_mode": self.precision_mode,
            "param_index_fingerprint": self.param_index_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "rng": self.seed.as_dict(),
            "sample_weighting": "sum_j(n_j/N * mean_gradient_j)",
            "loss_weighting": "sum_j(n_j/N * mean_loss_j)",
            "create_graph": False,
            "retain_graph": False,
        }


@dataclass(frozen=True)
class FullGradientResult:
    """Owned exact FP32 gradient plus immutable sweep metadata."""

    gradient: GradientState
    metadata: FullGradientSweepMetadata

    @property
    def mean_loss(self) -> float:
        return self.metadata.mean_loss


@dataclass(frozen=True)
class FullGradientClass:
    """Immutable base-class mapping used by the dedicated source."""

    original_label: int
    training_label: int
    classname: str

    def __post_init__(self) -> None:
        if isinstance(self.original_label, bool) or not isinstance(
            self.original_label, int
        ):
            raise FullGradientDataError("original_label must be an integer")
        if isinstance(self.training_label, bool) or not isinstance(
            self.training_label, int
        ):
            raise FullGradientDataError("training_label must be an integer")
        if self.original_label < 0 or self.training_label < 0:
            raise FullGradientDataError("class labels must be nonnegative")
        if not isinstance(self.classname, str) or not self.classname:
            raise FullGradientDataError("classname must be a nonempty string")


@dataclass(frozen=True)
class FullGradientRecord:
    """One immutable selected base-training record, before transformation."""

    position: int
    sample_id: str
    image_path: Path
    original_label: int
    training_label: int
    classname: str
    domain: int
    dataset: str
    shots: int
    seed: int

    def __post_init__(self) -> None:
        integer_fields = {
            "position": self.position,
            "original_label": self.original_label,
            "training_label": self.training_label,
            "domain": self.domain,
            "shots": self.shots,
            "seed": self.seed,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise FullGradientDataError(f"{name} must be an integer")
        if self.position < 0 or self.original_label < 0 or self.training_label < 0:
            raise FullGradientDataError("position and labels must be nonnegative")
        if self.shots < 1 or self.seed < 0:
            raise FullGradientDataError("shots must be positive and seed nonnegative")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise FullGradientDataError("sample_id must be a nonempty string")
        if not isinstance(self.image_path, Path):
            raise FullGradientDataError("image_path must be pathlib.Path")
        if not isinstance(self.classname, str) or not self.classname:
            raise FullGradientDataError("classname must be a nonempty string")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise FullGradientDataError("dataset must be a nonempty string")

    def fingerprint_entry(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "original_label": self.original_label,
            "training_label": self.training_label,
            "classname": self.classname,
        }


@dataclass(frozen=True)
class FullGradientSource(Sequence[FullGradientRecord]):
    """Validated, immutable complete base few-shot source in canonical order."""

    dataset: str
    dataset_name: str
    dataset_root: Path
    shots: int
    seed: int
    base_classes: tuple[FullGradientClass, ...]
    records: tuple[FullGradientRecord, ...]
    manifest_path: Path | None = None
    official_split_sha256: str | None = None
    fewshot_cache_path: Path | None = None
    fewshot_cache_sha256: str | None = None
    ordering: str = FULL_GRADIENT_SOURCE_ORDERING
    schema_version: str = FULL_GRADIENT_SOURCE_SCHEMA_VERSION
    fingerprint_schema_version: str = FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != FULL_GRADIENT_SOURCE_SCHEMA_VERSION:
            raise FullGradientDataError("Unsupported full-gradient source schema")
        if self.fingerprint_schema_version != FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION:
            raise FullGradientDataError("Unsupported source fingerprint schema")
        if self.ordering != FULL_GRADIENT_SOURCE_ORDERING:
            raise FullGradientDataError(f"Unsupported source ordering: {self.ordering}")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise FullGradientDataError("dataset must be a nonempty string")
        if not isinstance(self.dataset_name, str) or not self.dataset_name:
            raise FullGradientDataError("dataset_name must be a nonempty string")
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots < 1:
            raise FullGradientDataError("shots must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise FullGradientDataError("seed must be a nonnegative integer")
        if not isinstance(self.dataset_root, Path) or not self.dataset_root.is_dir():
            raise FullGradientDataError(
                f"dataset_root must be an existing directory: {self.dataset_root}"
            )
        if not isinstance(self.base_classes, tuple) or not self.base_classes:
            raise FullGradientDataError("base_classes must be a nonempty tuple")
        if any(not isinstance(item, FullGradientClass) for item in self.base_classes):
            raise FullGradientDataError("base_classes contains a malformed entry")
        if not isinstance(self.records, tuple) or not self.records:
            raise FullGradientDataError("complete selected source must not be empty")
        if any(not isinstance(item, FullGradientRecord) for item in self.records):
            raise FullGradientDataError("records contains a malformed entry")

        training_labels = [item.training_label for item in self.base_classes]
        original_labels = [item.original_label for item in self.base_classes]
        classnames = [item.classname for item in self.base_classes]
        if training_labels != list(range(len(self.base_classes))):
            raise FullGradientDataError(
                "base classes must use contiguous pinned training labels in order"
            )
        if len(original_labels) != len(set(original_labels)):
            raise FullGradientDataError("duplicate base original label")
        if len(classnames) != len(set(classnames)):
            raise FullGradientDataError("duplicate base classname")

        class_by_original = {
            item.original_label: item for item in self.base_classes
        }
        seen_ids: set[str] = set()
        counts: Counter[int] = Counter()
        root = self.dataset_root.resolve(strict=True)
        for expected_position, record in enumerate(self.records):
            if record.position != expected_position:
                raise FullGradientDataError(
                    "record positions must be contiguous and match canonical order"
                )
            if record.dataset != self.dataset:
                raise FullGradientDataError("record dataset differs from source")
            if record.shots != self.shots or record.seed != self.seed:
                raise FullGradientDataError("record shots/seed differs from source")
            _validate_portable_sample_id(record.sample_id)
            if record.sample_id in seen_ids:
                raise FullGradientDataError(
                    f"duplicate selected sample identity: {record.sample_id}"
                )
            seen_ids.add(record.sample_id)
            class_entry = class_by_original.get(record.original_label)
            if class_entry is None:
                raise FullGradientDataError(
                    f"record has a non-base/novel label: {record.original_label}"
                )
            if (
                record.training_label != class_entry.training_label
                or record.classname != class_entry.classname
            ):
                raise FullGradientDataError(
                    f"inconsistent label/class metadata for {record.sample_id}"
                )
            expected_path = _resolve_under_root(root, record.sample_id)
            try:
                actual_path = record.image_path.resolve(strict=True)
            except FileNotFoundError as error:
                raise FullGradientDataError(
                    f"selected image is missing: {record.image_path}"
                ) from error
            if not actual_path.is_file() or actual_path != expected_path:
                raise FullGradientDataError(
                    f"image path/sample ID mismatch: {record.sample_id}"
                )
            counts[record.original_label] += 1

        expected_count = len(self.base_classes) * self.shots
        if len(self.records) != expected_count:
            raise FullGradientDataError(
                f"selected count {len(self.records)} differs from derived {expected_count}"
            )
        if set(counts) != set(original_labels) or any(
            counts[label] != self.shots for label in original_labels
        ):
            raise FullGradientDataError(
                f"source is not exactly {self.shots} records per base class: {counts}"
            )

        canonical_payload = {
            "schema_version": self.fingerprint_schema_version,
            "records": [record.fingerprint_entry() for record in self.records],
        }
        canonical = json.dumps(
            canonical_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        object.__setattr__(self, "fingerprint", hashlib.sha256(canonical).hexdigest())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice):
        return self.records[index]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(record.sample_id for record in self.records)

    @property
    def count_per_class(self) -> tuple[tuple[int, int], ...]:
        counts = Counter(record.original_label for record in self.records)
        return tuple((item.original_label, counts[item.original_label]) for item in self.base_classes)

    def to_datum_list(self) -> list[Datum]:
        """Return fresh pinned-Dassl Datum objects in canonical source order."""

        return [
            Datum(
                impath=str(record.image_path),
                label=record.training_label,
                domain=record.domain,
                classname=record.classname,
            )
            for record in self.records
        ]

    def as_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "dataset_name": self.dataset_name,
            "shots": self.shots,
            "seed": self.seed,
            "count": len(self),
            "ordering": self.ordering,
            "fingerprint_schema_version": self.fingerprint_schema_version,
            "fingerprint": self.fingerprint,
            "sample_ids": list(self.sample_ids),
            "count_per_class": [
                {"original_label": label, "count": count}
                for label, count in self.count_per_class
            ],
            "official_split_sha256": self.official_split_sha256,
            "fewshot_cache_sha256": self.fewshot_cache_sha256,
            "contains_transformed_tensors": False,
        }


class FullGradientDataset(Dataset):
    """Thin metadata-preserving delegate to pinned Dassl DatasetWrapper."""

    def __init__(self, cfg: Any, source: FullGradientSource, transform: Any):
        if not isinstance(source, FullGradientSource):
            raise FullGradientDataError("source must be FullGradientSource")
        if int(cfg.DATALOADER.K_TRANSFORMS) != 1:
            raise FullGradientDataError(
                "The primary full-gradient path requires K_TRANSFORMS=1"
            )
        self.source = source
        self.transform = transform
        self._delegate = DatasetWrapper(
            cfg,
            source.to_datum_list(),
            transform=transform,
            is_train=True,
        )

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.source[index]
        output = dict(self._delegate[index])
        if int(output["index"]) != record.position:
            raise FullGradientDataError("Pinned dataset wrapper changed source position")
        if int(output["label"]) != record.training_label:
            raise FullGradientDataError("Pinned dataset wrapper changed training label")
        output.update(
            {
                "sample_id": record.sample_id,
                "classname": record.classname,
                "original_label": record.original_label,
                "source_position": record.position,
                "dataset": record.dataset,
                "shots": record.shots,
                "seed": record.seed,
            }
        )
        return output


def load_full_gradient_source(
    loaded: LoadedDataset,
    manifest_path: Path,
) -> FullGradientSource:
    """Build Task-9 source from a Task-2 manifest without sampling or writing."""

    if not isinstance(loaded, LoadedDataset):
        raise FullGradientDataError("loaded must be a validated LoadedDataset")
    manifest_path = Path(manifest_path).resolve(strict=True)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FullGradientDataError(f"Cannot read Task-2 manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise FullGradientDataError("Task-2 manifest root must be an object")
    if manifest.get("schema_version") != DATA_MANIFEST_SCHEMA_VERSION:
        raise FullGradientDataError("Task-2 manifest schema mismatch")

    dataset_meta = _require_mapping(manifest, "dataset")
    if dataset_meta.get("name") != loaded.spec.canonical_name:
        raise FullGradientDataError("Manifest dataset does not match loaded dataset")
    if dataset_meta.get("image_root_relative") != loaded.spec.image_dir:
        raise FullGradientDataError("Manifest image root differs from pinned source")

    provenance = _require_mapping(manifest, "provenance")
    if provenance.get("coop_commit") != COOP_COMMIT:
        raise FullGradientDataError("Manifest CoOp provenance mismatch")
    if provenance.get("dassl_commit") != DASSL_COMMIT:
        raise FullGradientDataError("Manifest Dassl provenance mismatch")

    split_meta = _require_mapping(manifest, "official_split")
    if split_meta.get("filename") != loaded.spec.split_filename:
        raise FullGradientDataError("Manifest fixed-split filename mismatch")
    if split_meta.get("sha256") != loaded.split_sha256:
        raise FullGradientDataError("Manifest fixed-split hash mismatch")

    partition = _require_mapping(manifest, "class_partition")
    expected_base = [
        {"original_label": int(label), "classname": loaded.class_map[label]}
        for label in loaded.base_labels
    ]
    expected_new = [
        {"original_label": int(label), "classname": loaded.class_map[label]}
        for label in loaded.new_labels
    ]
    if partition.get("base_classes") != expected_base:
        raise FullGradientDataError("Manifest base-class partition mismatch")
    if partition.get("new_classes") != expected_new:
        raise FullGradientDataError("Manifest new-class partition mismatch")

    fewshot = _require_mapping(manifest, "few_shot")
    shots = _require_int(fewshot, "shots", minimum=1)
    seed = _require_int(fewshot, "seed", minimum=0)
    complete = _require_mapping(manifest, "complete_selected_source")
    if complete.get("ordering") != FULL_GRADIENT_SOURCE_ORDERING:
        raise FullGradientDataError("Manifest selected-source ordering mismatch")
    if complete.get("independent_of_normal_loader_state") is not True:
        raise FullGradientDataError("Manifest source is not loader-independent")
    raw_records = complete.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise FullGradientDataError("Manifest complete source records are missing")

    train_by_id: dict[str, Any] = {}
    for item in loaded.train:
        sample_id = stable_sample_id(item, loaded.dataset_root)
        if sample_id in train_by_id:
            raise FullGradientDataError(f"Duplicate official train identity: {sample_id}")
        train_by_id[sample_id] = item
    test_ids = {
        stable_sample_id(item, loaded.dataset_root) for item in loaded.test
    }
    base_training_label = {
        label: index for index, label in enumerate(loaded.base_labels)
    }
    base_classes = tuple(
        FullGradientClass(
            original_label=label,
            training_label=base_training_label[label],
            classname=loaded.class_map[label],
        )
        for label in loaded.base_labels
    )

    records: list[FullGradientRecord] = []
    for position, raw in enumerate(raw_records):
        if not isinstance(raw, dict) or set(raw) != {
            "sample_id",
            "original_label",
            "classname",
        }:
            raise FullGradientDataError("Malformed complete selected-source record")
        sample_id = raw["sample_id"]
        if not isinstance(sample_id, str):
            raise FullGradientDataError("Manifest sample_id must be a string")
        item = train_by_id.get(sample_id)
        if item is None:
            raise FullGradientDataError(
                f"Selected identity is not in the official train split: {sample_id}"
            )
        if sample_id in test_ids:
            raise FullGradientDataError(f"Selected identity is also test data: {sample_id}")
        original_label = raw["original_label"]
        if isinstance(original_label, bool) or not isinstance(original_label, int):
            raise FullGradientDataError("Manifest original_label must be an integer")
        if original_label not in base_training_label:
            raise FullGradientDataError(
                f"Manifest selected a novel-class record: {sample_id}"
            )
        if int(item.label) != original_label or str(item.classname) != raw["classname"]:
            raise FullGradientDataError(
                f"Manifest label/class differs from official split: {sample_id}"
            )
        records.append(
            FullGradientRecord(
                position=position,
                sample_id=sample_id,
                image_path=Path(item.impath).resolve(strict=True),
                original_label=original_label,
                training_label=base_training_label[original_label],
                classname=str(item.classname),
                domain=int(item.domain),
                dataset=loaded.spec.key,
                shots=shots,
                seed=seed,
            )
        )

    manifest_ids = fewshot.get("selected_sample_ids")
    if manifest_ids != [record.sample_id for record in records]:
        raise FullGradientDataError("Manifest few-shot IDs and complete source differ")
    if complete.get("count") != len(records):
        raise FullGradientDataError("Manifest complete-source count mismatch")
    if fewshot.get("total_selected_count") != len(records):
        raise FullGradientDataError("Manifest few-shot count mismatch")

    cache_meta = _require_mapping(fewshot, "cache")
    cache_relative = cache_meta.get("path_relative_to_dataset_root")
    if not isinstance(cache_relative, str):
        raise FullGradientDataError("Manifest cache path is missing")
    cache_path = _resolve_under_root(loaded.dataset_root.resolve(strict=True), cache_relative)
    if not cache_path.is_file():
        raise FullGradientDataError(f"Pinned few-shot cache is missing: {cache_path}")
    cache_hash = sha256_file(cache_path)
    if cache_meta.get("sha256") != cache_hash:
        raise FullGradientDataError("Pinned few-shot cache hash differs from manifest")
    _validate_cache_matches_records(loaded, cache_path, records)

    return FullGradientSource(
        dataset=loaded.spec.key,
        dataset_name=loaded.spec.canonical_name,
        dataset_root=loaded.dataset_root.resolve(strict=True),
        shots=shots,
        seed=seed,
        base_classes=base_classes,
        records=tuple(records),
        manifest_path=manifest_path,
        official_split_sha256=loaded.split_sha256,
        fewshot_cache_path=cache_path,
        fewshot_cache_sha256=cache_hash,
    )


def build_full_gradient_loader(
    cfg: Any,
    source: FullGradientSource,
    *,
    micro_batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    """Build a dedicated sequential loader; caller owns RNG isolation/seeding."""

    if (
        isinstance(micro_batch_size, bool)
        or not isinstance(micro_batch_size, int)
        or micro_batch_size <= 0
    ):
        raise FullGradientDataError("micro_batch_size must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise FullGradientDataError("num_workers must be an integer")
    if num_workers != 0:
        raise FullGradientDataError(
            "Scientific full-gradient loader requires num_workers=0"
        )
    # Scientific callers cannot inject an evaluation or substitute transform:
    # resolve the pinned training family through Dassl every time.
    transform = build_transform(cfg, is_train=True)
    dataset = FullGradientDataset(cfg, source, transform)
    sampler = SequentialSampler(dataset)
    # This generator belongs only to this dedicated loader. The loader never
    # derives/seeds it; Task-8 isolated_rng does that at sweep time.
    generator = torch.Generator(device="cpu")
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        pin_memory=bool(torch.cuda.is_available() and cfg.USE_CUDA),
        generator=generator,
    )
    if not isinstance(loader.sampler, SequentialSampler):
        raise FullGradientDataError("Dedicated loader did not retain SequentialSampler")
    if loader.drop_last or loader.num_workers != 0:
        raise FullGradientDataError("Dedicated loader mechanics changed unexpectedly")
    return loader


def describe_full_gradient_loader(loader: DataLoader) -> dict[str, object]:
    """Return stable inspectable Task-9 loader metadata."""

    if not isinstance(loader.sampler, SequentialSampler):
        raise FullGradientDataError("Full-gradient loader sampler is not sequential")
    if loader.drop_last or loader.num_workers != 0:
        raise FullGradientDataError("Full-gradient loader violates worker/drop policy")
    return {
        "sampler": type(loader.sampler).__name__,
        "shuffle": False,
        "drop_last": bool(loader.drop_last),
        "num_workers": int(loader.num_workers),
        "micro_batch_size": int(loader.batch_size),
        "sample_count": len(loader.dataset),
        "micro_batch_count": len(loader),
        "expected_micro_batch_count": math.ceil(
            len(loader.dataset) / int(loader.batch_size)
        ),
        "transform_applications_per_record": 1,
        "dedicated_generator": True,
        "generator_device": str(loader.generator.device),
    }


class FullGradientService:
    """Side-effect-free exact gradient of one complete selected data source.

    ``mean_loss_fn`` must return the mean loss for the supplied micro-batch.
    The production default implements pinned CoOp classification loss.  Each
    micro-batch is differentiated independently with ``torch.autograd.grad``;
    the service never invokes backward, an optimizer, or a scheduler.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        param_index: ParamIndex,
        loader: DataLoader,
        precision_controller: PrecisionController,
        protocol_seed: int,
        dataset: str,
        shots: int,
        config_hash: str,
        mean_loss_fn: Callable[[nn.Module, Any], torch.Tensor] | None = None,
        batch_size_fn: Callable[[Any], int] | None = None,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(param_index, ParamIndex):
            raise TypeError("param_index must be a ParamIndex")
        if not isinstance(loader, DataLoader):
            raise TypeError("loader must be a torch DataLoader")
        if not isinstance(precision_controller, PrecisionController):
            raise TypeError("precision_controller must be a PrecisionController")
        if not isinstance(loader.sampler, SequentialSampler):
            raise FullGradientServiceError(
                "Full-gradient service requires a SequentialSampler"
            )
        if loader.drop_last or loader.num_workers != 0:
            raise FullGradientServiceError(
                "Full-gradient service requires drop_last=False and num_workers=0"
            )
        if not isinstance(loader.generator, torch.Generator):
            raise FullGradientServiceError(
                "Full-gradient service requires its own explicit torch.Generator"
            )
        if (
            isinstance(loader.batch_size, bool)
            or not isinstance(loader.batch_size, int)
            or loader.batch_size <= 0
        ):
            raise FullGradientServiceError(
                "Full-gradient service requires a positive fixed micro-batch size"
            )
        if len(loader.dataset) <= 0:
            raise FullGradientServiceError("Full-gradient source must not be empty")
        if not isinstance(dataset, str) or not dataset:
            raise FullGradientServiceError("dataset must be a nonempty string")
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise FullGradientServiceError("shots must be a positive integer")
        if not isinstance(config_hash, str) or not config_hash:
            raise FullGradientServiceError("config_hash must be a nonempty string")
        if isinstance(protocol_seed, bool) or not isinstance(protocol_seed, int):
            raise FullGradientServiceError("protocol_seed must be an integer")
        if protocol_seed < 0:
            raise FullGradientServiceError("protocol_seed must be nonnegative")

        source = getattr(loader.dataset, "source", None)
        if source is not None:
            if not isinstance(source, FullGradientSource):
                raise FullGradientServiceError(
                    "loader.dataset.source is not a FullGradientSource"
                )
            if source.dataset != dataset or source.shots != shots:
                raise FullGradientServiceError(
                    "Service seed metadata differs from the dedicated source"
                )

        self.model = model
        self.param_index = param_index
        self.loader = loader
        self.precision_controller = precision_controller
        self.protocol_seed = protocol_seed
        self.dataset = dataset
        self.shots = shots
        self.config_hash = config_hash
        self.mean_loss_fn = mean_loss_fn or self._default_coop_mean_loss
        self.batch_size_fn = batch_size_fn or self._default_batch_size
        self.source = source
        self.source_fingerprint = source.fingerprint if source is not None else None

    def compute(self, *, optimizer_step: int, purpose: str) -> FullGradientResult:
        """Return the exact sample mean at current unperturbed parameters."""

        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
        ):
            raise FullGradientServiceError(
                "optimizer_step must be a zero-based nonnegative integer"
            )
        if not isinstance(purpose, str) or not purpose:
            raise FullGradientServiceError("purpose must be a nonempty string")

        self.param_index.assert_matches_model(self.model)
        trainable_snapshot = tuple(
            entry.parameter.detach().clone() for entry in self.param_index
        )
        mode_snapshot = tuple(module.training for module in self.model.modules())
        accumulator = GradientState.zeros(self.param_index)
        total_samples = len(self.loader.dataset)
        observed_samples = 0
        observed_batch_sizes: list[int] = []
        observed_sample_ids: list[str] = []
        weighted_loss = 0.0
        forward_calls = 0
        autograd_grad_calls = 0
        result: FullGradientResult | None = None

        cuda_devices = self._cuda_parameter_devices()
        self._synchronize_cuda(cuda_devices)
        started = time.perf_counter()
        body_error: BaseException | None = None
        try:
            with isolated_rng(
                protocol_seed=self.protocol_seed,
                dataset=self.dataset,
                shots=self.shots,
                config_hash=self.config_hash,
                optimizer_step=optimizer_step,
                purpose=purpose,
                explicit_generators=(self.loader.generator,),
            ) as derived_seed:
                for batch in self.loader:
                    batch_size = self.batch_size_fn(batch)
                    if (
                        isinstance(batch_size, bool)
                        or not isinstance(batch_size, int)
                        or batch_size <= 0
                    ):
                        raise FullGradientServiceError(
                            "batch_size_fn must return a positive integer"
                        )
                    if observed_samples + batch_size > total_samples:
                        raise FullGradientServiceError(
                            "Dedicated loader yielded more samples than its source"
                        )
                    observed_batch_sizes.append(batch_size)
                    self._collect_sample_ids(batch, observed_sample_ids, batch_size)

                    with self.precision_controller.autocast_context():
                        loss = self.mean_loss_fn(self.model, batch)
                    forward_calls += 1
                    self._validate_mean_loss(loss)
                    gradients = torch.autograd.grad(
                        loss,
                        self.param_index.parameters,
                        create_graph=False,
                        retain_graph=False,
                        allow_unused=False,
                    )
                    autograd_grad_calls += 1
                    micro_gradient = GradientState.from_tensors(
                        self.param_index, gradients
                    )
                    if not micro_gradient.is_finite():
                        raise FullGradientNumericalError(
                            "Exact micro-batch gradient contains NaN or Inf"
                        )

                    weight = batch_size / total_samples
                    accumulator.accumulate_(micro_gradient, weight=weight)
                    if not accumulator.is_finite():
                        raise FullGradientNumericalError(
                            "Exact FP32 gradient accumulator became NaN or Inf"
                        )
                    weighted_loss += weight * float(loss.detach().item())
                    observed_samples += batch_size

                    # Do not retain any micro-batch graph/tensor reference in the
                    # result or across the next iteration.
                    del gradients, micro_gradient, loss, batch

                if observed_samples != total_samples:
                    raise FullGradientServiceError(
                        f"Dedicated loader yielded {observed_samples}/{total_samples} samples"
                    )
                if len(observed_batch_sizes) != len(self.loader):
                    raise FullGradientServiceError(
                        "Observed micro-batch count differs from dedicated loader length"
                    )
                if self.source is not None:
                    if tuple(observed_sample_ids) != self.source.sample_ids:
                        raise FullGradientServiceError(
                            "Exact sweep identities/order differ from dedicated source"
                        )
                elif observed_sample_ids and len(observed_sample_ids) != total_samples:
                    raise FullGradientServiceError(
                        "Synthetic loader supplied incomplete sample identities"
                    )
                if observed_sample_ids and len(set(observed_sample_ids)) != total_samples:
                    raise FullGradientServiceError(
                        "Exact sweep contains duplicate sample identities"
                    )
                if not math.isfinite(weighted_loss):
                    raise FullGradientNumericalError(
                        "Exact dataset-mean loss is NaN or Inf"
                    )

            self._synchronize_cuda(cuda_devices)
            elapsed_s = time.perf_counter() - started
            metadata = FullGradientSweepMetadata(
                sample_count=observed_samples,
                micro_batch_count=len(observed_batch_sizes),
                configured_micro_batch_size=int(self.loader.batch_size),
                observed_micro_batch_sizes=tuple(observed_batch_sizes),
                forward_calls=forward_calls,
                autograd_grad_calls=autograd_grad_calls,
                mean_loss=weighted_loss,
                elapsed_s=elapsed_s,
                precision_mode=self.precision_controller.mode,
                param_index_fingerprint=self.param_index.fingerprint,
                source_fingerprint=self.source_fingerprint,
                seed=derived_seed,
            )
            result = FullGradientResult(
                gradient=accumulator.clone(),
                metadata=metadata,
            )
        except BaseException as error:
            body_error = error
            raise
        finally:
            contamination = self._contamination_messages(
                trainable_snapshot, mode_snapshot
            )
            if contamination:
                message = "; ".join(contamination)
                if body_error is None:
                    raise FullGradientServiceError(message)
                if hasattr(body_error, "add_note"):
                    body_error.add_note(f"Full-gradient purity violation: {message}")

        if result is None:  # Defensive guard for static/type reasoning.
            raise FullGradientServiceError("Exact-gradient query produced no result")
        if not result.gradient.is_finite():
            raise FullGradientNumericalError("Exact dataset-mean gradient is nonfinite")
        return result

    def _default_coop_mean_loss(
        self, model: nn.Module, batch: Any
    ) -> torch.Tensor:
        if not isinstance(batch, Mapping):
            raise FullGradientServiceError(
                "Default CoOp loss requires a mapping batch"
            )
        image = batch.get("img")
        label = batch.get("label")
        if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
            raise FullGradientServiceError(
                "Default CoOp loss requires tensor 'img' and 'label' fields"
            )
        devices = {entry.parameter.device for entry in self.param_index}
        if len(devices) != 1:
            raise FullGradientServiceError(
                "Default CoOp batch transfer requires one trainable-parameter device"
            )
        device = next(iter(devices))
        logits = model(image.to(device),)
        return F.cross_entropy(logits, label.to(device))

    @staticmethod
    def _default_batch_size(batch: Any) -> int:
        if isinstance(batch, Mapping):
            for key in ("label", "img", "sample_id"):
                value = batch.get(key)
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    return int(value.shape[0])
                if isinstance(value, (list, tuple)):
                    return len(value)
        if isinstance(batch, (list, tuple)) and batch:
            value = batch[0]
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
        raise FullGradientServiceError(
            "Cannot derive micro-batch size; supply batch_size_fn"
        )

    @staticmethod
    def _validate_mean_loss(loss: object) -> None:
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise FullGradientServiceError(
                "mean_loss_fn must return one scalar tensor"
            )
        if not loss.is_floating_point():
            raise FullGradientServiceError("mean loss must have floating dtype")
        if not bool(torch.isfinite(loss.detach()).item()):
            raise FullGradientNumericalError(
                "Exact micro-batch mean loss contains NaN or Inf"
            )

    @staticmethod
    def _collect_sample_ids(
        batch: Any, target: list[str], batch_size: int
    ) -> None:
        if not isinstance(batch, Mapping) or "sample_id" not in batch:
            return
        raw = batch["sample_id"]
        if not isinstance(raw, (list, tuple)) or len(raw) != batch_size:
            raise FullGradientServiceError(
                "Collated sample_id field differs from micro-batch size"
            )
        target.extend(str(item) for item in raw)

    def _contamination_messages(
        self,
        trainable_snapshot: tuple[torch.Tensor, ...],
        mode_snapshot: tuple[bool, ...],
    ) -> list[str]:
        messages = []
        for entry, before in zip(self.param_index, trainable_snapshot):
            if not torch.equal(entry.parameter.detach(), before):
                messages.append(
                    f"trainable parameter changed during query: {entry.name}"
                )
        after_modes = tuple(module.training for module in self.model.modules())
        if after_modes != mode_snapshot:
            messages.append("model/module training modes changed during query")
        return messages

    def _cuda_parameter_devices(self) -> tuple[torch.device, ...]:
        devices = sorted(
            {
                entry.parameter.device
                for entry in self.param_index
                if entry.parameter.device.type == "cuda"
            },
            key=lambda device: device.index if device.index is not None else 0,
        )
        return tuple(devices)

    @staticmethod
    def _synchronize_cuda(devices: Sequence[torch.device]) -> None:
        for device in devices:
            torch.cuda.synchronize(device)


def iter_batch_sample_ids(loader: DataLoader) -> Iterator[tuple[str, ...]]:
    """Yield only portable IDs while still executing the real loader/transform."""

    for batch in loader:
        sample_ids = batch.get("sample_id")
        if not isinstance(sample_ids, (list, tuple)):
            raise FullGradientDataError("Collated batch is missing portable sample IDs")
        yield tuple(str(sample_id) for sample_id in sample_ids)


def _validate_cache_matches_records(
    loaded: LoadedDataset,
    cache_path: Path,
    records: Sequence[FullGradientRecord],
) -> None:
    try:
        with cache_path.open("rb") as stream:
            cached = pickle.load(stream)
    except (OSError, pickle.UnpicklingError) as error:
        raise FullGradientDataError(f"Cannot read pinned cache: {cache_path}") from error
    if not isinstance(cached, dict) or set(cached) != {"train", "val"}:
        raise FullGradientDataError("Malformed pinned few-shot cache")
    cached_train = list(cached["train"])
    cached_base, = OxfordPets.subsample_classes(cached_train, subsample="base")
    cache_records = sorted(
        [
            (
                stable_sample_id(cached_train_item, loaded.dataset_root),
                int(cached_train_item.label),
                str(cached_train_item.classname),
            )
            for cached_train_item in cached_train
            if int(cached_train_item.label) in set(loaded.base_labels)
        ],
        key=lambda row: (row[1], row[0]),
    )
    expected = [
        (record.sample_id, record.original_label, record.classname) for record in records
    ]
    if cache_records != expected:
        raise FullGradientDataError("Pinned cache selection differs from Task-2 manifest")
    if len(cached_base) != len(records):
        raise FullGradientDataError("Pinned base-subsampled cache count differs from source")


def _validate_portable_sample_id(sample_id: str) -> PurePosixPath:
    if "\\" in sample_id:
        raise FullGradientDataError("sample_id must use POSIX separators")
    path = PurePosixPath(sample_id)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise FullGradientDataError(f"sample_id is not a safe relative path: {sample_id}")
    return path


def _resolve_under_root(root: Path, relative: str) -> Path:
    portable = _validate_portable_sample_id(relative)
    try:
        candidate = root.joinpath(*portable.parts).resolve(strict=True)
    except FileNotFoundError as error:
        raise FullGradientDataError(
            f"Required dataset-relative path is missing: {relative}"
        ) from error
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise FullGradientDataError(f"Path escapes dataset root: {relative}") from error
    return candidate


def _require_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise FullGradientDataError(f"Manifest field {key!r} must be an object")
    return value


def _require_int(parent: dict[str, Any], key: str, *, minimum: int) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FullGradientDataError(
            f"Manifest field {key!r} must be an integer >= {minimum}"
        )
    return value
