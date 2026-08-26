"""Scientific data-protocol validation for the pinned CoOp datasets.

This module deliberately checks the fixed split before calling any CoOp
dataset constructor. DTD and EuroSAT constructors generate a replacement
random split when the JSON is absent, which is forbidden for scientific use.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import pickle
import random
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import yaml

from dassl.data.datasets import DatasetBase
from datasets.oxford_pets import OxfordPets


SCHEMA_VERSION = "sample_fg.data_manifest.v1"
COOP_COMMIT = "ff61507c790454bce7c5052c3ac39e60772f1f89"
DASSL_COMMIT = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ProtocolValidationError(RuntimeError):
    """Raised when benchmark data violate the pinned scientific protocol."""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    canonical_name: str
    registry_name: str
    dataset_dir: str
    image_dir: str
    split_filename: str
    split_drive_id: str
    pinned_raw_url: str
    local_acquisition_url: str
    archive_filename: str
    expected_total_classes: int
    expected_base_classes: int
    expected_new_classes: int
    expected_image_count: int
    expected_split_counts: tuple[int, int, int]


DATASET_SPECS = {
    "dtd": DatasetSpec(
        key="dtd",
        canonical_name="DTD",
        registry_name="DescribableTextures",
        dataset_dir="dtd",
        image_dir="images",
        split_filename="split_zhou_DescribableTextures.json",
        split_drive_id="1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x",
        pinned_raw_url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        local_acquisition_url="https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        archive_filename="dtd-r1.0.1.tar.gz",
        expected_total_classes=47,
        expected_base_classes=24,
        expected_new_classes=23,
        expected_image_count=5640,
        expected_split_counts=(2820, 1128, 1692),
    ),
    "eurosat": DatasetSpec(
        key="eurosat",
        canonical_name="EuroSAT",
        registry_name="EuroSAT",
        dataset_dir="eurosat",
        image_dir="2750",
        split_filename="split_zhou_EuroSAT.json",
        split_drive_id="1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o",
        pinned_raw_url="http://madm.dfki.de/files/sentinel/EuroSAT.zip",
        local_acquisition_url="https://madm.dfki.de/files/sentinel/EuroSAT.zip",
        archive_filename="EuroSAT.zip",
        expected_total_classes=10,
        expected_base_classes=5,
        expected_new_classes=5,
        expected_image_count=27000,
        expected_split_counts=(13500, 5400, 8100),
    ),
}


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    data_root: Path
    dataset_root: Path
    image_root: Path
    split_path: Path
    split_sha256: str
    archive_path: Path
    archive_size: int
    archive_sha256: str
    archive_timestamp_utc: str
    train: list[Any]
    val: list[Any]
    test: list[Any]
    class_map: dict[int, str]
    base_labels: tuple[int, ...]
    new_labels: tuple[int, ...]
    image_count: int
    class_directory_count: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require_official_split(dataset_root: Path, split_filename: str) -> Path:
    split_path = dataset_root / split_filename
    if not split_path.is_file():
        raise FileNotFoundError(
            "Scientific mode requires the fixed CoOp split before dataset "
            f"construction: {split_path}. Refusing upstream random split generation."
        )
    return split_path


def _read_split_json(split_path: Path) -> dict[str, list[Any]]:
    with split_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if set(payload) != {"train", "val", "test"}:
        raise ProtocolValidationError(
            f"Split must contain exactly train/val/test keys: {split_path}"
        )
    for split_name, entries in payload.items():
        if not isinstance(entries, list):
            raise ProtocolValidationError(f"Split {split_name} is not a list")
        for entry in entries:
            if not isinstance(entry, list) or len(entry) != 3:
                raise ProtocolValidationError(
                    f"Malformed {split_name} record in {split_path}: {entry!r}"
                )
    return payload


def stable_sample_id(item: Any, dataset_root: Path) -> str:
    image_path = Path(item.impath).resolve(strict=True)
    root = dataset_root.resolve(strict=True)
    try:
        relative = image_path.relative_to(root)
    except ValueError as exc:
        raise ProtocolValidationError(
            f"Sample escapes dataset root: {image_path} not under {root}"
        ) from exc
    return relative.as_posix()


def _item_signature(items: Sequence[Any], dataset_root: Path) -> list[tuple[str, int, str]]:
    return [
        (stable_sample_id(item, dataset_root), int(item.label), str(item.classname))
        for item in items
    ]


def _derive_class_map(splits: Iterable[Sequence[Any]]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for records in splits:
        for item in records:
            label = int(item.label)
            classname = str(item.classname)
            previous = mapping.setdefault(label, classname)
            if previous != classname:
                raise ProtocolValidationError(
                    f"Label {label} maps to both {previous!r} and {classname!r}"
                )
    return dict(sorted(mapping.items()))


def _validate_partition(
    class_map: dict[int, str], spec: DatasetSpec
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    labels = tuple(sorted(class_map))
    midpoint = math.ceil(len(labels) / 2)
    base_labels = labels[:midpoint]
    new_labels = labels[midpoint:]
    if len(labels) != spec.expected_total_classes:
        raise ProtocolValidationError(
            f"{spec.canonical_name}: observed {len(labels)} classes, "
            f"expected {spec.expected_total_classes}"
        )
    if len(base_labels) != spec.expected_base_classes or len(new_labels) != spec.expected_new_classes:
        raise ProtocolValidationError(
            f"{spec.canonical_name}: observed base/new "
            f"{len(base_labels)}/{len(new_labels)}, expected "
            f"{spec.expected_base_classes}/{spec.expected_new_classes}"
        )
    if set(base_labels) & set(new_labels) or set(base_labels) | set(new_labels) != set(labels):
        raise ProtocolValidationError("Base/new label partition is not disjoint and complete")
    base_names = {class_map[label] for label in base_labels}
    new_names = {class_map[label] for label in new_labels}
    if base_names & new_names:
        raise ProtocolValidationError("Base/new class-name sets overlap")
    return base_labels, new_labels


def _validate_split_leakage(
    train: Sequence[Any], test: Sequence[Any], dataset_root: Path
) -> None:
    train_ids = {stable_sample_id(item, dataset_root) for item in train}
    test_ids = {stable_sample_id(item, dataset_root) for item in test}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ProtocolValidationError(
            f"Train/test identity leakage detected ({len(overlap)}): {overlap[:5]}"
        )


def load_dataset(data_root: Path, spec: DatasetSpec) -> LoadedDataset:
    data_root = data_root.resolve(strict=True)
    dataset_root = data_root / spec.dataset_dir
    image_root = dataset_root / spec.image_dir
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root is missing: {dataset_root}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Dataset image root is missing: {image_root}")

    # This check must precede any CoOp dataset construction or cache directory creation.
    split_path = require_official_split(dataset_root, spec.split_filename)
    raw_split = _read_split_json(split_path)

    # Use the pinned CoOp split reader and Datum representation verbatim.
    train, val, test = OxfordPets.read_split(str(split_path), str(image_root))
    observed_counts = (len(train), len(val), len(test))
    if observed_counts != spec.expected_split_counts:
        raise ProtocolValidationError(
            f"{spec.canonical_name}: observed split counts {observed_counts}, "
            f"expected {spec.expected_split_counts}"
        )
    if observed_counts != tuple(len(raw_split[name]) for name in ("train", "val", "test")):
        raise ProtocolValidationError("Pinned CoOp reader count differs from JSON count")

    class_map = _derive_class_map((train, val, test))
    train_labels = {int(item.label) for item in train}
    if train_labels != set(class_map):
        raise ProtocolValidationError("Training split does not cover every class")
    base_labels, new_labels = _validate_partition(class_map, spec)
    _validate_split_leakage(train, test, dataset_root)

    image_count = sum(
        1
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    class_directory_count = sum(1 for path in image_root.iterdir() if path.is_dir())
    if image_count != spec.expected_image_count:
        raise ProtocolValidationError(
            f"{spec.canonical_name}: observed {image_count} images, "
            f"expected {spec.expected_image_count}"
        )
    if class_directory_count != spec.expected_total_classes:
        raise ProtocolValidationError(
            f"{spec.canonical_name}: observed {class_directory_count} class directories, "
            f"expected {spec.expected_total_classes}"
        )

    archive_path = data_root / "_archives" / spec.archive_filename
    if not archive_path.is_file():
        raise FileNotFoundError(f"Retained raw archive is missing: {archive_path}")
    archive_stat = archive_path.stat()

    return LoadedDataset(
        spec=spec,
        data_root=data_root,
        dataset_root=dataset_root,
        image_root=image_root,
        split_path=split_path,
        split_sha256=sha256_file(split_path),
        archive_path=archive_path,
        archive_size=archive_stat.st_size,
        archive_sha256=sha256_file(archive_path),
        archive_timestamp_utc=_utc_timestamp(archive_stat.st_mtime),
        train=train,
        val=val,
        test=test,
        class_map=class_map,
        base_labels=base_labels,
        new_labels=new_labels,
        image_count=image_count,
        class_directory_count=class_directory_count,
    )


def _utc_timestamp(seconds_since_epoch: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(seconds_since_epoch, tz=timezone.utc).isoformat()


@contextmanager
def _preserve_python_random_state():
    state = random.getstate()
    try:
        yield
    finally:
        random.setstate(state)


def select_upstream_fewshot(
    train: Sequence[Any], val: Sequence[Any], shots: int, seed: int
) -> tuple[list[Any], list[Any]]:
    if shots < 1:
        raise ValueError("shots must be positive")
    with _preserve_python_random_state():
        # Pinned CoOp calls set_random_seed(seed) before dataset construction.
        # Python random is the RNG actually consumed by Dassl's sampler.
        random.seed(seed)
        helper = DatasetBase(train_x=list(train))
        selected_train = helper.generate_fewshot_dataset(
            list(train), num_shots=shots
        )
        selected_val = helper.generate_fewshot_dataset(
            list(val), num_shots=min(shots, 4)
        )
    return list(selected_train), list(selected_val)


def ensure_upstream_cache(
    loaded: LoadedDataset, shots: int, seed: int
) -> tuple[list[Any], list[Any], Path, str, str]:
    fresh_train, fresh_val = select_upstream_fewshot(
        loaded.train, loaded.val, shots, seed
    )
    cache_dir = loaded.dataset_root / "split_fewshot"
    cache_path = cache_dir / f"shot_{shots}-seed_{seed}.pkl"
    existed_before = cache_path.exists()

    if existed_before:
        with cache_path.open("rb") as stream:
            cached = pickle.load(stream)
        if not isinstance(cached, dict) or set(cached) != {"train", "val"}:
            raise ProtocolValidationError(f"Malformed upstream cache: {cache_path}")
        cached_train = list(cached["train"])
        cached_val = list(cached["val"])
        if _item_signature(cached_train, loaded.dataset_root) != _item_signature(
            fresh_train, loaded.dataset_root
        ) or _item_signature(cached_val, loaded.dataset_root) != _item_signature(
            fresh_val, loaded.dataset_root
        ):
            raise ProtocolValidationError(
                f"Existing cache disagrees with a fresh pinned-upstream selection: {cache_path}"
            )
        action = "reused_verified"
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        if temporary.exists():
            raise ProtocolValidationError(
                f"Refusing to overwrite stale cache temporary: {temporary}"
            )
        with temporary.open("wb") as stream:
            pickle.dump(
                {"train": fresh_train, "val": fresh_val},
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        if cache_path.exists():
            raise ProtocolValidationError(f"Cache appeared concurrently: {cache_path}")
        os.replace(temporary, cache_path)
        cached_train, cached_val = fresh_train, fresh_val
        action = "generated"

    return cached_train, cached_val, cache_path, sha256_file(cache_path), action


def _class_entries(labels: Sequence[int], class_map: dict[int, str]) -> list[dict[str, Any]]:
    return [
        {"original_label": int(label), "classname": class_map[int(label)]}
        for label in labels
    ]


def _record_entry(item: Any, dataset_root: Path) -> dict[str, Any]:
    return {
        "sample_id": stable_sample_id(item, dataset_root),
        "original_label": int(item.label),
        "classname": str(item.classname),
    }


def _loader_and_transform_metadata(
    coop_root: Path, selected_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = coop_root / "configs" / "trainers" / "CoOp" / "vit_b16_ctxv1.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    batch_size = int(config["DATALOADER"]["TRAIN_X"]["BATCH_SIZE"])
    workers = int(config["DATALOADER"]["NUM_WORKERS"])
    drop_last = selected_count >= batch_size
    steps = selected_count // batch_size if drop_last else math.ceil(selected_count / batch_size)
    consumed = steps * batch_size if drop_last else selected_count
    raw_input_size = config["INPUT"]["SIZE"]
    if isinstance(raw_input_size, str):
        raw_input_size = ast.literal_eval(raw_input_size)
    input_size = [int(value) for value in raw_input_size]
    if len(input_size) != 2 or any(value <= 0 for value in input_size):
        raise ProtocolValidationError(
            f"Invalid resolved CoOp input size: {raw_input_size!r}"
        )
    loader = {
        "sampler": "RandomSampler",
        "sampler_source": "pinned Dassl defaults.py",
        "drop_last": drop_last,
        "drop_last_rule": "is_train and len(data_source) >= batch_size",
        "batch_size": batch_size,
        "workers": workers,
        "steps_per_epoch": steps,
        "samples_consumed_per_epoch": consumed,
        "complete_selected_source_count": selected_count,
    }
    input_cfg = config["INPUT"]
    transforms = {
        "config_source": "configs/trainers/CoOp/vit_b16_ctxv1.yaml",
        "input_size": input_size,
        "interpolation": input_cfg["INTERPOLATION"],
        "pixel_mean": list(input_cfg["PIXEL_MEAN"]),
        "pixel_std": list(input_cfg["PIXEL_STD"]),
        "configured_train_choices": list(input_cfg["TRANSFORMS"]),
        "train_pipeline": [
            {
                "name": "RandomResizedCrop",
                "size": input_size,
                "scale": [0.08, 1.0],
                "interpolation": input_cfg["INTERPOLATION"],
            },
            {"name": "RandomHorizontalFlip"},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "mean": list(input_cfg["PIXEL_MEAN"]),
                "std": list(input_cfg["PIXEL_STD"]),
            },
        ],
        "test_pipeline": [
            {"name": "ResizeSmallerEdge", "size": max(input_size)},
            {"name": "CenterCrop", "size": input_size},
            {"name": "ToTensor"},
            {
                "name": "Normalize",
                "mean": list(input_cfg["PIXEL_MEAN"]),
                "std": list(input_cfg["PIXEL_STD"]),
            },
        ],
    }
    return loader, transforms


def _cross_check_upstream_dataset_constructor(
    loaded: LoadedDataset,
    shots: int,
    seed: int,
    selected_ids: Sequence[str],
) -> int:
    from types import SimpleNamespace

    if loaded.spec.key == "dtd":
        from datasets.dtd import DescribableTextures as dataset_class
    elif loaded.spec.key == "eurosat":
        from datasets.eurosat import EuroSAT as dataset_class
    else:
        return -1
    cfg = SimpleNamespace(
        SEED=seed,
        DATASET=SimpleNamespace(
            ROOT=str(loaded.data_root),
            NUM_SHOTS=shots,
            SUBSAMPLE_CLASSES="base",
        ),
    )
    upstream_dataset = dataset_class(cfg)
    upstream_ids = [
        stable_sample_id(item, loaded.dataset_root)
        for item in upstream_dataset.train_x
    ]
    if upstream_ids != list(selected_ids):
        raise ProtocolValidationError(
            "Actual pinned CoOp dataset constructor disagrees with validated cache selection"
        )
    return len(upstream_dataset.test)


def validate_cell(
    loaded: LoadedDataset,
    shots: int,
    seed: int,
    coop_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cached_train, _, cache_path, cache_sha256, cache_action = ensure_upstream_cache(
        loaded, shots, seed
    )
    base_label_set = set(loaded.base_labels)
    selected = [item for item in cached_train if int(item.label) in base_label_set]
    selected_ids = [stable_sample_id(item, loaded.dataset_root) for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ProtocolValidationError(
            f"{loaded.spec.canonical_name} shots={shots} seed={seed}: duplicate selected identity"
        )
    train_ids = {stable_sample_id(item, loaded.dataset_root) for item in loaded.train}
    test_ids = {stable_sample_id(item, loaded.dataset_root) for item in loaded.test}
    if not set(selected_ids) <= train_ids:
        raise ProtocolValidationError("Few-shot selection contains a non-training identity")
    if set(selected_ids) & test_ids:
        raise ProtocolValidationError("Few-shot selection contains a test identity")
    if any(int(item.label) not in base_label_set for item in selected):
        raise ProtocolValidationError("Few-shot selection contains a new-class label")

    per_class = Counter(int(item.label) for item in selected)
    if set(per_class) != base_label_set:
        raise ProtocolValidationError("Few-shot selection does not cover every base class")
    if any(count != shots for count in per_class.values()):
        raise ProtocolValidationError(
            f"Few-shot selection is not exactly {shots} examples per base class: {per_class}"
        )
    derived_expected_total = len(loaded.base_labels) * shots
    specification_expected_total = loaded.spec.expected_base_classes * shots
    if len(selected) != derived_expected_total or len(selected) != specification_expected_total:
        raise ProtocolValidationError(
            f"Selected count {len(selected)} differs from derived/specification totals "
            f"{derived_expected_total}/{specification_expected_total}"
        )

    deterministic_train, _ = select_upstream_fewshot(
        loaded.train, loaded.val, shots, seed
    )
    deterministic_base = [
        item for item in deterministic_train if int(item.label) in base_label_set
    ]
    if _item_signature(selected, loaded.dataset_root) != _item_signature(
        deterministic_base, loaded.dataset_root
    ):
        raise ProtocolValidationError("Same-seed pinned-upstream selection is not deterministic")

    # Validate pinned relabeling mechanics without using relabeled values as provenance.
    relabeled_base, = OxfordPets.subsample_classes(
        list(cached_train), subsample="base"
    )
    if {stable_sample_id(item, loaded.dataset_root) for item in relabeled_base} != set(selected_ids):
        raise ProtocolValidationError("Pinned CoOp base subsampling disagrees with original-label filtering")

    stable_records = sorted(
        (_record_entry(item, loaded.dataset_root) for item in selected),
        key=lambda record: (record["original_label"], record["sample_id"]),
    )
    loader, transforms = _loader_and_transform_metadata(coop_root, len(stable_records))
    if loader["complete_selected_source_count"] != len(stable_records):
        raise ProtocolValidationError("Complete selected source was altered by loader semantics")
    upstream_base_test_count = _cross_check_upstream_dataset_constructor(
        loaded, shots, seed, selected_ids
    )

    counts = [
        {
            "original_label": label,
            "classname": loaded.class_map[label],
            "count": per_class[label],
        }
        for label in loaded.base_labels
    ]
    cache_relative = cache_path.relative_to(loaded.dataset_root).as_posix()
    archive_relative = loaded.archive_path.relative_to(loaded.data_root).as_posix()
    acquisition_deviation = None
    if loaded.spec.pinned_raw_url != loaded.spec.local_acquisition_url:
        acquisition_deviation = {
            "type": "transport-level",
            "pinned_documented_url": loaded.spec.pinned_raw_url,
            "observed_failure": "HTTP 403 Forbidden",
            "local_acquisition_url": loaded.spec.local_acquisition_url,
            "rationale": (
                "same official DFKI host and exact /files/sentinel/EuroSAT.zip "
                "path; HTTPS substituted after the documented HTTP endpoint rejected the request"
            ),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "name": loaded.spec.canonical_name,
            "registry_name": loaded.spec.registry_name,
            "root_absolute": str(loaded.dataset_root),
            "image_root_relative": loaded.spec.image_dir,
            "image_count": loaded.image_count,
            "expected_image_count": loaded.spec.expected_image_count,
            "class_directory_count": loaded.class_directory_count,
            "expected_class_directory_count": loaded.spec.expected_total_classes,
        },
        "provenance": {
            "coop_commit": COOP_COMMIT,
            "dassl_commit": DASSL_COMMIT,
            "protocol_source": "pinned KaiyangZhou/CoOp",
            "fewshot_source": "dassl.data.datasets.DatasetBase.generate_fewshot_dataset",
            "class_partition_source": "datasets.oxford_pets.OxfordPets.subsample_classes",
            "sample_identity_schema": "dataset-root-relative POSIX path v1",
        },
        "raw_archive": {
            "pinned_documented_url": loaded.spec.pinned_raw_url,
            "local_acquisition_url": loaded.spec.local_acquisition_url,
            "filename": loaded.spec.archive_filename,
            "path_relative_to_data_root": archive_relative,
            "byte_size": loaded.archive_size,
            "sha256": loaded.archive_sha256,
            "checksum_status": "observed_local_not_externally_verified",
            "download_timestamp_utc": loaded.archive_timestamp_utc,
            "acquisition_deviation": acquisition_deviation,
        },
        "official_split": {
            "filename": loaded.spec.split_filename,
            "source_drive_id": loaded.spec.split_drive_id,
            "sha256": loaded.split_sha256,
            "counts": {
                "train": len(loaded.train),
                "val": len(loaded.val),
                "test": len(loaded.test),
            },
            "expected_counts": {
                "train": loaded.spec.expected_split_counts[0],
                "val": loaded.spec.expected_split_counts[1],
                "test": loaded.spec.expected_split_counts[2],
            },
        },
        "class_partition": {
            "algorithm": "sorted original labels; first ceil(C/2) base, remainder new",
            "total_class_count": len(loaded.class_map),
            "expected_total_class_count": loaded.spec.expected_total_classes,
            "expected_base_class_count": loaded.spec.expected_base_classes,
            "expected_new_class_count": loaded.spec.expected_new_classes,
            "base_classes": _class_entries(loaded.base_labels, loaded.class_map),
            "new_classes": _class_entries(loaded.new_labels, loaded.class_map),
        },
        "few_shot": {
            "shots": shots,
            "seed": seed,
            "selected_count_per_class": counts,
            "total_selected_count": len(stable_records),
            "derived_expected_total_selected_count": derived_expected_total,
            "specification_expected_total_selected_count": specification_expected_total,
            "selected_sample_ids": [record["sample_id"] for record in stable_records],
            "cache": {
                "path_relative_to_dataset_root": cache_relative,
                "sha256": cache_sha256,
                "format": "pinned CoOp pickle with train/val Datum lists",
            },
        },
        "complete_selected_source": {
            "ordering": "original_label_then_sample_id_v1",
            "count": len(stable_records),
            "records": stable_records,
            "independent_of_normal_loader_state": True,
        },
        "normal_train_loader": loader,
        "transforms": transforms,
        "integrity": {
            "train_test_overlap_count": 0,
            "base_new_label_overlap_count": 0,
            "base_new_classname_overlap_count": 0,
            "selected_duplicate_identity_count": 0,
            "selected_new_class_count": 0,
            "same_seed_deterministic": True,
            "fixed_split_required_before_dataset_construction": True,
            "upstream_dataset_constructor_crosscheck": True,
            "upstream_base_test_count": upstream_base_test_count,
        },
    }
    summary = {
        "dataset": loaded.spec.canonical_name,
        "shots": shots,
        "seed": seed,
        "selected_count": len(stable_records),
        "expected_count": derived_expected_total,
        "per_class_count_pass": True,
        "cache_path": str(cache_path),
        "cache_sha256": cache_sha256,
        "cache_action": cache_action,
        "status": "PASS",
    }
    return manifest, summary


def write_deterministic_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def validate_matrix(
    data_root: Path,
    dataset_keys: Sequence[str],
    shots_values: Sequence[int],
    seeds: Sequence[int],
    output_root: Path,
    coop_root: Path,
) -> dict[str, Any]:
    loaded_datasets = {
        key: load_dataset(data_root, DATASET_SPECS[key]) for key in dataset_keys
    }
    cells: list[dict[str, Any]] = []
    for key in dataset_keys:
        loaded = loaded_datasets[key]
        for shots in shots_values:
            for seed in seeds:
                manifest, summary = validate_cell(loaded, shots, seed, coop_root)
                manifest_path = (
                    output_root
                    / key
                    / f"shots_{shots}"
                    / f"seed_{seed}"
                    / "data_manifest.json"
                )
                write_deterministic_json(manifest_path, manifest)
                summary["manifest_path"] = str(manifest_path)
                cells.append(summary)
    return {
        "schema_version": "sample_fg.data_validation_summary.v1",
        "data_root": str(data_root.resolve()),
        "cell_count": len(cells),
        "cells": cells,
        "status": "PASS",
    }
