"""Validate Task-9 selected-record sources and dedicated data loaders only."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dassl.config import get_cfg_default
from dassl.data.data_manager import DatasetWrapper, build_data_loader
from train import extend_cfg

from sample_fg.data_protocol import DATASET_SPECS, load_dataset, sha256_file
from sample_fg.full_gradient import (
    FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION,
    FULL_GRADIENT_SOURCE_SCHEMA_VERSION,
    FullGradientDataset,
    build_full_gradient_loader,
    describe_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.rng import capture_rng_state, isolated_rng, restore_rng_state


TASK8_COOP_SHA = "ff5339fdc2a125008d06df8a5a3b8468abd3a007"
COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
SHOTS = (4, 8, 16)
SEEDS = (1, 2, 3)
EXPECTED_COUNTS = {
    "dtd": {4: 96, 8: 192, 16: 384},
    "eurosat": {4: 20, 8: 40, 16: 80},
}
VALIDATION_CONFIG_HASH = "task9-full-gradient-data-path-v1"


def _run(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _assert_preflight() -> dict[str, Any]:
    branch = _run(["git", "branch", "--show-current"], REPO_ROOT)
    head = _run(["git", "rev-parse", "HEAD"], REPO_ROOT)
    dirty = _run(["git", "status", "--short"], REPO_ROOT)
    dassl_root = REPO_ROOT.parent / "Dassl.pytorch"
    dassl_head = _run(["git", "rev-parse", "HEAD"], dassl_root)
    dassl_dirty = _run(["git", "status", "--short"], dassl_root)
    if branch != "sample-full-gradient":
        raise AssertionError(f"Unexpected branch: {branch}")
    merge_base = _run(
        ["git", "merge-base", TASK8_COOP_SHA, head], REPO_ROOT
    )
    if merge_base != TASK8_COOP_SHA:
        raise AssertionError(
            f"Runtime HEAD {head} does not descend from Task-8 {TASK8_COOP_SHA}"
        )
    if head != TASK8_COOP_SHA:
        committed_paths = set(
            _run(
                ["git", "diff", "--name-only", f"{TASK8_COOP_SHA}..{head}"],
                REPO_ROOT,
            ).splitlines()
        )
        expected_paths = {
            "sample_fg/full_gradient.py",
            "scripts/validate_full_gradient_data_path.py",
            "tests/test_full_gradient_data.py",
        }
        if committed_paths != expected_paths:
            raise AssertionError(
                f"Unexpected committed paths since Task 8: {committed_paths}"
            )
        commit_count = int(
            _run(
                ["git", "rev-list", "--count", f"{TASK8_COOP_SHA}..{head}"],
                REPO_ROOT,
            )
        )
        subject = _run(["git", "log", "-1", "--format=%s"], REPO_ROOT)
        if commit_count != 1 or subject != "feat: add dedicated full-gradient data source":
            raise AssertionError(
                f"Unexpected Task-9 commit boundary: count={commit_count}, subject={subject!r}"
            )
    if dirty:
        # The validator is intentionally run before the Task-9 commit, so only
        # the known Task-9 implementation paths may be uncommitted.
        paths = set(
            _run(["git", "diff", "--name-only"], REPO_ROOT).splitlines()
        )
        paths.update(
            _run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                REPO_ROOT,
            ).splitlines()
        )
        paths = {path.replace("\\", "/") for path in paths if path}
        allowed = {
            "sample_fg/full_gradient.py",
            "scripts/validate_full_gradient_data_path.py",
            "tests/test_full_gradient_data.py",
        }
        if not paths <= allowed:
            raise AssertionError(f"Unexpected CoOp worktree changes: {dirty}")
    if dassl_head != DASSL_SHA or dassl_dirty:
        raise AssertionError(
            f"Dassl provenance/state mismatch: {dassl_head} dirty={bool(dassl_dirty)}"
        )
    return {
        "branch": branch,
        "task8_base_sha": TASK8_COOP_SHA,
        "runtime_sha": head,
        "upstream_provenance_sha": COOP_UPSTREAM_SHA,
        "task9_paths_only_dirty": bool(dirty),
        "dassl_sha": dassl_head,
        "dassl_clean": not bool(dassl_dirty),
    }


def _build_cfg(dataset_key: str, data_root: Path, shots: int, seed: int):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(
        str(REPO_ROOT / "configs" / "datasets" / f"{dataset_key}.yaml")
    )
    cfg.merge_from_file(
        str(REPO_ROOT / "configs" / "trainers" / "CoOp" / "vit_b16_ctxv1.yaml")
    )
    cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.NUM_SHOTS = shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.K_TRANSFORMS = 1
    cfg.SEED = seed
    # Task 9 is a CPU data validation. This affects pinning only, not the
    # transform family resolved from the pinned config.
    cfg.USE_CUDA = False
    cfg.freeze()
    return cfg


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _sweep(loader, fields, generator):
    sample_ids: list[str] = []
    labels: list[int] = []
    tensor_hashes: list[str] = []
    batch_sizes: list[int] = []
    with isolated_rng(**fields, explicit_generators=(generator,)) as derived:
        for batch in loader:
            batch_ids = [str(value) for value in batch["sample_id"]]
            batch_sizes.append(len(batch_ids))
            sample_ids.extend(batch_ids)
            labels.extend(int(value) for value in batch["original_label"].tolist())
            tensor_hashes.extend(_tensor_sha256(image) for image in batch["img"])
    return {
        "sample_ids": sample_ids,
        "labels": labels,
        "tensor_hashes": tensor_hashes,
        "batch_sizes": batch_sizes,
        "seed_digest": derived.sha256,
    }


def _seed_globals(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state())
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def _draw_bundle(count: int = 8):
    cuda = ()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda = tuple(
            torch.rand(count, device=f"cuda:{index}").cpu()
            for index in range(torch.cuda.device_count())
        )
    return {
        "python": tuple(random.random() for _ in range(count)),
        "numpy": np.random.random(count),
        "torch_cpu": torch.rand(count),
        "torch_cuda": cuda,
    }


def _bundle_checks(expected, actual) -> dict[str, bool]:
    return {
        "python": expected["python"] == actual["python"],
        "numpy": bool(np.array_equal(expected["numpy"], actual["numpy"])),
        "torch_cpu": bool(torch.equal(expected["torch_cpu"], actual["torch_cpu"])),
        "torch_cuda": len(expected["torch_cuda"]) == len(actual["torch_cuda"])
        and all(
            torch.equal(left, right)
            for left, right in zip(expected["torch_cuda"], actual["torch_cuda"])
        ),
    }


def _assert_batch_equal(expected, actual) -> None:
    for key in ("index", "label"):
        if not torch.equal(expected[key], actual[key]):
            raise AssertionError(f"Normal loader {key} changed after dedicated sweep")
    if not torch.equal(expected["img"], actual["img"]):
        raise AssertionError("Normal loader transformed tensors changed after sweep")
    if list(expected["impath"]) != list(actual["impath"]):
        raise AssertionError("Normal loader image identities changed after sweep")


def _normal_iterator_noninterference(
    dataset_key: str,
    cfg,
    source,
    transform,
    dedicated_loader,
    dedicated_generator,
) -> dict[str, Any]:
    if dataset_key == "dtd":
        from datasets.dtd import DescribableTextures as dataset_class
    elif dataset_key == "eurosat":
        from datasets.eurosat import EuroSAT as dataset_class
    else:
        raise AssertionError(dataset_key)
    normal_data = dataset_class(cfg).train_x

    def two_batches(with_auxiliary: bool):
        normal_loader = build_data_loader(
            cfg,
            sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
            data_source=normal_data,
            batch_size=2,
            tfm=transform,
            is_train=True,
        )
        iterator = iter(normal_loader)
        first = next(iterator)
        if with_auxiliary:
            fields = {
                "protocol_seed": int(cfg.SEED),
                "dataset": dataset_key,
                "shots": int(cfg.DATASET.NUM_SHOTS),
                "config_hash": VALIDATION_CONFIG_HASH,
                "optimizer_step": 5,
                "purpose": "task9_normal_loader_noninterference",
            }
            with isolated_rng(
                **fields,
                explicit_generators=(dedicated_generator,),
            ):
                list(dedicated_loader)
        second = next(iterator)
        return first, second

    process = capture_rng_state((dedicated_generator,))
    try:
        _seed_globals(9100 + int(cfg.SEED))
        initial = capture_rng_state((dedicated_generator,))
        control = two_batches(False)
        restore_rng_state(initial)
        isolated = two_batches(True)
    finally:
        restore_rng_state(process)
    _assert_batch_equal(control[0], isolated[0])
    _assert_batch_equal(control[1], isolated[1])
    return {
        "dataset": dataset_key,
        "shots": int(cfg.DATASET.NUM_SHOTS),
        "seed": int(cfg.SEED),
        "first_batch_ids": list(control[0]["impath"]),
        "second_batch_ids": list(control[1]["impath"]),
        "iterator_position_unchanged": True,
        "next_batch_transform_unchanged": True,
    }


def _transform_equivalence(cfg, source, transform) -> dict[str, Any]:
    dedicated = FullGradientDataset(cfg, source, transform)
    ordinary = DatasetWrapper(
        cfg,
        source.to_datum_list(),
        transform=transform,
        is_train=True,
    )
    process = capture_rng_state()
    try:
        _seed_globals(1209)
        state = capture_rng_state()
        expected = ordinary[0]
        restore_rng_state(state)
        actual = dedicated[0]
    finally:
        restore_rng_state(process)
    if not torch.equal(expected["img"], actual["img"]):
        raise AssertionError("Dedicated transform output differs from pinned DatasetWrapper")
    if expected["label"] != actual["label"] or expected["impath"] != actual["impath"]:
        raise AssertionError("Dedicated record mapping differs from pinned DatasetWrapper")
    return {
        "sample_id": actual["sample_id"],
        "tensor_sha256": _tensor_sha256(actual["img"]),
        "label": int(actual["label"]),
        "exact_tensor_equality": True,
        "same_pinned_dataset_wrapper_path": True,
    }


def _snapshot_cache_files(manifest_root: Path) -> dict[str, dict[str, Any]]:
    snapshots = {}
    for dataset_key in ("dtd", "eurosat"):
        for shots in SHOTS:
            for seed in SEEDS:
                path = (
                    manifest_root
                    / dataset_key
                    / f"shots_{shots}"
                    / f"seed_{seed}"
                    / "data_manifest.json"
                )
                manifest = json.loads(path.read_text(encoding="utf-8"))
                root = Path(manifest["dataset"]["root_absolute"])
                relative = manifest["few_shot"]["cache"][
                    "path_relative_to_dataset_root"
                ]
                cache = root / Path(relative)
                stat = cache.stat()
                snapshots[str(cache)] = {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": sha256_file(cache),
                }
    return snapshots


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8", newline="\n")
    temporary.replace(path)


def run(args) -> Path:
    preflight = _assert_preflight()
    data_root = Path(args.root).resolve(strict=True)
    manifest_root = Path(args.manifest_root).resolve(strict=True)
    output_path = Path(args.output).resolve()
    cache_before = _snapshot_cache_files(manifest_root)

    loaded = {
        key: load_dataset(data_root, DATASET_SPECS[key])
        for key in ("dtd", "eurosat")
    }
    sources = {}
    cells = []
    for dataset_key in ("dtd", "eurosat"):
        for shots in SHOTS:
            for seed in SEEDS:
                manifest_path = (
                    manifest_root
                    / dataset_key
                    / f"shots_{shots}"
                    / f"seed_{seed}"
                    / "data_manifest.json"
                )
                source = load_full_gradient_source(loaded[dataset_key], manifest_path)
                sources[(dataset_key, shots, seed)] = source
                derived = len(source.base_classes) * shots
                expected = EXPECTED_COUNTS[dataset_key][shots]
                if len(source) != derived or len(source) != expected:
                    raise AssertionError(
                        f"{dataset_key}/{shots}/{seed}: {len(source)} != {derived}/{expected}"
                    )
                cells.append(
                    {
                        "dataset": dataset_key,
                        "shots": shots,
                        "seed": seed,
                        "selected_count": len(source),
                        "derived_expected_count": derived,
                        "specification_expected_count": expected,
                        "per_class_counts": [
                            {"original_label": label, "count": count}
                            for label, count in source.count_per_class
                        ],
                        "fingerprint": source.fingerprint,
                        "manifest_ids_match": True,
                        "official_train_only": True,
                        "base_only": True,
                        "duplicate_ids": 0,
                        "status": "PASS",
                    }
                )

    configs = {
        (key, shots, seed): _build_cfg(key, data_root, shots, seed)
        for key, shots, seed in (
            ("dtd", 4, 1),
            ("eurosat", 4, 1),
            ("eurosat", 16, 1),
        )
    }
    loader_checks = {}
    dtd_source = sources[("dtd", 4, 1)]
    dtd_cfg = configs[("dtd", 4, 1)]
    for micro_batch_size in (1, 7, 32):
        loader = build_full_gradient_loader(
            dtd_cfg,
            dtd_source,
            micro_batch_size=micro_batch_size,
        )
        generator = loader.generator
        result = _sweep(
            loader,
            {
                "protocol_seed": 1,
                "dataset": "dtd",
                "shots": 4,
                "config_hash": VALIDATION_CONFIG_HASH,
                "optimizer_step": 0,
                "purpose": "task9_microbatch_invariance",
            },
            generator,
        )
        if tuple(result["sample_ids"]) != dtd_source.sample_ids:
            raise AssertionError("DTD metadata order changed with micro-batch size")
        loader_checks[f"dtd_4_seed1_batch{micro_batch_size}"] = {
            **describe_full_gradient_loader(loader),
            "observed_batch_sizes": result["batch_sizes"],
            "ordered_ids_match": True,
        }
    dtd_transform = loader.dataset.transform

    eurosat_source = sources[("eurosat", 16, 1)]
    eurosat_cfg = configs[("eurosat", 16, 1)]
    eurosat_loader = build_full_gradient_loader(
        eurosat_cfg,
        eurosat_source,
        micro_batch_size=32,
    )
    eurosat_generator = eurosat_loader.generator
    eurosat_result = _sweep(
        eurosat_loader,
        {
            "protocol_seed": 1,
            "dataset": "eurosat",
            "shots": 16,
            "config_hash": VALIDATION_CONFIG_HASH,
            "optimizer_step": 0,
            "purpose": "task9_complete_source",
        },
        eurosat_generator,
    )
    if tuple(eurosat_result["sample_ids"]) != eurosat_source.sample_ids:
        raise AssertionError("EuroSAT 16-shot source order mismatch")
    if eurosat_result["batch_sizes"] != [32, 32, 16]:
        raise AssertionError(
            f"EuroSAT final partial batch was not preserved: {eurosat_result['batch_sizes']}"
        )
    loader_checks["eurosat_16_seed1_batch32"] = {
        **describe_full_gradient_loader(eurosat_loader),
        "observed_batch_sizes": eurosat_result["batch_sizes"],
        "ordered_ids_match": True,
    }

    transform_equivalence = _transform_equivalence(
        dtd_cfg, dtd_source, dtd_transform
    )

    repeat_loader = build_full_gradient_loader(
        dtd_cfg,
        dtd_source,
        micro_batch_size=7,
    )
    repeat_generator = repeat_loader.generator
    repeat_fields = {
        "protocol_seed": 1,
        "dataset": "dtd",
        "shots": 4,
        "config_hash": VALIDATION_CONFIG_HASH,
        "optimizer_step": 7,
        "purpose": "task9_repeatability",
    }
    repeat_first = _sweep(repeat_loader, repeat_fields, repeat_generator)
    repeat_second = _sweep(repeat_loader, repeat_fields, repeat_generator)
    if repeat_first != repeat_second:
        raise AssertionError("Same isolated sweep metadata did not reproduce exactly")
    alternate = _sweep(
        repeat_loader,
        {**repeat_fields, "optimizer_step": 8, "purpose": "task9_alternate"},
        repeat_generator,
    )
    changed_tensors = sum(
        left != right
        for left, right in zip(
            repeat_first["tensor_hashes"], alternate["tensor_hashes"]
        )
    )
    if alternate["seed_digest"] == repeat_first["seed_digest"] or changed_tensors == 0:
        raise AssertionError("Purpose/step-separated isolated stream did not change")

    # Initialize CUDA only for a tiny state-continuation check, never for data/model work.
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        torch.empty(0, device="cuda:0")
    purity_loader = build_full_gradient_loader(
        dtd_cfg,
        dtd_source,
        micro_batch_size=32,
    )
    purity_generator = purity_loader.generator
    process = capture_rng_state((purity_generator,))
    try:
        _seed_globals(1409)
        continuation = capture_rng_state((purity_generator,))
        expected_bundle = _draw_bundle()
        restore_rng_state(continuation)
        _sweep(
            purity_loader,
            {
                "protocol_seed": 1,
                "dataset": "dtd",
                "shots": 4,
                "config_hash": VALIDATION_CONFIG_HASH,
                "optimizer_step": 9,
                "purpose": "task9_rng_purity",
            },
            purity_generator,
        )
        actual_bundle = _draw_bundle()
    finally:
        restore_rng_state(process)
    rng_purity = _bundle_checks(expected_bundle, actual_bundle)
    if not all(rng_purity.values()):
        raise AssertionError(f"Dedicated pass contaminated normal RNG: {rng_purity}")

    normal_iterator = []
    for dataset_key, shots, seed in (("dtd", 4, 1), ("eurosat", 4, 1)):
        cfg = configs[(dataset_key, shots, seed)]
        source = sources[(dataset_key, shots, seed)]
        dedicated = build_full_gradient_loader(
            cfg,
            source,
            micro_batch_size=7,
        )
        generator = dedicated.generator
        transform = dedicated.dataset.transform
        normal_iterator.append(
            _normal_iterator_noninterference(
                dataset_key,
                cfg,
                source,
                transform,
                dedicated,
                generator,
            )
        )

    cache_after = _snapshot_cache_files(manifest_root)
    if cache_after != cache_before:
        raise AssertionError("Task-9 validation modified a pinned few-shot cache")

    report = {
        "schema_version": "sample_fg.full_gradient_data_validation.v1",
        "status": "PASS",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": preflight,
        "source_schema": {
            "version": FULL_GRADIENT_SOURCE_SCHEMA_VERSION,
            "fingerprint_schema": FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION,
            "portable_identity": "dataset-root-relative POSIX path v1",
            "ordering": "original_label_then_sample_id_v1",
            "stores_transformed_tensors": False,
        },
        "matrix": {
            "cell_count": len(cells),
            "passing_cells": sum(cell["status"] == "PASS" for cell in cells),
            "cells": cells,
            "all_task2_manifest_sequences_match": True,
            "caches_read_only_unchanged": True,
        },
        "representative_sources": {
            "dtd_4_seed1": dtd_source.as_metadata(),
            "eurosat_16_seed1": eurosat_source.as_metadata(),
        },
        "loaders": loader_checks,
        "transform_equivalence": transform_equivalence,
        "isolated_sweep": {
            "same_metadata_same_ids_labels_and_tensor_hashes": True,
            "same_seed_digest": repeat_first["seed_digest"],
            "alternate_seed_digest": alternate["seed_digest"],
            "alternate_transformed_tensor_count_changed": changed_tensors,
            "normal_rng_continuation": rng_purity,
            "explicit_loader_generator_restored": True,
        },
        "normal_loader_noninterference": normal_iterator,
        "transform": {
            "builder": "dassl.data.transforms.build_transform(cfg, is_train=True)",
            "dataset_wrapper": "dassl.data.data_manager.DatasetWrapper",
            "choices": list(dtd_cfg.INPUT.TRANSFORMS),
            "size": list(dtd_cfg.INPUT.SIZE),
            "interpolation": dtd_cfg.INPUT.INTERPOLATION,
            "k_transforms": int(dtd_cfg.DATALOADER.K_TRANSFORMS),
        },
        "scope": {
            "optimizer_steps": 0,
            "gradient_evaluations": 0,
            "model_loaded": False,
            "clip_loaded": False,
            "full_gradient_computation": False,
            "full_gradient_service": False,
            "estimators": False,
            "sam": False,
            "sample": False,
        },
    }
    _write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cells": report["matrix"]["cell_count"],
                "dtd_4_count": len(dtd_source),
                "dtd_4_fingerprint": dtd_source.fingerprint,
                "eurosat_16_count": len(eurosat_source),
                "eurosat_16_fingerprint": eurosat_source.fingerprint,
                "eurosat_16_batches": eurosat_result["batch_sizes"],
                "rng_purity": rng_purity,
                "normal_iterator_noninterference": [
                    item["iterator_position_unchanged"] for item in normal_iterator
                ],
                "optimizer_steps": 0,
                "gradient_evaluations": 0,
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Prepared CoOp data root")
    parser.add_argument(
        "--manifest-root", required=True, help="Task-2 per-cell manifest root"
    )
    parser.add_argument("--output", required=True, help="Task-9 JSON report path")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
