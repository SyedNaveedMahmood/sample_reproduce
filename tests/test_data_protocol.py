from __future__ import annotations

import json
import pickle
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dassl.data.datasets import DatasetBase
from datasets.dtd import DescribableTextures

from sample_fg.data_protocol import (
    DatasetSpec,
    ProtocolValidationError,
    _item_signature,
    _loader_and_transform_metadata,
    ensure_upstream_cache,
    load_dataset,
    select_upstream_fewshot,
    sha256_file,
    write_deterministic_json,
)


def fixture_spec(total_classes: int = 5) -> DatasetSpec:
    return DatasetSpec(
        key="fixture",
        canonical_name="Fixture",
        registry_name="Fixture",
        dataset_dir="fixture",
        image_dir="images",
        split_filename="split.json",
        split_drive_id="fixture",
        pinned_raw_url="fixture://archive",
        local_acquisition_url="fixture://archive",
        archive_filename="fixture.zip",
        expected_total_classes=total_classes,
        expected_base_classes=(total_classes + 1) // 2,
        expected_new_classes=total_classes // 2,
        expected_image_count=total_classes * 8,
        expected_split_counts=(total_classes * 4, total_classes * 2, total_classes * 2),
    )


def create_fixture(root: Path, spec: DatasetSpec, overlap: bool = False) -> None:
    dataset_root = root / spec.dataset_dir
    image_root = dataset_root / spec.image_dir
    archive_root = root / "_archives"
    archive_root.mkdir(parents=True)
    (archive_root / spec.archive_filename).write_bytes(b"fixture archive")
    split = {"train": [], "val": [], "test": []}
    for label in range(spec.expected_total_classes):
        classname = f"class_{label}"
        class_dir = image_root / classname
        class_dir.mkdir(parents=True)
        paths = []
        for index in range(8):
            path = class_dir / f"image_{index}.jpg"
            path.touch()
            paths.append(path.relative_to(image_root).as_posix())
        split["train"].extend([[path, label, classname] for path in paths[:4]])
        split["val"].extend([[path, label, classname] for path in paths[4:6]])
        split["test"].extend([[path, label, classname] for path in paths[6:]])
    if overlap:
        split["test"][0] = list(split["train"][0])
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / spec.split_filename).write_text(
        json.dumps(split), encoding="utf-8"
    )


class DataProtocolTests(unittest.TestCase):
    def test_missing_split_fails_before_random_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = fixture_spec()
            (root / spec.dataset_dir / spec.image_dir).mkdir(parents=True)
            (root / "_archives").mkdir()
            (root / "_archives" / spec.archive_filename).write_bytes(b"archive")
            with mock.patch.object(
                DescribableTextures,
                "read_and_split_data",
                side_effect=AssertionError("fallback must not run"),
            ) as fallback:
                with self.assertRaisesRegex(FileNotFoundError, "Refusing upstream random split"):
                    load_dataset(root, spec)
                fallback.assert_not_called()

    def test_sha256_known_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "known.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
            )

    def test_partition_is_disjoint_complete_and_expected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = fixture_spec(total_classes=5)
            create_fixture(root, spec)
            loaded = load_dataset(root, spec)
            self.assertEqual(loaded.base_labels, (0, 1, 2))
            self.assertEqual(loaded.new_labels, (3, 4))
            self.assertFalse(set(loaded.base_labels) & set(loaded.new_labels))
            self.assertEqual(
                set(loaded.base_labels) | set(loaded.new_labels), set(loaded.class_map)
            )

    def test_sampling_exact_count_and_matches_upstream_seed_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = fixture_spec(total_classes=4)
            create_fixture(root, spec)
            loaded = load_dataset(root, spec)
            first_train, first_val = select_upstream_fewshot(
                loaded.train, loaded.val, shots=2, seed=3
            )
            second_train, second_val = select_upstream_fewshot(
                loaded.train, loaded.val, shots=2, seed=3
            )
            self.assertEqual(
                _item_signature(first_train, loaded.dataset_root),
                _item_signature(second_train, loaded.dataset_root),
            )
            self.assertEqual(len(first_train), 8)
            self.assertTrue(all(value == 2 for value in _counts(first_train).values()))

            state = random.getstate()
            try:
                random.seed(3)
                helper = DatasetBase(train_x=loaded.train)
                direct_train = helper.generate_fewshot_dataset(
                    loaded.train, num_shots=2
                )
                direct_val = helper.generate_fewshot_dataset(
                    loaded.val, num_shots=2
                )
            finally:
                random.setstate(state)
            self.assertEqual(
                _item_signature(first_train, loaded.dataset_root),
                _item_signature(direct_train, loaded.dataset_root),
            )
            self.assertEqual(
                _item_signature(first_val, loaded.dataset_root),
                _item_signature(direct_val, loaded.dataset_root),
            )

            other_train, _ = select_upstream_fewshot(
                loaded.train, loaded.val, shots=2, seed=4
            )
            state = random.getstate()
            try:
                random.seed(4)
                helper = DatasetBase(train_x=loaded.train)
                expected_other = helper.generate_fewshot_dataset(
                    loaded.train, num_shots=2
                )
            finally:
                random.setstate(state)
            self.assertEqual(
                _item_signature(other_train, loaded.dataset_root),
                _item_signature(expected_other, loaded.dataset_root),
            )

    def test_train_test_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = fixture_spec()
            create_fixture(root, spec, overlap=True)
            with self.assertRaisesRegex(ProtocolValidationError, "leakage"):
                load_dataset(root, spec)

    def test_incompatible_existing_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = fixture_spec(total_classes=4)
            create_fixture(root, spec)
            loaded = load_dataset(root, spec)
            _, _, cache_path, _, action = ensure_upstream_cache(
                loaded, shots=2, seed=1
            )
            self.assertEqual(action, "generated")
            with cache_path.open("wb") as stream:
                pickle.dump({"train": [], "val": []}, stream)
            with self.assertRaisesRegex(
                ProtocolValidationError, "disagrees with a fresh pinned-upstream selection"
            ):
                ensure_upstream_cache(loaded, shots=2, seed=1)

    def test_manifest_serialization_is_stable_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data_manifest.json"
            payload_a = {
                "schema_version": "test.v1",
                "dataset": {"name": "Fixture"},
                "complete_selected_source": {
                    "count": 2,
                    "records": [
                        {"sample_id": "images/a.jpg", "original_label": 0},
                        {"sample_id": "images/b.jpg", "original_label": 0},
                    ],
                },
            }
            payload_b = {
                "complete_selected_source": payload_a["complete_selected_source"],
                "dataset": payload_a["dataset"],
                "schema_version": "test.v1",
            }
            write_deterministic_json(path, payload_a)
            first = path.read_bytes()
            write_deterministic_json(path, payload_b)
            second = path.read_bytes()
            self.assertEqual(first, second)
            parsed = json.loads(second)
            self.assertEqual(parsed["complete_selected_source"]["count"], 2)
            self.assertTrue(
                all(
                    not Path(record["sample_id"]).is_absolute()
                    for record in parsed["complete_selected_source"]["records"]
                )
            )

    def test_complete_source_is_independent_of_drop_last(self):
        coop_root = Path(__file__).resolve().parents[1]
        loader, transforms = _loader_and_transform_metadata(
            coop_root, selected_count=80
        )
        self.assertTrue(loader["drop_last"])
        self.assertEqual(loader["samples_consumed_per_epoch"], 64)
        self.assertEqual(loader["complete_selected_source_count"], 80)
        self.assertEqual(transforms["input_size"], [224, 224])
        self.assertEqual(transforms["test_pipeline"][0]["size"], 224)


def _counts(items):
    counts = {}
    for item in items:
        counts[item.label] = counts.get(item.label, 0) + 1
    return counts


if __name__ == "__main__":
    unittest.main()
