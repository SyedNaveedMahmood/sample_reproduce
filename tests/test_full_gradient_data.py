import dataclasses
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from dassl.config import get_cfg_default
from dassl.data.data_manager import DatasetWrapper, build_data_loader

from sample_fg.full_gradient import (
    FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION,
    FULL_GRADIENT_SOURCE_ORDERING,
    FullGradientClass,
    FullGradientDataError,
    FullGradientDataset,
    FullGradientRecord,
    FullGradientSource,
    build_full_gradient_loader,
    describe_full_gradient_loader,
    iter_batch_sample_ids,
)
from sample_fg.rng import capture_rng_state, isolated_rng, restore_rng_state


RNG_FIELDS = {
    "protocol_seed": 1,
    "dataset": "fixture",
    "shots": 2,
    "config_hash": "task9-fixture-config",
    "optimizer_step": 0,
    "purpose": "full_gradient_data_fixture",
}


def _tensor_digest(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _seed_globals(seed):
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def _draw_bundle(count=8):
    return (
        tuple(random.random() for _ in range(count)),
        np.random.random(count),
        torch.rand(count),
    )


class CountingTransform:
    def __init__(self):
        self.calls = 0
        self.to_tensor = T.ToTensor()

    def __call__(self, image):
        self.calls += 1
        return self.to_tensor(image)


class FullGradientDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "fixture"
        self.root.mkdir()
        self.classes = (
            FullGradientClass(10, 0, "alpha"),
            FullGradientClass(20, 1, "beta"),
        )
        records = []
        position = 0
        for class_entry in self.classes:
            class_dir = self.root / "images" / class_entry.classname
            class_dir.mkdir(parents=True)
            for sample_number in range(2):
                image_path = class_dir / f"sample_{sample_number}.png"
                grid = np.arange(16 * 16 * 3, dtype=np.uint16).reshape(16, 16, 3)
                array = ((grid + position * 29) % 256).astype(np.uint8)
                Image.fromarray(array, mode="RGB").save(image_path)
                records.append(
                    FullGradientRecord(
                        position=position,
                        sample_id=image_path.relative_to(self.root).as_posix(),
                        image_path=image_path,
                        original_label=class_entry.original_label,
                        training_label=class_entry.training_label,
                        classname=class_entry.classname,
                        domain=0,
                        dataset="fixture",
                        shots=2,
                        seed=1,
                    )
                )
                position += 1
        self.records = tuple(records)
        self.source = self.make_source(self.records)

        cfg = get_cfg_default()
        cfg.USE_CUDA = False
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.DATALOADER.K_TRANSFORMS = 1
        cfg.DATALOADER.RETURN_IMG0 = False
        cfg.INPUT.SIZE = (8, 8)
        cfg.INPUT.TRANSFORMS = ()
        cfg.freeze()
        self.cfg = cfg

    def random_cfg(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.INPUT.TRANSFORMS = ("random_resized_crop", "random_flip")
        cfg.INPUT.RRCROP_SCALE = (0.3, 1.0)
        cfg.freeze()
        return cfg

    def make_source(self, records, **changes):
        fields = {
            "dataset": "fixture",
            "dataset_name": "Fixture",
            "dataset_root": self.root,
            "shots": 2,
            "seed": 1,
            "base_classes": self.classes,
            "records": tuple(records),
        }
        fields.update(changes)
        return FullGradientSource(**fields)

    def test_source_metadata_is_immutable_complete_and_portable(self):
        self.assertEqual(len(self.source), 4)
        self.assertEqual(self.source.ordering, FULL_GRADIENT_SOURCE_ORDERING)
        self.assertEqual(
            self.source.fingerprint_schema_version,
            FULL_GRADIENT_FINGERPRINT_SCHEMA_VERSION,
        )
        self.assertEqual(self.source.count_per_class, ((10, 2), (20, 2)))
        self.assertTrue(all("\\" not in item for item in self.source.sample_ids))
        self.assertFalse(
            any(
                isinstance(value, torch.Tensor)
                for record in self.source
                for value in dataclasses.astuple(record)
            )
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.source.shots = 4

    def test_source_fingerprint_is_order_sensitive_and_path_portable(self):
        same = self.make_source(self.records)
        self.assertEqual(same.fingerprint, self.source.fingerprint)
        swapped_raw = (self.records[1], self.records[0], *self.records[2:])
        swapped = tuple(
            dataclasses.replace(record, position=position)
            for position, record in enumerate(swapped_raw)
        )
        reordered = self.make_source(swapped)
        self.assertNotEqual(reordered.fingerprint, self.source.fingerprint)
        self.assertNotIn(str(self.root), self.source.fingerprint)

    def test_source_rejects_empty_duplicate_novel_missing_and_inconsistent_records(self):
        with self.assertRaisesRegex(FullGradientDataError, "must not be empty"):
            self.make_source(())

        duplicate = list(self.records)
        duplicate[1] = dataclasses.replace(
            duplicate[0], position=1
        )
        with self.assertRaisesRegex(FullGradientDataError, "duplicate selected"):
            self.make_source(duplicate)

        novel = list(self.records)
        novel[0] = dataclasses.replace(
            novel[0], original_label=30, training_label=2, classname="novel"
        )
        with self.assertRaisesRegex(FullGradientDataError, "non-base/novel"):
            self.make_source(novel)

        missing = list(self.records)
        missing[0] = dataclasses.replace(
            missing[0],
            sample_id="images/alpha/missing.png",
            image_path=self.root / "images" / "alpha" / "missing.png",
        )
        with self.assertRaisesRegex(FullGradientDataError, "missing"):
            self.make_source(missing)

        inconsistent = list(self.records)
        inconsistent[0] = dataclasses.replace(inconsistent[0], classname="wrong")
        with self.assertRaisesRegex(FullGradientDataError, "inconsistent"):
            self.make_source(inconsistent)

    def test_source_rejects_count_disagreement_and_unsafe_sample_id(self):
        with self.assertRaisesRegex(FullGradientDataError, "selected count"):
            self.make_source(self.records[:-1])
        unsafe = list(self.records)
        unsafe[0] = dataclasses.replace(unsafe[0], sample_id="../escape.png")
        with self.assertRaisesRegex(FullGradientDataError, "safe relative"):
            self.make_source(unsafe)

    def test_loader_is_sequential_complete_and_preserves_short_final_batch(self):
        transform = CountingTransform()
        with mock.patch(
            "sample_fg.full_gradient.build_transform", return_value=transform
        ):
            loader = build_full_gradient_loader(
                self.cfg,
                self.source,
                micro_batch_size=3,
            )
        batches = list(loader)
        metadata = describe_full_gradient_loader(loader)
        self.assertEqual(metadata["sampler"], "SequentialSampler")
        self.assertFalse(metadata["shuffle"])
        self.assertFalse(metadata["drop_last"])
        self.assertEqual(metadata["num_workers"], 0)
        self.assertEqual([len(batch["sample_id"]) for batch in batches], [3, 1])
        self.assertEqual(
            tuple(sample_id for batch in batches for sample_id in batch["sample_id"]),
            self.source.sample_ids,
        )
        self.assertEqual(transform.calls, len(self.source))

    def test_micro_batch_size_does_not_change_metadata_sequence(self):
        observed = []
        for size in (1, 3, 32):
            loader = build_full_gradient_loader(
                self.cfg,
                self.source,
                micro_batch_size=size,
            )
            observed.append(
                tuple(
                    sample_id
                    for batch_ids in iter_batch_sample_ids(loader)
                    for sample_id in batch_ids
                )
            )
        self.assertEqual(observed, [self.source.sample_ids] * 3)

    def test_workers_and_micro_batch_fail_fast(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(FullGradientDataError):
                build_full_gradient_loader(
                    self.cfg,
                    self.source,
                    micro_batch_size=value,
                )
        with self.assertRaisesRegex(FullGradientDataError, "num_workers=0"):
            build_full_gradient_loader(
                self.cfg,
                self.source,
                micro_batch_size=2,
                num_workers=1,
            )

    def test_dedicated_source_is_not_reduced_by_normal_drop_last(self):
        normal_loader = build_data_loader(
            self.cfg,
            sampler_type="SequentialSampler",
            data_source=self.source.to_datum_list(),
            batch_size=3,
            tfm=T.ToTensor(),
            is_train=True,
        )
        dedicated = build_full_gradient_loader(
            self.cfg,
            self.source,
            micro_batch_size=3,
        )
        self.assertTrue(normal_loader.drop_last)
        self.assertEqual(sum(len(batch["label"]) for batch in normal_loader), 3)
        self.assertEqual(sum(len(batch["label"]) for batch in dedicated), 4)

    def test_pinned_dataset_wrapper_transform_equivalence(self):
        transform = T.Compose(
            [
                T.RandomResizedCrop((8, 8), scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )
        dedicated = FullGradientDataset(self.cfg, self.source, transform)
        ordinary = DatasetWrapper(
            self.cfg,
            self.source.to_datum_list(),
            transform=transform,
            is_train=True,
        )
        snapshot = capture_rng_state()
        try:
            _seed_globals(909)
            seeded = capture_rng_state()
            expected = ordinary[0]
            restore_rng_state(seeded)
            actual = dedicated[0]
        finally:
            restore_rng_state(snapshot)
        self.assertTrue(torch.equal(expected["img"], actual["img"]))
        self.assertEqual(expected["label"], actual["label"])
        self.assertEqual(expected["impath"], actual["impath"])
        self.assertEqual(actual["sample_id"], self.source.sample_ids[0])

    def test_isolated_full_pass_is_repeatable_and_rng_invisible(self):
        cfg = self.random_cfg()
        loader = build_full_gradient_loader(
            cfg,
            self.source,
            micro_batch_size=3,
        )
        generator = loader.generator
        process = capture_rng_state((generator,))
        try:
            _seed_globals(808)
            parent = capture_rng_state((generator,))
            expected_continuation = _draw_bundle()
            restore_rng_state(parent)

            runs = []
            for _ in range(2):
                with isolated_rng(
                    **RNG_FIELDS,
                    explicit_generators=(generator,),
                ):
                    runs.append(
                        tuple(
                            _tensor_digest(image)
                            for batch in loader
                            for image in batch["img"]
                        )
                    )
            actual_continuation = _draw_bundle()
        finally:
            restore_rng_state(process)
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(expected_continuation[0], actual_continuation[0])
        np.testing.assert_array_equal(expected_continuation[1], actual_continuation[1])
        self.assertTrue(torch.equal(expected_continuation[2], actual_continuation[2]))

    def test_isolated_dedicated_pass_does_not_advance_normal_iterator(self):
        cfg = self.random_cfg()
        dedicated = build_full_gradient_loader(
            cfg,
            self.source,
            micro_batch_size=3,
        )
        dedicated_generator = dedicated.generator
        transform = dedicated.dataset.transform

        def normal_two_batches(run_auxiliary):
            normal = build_data_loader(
                cfg,
                sampler_type="RandomSampler",
                data_source=self.source.to_datum_list(),
                batch_size=2,
                tfm=transform,
                is_train=True,
            )
            iterator = iter(normal)
            first = next(iterator)
            if run_auxiliary:
                with isolated_rng(
                    **RNG_FIELDS,
                    explicit_generators=(dedicated_generator,),
                ):
                    list(dedicated)
            second = next(iterator)
            return first, second

        process = capture_rng_state((dedicated_generator,))
        try:
            _seed_globals(707)
            initial = capture_rng_state((dedicated_generator,))
            expected = normal_two_batches(False)
            restore_rng_state(initial)
            actual = normal_two_batches(True)
        finally:
            restore_rng_state(process)
        for expected_batch, actual_batch in zip(expected, actual):
            self.assertTrue(torch.equal(expected_batch["index"], actual_batch["index"]))
            self.assertTrue(torch.equal(expected_batch["label"], actual_batch["label"]))
            self.assertTrue(torch.equal(expected_batch["img"], actual_batch["img"]))


if __name__ == "__main__":
    unittest.main()
