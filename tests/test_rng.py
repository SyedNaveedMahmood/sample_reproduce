import random
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sample_fg.coop_anchor import build_smoke_cfg
from sample_fg.rng import (
    RNG_SEED_SCHEMA_VERSION,
    RNGError,
    capture_rng_state,
    derive_auxiliary_seed,
    isolated_rng,
    restore_rng_state,
)


SEED_FIELDS = {
    "protocol_seed": 1,
    "dataset": "dtd",
    "shots": 4,
    "config_hash": "0123456789abcdef",
    "optimizer_step": 0,
    "purpose": "diagnostic",
}


def _seed_cpu_domains(seed):
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.set_rng_state(generator.get_state())


def _draw_cpu_bundle(count=5):
    return (
        tuple(random.random() for _ in range(count)),
        np.random.random(count),
        torch.rand(count),
    )


@contextmanager
def _restore_after_test(explicit_generators=()):
    snapshot = capture_rng_state(explicit_generators)
    try:
        yield
    finally:
        restore_rng_state(snapshot)


class RNGIsolationTest(unittest.TestCase):
    def assertBundlesEqual(self, left, right):
        self.assertEqual(left[0], right[0])
        np.testing.assert_array_equal(left[1], right[1])
        self.assertTrue(torch.equal(left[2], right[2]))

    def assertSnapshotsEqual(self, left, right):
        self.assertEqual(left.python_state, right.python_state)
        self.assertEqual(left.numpy_state[0], right.numpy_state[0])
        np.testing.assert_array_equal(left.numpy_state[1], right.numpy_state[1])
        self.assertEqual(left.numpy_state[2:], right.numpy_state[2:])
        self.assertTrue(torch.equal(left.torch_cpu_state, right.torch_cpu_state))
        self.assertEqual(left.cuda_was_initialized, right.cuda_was_initialized)
        self.assertEqual(len(left.torch_cuda_states), len(right.torch_cuda_states))
        for expected, actual in zip(
            left.torch_cuda_states, right.torch_cuda_states
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(
            len(left.explicit_generators), len(right.explicit_generators)
        )
        for expected, actual in zip(
            left.explicit_generators, right.explicit_generators
        ):
            self.assertIs(expected.generator, actual.generator)
            self.assertEqual(expected.device, actual.device)
            self.assertTrue(torch.equal(expected.state, actual.state))

    def test_known_answer_seed_vector_and_canonical_encoding(self):
        seed = derive_auxiliary_seed(**SEED_FIELDS)
        self.assertEqual(RNG_SEED_SCHEMA_VERSION, "sample_fg.fullgrad_rng.v1")
        self.assertEqual(
            seed.canonical_preimage,
            "sample_fg.fullgrad_rng.v1|1|dtd|4|0123456789abcdef|0|diagnostic",
        )
        self.assertEqual(
            seed.sha256,
            "d48d2b4c0086d21488b4d472f339b9ee0135d176348b5753af3bf2e42cb0a230",
        )
        self.assertEqual(seed.raw_uint64, 15315945513183269396)
        self.assertEqual(seed.python_seed, 15315945513183269396)
        self.assertEqual(seed.numpy_seed, 8835604)
        self.assertEqual(seed.torch_seed, 15315945513183269396)
        self.assertEqual(seed.as_dict()["sha256"], seed.sha256)

    def test_seed_is_deterministic_and_every_normative_field_separates(self):
        reference = derive_auxiliary_seed(**SEED_FIELDS)
        self.assertEqual(reference, derive_auxiliary_seed(**SEED_FIELDS))
        changes = {
            "protocol_seed": 2,
            "dataset": "eurosat",
            "shots": 8,
            "config_hash": "fedcba9876543210",
            "optimizer_step": 1,
            "purpose": "estimator_refresh",
        }
        for field, value in changes.items():
            candidate_fields = dict(SEED_FIELDS)
            candidate_fields[field] = value
            candidate = derive_auxiliary_seed(**candidate_fields)
            self.assertNotEqual(candidate.sha256, reference.sha256, field)
            self.assertNotEqual(candidate.raw_uint64, reference.raw_uint64, field)

    def test_seed_metadata_rejects_ambiguous_or_invalid_values(self):
        for field, value in (
            ("protocol_seed", -1),
            ("shots", 0),
            ("optimizer_step", -1),
            ("dataset", ""),
            ("config_hash", "bad|hash"),
            ("purpose", "bad\npurpose"),
        ):
            fields = dict(SEED_FIELDS)
            fields[field] = value
            with self.assertRaises(RNGError, msg=field):
                derive_auxiliary_seed(**fields)

    def test_python_rng_continuation_is_exact(self):
        with _restore_after_test():
            random.seed(101)
            random.random()
            state = random.getstate()
            expected = tuple(random.random() for _ in range(8))
            random.setstate(state)
            with isolated_rng(**SEED_FIELDS):
                [random.random() for _ in range(100)]
            self.assertEqual(tuple(random.random() for _ in range(8)), expected)

    def test_numpy_rng_continuation_is_exact(self):
        with _restore_after_test():
            np.random.seed(202)
            np.random.random(1)
            state = np.random.get_state()
            expected = np.random.random(8)
            np.random.set_state(state)
            with isolated_rng(**SEED_FIELDS):
                np.random.random(100)
            np.testing.assert_array_equal(np.random.random(8), expected)

    def test_torch_cpu_rng_continuation_is_exact(self):
        with _restore_after_test():
            _seed_cpu_domains(303)
            torch.rand(1)
            state = torch.get_rng_state()
            expected = torch.rand(8)
            torch.set_rng_state(state)
            with isolated_rng(**SEED_FIELDS):
                torch.rand(100)
            self.assertTrue(torch.equal(torch.rand(8), expected))

    def test_combined_domain_purity_and_same_seed_body_determinism(self):
        with _restore_after_test():
            _seed_cpu_domains(404)
            _draw_cpu_bundle(1)
            continuation = capture_rng_state()
            expected = _draw_cpu_bundle(8)
            restore_rng_state(continuation)
            with isolated_rng(**SEED_FIELDS) as first_seed:
                first_body = _draw_cpu_bundle(32)
            with isolated_rng(**SEED_FIELDS) as second_seed:
                second_body = _draw_cpu_bundle(32)
            self.assertEqual(first_seed, second_seed)
            self.assertBundlesEqual(first_body, second_body)
            self.assertBundlesEqual(_draw_cpu_bundle(8), expected)

    def test_explicit_cpu_generator_is_owned_restored_and_not_replaced(self):
        generator = torch.Generator(device="cpu").manual_seed(505)
        with _restore_after_test((generator,)):
            identity = id(generator)
            torch.rand(1, generator=generator)
            state = generator.get_state()
            expected = torch.rand(12, generator=generator)
            generator.set_state(state)
            with isolated_rng(**SEED_FIELDS, explicit_generators=(generator,)):
                torch.rand(100, generator=generator)
            self.assertEqual(id(generator), identity)
            self.assertTrue(torch.equal(torch.rand(12, generator=generator), expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_global_and_explicit_generator_continuation_is_exact(self):
        torch.empty(0, device="cuda:0")
        generator = torch.Generator(device="cuda:0").manual_seed(606)
        with _restore_after_test((generator,)):
            torch.cuda.manual_seed_all(607)
            expected_global = []
            states = torch.cuda.get_rng_state_all()
            for device in range(torch.cuda.device_count()):
                expected_global.append(torch.rand(8, device=f"cuda:{device}"))
            torch.cuda.set_rng_state_all(states)
            generator_state = generator.get_state()
            expected_explicit = torch.rand(8, device="cuda:0", generator=generator)
            generator.set_state(generator_state)
            with isolated_rng(**SEED_FIELDS, explicit_generators=(generator,)):
                for device in range(torch.cuda.device_count()):
                    torch.rand(100, device=f"cuda:{device}")
                torch.rand(100, device="cuda:0", generator=generator)
            for device, expected in enumerate(expected_global):
                self.assertTrue(torch.equal(torch.rand(8, device=f"cuda:{device}"), expected))
            self.assertTrue(
                torch.equal(
                    torch.rand(8, device="cuda:0", generator=generator),
                    expected_explicit,
                )
            )

    def test_exception_propagates_after_all_states_are_restored(self):
        generator = torch.Generator(device="cpu").manual_seed(707)
        with _restore_after_test((generator,)):
            _seed_cpu_domains(708)
            before = capture_rng_state((generator,))
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                with isolated_rng(
                    **SEED_FIELDS, explicit_generators=(generator,)
                ):
                    _draw_cpu_bundle(20)
                    torch.rand(20, generator=generator)
                    raise RuntimeError("intentional isolation failure")
            after = capture_rng_state((generator,))
            self.assertSnapshotsEqual(before, after)

    def test_nested_context_restores_immediate_caller_and_normal_stream(self):
        outer_fields = dict(SEED_FIELDS, purpose="estimator_refresh")
        inner_fields = dict(SEED_FIELDS, purpose="diagnostic", optimizer_step=1)
        with _restore_after_test():
            _seed_cpu_domains(808)
            normal_continuation = capture_rng_state()
            expected_normal = _draw_cpu_bundle(6)
            restore_rng_state(normal_continuation)

            with isolated_rng(**outer_fields):
                expected_outer_first = _draw_cpu_bundle(6)
                expected_outer_second = _draw_cpu_bundle(6)
            with isolated_rng(**outer_fields):
                actual_outer_first = _draw_cpu_bundle(6)
                with isolated_rng(**inner_fields):
                    _draw_cpu_bundle(100)
                actual_outer_second = _draw_cpu_bundle(6)

            self.assertBundlesEqual(actual_outer_first, expected_outer_first)
            self.assertBundlesEqual(actual_outer_second, expected_outer_second)
            self.assertBundlesEqual(_draw_cpu_bundle(6), expected_normal)

    def test_snapshot_owns_tensor_numpy_and_generator_state(self):
        generator = torch.Generator(device="cpu").manual_seed(909)
        with _restore_after_test((generator,)):
            _seed_cpu_domains(910)
            snapshot = capture_rng_state((generator,))
            cpu_copy = snapshot.torch_cpu_state.clone()
            numpy_copy = snapshot.numpy_state[1].copy()
            generator_copy = snapshot.explicit_generators[0].state.clone()
            _draw_cpu_bundle(100)
            torch.rand(100, generator=generator)
            self.assertTrue(torch.equal(snapshot.torch_cpu_state, cpu_copy))
            np.testing.assert_array_equal(snapshot.numpy_state[1], numpy_copy)
            self.assertTrue(
                torch.equal(snapshot.explicit_generators[0].state, generator_copy)
            )

    def test_actual_dtd_train_transform_is_deterministic_and_isolated(self):
        from dassl.data.transforms import build_transform
        from dassl.utils import read_image

        coop_root = Path(__file__).resolve().parents[1]
        project_root = coop_root.parents[1]
        data_root = project_root / "data"
        image_path = data_root / "dtd" / "images" / "banded" / "banded_0133.jpg"
        if image_path.is_file():
            image = read_image(str(image_path))
        else:
            # The documented source-only transfer excludes datasets.  The
            # transform/RNG invariant does not depend on one particular JPEG,
            # so keep this unit test portable while real-data smoke validation
            # remains responsible for checking the actual DTD source.
            pixels = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)
            image = Image.fromarray(pixels, mode="RGB")
        with tempfile.TemporaryDirectory() as directory:
            cfg = build_smoke_cfg(
                coop_root, data_root, Path(directory), class_subsample="base"
            )
            transform = build_transform(cfg, is_train=True)

        transform_fields = dict(SEED_FIELDS, purpose="diagnostic", optimizer_step=3)
        with _restore_after_test():
            _seed_cpu_domains(1001)
            _draw_cpu_bundle(1)
            continuation = capture_rng_state()
            expected_normal = _draw_cpu_bundle(8)
            restore_rng_state(continuation)
            with isolated_rng(**transform_fields):
                first = transform(image)
                transform(image)
            with isolated_rng(**transform_fields):
                second = transform(image)
            self.assertTrue(torch.equal(first, second))
            self.assertBundlesEqual(_draw_cpu_bundle(8), expected_normal)

    def test_task8_is_not_wired_into_ordinary_coop_training(self):
        coop_root = Path(__file__).resolve().parents[1]
        for relative in ("train.py", "trainers/coop.py"):
            source = (coop_root / relative).read_text(encoding="utf-8")
            self.assertNotIn("sample_fg.rng", source)
            self.assertNotIn("isolated_rng", source)


if __name__ == "__main__":
    unittest.main()
