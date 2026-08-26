"""Validate Task-8 RNG isolation without loading CLIP or taking a step."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from dassl.data.transforms import build_transform
from dassl.utils import read_image
from sample_fg.coop_anchor import build_smoke_cfg, sha256_file
from sample_fg.rng import (
    RNG_SEED_SCHEMA_VERSION,
    capture_rng_state,
    derive_auxiliary_seed,
    isolated_rng,
    restore_rng_state,
)


TASK7_COOP_SHA = "d8ce29298e964c1c44abd6b525b6b56addcc1da7"
COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
DTD_SPLIT_SHA = "F26EBECCD2B58E68D70F07A0DC39FE49BF7B69024BAC34396B954DFD87969C38"
SEED_FIELDS = {
    "protocol_seed": 1,
    "dataset": "dtd",
    "shots": 4,
    "config_hash": "0123456789abcdef",
    "optimizer_step": 0,
    "purpose": "diagnostic",
}


def _run(command, cwd):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _seed_validation_streams(seed):
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    torch.set_rng_state(cpu_generator.get_state())
    torch.cuda.manual_seed_all(seed)


def _draw_globals(count):
    return {
        "python": tuple(random.random() for _ in range(count)),
        "numpy": np.random.random(count),
        "torch_cpu": torch.rand(count),
        "torch_cuda": tuple(
            torch.rand(count, device=f"cuda:{device}")
            for device in range(torch.cuda.device_count())
        ),
    }


def _bundle_checks(left, right):
    return {
        "python": left["python"] == right["python"],
        "numpy": bool(np.array_equal(left["numpy"], right["numpy"])),
        "torch_cpu": bool(torch.equal(left["torch_cpu"], right["torch_cpu"])),
        "torch_cuda": len(left["torch_cuda"]) == len(right["torch_cuda"])
        and all(
            torch.equal(expected, actual)
            for expected, actual in zip(left["torch_cuda"], right["torch_cuda"])
        ),
    }


def _snapshots_equal(left, right):
    if left.python_state != right.python_state:
        return False
    if left.numpy_state[0] != right.numpy_state[0]:
        return False
    if not np.array_equal(left.numpy_state[1], right.numpy_state[1]):
        return False
    if left.numpy_state[2:] != right.numpy_state[2:]:
        return False
    if not torch.equal(left.torch_cpu_state, right.torch_cpu_state):
        return False
    if left.cuda_was_initialized != right.cuda_was_initialized:
        return False
    if len(left.torch_cuda_states) != len(right.torch_cuda_states):
        return False
    if any(
        not torch.equal(expected, actual)
        for expected, actual in zip(left.torch_cuda_states, right.torch_cuda_states)
    ):
        return False
    if len(left.explicit_generators) != len(right.explicit_generators):
        return False
    return all(
        expected.generator is actual.generator
        and expected.device == actual.device
        and torch.equal(expected.state, actual.state)
        for expected, actual in zip(
            left.explicit_generators, right.explicit_generators
        )
    )


def _tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest().upper()


def run(args):
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    data_root = Path(args.root).resolve(strict=True)
    output_path = Path(args.output).resolve()

    if _run(["git", "branch", "--show-current"], REPO_ROOT) != "sample-full-gradient":
        raise AssertionError("Task-8 validation requires sample-full-gradient")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK7_COOP_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-7 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Dassl SHA changed")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Dassl worktree is dirty")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-8 Stage-0 validation requires CUDA")

    split_path = data_root / "dtd" / "split_zhou_DescribableTextures.json"
    if sha256_file(split_path) != DTD_SPLIT_SHA:
        raise AssertionError("DTD fixed split changed")
    image_path = data_root / "dtd" / "images" / "banded" / "banded_0133.jpg"
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    torch.empty(0, device="cuda:0")
    cpu_explicit = torch.Generator(device="cpu").manual_seed(1200)
    cuda_explicit = torch.Generator(device="cuda:0").manual_seed(1201)
    generators = (cpu_explicit, cuda_explicit)
    process_snapshot = capture_rng_state(generators)
    try:
        derived = derive_auxiliary_seed(**SEED_FIELDS)

        _seed_validation_streams(1300)
        _draw_globals(1)
        continuation = capture_rng_state(generators)
        expected = _draw_globals(12)
        expected_cpu_generator = torch.rand(12, generator=cpu_explicit)
        expected_cuda_generator = torch.rand(
            12, device="cuda:0", generator=cuda_explicit
        )
        restore_rng_state(continuation)
        with isolated_rng(**SEED_FIELDS, explicit_generators=generators):
            _draw_globals(128)
            torch.rand(128, generator=cpu_explicit)
            torch.rand(128, device="cuda:0", generator=cuda_explicit)
        observed = _draw_globals(12)
        observed_cpu_generator = torch.rand(12, generator=cpu_explicit)
        observed_cuda_generator = torch.rand(
            12, device="cuda:0", generator=cuda_explicit
        )
        global_purity = _bundle_checks(expected, observed)
        explicit_cpu_purity = bool(
            torch.equal(expected_cpu_generator, observed_cpu_generator)
        )
        explicit_cuda_purity = bool(
            torch.equal(expected_cuda_generator, observed_cuda_generator)
        )

        before_exception = capture_rng_state(generators)
        propagated = False
        try:
            with isolated_rng(**SEED_FIELDS, explicit_generators=generators):
                _draw_globals(64)
                torch.rand(64, generator=cpu_explicit)
                torch.rand(64, device="cuda:0", generator=cuda_explicit)
                raise RuntimeError("intentional-task8-exception")
        except RuntimeError as error:
            propagated = str(error) == "intentional-task8-exception"
        after_exception = capture_rng_state(generators)
        exception_restored = propagated and _snapshots_equal(
            before_exception, after_exception
        )

        outer_fields = dict(SEED_FIELDS, purpose="estimator_refresh")
        inner_fields = dict(SEED_FIELDS, purpose="diagnostic", optimizer_step=1)
        with isolated_rng(**outer_fields, explicit_generators=generators):
            expected_outer_first = _draw_globals(8)
            expected_outer_second = _draw_globals(8)
        with isolated_rng(**outer_fields, explicit_generators=generators):
            actual_outer_first = _draw_globals(8)
            with isolated_rng(**inner_fields, explicit_generators=generators):
                _draw_globals(64)
            actual_outer_second = _draw_globals(8)
        nesting_checks = _bundle_checks(expected_outer_first, actual_outer_first)
        nesting_checks.update(
            {
                f"continued_{key}": value
                for key, value in _bundle_checks(
                    expected_outer_second, actual_outer_second
                ).items()
            }
        )
        nesting_restored = all(nesting_checks.values())

        cfg = build_smoke_cfg(
            REPO_ROOT, data_root, output_path.parent / "task8_no_training", "base"
        )
        transform = build_transform(cfg, is_train=True)
        image = read_image(str(image_path))
        transform_fields = dict(SEED_FIELDS, optimizer_step=3)
        transform_continuation = capture_rng_state(generators)
        transform_expected_normal = _draw_globals(8)
        restore_rng_state(transform_continuation)
        with isolated_rng(**transform_fields):
            first_transform = transform(image)
            transform(image)
        with isolated_rng(**transform_fields):
            second_transform = transform(image)
        transform_deterministic = bool(
            torch.equal(first_transform, second_transform)
        )
        transform_normal_checks = _bundle_checks(
            transform_expected_normal, _draw_globals(8)
        )

        all_checks = {
            **global_purity,
            "explicit_cpu_generator": explicit_cpu_purity,
            "explicit_cuda_generator": explicit_cuda_purity,
            "exception_restoration": exception_restored,
            "nesting": nesting_restored,
            "transform_deterministic": transform_deterministic,
            "transform_normal_continuation": all(transform_normal_checks.values()),
        }
        if not all(all_checks.values()):
            raise AssertionError(f"Task-8 RNG isolation failed: {all_checks}")

        report = {
            "schema_version": "sample_fg.rng_isolation_validation.v1",
            "status": "PASS",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "coop_upstream_provenance_sha": COOP_UPSTREAM_SHA,
                "coop_task7_sha": TASK7_COOP_SHA,
                "coop_runtime_sha": _run(["git", "rev-parse", "HEAD"], REPO_ROOT),
                "coop_runtime_dirty": bool(
                    _run(["git", "status", "--short"], REPO_ROOT)
                ),
                "dassl_sha": DASSL_SHA,
                "dassl_clean": True,
            },
            "environment": {
                "python": sys.version.replace("\n", " "),
                "pytorch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cuda_device_count": torch.cuda.device_count(),
            },
            "seed_derivation": derived.as_dict(),
            "rng_domains": {
                "python_restoration": global_purity["python"],
                "numpy_restoration": global_purity["numpy"],
                "torch_cpu_restoration": global_purity["torch_cpu"],
                "torch_cuda_all_device_restoration": global_purity["torch_cuda"],
                "explicit_cpu_generator_restoration": explicit_cpu_purity,
                "explicit_cuda_generator_restoration": explicit_cuda_purity,
                "exception_restoration_and_propagation": exception_restored,
                "nested_immediate_caller_restoration": nesting_restored,
                "cuda_states_captured": len(process_snapshot.torch_cuda_states),
            },
            "dtd_transform_fixture": {
                "dataset_relative_image": "dtd/images/banded/banded_0133.jpg",
                "image_sha256": sha256_file(image_path),
                "transform": [
                    "RandomResizedCrop(size=(224,224), scale=(0.08,1.0))",
                    "RandomHorizontalFlip",
                    "ToTensor",
                    "CLIP Normalize",
                ],
                "same_auxiliary_metadata_same_tensor": transform_deterministic,
                "isolated_tensor_sha256": _tensor_sha256(first_transform),
                "normal_rng_continuation": transform_normal_checks,
            },
            "scope": {
                "optimizer_steps": 0,
                "clip_loaded": False,
                "full_gradient_loader": False,
                "full_gradient_computation": False,
                "full_gradient_service": False,
                "ema": False,
                "exact_estimator": False,
                "periodic_estimator": False,
                "sam": False,
                "sample": False,
            },
        }
        _write_json(output_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "seed": report["seed_derivation"],
                    "rng_domains": report["rng_domains"],
                    "dtd_transform_fixture": report["dtd_transform_fixture"],
                    "optimizer_steps": 0,
                    "output": str(output_path),
                },
                indent=2,
            )
        )
        return output_path
    finally:
        restore_rng_state(process_snapshot)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Prepared CoOp data root")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
