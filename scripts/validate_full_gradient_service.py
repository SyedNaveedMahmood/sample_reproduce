"""Run Task-10 exact-gradient acceptance on real DTD and EuroSAT CoOp paths."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from dassl.config import get_cfg_default
from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    build_coop_trainer,
    count_optimizer_steps,
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.full_gradient import (
    FullGradientService,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.rng import RNGSnapshot, capture_rng_state
from train import extend_cfg


TASK9_SHA = "2c7204dda56b52271918b5065c5847204a14cf30"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
CONFIG_HASH = "task10-real-coop-fp32-v1"


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _build_cfg(dataset: str, root: Path, shots: int, output_dir: Path):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"))
    cfg.merge_from_file(
        str(REPO_ROOT / "configs" / "trainers" / "CoOp" / "vit_b16_ctxv1.yaml")
    )
    cfg.merge_from_file(
        str(REPO_ROOT / "configs" / "sample_fg" / "smoke_task3_coop_anchor.yaml")
    )
    cfg.DATASET.ROOT = str(root)
    cfg.DATASET.NUM_SHOTS = shots
    cfg.DATASET.SUBSAMPLE_CLASSES = "base"
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.K_TRANSFORMS = 1
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.TRAINER.NAME = "CoOp"
    cfg.SEED = 1
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()
    return cfg


def _rng_equal(left: RNGSnapshot, right: RNGSnapshot) -> bool:
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
        not torch.equal(first, second)
        for first, second in zip(left.torch_cuda_states, right.torch_cuda_states)
    ):
        return False
    if len(left.explicit_generators) != len(right.explicit_generators):
        return False
    return all(
        first.generator is second.generator
        and first.device == second.device
        and torch.equal(first.state, second.state)
        for first, second in zip(
            left.explicit_generators, right.explicit_generators
        )
    )


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _run_dataset(
    *,
    dataset: str,
    shots: int,
    micro_batch_sizes: tuple[int, ...],
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    cfg = _build_cfg(dataset, data_root, shots, runtime_root / dataset)
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    param_index = ParamIndex.from_model(model)
    loaded = load_dataset(data_root, DATASET_SPECS[dataset])
    manifest = (
        manifest_root
        / dataset
        / f"shots_{shots}"
        / "seed_1"
        / "data_manifest.json"
    )
    source = load_full_gradient_source(loaded, manifest)

    prompt_before = tuple(entry.parameter.detach().clone() for entry in param_index)
    frozen_hash_before = hash_frozen_parameters(model)
    grad_before = tuple(
        None if entry.parameter.grad is None else entry.parameter.grad.detach().clone()
        for entry in param_index
    )
    optimizer_before = copy.deepcopy(trainer.optim.state_dict())
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    mode_before = tuple(module.training for module in model.modules())

    sweeps = []
    gradients = []
    with count_optimizer_steps(trainer.optim) as optimizer_counter:
        for micro_batch_size in micro_batch_sizes:
            loader = build_full_gradient_loader(
                cfg, source, micro_batch_size=micro_batch_size
            )
            rng_before = capture_rng_state((loader.generator,))
            torch.cuda.reset_peak_memory_stats()
            service = FullGradientService(
                model=model,
                param_index=param_index,
                loader=loader,
                precision_controller=PrecisionController("fp32"),
                protocol_seed=1,
                dataset=dataset,
                shots=shots,
                config_hash=CONFIG_HASH,
            )
            result = service.compute(optimizer_step=0, purpose="diagnostic")
            rng_after = capture_rng_state((loader.generator,))
            if not _rng_equal(rng_before, rng_after):
                raise AssertionError(f"{dataset}: exact sweep changed normal RNG state")
            gradients.append(result.gradient)
            sweep = result.metadata.as_dict()
            sweep.update(
                {
                    "gradient_norm": float(result.gradient.norm().item()),
                    "gradient_finite": result.gradient.is_finite(),
                    "gradient_dtypes": [str(item.dtype) for item in result.gradient],
                    "gradient_devices": [str(item.device) for item in result.gradient],
                    "peak_cuda_allocated_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "peak_cuda_reserved_bytes": int(
                        torch.cuda.max_memory_reserved()
                    ),
                    "rng_restored": True,
                }
            )
            sweeps.append(sweep)

    if optimizer_counter["count"] != 0:
        raise AssertionError("Full-gradient service called optimizer.step")
    for before, entry in zip(prompt_before, param_index):
        if not torch.equal(before, entry.parameter.detach()):
            raise AssertionError(f"{dataset}: prompt changed during exact query")
    if hash_frozen_parameters(model) != frozen_hash_before:
        raise AssertionError(f"{dataset}: frozen CLIP changed during exact query")
    for before, entry in zip(grad_before, param_index):
        after = entry.parameter.grad
        if before is None and after is not None:
            raise AssertionError(f"{dataset}: query populated live .grad")
        if before is not None and (after is None or not torch.equal(before, after)):
            raise AssertionError(f"{dataset}: query changed live .grad")
    if not _nested_equal(optimizer_before, trainer.optim.state_dict()):
        raise AssertionError(f"{dataset}: optimizer state changed")
    if not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError(f"{dataset}: scheduler state changed")
    if mode_before != tuple(module.training for module in model.modules()):
        raise AssertionError(f"{dataset}: module modes changed")

    invariance = None
    if len(gradients) > 1:
        torch.testing.assert_close(
            gradients[0].components,
            gradients[1].components,
            rtol=1e-4,
            atol=1e-6,
        )
        difference = gradients[0].subtract(gradients[1])
        invariance = {
            "compared_micro_batch_sizes": list(micro_batch_sizes[:2]),
            "difference_norm": float(difference.norm().item()),
            "rtol": 1e-4,
            "atol": 1e-6,
            "status": "PASS",
        }

    record = {
        "dataset": dataset,
        "shots": shots,
        "seed": 1,
        "source_count": len(source),
        "source_fingerprint": source.fingerprint,
        "param_index_fingerprint": param_index.fingerprint,
        "sweeps": sweeps,
        "microbatch_invariance": invariance,
        "purity": {
            "prompt_unchanged": True,
            "frozen_clip_unchanged": True,
            "live_grads_unchanged": True,
            "optimizer_unchanged": True,
            "scheduler_unchanged": True,
            "modes_unchanged": True,
            "optimizer_steps": 0,
        },
    }
    del trainer, model, gradients
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if _run(["git", "branch", "--show-current"], REPO_ROOT) != "sample-full-gradient":
        raise AssertionError("Expected sample-full-gradient branch")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TASK9_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-9 commit is not an ancestor of HEAD")
    if _run(["git", "rev-parse", "HEAD"], dassl_root) != DASSL_SHA:
        raise AssertionError("Pinned Dassl SHA changed")
    if _run(["git", "status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl worktree is dirty")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-10 real integration requires CUDA")

    data_root = Path(args.root).resolve(strict=True)
    manifest_root = Path(args.manifest_root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output = Path(args.output).resolve()
    runtime_root = output.parent / "task10_runtime"
    records = [
        _run_dataset(
            dataset="dtd",
            shots=4,
            micro_batch_sizes=(7, 32),
            data_root=data_root,
            manifest_root=manifest_root,
            clip_cache=clip_cache,
            runtime_root=runtime_root,
        ),
        _run_dataset(
            dataset="eurosat",
            shots=16,
            micro_batch_sizes=(32,),
            data_root=data_root,
            manifest_root=manifest_root,
            clip_cache=clip_cache,
            runtime_root=runtime_root,
        ),
    ]
    if records[1]["sweeps"][0]["observed_micro_batch_sizes"] != [32, 32, 16]:
        raise AssertionError("EuroSAT partial-batch shape differs from [32,32,16]")
    payload = {
        "schema_version": "sample_fg.task10_full_gradient_validation.v1",
        "status": "PASS",
        "environment": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "config_hash": CONFIG_HASH,
        "datasets": records,
        "scope": {
            "optimizer_steps": 0,
            "scheduler_steps": 0,
            "parameter_perturbations": 0,
            "estimators": False,
            "sam": False,
            "sample": False,
        },
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
