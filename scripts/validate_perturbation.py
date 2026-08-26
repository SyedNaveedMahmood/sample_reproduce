"""Validate Task-13 prompt perturbation on the real CoOp anchor."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.rng import capture_rng_state


TASK12_SHA = "8d7cfae85eb4793ee5c0ee41766761e714d06adb"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _git(args: list[str], root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _rng_equal(left, right) -> bool:
    import numpy as np

    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(torch.equal(a, b) for a, b in zip(left.torch_cuda_states, right.torch_cuda_states))
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK12_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-12 commit is not an ancestor of HEAD")
    if _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA or _git(["status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-13 real integration requires CUDA")

    output = Path(args.output).resolve()
    cfg = build_smoke_cfg(
        REPO_ROOT,
        Path(args.root).resolve(strict=True),
        output.parent / "task13_runtime",
        "base",
    )
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, Path(args.clip_cache).resolve(strict=True))
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    if index.names != ("prompt_learner.ctx",):
        raise AssertionError(f"Unexpected trainable set: {index.names}")

    displacement = GradientState.from_tensors(
        index,
        (torch.full(index[0].shape, 1.0e-3, dtype=torch.float32, device=index[0].device),),
    )
    prompt = index[0].parameter
    prompt_id = id(prompt)
    prompt_before = prompt.detach().clone()
    prompt.grad = torch.full_like(prompt, 0.25)
    grad_before = prompt.grad.detach().clone()
    frozen_before = hash_frozen_parameters(model)
    modes_before = tuple(module.training for module in model.modules())
    rng_before = capture_rng_state()
    controller = PromptPerturbation(index)

    with count_optimizer_steps(trainer.optim) as counter:
        with controller.displaced(displacement) as snapshot:
            expected = snapshot[0] + displacement[0].to(dtype=prompt.dtype)
            inside_exact = torch.equal(prompt, expected)
            inside_delta_norm = float((prompt.float() - snapshot[0].float()).norm().item())
        restored_normal = torch.equal(prompt, prompt_before)
        try:
            with controller.displaced(displacement):
                raise RuntimeError("task13 injected integration failure")
        except RuntimeError as error:
            if str(error) != "task13 injected integration failure":
                raise
        restored_exception = torch.equal(prompt, prompt_before)

    rng_after = capture_rng_state()
    checks = {
        "inside_value_exact": inside_exact,
        "restored_after_normal_exit": restored_normal,
        "restored_after_exception": restored_exception,
        "parameter_identity_unchanged": id(prompt) == prompt_id,
        "live_grad_unchanged": torch.equal(prompt.grad, grad_before),
        "frozen_parameters_unchanged": hash_frozen_parameters(model) == frozen_before,
        "module_modes_unchanged": tuple(module.training for module in model.modules()) == modes_before,
        "rng_unchanged": _rng_equal(rng_before, rng_after),
        "optimizer_steps_zero": counter["count"] == 0,
        "controller_inactive": not controller.active,
    }
    if not all(checks.values()):
        raise AssertionError(f"Task-13 real integration checks failed: {checks}")

    payload = {
        "schema_version": "sample_fg.task13_perturbation.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "backbone": "ViT-B/16",
        "precision": "fp32",
        "trainable_names": list(index.names),
        "param_index_fingerprint": index.fingerprint,
        "parameter_dtype": str(prompt.dtype),
        "parameter_device": str(prompt.device),
        "displacement_dtype": str(displacement[0].dtype),
        "displacement_norm": float(displacement.norm().item()),
        "inside_applied_delta_norm": inside_delta_norm,
        "checks": checks,
        "optimizer_steps": counter["count"],
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
