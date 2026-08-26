"""Run one bounded real DTD diagnostic-only SAMPLe geometry fixture."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.diagnostics import compute_gradient_diagnostics
from sample_fg.estimators import EMAEstimator
from sample_fg.full_gradient import (
    FullGradientService,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.precision import PrecisionController
from sample_fg.projection import project_batch_gradient, safe_unit


TASK17_SHA = "43877a3c5b50ca6fc0800a0365ccee6f3c21d879"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"


def _git(args: list[str], root: Path) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _nested_equal(left, right) -> bool:
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


def _capture_gradient(model, batch, optimizer, index):
    controller = PrecisionController("fp32")
    controller.begin(optimizer)
    with controller.autocast_context():
        loss = F.cross_entropy(model(batch[0]), batch[1])
    controller.backward(loss)
    gradient = controller.capture_gradients(index, optimizer).state
    return float(loss.detach().item()), gradient


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Path:
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={REPO_ROOT}",
            "merge-base",
            "--is-ancestor",
            TASK17_SHA,
            "HEAD",
        ],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-17 commit is not an ancestor of HEAD")
    if (
        _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA
        or _git(["status", "--short"], dassl_root)
    ):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-18 real integration requires CUDA")

    output = Path(args.output).resolve()
    data_root = Path(args.root).resolve(strict=True)
    cfg = build_smoke_cfg(
        REPO_ROOT,
        data_root,
        output.parent / "task18_runtime",
        "base",
    )
    cfg.defrost()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.freeze()
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(
        cfg, Path(args.clip_cache).resolve(strict=True)
    )
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    prompt_before = tuple(entry.parameter.detach().clone() for entry in index)
    frozen_before = hash_frozen_parameters(model)
    optimizer_before = copy.deepcopy(trainer.optim.state_dict())
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())

    loaded = load_dataset(data_root, DATASET_SPECS["dtd"])
    manifest = (
        Path(args.manifest_root).resolve(strict=True)
        / "dtd"
        / "shots_4"
        / "seed_1"
        / "data_manifest.json"
    )
    source = load_full_gradient_source(loaded, manifest)
    loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    service = FullGradientService(
        model=model,
        param_index=index,
        loader=loader,
        precision_controller=PrecisionController("fp32"),
        protocol_seed=1,
        dataset="dtd",
        shots=4,
        config_hash="task18-real-diagnostics-v1",
    )
    estimator = EMAEstimator(index, ema_lambda=0.15)
    iterator = iter(trainer.train_loader_x)

    # Prime EMA with a different real batch at the same unchanged theta so the
    # diagnostic residual is nontrivial and its cosine is well conditioned.
    prime_raw = next(iterator)
    prime = trainer.parse_batch_train(prime_raw)
    _, prime_gradient = _capture_gradient(
        model, prime, trainer.optim, index
    )
    estimator.global_direction(batch_grad=prime_gradient, optimizer_step=0)

    raw_batch = next(iterator)
    batch = trainer.parse_batch_train(raw_batch)
    loss_current, batch_gradient = _capture_gradient(
        model, batch, trainer.optim, index
    )
    estimator_result = estimator.global_direction(
        batch_grad=batch_gradient, optimizer_step=1
    )
    active = estimator_result.active_global_estimate
    exact = service.compute(optimizer_step=1, purpose="diagnostic")
    projection = project_batch_gradient(batch_gradient, active)
    epsilon = safe_unit(batch_gradient).unit.scale(0.05)
    displacement = epsilon.subtract(projection.batch_component.scale(0.0015))

    perturbation = PromptPerturbation(index)
    with perturbation.displaced(displacement):
        loss_displaced, perturbed_gradient = _capture_gradient(
            model, batch, trainer.optim, index
        )
    perturbation.assert_inactive()

    state_before = {
        "batch": batch_gradient.clone(),
        "active": active.clone(),
        "perturbed": perturbed_gradient.clone(),
        "exact": exact.gradient.clone(),
        "estimator": copy.deepcopy(estimator.state_dict()),
    }
    metrics = compute_gradient_diagnostics(
        batch_gradient=batch_gradient,
        active_global_estimate=active,
        projection=projection,
        perturbed_gradient=perturbed_gradient,
        exact_full_gradient=exact.gradient,
        alpha=0.0015,
    )
    for label, state in (
        ("batch", batch_gradient),
        ("active", active),
        ("perturbed", perturbed_gradient),
        ("exact", exact.gradient),
    ):
        if not all(
            torch.equal(a, b) for a, b in zip(state, state_before[label])
        ):
            raise AssertionError(f"Diagnostic metrics mutated {label} state")
    if not _nested_equal(state_before["estimator"], estimator.state_dict()):
        raise AssertionError("Diagnostic metrics mutated EMA state")

    scalar_values = [
        value for value in metrics.values() if isinstance(value, float)
    ]
    if not all(math.isfinite(value) for value in scalar_values):
        raise FloatingPointError("Real diagnostics contain nonfinite values")
    active_cosine = metrics["grad/batch_component_estimator_cosine"]
    reference_cosine = metrics["grad/reference_batch_component_exact_cosine"]
    if active_cosine is None or abs(active_cosine) > 2e-5:
        raise AssertionError("Estimator projection sanity cosine is not near zero")
    if reference_cosine is None or abs(reference_cosine) > 2e-5:
        raise AssertionError("Exact-reference projection sanity cosine is not near zero")

    trainer.optim.zero_grad(set_to_none=True)
    if not all(
        torch.equal(entry.parameter, before)
        for entry, before in zip(index, prompt_before)
    ):
        raise AssertionError("Diagnostic-only fixture changed prompt parameters")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Diagnostic-only fixture changed frozen CLIP")
    if not _nested_equal(optimizer_before, trainer.optim.state_dict()):
        raise AssertionError("Diagnostic-only fixture changed optimizer state")
    if not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError("Diagnostic-only fixture changed scheduler state")

    with count_optimizer_steps(trainer.optim) as counter:
        pass
    payload = {
        "schema_version": "sample_fg.task18_diagnostics.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "backbone": "ViT-B/16",
        "precision": "fp32",
        "batch_size": 2,
        "workers": 0,
        "estimator": "ema",
        "ema_priming_steps": 1,
        "diagnostic_optimizer_step": 1,
        "loss_current": loss_current,
        "loss_displaced": loss_displaced,
        "metrics": metrics.as_dict(),
        "exact_query": exact.metadata.as_dict(),
        "exact_query_purpose": "diagnostic",
        "exact_query_at_unperturbed_theta": True,
        "optimizer_steps": counter["count"],
        "prompt_unchanged": True,
        "frozen_clip_unchanged": True,
        "optimizer_unchanged": True,
        "scheduler_unchanged": True,
        "estimator_unchanged_by_metric_computation": True,
    }
    _write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
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
