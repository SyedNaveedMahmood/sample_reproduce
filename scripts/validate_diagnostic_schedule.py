"""Validate diagnostic exact-query scheduling/reuse on real DTD/CoOp."""

from __future__ import annotations

import argparse
import copy
import json
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
    hash_frozen_parameters,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.diagnostic_schedule import (
    DiagnosticCoordinator,
    DiagnosticSchedule,
)
from sample_fg.estimators import EMAEstimator, ExactEstimator, PeriodicEstimator
from sample_fg.full_gradient import (
    FullGradientService,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController


TASK18_SHA = "58a47ab62a910c95b8f9ca4ea82998470caa0371"
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


class _CountingService:
    def __init__(self, service):
        self.service = service
        self.calls = []

    def compute(self, *, optimizer_step, purpose):
        result = self.service.compute(
            optimizer_step=optimizer_step, purpose=purpose
        )
        self.calls.append(
            {
                "optimizer_step": optimizer_step,
                "purpose": purpose,
                "elapsed_s": result.metadata.elapsed_s,
                "sample_count": result.metadata.sample_count,
                "micro_batch_count": result.metadata.micro_batch_count,
                "seed_sha256": result.metadata.seed.sha256,
            }
        )
        return result


def _capture_gradient(model, batch, optimizer, index):
    controller = PrecisionController("fp32")
    controller.begin(optimizer)
    loss = F.cross_entropy(model(batch[0]), batch[1])
    controller.backward(loss)
    return controller.capture_gradients(index, optimizer).state


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
        ["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK18_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-18 commit is not an ancestor of HEAD")
    if _git(["rev-parse", "HEAD"], dassl_root) != DASSL_SHA or _git(["status", "--short"], dassl_root):
        raise AssertionError("Pinned Dassl checkout changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Task-19 real integration requires CUDA")

    output = Path(args.output).resolve()
    data_root = Path(args.root).resolve(strict=True)
    cfg = build_smoke_cfg(REPO_ROOT, data_root, output.parent / "task19_runtime", "base")
    cfg.defrost()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.freeze()
    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, Path(args.clip_cache).resolve(strict=True))
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    prompt_before = tuple(entry.parameter.detach().clone() for entry in index)
    frozen_before = hash_frozen_parameters(model)
    optimizer_before = copy.deepcopy(trainer.optim.state_dict())
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())

    iterator = iter(trainer.train_loader_x)
    first = trainer.parse_batch_train(next(iterator))
    second = trainer.parse_batch_train(next(iterator))
    g0 = _capture_gradient(model, first, trainer.optim, index)
    g1 = _capture_gradient(model, second, trainer.optim, index)
    trainer.optim.zero_grad(set_to_none=True)

    loaded = load_dataset(data_root, DATASET_SPECS["dtd"])
    source = load_full_gradient_source(
        loaded,
        Path(args.manifest_root).resolve(strict=True)
        / "dtd"
        / "shots_4"
        / "seed_1"
        / "data_manifest.json",
    )

    def service(tag):
        loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
        query = FullGradientService(
            model=model,
            param_index=index,
            loader=loader,
            precision_controller=PrecisionController("fp32"),
            protocol_seed=1,
            dataset="dtd",
            shots=4,
            config_hash=f"task19-{tag}-v1",
        )
        return _CountingService(query)

    schedule = DiagnosticSchedule(1)
    cases = {}

    ema_service = service("ema")
    ema = EMAEstimator(index, ema_lambda=0.15)
    ema_result = ema.global_direction(batch_grad=g0, optimizer_step=0)
    ema_before = copy.deepcopy(ema.state_dict())
    ema_reference = DiagnosticCoordinator(
        schedule=schedule, full_gradient_service=ema_service
    ).reference_for_step(ema_result, optimizer_step=0, epoch=0, batch_index=0)
    if not _nested_equal(ema_before, ema.state_dict()):
        raise AssertionError("EMA diagnostic query mutated estimator state")
    cases["ema_independent"] = {
        "optimization_without_diagnostic_calls": 0,
        "total_calls_with_diagnostic": len(ema_service.calls),
        "source": ema_reference.source,
        "query_issued": ema_reference.exact_service_query_issued,
        "calls": ema_service.calls,
        "estimator_state_unchanged": True,
    }

    exact_service = service("exact")
    exact = ExactEstimator(index, full_gradient_service=exact_service)
    exact_result = exact.global_direction(batch_grad=g0, optimizer_step=0)
    exact_reference = DiagnosticCoordinator(
        schedule=schedule, full_gradient_service=exact_service
    ).reference_for_step(exact_result, optimizer_step=0, epoch=0, batch_index=0)
    cases["exact_reuse"] = {
        "total_calls_with_diagnostic": len(exact_service.calls),
        "source": exact_reference.source,
        "query_issued": exact_reference.exact_service_query_issued,
        "calls": exact_service.calls,
    }

    refresh_service = service("periodic-refresh")
    refresh_estimator = PeriodicEstimator(
        index,
        ema_lambda=0.15,
        refresh_k_steps=2,
        full_gradient_service=refresh_service,
    )
    refresh_result = refresh_estimator.global_direction(
        batch_grad=g0, optimizer_step=0
    )
    refresh_reference = DiagnosticCoordinator(
        schedule=schedule, full_gradient_service=refresh_service
    ).reference_for_step(refresh_result, optimizer_step=0, epoch=0, batch_index=0)
    cases["periodic_refresh_reuse"] = {
        "total_calls_with_diagnostic": len(refresh_service.calls),
        "source": refresh_reference.source,
        "query_issued": refresh_reference.exact_service_query_issued,
        "calls": refresh_service.calls,
    }

    nonrefresh_service = service("periodic-nonrefresh")
    nonrefresh_estimator = PeriodicEstimator(
        index,
        ema_lambda=0.15,
        refresh_k_steps=2,
        full_gradient_service=nonrefresh_service,
    )
    nonrefresh_estimator.global_direction(batch_grad=g0, optimizer_step=0)
    nonrefresh_result = nonrefresh_estimator.global_direction(
        batch_grad=g1, optimizer_step=1
    )
    nonrefresh_before = copy.deepcopy(nonrefresh_estimator.state_dict())
    nonrefresh_reference = DiagnosticCoordinator(
        schedule=schedule, full_gradient_service=nonrefresh_service
    ).reference_for_step(nonrefresh_result, optimizer_step=1, epoch=0, batch_index=1)
    if not _nested_equal(nonrefresh_before, nonrefresh_estimator.state_dict()):
        raise AssertionError("Periodic non-refresh diagnostic mutated estimator")
    cases["periodic_nonrefresh_independent"] = {
        "total_calls_including_step0_refresh": len(nonrefresh_service.calls),
        "diagnostic_calls_at_step1": sum(
            call["optimizer_step"] == 1 and call["purpose"] == "diagnostic"
            for call in nonrefresh_service.calls
        ),
        "source": nonrefresh_reference.source,
        "query_issued": nonrefresh_reference.exact_service_query_issued,
        "calls": nonrefresh_service.calls,
        "last_refresh_step": nonrefresh_estimator.last_refresh_step,
        "age_steps": nonrefresh_estimator.age_steps,
        "estimator_state_unchanged": True,
    }

    expected = {
        "ema_independent": (1, "independent_diagnostic_query", True),
        "exact_reuse": (1, "exact_estimator_reuse", False),
        "periodic_refresh_reuse": (1, "periodic_refresh_reuse", False),
    }
    for name, (count, source_name, issued) in expected.items():
        case = cases[name]
        if case["total_calls_with_diagnostic"] != count or case["source"] != source_name or case["query_issued"] != issued:
            raise AssertionError(f"Real query provenance differs for {name}")
    nonrefresh = cases["periodic_nonrefresh_independent"]
    if (
        nonrefresh["total_calls_including_step0_refresh"] != 2
        or nonrefresh["diagnostic_calls_at_step1"] != 1
        or nonrefresh["last_refresh_step"] != 0
        or nonrefresh["age_steps"] != 1
    ):
        raise AssertionError("Periodic non-refresh diagnostic behavior differs")

    if not all(torch.equal(entry.parameter, value) for entry, value in zip(index, prompt_before)):
        raise AssertionError("Diagnostic scheduling changed prompt parameters")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Diagnostic scheduling changed frozen CLIP")
    if not _nested_equal(optimizer_before, trainer.optim.state_dict()) or not _nested_equal(scheduler_before, trainer.sched.state_dict()):
        raise AssertionError("Diagnostic scheduling changed optimizer/scheduler")

    payload = {
        "schema_version": "sample_fg.task19_diagnostic_schedule.v1",
        "status": "PASS",
        "smoke": True,
        "allow_scientific_summary": False,
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "backbone": "ViT-B/16",
        "precision": "fp32",
        "diagnostic_interval_steps": 1,
        "cases": cases,
        "all_queries_at_unperturbed_theta": True,
        "optimizer_steps": 0,
        "prompt_unchanged": True,
        "frozen_clip_unchanged": True,
        "optimizer_unchanged": True,
        "scheduler_unchanged": True,
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
