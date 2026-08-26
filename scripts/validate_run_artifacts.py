"""Create and validate one bounded real periodic-SAMPLe run artifact tree."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    EXPECTED_CLIP_SHA256,
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    hash_frozen_parameters,
    sha256_file,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.diagnostic_schedule import DiagnosticCoordinator, DiagnosticSchedule
from sample_fg.environment import capture_environment
from sample_fg.estimators import PeriodicEstimator
from sample_fg.full_gradient import (
    FullGradientService,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.param_index import ParamIndex
from sample_fg.precision import PrecisionController
from sample_fg.results import (
    METRICS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    RunAccounting,
    RunArtifacts,
    RunIdentity,
    bind_run_identity,
    load_jsonl,
    resolve_config,
)
from sample_fg.step_engine import StepEngine


COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
TASK19_SHA = "3fe68e0e86f3f4be6cde178414faa1c424ed7203"


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolved_config(
    *,
    cfg,
    data_root: Path,
    output_root: Path,
    clip_checkpoint: Path,
) -> dict[str, object]:
    return resolve_config(
        {
            "run": {
                "experiment_id": "T20",
                "output_root": str(output_root),
                "notes": "bounded Task-20 artifact writer validation",
                "smoke": True,
            },
            "data": {
                "dataset": "dtd",
                "root": str(data_root),
                "shots": 4,
                "seed": 1,
                "class_subsample": "base",
                "split_policy": "official_coop_fixed",
                "require_split_checksum": True,
                "train_batch_size": 2,
                "test_batch_size": int(cfg.DATALOADER.TEST.BATCH_SIZE),
                "num_workers": 0,
                "preserve_upstream_drop_last": True,
                "augmentation_policy": "pinned_coop_train_transform",
                "seed_policy": "coop_legacy_plus_isolated_fullgrad_v1",
            },
            "model": {
                "backbone": "ViT-B/16",
                "prompt_learner": "CoOp",
                "nominal_n_ctx": int(cfg.TRAINER.COOP.N_CTX),
                "effective_n_ctx": 4,
                "ctx_init": str(cfg.TRAINER.COOP.CTX_INIT),
                "class_specific_context": bool(cfg.TRAINER.COOP.CSC),
                "class_token_position": str(cfg.TRAINER.COOP.CLASS_TOKEN_POSITION),
                "freeze_clip": True,
                "checkpoint_path": str(clip_checkpoint),
                "checkpoint_sha256": EXPECTED_CLIP_SHA256,
            },
            "method": {
                "name": "sample",
                "rho": 0.05,
                "alpha": 0.0015,
                "ema_lambda": 0.15,
                "norm_eps": 1e-12,
                "nonfinite_policy": "abort",
                "first_order_stop_gradient": True,
            },
            "estimator": {
                "mode": "periodic",
                "refresh_k_steps": 2,
                "full_gradient_micro_batch_size": 32,
                "full_gradient_num_workers": 0,
                "full_gradient_transform_policy": "train_aug_isolated_conditional_exact_v1",
                "full_gradient_accum_dtype": "fp32",
            },
            "diagnostics": {
                "enabled": True,
                "full_gradient_interval_steps": 1,
                "log_step_interval": 1,
                "eval_interval_epochs": None,
                "store_gradient_vectors": False,
                "write_separate_jsonl": True,
                "purity_assertions": True,
            },
            "optim": {
                "name": str(cfg.OPTIM.NAME).lower(),
                "lr": float(cfg.OPTIM.LR),
                "weight_decay": float(cfg.OPTIM.WEIGHT_DECAY),
                "momentum": float(cfg.OPTIM.MOMENTUM),
                "nesterov": bool(cfg.OPTIM.SGD_NESTEROV),
                "max_epoch": int(cfg.OPTIM.MAX_EPOCH),
                "scheduler": str(cfg.OPTIM.LR_SCHEDULER),
                "warmup_epoch": int(cfg.OPTIM.WARMUP_EPOCH),
                "warmup_type": str(cfg.OPTIM.WARMUP_TYPE),
                "warmup_cons_lr": float(cfg.OPTIM.WARMUP_CONS_LR),
                "scheduler_step_unit": "epoch",
            },
            "runtime": {
                "device": "cuda:0",
                "precision": "fp32",
                "gradient_state_dtype": "fp32",
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_sync_timing": True,
            },
            "checkpoint": {
                "enabled": False,
                "reason": "Task-20 artifact smoke; complete checkpoints begin at Task 21",
                "recovery_interval_steps": None,
                "save_final": False,
                "save_best": False,
                "resume_from": None,
                "strict_config_match": True,
                "save_rng_state": True,
            },
            "logging": {
                "metrics_format": "jsonl",
                "console_level": "human_only",
                "capture_package_freeze": True,
                "capture_git_diff_hash": True,
                "capture_gpu_memory": True,
                "schema_version": "sample_fg.logging.v1",
            },
            "smoke": {
                "max_optimizer_steps": 1,
                "limit_train_samples_per_class": None,
                "limit_eval_samples": None,
                "force_num_workers_zero": True,
                "allow_scientific_summary": False,
            },
            "provenance": {
                "coop_upstream_commit": COOP_UPSTREAM_SHA,
                "coop_task19_commit": TASK19_SHA,
                "dassl_commit": DASSL_SHA,
            },
        }
    )


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Task-20 real validation requires CUDA")
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK19_SHA, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    ).returncode:
        raise AssertionError("Accepted Task-19 commit is not an ancestor of HEAD")
    if _git(dassl_root, "rev-parse", "HEAD") != DASSL_SHA or _git(dassl_root, "status", "--short"):
        raise AssertionError("Pinned Dassl checkout changed")

    output_root = Path(args.output_root).resolve()
    data_root = Path(args.root).resolve(strict=True)
    manifest = Path(args.manifest_root).resolve(strict=True) / "dtd" / "shots_4" / "seed_1" / "data_manifest.json"
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    clip_checkpoint = clip_cache / "ViT-B-16.pt"
    if sha256_file(clip_checkpoint).lower() != EXPECTED_CLIP_SHA256:
        raise AssertionError("Pinned CLIP checkpoint hash differs")

    cfg = build_smoke_cfg(REPO_ROOT, data_root, output_root / "task20_runtime", "base")
    cfg.defrost()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.freeze()
    config = _resolved_config(
        cfg=cfg,
        data_root=data_root,
        output_root=output_root,
        clip_checkpoint=clip_checkpoint,
    )
    identity = RunIdentity.now(
        dataset="dtd",
        shots=4,
        method_tag="sample",
        estimator_tag="periodic_k2",
        seed=1,
        config_sha256=str(config["config_sha256"]),
        experiment_id="T20",
        smoke=True,
        allow_scientific_summary=False,
    )
    config = bind_run_identity(config, identity)
    environment = capture_environment(
        project_repo=REPO_ROOT,
        coop_upstream_commit=COOP_UPSTREAM_SHA,
        dassl_commit=DASSL_SHA,
        precision_mode="fp32",
        clip_backbone="ViT-B/16",
        clip_checkpoint_identifier=str(clip_checkpoint),
        clip_checkpoint_sha256=EXPECTED_CLIP_SHA256,
        capture_package_freeze=True,
    )
    artifacts = RunArtifacts(output_root, identity)
    run_dir = artifacts.create(
        resolved_config=config,
        environment=environment,
        data_manifest_source=manifest,
    )
    artifacts.append_log("Task-20 bounded periodic-SAMPLe smoke started")

    set_random_seed(cfg.SEED)
    trainer = build_coop_trainer(cfg, clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    if index.names != ("prompt_learner.ctx",):
        raise AssertionError("Real CoOp trainable subspace changed")
    prompt_before = index[0].parameter.detach().clone()
    frozen_before = hash_frozen_parameters(model)
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    loaded = load_dataset(data_root, DATASET_SPECS["dtd"])
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
        config_hash=str(config["config_sha256"]),
    )
    estimator = PeriodicEstimator(
        index,
        ema_lambda=0.15,
        refresh_k_steps=2,
        full_gradient_service=service,
    )
    coordinator = DiagnosticCoordinator(
        schedule=DiagnosticSchedule(1), full_gradient_service=service
    )
    engine = StepEngine(
        param_index=index,
        optimizer=trainer.optim,
        precision_controller=PrecisionController("fp32"),
        rho=0.05,
        alpha=0.0015,
        diagnostic_coordinator=coordinator,
    )
    raw_batch = next(iter(trainer.train_loader_x))
    batch = trainer.parse_batch_train(raw_batch)
    lr = float(trainer.get_current_lr())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with count_optimizer_steps(trainer.optim) as counter:
        record = engine.step_sample(
            batch,
            lambda item: F.cross_entropy(model(item[0]), item[1]),
            estimator,
            epoch=0,
            batch_index=0,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if counter["count"] != 1 or engine.optimizer_step != 1:
        raise AssertionError("Bounded artifact smoke did not perform exactly one step")
    if record.diagnostic_event is None:
        raise AssertionError("Scheduled diagnostic event is missing")
    if record.diagnostic_event.reference.source != "periodic_refresh_reuse":
        raise AssertionError("Task-20 periodic diagnostic did not reuse refresh")
    metadata = record.estimator_result.full_gradient_metadata
    if metadata is None or estimator.exact_query_count != 1:
        raise AssertionError("Task-20 periodic refresh accounting differs")
    if torch.equal(index[0].parameter, prompt_before):
        raise AssertionError("Prompt did not update")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Frozen CLIP changed")
    if trainer.sched.state_dict() != scheduler_before:
        raise AssertionError("Epoch scheduler changed during one bounded step")

    common = {
        "run_id": identity.run_id,
        "experiment_id": identity.experiment_id,
        "dataset": "dtd",
        "shots": 4,
        "seed": 1,
        "method": "sample",
        "estimator_mode": "periodic",
        "periodic_k_steps": 2,
        "epoch": 0,
        "batch_index": 0,
        "optimizer_step": record.optimizer_step,
        "wall_time_utc": _utc_now(),
    }
    metric = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "event_type": "train_step",
        **common,
        "loss/current": record.loss_current,
        "loss/displaced": record.loss_displaced,
        "loss/sample_objective": record.loss_sample_objective,
        "optim/learning_rate": lr,
        "optim/nonfinite_event": False,
        "grad/batch_norm": record.batch_gradient_norm,
        "grad/global_estimate_norm": record.global_direction_norm,
        "grad/batch_component_norm": record.batch_component_norm,
        "grad/perturbed_norm": record.perturbed_gradient_norm,
        "grad/update_norm": record.final_gradient_norm,
        "grad/sam_perturb_norm": record.sam_perturbation_norm,
        "grad/batch_correction_norm": record.batch_correction_norm,
        "grad/total_displacement_norm": record.total_displacement_norm,
        "grad/xi": record.projection.xi,
        "grad/sigma": record.projection.sigma,
        "grad/projection_coefficient": record.projection.projection_coefficient,
        "estimator/refreshed": record.estimator_result.refreshed,
        "estimator/age_steps": record.estimator_result.age_steps,
        "estimator/exact_query_count": record.estimator_result.exact_query_count,
        "timing/train_step_s": elapsed,
    }
    artifacts.append_metric(metric)
    diagnostic = {**record.diagnostic_event.as_dict(), **common}
    diagnostic["full_gradient_elapsed_s"] = metadata.elapsed_s
    artifacts.append_diagnostic(diagnostic)

    accounting = RunAccounting(
        train_total_s=elapsed,
        full_gradient_total_s=metadata.elapsed_s,
        peak_cuda_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()),
    )
    for name, amount in {
        "optimizer_steps": 1,
        "normal_forward_calls": 2,
        "normal_backward_calls": 2,
        "full_gradient_sweeps": 1,
        "full_gradient_forward_calls": metadata.forward_calls,
        "full_gradient_autograd_grad_calls": metadata.autograd_grad_calls,
        "exact_reference_points": 1,
    }.items():
        accounting.increment(name, amount)
    values = record.diagnostic_event.metrics.as_dict()
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_identity": identity.as_dict(),
        "status": "completed",
        "smoke": True,
        "allow_scientific_summary": False,
        "evaluation": {
            "base_accuracy_pct": None,
            "new_accuracy_pct": None,
            "hm_pct": None,
            "unavailable_reason": "Task-20 artifact smoke performs no evaluation",
        },
        "efficiency": accounting.as_dict(),
        "estimator_diagnostics": {
            "num_exact_reference_points": 1,
            "global_estimate_exact_cosine_mean": values["grad/global_estimate_exact_cosine"],
            "global_estimate_exact_cosine_median": values["grad/global_estimate_exact_cosine"],
            "global_estimate_exact_relative_l2_mean": values["grad/global_estimate_exact_relative_l2"],
            "global_estimate_exact_log_norm_ratio_mean": values["grad/global_estimate_exact_log_norm_ratio"],
            "batch_component_exact_abs_cosine_mean": abs(values["grad/batch_component_exact_cosine"]),
            "perturbed_gradient_exact_abs_cosine_mean": abs(values["grad/perturbed_gradient_exact_cosine"]),
        },
        "artifacts": {
            "config": "config.yaml",
            "environment": "environment.json",
            "data_manifest": "data_manifest.json",
            "metrics": "metrics.jsonl",
            "gradient_diagnostics": "gradient_diagnostics.jsonl",
            "checkpoints": "checkpoints/ (empty until Task 21)",
            "log": "logs/run.log",
        },
    }
    artifacts.write_summary(summary)
    artifacts.append_log("Task-20 bounded periodic-SAMPLe smoke completed")

    with (run_dir / "config.yaml").open(encoding="utf-8") as stream:
        if yaml.safe_load(stream) != config:
            raise AssertionError("Resolved config did not round-trip")
    if json.loads((run_dir / "environment.json").read_text(encoding="utf-8")) != environment:
        raise AssertionError("Environment did not round-trip")
    if (run_dir / "data_manifest.json").read_bytes() != manifest.read_bytes():
        raise AssertionError("Authoritative data manifest was not copied byte-for-byte")
    if len(load_jsonl(artifacts.metrics_path)) != 1 or len(load_jsonl(artifacts.diagnostics_path)) != 1:
        raise AssertionError("JSONL artifact record count differs")
    parsed_summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if parsed_summary["status"] != "completed" or parsed_summary["allow_scientific_summary"]:
        raise AssertionError("Summary status/gating differs")
    if not (run_dir / "checkpoints").is_dir() or not (run_dir / "logs" / "run.log").is_file():
        raise AssertionError("Canonical run subdirectories are incomplete")

    result = {
        "status": "PASS",
        "run_dir": str(run_dir),
        "run_id": identity.run_id,
        "config_sha256": identity.config_sha256,
        "optimizer_steps": 1,
        "full_gradient_sweeps": 1,
        "diagnostic_source": record.diagnostic_event.reference.source,
        "metrics_records": 1,
        "diagnostic_records": 1,
        "manifest_byte_exact": True,
        "smoke": True,
        "allow_scientific_summary": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
