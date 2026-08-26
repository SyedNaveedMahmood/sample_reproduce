"""Validate uninterrupted-vs-resumed real DTD periodic SAMPLe execution."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointProgress,
    load_scientific_checkpoint,
    save_scientific_checkpoint,
)
from sample_fg.coop_anchor import (
    EXPECTED_CLIP_SHA256,
    build_coop_trainer,
    build_smoke_cfg,
    hash_frozen_parameters,
    sha256_file,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.environment import capture_environment
from sample_fg.estimators import PeriodicEstimator
from sample_fg.full_gradient import (
    FullGradientService,
    FullGradientSource,
    build_full_gradient_loader,
    load_full_gradient_source,
)
from sample_fg.param_index import ParamIndex
from sample_fg.perturbation import PromptPerturbation
from sample_fg.precision import PrecisionController
from sample_fg.results import (
    METRICS_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    RunAccounting,
    RunArtifacts,
    RunIdentity,
    atomic_write_json,
    bind_run_identity,
    load_jsonl,
    resolve_config,
)
from sample_fg.rng import RNGSnapshot, capture_rng_state
from sample_fg.step_engine import StepEngine


COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
TASK20_SHA = "713056f35eb224de0bd3babf31037cbc94361bc5"


@dataclass
class _Runtime:
    trainer: object
    cfg: object
    model: torch.nn.Module
    index: ParamIndex
    precision: PrecisionController
    perturbation: PromptPerturbation
    engine: StepEngine
    estimator: PeriodicEstimator
    full_loader: object
    frozen_hash: str


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _rng_equal(left: RNGSnapshot, right: RNGSnapshot) -> bool:
    return (
        left.python_state == right.python_state
        and left.numpy_state[0] == right.numpy_state[0]
        and np.array_equal(left.numpy_state[1], right.numpy_state[1])
        and left.numpy_state[2:] == right.numpy_state[2:]
        and torch.equal(left.torch_cpu_state, right.torch_cpu_state)
        and left.cuda_was_initialized == right.cuda_was_initialized
        and len(left.torch_cuda_states) == len(right.torch_cuda_states)
        and all(torch.equal(a, b) for a, b in zip(left.torch_cuda_states, right.torch_cuda_states))
        and len(left.explicit_generators) == len(right.explicit_generators)
        and all(
            a.device == b.device and torch.equal(a.state, b.state)
            for a, b in zip(left.explicit_generators, right.explicit_generators)
        )
    )


def _cfg(data_root: Path, output: Path):
    cfg = build_smoke_cfg(REPO_ROOT, data_root, output, "base")
    cfg.defrost()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TRAINER.COOP.PREC = "fp32"
    cfg.freeze()
    return cfg


def _config(
    *, cfg, data_root: Path, output_root: Path, clip_checkpoint: Path,
    source: FullGradientSource,
) -> dict[str, object]:
    return resolve_config(
        {
            "run": {
                "experiment_id": "T21",
                "output_root": str(output_root),
                "notes": "bounded Task-21 interrupted-vs-uninterrupted resume smoke",
                "smoke": True,
            },
            "data": {
                "dataset": "dtd", "root": str(data_root), "shots": 4, "seed": 1,
                "class_subsample": "base", "split_policy": "official_coop_fixed",
                "require_split_checksum": True, "train_batch_size": 2,
                "test_batch_size": int(cfg.DATALOADER.TEST.BATCH_SIZE), "num_workers": 0,
                "preserve_upstream_drop_last": True,
                "augmentation_policy": "pinned_coop_train_transform",
                "seed_policy": "coop_legacy_plus_isolated_fullgrad_v1",
                "selected_source_fingerprint": source.fingerprint,
                "selected_count": len(source),
            },
            "model": {
                "backbone": "ViT-B/16", "prompt_learner": "CoOp",
                "nominal_n_ctx": int(cfg.TRAINER.COOP.N_CTX), "effective_n_ctx": 4,
                "ctx_init": str(cfg.TRAINER.COOP.CTX_INIT),
                "class_specific_context": bool(cfg.TRAINER.COOP.CSC),
                "class_token_position": str(cfg.TRAINER.COOP.CLASS_TOKEN_POSITION),
                "freeze_clip": True, "checkpoint_path": str(clip_checkpoint),
                "checkpoint_sha256": EXPECTED_CLIP_SHA256,
            },
            "method": {
                "name": "sample", "rho": 0.05, "alpha": 0.0015,
                "ema_lambda": 0.15, "norm_eps": 1e-12,
                "nonfinite_policy": "abort", "first_order_stop_gradient": True,
            },
            "estimator": {
                "mode": "periodic", "refresh_k_steps": 2,
                "full_gradient_micro_batch_size": 32, "full_gradient_num_workers": 0,
                "full_gradient_transform_policy": "train_aug_isolated_conditional_exact_v1",
                "full_gradient_accum_dtype": "fp32",
            },
            "diagnostics": {
                "enabled": False, "full_gradient_interval_steps": None,
                "log_step_interval": 1, "eval_interval_epochs": None,
                "store_gradient_vectors": False, "write_separate_jsonl": True,
                "purity_assertions": True,
            },
            "optim": {
                "name": str(cfg.OPTIM.NAME).lower(), "lr": float(cfg.OPTIM.LR),
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
                "device": "cuda:0", "precision": "fp32",
                "gradient_state_dtype": "fp32", "deterministic_algorithms": False,
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_sync_timing": True,
            },
            "checkpoint": {
                "enabled": True, "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "recovery_interval_steps": 3, "save_final": True, "save_best": False,
                "resume_from": None, "strict_config_match": True,
                "save_rng_state": True,
                "boundary": "after_logical_optimizer_step_unperturbed_v1",
                "normal_loader_resume": "workers0_epoch_rng_replay_v1",
            },
            "logging": {
                "metrics_format": "jsonl", "console_level": "human_only",
                "capture_package_freeze": True, "capture_git_diff_hash": True,
                "capture_gpu_memory": True, "schema_version": "sample_fg.logging.v1",
            },
            "smoke": {
                "max_optimizer_steps": 6, "limit_train_samples_per_class": None,
                "limit_eval_samples": None, "force_num_workers_zero": True,
                "allow_scientific_summary": False,
            },
            "provenance": {
                "coop_upstream_commit": COOP_UPSTREAM_SHA,
                "coop_task20_commit": TASK20_SHA,
                "dassl_commit": DASSL_SHA,
            },
        }
    )


def _build_runtime(
    *, data_root: Path, clip_cache: Path, output: Path,
    source: FullGradientSource, config_hash: str,
) -> _Runtime:
    set_random_seed(1)
    cfg = _cfg(data_root, output)
    trainer = build_coop_trainer(cfg, clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    if index.names != ("prompt_learner.ctx",) or index[0].shape != (4, 512):
        raise AssertionError("Real CoOp ParamIndex differs")
    full_loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    service = FullGradientService(
        model=model, param_index=index, loader=full_loader,
        precision_controller=PrecisionController("fp32"), protocol_seed=1,
        dataset="dtd", shots=4, config_hash=config_hash,
    )
    estimator = PeriodicEstimator(
        index, ema_lambda=0.15, refresh_k_steps=2,
        full_gradient_service=service,
    )
    precision = PrecisionController("fp32")
    perturbation = PromptPerturbation(index)
    engine = StepEngine(
        param_index=index, optimizer=trainer.optim,
        precision_controller=precision, rho=0.05, alpha=0.0015,
        perturbation=perturbation,
    )
    return _Runtime(
        trainer, cfg, model, index, precision, perturbation, engine,
        estimator, full_loader, hash_frozen_parameters(model),
    )


def _run_steps(runtime: _Runtime, iterator, start: int, end: int):
    records = []
    dataset_root = (Path(runtime.cfg.DATASET.ROOT).resolve() / "dtd").resolve()
    for expected_step in range(start, end):
        raw = next(iterator)
        sample_ids = tuple(
            Path(str(item)).resolve().relative_to(dataset_root).as_posix()
            for item in raw.get("impath", ())
        )
        batch = runtime.trainer.parse_batch_train(raw)
        record = runtime.engine.step_sample(
            batch,
            lambda item: F.cross_entropy(runtime.model(item[0]), item[1]),
            runtime.estimator,
            epoch=0,
            batch_index=expected_step,
        )
        if record.optimizer_step != expected_step:
            raise AssertionError("Logical optimizer step differs")
        metadata = record.estimator_result.full_gradient_metadata
        records.append(
            {
                "optimizer_step": record.optimizer_step,
                "batch_index": expected_step,
                "sample_ids": list(sample_ids),
                "loss_current": record.loss_current,
                "loss_displaced": record.loss_displaced,
                "batch_gradient_norm": record.batch_gradient_norm,
                "global_direction_norm": record.global_direction_norm,
                "final_gradient_norm": record.final_gradient_norm,
                "refreshed": record.estimator_result.refreshed,
                "age_steps": record.estimator_result.age_steps,
                "last_refresh_step": record.estimator_result.last_refresh_step,
                "exact_query_count": record.estimator_result.exact_query_count,
                "full_gradient_elapsed_s": metadata.elapsed_s if metadata else 0.0,
                "full_gradient_sample_count": metadata.sample_count if metadata else 0,
            }
        )
    return records


def _deterministic_record(record):
    return {key: value for key, value in record.items() if key != "full_gradient_elapsed_s"}


def _capture_final(runtime: _Runtime, explicit) -> dict[str, object]:
    return {
        "prompt": tuple(entry.parameter.detach().clone() for entry in runtime.index),
        "optimizer": copy.deepcopy(runtime.trainer.optim.state_dict()),
        "scheduler": copy.deepcopy(runtime.trainer.sched.state_dict()),
        "precision": copy.deepcopy(runtime.precision.state_dict()),
        "estimator": copy.deepcopy(runtime.estimator.state_dict()),
        "engine": copy.deepcopy(runtime.engine.state_dict()),
        "rng": capture_rng_state(explicit),
        "frozen_hash": hash_frozen_parameters(runtime.model),
    }


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Task-21 real validation requires CUDA")
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if subprocess.run(
        ["git", "-c", f"safe.directory={REPO_ROOT}", "merge-base", "--is-ancestor", TASK20_SHA, "HEAD"],
        cwd=REPO_ROOT, check=False,
    ).returncode:
        raise AssertionError("Accepted Task-20 commit is not an ancestor of HEAD")
    if _git(dassl_root, "rev-parse", "HEAD") != DASSL_SHA or _git(dassl_root, "status", "--short"):
        raise AssertionError("Pinned Dassl checkout changed")

    data_root = Path(args.root).resolve(strict=True)
    manifest = Path(args.manifest_root).resolve(strict=True) / "dtd" / "shots_4" / "seed_1" / "data_manifest.json"
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    clip_checkpoint = clip_cache / "ViT-B-16.pt"
    if sha256_file(clip_checkpoint).lower() != EXPECTED_CLIP_SHA256:
        raise AssertionError("Pinned CLIP checkpoint hash differs")
    output_root = Path(args.output_root).resolve()
    report_path = Path(args.report).resolve()
    loaded = load_dataset(data_root, DATASET_SPECS["dtd"])
    source = load_full_gradient_source(loaded, manifest)
    base_cfg = _cfg(data_root, output_root / "task21_config_runtime")
    config = _config(
        cfg=base_cfg, data_root=data_root, output_root=output_root,
        clip_checkpoint=clip_checkpoint, source=source,
    )
    identity = RunIdentity.now(
        dataset="dtd", shots=4, method_tag="sample", estimator_tag="periodic_k2",
        seed=1, config_sha256=str(config["config_sha256"]),
        experiment_id="T21", smoke=True, allow_scientific_summary=False,
    )
    config = bind_run_identity(config, identity)
    environment = capture_environment(
        project_repo=REPO_ROOT, coop_upstream_commit=COOP_UPSTREAM_SHA,
        dassl_commit=DASSL_SHA, precision_mode="fp32", clip_backbone="ViT-B/16",
        clip_checkpoint_identifier=str(clip_checkpoint),
        clip_checkpoint_sha256=EXPECTED_CLIP_SHA256,
        capture_package_freeze=True,
    )
    artifacts = RunArtifacts(output_root, identity)
    run_dir = artifacts.create(
        resolved_config=config, environment=environment,
        data_manifest_source=manifest,
    )
    artifacts.append_log("Task-21 real checkpoint/resume smoke started")

    # Uninterrupted six-step control.
    control = _build_runtime(
        data_root=data_root, clip_cache=clip_cache,
        output=run_dir / "control_runtime", source=source,
        config_hash=identity.config_sha256,
    )
    control_iterator = iter(control.trainer.train_loader_x)
    control_records = _run_steps(control, control_iterator, 0, 6)
    control_final = _capture_final(control, (control.full_loader.generator,))
    control_frozen_initial = control.frozen_hash
    del control_iterator, control
    gc.collect()
    torch.cuda.empty_cache()

    # First half of interrupted trajectory and recovery save.
    first = _build_runtime(
        data_root=data_root, clip_cache=clip_cache,
        output=run_dir / "split_first_runtime", source=source,
        config_hash=identity.config_sha256,
    )
    explicit_first = {"full_gradient_loader": first.full_loader.generator}
    epoch_start_rng = capture_rng_state(explicit_first.values())
    first_iterator = iter(first.trainer.train_loader_x)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    split_started = time.perf_counter()
    first_records = _run_steps(first, first_iterator, 0, 3)
    recovery_path = run_dir / "checkpoints" / "recovery_step_3.pt"
    recovery_metadata = save_scientific_checkpoint(
        recovery_path, param_index=first.index, optimizer=first.trainer.optim,
        scheduler=first.trainer.sched, precision_controller=first.precision,
        step_engine=first.engine, estimator=first.estimator,
        perturbation=first.perturbation,
        progress=CheckpointProgress(3, 0, 3, 6), method="sample",
        config_sha256=identity.config_sha256,
        source_fingerprint=source.fingerprint,
        result_state={
            "optimizer_steps": 3, "normal_samples_seen": 6,
            "metrics_records": 3, "full_gradient_sweeps": first.estimator.exact_query_count,
        },
        explicit_generators=explicit_first,
        normal_loader_epoch_start_rng=epoch_start_rng,
        normal_loader_length=len(first.trainer.train_loader_x),
    )
    first_prompt_at_save = tuple(entry.parameter.detach().clone() for entry in first.index)
    del first_iterator, first
    gc.collect()
    torch.cuda.empty_cache()

    # Fresh reconstruction, transactional load, exact worker-0 batch replay.
    resumed = _build_runtime(
        data_root=data_root, clip_cache=clip_cache,
        output=run_dir / "split_resumed_runtime", source=source,
        config_hash=identity.config_sha256,
    )
    explicit_resumed = {"full_gradient_loader": resumed.full_loader.generator}
    load_result = load_scientific_checkpoint(
        recovery_path, param_index=resumed.index, optimizer=resumed.trainer.optim,
        scheduler=resumed.trainer.sched, precision_controller=resumed.precision,
        step_engine=resumed.engine, estimator=resumed.estimator,
        perturbation=resumed.perturbation, expected_method="sample",
        expected_config_sha256=identity.config_sha256,
        expected_source_fingerprint=source.fingerprint,
        explicit_generators=explicit_resumed,
    )
    if not all(torch.equal(value, entry.parameter) for value, entry in zip(first_prompt_at_save, resumed.index)):
        raise AssertionError("Recovery load did not restore the saved prompt")
    resumed_iterator = load_result.resume_worker0_loader(
        resumed.trainer.train_loader_x, explicit_generators=explicit_resumed
    )
    resumed_records = _run_steps(resumed, resumed_iterator, 3, 6)
    torch.cuda.synchronize()
    split_elapsed = time.perf_counter() - split_started
    split_records = first_records + resumed_records
    resumed_final = _capture_final(resumed, explicit_resumed.values())

    deterministic_control = [_deterministic_record(item) for item in control_records]
    deterministic_split = [_deterministic_record(item) for item in split_records]
    checks = {
        "step_records": deterministic_control == deterministic_split,
        "prompt": all(torch.equal(a, b) for a, b in zip(control_final["prompt"], resumed_final["prompt"])),
        "optimizer": _nested_equal(control_final["optimizer"], resumed_final["optimizer"]),
        "scheduler": _nested_equal(control_final["scheduler"], resumed_final["scheduler"]),
        "precision": _nested_equal(control_final["precision"], resumed_final["precision"]),
        "estimator": _nested_equal(control_final["estimator"], resumed_final["estimator"]),
        "engine": _nested_equal(control_final["engine"], resumed_final["engine"]),
        "rng": _rng_equal(control_final["rng"], resumed_final["rng"]),
        "normal_batch_ids": [item["sample_ids"] for item in control_records] == [item["sample_ids"] for item in split_records],
        "frozen_clip": control_final["frozen_hash"] == resumed_final["frozen_hash"] == control_frozen_initial == resumed.frozen_hash,
    }
    if not all(checks.values()):
        raise AssertionError(f"Real checkpoint/resume comparison failed: {checks}")
    if [item["refreshed"] for item in split_records] != [True, False, True, False, True, False]:
        raise AssertionError("Periodic refresh clock changed after resume")
    if resumed.estimator.exact_query_count != 3 or resumed.engine.optimizer_step != 6:
        raise AssertionError("Final periodic query/optimizer counters differ")

    epoch_start_resumed = load_result.epoch_start_rng_snapshot(
        explicit_generators=explicit_resumed
    )
    final_path = run_dir / "checkpoints" / "final.pt"
    final_metadata = save_scientific_checkpoint(
        final_path, param_index=resumed.index, optimizer=resumed.trainer.optim,
        scheduler=resumed.trainer.sched, precision_controller=resumed.precision,
        step_engine=resumed.engine, estimator=resumed.estimator,
        perturbation=resumed.perturbation,
        progress=CheckpointProgress(6, 0, 6, 12), method="sample",
        config_sha256=identity.config_sha256,
        source_fingerprint=source.fingerprint,
        result_state={
            "optimizer_steps": 6, "normal_samples_seen": 12,
            "metrics_records": 6, "full_gradient_sweeps": 3,
        },
        explicit_generators=explicit_resumed,
        normal_loader_epoch_start_rng=epoch_start_resumed,
        normal_loader_length=len(resumed.trainer.train_loader_x),
    )
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())

    for record in split_records:
        artifacts.append_metric(
            {
                "schema_version": METRICS_SCHEMA_VERSION,
                "event_type": "train_step", "run_id": identity.run_id,
                "experiment_id": identity.experiment_id, "dataset": "dtd",
                "shots": 4, "seed": 1, "method": "sample",
                "estimator_mode": "periodic", "periodic_k_steps": 2,
                "epoch": 0, "batch_index": record["batch_index"],
                "optimizer_step": record["optimizer_step"],
                "wall_time_utc": _utc_now(),
                "loss/current": record["loss_current"],
                "loss/displaced": record["loss_displaced"],
                "grad/batch_norm": record["batch_gradient_norm"],
                "grad/global_estimate_norm": record["global_direction_norm"],
                "grad/update_norm": record["final_gradient_norm"],
                "estimator/refreshed": record["refreshed"],
                "estimator/age_steps": record["age_steps"],
                "estimator/last_refresh_step": record["last_refresh_step"],
                "estimator/exact_query_count": record["exact_query_count"],
                "full_gradient/elapsed_s": record["full_gradient_elapsed_s"],
                "full_gradient/sample_count": record["full_gradient_sample_count"],
            }
        )
    for kind, metadata in (("recovery", recovery_metadata), ("final", final_metadata)):
        artifacts.append_metric(
            {
                "schema_version": METRICS_SCHEMA_VERSION,
                "event_type": "checkpoint", "checkpoint_kind": kind,
                "run_id": identity.run_id, "experiment_id": identity.experiment_id,
                "dataset": "dtd", "shots": 4, "seed": 1, "method": "sample",
                "estimator_mode": "periodic", "periodic_k_steps": 2,
                "epoch": metadata.epoch_zero_based,
                "batch_index": metadata.next_batch_index_zero_based,
                "optimizer_step": metadata.next_optimizer_step,
                "wall_time_utc": _utc_now(), "checkpoint": metadata.as_dict(),
            }
        )
    full_gradient_time = sum(item["full_gradient_elapsed_s"] for item in split_records)
    accounting = RunAccounting(
        train_total_s=split_elapsed, full_gradient_total_s=full_gradient_time,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
    )
    for name, amount in {
        "optimizer_steps": 6, "normal_forward_calls": 12,
        "normal_backward_calls": 12, "normal_samples_processed": 12,
        "full_gradient_sweeps": 3, "full_gradient_forward_calls": 9,
        "full_gradient_autograd_grad_calls": 9,
        "full_gradient_samples_processed": 288, "checkpoints_written": 2,
    }.items():
        accounting.increment(name, amount)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_identity": identity.as_dict(), "status": "completed",
        "smoke": True, "allow_scientific_summary": False,
        "evaluation": {
            "base_accuracy_pct": None, "new_accuracy_pct": None, "hm_pct": None,
            "unavailable_reason": "Task-21 resume smoke performs no evaluation",
        },
        "efficiency": accounting.as_dict(),
        "estimator_diagnostics": {"num_exact_reference_points": 0},
        "resume_validation": {
            "control_steps": 6, "split_step": 3,
            "resumed_steps": 3, "checks": checks,
            "refresh_flags": [item["refreshed"] for item in split_records],
            "exact_query_count": resumed.estimator.exact_query_count,
        },
        "artifacts": {
            "recovery_checkpoint": "checkpoints/recovery_step_3.pt",
            "final_checkpoint": "checkpoints/final.pt",
            "metrics": "metrics.jsonl", "log": "logs/run.log",
        },
    }
    artifacts.write_summary(summary)
    artifacts.append_log("Task-21 real checkpoint/resume smoke completed")
    if len(load_jsonl(artifacts.metrics_path)) != 8:
        raise AssertionError("Task-21 metrics/checkpoint event count differs")
    if load_jsonl(artifacts.diagnostics_path):
        raise AssertionError("Task-21 unexpectedly wrote diagnostic events")

    report = {
        "schema_version": "sample_fg.task21_checkpoint_resume.v1",
        "status": "PASS", "smoke": True, "allow_scientific_summary": False,
        "run_dir": str(run_dir), "run_id": identity.run_id,
        "checkpoint_schema": CHECKPOINT_SCHEMA_VERSION,
        "recovery_checkpoint": recovery_metadata.as_dict(),
        "final_checkpoint": final_metadata.as_dict(),
        "control_steps": 6, "split_before_save_steps": 3,
        "resumed_steps": 3, "final_optimizer_step": resumed.engine.optimizer_step,
        "final_exact_query_count": resumed.estimator.exact_query_count,
        "refresh_flags": [item["refreshed"] for item in split_records],
        "age_steps": [item["age_steps"] for item in split_records],
        "last_refresh_steps": [item["last_refresh_step"] for item in split_records],
        "checks": checks, "worker0_loader_replay": True,
        "normal_batch_ids": [item["sample_ids"] for item in split_records],
        "checkpoint_metrics_records": 2, "train_metrics_records": 6,
        "split_train_total_s": split_elapsed,
        "split_full_gradient_total_s": full_gradient_time,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
    }
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
