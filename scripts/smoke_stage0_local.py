"""Run the complete non-scientific local Stage-0 correctness suite."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import (
    EXPECTED_CLIP_SHA256,
    audit_prompt_only_training,
    build_coop_trainer,
    build_smoke_cfg,
    count_optimizer_steps,
    evaluate_bounded,
    hash_frozen_parameters,
    sha256_file,
    unwrap_model,
)
from sample_fg.data_protocol import DATASET_SPECS, load_dataset
from sample_fg.diagnostic_schedule import DiagnosticCoordinator, DiagnosticSchedule
from sample_fg.environment import capture_environment
from sample_fg.estimators import EMAEstimator, ExactEstimator, PeriodicEstimator
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
    RunArtifactError,
    RunArtifacts,
    RunIdentity,
    atomic_write_json,
    bind_run_identity,
    copy_artifact,
    load_jsonl,
    resolve_config,
)
from sample_fg.step_engine import SAMPLeStepRecord, SAMStepRecord, StepEngine


STAGE0_SCHEMA_VERSION = "sample_fg.stage0_suite.v1"
COOP_UPSTREAM_SHA = "ff61507c790454bce7c5052c3ac39e60772f1f89"
DASSL_SHA = "c61a1b570ac6333bd50fb5ae06aea59002fb20bb"
STARTING_SHA = "3c3bb93344a8ea51fe9704226c92f58a656011b0"
DTD_SPLIT_SHA = "F26EBECCD2B58E68D70F07A0DC39FE49BF7B69024BAC34396B954DFD87969C38"
DTD_CACHE_SHA = "81EE5B688EC9D80BBE424522A7638FCE5AAE84F07EC40928CBBA2B57B2B142AD"
REQUIRED_ARTIFACTS = (
    "config.yaml",
    "environment.json",
    "data_manifest.json",
    "metrics.jsonl",
    "gradient_diagnostics.jsonl",
    "summary.json",
    "checkpoints/final.pt",
    "logs/run.log",
)
FAILURE_PATTERN = re.compile(r"\b(?:nan|inf|traceback)\b", re.IGNORECASE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root}", *args], cwd=root, text=True
    ).strip()


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cfg(data_root: Path, output: Path, precision: str, class_subsample: str = "base"):
    cfg = build_smoke_cfg(REPO_ROOT, data_root, output, class_subsample)
    cfg.defrost()
    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = 2
    cfg.DATALOADER.TEST.BATCH_SIZE = 8
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.TRAINER.COOP.PREC = precision
    cfg.OPTIM.MAX_EPOCH = 1
    cfg.freeze()
    return cfg


def _method_tag(method: str, estimator: str | None) -> tuple[str, str]:
    if method != "sample":
        return method, "none"
    if estimator == "periodic":
        return method, "periodic_k2"
    if estimator in {"ema", "exact"}:
        return method, estimator
    raise ValueError("SAMPLe requires ema, exact, or periodic")


def _resolved_config(
    *, cfg, data_root: Path, output_root: Path, clip_checkpoint: Path,
    method: str, estimator: str | None, precision: str, steps: int,
    source_fingerprint: str,
) -> dict[str, Any]:
    method_tag, estimator_tag = _method_tag(method, estimator)
    diagnostics = method == "sample"
    return resolve_config(
        {
            "run": {
                "experiment_id": "S1",
                "output_root": str(output_root),
                "notes": "formal local Stage-0 correctness smoke; never scientific",
                "smoke": True,
            },
            "data": {
                "dataset": "dtd", "root": str(data_root), "shots": 4,
                "seed": 1, "class_subsample": "base",
                "split_policy": "official_coop_fixed",
                "require_split_checksum": True,
                "split_sha256": DTD_SPLIT_SHA,
                "fewshot_cache_sha256": DTD_CACHE_SHA,
                "train_batch_size": 2, "test_batch_size": 8,
                "num_workers": 0, "preserve_upstream_drop_last": True,
                "augmentation_policy": "pinned_coop_train_transform",
                "seed_policy": "coop_legacy_plus_isolated_fullgrad_v1",
                "selected_source_fingerprint": source_fingerprint,
                "selected_count": 96,
            },
            "model": {
                "backbone": "ViT-B/16", "prompt_learner": "CoOp",
                "nominal_n_ctx": int(cfg.TRAINER.COOP.N_CTX),
                "effective_n_ctx": 4, "ctx_init": str(cfg.TRAINER.COOP.CTX_INIT),
                "class_specific_context": bool(cfg.TRAINER.COOP.CSC),
                "class_token_position": str(cfg.TRAINER.COOP.CLASS_TOKEN_POSITION),
                "freeze_clip": True, "checkpoint_path": str(clip_checkpoint),
                "checkpoint_sha256": EXPECTED_CLIP_SHA256,
            },
            "method": {
                "name": method_tag,
                "rho": 0.05 if method in {"sam", "sample"} else None,
                "alpha": 0.0015 if method == "sample" else None,
                "ema_lambda": 0.15 if estimator in {"ema", "periodic"} else None,
                "norm_eps": 1e-12, "nonfinite_policy": "abort",
                "first_order_stop_gradient": True,
            },
            "estimator": {
                "mode": estimator or "none",
                "refresh_k_steps": 2 if estimator == "periodic" else (1 if estimator == "exact" else None),
                "full_gradient_micro_batch_size": 2,
                "full_gradient_num_workers": 0,
                "full_gradient_transform_policy": "train_aug_isolated_conditional_exact_v1",
                "full_gradient_accum_dtype": "fp32",
            },
            "diagnostics": {
                "enabled": diagnostics,
                "full_gradient_interval_steps": 1 if diagnostics else None,
                "log_step_interval": 1, "eval_interval_epochs": None,
                "store_gradient_vectors": False,
                "write_separate_jsonl": True, "purity_assertions": True,
            },
            "optim": {
                "name": str(cfg.OPTIM.NAME).lower(), "lr": float(cfg.OPTIM.LR),
                "weight_decay": float(cfg.OPTIM.WEIGHT_DECAY),
                "momentum": float(cfg.OPTIM.MOMENTUM),
                "nesterov": bool(cfg.OPTIM.SGD_NESTEROV),
                "max_epoch": 1, "scheduler": str(cfg.OPTIM.LR_SCHEDULER),
                "warmup_epoch": int(cfg.OPTIM.WARMUP_EPOCH),
                "warmup_type": str(cfg.OPTIM.WARMUP_TYPE),
                "warmup_cons_lr": float(cfg.OPTIM.WARMUP_CONS_LR),
                "scheduler_step_unit": "epoch",
            },
            "runtime": {
                "device": "cuda:0", "precision": precision,
                "gradient_state_dtype": "fp32",
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cuda_sync_timing": True,
            },
            "checkpoint": {
                "enabled": True, "recovery_interval_steps": None,
                "save_final": True, "save_best": False, "resume_from": None,
                "strict_config_match": True, "save_rng_state": True,
                "format": "pinned_coop_prompt_checkpoint",
            },
            "logging": {
                "metrics_format": "jsonl", "console_level": "human_only",
                "capture_package_freeze": True, "capture_git_diff_hash": True,
                "capture_gpu_memory": True, "schema_version": "sample_fg.logging.v1",
            },
            "smoke": {
                "max_optimizer_steps": steps,
                "limit_train_samples_per_class": None, "limit_eval_samples": 8,
                "force_num_workers_zero": True,
                "allow_scientific_summary": False,
            },
            "provenance": {
                "coop_upstream_commit": COOP_UPSTREAM_SHA,
                "coop_stage0_start_commit": STARTING_SHA,
                "dassl_commit": DASSL_SHA,
                "method_tag": method_tag, "estimator_tag": estimator_tag,
            },
        }
    )


def _common(identity: RunIdentity, *, method: str, estimator: str | None,
            step: int, batch_index: int) -> dict[str, Any]:
    return {
        "run_id": identity.run_id, "experiment_id": identity.experiment_id,
        "dataset": "dtd", "shots": 4, "seed": 1, "method": method,
        "estimator_mode": estimator, "periodic_k_steps": 2 if estimator == "periodic" else None,
        "epoch": 0, "batch_index": batch_index, "optimizer_step": step,
        "wall_time_utc": _utc_now(),
    }


def _append_sharp_metric(artifacts: RunArtifacts, identity: RunIdentity,
                         record: SAMStepRecord | SAMPLeStepRecord,
                         *, estimator: str | None, batch_index: int,
                         elapsed: float, lr: float) -> None:
    metric: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA_VERSION, "event_type": "train_step",
        **_common(identity, method=record.method, estimator=estimator,
                  step=record.optimizer_step, batch_index=batch_index),
        "loss/current": record.loss_current, "loss/displaced": record.loss_displaced,
        "loss/sample_objective": getattr(record, "loss_sample_objective", None),
        "optim/learning_rate": lr, "optim/nonfinite_event": False,
        "grad/batch_norm": record.batch_gradient_norm,
        "grad/perturbed_norm": record.perturbed_gradient_norm,
        "grad/update_norm": record.final_gradient_norm,
        "grad/sam_perturb_norm": record.sam_perturbation_norm,
        "timing/train_step_s": elapsed,
    }
    if isinstance(record, SAMPLeStepRecord):
        metric.update(
            {
                "grad/global_estimate_norm": record.global_direction_norm,
                "grad/batch_component_norm": record.batch_component_norm,
                "grad/batch_correction_norm": record.batch_correction_norm,
                "grad/total_displacement_norm": record.total_displacement_norm,
                "grad/xi": record.projection.xi,
                "grad/sigma": record.projection.sigma,
                "grad/projection_coefficient": record.projection.projection_coefficient,
                "estimator/refreshed": record.estimator_result.refreshed,
                "estimator/age_steps": record.estimator_result.age_steps,
                "estimator/exact_query_count": record.estimator_result.exact_query_count,
            }
        )
    artifacts.append_metric(metric)


def _save_upstream_checkpoint(trainer, run_dir: Path) -> tuple[Path, str]:
    checkpoint_root = run_dir / "checkpoints" / "coop"
    trainer.save_model(epoch=0, directory=str(checkpoint_root))
    upstream = checkpoint_root / "prompt_learner" / "model.pth.tar-1"
    payload = torch.load(upstream, map_location="cpu", weights_only=False)
    learned = unwrap_model(trainer.model).prompt_learner.ctx.detach().cpu()
    if not torch.equal(payload["state_dict"]["ctx"].detach().cpu(), learned):
        raise AssertionError("Saved upstream checkpoint does not equal learned context")
    final = run_dir / "checkpoints" / "final.pt"
    copy_artifact(upstream, final)
    return checkpoint_root, sha256_file(final)


def _evaluate(
    *, trainer, learned_context: torch.Tensor, checkpoint_root: Path,
    data_root: Path, clip_cache: Path, runtime_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], float, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    base = evaluate_bounded(trainer, 8, 24)
    torch.cuda.synchronize()
    base_s = time.perf_counter() - started
    if not torch.equal(
        unwrap_model(trainer.model).prompt_learner.ctx.detach().cpu().float(),
        learned_context,
    ):
        raise AssertionError("Base evaluation changed learned context")

    del trainer
    _release_cuda()
    new_cfg = _cfg(data_root, runtime_root / "new", "fp32", "new")
    set_random_seed(1)
    new_trainer = build_coop_trainer(new_cfg, clip_cache)
    if len(new_trainer.dm.dataset.classnames) != 23:
        raise AssertionError("DTD new-class dimension differs from 23")
    new_trainer.load_model(str(checkpoint_root), epoch=1)
    loaded = unwrap_model(new_trainer.model).prompt_learner.ctx.detach().cpu().float()
    if not torch.equal(loaded, learned_context):
        raise AssertionError("New-class model did not reuse the saved base context")
    torch.cuda.synchronize()
    started = time.perf_counter()
    new = evaluate_bounded(new_trainer, 8, 23)
    torch.cuda.synchronize()
    new_s = time.perf_counter() - started
    if not torch.equal(
        unwrap_model(new_trainer.model).prompt_learner.ctx.detach().cpu().float(),
        learned_context,
    ):
        raise AssertionError("New-class evaluation changed learned context")
    del new_trainer
    _release_cuda()
    return base, new, base_s, new_s


def _diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_exact_reference_points": 0,
            "global_estimate_exact_cosine_mean": None,
            "global_estimate_exact_cosine_median": None,
            "global_estimate_exact_relative_l2_mean": None,
            "global_estimate_exact_log_norm_ratio_mean": None,
            "batch_component_exact_abs_cosine_mean": None,
            "perturbed_gradient_exact_abs_cosine_mean": None,
        }
    keys = (
        "grad/global_estimate_exact_cosine",
        "grad/global_estimate_exact_relative_l2",
        "grad/global_estimate_exact_log_norm_ratio",
        "grad/batch_component_exact_cosine",
        "grad/perturbed_gradient_exact_cosine",
    )
    values = {key: [item["metrics"][key] for item in records if item["metrics"][key] is not None] for key in keys}
    cosine = sorted(values[keys[0]])
    median = cosine[len(cosine) // 2] if len(cosine) % 2 else (cosine[len(cosine)//2-1] + cosine[len(cosine)//2]) / 2
    mean = lambda sequence: sum(sequence) / len(sequence) if sequence else None
    return {
        "num_exact_reference_points": len(records),
        "global_estimate_exact_cosine_mean": mean(values[keys[0]]),
        "global_estimate_exact_cosine_median": median,
        "global_estimate_exact_relative_l2_mean": mean(values[keys[1]]),
        "global_estimate_exact_log_norm_ratio_mean": mean(values[keys[2]]),
        "batch_component_exact_abs_cosine_mean": mean([abs(v) for v in values[keys[3]]]),
        "perturbed_gradient_exact_abs_cosine_mean": mean([abs(v) for v in values[keys[4]]]),
    }


def audit_run_directory(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).is_file()]
    if missing:
        raise RunArtifactError(f"Missing Stage-0 artifacts: {missing}")
    import yaml
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = load_jsonl(run_dir / "metrics.jsonl")
    diagnostics = load_jsonl(run_dir / "gradient_diagnostics.jsonl")
    if not config["run"]["smoke"] or config["smoke"]["allow_scientific_summary"]:
        raise RunArtifactError("Stage-0 config scientific gate is invalid")
    if not summary["smoke"] or summary["allow_scientific_summary"]:
        raise RunArtifactError("Stage-0 summary scientific gate is invalid")
    if summary["status"] != "completed" or not metrics:
        raise RunArtifactError("Stage-0 run is incomplete")
    if environment.get("gpu", {}).get("name") is None:
        raise RunArtifactError("Stage-0 environment lacks detected GPU")
    selected_samples = manifest.get("selected_samples") or manifest.get("few_shot", {}).get("selected_sample_ids")
    if not selected_samples:
        raise RunArtifactError("Stage-0 manifest lacks selected samples")
    suspect: list[str] = []
    for relative in ("logs/run.log", "metrics.jsonl", "gradient_diagnostics.jsonl"):
        for number, line in enumerate((run_dir / relative).read_text(encoding="utf-8").splitlines(), 1):
            if FAILURE_PATTERN.search(line):
                suspect.append(f"{relative}:{number}")
    if suspect:
        raise RunArtifactError(f"Failure token found in artifacts: {suspect}")
    return {
        "status": "PASS", "run_id": summary["run_identity"]["run_id"],
        "metric_records": len(metrics), "diagnostic_records": len(diagnostics),
        "required_artifacts": len(REQUIRED_ARTIFACTS), "failure_scan": "PASS",
    }


def _run_cell(
    *, method: str, estimator_mode: str | None, precision: str, steps: int,
    data_root: Path, manifest: Path, clip_cache: Path, output_root: Path,
    source,
) -> dict[str, Any]:
    method_tag, estimator_tag = _method_tag(method, estimator_mode)
    cfg_precision = "fp16" if precision == "coop_fp16" else precision
    cfg = _cfg(data_root, output_root / "runtime" / f"{method_tag}_{estimator_tag}_{precision}", cfg_precision)
    config = _resolved_config(
        cfg=cfg, data_root=data_root, output_root=output_root,
        clip_checkpoint=clip_cache / "ViT-B-16.pt", method=method,
        estimator=estimator_mode, precision=precision, steps=steps,
        source_fingerprint=source.fingerprint,
    )
    identity = RunIdentity.now(
        dataset="dtd", shots=4, method_tag=method_tag,
        estimator_tag=f"{estimator_tag}_{precision}", seed=1,
        config_sha256=config["config_sha256"], experiment_id="S1",
        smoke=True, allow_scientific_summary=False,
    )
    config = bind_run_identity(config, identity)
    environment = capture_environment(
        project_repo=REPO_ROOT, coop_upstream_commit=COOP_UPSTREAM_SHA,
        dassl_commit=DASSL_SHA, precision_mode=precision,
        clip_backbone="ViT-B/16",
        clip_checkpoint_identifier=str(clip_cache / "ViT-B-16.pt"),
        clip_checkpoint_sha256=EXPECTED_CLIP_SHA256,
        capture_package_freeze=True,
    )
    artifacts = RunArtifacts(output_root / "runs", identity)
    run_dir = artifacts.create(
        resolved_config=config, environment=environment,
        data_manifest_source=manifest,
    )
    artifacts.append_log(f"Stage-0 {method_tag}/{estimator_tag}/{precision} started")

    set_random_seed(1)
    trainer = build_coop_trainer(cfg, clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    audit = audit_prompt_only_training(model, trainer.optim)
    index = ParamIndex.from_model(model)
    if index.names != ("prompt_learner.ctx",) or index[0].shape != (4, 512):
        raise AssertionError("Stage-0 CoOp ParamIndex differs")
    if len(trainer.dm.dataset.classnames) != 24 or len(trainer.dm.dataset.train_x) != 96:
        raise AssertionError("Stage-0 DTD base protocol differs")
    prompt_before = index[0].parameter.detach().clone()
    frozen_before = hash_frozen_parameters(model)
    scheduler_before = copy.deepcopy(trainer.sched.state_dict())
    precision_controller = PrecisionController(cfg_precision)
    service = None
    estimator = None
    coordinator = None
    if method == "sample":
        full_loader = build_full_gradient_loader(cfg, source, micro_batch_size=2)
        service = FullGradientService(
            model=model, param_index=index, loader=full_loader,
            precision_controller=PrecisionController(cfg_precision),
            protocol_seed=1, dataset="dtd", shots=4,
            config_hash=config["config_sha256"],
        )
        if estimator_mode == "ema":
            estimator = EMAEstimator(index, ema_lambda=0.15)
        elif estimator_mode == "exact":
            estimator = ExactEstimator(index, full_gradient_service=service)
        elif estimator_mode == "periodic":
            estimator = PeriodicEstimator(index, ema_lambda=0.15, refresh_k_steps=2, full_gradient_service=service)
        else:
            raise AssertionError("Unsupported Stage-0 estimator")
        coordinator = DiagnosticCoordinator(
            schedule=DiagnosticSchedule(1), full_gradient_service=service
        )
    engine = None if method == "coop" else StepEngine(
        param_index=index, optimizer=trainer.optim,
        precision_controller=precision_controller, rho=0.05,
        alpha=0.0015 if method == "sample" else None,
        diagnostic_coordinator=coordinator,
    )

    accounting = RunAccounting()
    diagnostic_records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    with count_optimizer_steps(trainer.optim) as optimizer_counter:
        for batch_index, raw_batch in enumerate(trainer.train_loader_x):
            if optimizer_counter["count"] >= steps:
                break
            lr = float(trainer.get_current_lr())
            if method == "coop":
                trainer.batch_idx = batch_index
                trainer.num_batches = len(trainer.train_loader_x)
                step_started = time.perf_counter()
                values = trainer.forward_backward(raw_batch)
                elapsed = time.perf_counter() - step_started
                metric = {
                    "schema_version": METRICS_SCHEMA_VERSION, "event_type": "train_step",
                    **_common(identity, method="coop", estimator=None,
                              step=optimizer_counter["count"] - 1, batch_index=batch_index),
                    "loss/current": float(values["loss"]), "loss/displaced": None,
                    "loss/sample_objective": None, "optim/learning_rate": lr,
                    "optim/nonfinite_event": False, "grad/batch_norm": None,
                    "grad/perturbed_norm": None, "grad/update_norm": None,
                    "timing/train_step_s": elapsed,
                }
                artifacts.append_metric(metric)
            else:
                image, label = trainer.parse_batch_train(raw_batch)
                batch = (image, label)
                torch.cuda.synchronize()
                step_started = time.perf_counter()
                if method == "sam":
                    record = engine.step_sam(batch, lambda item: F.cross_entropy(model(item[0]), item[1]))
                else:
                    record = engine.step_sample(
                        batch, lambda item: F.cross_entropy(model(item[0]), item[1]),
                        estimator, epoch=0, batch_index=batch_index,
                    )
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - step_started
                _append_sharp_metric(
                    artifacts, identity, record, estimator=estimator_mode,
                    batch_index=batch_index, elapsed=elapsed, lr=lr,
                )
                if isinstance(record, SAMPLeStepRecord) and record.diagnostic_event is not None:
                    diagnostic = {
                        **record.diagnostic_event.as_dict(),
                        **_common(identity, method="sample", estimator=estimator_mode,
                                  step=record.optimizer_step, batch_index=batch_index),
                    }
                    artifacts.append_diagnostic(diagnostic)
                    diagnostic_records.append(diagnostic)
                    diag_values = diagnostic["metrics"]
                    residual_norm = diag_values["grad/batch_component_norm"]
                    batch_norm = diag_values["grad/batch_norm"]
                    direction_norm = diag_values["grad/global_estimate_norm"]
                    if residual_norm <= 1e-6 * batch_norm:
                        scale = batch_norm * direction_norm
                        active_error = 0.0 if scale == 0 else abs(diag_values["raw/dot_batch_component_global"]) / scale
                    else:
                        active_error = abs(diag_values["grad/batch_component_estimator_cosine"] or 0.0)
                    if active_error > 2e-5:
                        raise AssertionError("Active projection orthogonality differs")
                    if abs(diagnostic["metrics"]["grad/reference_batch_component_exact_cosine"] or 0.0) > 2e-5:
                        raise AssertionError("Exact-reference projection orthogonality differs")
            accounting.increment("optimizer_steps")
            accounting.increment("current_forward_batches")
            accounting.increment("current_backward_batches")
            accounting.increment("current_samples", 2)
            if method in {"sam", "sample"}:
                accounting.increment("displaced_forward_batches")
                accounting.increment("displaced_backward_batches")
                accounting.increment("displaced_samples", 2)
            if method == "sample":
                metadata = record.estimator_result.full_gradient_metadata
                if metadata is not None:
                    accounting.increment("full_gradient_sweeps")
                    accounting.increment("full_gradient_forward_microbatches", metadata.forward_calls)
                    accounting.increment("full_gradient_backward_microbatches", metadata.autograd_grad_calls)
                    accounting.increment("full_gradient_samples", metadata.sample_count)
                    accounting.full_gradient_total_s += metadata.elapsed_s
                event = record.diagnostic_event
                if event is not None and event.reference.exact_service_query_issued:
                    metadata = event.reference.full_gradient_metadata
                    accounting.increment("full_gradient_sweeps")
                    accounting.increment("full_gradient_forward_microbatches", metadata.forward_calls)
                    accounting.increment("full_gradient_backward_microbatches", metadata.autograd_grad_calls)
                    accounting.increment("full_gradient_samples", metadata.sample_count)
                    accounting.full_gradient_total_s += metadata.elapsed_s
                if event is not None:
                    accounting.increment("exact_reference_points")
    torch.cuda.synchronize()
    accounting.train_total_s = time.perf_counter() - training_started
    accounting.peak_cuda_allocated_bytes = int(torch.cuda.max_memory_allocated())
    accounting.peak_cuda_reserved_bytes = int(torch.cuda.max_memory_reserved())
    if optimizer_counter["count"] != steps:
        raise AssertionError("Stage-0 optimizer-step count differs")
    if torch.equal(index[0].parameter, prompt_before):
        raise AssertionError("Stage-0 prompt did not change")
    if hash_frozen_parameters(model) != frozen_before:
        raise AssertionError("Stage-0 frozen CLIP changed")
    if trainer.sched.state_dict() != scheduler_before:
        raise AssertionError("Stage-0 advanced epoch scheduler before boundary")

    learned = index[0].parameter.detach().cpu().float().clone()
    checkpoint_root, checkpoint_hash = _save_upstream_checkpoint(trainer, run_dir)
    base, new, base_s, new_s = _evaluate(
        trainer=trainer, learned_context=learned, checkpoint_root=checkpoint_root,
        data_root=data_root, clip_cache=clip_cache,
        runtime_root=output_root / "runtime" / identity.run_id,
    )
    denominator = base["accuracy_pct"] + new["accuracy_pct"]
    hm = 0.0 if denominator == 0 else 2 * base["accuracy_pct"] * new["accuracy_pct"] / denominator
    for event_type, result, seconds in (("eval_base", base, base_s), ("eval_new", new, new_s)):
        artifacts.append_metric(
            {
                "schema_version": METRICS_SCHEMA_VERSION, "event_type": event_type,
                **_common(identity, method=method, estimator=estimator_mode,
                          step=steps, batch_index=-1),
                "eval/accuracy_pct": result["accuracy_pct"],
                "eval/num_samples": result["sample_count"],
                "eval/class_count": result["class_count"],
                "timing/eval_s": seconds,
            }
        )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_identity": identity.as_dict(), "status": "completed",
        "smoke": True, "allow_scientific_summary": False,
        "evaluation": {
            "base_accuracy_pct": base["accuracy_pct"],
            "new_accuracy_pct": new["accuracy_pct"], "hm_pct": hm,
            "base_num_samples": base["sample_count"], "new_num_samples": new["sample_count"],
            "base_class_count": 24, "new_class_count": 23,
            "same_unified_context": True, "new_retrained": False,
            "interpretation": "non-scientific execution check only",
        },
        "efficiency": {
            **accounting.as_dict(), "eval_base_s": base_s, "eval_new_s": new_s,
            "evaluation_total_s": base_s + new_s,
            "diagnostic_overhead_s": sum(
                item["full_gradient"]["elapsed_s"]
                for item in diagnostic_records
                if item["exact_service_query_issued"]
            ),
        },
        "estimator_diagnostics": _diagnostic_summary(diagnostic_records),
        "invariants": {
            "prompt_changed": True, "frozen_clip_unchanged": True,
            "scheduler_steps": 0, "optimizer_steps": steps,
            "trainable_names": [item["name"] for item in audit["trainable_parameters"]],
            "checkpoint_sha256": checkpoint_hash,
        },
        "artifacts": {
            "config": "config.yaml", "environment": "environment.json",
            "data_manifest": "data_manifest.json", "metrics": "metrics.jsonl",
            "gradient_diagnostics": "gradient_diagnostics.jsonl",
            "checkpoint": "checkpoints/final.pt", "log": "logs/run.log",
        },
    }
    artifacts.write_summary(summary)
    artifacts.append_log(f"Stage-0 {method_tag}/{estimator_tag}/{precision} completed PASS")
    audit_result = audit_run_directory(run_dir)
    return {
        "status": "PASS", "run_dir": str(run_dir), "run_id": identity.run_id,
        "config_sha256": identity.config_sha256, "method": method,
        "estimator": estimator_mode, "precision": precision, "steps": steps,
        "full_gradient_sweeps": accounting.compute_counts.get("full_gradient_sweeps", 0),
        "full_gradient_samples": accounting.compute_counts.get("full_gradient_samples", 0),
        "diagnostics": len(diagnostic_records),
        "train_total_s": accounting.train_total_s, "eval_total_s": base_s + new_s,
        "peak_cuda_allocated_bytes": accounting.peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": accounting.peak_cuda_reserved_bytes,
        "evaluation": summary["evaluation"], "artifact_audit": audit_result,
        "deterministic_metrics": [
            {key: value for key, value in metric.items() if key in {
                "optimizer_step", "loss/current", "loss/displaced",
                "loss/sample_objective", "grad/batch_norm",
                "grad/global_estimate_norm", "grad/perturbed_norm", "grad/update_norm",
            }}
            for metric in load_jsonl(run_dir / "metrics.jsonl")
            if metric["event_type"] == "train_step"
        ],
    }


def _memory_leak_check(*, data_root: Path, manifest: Path, clip_cache: Path,
                       output_root: Path, source, sweeps: int) -> dict[str, Any]:
    cfg = _cfg(data_root, output_root / "memory_runtime", "fp32")
    set_random_seed(1)
    trainer = build_coop_trainer(cfg, clip_cache)
    trainer.set_model_mode("train")
    model = unwrap_model(trainer.model)
    index = ParamIndex.from_model(model)
    prompt = index[0].parameter.detach().clone()
    frozen = hash_frozen_parameters(model)
    loader = build_full_gradient_loader(cfg, source, micro_batch_size=32)
    service = FullGradientService(
        model=model, param_index=index, loader=loader,
        precision_controller=PrecisionController("fp32"), protocol_seed=1,
        dataset="dtd", shots=4, config_hash="stage0-memory-leak-v1",
    )
    allocated: list[int] = []
    losses: list[float] = []
    for step in range(sweeps):
        result = service.compute(optimizer_step=step, purpose="stage0_memory_check")
        losses.append(result.mean_loss)
        del result
        gc.collect()
        torch.cuda.synchronize()
        allocated.append(int(torch.cuda.memory_allocated()))
    if any(not math.isfinite(value) for value in losses):
        raise FloatingPointError("Repeated exact sweep produced nonfinite loss")
    if allocated[-1] > allocated[0] + 8 * 1024 * 1024:
        raise AssertionError(f"Repeated exact sweeps retained CUDA memory: {allocated}")
    if not torch.equal(index[0].parameter, prompt) or hash_frozen_parameters(model) != frozen:
        raise AssertionError("Repeated exact sweeps mutated model parameters")
    return {
        "status": "PASS", "sweeps": sweeps, "micro_batch_size": 32,
        "sample_count_per_sweep": len(source), "allocated_bytes": allocated,
        "allocated_growth_bytes": allocated[-1] - allocated[0],
        "reserved_memory_is_not_used_as_leak_evidence": True,
        "optimizer_steps": 0,
    }


def run(args: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-0 requires CUDA")
    if _git(REPO_ROOT, "rev-parse", "HEAD") != STARTING_SHA:
        raise AssertionError("Stage-0 starting CoOp SHA differs")
    project_root = REPO_ROOT.parents[1]
    dassl_root = project_root / "implementation" / "Dassl.pytorch"
    if _git(dassl_root, "rev-parse", "HEAD") != DASSL_SHA or _git(dassl_root, "status", "--short"):
        raise AssertionError("Pinned Dassl checkout changed")
    data_root = Path(args.root).resolve(strict=True)
    manifest_root = Path(args.manifest_root).resolve(strict=True)
    clip_cache = Path(args.clip_cache).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    report_path = Path(args.report).resolve()
    checkpoint = clip_cache / "ViT-B-16.pt"
    if sha256_file(checkpoint).lower() != EXPECTED_CLIP_SHA256:
        raise AssertionError("Pinned CLIP hash differs")
    if sha256_file(data_root / "dtd" / "split_zhou_DescribableTextures.json") != DTD_SPLIT_SHA:
        raise AssertionError("DTD split hash differs")
    if sha256_file(data_root / "dtd" / "split_fewshot" / "shot_4-seed_1.pkl") != DTD_CACHE_SHA:
        raise AssertionError("DTD cache hash differs")

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    loaded_dtd = load_dataset(data_root, DATASET_SPECS["dtd"])
    manifest = manifest_root / "dtd" / "shots_4" / "seed_1" / "data_manifest.json"
    source = load_full_gradient_source(loaded_dtd, manifest)
    euro_manifest = json.loads(
        (manifest_root / "eurosat" / "shots_4" / "seed_1" / "data_manifest.json").read_text(encoding="utf-8")
    )
    euro_selected_count = euro_manifest["few_shot"]["total_selected_count"]
    if len(source) != 96 or euro_selected_count != 20:
        raise AssertionError("Stage-0 DTD/EuroSAT manifest counts differ")

    matrix = [
        ("coop", None, "fp32", 3),
        ("sam", None, "fp32", 3),
        ("sample", "ema", "fp32", 3),
        ("sample", "exact", "fp32", 3),
        ("sample", "periodic", "fp32", 5),
        ("sample", "ema", "coop_fp16", 3),
    ]
    runs: list[dict[str, Any]] = []
    for method, estimator, precision, steps in matrix:
        print(f"[stage0] {method}/{estimator or 'none'}/{precision}: {steps} steps", flush=True)
        runs.append(
            _run_cell(
                method=method, estimator_mode=estimator, precision=precision,
                steps=steps, data_root=data_root, manifest=manifest,
                clip_cache=clip_cache, output_root=output_root, source=source,
            )
        )
        _release_cuda()

    print("[stage0] repeatability: sample/ema/fp32", flush=True)
    repeated = _run_cell(
        method="sample", estimator_mode="ema", precision="fp32", steps=3,
        data_root=data_root, manifest=manifest, clip_cache=clip_cache,
        output_root=output_root, source=source,
    )
    reference = next(item for item in runs if item["method"] == "sample" and item["estimator"] == "ema" and item["precision"] == "fp32")
    if reference["config_sha256"] != repeated["config_sha256"]:
        raise AssertionError("Repeatability run config hash differs")
    if reference["deterministic_metrics"] != repeated["deterministic_metrics"]:
        raise AssertionError("Repeatability training metrics differ")
    if reference["evaluation"] != repeated["evaluation"]:
        raise AssertionError("Repeatability evaluation differs")
    repeated["repeat_of_run_id"] = reference["run_id"]
    repeated["deterministic_match"] = True
    runs.append(repeated)
    _release_cuda()

    print(f"[stage0] exact-query memory check: {args.memory_sweeps} sweeps", flush=True)
    memory = _memory_leak_check(
        data_root=data_root, manifest=manifest, clip_cache=clip_cache,
        output_root=output_root, source=source, sweeps=args.memory_sweeps,
    )
    _release_cuda()

    resume_report_path = output_root / "checkpoint_resume.json"
    subprocess.check_call(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "validate_checkpoint_resume.py"),
            "--root", str(data_root), "--manifest-root", str(manifest_root),
            "--clip-cache", str(clip_cache),
            "--output-root", str(output_root / "checkpoint_resume_runs"),
            "--report", str(resume_report_path),
        ], cwd=REPO_ROOT,
    )
    resume = json.loads(resume_report_path.read_text(encoding="utf-8"))
    if resume.get("status") != "PASS":
        raise AssertionError("Integrated checkpoint/resume validation failed")

    report = {
        "schema_version": STAGE0_SCHEMA_VERSION, "status": "PASS",
        "smoke": True, "allow_scientific_summary": False,
        "completed_at_utc": _utc_now(),
        "source": {
            "coop_upstream_sha": COOP_UPSTREAM_SHA,
            "coop_stage0_start_sha": STARTING_SHA, "dassl_sha": DASSL_SHA,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        },
        "data_validation": {
            "dtd_selected_4shot": len(source),
            "eurosat_selected_4shot": euro_selected_count,
            "official_fixed_splits": True,
        },
        "matrix": runs, "repeatability": {
            "reference_run_id": reference["run_id"],
            "repeat_run_id": repeated["run_id"],
            "deterministic_fields_match": True,
            "legitimately_variable_fields": ["run_id", "timestamps", "timing", "peak allocator statistics"],
        },
        "memory_leak_check": memory,
        "checkpoint_resume": {
            "status": resume["status"], "report": str(resume_report_path),
            "uninterrupted_vs_resumed": resume.get("uninterrupted_vs_resumed"),
        },
        "artifact_audit": {
            "runs_audited": len(runs), "all_required_files": True,
            "failure_token_scan": "PASS", "method_ranking_performed": False,
        },
        "warnings": {
            "benign": ["torch.cuda.amp API deprecation warnings in retained tests"],
            "scientific_numerical": [],
        },
        "interpretation": "correctness/operational viability only; no scientific result",
    }
    atomic_write_json(report_path, report)
    print(json.dumps({"status": "PASS", "report": str(report_path), "runs": len(runs)}, indent=2))
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--memory-sweeps", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
