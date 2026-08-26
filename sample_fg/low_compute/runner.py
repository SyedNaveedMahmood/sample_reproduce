"""Integrated, frozen-state LC01+LC04 execution lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml
from torch.nn import functional as F

from sample_fg.coop_anchor import build_coop_trainer, hash_frozen_parameters, unwrap_model
from sample_fg.environment import capture_environment
from sample_fg.gradient_state import GradientState
from sample_fg.param_index import ParamIndex
from sample_fg.paper_runner import (
    DEFAULT_CONFIG as DEFAULT_PAPER_CONFIG,
    _build_runtime,
    build_scientific_cfg,
    build_scientific_plan,
    resolve_method,
)
from sample_fg.rng import isolated_rng
from sample_fg.results import atomic_write_json
from dassl.utils import set_random_seed

from .artifacts import (
    CONFIG_SCHEMA_VERSION,
    METRICS_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    LowComputeArtifacts,
)
from .budget import TransitionGuard
from .checkpoint_probe import load_probe_checkpoint, verify_source_immutable
from .functional_probe import compare_functional_directions
from .feature_cache import FeatureCacheKey, materialization_seed_clock, save_feature_cache
from .gradient_bank import GradientBank, build_gradient_bank, gradient_sha256
from .math import gradient_metrics
from .planner import IntegratedProbePlan
from .replay import (
    effective_sample_size,
    ema_replay,
    history_length,
    permutation_trials,
    projection_displacement_metrics,
    stationary_ema_replay,
    evaluate_lc02_gate,
)


class IntegratedProbeError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntegratedProbeError(f"YAML root is not a mapping: {path}")
    return value


def _portable_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "data": (
            "dataset", "shots", "seed", "split_policy", "train_batch_size",
            "test_batch_size", "num_workers", "preserve_upstream_drop_last",
            "augmentation_policy", "selected_source_fingerprint", "selected_count",
        ),
        "model": (
            "backbone", "prompt_learner", "effective_n_ctx", "ctx_init",
            "class_specific_context", "class_token_position", "freeze_clip",
            "checkpoint_sha256",
        ),
        "method": (
            "name", "rho", "alpha", "ema_lambda", "norm_eps",
            "first_order_stop_gradient",
        ),
        "estimator": ("mode", "full_gradient_micro_batch_size"),
        "optim": (
            "name", "lr", "weight_decay", "momentum", "nesterov", "max_epoch",
            "scheduler", "warmup_epoch", "warmup_type", "warmup_cons_lr",
            "scheduler_step_unit",
        ),
        "runtime": ("precision", "gradient_state_dtype"),
    }
    return {
        section: {
            key: config.get(section, {}).get(key)
            for key in keys
        }
        for section, keys in fields.items()
    }


def build_source_scientific_plan(
    *,
    source_run: Path,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    paper_config: Path = DEFAULT_PAPER_CONFIG,
    runtime_output_root: Path,
):
    """Rebuild and hash-check the canonical source protocol without training."""

    config = _read_yaml(Path(source_run) / "config.yaml")
    run = config.get("run", {})
    checkpoint = config.get("checkpoint", {})
    diagnostics = config.get("diagnostics", {})
    estimator = config.get("estimator", {})
    selection = resolve_method("sample", "ema")
    plan = build_scientific_plan(
        dataset="dtd",
        shots=16,
        seed=1,
        experiment_id=str(run.get("experiment_id", "R2")),
        selection=selection,
        data_root=Path(data_root),
        manifest_root=Path(manifest_root),
        clip_cache=Path(clip_cache),
        output_root=Path(run.get("output_root", runtime_output_root)),
        config_path=Path(paper_config),
        recovery_interval_epochs=int(checkpoint.get("recovery_interval_epochs", 10)),
        epochs=int(config.get("optim", {}).get("max_epoch", 200)),
        diagnostic_interval_steps=diagnostics.get("full_gradient_interval_steps"),
        full_gradient_micro_batch_size=int(estimator.get("full_gradient_micro_batch_size", 32)),
        notes=str(run.get("notes", "Primary DTD/EuroSAT 16-shot CoOp paper reproduction")),
    )
    if _portable_protocol(plan.resolved_config) != _portable_protocol(config):
        raise IntegratedProbeError(
            "Reconstructed source scientific protocol differs; refuse checkpoint analysis"
        )
    return replace(plan, output_root=Path(runtime_output_root).resolve())


def _cpu_state(state: GradientState) -> GradientState:
    return GradientState.from_tensors(
        state.param_index, (component.detach().cpu() for component in state)
    )


def _device_state(state: GradientState, param_index: ParamIndex) -> GradientState:
    return GradientState.from_tensors(
        param_index,
        (
            component.detach().to(device=entry.parameter.device)
            for entry, component in zip(param_index, state)
        ),
    )


def _text_features(model) -> torch.Tensor:
    prompts = model.prompt_learner()
    text = model.text_encoder(prompts, model.tokenized_prompts)
    return text / text.norm(dim=-1, keepdim=True)


def _materialize_features(runtime, checkpoint_sha: str, replicate: int) -> tuple[dict[str, Any], ...]:
    dataset = runtime.full_gradient_loader.dataset
    model = runtime.model
    device = runtime.param_index[0].parameter.device
    rows = []
    pending = []
    model.train()
    for index, record in enumerate(dataset.source):
        seed_clock = materialization_seed_clock(
            checkpoint_sha, replicate, record.sample_id
        )
        with isolated_rng(
            protocol_seed=1,
            dataset="dtd",
            shots=16,
            config_hash=checkpoint_sha,
            optimizer_step=seed_clock,
            purpose=f"lc01_materialize_r{replicate}_{record.sample_id}",
        ) as derived:
            item = dataset[index]
        pending.append((record, item, derived.as_dict()))
        if len(pending) == 32 or index + 1 == len(dataset.source):
            images = torch.stack([entry[1]["img"] for entry in pending]).to(device)
            with torch.no_grad():
                features = model.image_encoder(images.type(model.dtype))
                features = features / features.norm(dim=-1, keepdim=True)
            for (pending_record, pending_item, seed), feature in zip(pending, features):
                rows.append(
                    {
                        "sample_id": pending_record.sample_id,
                        "label": int(pending_item["label"]),
                        "feature": feature.detach().to(dtype=torch.float32).cpu(),
                        "derived_seed": seed,
                    }
                )
            pending.clear()
    return tuple(rows)


def _gradient_bank(runtime, materialized: Sequence[Mapping[str, Any]], replicate: int) -> GradientBank:
    device = runtime.param_index[0].parameter.device
    model = runtime.model
    model.train()
    batches = []
    for start in range(0, len(materialized), 32):
        records = materialized[start : start + 32]
        sample_ids = tuple(str(row["sample_id"]) for row in records)
        features = torch.stack([row["feature"] for row in records]).to(device)
        labels = torch.tensor([row["label"] for row in records], device=device, dtype=torch.long)

        def loss_closure(features=features, labels=labels):
            text = _text_features(model)
            logits = model.logit_scale.exp() * features.to(text) @ text.t()
            return F.cross_entropy(logits, labels)

        batches.append((sample_ids, len(records), loss_closure))
    bank = build_gradient_bank(
        param_index=runtime.param_index,
        materialized_batches=batches,
        materialization_replicate=replicate,
    )
    cpu_batches = tuple(
        replace(item, gradient=_cpu_state(item.gradient)) for item in bank.batches
    )
    return GradientBank(
        batches=cpu_batches,
        exact=_cpu_state(bank.exact),
        total_samples=bank.total_samples,
        materialization_replicate=replicate,
    )


def _eval_feature_cache(
    trainer, model, *, label_offset: int
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    model.eval()
    features = []
    labels = []
    sample_ids = []
    with torch.no_grad():
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_test(batch)
            value = model.image_encoder(image.type(model.dtype))
            value = value / value.norm(dim=-1, keepdim=True)
            features.append(value.detach().to(dtype=torch.float32).cpu())
            labels.append(label.detach().cpu().to(dtype=torch.long) + label_offset)
            raw_ids = batch.get("impath") if isinstance(batch, Mapping) else None
            if not isinstance(raw_ids, (tuple, list)) or len(raw_ids) != len(label):
                raw_ids = tuple(
                    f"{label_offset}:{len(sample_ids) + index}"
                    for index in range(len(label))
                )
            root = Path(trainer.cfg.DATASET.ROOT).resolve()
            for value in raw_ids:
                try:
                    sample_ids.append(Path(value).resolve().relative_to(root).as_posix())
                except (OSError, ValueError, TypeError):
                    sample_ids.append(str(value))
    return torch.cat(features), torch.cat(labels), tuple(sample_ids)


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    data = sorted(float(value) for value in values if value is not None)
    if not data:
        return {
            "defined_count": 0, "mean": None, "sd": None, "min": None,
            "q05": None, "q25": None, "median": None, "q75": None,
            "q95": None, "max": None,
        }
    tensor = torch.tensor(data, dtype=torch.float64)
    return {
        "defined_count": len(data),
        "mean": statistics.fmean(data),
        "sd": statistics.stdev(data) if len(data) > 1 else 0.0,
        "min": data[0],
        "q05": float(torch.quantile(tensor, 0.05).item()),
        "q25": float(torch.quantile(tensor, 0.25).item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "q75": float(torch.quantile(tensor, 0.75).item()),
        "q95": float(torch.quantile(tensor, 0.95).item()),
        "max": data[-1],
    }


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")


def run_integrated_probe(
    plan: IntegratedProbePlan,
    *,
    scientific_plan,
    output_root: Path,
    config_path: Path,
) -> Path:
    """Execute the explicitly authorized frozen audit; never take a step."""

    plan.budget.require_read_only()
    runtime_root = Path(output_root).resolve() / "_runtime"
    runtime = _build_runtime(scientific_plan, runtime_root)
    source_config = _read_yaml(Path(plan.source_run) / "config.yaml")
    probe_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (
        Path(output_root).resolve() / "lc01_lc04" / "dtd" / "shots_16"
        / "sample_ema" / "seed_1" / probe_id
    )
    artifacts = LowComputeArtifacts(run_dir)
    config = _read_yaml(config_path)
    environment = capture_environment(
        project_repo=Path(__file__).resolve().parents[2],
        coop_upstream_commit=source_config["provenance"]["coop_upstream_commit"],
        dassl_commit=source_config["provenance"]["dassl_commit"],
        precision_mode="coop_fp16",
        clip_backbone="ViT-B/16",
        clip_checkpoint_identifier=str(scientific_plan.clip_checkpoint),
        clip_checkpoint_sha256=source_config["model"]["checkpoint_sha256"],
        capture_package_freeze=True,
    )
    source = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "source_run": str(plan.source_run),
        "source_config_sha256": source_config["config_sha256"],
        "selected_source_fingerprint": scientific_plan.source.fingerprint,
        "checkpoints": [
            {"path": str(item.path), "sha256": item.sha256, "epoch": item.epoch}
            for item in plan.checkpoints
        ],
    }
    artifacts.create(config=config, environment=environment, source=source, budget=plan.budget.as_dict())

    # Build the shared Base+New evaluation image cache once.
    eval_cache_started = time.perf_counter()
    base_features, base_labels, base_sample_ids = _eval_feature_cache(
        runtime.trainer, runtime.model, label_offset=0
    )
    base_class_count = len(runtime.trainer.dm.dataset.classnames)
    new_cfg = build_scientific_cfg(
        dataset="dtd", seed=1, data_root=scientific_plan.data_root,
        output_dir=runtime_root / "new", config_path=scientific_plan.config_path,
        class_subsample="new", shots=16, epochs=200,
    )
    set_random_seed(1)
    new_trainer = build_coop_trainer(new_cfg, scientific_plan.clip_cache)
    new_model = unwrap_model(new_trainer.model)
    new_index = ParamIndex.from_model(new_model)
    new_features, new_labels, new_sample_ids = _eval_feature_cache(
        new_trainer, new_model, label_offset=base_class_count
    )
    eval_features = torch.cat((base_features, new_features))
    eval_labels = torch.cat((base_labels, new_labels))
    eval_key = FeatureCacheKey(
        dataset="dtd", split="base_new_test",
        clip_sha256=source_config["model"]["checkpoint_sha256"],
        transform_signature="canonical_evaluation_transform",
    )
    cache_digest = save_feature_cache(
        run_dir / "cache" / f"eval_{eval_key.digest}.pt",
        key=eval_key, features=eval_features, labels=eval_labels,
        sample_ids=base_sample_ids + new_sample_ids,
    )
    atomic_write_json(
        run_dir / "cache" / "eval_feature_index.json",
        {
            "schema_version": "sample_fg.low_compute_eval_feature_cache.v1",
            "sample_count": len(eval_labels), "sha256": cache_digest,
            "base_class_count": base_class_count,
            "new_class_count": len(new_trainer.dm.dataset.classnames),
            "transform": "canonical_evaluation_transform",
        },
    )
    eval_cache_wall_s = time.perf_counter() - eval_cache_started

    replay_path = run_dir / "lc01" / "replay_trials.jsonl"
    fidelity_path = run_dir / "lc01" / "checkpoint_fidelity.jsonl"
    geometry_path = run_dir / "lc01" / "projection_geometry.jsonl"
    function_path = run_dir / "lc04" / "function_space_fidelity.jsonl"
    for path in (replay_path, fidelity_path, geometry_path, function_path):
        path.write_bytes(b"")
    replay_summaries = []
    geometry_summaries = []
    function_rows = []
    fidelity_rows = []
    bank_index = []
    exact_reference = None
    exact_metadata = None
    gradient_bank_wall_s = 0.0
    cpu_replay_wall_s = 0.0
    function_space_wall_s = 0.0
    started = time.perf_counter()
    frozen_base = hash_frozen_parameters(runtime.model)
    frozen_new = hash_frozen_parameters(new_model)
    with (
        TransitionGuard(runtime.trainer.optim, runtime.trainer.sched),
        TransitionGuard(new_trainer.optim, new_trainer.sched),
    ):
        for checkpoint_index, selected in enumerate(plan.checkpoints):
            checkpoint = load_probe_checkpoint(plan.source_run, selected.path)
            checkpoint.install_prompt(runtime.param_index)
            checkpoint.install_prompt(new_index)
            prompt_before = tuple(entry.parameter.detach().clone() for entry in runtime.param_index)
            actual_ema = checkpoint.actual_ema(runtime.param_index)
            for replicate in range(selected.materialization_replicates):
                bank_started = time.perf_counter()
                materialized = _materialize_features(runtime, selected.sha256, replicate)
                materialized_key = FeatureCacheKey(
                    dataset="dtd", split="fixed_materialized_train",
                    clip_sha256=source_config["model"]["checkpoint_sha256"],
                    transform_signature="pinned_coop_train_transform_per_sample_seed_v1",
                    checkpoint_sha256=selected.sha256, replicate=replicate,
                )
                materialized_hash = save_feature_cache(
                    run_dir / "cache" / f"train_{materialized_key.digest}.pt",
                    key=materialized_key,
                    features=torch.stack([row["feature"] for row in materialized]),
                    labels=torch.tensor([row["label"] for row in materialized]),
                    sample_ids=tuple(str(row["sample_id"]) for row in materialized),
                )
                bank = _gradient_bank(runtime, materialized, replicate)
                gradient_bank_wall_s += time.perf_counter() - bank_started
                bank_index.append(
                    {
                        "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                        "materialization_replicate": replicate,
                        "sample_count": bank.total_samples,
                        "batch_count": len(bank.batches),
                        "exact_gradient_sha256": gradient_sha256(bank.exact),
                        "materialized_feature_cache_sha256": materialized_hash,
                        "batches": [
                            {
                                "batch_index": item.batch_index,
                                "sample_ids": list(item.sample_ids),
                                "sample_count": item.sample_count,
                                "mean_loss": item.mean_loss,
                                "gradient_norm": float(item.gradient.norm().item()),
                                "gradient_sha256": item.gradient_sha256,
                            }
                            for item in bank.batches
                        ],
                    }
                )
                fidelity = gradient_metrics(actual_ema, bank.exact)
                fidelity_payload = {
                        "schema_version": METRICS_SCHEMA_VERSION,
                        "metric": "actual_checkpoint_ema_vs_materialized_exact",
                        "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                        "materialization_replicate": replicate, **fidelity,
                }
                _append_jsonl(fidelity_path, fidelity_payload)
                fidelity_rows.append(fidelity_payload)
                artifacts.append_metric(
                    {**fidelity_payload, "task": "lc01", "event_type": "checkpoint_fidelity"}
                )
                gradients = tuple(item.gradient for item in bank.batches)
                replay_started = time.perf_counter()
                for lambda_index, ema_lambda in enumerate(plan.lambda_grid):
                    canonical = ema_replay(gradients, ema_lambda)
                    canonical_metrics = gradient_metrics(canonical, bank.exact)
                    trials = permutation_trials(
                        gradients, bank.exact, ema_lambda=ema_lambda,
                        trial_count=plan.order_trials,
                        seed=plan.analysis_seed + checkpoint_index * 10000 + replicate * 1000 + lambda_index,
                    )
                    trial_gb_cosines = []
                    trial_degenerate = []
                    for trial in trials:
                        estimate = ema_replay(gradients, ema_lambda, order=trial.order)
                        geometry = [
                            projection_displacement_metrics(
                                gradient, estimate, bank.exact, rho=0.05, alpha=0.0015
                            )
                            for gradient in gradients
                        ]
                        row = {
                            "schema_version": METRICS_SCHEMA_VERSION,
                            "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                            "materialization_replicate": replicate,
                            "lambda": ema_lambda, "replay_mode": "cold_start",
                            "order_trial": trial.trial, "order": list(trial.order),
                            "last_batch_id": trial.last_batch_id,
                            **trial.metrics,
                            "gB_exact_cosine_mean": _stats(item["gB"]["cosine"] for item in geometry)["mean"],
                            "gB_exact_cosine_min": _stats(item["gB"]["cosine"] for item in geometry)["min"],
                            "delta_exact_cosine_mean": _stats(item["delta"]["cosine"] for item in geometry)["mean"],
                        }
                        _append_jsonl(replay_path, row)
                        trial_gb_cosines.append(row["gB_exact_cosine_mean"])
                        trial_degenerate.extend(
                            bool(item["estimate_projection_degenerate"])
                            for item in geometry
                        )
                    stationary_rows = []
                    for trial_index in range(plan.order_trials):
                        stationary_trial, trial_orders = stationary_ema_replay(
                            gradients, ema_lambda,
                            epochs=plan.stationary_replay_epochs,
                            seed=(
                                plan.analysis_seed + 1000000 + checkpoint_index * 100000
                                + replicate * 10000 + lambda_index * 1000 + trial_index
                            ),
                        )
                        stationary_geometry = [
                            projection_displacement_metrics(
                                gradient, stationary_trial, bank.exact,
                                rho=0.05, alpha=0.0015,
                            ) for gradient in gradients
                        ]
                        stationary_row = {
                            "schema_version": METRICS_SCHEMA_VERSION,
                            "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                            "materialization_replicate": replicate,
                            "lambda": ema_lambda, "replay_mode": "stationary_20_epochs",
                            "order_trial": trial_index,
                            "orders": [list(order) for order in trial_orders],
                            "last_batch_id": trial_orders[-1][-1],
                            **gradient_metrics(stationary_trial, bank.exact),
                            "gB_exact_cosine_mean": _stats(
                                item["gB"]["cosine"] for item in stationary_geometry
                            )["mean"],
                            "gB_exact_cosine_min": _stats(
                                item["gB"]["cosine"] for item in stationary_geometry
                            )["min"],
                            "delta_exact_cosine_mean": _stats(
                                item["delta"]["cosine"] for item in stationary_geometry
                            )["mean"],
                        }
                        _append_jsonl(replay_path, stationary_row)
                        stationary_rows.append(stationary_row)
                    replay_summaries.append(
                        {
                            "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                            "materialization_replicate": replicate, "lambda": ema_lambda,
                            "effective_sample_size": effective_sample_size(ema_lambda),
                            "history_k90": history_length(ema_lambda, 0.90),
                            "history_k95": history_length(ema_lambda, 0.95),
                            "history_k99": history_length(ema_lambda, 0.99),
                            "canonical_order": canonical_metrics,
                            "order_cosine": _stats(row.metrics["cosine"] for row in trials),
                            "order_relative_l2": _stats(row.metrics["relative_l2"] for row in trials),
                            "order_norm_ratio": _stats(row.metrics["norm_ratio"] for row in trials),
                            "order_gB_cosine": _stats(trial_gb_cosines),
                            "degenerate_projection_rate": (
                                sum(trial_degenerate) / len(trial_degenerate)
                            ),
                            "stationary_order_cosine": _stats(
                                row["cosine"] for row in stationary_rows
                                if row["cosine"] is not None
                            ),
                            "stationary_order_relative_l2": _stats(
                                row["relative_l2"] for row in stationary_rows
                                if row["relative_l2"] is not None
                            ),
                            "stationary_order_norm_ratio": _stats(
                                row["norm_ratio"] for row in stationary_rows
                                if row["norm_ratio"] is not None
                            ),
                        }
                    )
                    stationary, orders = stationary_ema_replay(
                        gradients, ema_lambda, epochs=plan.stationary_replay_epochs,
                        seed=plan.analysis_seed + 500000 + checkpoint_index * 10000 + replicate * 1000 + lambda_index,
                    )
                    stationary_metrics = gradient_metrics(stationary, bank.exact)
                    geometry = [
                        projection_displacement_metrics(
                            gradient, stationary, bank.exact, rho=0.05, alpha=0.0015
                        ) for gradient in gradients
                    ]
                    geometry_row = {
                        "schema_version": METRICS_SCHEMA_VERSION,
                        "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                        "materialization_replicate": replicate, "lambda": ema_lambda,
                        "replay_mode": "stationary_20_epochs",
                        "exact_cosine": stationary_metrics["cosine"],
                        "relative_l2": stationary_metrics["relative_l2"],
                        "norm_ratio": stationary_metrics["norm_ratio"],
                        "gB_exact_cosine_mean": _stats(item["gB"]["cosine"] for item in geometry)["mean"],
                        "gB_exact_cosine_min": _stats(item["gB"]["cosine"] for item in geometry)["min"],
                        "delta_exact_cosine_mean": _stats(item["delta"]["cosine"] for item in geometry)["mean"],
                    }
                    _append_jsonl(geometry_path, geometry_row)
                    geometry_summaries.append(geometry_row)
                    artifacts.append_metric(
                        {**geometry_row, "task": "lc01", "event_type": "projection_geometry"}
                    )
                cpu_replay_wall_s += time.perf_counter() - replay_started

                if replicate == 0:
                    # LC04 uses the primary materialization direction only; the
                    # two extra final banks are LC01 augmentation robustness.
                    def all_text_features():
                        with torch.no_grad():
                            new_index[0].parameter.copy_(
                                runtime.param_index[0].parameter.detach().to(new_index[0].parameter)
                            )
                        return torch.cat((_text_features(runtime.model), _text_features(new_model)))

                    runtime.model.eval()
                    new_model.eval()
                    function_started = time.perf_counter()
                    lc04_rows = compare_functional_directions(
                        param_index=runtime.param_index,
                        ema_direction=_device_state(actual_ema, runtime.param_index),
                        exact_direction=_device_state(bank.exact, runtime.param_index),
                        radii=plan.radii,
                        text_feature_fn=all_text_features,
                        eval_image_features=eval_features,
                        eval_labels=eval_labels,
                        base_class_count=base_class_count,
                        logit_scale=runtime.model.logit_scale.exp().detach().cpu(),
                    )
                    function_space_wall_s += time.perf_counter() - function_started
                    with torch.no_grad():
                        new_index[0].parameter.copy_(runtime.param_index[0].parameter.detach().to(new_index[0].parameter))
                    for row in lc04_rows:
                        payload = {
                            "schema_version": METRICS_SCHEMA_VERSION,
                            "checkpoint_sha256": selected.sha256, "epoch": selected.epoch,
                            "materialization_replicate": replicate, **row,
                        }
                        _append_jsonl(function_path, payload)
                        function_rows.append(payload)
                        artifacts.append_metric(
                            {**payload, "task": "lc04", "event_type": "function_space_fidelity"}
                        )
            if selected.epoch == max(item.epoch for item in plan.checkpoints):
                runtime.model.train()
                service = runtime.engine.diagnostic_coordinator.full_gradient_service
                queried = service.compute(
                    optimizer_step=selected.optimizer_step,
                    purpose="lc01_optional_independent_exact_reference",
                )
                exact_reference = _cpu_state(queried.gradient)
                exact_metadata = queried.metadata.as_dict()
                exact_metadata["materialized_reference_comparison"] = gradient_metrics(
                    bank.exact, exact_reference
                )
            if any(not torch.equal(entry.parameter, before) for entry, before in zip(runtime.param_index, prompt_before)):
                raise IntegratedProbeError("Prompt changed while analyzing a checkpoint")
            if hash_frozen_parameters(runtime.model) != frozen_base or hash_frozen_parameters(new_model) != frozen_new:
                raise IntegratedProbeError("Frozen CLIP changed during the probe")
            verify_source_immutable(checkpoint)

    elapsed = time.perf_counter() - started
    artifacts.write_lc01(
        "gradient_bank_index.json",
        {"schema_version": "sample_fg.low_compute_gradient_bank_index.v1", "banks": bank_index},
    )
    gate_rows = []
    for row in replay_summaries:
        if row["materialization_replicate"] != 0:
            continue
        gate_rows.append(
            {
                "checkpoint": row["epoch"], "lambda": row["lambda"],
                "median_exact_cosine": row["order_cosine"]["median"],
                "median_relative_l2": row["order_relative_l2"]["median"],
                "median_gB_exact_cosine": row["order_gB_cosine"]["median"],
                "degenerate_projection_rate": row["degenerate_projection_rate"],
            }
        )
    lc02_gate = evaluate_lc02_gate(gate_rows)
    artifacts.write_lc01(
        "replay_summary.json",
        {
            "schema_version": "sample_fg.low_compute_replay_summary.v1",
            "rows": replay_summaries, "lc02_gate": lc02_gate,
        },
    )
    artifacts.write_lc01(
        "geometry_summary.json",
        {"schema_version": "sample_fg.low_compute_geometry_summary.v1", "rows": geometry_summaries},
    )
    accounting = {
        "schema_version": "sample_fg.low_compute_accounting.v1",
        "planned": plan.budget.as_dict(),
        "optimizer_steps_executed": 0, "scheduler_steps_executed": 0,
        "total_wall_s": elapsed,
        "exact_gradient_gpu_wall_s": (
            None if exact_metadata is None else exact_metadata["elapsed_s"]
        ),
        "minibatch_gradient_capture_gpu_wall_s": gradient_bank_wall_s,
        "evaluation_image_feature_cache_wall_s": eval_cache_wall_s,
        "function_space_forward_wall_s": function_space_wall_s,
        "cpu_permutation_replay_wall_s": cpu_replay_wall_s,
        "plotting_aggregation_wall_s": 0.0,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        "optional_exact_reference": exact_metadata,
    }
    artifacts.write_lc01("compute_accounting.json", accounting)
    artifacts.write_lc04(
        {"schema_version": "sample_fg.low_compute_function_space.v1", "rows": function_rows}
    )
    artifacts.write_summary(
        {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "completed",
            "title": "How Global Is SAMPLe's Global Gradient?",
            "checkpoint_count": len(plan.checkpoints),
            "lambda_grid": list(plan.lambda_grid),
            "order_trials": plan.order_trials,
            "finite_difference_h": list(plan.radii),
            "primary_findings": {
                "actual_checkpoint_ema_vs_materialized_exact": fidelity_rows,
                "function_space_radius_0_005": [
                    row for row in function_rows if row["radius"] == 0.005
                ],
                "lc02_gate": lc02_gate,
            },
            "safety": {
                "optimizer_steps_executed": 0,
                "scheduler_steps_executed": 0,
                "model_parameters_changed": False,
                "source_artifacts_changed": False,
                "frozen_clip_changed": False,
            },
            "interpretation_policy": "diagnostic_estimator_fidelity_not_algorithm_validity",
        }
    )
    return run_dir
