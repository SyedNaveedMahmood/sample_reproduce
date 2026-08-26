"""LC06 fixed-materialization prompt-space sharpness campaign lifecycle."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sample_fg.coop_anchor import hash_frozen_parameters
from sample_fg.gradient_state import GradientState
from sample_fg.results import atomic_write_json

from .budget import ComputeBudget, TransitionGuard
from .campaign_sources import DiscoveryReport, R2Source, build_r2_scientific_plan
from .checkpoint_probe import verify_source_immutable
from .probe_runtime import (
    build_dataset_runtime,
    build_or_reuse_training_cache,
    fixed_feature_loss_fn,
    write_cache_index,
)
from .semantic import pearson_spearman
from .sharpness import (
    NUM_RANDOM_DIRECTIONS,
    RADII,
    exact_materialized_gradient,
    parameter_sha256,
    probe_structured_direction,
    probe_symmetric_loss_sharpness,
    sample_prompt_directions,
    summarize_sharpness,
)


LC06_SCHEMA = "sample_fg.low_compute_lc06.v1"


class SharpnessCampaignError(RuntimeError):
    pass


def _append(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(payload), allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n")


def _restore(index, values) -> None:
    with torch.no_grad():
        for entry, value in zip(index, values):
            entry.parameter.copy_(value)
            entry.parameter.grad = None


def _method_label(source: R2Source) -> str:
    return "sample_ema" if source.key.method == "sample" else source.key.method


def _device_state(state: GradientState, param_index) -> GradientState:
    return GradientState.from_tensors(
        param_index,
        (
            value.to(device=entry.parameter.device)
            for entry, value in zip(param_index, state)
        ),
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _lc05_index(lc05_run: Path | None) -> dict[str, dict[str, Any]]:
    if lc05_run is None:
        return {}
    return {
        str(row["checkpoint_sha256"]): row
        for row in _jsonl(Path(lc05_run) / "semantic_drift.jsonl")
        if isinstance(row.get("checkpoint_sha256"), str)
    }


def _linked_mechanism_evidence(root: Path | None, checkpoint_sha: str) -> list[dict[str, Any]]:
    if root is None or not Path(root).is_dir():
        return []
    rows = []
    names = {
        "checkpoint_fidelity.jsonl": "LC01",
        "function_space_fidelity.jsonl": "LC04",
        "checkpoint_trajectory.jsonl": "LC03",
    }
    for path in sorted(Path(root).rglob("*.jsonl")):
        if path.name not in names:
            continue
        for row in _jsonl(path):
            if row.get("checkpoint_sha256") == checkpoint_sha:
                rows.append(
                    {
                        "task": names[path.name], "artifact": str(path.resolve()),
                        "artifact_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                        "row": row,
                    }
                )
    return rows


def build_lc06_dry_run(
    discovery: DiscoveryReport,
    *,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    output_root: Path,
) -> dict[str, Any]:
    resources = []
    for source in discovery.compatible:
        plan = build_r2_scientific_plan(
            source,
            data_root=data_root,
            manifest_root=manifest_root,
            clip_cache=clip_cache,
            runtime_output_root=Path(output_root) / "_runtime",
        )
        resources.append(
            {
                **source.key.as_dict(), "manifest": str(plan.manifest_path),
                "selected_source_fingerprint": plan.source.fingerprint,
                "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
            }
        )
    dataset_seeds = {
        (source.key.dataset, source.key.seed) for source in discovery.compatible
    }
    image_batches = sum(
        (next(
            int(source.source_config["data"]["selected_count"])
            for source in discovery.compatible
            if (source.key.dataset, source.key.seed) == key
        ) + 31) // 32
        for key in dataset_seeds
    )
    structured_calls = sum(7 + (7 if source.key.method == "sample" else 0) for source in discovery.compatible)
    budget = ComputeBudget(
        optimizer_steps=0,
        scheduler_steps=0,
        exact_forward_batches=len(discovery.compatible),
        exact_backward_batches=len(discovery.compatible),
        image_encoder_forward_batches=image_batches,
        text_encoder_forward_calls=(
            len(discovery.compatible) * (2 + NUM_RANDOM_DIRECTIONS * 2 * len(RADII))
            + structured_calls
        ),
    )
    budget.require_read_only()
    return {
        "schema_version": LC06_SCHEMA,
        "status": "DRY_RUN_VALIDATED", "task": "lc06", "dry_run": True,
        "training_started": False, "source_artifacts_read_only": True,
        "discovery": discovery.as_dict(), "resolved_resources": resources,
        "probe": {
            "objective": "fixed-materialization prompt-space sharpness",
            "radii": list(RADII), "num_random_directions": NUM_RANDOM_DIRECTIONS,
            "directions_per_checkpoint_keyed_by_checkpoint_sha256": True,
            "image_features_cached_not_recomputed_per_direction": True,
            "hessian_or_hvp": False,
        },
        "budget": budget.as_dict(),
    }


def _correlations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for radius in RADII:
        selected = [row for row in rows if row["radius"] == radius]
        specs = {
            "sharpness_vs_standard_new": lambda row: row["source_evaluation"]["new_accuracy_pct"],
            "sharpness_vs_open_world_new": lambda row: None if row["lc05"] is None else row["lc05"]["open_world"]["open_world_new_accuracy_pct"],
            "sharpness_vs_semantic_drift": lambda row: None if row["lc05"] is None else row["lc05"]["semantic_drift"]["all"]["mean_cosine_drift"],
        }
        output[str(radius)] = {}
        for name, getter in specs.items():
            pairs = []
            for row in selected:
                right = getter(row)
                if right is not None:
                    pairs.append((float(row["sharpness_mean"]), float(right)))
            output[str(radius)][name] = pearson_spearman(pairs)
    return output


def run_lc06(
    discovery: DiscoveryReport,
    *,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    output_root: Path,
    lc05_run: Path | None,
    evidence_root: Path | None,
) -> Path:
    if not discovery.compatible:
        raise SharpnessCampaignError("LC06 has no compatible completed R2 checkpoints")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root).resolve() / "lc06" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "cache").mkdir(); (run_dir / "plots").mkdir()
    probe_path = run_dir / "sharpness_probe.jsonl"
    structured_path = run_dir / "structured_direction_probe.jsonl"
    probe_path.write_bytes(b""); structured_path.write_bytes(b"")
    atomic_write_json(run_dir / "source_discovery.json", discovery.as_dict())
    lc05 = _lc05_index(lc05_run)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cpu_started = time.process_time()
    summary_rows = []
    structured_rows = []
    image_batches = 0
    text_calls = 0
    backwards = 0
    groups: dict[tuple[str, int], list[R2Source]] = defaultdict(list)
    for source in discovery.compatible:
        groups[(source.key.dataset, source.key.seed)].append(source)

    for (dataset, seed), sources in sorted(groups.items()):
        anchor = sorted(sources, key=lambda value: value.key)[0]
        plan = build_r2_scientific_plan(
            anchor,
            data_root=data_root,
            manifest_root=manifest_root,
            clip_cache=clip_cache,
            runtime_output_root=run_dir / "_runtime",
        )
        runtime = build_dataset_runtime(anchor, plan, runtime_root=run_dir / "_runtime")
        index = runtime.base_runtime.param_index
        initial = tuple(entry.parameter.detach().clone() for entry in index)
        frozen = hash_frozen_parameters(runtime.base_runtime.model)
        cache = build_or_reuse_training_cache(
            runtime,
            destination=run_dir / "cache" / f"train_{dataset}_seed{seed}.pt",
        )
        image_batches += int(cache["image_encoder_forward_batches"])
        write_cache_index(
            run_dir / "cache" / f"train_{dataset}_seed{seed}_index.json",
            cache,
            dataset=dataset, seed=seed,
            selected_source_fingerprint=plan.source.fingerprint,
            transform_signature="pinned_coop_train_transform_per_sample_seed_v1",
            metric_scope="fixed-materialization prompt-space sharpness",
        )
        with TransitionGuard(runtime.base_runtime.trainer.optim, runtime.base_runtime.trainer.sched):
            for source in sorted(sources, key=lambda value: value.key):
                build_r2_scientific_plan(
                    source,
                    data_root=data_root,
                    manifest_root=manifest_root,
                    clip_cache=clip_cache,
                    runtime_output_root=run_dir / "_runtime",
                )
                source.checkpoint.install_prompt(index)
                checkpoint_parameter_hash = parameter_sha256(index)
                loss_fn = fixed_feature_loss_fn(
                    runtime.base_runtime.model,
                    features=cache["features"], labels=cache["labels"],
                )
                reference_loss, exact = exact_materialized_gradient(
                    param_index=index, loss_fn=loss_fn
                )
                backwards += 1; text_calls += 1
                directions = sample_prompt_directions(
                    index, checkpoint_sha256=source.checkpoint.checkpoint_sha256
                )
                raw_rows = probe_symmetric_loss_sharpness(
                    param_index=index, loss_fn=loss_fn, directions=directions
                )
                text_calls += 1 + len(raw_rows) * 2
                max_logical_norm_error = 0.0
                max_live_norm_error = 0.0
                for raw in raw_rows:
                    for label in ("plus", "minus"):
                        max_logical_norm_error = max(
                            max_logical_norm_error,
                            abs(float(raw[f"logical_displacement_norm_{label}"]) - float(raw["radius"])),
                        )
                        max_live_norm_error = max(
                            max_live_norm_error,
                            abs(float(raw[f"live_displacement_norm_{label}"]) - float(raw["radius"])),
                        )
                    _append(probe_path, {
                        "schema_version": LC06_SCHEMA, **source.key.as_dict(),
                        "method_key": _method_label(source),
                        "source_config_sha256": source.checkpoint.source_config_sha256,
                        "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
                        "reference_loss": reference_loss, **raw,
                    })
                if max_logical_norm_error > 1e-6:
                    raise SharpnessCampaignError(
                        "Logical FP32 prompt displacement norm tolerance failed: "
                        f"{max_logical_norm_error}"
                    )
                relevant = {"exact_materialized_gradient": exact}
                if source.key.method == "sample":
                    relevant["stored_ema"] = _device_state(
                        source.checkpoint.actual_ema(index), index
                    )
                checkpoint_structured = []
                for name, direction in relevant.items():
                    values = probe_structured_direction(
                        name=name, param_index=index, loss_fn=loss_fn, direction=direction
                    )
                    text_calls += 1 + len(values) * 2
                    for value in values:
                        row = {
                            "schema_version": LC06_SCHEMA, **source.key.as_dict(),
                            "method_key": _method_label(source),
                            "source_config_sha256": source.checkpoint.source_config_sha256,
                            "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
                            "reference_loss": reference_loss, **value,
                        }
                        checkpoint_structured.append(row); structured_rows.append(row)
                        _append(structured_path, row)
                linked_lc05 = lc05.get(source.checkpoint.checkpoint_sha256)
                for radius in RADII:
                    selected = [row for row in raw_rows if row["radius"] == radius]
                    summary = summarize_sharpness(
                        selected, baseline_loss=reference_loss, radius=radius
                    )
                    summary_rows.append(
                        {
                            "schema_version": LC06_SCHEMA, **source.key.as_dict(),
                            "method_key": _method_label(source),
                            "source_run": str(source.run_dir),
                            "source_config_sha256": source.checkpoint.source_config_sha256,
                            "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
                            "checkpoint_parameter_sha256": checkpoint_parameter_hash,
                            "selected_source_fingerprint": plan.source.fingerprint,
                            "metric_scope": "fixed-materialization prompt-space sharpness",
                            "exact_materialized_gradient_norm": float(exact.norm().detach().cpu().item()),
                            "source_evaluation": {
                                key: float(source.source_summary["evaluation"][key])
                                for key in ("base_accuracy_pct", "new_accuracy_pct", "hm_pct")
                            },
                            "lc05": linked_lc05,
                            "mechanism_evidence": _linked_mechanism_evidence(
                                evidence_root, source.checkpoint.checkpoint_sha256
                            ),
                            "max_logical_displacement_norm_absolute_error": max_logical_norm_error,
                            "max_live_fp16_displacement_norm_absolute_error": max_live_norm_error,
                            "live_fp16_radius_policy": "record_quantization_error_do_not_change_direction",
                            **summary,
                        }
                    )
                if parameter_sha256(index) != checkpoint_parameter_hash:
                    raise SharpnessCampaignError("Checkpoint prompt was not restored after LC06")
                verify_source_immutable(source.checkpoint)
        _restore(index, initial)
        if hash_frozen_parameters(runtime.base_runtime.model) != frozen:
            raise SharpnessCampaignError("Frozen CLIP changed during LC06")

    atomic_write_json(run_dir / "sharpness_summary.json", {
        "schema_version": LC06_SCHEMA, "rows": summary_rows,
    })
    atomic_write_json(run_dir / "structured_direction_probe.json", {
        "schema_version": LC06_SCHEMA, "rows": structured_rows,
    })
    atomic_write_json(run_dir / "sharpness_correlations.json", {
        "schema_version": LC06_SCHEMA, "correlations": _correlations(summary_rows),
        "significance_tests": False,
    })
    atomic_write_json(run_dir / "source_hashes.json", {
        "schema_version": LC06_SCHEMA,
        "sources": {
            str(source.run_dir): dict(source.checkpoint.source_artifact_sha256)
            for source in discovery.compatible
        },
    })
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - started
    accounting = {
        "schema_version": LC06_SCHEMA, "optimizer_steps": 0, "scheduler_steps": 0,
        "image_encoder_forward_batches": image_batches,
        "prompt_backward_operations": backwards,
        "text_encoder_forward_calls": text_calls,
        "random_directions_per_checkpoint": NUM_RANDOM_DIRECTIONS,
        "radii": list(RADII), "hessian_or_hvp_calls": 0,
        "compatible_checkpoint_count": len(discovery.compatible),
        "missing_checkpoint_count": len(discovery.missing),
        "wall_time_s": wall_time_s,
        "cpu_wall_time_s": wall_time_s,
        "cpu_process_time_s": time.process_time() - cpu_started,
        "gpu_wall_time_s": wall_time_s if torch.cuda.is_available() else 0.0,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
        ),
        "timing_semantics": {
            "cpu_wall_time_s": "host-observed end-to-end wall time",
            "cpu_process_time_s": "process CPU time",
            "gpu_wall_time_s": "CUDA-synchronized end-to-end wall time",
        },
    }
    atomic_write_json(run_dir / "compute_accounting.json", accounting)
    atomic_write_json(run_dir / "summary.json", {
        "schema_version": LC06_SCHEMA, "status": "completed", "task": "lc06",
        "compatible_checkpoint_count": len(discovery.compatible),
        "missing_checkpoint_count": len(discovery.missing),
        "safety": {
            "optimizer_steps_executed": 0, "scheduler_steps_executed": 0,
            "source_artifacts_changed": False, "model_parameters_changed": False,
            "hessian_or_hvp_executed": False,
        },
        "artifacts": {
            "sharpness_probe": "sharpness_probe.jsonl",
            "sharpness_summary": "sharpness_summary.json",
            "structured_direction_probe": "structured_direction_probe.jsonl",
            "sharpness_correlations": "sharpness_correlations.json",
            "compute_accounting": "compute_accounting.json",
        },
    })
    return run_dir
