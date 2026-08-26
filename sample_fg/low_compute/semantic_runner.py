"""LC05 frozen-checkpoint semantic drift and open-world campaign lifecycle."""

from __future__ import annotations

import json
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from sample_fg.coop_anchor import hash_frozen_parameters
from sample_fg.data_protocol import DATASET_SPECS
from sample_fg.results import atomic_write_json

from .budget import ComputeBudget, TransitionGuard
from .campaign_sources import (
    DiscoveryReport,
    R2Source,
    build_r2_scientific_plan,
)
from .checkpoint_probe import sha256_file, verify_source_immutable
from .probe_runtime import (
    build_dataset_runtime,
    build_or_reuse_eval_cache,
    install_checkpoint,
    text_features,
    write_cache_index,
)
from .semantic import (
    compute_neighbor_preservation,
    compute_semantic_drift,
    compute_topology_distortion,
    descriptive,
    evaluate_open_world_logits,
    evaluate_standard_logits,
    pearson_spearman,
)


LC05_SCHEMA = "sample_fg.low_compute_lc05.v1"
PARITY_TOLERANCE_PCT = 1e-4


class SemanticCampaignError(RuntimeError):
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


def build_lc05_dry_run(
    discovery: DiscoveryReport,
    *,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    output_root: Path,
    reusable_cache_root: Path | None,
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
                **source.key.as_dict(),
                "manifest": str(plan.manifest_path),
                "selected_source_fingerprint": plan.source.fingerprint,
                "clip_checkpoint": str(plan.clip_checkpoint),
                "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
            }
        )
    datasets = sorted({source.key.dataset for source in discovery.compatible})
    eval_batches = sum(
        (DATASET_SPECS[name].expected_split_counts[2] + 99) // 100 + 1
        for name in datasets
    )
    budget = ComputeBudget(
        optimizer_steps=0,
        scheduler_steps=0,
        image_encoder_forward_batches=eval_batches,
        text_encoder_forward_calls=2 * len(discovery.compatible) + 2 * len(datasets),
    )
    budget.require_read_only()
    return {
        "schema_version": LC05_SCHEMA,
        "status": "DRY_RUN_VALIDATED",
        "task": "lc05",
        "dry_run": True,
        "training_started": False,
        "source_artifacts_read_only": True,
        "discovery": discovery.as_dict(),
        "resolved_resources": resources,
        "cache_policy": {
            "one_eval_image_feature_cache_per_dataset_clip": True,
            "reuse_root": None if reusable_cache_root is None else str(Path(reusable_cache_root).resolve()),
            "estimated_image_batches_is_upper_bound_before_cache_lookup": True,
        },
        "budget": budget.as_dict(),
    }


def _summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["method_key"])].append(row)
    summaries = []
    scalar_paths = {
        "semantic_drift_all": ("semantic_drift", "all", "mean_cosine_drift"),
        "topology_distortion_all": ("topology_distortion", "all_off_diagonal"),
        "standard_base": ("standard_evaluation", "base_accuracy_pct"),
        "standard_new": ("standard_evaluation", "new_accuracy_pct"),
        "standard_hm": ("standard_evaluation", "hm_pct"),
        "open_world_base": ("open_world", "open_world_base_accuracy_pct"),
        "open_world_new": ("open_world", "open_world_new_accuracy_pct"),
        "open_world_hm": ("open_world", "open_world_hm_pct"),
    }

    def resolve(row, path):
        value = row
        for key in path:
            value = value[key]
        return float(value)

    for (dataset, method), values in sorted(groups.items()):
        summaries.append(
            {
                "dataset": dataset,
                "method_key": method,
                "seeds": sorted(row["seed"] for row in values),
                "metrics": {
                    label: descriptive(resolve(row, path) for row in values)
                    for label, path in scalar_paths.items()
                },
            }
        )
    correlations = {}
    pair_specs = {
        "semantic_drift_vs_standard_new": (
            ("semantic_drift", "all", "mean_cosine_drift"),
            ("standard_evaluation", "new_accuracy_pct"),
        ),
        "semantic_drift_vs_open_world_new": (
            ("semantic_drift", "all", "mean_cosine_drift"),
            ("open_world", "open_world_new_accuracy_pct"),
        ),
        "topology_distortion_vs_standard_hm": (
            ("topology_distortion", "all_off_diagonal"),
            ("standard_evaluation", "hm_pct"),
        ),
        "standard_hm_vs_open_world_hm": (
            ("standard_evaluation", "hm_pct"),
            ("open_world", "open_world_hm_pct"),
        ),
    }
    for label, (left, right) in pair_specs.items():
        correlations[label] = pearson_spearman(
            [(resolve(row, left), resolve(row, right)) for row in rows]
        )
    return {
        "schema_version": LC05_SCHEMA,
        "groups": summaries,
        "correlations": correlations,
        "significance_tests": False,
    }


def run_lc05(
    discovery: DiscoveryReport,
    *,
    data_root: Path,
    manifest_root: Path,
    clip_cache: Path,
    output_root: Path,
    reusable_cache_root: Path | None,
) -> Path:
    """Execute inference-only LC05 over every locally compatible source."""

    if not discovery.compatible:
        raise SemanticCampaignError("LC05 has no compatible completed R2 checkpoints")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output_root).resolve() / "lc05" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "cache").mkdir()
    (run_dir / "plots").mkdir()
    semantic_path = run_dir / "semantic_drift.jsonl"
    open_path = run_dir / "open_world_eval.jsonl"
    semantic_path.write_bytes(b""); open_path.write_bytes(b"")
    atomic_write_json(run_dir / "source_discovery.json", discovery.as_dict())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cpu_started = time.process_time()
    rows: list[dict[str, Any]] = []
    parity_rows = []
    zero_shot_rows = []
    cache_rows = []
    image_batches = 0
    text_calls = 0
    source_by_dataset: dict[str, list[R2Source]] = defaultdict(list)
    for source in discovery.compatible:
        source_by_dataset[source.key.dataset].append(source)

    for dataset, sources in sorted(source_by_dataset.items()):
        anchor = sorted(sources, key=lambda value: value.key)[0]
        plan = build_r2_scientific_plan(
            anchor,
            data_root=data_root,
            manifest_root=manifest_root,
            clip_cache=clip_cache,
            runtime_output_root=run_dir / "_runtime",
        )
        runtime = build_dataset_runtime(anchor, plan, runtime_root=run_dir / "_runtime")
        base_index = runtime.base_runtime.param_index
        new_index = runtime.new_index
        base_initial = tuple(entry.parameter.detach().clone() for entry in base_index)
        new_initial = tuple(entry.parameter.detach().clone() for entry in new_index)
        frozen_base = hash_frozen_parameters(runtime.base_runtime.model)
        frozen_new = hash_frozen_parameters(runtime.new_model)
        expected = DATASET_SPECS[dataset]
        if (
            len(runtime.base_classnames) != expected.expected_base_classes
            or len(runtime.new_classnames) != expected.expected_new_classes
            or len(set(runtime.base_classnames + runtime.new_classnames))
            != expected.expected_total_classes
        ):
            raise SemanticCampaignError(f"Canonical class ordering/count differs for {dataset}")
        reference_base = text_features(runtime.base_runtime.model)
        reference_new = text_features(runtime.new_model)
        reference = torch.cat((reference_base, reference_new))
        text_calls += 2
        cache = build_or_reuse_eval_cache(
            runtime,
            destination=run_dir / "cache" / f"eval_{dataset}.pt",
            reusable_cache_root=reusable_cache_root,
        )
        image_batches += int(cache["image_encoder_forward_batches"])
        base_count = len(reference_base)
        base_mask = cache["labels"] < base_count
        new_mask = ~base_mask
        base_features = cache["features"][base_mask]
        base_labels = cache["labels"][base_mask]
        new_features = cache["features"][new_mask]
        new_labels_local = cache["labels"][new_mask] - base_count
        write_cache_index(
            run_dir / "cache" / f"eval_{dataset}_index.json",
            cache,
            dataset=dataset,
            class_order=list(runtime.base_classnames + runtime.new_classnames),
            base_class_count=base_count,
            new_class_count=len(reference_new),
            clip_sha256=anchor.source_config["model"]["checkpoint_sha256"],
            transform_signature="canonical_evaluation_transform",
        )
        cache_rows.append(
            {
                "dataset": dataset, "reused": cache["reused"],
                "source_path": cache["source_path"], "sample_count": len(cache["labels"]),
            }
        )
        zero_standard = evaluate_standard_logits(
            base_image_features=base_features,
            base_labels=base_labels,
            base_text_features=reference_base,
            new_image_features=new_features,
            new_labels=new_labels_local,
            new_text_features=reference_new,
            logit_scale=runtime.base_runtime.model.logit_scale.exp(),
        )
        zero_open = evaluate_open_world_logits(
            image_features=cache["features"], labels=cache["labels"],
            text_features=reference, base_class_count=base_count,
            logit_scale=runtime.base_runtime.model.logit_scale.exp(),
        )
        zero_shot_rows.append(
            {
                "schema_version": LC05_SCHEMA, "dataset": dataset,
                "method_key": "zero_shot_clip", "standard_evaluation": zero_standard,
                "open_world": zero_open,
            }
        )
        with (
            TransitionGuard(runtime.base_runtime.trainer.optim, runtime.base_runtime.trainer.sched),
            TransitionGuard(runtime.new_trainer.optim, runtime.new_trainer.sched),
        ):
            for source in sorted(sources, key=lambda value: value.key):
                # Validate every method/seed protocol, not only the dataset anchor.
                build_r2_scientific_plan(
                    source,
                    data_root=data_root,
                    manifest_root=manifest_root,
                    clip_cache=clip_cache,
                    runtime_output_root=run_dir / "_runtime",
                )
                install_checkpoint(runtime, source)
                learned_base = text_features(runtime.base_runtime.model)
                learned_new = text_features(runtime.new_model)
                learned = torch.cat((learned_base, learned_new))
                text_calls += 2
                standard = evaluate_standard_logits(
                    base_image_features=base_features,
                    base_labels=base_labels,
                    base_text_features=learned_base,
                    new_image_features=new_features,
                    new_labels=new_labels_local,
                    new_text_features=learned_new,
                    logit_scale=runtime.base_runtime.model.logit_scale.exp(),
                )
                expected_eval = source.source_summary["evaluation"]
                differences = {
                    name: abs(standard[name] - float(expected_eval[name]))
                    for name in ("base_accuracy_pct", "new_accuracy_pct", "hm_pct")
                }
                parity_passed = all(value <= PARITY_TOLERANCE_PCT for value in differences.values())
                parity = {
                    **source.key.as_dict(), "run_dir": str(source.run_dir),
                    "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
                    "observed": standard,
                    "source": {name: float(expected_eval[name]) for name in differences},
                    "absolute_difference_pct": differences,
                    "tolerance_pct": PARITY_TOLERANCE_PCT,
                    "passed": parity_passed,
                }
                parity_rows.append(parity)
                if not parity_passed:
                    raise SemanticCampaignError(
                        f"Cached standard evaluation parity failed for {source.key.as_dict()}"
                    )
                row = {
                    "schema_version": LC05_SCHEMA,
                    **source.key.as_dict(),
                    "method_key": _method_label(source),
                    "source_run": str(source.run_dir),
                    "source_config_sha256": source.checkpoint.source_config_sha256,
                    "checkpoint_sha256": source.checkpoint.checkpoint_sha256,
                    "class_order": list(runtime.base_classnames + runtime.new_classnames),
                    "semantic_drift": compute_semantic_drift(
                        reference, learned, base_class_count=base_count
                    ),
                    "topology_distortion": compute_topology_distortion(
                        reference, learned, base_class_count=base_count
                    ),
                    "neighbor_preservation": compute_neighbor_preservation(
                        reference, learned, base_class_count=base_count
                    ),
                    "standard_evaluation": standard,
                    "open_world": evaluate_open_world_logits(
                        image_features=cache["features"], labels=cache["labels"],
                        text_features=learned, base_class_count=base_count,
                        logit_scale=runtime.base_runtime.model.logit_scale.exp(),
                    ),
                }
                rows.append(row)
                _append(semantic_path, row)
                _append(open_path, {
                    key: row[key] for key in (
                        "schema_version", "dataset", "shots", "method", "estimator",
                        "method_key", "seed", "source_run", "source_config_sha256",
                        "checkpoint_sha256", "standard_evaluation", "open_world",
                    )
                })
                verify_source_immutable(source.checkpoint)
        _restore(base_index, base_initial); _restore(new_index, new_initial)
        if hash_frozen_parameters(runtime.base_runtime.model) != frozen_base:
            raise SemanticCampaignError("Frozen Base CLIP changed during LC05")
        if hash_frozen_parameters(runtime.new_model) != frozen_new:
            raise SemanticCampaignError("Frozen New CLIP changed during LC05")

    summary_payload = _summarize(rows)
    atomic_write_json(run_dir / "standard_eval_parity.json", {
        "schema_version": LC05_SCHEMA, "all_passed": all(row["passed"] for row in parity_rows),
        "rows": parity_rows,
    })
    atomic_write_json(run_dir / "zero_shot_reference.json", {
        "schema_version": LC05_SCHEMA, "rows": zero_shot_rows,
    })
    atomic_write_json(run_dir / "semantic_summary.json", summary_payload)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - started
    accounting = {
        "schema_version": LC05_SCHEMA,
        "optimizer_steps": 0, "scheduler_steps": 0,
        "normal_backward_batches": 0, "exact_backward_batches": 0,
        "image_encoder_forward_batches": image_batches,
        "text_encoder_forward_calls": text_calls,
        "compatible_checkpoint_count": len(rows),
        "missing_checkpoint_count": len(discovery.missing),
        "cache_rows": cache_rows,
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
    source_hashes = {
        str(source.run_dir): dict(source.checkpoint.source_artifact_sha256)
        for source in discovery.compatible
    }
    atomic_write_json(run_dir / "source_hashes.json", {
        "schema_version": LC05_SCHEMA, "sources": source_hashes,
    })
    atomic_write_json(run_dir / "summary.json", {
        "schema_version": LC05_SCHEMA, "status": "completed", "task": "lc05",
        "compatible_checkpoint_count": len(rows),
        "missing_checkpoint_count": len(discovery.missing),
        "datasets": sorted(source_by_dataset),
        "safety": {
            "optimizer_steps_executed": 0, "scheduler_steps_executed": 0,
            "source_artifacts_changed": False, "model_parameters_changed": False,
            "new_class_retraining": False,
        },
        "artifacts": {
            "semantic_drift": "semantic_drift.jsonl",
            "open_world_eval": "open_world_eval.jsonl",
            "standard_eval_parity": "standard_eval_parity.json",
            "semantic_summary": "semantic_summary.json",
            "zero_shot_reference": "zero_shot_reference.json",
            "compute_accounting": "compute_accounting.json",
        },
    })
    return run_dir
