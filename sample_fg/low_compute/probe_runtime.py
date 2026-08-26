"""Shared frozen CLIP runtime and feature-cache helpers for LC05/LC06."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from dassl.utils import set_random_seed
from sample_fg.coop_anchor import build_coop_trainer, unwrap_model
from sample_fg.full_gradient import build_full_gradient_loader
from sample_fg.param_index import ParamIndex
from sample_fg.paper_runner import _build_runtime, build_scientific_cfg
from sample_fg.rng import isolated_rng

from .campaign_sources import R2Source
from .feature_cache import (
    FeatureCacheKey,
    load_feature_cache,
    materialization_seed_clock,
    save_feature_cache,
)
from .runner import _eval_feature_cache, _text_features


@dataclass
class DatasetRuntime:
    source: R2Source
    scientific_plan: Any
    base_runtime: Any
    new_trainer: Any
    new_model: torch.nn.Module
    new_index: ParamIndex
    base_classnames: tuple[str, ...]
    new_classnames: tuple[str, ...]


def build_dataset_runtime(
    source: R2Source,
    scientific_plan,
    *,
    runtime_root: Path,
) -> DatasetRuntime:
    base_runtime = _build_runtime(scientific_plan, Path(runtime_root) / source.key.dataset)
    new_cfg = build_scientific_cfg(
        dataset=source.key.dataset,
        seed=source.key.seed,
        data_root=scientific_plan.data_root,
        output_dir=Path(runtime_root) / source.key.dataset / "new",
        config_path=scientific_plan.config_path,
        class_subsample="new",
        shots=source.key.shots,
        epochs=scientific_plan.epochs,
    )
    set_random_seed(source.key.seed)
    new_trainer = build_coop_trainer(new_cfg, scientific_plan.clip_cache)
    new_model = unwrap_model(new_trainer.model)
    new_index = ParamIndex.from_model(new_model)
    base_names = tuple(str(value) for value in base_runtime.trainer.dm.dataset.classnames)
    new_names = tuple(str(value) for value in new_trainer.dm.dataset.classnames)
    if set(base_names) & set(new_names):
        raise RuntimeError("Base/New class vocabularies overlap")
    return DatasetRuntime(
        source=source,
        scientific_plan=scientific_plan,
        base_runtime=base_runtime,
        new_trainer=new_trainer,
        new_model=new_model,
        new_index=new_index,
        base_classnames=base_names,
        new_classnames=new_names,
    )


def text_features(model) -> torch.Tensor:
    return _text_features(model).detach().to(dtype=torch.float32).cpu()


def _cache_candidates(search_root: Path | None, key: FeatureCacheKey):
    if search_root is None or not Path(search_root).is_dir():
        return []
    # Validate the embedded key rather than trusting a filename convention;
    # this permits reuse of both LC01's digest name and LC05's dataset name.
    return sorted(Path(search_root).rglob("eval_*.pt"))


def build_or_reuse_eval_cache(
    runtime: DatasetRuntime,
    *,
    destination: Path,
    reusable_cache_root: Path | None,
) -> dict[str, Any]:
    """Return canonical Base+New test features, preferring a hashed LC cache."""

    config = runtime.source.source_config
    key = FeatureCacheKey(
        dataset=runtime.source.key.dataset,
        split="base_new_test",
        clip_sha256=str(config["model"]["checkpoint_sha256"]),
        transform_signature="canonical_evaluation_transform",
    )
    for candidate in _cache_candidates(reusable_cache_root, key):
        try:
            features, labels, sample_ids = load_feature_cache(candidate, expected_key=key)
        except Exception:
            continue
        base_count = len(runtime.base_classnames)
        if (
            len(features) == len(labels) == len(sample_ids)
            and bool((labels >= 0).all().item())
            and bool((labels < base_count + len(runtime.new_classnames)).all().item())
            and bool((labels < base_count).any().item())
            and bool((labels >= base_count).any().item())
        ):
            return {
                "features": features,
                "labels": labels,
                "sample_ids": sample_ids,
                "key": key,
                "reused": True,
                "source_path": str(candidate.resolve()),
                "image_encoder_forward_batches": 0,
            }

    base_features, base_labels, base_ids = _eval_feature_cache(
        runtime.base_runtime.trainer, runtime.base_runtime.model, label_offset=0
    )
    base_count = len(runtime.base_classnames)
    new_features, new_labels, new_ids = _eval_feature_cache(
        runtime.new_trainer, runtime.new_model, label_offset=base_count
    )
    features = torch.cat((base_features, new_features))
    labels = torch.cat((base_labels, new_labels))
    sample_ids = base_ids + new_ids
    destination = Path(destination)
    digest = save_feature_cache(
        destination,
        key=key,
        features=features,
        labels=labels,
        sample_ids=sample_ids,
    )
    return {
        "features": features,
        "labels": labels,
        "sample_ids": sample_ids,
        "key": key,
        "reused": False,
        "source_path": str(destination.resolve()),
        "content_sha256": digest,
        "image_encoder_forward_batches": (
            len(runtime.base_runtime.trainer.test_loader)
            + len(runtime.new_trainer.test_loader)
        ),
    }


def install_checkpoint(runtime: DatasetRuntime, source: R2Source) -> None:
    source.checkpoint.install_prompt(runtime.base_runtime.param_index)
    source.checkpoint.install_prompt(runtime.new_index)


def build_or_reuse_training_cache(
    runtime: DatasetRuntime,
    *,
    destination: Path,
) -> dict[str, Any]:
    """Materialize one seed-specific selected source and encode images once."""

    source = runtime.source
    plan = runtime.scientific_plan
    key = FeatureCacheKey(
        dataset=source.key.dataset,
        split="fixed_materialized_train",
        clip_sha256=str(source.source_config["model"]["checkpoint_sha256"]),
        transform_signature="pinned_coop_train_transform_per_sample_seed_v1",
        checkpoint_sha256=plan.source.fingerprint,
        replicate=0,
    )
    destination = Path(destination)
    if destination.is_file():
        features, labels, sample_ids = load_feature_cache(destination, expected_key=key)
        return {
            "features": features, "labels": labels, "sample_ids": sample_ids,
            "key": key, "reused": True, "source_path": str(destination.resolve()),
            "image_encoder_forward_batches": 0,
        }
    loader = runtime.base_runtime.full_gradient_loader
    if loader is None:
        cfg = runtime.base_runtime.trainer.cfg
        loader = build_full_gradient_loader(
            cfg,
            plan.source,
            micro_batch_size=plan.full_gradient_micro_batch_size,
            num_workers=0,
        )
    dataset = loader.dataset
    model = runtime.base_runtime.model
    device = runtime.base_runtime.param_index[0].parameter.device
    records = []
    pending = []
    model.eval()
    for index, record in enumerate(dataset.source):
        clock = materialization_seed_clock(plan.source.fingerprint, 0, record.sample_id)
        with isolated_rng(
            protocol_seed=source.key.seed,
            dataset=source.key.dataset,
            shots=source.key.shots,
            config_hash=plan.source.fingerprint,
            optimizer_step=clock,
            purpose=f"lc06_materialize_{record.sample_id}",
        ):
            item = dataset[index]
        pending.append((record, item))
        if len(pending) == 32 or index + 1 == len(dataset.source):
            images = torch.stack([entry[1]["img"] for entry in pending]).to(device)
            with torch.no_grad():
                values = model.image_encoder(images.type(model.dtype))
                values = values / values.norm(dim=-1, keepdim=True)
            for (selected, item), value in zip(pending, values):
                records.append(
                    (selected.sample_id, int(item["label"]), value.detach().float().cpu())
                )
            pending.clear()
    features = torch.stack([row[2] for row in records])
    labels = torch.tensor([row[1] for row in records], dtype=torch.long)
    sample_ids = tuple(row[0] for row in records)
    digest = save_feature_cache(
        destination, key=key, features=features, labels=labels, sample_ids=sample_ids
    )
    return {
        "features": features, "labels": labels, "sample_ids": sample_ids,
        "key": key, "reused": False, "source_path": str(destination.resolve()),
        "content_sha256": digest,
        "image_encoder_forward_batches": (len(records) + 31) // 32,
    }


def fixed_feature_loss_fn(
    model,
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
):
    device = next(model.prompt_learner.parameters()).device
    cached_features = features.to(device=device)
    cached_labels = labels.to(device=device, dtype=torch.long)

    def loss():
        text = _text_features(model)
        logits = model.logit_scale.exp() * cached_features.to(text) @ text.t()
        return F.cross_entropy(logits, cached_labels, reduction="mean")

    return loss


def write_cache_index(path: Path, cache: Mapping[str, Any], **metadata: Any) -> None:
    payload = {
        "schema_version": "sample_fg.low_compute_cache_index.v1",
        "key": cache["key"].__dict__,
        "key_sha256": cache["key"].digest,
        "reused": bool(cache["reused"]),
        "source_path": str(cache["source_path"]),
        "sample_count": len(cache["labels"]),
        **metadata,
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
