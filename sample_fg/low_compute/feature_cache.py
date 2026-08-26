"""Hashed feature caches and order-invariant materialization identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch


FEATURE_CACHE_SCHEMA_VERSION = "sample_fg.low_compute_feature_cache.v1"
MATERIALIZATION_SCHEMA_VERSION = "sample_fg.low_compute_materialization.v1"


class FeatureCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class FeatureCacheKey:
    dataset: str
    split: str
    clip_sha256: str
    transform_signature: str
    checkpoint_sha256: str | None = None
    replicate: int | None = None

    @property
    def digest(self) -> str:
        payload = {"schema_version": FEATURE_CACHE_SCHEMA_VERSION, **asdict(self)}
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def materialization_seed_clock(
    checkpoint_sha256: str, replicate: int, sample_id: str
) -> int:
    """Map normative per-sample identity to a stable nonnegative RNG clock."""

    if replicate < 0 or not sample_id:
        raise FeatureCacheError("Invalid materialization identity")
    payload = {
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "replicate": replicate,
        "sample_id": sample_id,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**63 - 1)


def _tensor_hash(features: torch.Tensor, labels: torch.Tensor, sample_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for tensor in (features, labels):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    digest.update(json.dumps(list(sample_ids), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def save_feature_cache(
    path: Path,
    *,
    key: FeatureCacheKey,
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: Sequence[str],
) -> str:
    features = features.detach().to(dtype=torch.float32).cpu().contiguous()
    labels = labels.detach().to(dtype=torch.long).cpu().contiguous()
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise FeatureCacheError("Feature cache tensors have incompatible shapes")
    if len(sample_ids) != len(labels) or len(set(sample_ids)) != len(sample_ids):
        raise FeatureCacheError("Feature cache sample IDs are incomplete or duplicated")
    content_hash = _tensor_hash(features, labels, sample_ids)
    payload = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "key": asdict(key),
        "key_sha256": key.digest,
        "content_sha256": content_hash,
        "features": features,
        "labels": labels,
        "sample_ids": tuple(sample_ids),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return content_hash


def load_feature_cache(
    path: Path, *, expected_key: FeatureCacheKey
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    try:
        payload = torch.load(Path(path).resolve(strict=True), map_location="cpu", weights_only=False)
    except Exception as error:
        raise FeatureCacheError(f"Cannot load feature cache: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise FeatureCacheError("Unsupported feature-cache schema")
    if payload.get("key_sha256") != expected_key.digest or payload.get("key") != asdict(expected_key):
        raise FeatureCacheError("Feature-cache key differs")
    features = payload.get("features")
    labels = payload.get("labels")
    sample_ids = payload.get("sample_ids")
    if not isinstance(features, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise FeatureCacheError("Feature-cache tensors are missing")
    if not isinstance(sample_ids, (tuple, list)) or any(not isinstance(item, str) for item in sample_ids):
        raise FeatureCacheError("Feature-cache sample identities are malformed")
    observed = _tensor_hash(features, labels, sample_ids)
    if observed != payload.get("content_sha256"):
        raise FeatureCacheError("Feature-cache content hash mismatch")
    return features.clone(), labels.clone(), tuple(sample_ids)
