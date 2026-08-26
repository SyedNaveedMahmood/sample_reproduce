"""Task-3 helpers for a tightly wrapped, ordinary CoOp smoke path."""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.nn import functional as F

from .scheduler_compat import build_lr_scheduler_compat


EXPECTED_CLIP_KEY = "ViT-B/16"
EXPECTED_CLIP_SHA256 = (
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
)
EXPECTED_TRAINABLE_NAME = "prompt_learner.ctx"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_smoke_cfg(
    coop_root: Path,
    data_root: Path,
    output_dir: Path,
    class_subsample: str,
):
    """Resolve config through the pinned CoOp/Dassl YACS definitions."""

    from dassl.config import get_cfg_default
    from train import extend_cfg

    if class_subsample not in {"base", "new"}:
        raise ValueError(f"Unexpected class subsample: {class_subsample}")

    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.merge_from_file(str(coop_root / "configs" / "datasets" / "dtd.yaml"))
    cfg.merge_from_file(
        str(coop_root / "configs" / "trainers" / "CoOp" / "vit_b16_ctxv1.yaml")
    )
    cfg.merge_from_file(
        str(coop_root / "configs" / "sample_fg" / "smoke_task3_coop_anchor.yaml")
    )
    cfg.DATASET.ROOT = str(data_root)
    cfg.DATASET.SUBSAMPLE_CLASSES = class_subsample
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.SEED = 1
    cfg.TRAINER.NAME = "CoOp"
    cfg.freeze()
    return cfg


@contextmanager
def scoped_coop_construction_compat(
    checkpoint_cache_dir: Path,
) -> Iterator[None]:
    """Scope local cache routing and scheduler adaptation to model build."""

    import trainers.coop as upstream_coop

    original_scheduler_builder = upstream_coop.build_lr_scheduler
    original_download = upstream_coop.clip._download
    expected_url = upstream_coop.clip._MODELS[EXPECTED_CLIP_KEY]

    def download_from_pinned_source(url: str):
        if url != expected_url:
            raise RuntimeError(f"Unexpected CLIP checkpoint URL: {url}")
        return original_download(url, root=str(checkpoint_cache_dir))

    upstream_coop.build_lr_scheduler = build_lr_scheduler_compat
    upstream_coop.clip._download = download_from_pinned_source
    try:
        yield
    finally:
        upstream_coop.clip._download = original_download
        upstream_coop.build_lr_scheduler = original_scheduler_builder


def build_coop_trainer(cfg, checkpoint_cache_dir: Path):
    """Use the real registry and real CoOp class with scoped construction fixes."""

    from dassl.engine import build_trainer

    with scoped_coop_construction_compat(checkpoint_cache_dir):
        trainer = build_trainer(cfg)
    if trainer.__class__.__name__ != "CoOp":
        raise AssertionError(f"Expected registered CoOp, got {trainer.__class__.__name__}")
    return trainer


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def audit_prompt_only_training(model, optimizer) -> dict[str, Any]:
    """Direct Task-3 audit; Task 4 will introduce the formal ParamIndex."""

    model = unwrap_model(model)
    named = list(model.named_parameters())
    trainable = [(name, param) for name, param in named if param.requires_grad]
    frozen = [(name, param) for name, param in named if not param.requires_grad]
    trainable_names = [name for name, _ in trainable]
    if trainable_names != [EXPECTED_TRAINABLE_NAME]:
        raise AssertionError(f"Unexpected trainable parameters: {trainable_names}")

    optimizer_ids = {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }
    if optimizer_ids != {id(trainable[0][1])}:
        raise AssertionError("Optimizer parameters differ from the prompt context tensor")

    def frozen_group(prefix: str) -> list[str]:
        names = [name for name, param in named if name.startswith(prefix)]
        if not names:
            raise AssertionError(f"Missing expected frozen component: {prefix}")
        if any(dict(named)[name].requires_grad for name in names):
            raise AssertionError(f"Component is not fully frozen: {prefix}")
        return names

    image_names = frozen_group("image_encoder.")
    transformer_names = frozen_group("text_encoder.transformer.")
    required_exact = [
        "text_encoder.positional_embedding",
        "text_encoder.text_projection",
        "text_encoder.ln_final.weight",
        "text_encoder.ln_final.bias",
        "logit_scale",
    ]
    named_map = dict(named)
    for name in required_exact:
        if name not in named_map or named_map[name].requires_grad:
            raise AssertionError(f"Expected retained frozen parameter: {name}")
    if any("token_embedding" in name for name, _ in named):
        raise AssertionError("Pinned CustomCLIP unexpectedly retained token_embedding")

    visual_projection_names = [
        name for name in image_names if name == "image_encoder.proj" or name.endswith(".proj")
    ]
    if not visual_projection_names:
        raise AssertionError("No retained frozen visual projection was found")

    return {
        "trainable_parameters": [
            {
                "name": name,
                "shape": list(param.shape),
                "numel": param.numel(),
                "dtype": str(param.dtype),
            }
            for name, param in trainable
        ],
        "trainable_numel": sum(param.numel() for _, param in trainable),
        "frozen_numel": sum(param.numel() for _, param in frozen),
        "frozen_parameter_count": len(frozen),
        "component_audit": {
            "image_encoder": {"parameter_count": len(image_names), "status": "frozen"},
            "text_transformer": {
                "parameter_count": len(transformer_names),
                "status": "frozen",
            },
            "token_embedding": {
                "status": "not_retained_after_prompt_initialization",
                "optimizer_reachable": False,
            },
            "positional_embedding": "frozen",
            "text_projection": "frozen",
            "visual_projection": {
                "status": "frozen",
                "names": visual_projection_names,
            },
            "logit_scale": "frozen",
        },
    }


def hash_frozen_parameters(model) -> str:
    """Strongly checksum every retained frozen tensor without keeping a clone."""

    model = unwrap_model(model)
    digest = hashlib.sha256()
    frozen_count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        frozen_count += 1
        tensor = param.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    if frozen_count == 0:
        raise AssertionError("No frozen parameters were available to checksum")
    return digest.hexdigest().upper()


@contextmanager
def count_optimizer_steps(optimizer) -> Iterator[dict[str, int]]:
    """Count calls while delegating each call to the real optimizer method."""

    original_step = optimizer.step
    counter = {"count": 0}

    def counted_step(*args, **kwargs):
        counter["count"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    try:
        yield counter
    finally:
        optimizer.step = original_step


def run_bounded_coop_steps(trainer, max_optimizer_steps: int) -> list[dict[str, Any]]:
    """Call pinned ``CoOp.forward_backward`` exactly the requested number of times."""

    if max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    trainer.set_model_mode("train")
    trainer.epoch = 0
    trainer.num_batches = len(trainer.train_loader_x)
    model = unwrap_model(trainer.model)
    ctx = model.prompt_learner.ctx
    records: list[dict[str, Any]] = []

    with count_optimizer_steps(trainer.optim) as counter:
        for batch_idx, batch in enumerate(trainer.train_loader_x):
            if counter["count"] >= max_optimizer_steps:
                break
            trainer.batch_idx = batch_idx
            before = ctx.detach().clone()
            lr_before = float(trainer.get_current_lr())
            loss_summary = trainer.forward_backward(batch)
            lr_after = float(trainer.get_current_lr())
            grad = ctx.grad
            if grad is None:
                raise AssertionError("Prompt context gradient is missing")
            grad_fp32 = grad.detach().float()
            finite_loss = math.isfinite(float(loss_summary["loss"]))
            finite_grad = bool(torch.isfinite(grad_fp32).all().item())
            grad_norm = float(torch.linalg.vector_norm(grad_fp32).item())
            delta_norm = float(
                torch.linalg.vector_norm((ctx.detach() - before).float()).item()
            )
            frozen_with_grad = [
                name
                for name, param in model.named_parameters()
                if not param.requires_grad and param.grad is not None
            ]
            if not finite_loss or not finite_grad or grad_norm <= 0:
                raise FloatingPointError(
                    f"Non-finite/zero CoOp step: loss={loss_summary['loss']} grad={grad_norm}"
                )
            if delta_norm <= 0:
                raise AssertionError("Prompt context did not change on an optimizer step")
            if frozen_with_grad:
                raise AssertionError(f"Frozen parameters received gradients: {frozen_with_grad}")
            records.append(
                {
                    "optimizer_step": counter["count"],
                    "epoch_zero_based": 0,
                    "epoch_one_based": 1,
                    "batch_index_zero_based": batch_idx,
                    "batch_position_one_based": batch_idx + 1,
                    "num_batches_in_epoch": trainer.num_batches,
                    "loss": float(loss_summary["loss"]),
                    "batch_accuracy_pct": float(loss_summary["acc"]),
                    "prompt_gradient_norm": grad_norm,
                    "prompt_step_delta_norm": delta_norm,
                    "lr_before": lr_before,
                    "lr_after": lr_after,
                    "loss_finite": finite_loss,
                    "gradient_finite": finite_grad,
                    "frozen_parameters_with_grad": 0,
                }
            )

    if counter["count"] != max_optimizer_steps or len(records) != max_optimizer_steps:
        raise AssertionError(
            f"Expected {max_optimizer_steps} optimizer steps, got {counter['count']}"
        )
    return records


@torch.no_grad()
def evaluate_bounded(trainer, max_samples: int, expected_classes: int) -> dict[str, Any]:
    """Exercise the pinned test loader/model path on a bounded sample count."""

    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    trainer.set_model_mode("eval")
    total = 0
    correct = 0
    loss_sum = 0.0
    output_dim = None

    with count_optimizer_steps(trainer.optim) as counter:
        for batch in trainer.test_loader:
            image, label = trainer.parse_batch_test(batch)
            remaining = max_samples - total
            image = image[:remaining]
            label = label[:remaining]
            logits = trainer.model_inference(image)
            output_dim = int(logits.shape[1])
            if output_dim != expected_classes:
                raise AssertionError(
                    f"Expected {expected_classes} logits, observed {output_dim}"
                )
            loss = F.cross_entropy(logits, label)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError("Non-finite bounded evaluation output")
            batch_count = int(label.shape[0])
            total += batch_count
            loss_sum += float(loss.item()) * batch_count
            correct += int(logits.argmax(dim=1).eq(label).sum().item())
            if total >= max_samples:
                break

    if total != max_samples:
        raise AssertionError(f"Expected {max_samples} evaluation samples, got {total}")
    if counter["count"] != 0:
        raise AssertionError("Evaluation performed an optimizer step")
    return {
        "sample_count": total,
        "class_count": expected_classes,
        "output_dimension": output_dim,
        "mean_loss": loss_sum / total,
        "accuracy_pct": 100.0 * correct / total,
        "finite": True,
        "optimizer_steps": 0,
    }
