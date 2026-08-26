"""Manifest-gated scientific runner for SAMPLe EMA/Exact/Periodic campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .campaign import (
    CampaignError,
    CampaignManifest,
    PeriodicKFreeze,
    load_periodic_k_freeze,
)
from .paper_runner import (
    FULL_GRADIENT_MICRO_BATCH_SIZE,
    MethodSelection,
    REPO_ROOT,
    ScientificPlan,
    ScientificRunnerError,
    build_scientific_plan,
    dry_run_report,
    run_scientific,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "sample_fg" / "extension.yaml"
DEFAULT_CAMPAIGN = (
    REPO_ROOT / "configs" / "sample_fg" / "extension_campaign.yaml"
)
EXTENSION_TASKS = ("task26", "task27", "task28")
EXTENSION_ESTIMATORS = ("ema", "exact", "periodic")


def resolve_extension_method(
    method: str,
    estimator: str,
    periodic_k_steps: int | None,
    *,
    allowed_k: tuple[int, ...],
) -> MethodSelection:
    """Validate the only estimator combinations authorized by E0/E1/E2."""

    if method != "sample":
        raise ScientificRunnerError(
            "The extension runner accepts --method sample only"
        )
    if estimator not in EXTENSION_ESTIMATORS:
        raise ScientificRunnerError(
            f"Extension estimator must be one of {EXTENSION_ESTIMATORS}"
        )
    if estimator == "periodic":
        if (
            isinstance(periodic_k_steps, bool)
            or not isinstance(periodic_k_steps, int)
            or periodic_k_steps < 1
        ):
            raise ScientificRunnerError("Periodic SAMPLe requires --periodic-k")
        if periodic_k_steps not in allowed_k:
            raise ScientificRunnerError(
                f"Periodic K must be one of the predeclared values {allowed_k}"
            )
        estimator_tag = f"periodic-k{periodic_k_steps}"
    else:
        if periodic_k_steps is not None:
            raise ScientificRunnerError(
                f"SAMPLe-{estimator} must not receive --periodic-k"
            )
        estimator_tag = estimator
    return MethodSelection(
        method="sample",
        estimator=estimator,
        method_tag="sample",
        estimator_tag=estimator_tag,
        rho=0.05,
        alpha=0.0015,
        ema_lambda=0.15 if estimator in {"ema", "periodic"} else None,
        refresh_k_steps=periodic_k_steps,
    )


def _freeze_for_args(
    args: argparse.Namespace,
    campaign: CampaignManifest,
) -> PeriodicKFreeze | None:
    if args.task == "task28":
        if not args.periodic_k_freeze:
            raise ScientificRunnerError(
                "Task 28 is gated by --periodic-k-freeze for every cell"
            )
        return load_periodic_k_freeze(
            Path(args.periodic_k_freeze), campaign=campaign
        )
    if args.periodic_k_freeze:
        raise ScientificRunnerError(
            "A periodic-K freeze is accepted only for Task 28"
        )
    return None


def build_extension_plan(args: argparse.Namespace) -> ScientificPlan:
    try:
        campaign = CampaignManifest.load(Path(args.campaign_config))
        freeze = _freeze_for_args(args, campaign)
        selection = resolve_extension_method(
            args.method,
            args.estimator,
            args.periodic_k,
            allowed_k=campaign.allowed_k,
        )
        frozen_values = freeze.selected_k_values if freeze is not None else None
        cell = campaign.validate_cell(
            task=args.task,
            dataset=args.dataset,
            shots=args.shots,
            seed=args.seed,
            method=args.method,
            estimator=args.estimator,
            periodic_k_steps=args.periodic_k,
            frozen_k_values=frozen_values,
        )
    except CampaignError as error:
        raise ScientificRunnerError(str(error)) from error

    diagnostic_interval = args.diagnostic_interval_steps
    if diagnostic_interval is not None:
        if diagnostic_interval != cell.steps_per_epoch:
            raise ScientificRunnerError(
                "Declared campaigns require once-per-normal-epoch diagnostics; "
                f"expected {cell.steps_per_epoch} steps"
            )
    if args.epochs != cell.epochs:
        raise ScientificRunnerError(
            f"Campaign fixes --epochs at {cell.epochs}, observed {args.epochs}"
        )
    if args.full_gradient_micro_batch_size != cell.full_gradient_micro_batch_size:
        raise ScientificRunnerError(
            "Full-gradient micro-batch size differs from the campaign contract"
        )

    campaign_metadata = {
        "schema_version": "sample_fg.run_campaign_binding.v1",
        "task": cell.task,
        "task_title": cell.title,
        "campaign_config": str(campaign.path),
        "campaign_config_sha256": campaign.sha256,
        "periodic_k_freeze": freeze.as_dict() if freeze is not None else None,
        "reuse_experiment_id": cell.reuse_experiment_id,
        "reproduction_gap_context": (
            "CoOp reproduced closely; audited public-spec SAM/SAMPLe did not "
            "numerically reproduce the published DTD novel-class gains; no "
            "unsupported tuning was performed."
        ),
    }
    plan = build_scientific_plan(
        dataset=cell.dataset,
        shots=cell.shots,
        seed=cell.seed,
        experiment_id=cell.experiment_id,
        selection=selection,
        data_root=Path(args.data_root),
        manifest_root=Path(args.manifest_root),
        clip_cache=Path(args.clip_cache),
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        recovery_interval_epochs=args.recovery_interval_epochs,
        epochs=cell.epochs,
        diagnostic_interval_steps=diagnostic_interval,
        full_gradient_micro_batch_size=cell.full_gradient_micro_batch_size,
        resume_from=Path(args.resume_from) if args.resume_from else None,
        notes=f"{cell.task}: {cell.title}",
        campaign_metadata=campaign_metadata,
    )
    if plan.steps_per_epoch != cell.steps_per_epoch:
        raise ScientificRunnerError(
            "Task-2 manifest steps_per_epoch differs from the campaign protocol"
        )
    if len(plan.source) != cell.selected_count:
        raise ScientificRunnerError(
            "Task-2 selected source count differs from the campaign protocol"
        )
    normal_samples = int(
        plan.manifest["normal_train_loader"]["samples_consumed_per_epoch"]
    )
    if normal_samples != cell.samples_consumed_per_epoch:
        raise ScientificRunnerError(
            "Task-2 normal samples per epoch differs from the campaign protocol"
        )
    return plan


def extension_dry_run_report(plan: ScientificPlan) -> dict[str, object]:
    report = dry_run_report(plan)
    report["campaign"] = plan.resolved_config["campaign"]
    report["cell"]["canonical_relative_root"] = str(
        Path(plan.dataset)
        / f"shots_{plan.shots}"
        / plan.selection.method_tag
        / plan.selection.estimator_tag
        / f"seed_{plan.seed}"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one manifest-approved SAMPLe estimator-extension cell"
    )
    parser.add_argument("--task", required=True, choices=EXTENSION_TASKS)
    parser.add_argument("--dataset", required=True, choices=("dtd", "eurosat"))
    parser.add_argument("--shots", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--periodic-k", type=int)
    parser.add_argument("--periodic-k-freeze")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--campaign-config", default=str(DEFAULT_CAMPAIGN))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument(
        "--diagnostic-interval-steps",
        type=int,
        help="Must equal one normal epoch for declared campaigns",
    )
    parser.add_argument(
        "--full-gradient-micro-batch-size",
        type=int,
        default=FULL_GRADIENT_MICRO_BATCH_SIZE,
    )
    parser.add_argument("--recovery-interval-epochs", type=int, default=10)
    parser.add_argument("--resume-from")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_extension_plan(args)
    if args.dry_run:
        print(json.dumps(extension_dry_run_report(plan), indent=2, sort_keys=True))
        return 0
    reuse_experiment_id = plan.resolved_config["campaign"].get(
        "reuse_experiment_id"
    )
    if reuse_experiment_id is not None:
        raise ScientificRunnerError(
            "This declared cell must reuse its completed "
            f"{reuse_experiment_id} artifact; direct retraining is prohibited"
        )
    run_dir = run_scientific(plan)
    print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
    return 0
