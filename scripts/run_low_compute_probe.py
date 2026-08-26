"""CLI for the integrated zero-step low-compute probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.low_compute.checkpoint_probe import load_probe_checkpoint
from sample_fg.low_compute.campaign_sources import discover_r2_sources
from sample_fg.low_compute.planner import build_integrated_plan
from sample_fg.low_compute.runner import build_source_scientific_plan, run_integrated_probe
from sample_fg.low_compute.semantic_runner import build_lc05_dry_run, run_lc05
from sample_fg.low_compute.sharpness_runner import build_lc06_dry_run, run_lc06
from sample_fg.low_compute.trajectory import build_trajectory_plan, run_trajectory_probe
from sample_fg.paper_runner import DEFAULT_CONFIG as DEFAULT_PAPER_CONFIG


DEFAULT_CONFIG = REPO_ROOT / "configs" / "sample_fg" / "low_compute_campaign.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only LC01/03/04/05/06 low-compute probes"
    )
    parser.add_argument(
        "--task", default="lc01_lc04",
        choices=("lc01_lc04", "lc03", "lc05", "lc06"),
    )
    parser.add_argument("--dataset", default="dtd", choices=("dtd",))
    parser.add_argument("--datasets", nargs="+", choices=("dtd", "eurosat"), default=("dtd", "eurosat"))
    parser.add_argument("--shots", default=16, type=int, choices=(16,))
    parser.add_argument("--seed", default=1, type=int, choices=(1,))
    parser.add_argument("--source-run")
    parser.add_argument("--runs-root", help="Paper-reproduction root discovered by LC05/LC06")
    parser.add_argument(
        "--lc01-run",
        help="Completed LC01+LC04 run whose identities/caches LC03 must consume",
    )
    parser.add_argument("--lc05-run", help="Completed LC05 run joined into LC06")
    parser.add_argument(
        "--evidence-root",
        help="Low-compute root searched for hash-matched LC01/LC03/LC04 evidence",
    )
    parser.add_argument("--checkpoint-policy", default="fractions", choices=("fractions",))
    parser.add_argument(
        "--checkpoint-fractions", nargs="+", type=float,
        default=(0.10, 0.30, 0.50, 0.70, 1.00),
    )
    parser.add_argument("--lambda-grid", nargs="+", type=float)
    parser.add_argument("--order-trials", type=int, default=512)
    parser.add_argument("--analysis-seed", type=int, default=10401)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paper-config", default=str(DEFAULT_PAPER_CONFIG))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--execute", action="store_true",
        help="Explicitly run frozen forward/backward probes; still performs zero steps",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.task in {"lc05", "lc06"}:
        if not args.runs_root:
            raise SystemExit(f"{args.task.upper()} requires --runs-root")
        discovery = discover_r2_sources(Path(args.runs_root), datasets=args.datasets)
        common = {
            "data_root": Path(args.data_root),
            "manifest_root": Path(args.manifest_root),
            "clip_cache": Path(args.clip_cache),
            "output_root": Path(args.output_root),
        }
        if args.task == "lc05":
            if args.dry_run:
                print(json.dumps(build_lc05_dry_run(
                    discovery, **common, reusable_cache_root=Path(args.output_root)
                ), indent=2, sort_keys=True))
                return 0
            run_dir = run_lc05(
                discovery, **common, reusable_cache_root=Path(args.output_root)
            )
        else:
            if args.dry_run:
                print(json.dumps(build_lc06_dry_run(
                    discovery, **common
                ), indent=2, sort_keys=True))
                return 0
            lc05_run = Path(args.lc05_run) if args.lc05_run else None
            run_dir = run_lc06(
                discovery, **common, lc05_run=lc05_run,
                evidence_root=(
                    Path(args.evidence_root) if args.evidence_root
                    else Path(args.output_root)
                ),
            )
        print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
        return 0
    if args.task == "lc03":
        if not args.source_run:
            raise SystemExit("LC03 requires --source-run")
        if not args.lc01_run:
            raise SystemExit("LC03 requires --lc01-run")
        plan = build_trajectory_plan(
            lc01_run=Path(args.lc01_run),
            source_run=Path(args.source_run),
            campaign_config=Path(args.config),
        )
        scientific = build_source_scientific_plan(
            source_run=Path(args.source_run),
            data_root=Path(args.data_root),
            manifest_root=Path(args.manifest_root),
            clip_cache=Path(args.clip_cache),
            paper_config=Path(args.paper_config),
            runtime_output_root=Path(args.output_root) / "_runtime",
        )
        if scientific.source.fingerprint != plan.source_fingerprint:
            raise SystemExit("LC03 reconstructed selected source differs from LC01")
        if args.dry_run:
            report = plan.as_dict()
            report["resolved_resources"] = {
                "dataset_root": str(scientific.data_root),
                "manifest": str(scientific.manifest_path),
                "gradient_source_fingerprint": scientific.source.fingerprint,
                "clip_checkpoint": str(scientific.clip_checkpoint),
                "source_config_sha256": plan.source_config_sha256,
            }
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        run_dir = run_trajectory_probe(
            plan, scientific_plan=scientific, output_root=Path(args.output_root)
        )
        print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
        return 0
    if not args.source_run:
        raise SystemExit("LC01+LC04 requires --source-run")
    expected_fractions = (0.10, 0.30, 0.50, 0.70, 1.00)
    if tuple(args.checkpoint_fractions) != expected_fractions:
        raise SystemExit("LC01 scientific checkpoint fractions are fixed by design")
    plan = build_integrated_plan(
        source_run=Path(args.source_run),
        config_path=Path(args.config),
        order_trials=args.order_trials,
        analysis_seed=args.analysis_seed,
    )
    if args.lambda_grid is not None and tuple(args.lambda_grid) != plan.lambda_grid:
        raise SystemExit("LC01 scientific lambda grid is fixed by design")
    scientific = build_source_scientific_plan(
        source_run=Path(args.source_run),
        data_root=Path(args.data_root),
        manifest_root=Path(args.manifest_root),
        clip_cache=Path(args.clip_cache),
        paper_config=Path(args.paper_config),
        runtime_output_root=Path(args.output_root) / "_runtime",
    )
    # Full source validation is part of preview, not deferred to execution.
    for checkpoint in plan.checkpoints:
        load_probe_checkpoint(Path(args.source_run), checkpoint.path)
    if args.dry_run:
        report = plan.as_dict()
        report["resolved_resources"] = {
            "dataset_root": str(scientific.data_root),
            "manifest": str(scientific.manifest_path),
            "gradient_source_fingerprint": scientific.source.fingerprint,
            "clip_checkpoint": str(scientific.clip_checkpoint),
            "source_config_sha256": scientific.resolved_config["config_sha256"],
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    run_dir = run_integrated_probe(
        plan,
        scientific_plan=scientific,
        output_root=Path(args.output_root),
        config_path=Path(args.config),
    )
    print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
