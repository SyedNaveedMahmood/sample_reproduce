"""Validated command line entry point for the single LC02 causal branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.low_compute.fork_runner import (
    build_low_compute_fork_plan,
    run_low_compute_fork,
)
from sample_fg.low_compute.runner import build_source_scientific_plan
from sample_fg.paper_runner import DEFAULT_CONFIG as DEFAULT_PAPER_CONFIG


DEFAULT_CONFIG = REPO_ROOT / "configs" / "sample_fg" / "low_compute_campaign.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LC02 coverage-aware SAMPLe state-preserving causal branch"
    )
    parser.add_argument("--task", default="lc02", choices=("lc02",))
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--lc01-summary", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--paper-config", default=str(DEFAULT_PAPER_CONFIG))
    parser.add_argument("--resume-from")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the registered 240-step LC02 branch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_plan = build_source_scientific_plan(
        source_run=Path(args.source_run),
        data_root=Path(args.data_root),
        manifest_root=Path(args.manifest_root),
        clip_cache=Path(args.clip_cache),
        paper_config=Path(args.paper_config),
        runtime_output_root=Path(args.output_root) / "_runtime",
    )
    plan = build_low_compute_fork_plan(
        source_run=Path(args.source_run),
        source_checkpoint=Path(args.source_checkpoint),
        lc01_summary=Path(args.lc01_summary),
        campaign_config=Path(args.config),
        source_plan=source_plan,
        output_root=Path(args.output_root),
        resume_from=Path(args.resume_from) if args.resume_from else None,
    )
    if args.dry_run:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return 0
    run_dir = run_low_compute_fork(plan)
    print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
