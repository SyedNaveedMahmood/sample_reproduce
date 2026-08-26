"""CLI for the read-only LC02 projected-component diagnostic correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.low_compute.lc02_audit import (
    build_lc02_audit_plan,
    run_lc02_diagnostic_audit,
)
from sample_fg.paper_runner import DEFAULT_CONFIG as DEFAULT_PAPER_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only audit of the completed LC02 g_B fidelity diagnostic"
    )
    parser.add_argument("--lc02-run", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--clip-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--paper-config", default=str(DEFAULT_PAPER_CONFIG))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "lc02_run": Path(args.lc02_run),
        "source_run": Path(args.source_run),
        "data_root": Path(args.data_root),
        "manifest_root": Path(args.manifest_root),
        "clip_cache": Path(args.clip_cache),
        "paper_config": Path(args.paper_config),
        "output_root": Path(args.output_root),
    }
    if args.dry_run:
        print(json.dumps(build_lc02_audit_plan(**common), indent=2, sort_keys=True))
        return 0
    run_dir = run_lc02_diagnostic_audit(**common)
    print(json.dumps({"status": "COMPLETED", "run_dir": str(run_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
