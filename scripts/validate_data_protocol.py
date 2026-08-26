#!/usr/bin/env python
"""Validate fixed CoOp data provenance without loading CLIP or training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.data_protocol import DATASET_SPECS, validate_matrix, write_deterministic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="CoOp $DATA root")
    parser.add_argument(
        "--datasets", nargs="+", choices=sorted(DATASET_SPECS), required=True
    )
    parser.add_argument("--shots", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scientific",
        action="store_true",
        help="Required: enforce fixed split and fail instead of generating one",
    )
    args = parser.parse_args()
    if not args.scientific:
        parser.error("This validator only runs in --scientific fail-fast mode")
    if any(value < 1 for value in args.shots):
        parser.error("shots must be positive")
    if any(value < 0 for value in args.seeds):
        parser.error("seeds must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    summary = validate_matrix(
        data_root=args.root,
        dataset_keys=args.datasets,
        shots_values=args.shots,
        seeds=args.seeds,
        output_root=args.output_dir,
        coop_root=REPO_ROOT,
    )
    summary_path = args.output_dir / "validation_summary.json"
    write_deterministic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
