"""CLI for LC07 predeclared triage and professor-facing reporting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.plot_low_compute_campaign import plot_lc07
from sample_fg.low_compute.reporting import (
    load_low_compute_bundle,
    render_low_compute_findings,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc01-run", required=True)
    parser.add_argument("--lc03-run", required=True)
    parser.add_argument("--lc05-run", required=True)
    parser.add_argument("--lc06-run", required=True)
    parser.add_argument("--lc02-run")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    bundle = load_low_compute_bundle(
        lc01_run=Path(args.lc01_run), lc03_run=Path(args.lc03_run),
        lc05_run=Path(args.lc05_run), lc06_run=Path(args.lc06_run),
        lc02_run=None if args.lc02_run is None else Path(args.lc02_run),
    )
    output = Path(args.output_dir)
    report = render_low_compute_findings(bundle, output_dir=output)
    plots = plot_lc07(
        lc01_run=Path(args.lc01_run), lc05_run=Path(args.lc05_run),
        lc06_run=Path(args.lc06_run), output_dir=output / "plots",
    )
    print(json.dumps({
        "status": "COMPLETED", "report": str(report),
        "plots": [str(path) for path in plots],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
