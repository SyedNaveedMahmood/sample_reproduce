"""Regenerate LC03 tables and plots from saved scalar artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.low_compute.trajectory import render_trajectory_artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a saved LC03 trajectory")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    outputs = render_trajectory_artifacts(Path(args.run_dir))
    print(json.dumps({"status": "COMPLETED", "outputs": [str(path) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
