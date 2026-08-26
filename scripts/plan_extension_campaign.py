"""Zero-training structural dry-run for the declared Task 25--28 matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.campaign import CampaignManifest, TASK_KEYS


def build_report(args: argparse.Namespace) -> dict[str, object]:
    campaign = CampaignManifest.load(Path(args.campaign_config))
    repo_root = Path(args.repo_root).resolve(strict=True)
    project_root = Path(args.project_root).resolve()
    resources = {
        "repo_root": repo_root,
        "data_root": project_root / "data",
        "manifest_root": project_root / "provenance" / "task2_data_manifests",
        "clip_cache": project_root / "implementation" / ".cache" / "clip",
        "r2_output_root": project_root / "runs" / "paper_reproduction",
        "extension_output_root": project_root / "runs" / "estimator_extension",
    }
    cells: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for task in args.tasks:
        frozen = (
            (args.hypothetical_frozen_k,)
            if task == "task28" and args.hypothetical_frozen_k is not None
            else None
        )
        expanded = campaign.cells(
            task,
            frozen_k_values=frozen,
            allow_unfrozen=task == "task28" and frozen is None,
        )
        counts[task] = len(expanded)
        for cell in expanded:
            item = cell.as_dict()
            item["output_root"] = str(
                resources[
                    "r2_output_root"
                    if task == "task25"
                    else "extension_output_root"
                ]
            )
            item["optimizer"] = campaign.protocol.get("optimizer")
            item["scheduler"] = campaign.protocol.get("scheduler")
            item["precision"] = campaign.protocol.get("precision")
            item["full_gradient_source"] = str(
                resources["manifest_root"]
                / cell.dataset
                / f"shots_{cell.shots}"
                / f"seed_{cell.seed}"
                / "data_manifest.json"
            )
            item["optimizer_steps_executed"] = 0
            cells.append(item)
    return {
        "schema_version": "sample_fg.campaign_dry_run.v1",
        "status": "STRUCTURAL_DRY_RUN_VALIDATED",
        "dry_run": True,
        "training_started": False,
        "optimizer_steps_executed": 0,
        "campaign_config": str(campaign.path),
        "campaign_config_sha256": campaign.sha256,
        "tasks": list(args.tasks),
        "cell_counts": counts,
        "total_cells": len(cells),
        "reuse_existing_artifact_cells": sum(
            item["execution_mode"] == "reuse_existing_artifact" for item in cells
        ),
        "new_scientific_run_cells": sum(
            item["execution_mode"] == "new_scientific_run" for item in cells
        ),
        "hypothetical_frozen_k": args.hypothetical_frozen_k,
        "hypothetical_freeze_is_execution_authority": False,
        "resources": {
            key: {"path": str(path), "exists": path.exists()}
            for key, path in resources.items()
        },
        "cells": cells,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--project-root", required=True)
    parser.add_argument(
        "--campaign-config",
        default=str(REPO_ROOT / "configs" / "sample_fg" / "extension_campaign.yaml"),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASK_KEYS,
        default=list(TASK_KEYS),
    )
    parser.add_argument(
        "--hypothetical-frozen-k",
        type=int,
        help=(
            "Dry-run-only Task-28 wiring value; does not create a freeze record "
            "or authorize execution"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = build_report(parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
