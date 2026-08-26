"""Deterministic LC05/LC06/LC07 plots from saved scalar artifacts only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample_fg.results import atomic_write_json


PLOT_SCHEMA = "sample_fg.low_compute_campaign_plots.v1"
matplotlib.rcParams["svg.hashsalt"] = "sample_fg_lc05_lc07_v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact root is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(fig, root: Path, stem: str) -> list[str]:
    root.mkdir(parents=True, exist_ok=True)
    png = root / f"{stem}.png"; svg = root / f"{stem}.svg"
    fig.savefig(png, dpi=160, metadata={"Software": "sample_fg"})
    fig.savefig(svg, metadata={"Creator": "sample_fg", "Date": None})
    plt.close(fig)
    return [png.name, svg.name]


def _identity(row):
    return f"{row['dataset']}:{row['method_key']}:s{row['seed']}"


def plot_lc05(run_dir: Path) -> tuple[Path, ...]:
    root = Path(run_dir).resolve(strict=True)
    rows = _jsonl(root / "semantic_drift.jsonl")
    zero = _load(root / "zero_shot_reference.json")["rows"]
    if not rows:
        raise ValueError("LC05 has no semantic rows")
    output = root / "plots"; files = []
    labels = [_identity(row) for row in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.1), 4.2), constrained_layout=True)
    ax.bar(labels, [row["semantic_drift"]["all"]["mean_cosine_drift"] for row in rows])
    ax.set(title="LC05 semantic drift", ylabel="mean 1-cosine"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    files += _save(fig, output, "semantic_drift_by_method_dataset")

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.1), 4.2), constrained_layout=True)
    ax.bar(labels, [row["topology_distortion"]["all_off_diagonal"] for row in rows])
    ax.set(title="LC05 off-diagonal topology distortion", ylabel="relative Frobenius distortion"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    files += _save(fig, output, "topology_distortion_by_method_dataset")

    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    for row in rows:
        ax.scatter(row["semantic_drift"]["all"]["mean_cosine_drift"], row["standard_evaluation"]["new_accuracy_pct"], label=_identity(row))
    ax.set(title="Semantic drift vs standard New", xlabel="mean all-class semantic drift", ylabel="New accuracy (%)"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    files += _save(fig, output, "semantic_drift_vs_new_accuracy")

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    x = range(len(rows)); width = .12
    metric_names = (("base_accuracy_pct", "open_world_base_accuracy_pct", "Base"), ("new_accuracy_pct", "open_world_new_accuracy_pct", "New"), ("hm_pct", "open_world_hm_pct", "HM"))
    for index, (standard, opened, name) in enumerate(metric_names):
        ax.bar([value + (2*index)*width for value in x], [row["standard_evaluation"][standard] for row in rows], width, label=f"standard {name}")
        ax.bar([value + (2*index+1)*width for value in x], [row["open_world"][opened] for row in rows], width, label=f"open {name}")
    ax.set_xticks([value + 2.5*width for value in x], labels); ax.tick_params(axis="x", rotation=25)
    ax.set(title="Standard split vs all-class evaluation", ylabel="accuracy (%)"); ax.legend(fontsize=7, ncol=3); ax.grid(axis="y", alpha=.25)
    files += _save(fig, output, "standard_vs_open_world")

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 1.2), 4.2), constrained_layout=True)
    x = list(range(len(rows)))
    ax.bar([v-.18 for v in x], [row["open_world"]["base_to_new_group_confusion_pct"] for row in rows], .36, label="Base to New")
    ax.bar([v+.18 for v in x], [row["open_world"]["new_to_base_group_confusion_pct"] for row in rows], .36, label="New to Base")
    ax.set_xticks(x, labels); ax.tick_params(axis="x", rotation=25); ax.set(title="Cross-group confusion", ylabel="samples (%)"); ax.legend(); ax.grid(axis="y", alpha=.25)
    files += _save(fig, output, "base_new_group_confusion")

    fig, ax = plt.subplots(figsize=(6.2, 4.3), constrained_layout=True)
    for row in zero:
        values = [row["standard_evaluation"]["base_accuracy_pct"], row["standard_evaluation"]["new_accuracy_pct"], row["open_world"]["open_world_base_accuracy_pct"], row["open_world"]["open_world_new_accuracy_pct"]]
        ax.plot(["standard Base", "standard New", "open Base", "open New"], values, marker="o", label=f"{row['dataset']}:zero-shot")
    ax.set(title="Zero-shot CLIP reference points", ylabel="accuracy (%)"); ax.legend(); ax.grid(alpha=.25)
    files += _save(fig, output, "zero_shot_clip_reference")
    sources = [root / "semantic_drift.jsonl", root / "open_world_eval.jsonl", root / "source_hashes.json"]
    atomic_write_json(output / "plot_manifest.json", {
        "schema_version": PLOT_SCHEMA, "task": "LC05", "plots": files,
        "source_artifact_sha256": {str(path): _sha(path) for path in sources},
    })
    return tuple(output / name for name in files)


def plot_lc06(run_dir: Path) -> tuple[Path, ...]:
    root = Path(run_dir).resolve(strict=True)
    rows = _load(root / "sharpness_summary.json")["rows"]
    structured = _jsonl(root / "structured_direction_probe.jsonl")
    if not rows:
        raise ValueError("LC06 has no sharpness rows")
    output = root / "plots"; files = []
    identities = sorted({_identity(row) for row in rows})
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for identity in identities:
        selected = sorted((row for row in rows if _identity(row) == identity), key=lambda row: row["radius"])
        ax.plot([row["radius"] for row in selected], [row["sharpness_mean"] for row in selected], marker="o", label=identity)
    ax.set(title="Fixed-materialization sampled sharpness", xlabel="global prompt radius", ylabel="mean symmetric loss increase"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    files += _save(fig, output, "sharpness_by_radius_method")

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    for row in rows:
        if row["radius"] == 0.05:
            ax.scatter(row["sharpness_mean"], row["source_evaluation"]["new_accuracy_pct"], label=_identity(row))
    ax.set(title="Prompt sharpness vs standard New", xlabel="mean sharpness at rho", ylabel="New accuracy (%)"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    files += _save(fig, output, "sharpness_vs_new_accuracy")

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    for direction in sorted({row["direction"] for row in structured}):
        selected = sorted((row for row in structured if row["direction"] == direction), key=lambda row: row["radius"])
        ax.plot([row["radius"] for row in selected], [row.get("sharpness", 0.0) for row in selected], marker="o", label=direction)
    ax.set(title="Optimizer-relevant prompt directions", xlabel="global prompt radius", ylabel="symmetric loss increase"); ax.legend(); ax.grid(alpha=.25)
    files += _save(fig, output, "structured_direction_sharpness")
    sources = [root / "sharpness_summary.json", root / "structured_direction_probe.jsonl", root / "source_hashes.json"]
    atomic_write_json(output / "plot_manifest.json", {
        "schema_version": PLOT_SCHEMA, "task": "LC06", "plots": files,
        "source_artifact_sha256": {str(path): _sha(path) for path in sources},
    })
    return tuple(output / name for name in files)


def plot_lc07(*, lc01_run: Path, lc05_run: Path, lc06_run: Path, output_dir: Path) -> tuple[Path, ...]:
    lc01 = Path(lc01_run).resolve(strict=True); lc05 = Path(lc05_run).resolve(strict=True); lc06 = Path(lc06_run).resolve(strict=True)
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    files = []
    primary = _load(lc01 / "summary.json")["primary_findings"]["actual_checkpoint_ema_vs_materialized_exact"]
    primary = sorted((row for row in primary if row["materialization_replicate"] == 0), key=lambda row: row["epoch"])
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    ax.plot([row["epoch"] for row in primary], [row["cosine"] for row in primary], marker="o", label="actual checkpoint EMA")
    replay = _load(lc01 / "lc01" / "replay_summary.json")["rows"]
    for decay, label in ((0.15, "paper replay"), (0.8461538461538461, "coverage replay")):
        selected = sorted((row for row in replay if row["materialization_replicate"] == 0 and row["lambda"] == decay), key=lambda row: row["epoch"])
        ax.plot([row["epoch"] for row in selected], [row["canonical_order"]["cosine"] for row in selected], marker="o", label=label)
    ax.axhline(0, color="black", linewidth=.7); ax.set(title="Estimator fidelity across checkpoints", xlabel="epoch", ylabel="cosine with materialized exact"); ax.legend(fontsize=8); ax.grid(alpha=.25)
    files += _save(fig, output, "mechanism_fidelity")

    semantic = _jsonl(lc05 / "semantic_drift.jsonl")
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for row in semantic:
        ax.scatter(row["standard_evaluation"]["new_accuracy_pct"], row["open_world"]["open_world_new_accuracy_pct"], s=70, label=_identity(row))
    ax.plot([0, 100], [0, 100], linestyle="--", color="gray"); ax.set(title="Standard vs open-world New", xlabel="standard New (%)", ylabel="open-world New (%)"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    files += _save(fig, output, "semantic_open_world")

    sharp = _load(lc06 / "sharpness_summary.json")["rows"]
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    for identity in sorted({_identity(row) for row in sharp}):
        selected = sorted((row for row in sharp if _identity(row) == identity), key=lambda row: row["radius"])
        ax.plot([row["radius"] for row in selected], [row["sharpness_mean"] for row in selected], marker="o", label=identity)
    ax.set(title="Prompt-space sampled sharpness", xlabel="radius", ylabel="mean symmetric loss increase"); ax.legend(fontsize=7); ax.grid(alpha=.25)
    files += _save(fig, output, "prompt_sharpness")
    sources = [lc01 / "summary.json", lc01 / "lc01" / "replay_summary.json", lc05 / "semantic_drift.jsonl", lc06 / "sharpness_summary.json"]
    atomic_write_json(output / "plot_manifest.json", {
        "schema_version": PLOT_SCHEMA, "task": "LC07", "plots": files,
        "source_artifact_sha256": {str(path): _sha(path) for path in sources},
    })
    return tuple(output / name for name in files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("lc05", "lc06", "lc07"))
    parser.add_argument("--run-dir")
    parser.add_argument("--lc01-run"); parser.add_argument("--lc05-run"); parser.add_argument("--lc06-run")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    if args.task == "lc05":
        paths = plot_lc05(Path(args.run_dir))
    elif args.task == "lc06":
        paths = plot_lc06(Path(args.run_dir))
    else:
        paths = plot_lc07(
            lc01_run=Path(args.lc01_run), lc05_run=Path(args.lc05_run),
            lc06_run=Path(args.lc06_run), output_dir=Path(args.output_dir),
        )
    print(json.dumps({"plots": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
