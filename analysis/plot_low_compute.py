"""Deterministic LC01+LC04 plots generated without model/data access."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.aggregate_low_compute import build_primary_rows


def plot_saved_artifacts(run_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    started = time.perf_counter()
    rows = build_primary_rows(run_dir)
    config = yaml.safe_load((Path(run_dir) / "config.yaml").read_text(encoding="utf-8"))
    paper_lambda = float(config["paper_constants"]["ema_lambda"])
    coverage_lambda = float(config["lc01"]["coverage_lambda"])
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    epochs = sorted({row["checkpoint_epoch"] for row in rows})
    colors = plt.cm.viridis([index / max(1, len(epochs) - 1) for index in range(len(epochs))])
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 11.0), constrained_layout=True)
    for epoch, color in zip(epochs, colors):
        selected = [row for row in rows if row["checkpoint_epoch"] == epoch]
        x = [row["lambda"] for row in selected]
        label = f"epoch {epoch}"
        axes[0, 0].plot(x, [row["ema_exact_cosine"] for row in selected], marker="o", color=color, label=label)
        axes[0, 1].plot(x, [row["ema_exact_relative_l2"] for row in selected], marker="o", color=color)
        axes[0, 2].errorbar(
            x, [row["ema_exact_cosine"] for row in selected],
            yerr=[row["order_sensitivity_sd"] for row in selected], marker="o", color=color,
        )
        axes[1, 0].plot(
            [row["effective_sample_size"] for row in selected],
            [row["ema_exact_cosine"] for row in selected], marker="o", color=color,
        )
        axes[1, 1].plot(x, [row["gB_exact_ema_cosine"] for row in selected], marker="o", color=color)
        axes[2, 2].plot(x, [row["delta_exact_ema_cosine"] for row in selected], marker="o", color=color)
        axes[1, 2].scatter(
            [row["ema_exact_cosine"] for row in selected],
            [row["gB_exact_ema_cosine"] for row in selected], color=color,
        )
    for row in rows:
        if row["lambda"] == paper_lambda:
            axes[2, 0].scatter(
                row["actual_checkpoint_parameter_cosine"], row["logit_agreement"],
                color=colors[epochs.index(row["checkpoint_epoch"])],
            )
            axes[2, 1].scatter(
                row["checkpoint_epoch"], row["ema_exact_cosine"],
                color=colors[epochs.index(row["checkpoint_epoch"])],
            )
    axes[0, 0].set(title="Exact vs EMA cosine", xlabel="lambda", ylabel="cosine")
    axes[0, 1].set(title="Exact vs EMA relative L2", xlabel="lambda", ylabel="relative L2")
    axes[0, 2].set(title="Minibatch-order sensitivity", xlabel="lambda", ylabel="cosine +/- SD")
    axes[1, 0].set(title="Effective sample size vs fidelity", xlabel="descriptive N_eff", ylabel="cosine")
    axes[1, 1].set(title="Projected batch-component fidelity", xlabel="lambda", ylabel="g_B cosine")
    axes[1, 2].set(title="Parameter vs projected fidelity", xlabel="EMA/exact cosine", ylabel="g_B cosine")
    axes[2, 0].set(title="Parameter vs function fidelity", xlabel="actual EMA/exact cosine", ylabel="logit-response cosine")
    axes[2, 1].set(title="Paper-lambda fidelity over checkpoints", xlabel="checkpoint epoch", ylabel="EMA/exact cosine")
    axes[2, 2].set(title="Displacement fidelity", xlabel="lambda", ylabel="delta cosine")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    for axis in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 1], axes[2, 2]):
        axis.axvline(paper_lambda, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(coverage_lambda, color="black", linestyle=":", linewidth=0.8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("How Global Is SAMPLe's Global Gradient?")
    matplotlib.rcParams["svg.hashsalt"] = "sample_fg_lc01_lc04_v1"
    png = destination / "global_gradient_fidelity.png"
    svg = destination / "global_gradient_fidelity.svg"
    fig.savefig(png, dpi=160, metadata={"Software": "sample_fg"})
    fig.savefig(svg, metadata={"Creator": "sample_fg", "Date": None})
    plt.close(fig)
    accounting_path = Path(run_dir) / "lc01" / "compute_accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8"))
    accounting["plotting_aggregation_wall_s"] = time.perf_counter() - started
    # Analysis owns only its derived output; source probe artifacts stay immutable.
    (destination / "analysis_compute_accounting.json").write_text(
        json.dumps(accounting, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return png, svg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    png, svg = plot_saved_artifacts(Path(args.run_dir), Path(args.output_dir))
    print(json.dumps({"png": str(png), "svg": str(svg)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
