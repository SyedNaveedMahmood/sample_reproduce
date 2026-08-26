"""Generate deterministic PNG/SVG figures from Task-23 aggregates."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

try:
    from analysis.make_tables import AnalysisInputError, SMOKE_LABEL, load_aggregate_bundle, method_label, method_sort
except ModuleNotFoundError:  # Direct execution from analysis/.
    from make_tables import AnalysisInputError, SMOKE_LABEL, load_aggregate_bundle, method_label, method_sort


PLOT_MANIFEST_SCHEMA_VERSION = "sample_fg.analysis_plots.v1"
WIDTH, HEIGHT = 1000, 640
LEFT, RIGHT, TOP, BOTTOM = 100, 35, 85, 90
COLORS = ("#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be185d", "#4b5563")


@dataclass
class Series:
    label: str
    points: list[tuple[float, float]]
    color: str
    connect: bool = True
    marker: str = "circle"
    error: list[tuple[float, float, float]] = field(default_factory=list)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise AnalysisInputError("Nonfinite value reached plot rendering")
    return number


def _bounds(series: Sequence[Series]) -> tuple[float, float, float, float]:
    points = [point for item in series for point in item.points]
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    for item in series:
        for x, low, high in item.error:
            xs.append(x); ys.extend((low, high))
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    xpad = (xmax - xmin) * 0.08 if xmax != xmin else 0.5
    ypad = (ymax - ymin) * 0.10 if ymax != ymin else max(abs(ymin) * 0.1, 0.5)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _geometry(series: Sequence[Series]):
    xmin, xmax, ymin, ymax = _bounds(series)
    plot_w, plot_h = WIDTH - LEFT - RIGHT, HEIGHT - TOP - BOTTOM
    def xy(x: float, y: float) -> tuple[float, float]:
        return LEFT + (x - xmin) / (xmax - xmin) * plot_w, TOP + (ymax - y) / (ymax - ymin) * plot_h
    return (xmin, xmax, ymin, ymax), xy


def _svg_plot(
    path: Path, title: str, x_label: str, y_label: str, series: Sequence[Series],
    *, smoke: bool, x_ticks: Mapping[float, str] | None = None,
    refresh_x: Sequence[float] = (), metric_keys: Sequence[str] = (),
) -> None:
    bounds, xy = _geometry(series)
    xmin, xmax, ymin, ymax = bounds
    label = SMOKE_LABEL if smoke else "SCIENTIFIC AGGREGATE"
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<metadata>mode={"smoke" if smoke else "scientific"}; metrics={escape(",".join(metric_keys))}; error=sample_std_ddof1</metadata>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{WIDTH/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{escape(title)}</text>',
        f'<text x="{WIDTH/2}" y="55" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#{"b91c1c" if smoke else "374151"}">{escape(label)}</text>',
        f'<line x1="{LEFT}" y1="{HEIGHT-BOTTOM}" x2="{WIDTH-RIGHT}" y2="{HEIGHT-BOTTOM}" stroke="#111827"/>',
        f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" y2="{HEIGHT-BOTTOM}" stroke="#111827"/>',
    ]
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        _, py = xy(xmin, value)
        parts.append(f'<line x1="{LEFT}" y1="{py:.3f}" x2="{WIDTH-RIGHT}" y2="{py:.3f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{LEFT-10}" y="{py+4:.3f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3g}</text>')
    ticks = x_ticks or {xmin: f"{xmin:.3g}", xmax: f"{xmax:.3g}"}
    for value, text_value in ticks.items():
        px, _ = xy(float(value), ymin)
        parts.append(f'<line x1="{px:.3f}" y1="{HEIGHT-BOTTOM}" x2="{px:.3f}" y2="{HEIGHT-BOTTOM+5}" stroke="#111827"/>')
        parts.append(f'<text x="{px:.3f}" y="{HEIGHT-BOTTOM+22}" text-anchor="middle" font-family="sans-serif" font-size="10">{escape(str(text_value))}</text>')
    for value in sorted(set(refresh_x)):
        px, _ = xy(float(value), ymin)
        parts.append(f'<line class="refresh-marker" x1="{px:.3f}" y1="{TOP}" x2="{px:.3f}" y2="{HEIGHT-BOTTOM}" stroke="#f59e0b" stroke-dasharray="5,4" opacity="0.65"/>')
    for item in series:
        points = sorted(item.points)
        coords = [xy(x, y) for x, y in points]
        if item.connect and len(coords) > 1:
            joined = " ".join(f"{x:.3f},{y:.3f}" for x, y in coords)
            parts.append(f'<polyline fill="none" stroke="{item.color}" stroke-width="2" points="{joined}"/>')
        for x, low, high in item.error:
            px, py_low = xy(x, low); _, py_high = xy(x, high)
            parts.append(f'<line class="sample-std-errorbar" x1="{px:.3f}" y1="{py_low:.3f}" x2="{px:.3f}" y2="{py_high:.3f}" stroke="{item.color}" stroke-width="2"/>')
        for px, py in coords:
            if item.marker == "triangle":
                parts.append(f'<polygon points="{px:.3f},{py-5:.3f} {px-5:.3f},{py+5:.3f} {px+5:.3f},{py+5:.3f}" fill="{item.color}"/>')
            else:
                parts.append(f'<circle cx="{px:.3f}" cy="{py:.3f}" r="4" fill="{item.color}"/>')
    parts.append(f'<text x="{(LEFT+WIDTH-RIGHT)/2}" y="{HEIGHT-28}" text-anchor="middle" font-family="sans-serif" font-size="13">{escape(x_label)}</text>')
    parts.append(f'<text x="22" y="{(TOP+HEIGHT-BOTTOM)/2}" text-anchor="middle" transform="rotate(-90 22 {(TOP+HEIGHT-BOTTOM)/2})" font-family="sans-serif" font-size="13">{escape(y_label)}</text>')
    legend_y = TOP + 5
    legend_items: list[Series] = []
    for item in series:
        if item.label not in {entry.label for entry in legend_items}:
            legend_items.append(item)
    for index, item in enumerate(legend_items[:12]):
        x = WIDTH - RIGHT - 225
        y = legend_y + index * 18
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x+18}" y2="{y}" stroke="{item.color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+24}" y="{y+4}" font-family="sans-serif" font-size="11">{escape(item.label)}</text>')
    if not any(item.points for item in series):
        parts.append(f'<text x="{WIDTH/2}" y="{HEIGHT/2}" text-anchor="middle" font-family="sans-serif" font-size="18" fill="#6b7280">No structured observations for this view</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _png_plot(
    path: Path, title: str, x_label: str, y_label: str, series: Sequence[Series],
    *, smoke: bool, x_ticks: Mapping[float, str] | None = None,
    refresh_x: Sequence[float] = (), metric_keys: Sequence[str] = (),
) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw, font = ImageDraw.Draw(image), ImageFont.load_default()
    bounds, xy = _geometry(series)
    xmin, xmax, ymin, ymax = bounds
    draw.text((WIDTH / 2, 22), title, fill="#111827", font=font, anchor="mm")
    draw.text((WIDTH / 2, 47), SMOKE_LABEL if smoke else "SCIENTIFIC AGGREGATE", fill="#b91c1c" if smoke else "#374151", font=font, anchor="mm")
    draw.line((LEFT, HEIGHT - BOTTOM, WIDTH - RIGHT, HEIGHT - BOTTOM), fill="#111827", width=2)
    draw.line((LEFT, TOP, LEFT, HEIGHT - BOTTOM), fill="#111827", width=2)
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        _, py = xy(xmin, value)
        draw.line((LEFT, py, WIDTH - RIGHT, py), fill="#e5e7eb")
        draw.text((LEFT - 8, py), f"{value:.3g}", fill="#111827", font=font, anchor="rm")
    ticks = x_ticks or {xmin: f"{xmin:.3g}", xmax: f"{xmax:.3g}"}
    for value, text_value in ticks.items():
        px, _ = xy(float(value), ymin)
        draw.line((px, HEIGHT - BOTTOM, px, HEIGHT - BOTTOM + 5), fill="#111827")
        draw.text((px, HEIGHT - BOTTOM + 17), str(text_value), fill="#111827", font=font, anchor="mm")
    for value in sorted(set(refresh_x)):
        px, _ = xy(float(value), ymin)
        for py in range(TOP, HEIGHT - BOTTOM, 9):
            draw.line((px, py, px, min(py + 5, HEIGHT - BOTTOM)), fill="#f59e0b", width=2)
    for item in series:
        coords = [xy(x, y) for x, y in sorted(item.points)]
        if item.connect and len(coords) > 1:
            draw.line(coords, fill=item.color, width=2)
        for x, low, high in item.error:
            px, py_low = xy(x, low); _, py_high = xy(x, high)
            draw.line((px, py_low, px, py_high), fill=item.color, width=2)
            draw.line((px - 4, py_low, px + 4, py_low), fill=item.color, width=2)
            draw.line((px - 4, py_high, px + 4, py_high), fill=item.color, width=2)
        for px, py in coords:
            if item.marker == "triangle":
                draw.polygon(((px, py - 5), (px - 5, py + 5), (px + 5, py + 5)), fill=item.color)
            else:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=item.color)
    draw.text(((LEFT + WIDTH - RIGHT) / 2, HEIGHT - 28), x_label, fill="#111827", font=font, anchor="mm")
    draw.text((24, TOP - 16), y_label, fill="#111827", font=font, anchor="lm")
    legend_items: list[Series] = []
    for item in series:
        if item.label not in {entry.label for entry in legend_items}:
            legend_items.append(item)
    for index, item in enumerate(legend_items[:12]):
        x, y = WIDTH - RIGHT - 225, TOP + 5 + index * 18
        draw.line((x, y, x + 18, y), fill=item.color, width=3)
        draw.text((x + 24, y), item.label, fill="#111827", font=font, anchor="lm")
    if not any(item.points for item in series):
        draw.text((WIDTH / 2, HEIGHT / 2), "No structured observations for this view", fill="#6b7280", font=font, anchor="mm")
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("analysis_mode", "smoke" if smoke else "scientific")
    metadata.add_text("metric_keys", ",".join(metric_keys))
    metadata.add_text("error_representation", "sample standard deviation ddof=1")
    image.save(path, format="PNG", pnginfo=metadata, optimize=False, compress_level=9)


def _emit_plot(
    output_dir: Path, name: str, title: str, x_label: str, y_label: str,
    series: Sequence[Series], *, smoke: bool, metric_keys: Sequence[str],
    x_ticks: Mapping[float, str] | None = None, refresh_x: Sequence[float] = (),
    raw_seed_points: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = dict(smoke=smoke, x_ticks=x_ticks, refresh_x=refresh_x, metric_keys=metric_keys)
    _svg_plot(output_dir / f"{name}.svg", title, x_label, y_label, series, **kwargs)
    _png_plot(output_dir / f"{name}.png", title, x_label, y_label, series, **kwargs)
    return {
        "files": [f"{name}.svg", f"{name}.png"], "title": title,
        "x_axis": x_label, "y_axis": y_label, "metric_keys": list(metric_keys),
        "series": len(series), "points": sum(len(item.points) for item in series),
        "raw_seed_points": raw_seed_points,
        "error_representation": "sample standard deviation ddof=1; none when n=1",
        "periodic_refresh_markers": bool(refresh_x),
    }


def _diagnostic_series(rows: Sequence[dict[str, Any]], field: str, *, absolute: bool = False) -> list[Series]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get(field) is None:
            continue
        groups.setdefault(str(row.get("run_id")), []).append(row)
    output, color_by_group = [], {}
    for index, run_id in enumerate(sorted(groups)):
        values = sorted(groups[run_id], key=lambda row: int(row.get("optimizer_step") or 0))
        group = (values[0].get("dataset"), method_label(values[0]))
        if group not in color_by_group:
            color_by_group[group] = COLORS[len(color_by_group) % len(COLORS)]
        max_step = max(int(row.get("optimizer_step") or 0) for row in values)
        points = []
        for row in values:
            x = 0.0 if max_step == 0 else int(row.get("optimizer_step") or 0) / max_step
            y = _finite(row.get(field))
            if y is not None:
                points.append((x, abs(y) if absolute else y))
        output.append(Series(f"{values[0].get('dataset')} {method_label(values[0])}", points, color_by_group[group]))
    return output


def _short_method(row: Mapping[str, Any]) -> str:
    label = method_label(row)
    return label.replace("SAMPLe-", "").replace("Periodic ", "P ")


def _base_new_hm_series(summary: Sequence[dict[str, Any]], runs: Sequence[dict[str, Any]]) -> tuple[list[Series], dict[float, str]]:
    selected = [row for row in summary if int(row.get("shots") or 0) == 16]
    ordered = sorted(selected or list(summary), key=method_sort)
    ticks = {
        float(index): f"{str(row.get('dataset')).upper()} {_short_method(row)}"
        for index, row in enumerate(ordered)
    }
    series = []
    for metric_index, metric in enumerate(("base", "new", "hm")):
        means, error, raw = [], [], []
        for index, row in enumerate(ordered):
            mean, std, n = _finite(row.get(f"{metric}_mean")), _finite(row.get(f"{metric}_std")), int(row.get(f"{metric}_n") or 0)
            if mean is not None:
                means.append((float(index), mean))
                if std is not None and n > 1:
                    error.append((float(index), mean - std, mean + std))
            matching = [item for item in runs if item.get("experiment_id") == row.get("experiment_id") and item.get("dataset") == row.get("dataset") and item.get("shots") == row.get("shots") and method_label(item) == method_label(row)]
            for item in matching:
                value = _finite(item.get(f"{metric}_accuracy_pct" if metric != "hm" else "hm_pct"))
                if value is not None:
                    raw.append((index + (int(item.get("seed") or 0) - 2) * 0.045, value))
        color = COLORS[metric_index]
        series.append(Series(f"{metric.title()} mean ± sample std", means, color, connect=False, error=error))
        series.append(Series(f"{metric.title()} raw seeds", raw, color, connect=False, marker="triangle"))
    return series, ticks


def _shot_series(summary: Sequence[dict[str, Any]], metric: str) -> list[Series]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary:
        groups.setdefault((str(row.get("dataset")), method_label(row)), []).append(row)
    output = []
    for index, ((dataset, label), rows) in enumerate(sorted(groups.items())):
        points = [(float(row["shots"]), float(row[f"{metric}_mean"])) for row in sorted(rows, key=lambda row: int(row["shots"])) if row.get(f"{metric}_mean") is not None]
        errors = []
        for row in rows:
            mean, std, n = _finite(row.get(f"{metric}_mean")), _finite(row.get(f"{metric}_std")), int(row.get(f"{metric}_n") or 0)
            if mean is not None and std is not None and n > 1:
                errors.append((float(row["shots"]), mean - std, mean + std))
        output.append(Series(f"{dataset} {label}", points, COLORS[index % len(COLORS)], error=errors))
    return output


def _runtime_series(rows: Sequence[dict[str, Any]]) -> tuple[list[Series], dict[float, str]]:
    shot16 = [row for row in rows if int(row.get("shots") or 0) == 16]
    rows = shot16 or rows
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("experiment_id"), row.get("dataset"), row.get("shots"),
            method_label(row), row.get("precision"),
        )
        groups.setdefault(key, []).append(row)
    ordered = sorted(groups.items(), key=lambda item: method_sort(sorted(item[1], key=method_sort)[0]))
    means, errors, raw, ticks = [], [], [], {}
    for index, ((_, dataset, shots, label, _), values) in enumerate(ordered):
        ticks[float(index)] = f"{str(dataset).upper()} {label.replace('SAMPLe-', '')}"
        numbers = [float(row["train_time_overhead_vs_sample_ema_pct"]) for row in values if row.get("train_time_overhead_vs_sample_ema_pct") is not None]
        if not numbers:
            continue
        mean = sum(numbers) / len(numbers)
        means.append((float(index), mean))
        if len(numbers) > 1:
            variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
            std = math.sqrt(variance)
            errors.append((float(index), mean - std, mean + std))
        for row in sorted(values, key=lambda item: int(item.get("seed") or 0)):
            value = _finite(row.get("train_time_overhead_vs_sample_ema_pct"))
            if value is not None:
                raw.append((index + (int(row.get("seed") or 0) - 2) * 0.045, value))
    return (
        [
            Series("mean ± sample std", means, COLORS[0], connect=False, error=errors),
            Series("raw seed points", raw, COLORS[2], connect=False, marker="triangle"),
        ],
        ticks,
    )


def _periodic_k_summary_series(
    rows: Sequence[dict[str, Any]],
    metric: str,
) -> list[Series]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if (
            row.get("experiment_id") != "E1"
            or row.get("estimator_mode") != "periodic"
            or row.get(f"{metric}_mean") is None
        ):
            continue
        groups.setdefault(
            (str(row.get("dataset")), int(row.get("shots") or 0)), []
        ).append(row)
    output: list[Series] = []
    for index, ((dataset, shots), values) in enumerate(sorted(groups.items())):
        points, errors = [], []
        for row in sorted(values, key=lambda item: int(item["periodic_k_steps"])):
            k = float(row["periodic_k_steps"])
            mean = float(row[f"{metric}_mean"])
            points.append((k, mean))
            std = _finite(row.get(f"{metric}_std"))
            if std is not None and int(row.get(f"{metric}_n") or 0) > 1:
                errors.append((k, mean - std, mean + std))
        output.append(
            Series(
                f"{dataset} {shots}shot",
                points,
                COLORS[index % len(COLORS)],
                error=errors,
            )
        )
    return output


def _periodic_k_efficiency_series(rows: Sequence[dict[str, Any]]) -> list[Series]:
    groups: dict[tuple[str, int, str], dict[int, list[float]]] = {}
    for row in rows:
        if row.get("experiment_id") != "E1" or row.get("estimator_mode") != "periodic":
            continue
        for label, field in (
            ("training wall time", "train_total_s"),
            ("exact-gradient wall time", "exact_gradient_total_s"),
        ):
            value = _finite(row.get(field))
            if value is None:
                continue
            key = (str(row.get("dataset")), int(row.get("shots") or 0), label)
            groups.setdefault(key, {}).setdefault(
                int(row.get("periodic_k_steps") or 0), []
            ).append(value)
    output: list[Series] = []
    for index, ((dataset, shots, label), by_k) in enumerate(sorted(groups.items())):
        points, errors = [], []
        for k, values in sorted(by_k.items()):
            mean = sum(values) / len(values)
            points.append((float(k), mean))
            if len(values) > 1:
                std = math.sqrt(
                    sum((value - mean) ** 2 for value in values)
                    / (len(values) - 1)
                )
                errors.append((float(k), mean - std, mean + std))
        output.append(
            Series(
                f"{dataset} {shots}shot {label}",
                points,
                COLORS[index % len(COLORS)],
                error=errors,
            )
        )
    return output


def make_plots(input_dir: Path, output_dir: Path, *, mode: str = "scientific") -> dict[str, Any]:
    bundle = load_aggregate_bundle(Path(input_dir), mode)
    output_dir, smoke = Path(output_dir).resolve(), mode == "smoke"
    plots: dict[str, Any] = {}
    diagnostic_specs = (
        ("global_estimate_fidelity", "Global-estimate fidelity through training", "grad/global_estimate_exact_cosine", "cos(F, exact)", False),
        ("estimator_relative_l2", "Estimator relative L2 error through training", "grad/global_estimate_exact_relative_l2", "relative L2 error", False),
        ("estimator_norm_ratio", "Estimator/exact norm ratio through training", "grad/global_estimate_exact_norm_ratio", "norm ratio", False),
        ("batch_component_estimator_alignment", "Batch-component alignment with active estimator", "grad/batch_component_estimator_cosine", "signed cos(g_B, active estimator)", False),
        ("construction_orthogonality", "Construction-level exact orthogonality", "grad/batch_component_exact_cosine", "signed cos(g_B, exact)", False),
        ("reference_construction_orthogonality", "Exact-reference construction orthogonality", "grad/reference_batch_component_exact_cosine", "signed cos(reference g_B, exact)", False),
        ("perturbed_estimator_alignment", "Perturbed-gradient alignment with active estimator", "grad/perturbed_gradient_estimator_cosine", "signed cos(p, active estimator)", False),
        ("objective_orthogonality", "Objective-level exact orthogonality", "grad/perturbed_gradient_exact_cosine", "abs cos(p, exact)", True),
        ("perturbed_batch_component_alignment", "Perturbed-gradient alignment with batch component", "grad/perturbed_gradient_batch_component_cosine", "signed cos(p, g_B)", False),
        ("perturbed_batch_alignment", "Perturbed-gradient alignment with batch gradient", "grad/perturbed_gradient_batch_cosine", "signed cos(p, g)", False),
        ("taylor_exploitation", "Taylor exploitation term", "taylor/exploitation_term", "exploitation term", False),
        ("taylor_exploration", "Taylor exploration term", "taylor/exploration_term", "exploration term", False),
        ("taylor_joint", "Taylor joint alignment term", "taylor/joint_alignment_term", "joint term", False),
    )
    for name, title, field, y_label, absolute in diagnostic_specs:
        plots[name] = _emit_plot(output_dir, name, title, "normalized optimizer progress", y_label, _diagnostic_series(bundle["diagnostics"], field, absolute=absolute), smoke=smoke, metric_keys=(field,), x_ticks={0.0: "0", 0.5: "0.5", 1.0: "1"})
    base_series, base_ticks = _base_new_hm_series(bundle["summary"], bundle["runs"])
    plots["base_new_hm_by_estimator"] = _emit_plot(output_dir, "base_new_hm_by_estimator", "Base/New/HM by estimator", "method (manifest order)", "accuracy (%)", base_series, smoke=smoke, metric_keys=("base_accuracy_pct", "new_accuracy_pct", "hm_pct"), x_ticks=base_ticks, raw_seed_points=True)
    for metric in ("base", "new", "hm"):
        name = f"shot_count_{metric}"
        plots[name] = _emit_plot(output_dir, name, f"Shot-count {metric.title()} generalization", "shots", f"{metric.title()} accuracy (%)", _shot_series(bundle["summary"], metric), smoke=smoke, metric_keys=("shots", f"{metric}_mean", f"{metric}_std"), x_ticks={4.0: "4", 8.0: "8", 16.0: "16"})
    hm_groups: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in sorted(bundle["runs"], key=method_sort):
        x, y = _finite(row.get("global_estimate_exact_relative_l2_mean")), _finite(row.get("hm_pct"))
        if x is not None and y is not None:
            hm_groups.setdefault((str(row.get("dataset")), method_label(row)), []).append((x, y))
    hm_series = [
        Series(f"{dataset} {label}", sorted(points), COLORS[index % len(COLORS)], connect=False)
        for index, ((dataset, label), points) in enumerate(sorted(hm_groups.items()))
    ]
    plots["hm_vs_estimator_error"] = _emit_plot(output_dir, "hm_vs_estimator_error", "HM versus estimator error (descriptive)", "run-level estimator relative L2", "HM (%)", hm_series, smoke=smoke, metric_keys=("global_estimate_exact_relative_l2_mean", "hm_pct"), raw_seed_points=True)
    efficiency_series, efficiency_ticks = _runtime_series(bundle["efficiency"])
    plots["runtime_overhead"] = _emit_plot(output_dir, "runtime_overhead", "Runtime overhead versus matched SAMPLe-EMA", "method (manifest order)", "train time overhead (%)", efficiency_series, smoke=smoke, metric_keys=("train_time_overhead_vs_sample_ema_pct",), x_ticks=efficiency_ticks, raw_seed_points=True)
    plots["periodic_k_fidelity"] = _emit_plot(
        output_dir,
        "periodic_k_fidelity",
        "Periodic estimator fidelity versus K (E1 selection evidence)",
        "periodic refresh K (optimizer steps)",
        "cos(active estimator, exact)",
        _periodic_k_summary_series(bundle["summary"], "estimator_exact_cosine"),
        smoke=smoke,
        metric_keys=("periodic_k_steps", "estimator_exact_cosine_mean"),
        x_ticks={2.0: "2", 4.0: "4", 8.0: "8", 16.0: "16"},
    )
    plots["periodic_k_efficiency"] = _emit_plot(
        output_dir,
        "periodic_k_efficiency",
        "Periodic estimator cost versus K (E1 selection evidence)",
        "periodic refresh K (optimizer steps)",
        "wall time (seconds)",
        _periodic_k_efficiency_series(bundle["efficiency"]),
        smoke=smoke,
        metric_keys=("periodic_k_steps", "train_total_s", "exact_gradient_total_s"),
        x_ticks={2.0: "2", 4.0: "4", 8.0: "8", 16.0: "16"},
    )
    periodic = [row for row in bundle["diagnostics"] if row.get("estimator_mode") == "periodic" and row.get("estimator_age_steps") is not None]
    periodic_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in periodic:
        periodic_groups.setdefault((row.get("dataset"), row.get("shots"), row.get("periodic_k_steps")), []).append(row)
    periodic_series, refresh_x = [], []
    for index, group in enumerate(sorted(periodic_groups, key=lambda value: tuple(str(item) for item in value))):
        rows = sorted(periodic_groups[group], key=lambda row: (int(row.get("optimizer_step") or 0), int(row.get("seed") or 0)))
        by_step: dict[int, list[float]] = {}
        for row in rows:
            by_step.setdefault(int(row["optimizer_step"]), []).append(float(row["estimator_age_steps"]))
        points = [(float(step), sum(ages) / len(ages)) for step, ages in sorted(by_step.items())]
        periodic_series.append(Series(f"{group[0]} {group[1]}shot periodic K={group[2]}", points, COLORS[index % len(COLORS)]))
        refresh_x.extend(float(row["optimizer_step"]) for row in rows if row.get("estimator_refreshed") is True)
    plots["periodic_refresh_age"] = _emit_plot(output_dir, "periodic_refresh_age", "Periodic estimator refresh and age", "optimizer step", "estimator age (steps)", periodic_series, smoke=smoke, metric_keys=("estimator_age_steps", "estimator_refreshed"), refresh_x=refresh_x)
    manifest = {
        "schema_version": PLOT_MANIFEST_SCHEMA_VERSION, "mode": mode,
        "scientific": not smoke, "smoke": smoke,
        "display_label": SMOKE_LABEL if smoke else None,
        "source_aggregation": str(bundle["input_dir"]),
        "ordering": "experiment manifest method order; never performance",
        "statistics": {"error_bars": "sample standard deviation (ddof=1)", "n1": "no error bar", "raw_seed_points": True, "significance_stars": False},
        "orthogonality_concepts": {
            "construction": "grad/batch_component_exact_cosine",
            "reference_construction": "grad/reference_batch_component_exact_cosine",
            "objective": "grad/perturbed_gradient_exact_cosine",
            "collapsed": False,
        },
        "periodic_k_selection": {
            "accuracy_used": False,
            "fidelity_metric": "estimator_exact_cosine_mean",
            "cost_metrics": ["train_total_s", "exact_gradient_total_s"],
        },
        "plots": plots,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_plots_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("scientific", "smoke"), default="scientific")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(make_plots(Path(arguments.input_dir), Path(arguments.output_dir), mode=arguments.mode), indent=2, sort_keys=True, ensure_ascii=False))
