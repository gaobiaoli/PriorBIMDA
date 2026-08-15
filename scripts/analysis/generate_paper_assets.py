#!/usr/bin/env python3
"""Generate paper-ready quantitative and process-diagram assets.

The generator is deliberately read-only with respect to registered experiment
results.  Every quantitative panel is reconstructed from the compact artifacts
under ``results/`` and ``data/provenance/``.  A machine-readable manifest records
the source hashes, plotted values, claim scope, and whether evidence already
exists or is only a proposed experiment.

The default output contains independent PNG, SVG, and PDF files.  PNG is useful
for quick previews, while SVG/PDF remain editable when assembling a paper or PPT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

GENERATOR = "scripts/analysis/generate_paper_assets.py"
DEFAULT_FORMATS = ("png", "svg", "pdf")
ALLOWED_FORMATS = frozenset(DEFAULT_FORMATS)

COLORS = {
    "raw": "#7A7A7A",
    "scale": "#4C78A8",
    "bim": "#F2A541",
    "learned": "#2A9D8F",
    "worse": "#D65F5F",
    "ink": "#203040",
    "muted": "#718096",
    "light": "#EDF2F7",
    "violet": "#7B61A8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate paper figures from registered compact results."
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--scale-selection",
        type=Path,
        default=Path("data/provenance/stanford_area1_robust_scale_selection_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/paper_evaluation"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=sorted(ALLOWED_FORMATS),
        help="One or more output formats (default: png svg pdf).",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


class SourceRegistry:
    """Load JSON evidence and retain a hash-addressed source receipt."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._records: dict[Path, dict[str, Any]] = {}

    def load_json(self, path: Path) -> Any:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required compact result is missing: {resolved}")
        raw = resolved.read_bytes()
        self._records.setdefault(
            resolved,
            {
                "path": _display_path(resolved, self.root),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
        )
        return json.loads(raw.decode("utf-8"))

    def reference(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._records:
            raise KeyError(f"Source has not been loaded: {resolved}")
        return str(self._records[resolved]["path"])

    @property
    def records(self) -> list[dict[str, Any]]:
        return sorted(self._records.values(), key=lambda item: str(item["path"]))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return "inf" if value > 0 else "-inf"
    return value


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#4A5568",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "PriorBIMDA-paper-assets-v1",
        }
    )


def _save_figure(
    figure: Figure,
    output_root: Path,
    relative_stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[dict[str, Any]]:
    stem = output_root / relative_stem
    stem.parent.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for extension in formats:
        if extension not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported output format: {extension}")
        path = stem.with_suffix(f".{extension}")
        metadata: dict[str, Any]
        if extension == "pdf":
            metadata = {
                "Creator": GENERATOR,
                "CreationDate": None,
                "ModDate": None,
            }
        elif extension == "svg":
            metadata = {"Creator": GENERATOR, "Date": None}
        else:
            metadata = {"Software": GENERATOR}
        figure.savefig(
            path,
            format=extension,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        if extension == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        artifacts.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "format": extension,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    plt.close(figure)
    return artifacts


def _annotate_bars(axis: Axes, bars: Any, *, digits: int = 3) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            f"{value:.{digits}f}",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )


def _metric(summary: dict[str, Any], subset: str, method: str) -> dict[str, Any]:
    return summary["aggregates"][subset][method]["pixel_micro"]


def _main_result_values(
    metrics: dict[str, Any], stanford_summary: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    slabim = metrics["slabim"]["methods"]
    stanford = stanford_summary["aggregates"]["all"]
    return {
        "SLABIM": [
            {"method": "Raw DA3", "abs_rel": slabim["raw_da3"]["abs_rel"]},
            {
                "method": "Scale only",
                "abs_rel": slabim["universal_global_scale"]["abs_rel"],
            },
            {
                "method": "Direct BIM",
                "abs_rel": slabim["universal_bim_direct"]["abs_rel"],
            },
            {
                "method": "Learned refiner",
                "abs_rel": slabim["learned_refiner"]["abs_rel"],
                "registered_method": "learned_refiner",
            },
        ],
        "2D-3D-S Area 1": [
            {
                "method": "Raw DA3",
                "abs_rel": stanford["raw_da3"]["pixel_micro"]["abs_rel"],
            },
            {
                "method": "Scale only",
                "abs_rel": stanford["robust_global_scale"]["pixel_micro"]["abs_rel"],
            },
            {
                "method": "Direct BIM",
                "abs_rel": stanford["robust_bim_direct"]["pixel_micro"]["abs_rel"],
            },
            {
                "method": "Learned refiner",
                "abs_rel": stanford["refined"]["pixel_micro"]["abs_rel"],
                "registered_method": "refined",
            },
        ],
    }


def render_main_results(values: dict[str, list[dict[str, Any]]]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), sharey=True)
    method_colors = {
        "Raw DA3": COLORS["raw"],
        "Scale only": COLORS["scale"],
        "Direct BIM": COLORS["bim"],
        "Frozen refiner": COLORS["learned"],
        "Partial E2E": COLORS["violet"],
        "Learned refiner": COLORS["learned"],
    }
    maximum = max(item["abs_rel"] for rows in values.values() for item in rows)
    for axis, (dataset, rows) in zip(axes, values.items()):
        heights = [float(item["abs_rel"]) for item in rows]
        colors = [method_colors[str(item["method"])] for item in rows]
        bars = axis.bar(range(len(rows)), heights, color=colors, width=0.68)
        _annotate_bars(axis, bars)
        axis.set_xticks(range(len(rows)), [item["method"] for item in rows], rotation=15)
        axis.set_title(dataset)
        axis.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.8)
        direct_index = next(
            index for index, row in enumerate(rows) if row["method"] == "Direct BIM"
        )
        direct = heights[direct_index]
        best_learned = min(heights[direct_index + 1 :])
        gain = 100.0 * (direct - best_learned) / direct
        axis.text(
            0.98,
            0.95,
            f"Best learned AbsRel vs direct BIM: {gain:.1f}% lower",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color=COLORS["learned"],
        )
    axes[0].set_ylabel("AbsRel ↓")
    axes[0].set_ylim(0.0, maximum * 1.22)
    figure.suptitle("Registered blind-test performance (0.2–5.0 m, pixel-micro)", y=1.02)
    figure.tight_layout()
    return figure


def _subset_values(stanford_summary: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = (
        ("all", "All valid pixels"),
        ("furniture", "Furniture"),
        ("bim_foreground_conflict", "BIM–foreground conflict"),
    )
    rows: list[dict[str, Any]] = []
    for key, label in definitions:
        methods = {}
        for method in ("raw_da3", "robust_bim_direct", "refined"):
            metric = _metric(stanford_summary, key, method)
            methods[method] = {
                "abs_rel": metric["abs_rel"],
                "count": metric["count"],
            }
        rows.append({"subset": key, "label": label, "methods": methods})
    return rows


def render_area1_subsets(rows: list[dict[str, Any]]) -> Figure:
    figure, axis = plt.subplots(figsize=(9.8, 5.1))
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.23
    specifications = (
        ("raw_da3", "Raw DA3", COLORS["raw"]),
        ("robust_bim_direct", "Universal direct BIM", COLORS["bim"]),
        ("refined", "Learned refiner", COLORS["learned"]),
    )
    for offset, (method, label, color) in zip((-width, 0.0, width), specifications):
        values = [float(row["methods"][method]["abs_rel"]) for row in rows]
        bars = axis.bar(x + offset, values, width, label=label, color=color)
        _annotate_bars(axis, bars)
    axis.set_xticks(x, [row["label"] for row in rows])
    axis.set_ylabel("AbsRel ↓")
    axis.set_title("Area 1 blind test by semantic/BIM-conflict subset")
    axis.set_ylim(0.0, 0.36)
    axis.grid(axis="y", color="#D8DEE9", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False, ncol=3, loc="upper center")
    conflict = rows[-1]["methods"]
    difference = float(conflict["refined"]["abs_rel"]) - float(
        conflict["robust_bim_direct"]["abs_rel"]
    )
    direction = "higher (worse)" if difference > 0 else "lower (better)"
    axis.text(
        0.99,
        0.70,
        f"Conflict subset is not hidden:\nlearned − direct = {difference:+.4f} ({direction})",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=COLORS["worse"] if difference > 0 else COLORS["learned"],
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#CBD5E0"},
    )
    figure.text(
        0.5,
        0.01,
        "Bars are pixel-micro point estimates on fixed support; room-bootstrap uncertainty "
        "is shown separately.",
        ha="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def _room_pairs(stanford_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for room, subsets in stanford_summary["per_room"].items():
        direct = float(subsets["all"]["robust_bim_direct"]["abs_rel"])
        refined = float(subsets["all"]["refined"]["abs_rel"])
        rows.append(
            {
                "room": room,
                "robust_bim_direct_abs_rel": direct,
                "refined_abs_rel": refined,
                "difference": refined - direct,
            }
        )
    return rows


def _bootstrap_rows(stanford_summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "all": "All pixels",
        "furniture": "Furniture",
        "bim_foreground_conflict": "BIM–foreground conflict",
    }
    rows = []
    for subset, label in labels.items():
        item = stanford_summary["paired_room_bootstrap"][subset]["abs_rel"]
        rows.append(
            {
                "subset": subset,
                "label": label,
                "mean_difference": item["mean_difference"],
                "confidence_interval_95": item["confidence_interval_95"],
                "rooms": item["rooms"],
                "bootstrap_repetitions": item["bootstrap_repetitions"],
                "seed": item["seed"],
                "candidate_better_room_fraction": item["candidate_better_room_fraction"],
            }
        )
    return rows


def render_room_pairs_and_bootstrap(
    room_rows: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]]
) -> Figure:
    figure, (slope_axis, forest_axis) = plt.subplots(
        1,
        2,
        figsize=(13.0, 5.5),
        gridspec_kw={"width_ratios": (1.25, 1.0)},
    )
    ordered = sorted(room_rows, key=lambda item: float(item["difference"]))
    y = np.arange(len(ordered), dtype=np.float64)
    for index, row in enumerate(ordered):
        direct = float(row["robust_bim_direct_abs_rel"])
        refined = float(row["refined_abs_rel"])
        improved = refined < direct
        slope_axis.plot(
            [direct, refined],
            [index, index],
            color=COLORS["learned"] if improved else COLORS["worse"],
            linewidth=2.2,
            alpha=0.85,
            zorder=1,
        )
    direct_values = [float(row["robust_bim_direct_abs_rel"]) for row in ordered]
    learned_values = [float(row["refined_abs_rel"]) for row in ordered]
    slope_axis.scatter(
        direct_values,
        y,
        marker="o",
        facecolors="white",
        edgecolors=COLORS["bim"],
        linewidths=1.8,
        s=55,
        label="Robust direct BIM",
        zorder=2,
    )
    slope_axis.scatter(
        learned_values,
        y,
        marker="o",
        color=COLORS["learned"],
        s=46,
        label="Learned refiner",
        zorder=3,
    )
    slope_axis.set_yticks(y, [str(row["room"]) for row in ordered])
    slope_axis.set_xlabel("Room-level AbsRel ↓")
    slope_axis.set_title("Paired held-out rooms")
    slope_axis.grid(axis="x", color="#D8DEE9", linewidth=0.7)
    slope_axis.legend(frameon=False, loc="lower right")

    forest_y = np.arange(len(bootstrap_rows), dtype=np.float64)
    for index, row in enumerate(bootstrap_rows):
        mean = float(row["mean_difference"])
        low, high = (float(value) for value in row["confidence_interval_95"])
        color = COLORS["learned"] if high < 0 else COLORS["worse"] if mean > 0 else COLORS["raw"]
        forest_axis.errorbar(
            mean,
            index,
            xerr=np.asarray([[mean - low], [high - mean]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=7,
            linewidth=2,
        )
        forest_axis.text(
            high + 0.001,
            index,
            f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]",
            va="center",
            fontsize=8.2,
        )
    forest_axis.axvline(0.0, color=COLORS["ink"], linestyle="--", linewidth=1)
    forest_axis.set_yticks(forest_y, [str(row["label"]) for row in bootstrap_rows])
    forest_axis.set_xlabel("Learned − direct BIM AbsRel (negative is better)")
    forest_axis.set_title("Paired room bootstrap: mean difference, 95% CI")
    forest_axis.grid(axis="x", color="#D8DEE9", linewidth=0.7)
    all_values = [
        float(value)
        for row in bootstrap_rows
        for value in (
            row["confidence_interval_95"][0],
            row["confidence_interval_95"][1],
        )
    ]
    span = max(all_values) - min(all_values)
    forest_axis.set_xlim(min(all_values) - 0.15 * span, max(all_values) + 0.75 * span)
    repetitions = int(bootstrap_rows[0]["bootstrap_repetitions"])
    seed = int(bootstrap_rows[0]["seed"])
    figure.suptitle(
        f"Area 1 room-disjoint blind test | paired bootstrap: {repetitions:,} resamples, seed {seed}",
        y=1.01,
    )
    figure.tight_layout()
    return figure


def _deterministic_ablation_rows(ablation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = {
        "scale_only": "Remove local stage",
        "no_q25_cap": "Remove Q25 cap",
        "wide_ratio_bounds": "Widen ratio bounds",
        "min_samples_1": "Min support: 1",
        "no_consistency_gate": "Remove consistency gate",
        "no_edge_gate": "Remove Sobel edge gate",
        "no_gaussian_propagation": "Remove Gaussian propagation",
        "no_support_cutoff": "Remove support cutoff",
        "alpha_1_0": "Local multiplier: 1.0",
    }
    rows: dict[str, list[dict[str, Any]]] = {}
    for dataset_key, dataset_label in (
        ("slabim", "SLABIM validation"),
        ("stanford_area1", "Area 1 validation"),
    ):
        bootstrap = ablation["datasets"][dataset_key]["paired_group_bootstrap_abs_rel"]["all"]
        rows[dataset_label] = [
            {
                "variant": variant,
                "label": labels[variant],
                "mean_difference": bootstrap[variant]["mean_difference"],
                "confidence_interval_95": bootstrap[variant]["confidence_interval_95"],
                "groups": bootstrap[variant]["groups"],
            }
            for variant in labels
        ]
    return rows


def render_deterministic_ablation(rows: dict[str, list[dict[str, Any]]]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 6.0), sharey=True)
    for axis, (dataset, dataset_rows) in zip(axes, rows.items()):
        y = np.arange(len(dataset_rows), dtype=np.float64)
        for index, row in enumerate(dataset_rows):
            mean = float(row["mean_difference"])
            low, high = (float(value) for value in row["confidence_interval_95"])
            color = COLORS["learned"] if mean > 0 else COLORS["worse"]
            axis.errorbar(
                mean,
                index,
                xerr=np.asarray([[mean - low], [high - mean]]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=6,
                linewidth=1.8,
            )
        axis.axvline(0.0, color=COLORS["ink"], linestyle="--", linewidth=1)
        axis.set_yticks(y, [str(row["label"]) for row in dataset_rows])
        axis.invert_yaxis()
        axis.set_xlabel("Variant − full group AbsRel (positive: factor helped)")
        axis.set_title(dataset)
        axis.grid(axis="x", color="#D8DEE9", linewidth=0.7)
    figure.suptitle(
        "Post-hoc deterministic BIM-direct factor ablation (validation only, 95% group bootstrap CI)",
        y=1.01,
    )
    figure.tight_layout()
    return figure


def _cap_number(value: Any) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "+inf", "infinity"}:
        return float("inf")
    return float(value)


def _cap_label(value: Any) -> str:
    numeric = _cap_number(value)
    return "∞" if math.isinf(numeric) else f"{numeric:g}"


def _candidate_rows(scale_selection: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in scale_selection["candidate_results"]:
        summary = candidate["scale_summary"]
        frames = int(summary["frames"])
        cap_events = int(summary["q10_cap_triggered_frames"]) + int(
            summary["q25_cap_triggered_frames"]
        )
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "q10_log_cap": _cap_number(candidate["q10_log_cap"]),
                "q25_log_cap": _cap_number(candidate["q25_log_cap"]),
                "room_macro_abs_rel": candidate["selection_objective"]["room_macro_abs_rel"],
                "furniture_room_macro_abs_rel": candidate["selection_objective"][
                    "furniture_room_macro_abs_rel"
                ],
                "frames": frames,
                "cap_events_per_frame": cap_events / max(frames, 1),
                "q10_cap_triggered_frames": summary["q10_cap_triggered_frames"],
                "q25_cap_triggered_frames": summary["q25_cap_triggered_frames"],
            }
        )
    return rows


def _selected_caps(scale_selection: dict[str, Any]) -> tuple[float, float]:
    selected = scale_selection["final_selection"]["canonical_scale_estimator"]
    return _cap_number(selected["q10_log_cap"]), _cap_number(selected["q25_log_cap"])


def render_scale_heatmap(
    candidates: list[dict[str, Any]], selected_caps: tuple[float, float]
) -> Figure:
    q10_values = sorted({float(row["q10_log_cap"]) for row in candidates})
    q25_values = sorted({float(row["q25_log_cap"]) for row in candidates})
    matrix = np.full((len(q25_values), len(q10_values)), np.nan, dtype=np.float64)
    lookup = {
        (float(row["q25_log_cap"]), float(row["q10_log_cap"])): float(row["room_macro_abs_rel"])
        for row in candidates
    }
    for row_index, q25 in enumerate(q25_values):
        for column_index, q10 in enumerate(q10_values):
            matrix[row_index, column_index] = lookup.get((q25, q10), np.nan)
    figure, axis = plt.subplots(figsize=(10.6, 5.7))
    image = axis.imshow(matrix, cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(q10_values)), [_cap_label(value) for value in q10_values])
    axis.set_yticks(range(len(q25_values)), [_cap_label(value) for value in q25_values])
    axis.set_xlabel("q10 log upper cap")
    axis.set_ylabel("q25 log upper cap")
    axis.set_title(
        f"Train-only robust-scale sensitivity ({len(candidates)} candidates; room-macro AbsRel)"
    )
    finite = matrix[np.isfinite(matrix)]
    threshold = float(np.median(finite))
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if not np.isfinite(value):
                continue
            color = "white" if value >= threshold else COLORS["ink"]
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=color,
                fontsize=7.7,
            )
    selected_q10, selected_q25 = selected_caps
    selected_column = q10_values.index(selected_q10)
    selected_row = q25_values.index(selected_q25)
    axis.scatter(
        [selected_column],
        [selected_row],
        marker="*",
        s=260,
        facecolor="none",
        edgecolor="#E639B5",
        linewidth=2.2,
        label="selected on train only",
    )
    axis.legend(frameon=True, facecolor="white", loc="upper right")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.03)
    colorbar.set_label("Room-macro AbsRel ↓")
    figure.text(
        0.5,
        0.01,
        "Validation and test samples opened during cap selection: 0",
        ha="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def pareto_frontier_indices(points: Sequence[tuple[float, float]]) -> list[int]:
    """Return non-dominated indices for a two-objective minimization problem."""

    frontier = []
    for index, (x_value, y_value) in enumerate(points):
        dominated = False
        for other_index, (other_x, other_y) in enumerate(points):
            if other_index == index:
                continue
            no_worse = other_x <= x_value and other_y <= y_value
            strictly_better = other_x < x_value or other_y < y_value
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(index)
    return sorted(frontier, key=lambda item: (points[item][0], points[item][1]))


def render_scale_pareto(
    candidates: list[dict[str, Any]], selected_caps: tuple[float, float]
) -> Figure:
    points = [
        (float(row["room_macro_abs_rel"]), float(row["furniture_room_macro_abs_rel"]))
        for row in candidates
    ]
    cap_events = np.asarray(
        [float(row["cap_events_per_frame"]) for row in candidates], dtype=np.float64
    )
    frontier = pareto_frontier_indices(points)
    figure, axis = plt.subplots(figsize=(7.8, 6.0))
    scatter = axis.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        c=cap_events,
        cmap="plasma",
        s=58,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    if frontier:
        axis.plot(
            [points[index][0] for index in frontier],
            [points[index][1] for index in frontier],
            color=COLORS["ink"],
            linestyle="--",
            linewidth=1.3,
            label="non-dominated frontier",
        )
    selected_q10, selected_q25 = selected_caps
    selected_index = next(
        index
        for index, row in enumerate(candidates)
        if float(row["q10_log_cap"]) == selected_q10 and float(row["q25_log_cap"]) == selected_q25
    )
    selected_point = points[selected_index]
    axis.scatter(
        [selected_point[0]],
        [selected_point[1]],
        marker="*",
        s=300,
        color="#E639B5",
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
        label=f"selected: q10={_cap_label(selected_q10)}, q25={_cap_label(selected_q25)}",
    )
    axis.annotate(
        "Lower is better on both axes",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        color=COLORS["muted"],
    )
    axis.set_xlabel("All-pixel train room-macro AbsRel ↓")
    axis.set_ylabel("Furniture train room-macro AbsRel ↓")
    axis.set_title(f"Train-only robust-scale trade-off ({len(candidates)} candidates)")
    axis.grid(color="#D8DEE9", linewidth=0.7, alpha=0.8)
    axis.legend(frameon=False, loc="upper right")
    colorbar = figure.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Cap trigger events / frame")
    figure.tight_layout()
    return figure


def _history_points(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "epoch": int(row["epoch"]),
            "val_refined_abs_rel": float(row["val_refined_abs_rel"]),
            "val_anchor_abs_rel": float(row["val_anchor_abs_rel"]),
            "accepted": bool(row.get("accepted", False)),
        }
        for row in history
    ]


def render_training_curves(
    histories: dict[str, dict[str, list[dict[str, Any]]]],
    _area1_e2e_promoted: bool | None = None,
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.7))
    for axis, (dataset, runs) in zip(axes, histories.items()):
        for run_name, color, marker in (("frozen", COLORS["scale"], "o"),):
            points = runs[run_name]
            axis.plot(
                [point["epoch"] for point in points],
                [point["val_refined_abs_rel"] for point in points],
                color=color,
                marker=marker,
                markersize=4,
                linewidth=1.8,
                label="Universal learned refiner",
            )
            best = min(points, key=lambda point: point["val_refined_abs_rel"])
            axis.scatter(
                [best["epoch"]],
                [best["val_refined_abs_rel"]],
                s=95,
                facecolor="none",
                edgecolor=color,
                linewidth=1.8,
                zorder=4,
            )
        anchor = float(runs["frozen"][0]["val_anchor_abs_rel"])
        axis.axhline(
            anchor,
            color=COLORS["bim"],
            linestyle="--",
            linewidth=1.4,
            label="Direct BIM validation baseline",
        )
        axis.set_xlabel("Epoch")
        axis.set_title(dataset)
        axis.grid(color="#D8DEE9", linewidth=0.7, alpha=0.8)
    axes[0].set_ylabel("Validation AbsRel ↓")
    axes[0].legend(frameon=False, fontsize=8.5)
    figure.suptitle("Universal-model validation trajectories (no test-set selection)", y=1.02)
    figure.tight_layout()
    return figure


def _diagram_box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str | None = None,
    fontsize: float = 10,
) -> FancyBboxPatch:
    edge = edgecolor or COLORS["ink"]
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edge,
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return patch


def _diagram_arrow(
    axis: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "#536273",
    style: str = "-",
    bend: float = 0.0,
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": 1.5,
            "linestyle": style,
            "connectionstyle": f"arc3,rad={bend}",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _diagram_canvas(figsize: tuple[float, float], title: str) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(figsize=figsize)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.text(0.01, 0.97, title, ha="left", va="top", fontsize=13, fontweight="bold")
    return figure, axis


def render_process_candidate_a() -> Figure:
    figure, axis = _diagram_canvas(
        (14.0, 5.2), "Candidate A — concise method pipeline (paper main figure)"
    )
    input_specs = (
        (0.69, "RGB image", "#DCEBFA"),
        (0.43, "Raw DA3 depth", "#E2E8F0"),
        (0.17, "Fixed BIM\nenvelope", "#FDE9C9"),
    )
    for y, text, color in input_specs:
        _diagram_box(axis, 0.03, y, 0.13, 0.14, text, facecolor=color)
    _diagram_box(
        axis,
        0.23,
        0.34,
        0.15,
        0.22,
        "Robust metric-scale\ncalibration\n(train-selected caps)",
        facecolor="#DCEBFA",
    )
    _diagram_box(
        axis,
        0.45,
        0.58,
        0.14,
        0.17,
        "RGB + DA3 geometry\ncondition encoders",
        facecolor="#D9F1ED",
    )
    _diagram_box(
        axis,
        0.45,
        0.23,
        0.14,
        0.17,
        "BIM geometry\nfeature encoder",
        facecolor="#FDE9C9",
    )
    _diagram_box(
        axis,
        0.65,
        0.38,
        0.13,
        0.20,
        "Multiscale fusion\n+ BIM adapters",
        facecolor="#E8E0F2",
    )
    _diagram_box(
        axis,
        0.83,
        0.38,
        0.13,
        0.20,
        "Frame + low + detail\nbounded log-residual\nanchor × exp(r)",
        facecolor="#CDEDE5",
    )
    _diagram_arrow(axis, (0.16, 0.50), (0.23, 0.47))
    _diagram_arrow(axis, (0.16, 0.24), (0.23, 0.40))
    _diagram_arrow(axis, (0.16, 0.76), (0.45, 0.68), bend=-0.08)
    _diagram_arrow(axis, (0.38, 0.47), (0.45, 0.64))
    _diagram_arrow(axis, (0.16, 0.24), (0.45, 0.31))
    _diagram_arrow(axis, (0.59, 0.66), (0.65, 0.52))
    _diagram_arrow(axis, (0.59, 0.31), (0.65, 0.43))
    _diagram_arrow(axis, (0.78, 0.48), (0.83, 0.48))
    axis.text(
        0.305,
        0.10,
        "scale first, then learn residual structure",
        ha="center",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    figure.tight_layout()
    return figure


def render_process_candidate_b() -> Figure:
    figure, axis = _diagram_canvas(
        (14.0, 6.8), "Candidate B — multi-condition network detail (architecture figure)"
    )
    _diagram_box(
        axis,
        0.03,
        0.66,
        0.13,
        0.16,
        "RGB + DA3 geometry\n(separate encoders)",
        facecolor="#DCEBFA",
    )
    _diagram_box(
        axis,
        0.03,
        0.19,
        0.13,
        0.20,
        "BIM depth + valid\n+ normals + edge\n+ disagreement",
        facecolor="#FDE9C9",
    )
    encoder_x = (0.23, 0.36, 0.49)
    for index, x in enumerate(encoder_x, start=1):
        _diagram_box(
            axis,
            x,
            0.66,
            0.09,
            0.16,
            f"Image\nlevel {index}",
            facecolor="#D9F1ED",
            fontsize=9,
        )
        _diagram_box(
            axis,
            x,
            0.21,
            0.09,
            0.16,
            f"BIM\nlevel {index}",
            facecolor="#FFF1D9",
            fontsize=9,
        )
        if index == 1:
            _diagram_arrow(axis, (0.16, 0.74), (x, 0.74))
            _diagram_arrow(axis, (0.16, 0.29), (x, 0.29))
        else:
            previous = encoder_x[index - 2]
            _diagram_arrow(axis, (previous + 0.09, 0.74), (x, 0.74))
            _diagram_arrow(axis, (previous + 0.09, 0.29), (x, 0.29))
    _diagram_box(
        axis,
        0.64,
        0.42,
        0.13,
        0.22,
        "RGB/geometry fusion\n+ additive BIM adapters\n(at every scale)",
        facecolor="#E8E0F2",
    )
    _diagram_arrow(axis, (0.58, 0.74), (0.64, 0.58))
    _diagram_arrow(axis, (0.58, 0.29), (0.64, 0.48))
    heads = (
        (0.73, "Frame residual"),
        (0.48, "Low-frequency residual"),
        (0.23, "Detail residual"),
    )
    for y, text in heads:
        _diagram_box(axis, 0.83, y, 0.13, 0.13, text, facecolor="#CDEDE5", fontsize=9)
        _diagram_arrow(axis, (0.77, 0.53), (0.83, y + 0.065), bend=(y - 0.48) * 0.25)
    axis.text(
        0.70,
        0.12,
        "Depth-aware routing gates the frame residual (and optionally the low residual);\n"
        "BIM reliability is auxiliary supervision, not a multiplicative output gate.",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    figure.tight_layout()
    return figure


def render_process_candidate_c() -> Figure:
    figure, axis = _diagram_canvas(
        (15.0, 5.2), "Candidate C — leakage-safe evaluation protocol (methods/evaluation figure)"
    )
    stages = (
        ("Public data\nSLABIM + Area 1", "#DCEBFA"),
        ("Exclude known bad\nframes before splitting", "#FCE8E6"),
        ("Fixed train/val/test\nroom-disjoint where possible", "#E2E8F0"),
        ("Train-only robust-scale\n48-candidate selection", "#FDE9C9"),
        ("Frozen / E2E training\nvalidation promotion gate", "#E8E0F2"),
        ("One-time blind test\nfixed 0.2–5.0 m support", "#D9F1ED"),
        ("Subsets + paired-room\nbootstrap + failure report", "#CDEDE5"),
    )
    x_positions = np.linspace(0.02, 0.84, len(stages))
    width = 0.12
    for index, ((text, color), x) in enumerate(zip(stages, x_positions)):
        y = 0.38 if index % 2 == 0 else 0.57
        _diagram_box(axis, float(x), y, width, 0.21, text, facecolor=color, fontsize=8.7)
        axis.text(
            float(x) + width / 2,
            y - 0.055,
            f"{index + 1}",
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            bbox={"boxstyle": "circle,pad=0.25", "facecolor": COLORS["ink"], "edgecolor": "none"},
        )
        if index:
            previous_x = float(x_positions[index - 1])
            previous_y = 0.38 if (index - 1) % 2 == 0 else 0.57
            _diagram_arrow(
                axis,
                (previous_x + width, previous_y + 0.105),
                (float(x), y + 0.105),
            )
    # Place the boundary after the validation promotion gate and before the
    # one-time blind test.  The fifth box ends near x=0.687.
    boundary_x = 0.69
    axis.text(
        boundary_x,
        0.18,
        "Selection boundary",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=COLORS["worse"],
    )
    axis.plot(
        [boundary_x, boundary_x],
        [0.27, 0.88],
        color=COLORS["worse"],
        linestyle="--",
        linewidth=1.3,
    )
    axis.text(
        boundary_x - 0.02,
        0.89,
        "train / validation only",
        ha="right",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    axis.text(
        boundary_x + 0.02,
        0.89,
        "locked blind-test reporting",
        ha="left",
        fontsize=8.5,
        color=COLORS["muted"],
    )
    figure.tight_layout()
    return figure


def _figure_record(
    *,
    figure_id: str,
    title: str,
    availability: str,
    evidence_scope: str,
    source_paths: Sequence[str],
    numeric_payload: Any,
    files: list[dict[str, Any]],
    cautions: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": figure_id,
        "title": title,
        "availability": availability,
        "planned": False,
        "evidence_scope": evidence_scope,
        "source_paths": list(source_paths),
        "numeric_payload": _json_safe(numeric_payload),
        "cautions": list(cautions),
        "files": files,
    }


def generate_assets(
    *,
    results_root: Path,
    scale_selection_path: Path,
    output_root: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 300,
) -> dict[str, Any]:
    """Generate all registered quantitative panels and three process candidates."""

    formats = tuple(dict.fromkeys(formats))
    if not formats or any(item not in ALLOWED_FORMATS for item in formats):
        raise ValueError(f"formats must be selected from {sorted(ALLOWED_FORMATS)}")
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    _configure_matplotlib()
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    repository_root = results_root.parent
    registry = SourceRegistry(repository_root)

    paths = {
        "metrics": results_root / "metrics.json",
        "area1_test": results_root / "stanford_area1" / "test_summary.json",
        "deterministic_ablation": (
            results_root / "deterministic_baseline_ablation" / "summary.json"
        ),
        "scale_selection": scale_selection_path.resolve(),
        "slabim_history": results_root / "slabim" / "history.json",
        "area1_history": results_root / "stanford_area1" / "history.json",
    }
    loaded = {name: registry.load_json(path) for name, path in paths.items()}
    references = {name: registry.reference(path) for name, path in paths.items()}
    output_root.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, Any]] = []

    main_values = _main_result_values(loaded["metrics"], loaded["area1_test"])
    figures.append(
        _figure_record(
            figure_id="main_blind_test_absrel",
            title="Registered blind-test AbsRel on SLABIM and 2D-3D-S Area 1",
            availability="existing_result",
            evidence_scope="blind_test_pixel_micro_fixed_0.2_5.0_m",
            source_paths=(references["metrics"], references["area1_test"]),
            numeric_payload=main_values,
            files=_save_figure(
                render_main_results(main_values),
                output_root,
                "quantitative/main_blind_test_absrel",
                formats,
                dpi,
            ),
        )
    )

    deterministic_rows = _deterministic_ablation_rows(loaded["deterministic_ablation"])
    figures.append(
        _figure_record(
            figure_id="deterministic_bim_direct_factor_ablation",
            title="Deterministic BIM-direct leave-one-factor-out ablation",
            availability="diagnostic_existing_result",
            evidence_scope="post_hoc_validation_group_bootstrap",
            source_paths=(references["deterministic_ablation"],),
            numeric_payload={"datasets": deterministic_rows},
            files=_save_figure(
                render_deterministic_ablation(deterministic_rows),
                output_root,
                "quantitative/deterministic_bim_direct_factor_ablation",
                formats,
                dpi,
            ),
            cautions=(
                "Post-hoc validation diagnostic produced after the registered blind tests.",
                "It must not be used to rewrite the frozen v1 test claim without a new protocol.",
            ),
        )
    )

    subset_rows = _subset_values(loaded["area1_test"])
    conflict = subset_rows[-1]["methods"]
    conflict_difference = float(conflict["refined"]["abs_rel"]) - float(
        conflict["robust_bim_direct"]["abs_rel"]
    )
    figures.append(
        _figure_record(
            figure_id="area1_subset_absrel",
            title="Area 1 semantic and BIM-conflict subset performance",
            availability="existing_result",
            evidence_scope="blind_test_pixel_micro_fixed_support",
            source_paths=(references["area1_test"],),
            numeric_payload={
                "subsets": subset_rows,
                "conflict_refined_minus_direct_abs_rel": conflict_difference,
            },
            files=_save_figure(
                render_area1_subsets(subset_rows),
                output_root,
                "quantitative/area1_subset_absrel",
                formats,
                dpi,
            ),
            cautions=(
                "Conflict point estimates improve, but the paired-room AbsRel CI crosses zero.",
                "Pixel-micro bars do not carry room-bootstrap intervals.",
            ),
        )
    )

    room_rows = _room_pairs(loaded["area1_test"])
    bootstrap_rows = _bootstrap_rows(loaded["area1_test"])
    figures.append(
        _figure_record(
            figure_id="area1_room_pairs_and_bootstrap",
            title="Area 1 paired-room slopes and paired bootstrap confidence intervals",
            availability="existing_result",
            evidence_scope="room_disjoint_blind_test_room_macro",
            source_paths=(references["area1_test"],),
            numeric_payload={"room_pairs": room_rows, "paired_bootstrap": bootstrap_rows},
            files=_save_figure(
                render_room_pairs_and_bootstrap(room_rows, bootstrap_rows),
                output_root,
                "quantitative/area1_room_pairs_and_bootstrap",
                formats,
                dpi,
            ),
            cautions=(
                "Confidence intervals summarize resampled held-out rooms, not pixels.",
                (
                    "The conflict-subset interval crosses zero and must not be claimed as a "
                    "significant improvement."
                ),
            ),
        )
    )

    candidates = _candidate_rows(loaded["scale_selection"])
    selected_caps = _selected_caps(loaded["scale_selection"])
    sensitivity_payload = {
        "selection_scope": loaded["scale_selection"]["final_selection"]["selection_scope"],
        "split_isolation": {
            "validation_samples_opened": loaded["scale_selection"]["split_isolation"][
                "validation_samples_opened"
            ],
            "test_samples_opened": loaded["scale_selection"]["split_isolation"][
                "test_samples_opened"
            ],
        },
        "selected_caps": {"q10_log_cap": selected_caps[0], "q25_log_cap": selected_caps[1]},
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    figures.append(
        _figure_record(
            figure_id="area1_train_only_scale_heatmap",
            title="Area 1 train-only robust-scale cap sensitivity heatmap",
            availability="existing_result",
            evidence_scope="train_only_room_macro_grid_search",
            source_paths=(references["scale_selection"],),
            numeric_payload=sensitivity_payload,
            files=_save_figure(
                render_scale_heatmap(candidates, selected_caps),
                output_root,
                "quantitative/area1_train_only_scale_heatmap",
                formats,
                dpi,
            ),
        )
    )
    figures.append(
        _figure_record(
            figure_id="area1_train_only_scale_pareto",
            title="Area 1 train-only all-pixel/furniture robust-scale trade-off",
            availability="existing_result",
            evidence_scope="train_only_room_macro_grid_search",
            source_paths=(references["scale_selection"],),
            numeric_payload={
                **sensitivity_payload,
                "pareto_candidate_ids": [
                    candidates[index]["candidate_id"]
                    for index in pareto_frontier_indices(
                        [
                            (
                                float(row["room_macro_abs_rel"]),
                                float(row["furniture_room_macro_abs_rel"]),
                            )
                            for row in candidates
                        ]
                    )
                ],
            },
            files=_save_figure(
                render_scale_pareto(candidates, selected_caps),
                output_root,
                "quantitative/area1_train_only_scale_pareto",
                formats,
                dpi,
            ),
        )
    )

    histories = {
        "SLABIM": {
            "frozen": _history_points(loaded["slabim_history"]),
        },
        "2D-3D-S Area 1": {
            "frozen": _history_points(loaded["area1_history"]),
        },
    }
    figures.append(
        _figure_record(
            figure_id="registered_training_curves",
            title="Universal-model validation trajectories",
            availability="existing_result",
            evidence_scope="validation_histories_no_test_selection",
            source_paths=(
                references["metrics"],
                references["slabim_history"],
                references["area1_history"],
            ),
            numeric_payload={"histories": histories},
            files=_save_figure(
                render_training_curves(histories),
                output_root,
                "quantitative/registered_training_curves",
                formats,
                dpi,
            ),
            cautions=(
                "Each curve is one deterministic training run; this is not a seed variance plot.",
            ),
        )
    )

    process_specs = (
        (
            "process_candidate_a_method_pipeline",
            "Candidate A: concise method pipeline",
            render_process_candidate_a,
            "process/candidate_a_method_pipeline",
        ),
        (
            "process_candidate_b_dual_stream",
            "Candidate B: dual-stream architecture detail",
            render_process_candidate_b,
            "process/candidate_b_dual_stream_architecture",
        ),
        (
            "process_candidate_c_evaluation_protocol",
            "Candidate C: leakage-safe evaluation protocol",
            render_process_candidate_c,
            "process/candidate_c_evaluation_protocol",
        ),
    )
    for figure_id, title, renderer, relative_stem in process_specs:
        figures.append(
            _figure_record(
                figure_id=figure_id,
                title=title,
                availability="design_candidate",
                evidence_scope="schematic_not_quantitative_evidence",
                source_paths=(),
                numeric_payload={},
                files=_save_figure(
                    renderer(),
                    output_root,
                    relative_stem,
                    formats,
                    dpi,
                ),
                cautions=("Schematic asset; it is not a measured result.",),
            )
        )

    script_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR,
        "generator_sha256": sha256_file(script_path),
        "output_formats": list(formats),
        "depth_protocol_m": loaded["metrics"].get("depth_protocol_m", [0.2, 5.0]),
        "status_legend": {
            "existing_result": "Measured registered result under the stated evidence scope.",
            "diagnostic_existing_result": (
                "Measured post-hoc validation diagnostic; not a registered blind-test result."
            ),
            "historical_existing_result": (
                "Measured historical result that is not the current final protocol."
            ),
            "design_candidate": "Editable visual candidate; not quantitative evidence.",
            "planned": "Experiment is proposed and has no numeric result in this manifest.",
        },
        "sources": registry.records,
        "figures": figures,
        "planned_evaluations_not_plotted": [
            {
                "id": "current_protocol_multiseed_ablation",
                "availability": "planned",
                "planned": True,
                "factors": [
                    "no_RGB",
                    "no_BIM_features",
                    "single_residual_head",
                    "no_depth_routing",
                    "frozen_DA3_vs_E2E_decoder",
                ],
                "reason_not_plotted": "No current-protocol multi-seed result artifact exists.",
            },
            {
                "id": "pose_and_bim_prior_robustness",
                "availability": "planned",
                "planned": True,
                "factors": [
                    "camera_translation_perturbation",
                    "camera_rotation_perturbation",
                    "BIM_coverage_dropout",
                ],
                "reason_not_plotted": "No registered perturbation sweep exists.",
            },
            {
                "id": "compute_data_sensitivity",
                "availability": "planned",
                "planned": True,
                "factors": ["image_resolution", "training_fraction", "batch_size"],
                "reason_not_plotted": "No registered factorial sweep exists.",
            },
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate_assets(
        results_root=args.results_root,
        scale_selection_path=args.scale_selection,
        output_root=args.output,
        formats=args.formats,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "figure_count": len(manifest["figures"]),
                "formats": manifest["output_formats"],
                "manifest": str((args.output / "manifest.json").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
