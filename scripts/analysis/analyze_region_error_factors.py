#!/usr/bin/env python3
"""Analyze why held-out depth errors differ across SLABIM regions.

The script joins the registered seed-42 cross-validation results with prepared
sample geometry, RGB statistics, and pose-recovery diagnostics. It performs no
training and does not alter registered evaluation artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bim_priorda3.baselines import (
    bim_scale_and_local_features,
    estimate_bim_scale,
)

REGION_ORDER = (
    "3F_Region2",
    "3F_Region3",
    "4F_Region2",
    "4F_Region3",
    "5F_Region2",
    "5F_Region3",
)
DEPTH_BINS = (0.2, 1.0, 2.0, 3.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/slabim_504_r50/manifest.jsonl"),
    )
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=Path("outputs/slabim_region_cv"),
    )
    parser.add_argument(
        "--slabim-root",
        type=Path,
        default=Path("../SLABIM"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/region_error_analysis"),
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records[record["id"]] = record
    return records


def load_evaluation_rows(cv_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = "fold_*/seed_42/evaluation_test/per_frame.csv"
    for path in sorted((cv_root / "folds").glob(pattern)):
        with path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                row: dict[str, Any] = {}
                for key, value in raw.items():
                    if key in {"sample_id", "region"}:
                        row[key] = value
                    elif key == "frame_index":
                        row[key] = int(float(value))
                    else:
                        row[key] = float(value)
                row["evaluation_csv"] = str(path)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No seed-42 evaluation rows under {cv_root}")
    return rows


def rgb_statistics(path: Path) -> tuple[float, float, float, float]:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return (float("nan"),) * 4
    small = cv2.resize(bgr, (160, 90), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return (
        float(gray.mean()),
        float(gray.std()),
        float(hsv[..., 1].mean() / 255.0),
        float(cv2.Laplacian(gray, cv2.CV_32F).var()),
    )


def pose_diagnostics(
    slabim_root: Path,
) -> dict[str, dict[str, np.ndarray]]:
    diagnostics = {}
    for region in REGION_ORDER:
        path = (
            slabim_root
            / "sensor_data"
            / region
            / "points"
            / "lidar_pose_local_to_slam.diagnostics.npz"
        )
        with np.load(path) as item:
            diagnostics[region] = {
                "fitness": item["fitness"].copy(),
                "rmse": item["rmse"].copy(),
                "recovered": item["recovered"].copy(),
            }
    return diagnostics


def finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def analyze_frame(
    evaluation: dict[str, Any],
    record: dict[str, Any],
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, float]]:
    sample_path = Path(record["sample"])
    with np.load(sample_path) as item:
        base = item["base_depth"].astype(np.float32)
        confidence = item["base_confidence"].astype(np.float32)
        bim = item["bim_depth"].astype(np.float32)
        bim_valid = item["bim_valid"].astype(bool)
        bim_edge = item["bim_edge"].astype(bool)
        gt = item["gt_depth"].astype(np.float32)
        gt_valid = item["gt_valid"].astype(bool) & np.isfinite(gt) & (gt > 0)
        gt_support = item["gt_support"].astype(np.float32)

    scaled, direct, field, support, scale = bim_scale_and_local_features(base, bim)
    gt_for_scale = np.where(gt_valid, gt, 0.0)
    oracle_scale = estimate_bim_scale(base, gt_for_scale)
    scale_mismatch_abs_log = abs(float(np.log(max(scale, 1e-6) / max(oracle_scale, 1e-6))))
    overlap = gt_valid & bim_valid & (bim > 0)
    valid_bim = bim_valid & (bim > 0) & np.isfinite(bim)
    safe_bim = np.maximum(bim, 1e-6)
    safe_scaled = np.maximum(scaled, 1e-6)
    log_disagreement = np.abs(np.log(safe_bim) - np.log(safe_scaled))
    gradient_x = cv2.Sobel(bim, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(bim, cv2.CV_32F, 0, 1, ksize=3)
    seed_mask = (
        valid_bim
        & np.isfinite(log_disagreement)
        & (log_disagreement <= 0.10)
        & (np.hypot(gradient_x, gradient_y) < 0.25)
    )
    gt_count = int(gt_valid.sum())
    overlap_count = int(overlap.sum())
    bim_count = int(valid_bim.sum())
    total_pixels = int(gt.size)
    bin_counts = {}
    for lower, upper in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
        bin_counts[f"gt_{lower:g}_{upper:g}_count"] = int(
            (gt_valid & (gt >= lower) & (gt < upper)).sum()
        )

    if overlap_count:
        bim_abs_rel_sum = float(
            (np.abs(bim[overlap] - gt[overlap]) / np.maximum(gt[overlap], 1e-6)).sum()
        )
        bim_log_error_sum = float(
            np.abs(
                np.log(np.maximum(bim[overlap], 1e-6)) - np.log(np.maximum(gt[overlap], 1e-6))
            ).sum()
        )
        scaled_log_error = np.abs(
            np.log(np.maximum(scaled[overlap], 1e-6)) - np.log(np.maximum(gt[overlap], 1e-6))
        )
        bim_log_error = np.abs(
            np.log(np.maximum(bim[overlap], 1e-6)) - np.log(np.maximum(gt[overlap], 1e-6))
        )
        bim_wins = int((bim_log_error < scaled_log_error).sum())
    else:
        bim_abs_rel_sum = 0.0
        bim_log_error_sum = 0.0
        bim_wins = 0

    rgb_mean, rgb_std, rgb_saturation, rgb_laplacian = rgb_statistics(Path(record["image"]))
    frame_index = int(evaluation["frame_index"])
    pose = diagnostics[evaluation["region"]]
    pose_fitness = (
        float(pose["fitness"][frame_index])
        if 0 <= frame_index < len(pose["fitness"])
        else float("nan")
    )
    pose_rmse = (
        float(pose["rmse"][frame_index]) if 0 <= frame_index < len(pose["rmse"]) else float("nan")
    )
    direct_abs_rel = float(evaluation["previous_scale_local_abs_rel"])
    refined_abs_rel = float(evaluation["refined_abs_rel"])
    scaled_abs_rel = float(evaluation["global_scale_abs_rel"])
    row = {
        "sample_id": evaluation["sample_id"],
        "region": evaluation["region"],
        "image": record["image"],
        "sample": record["sample"],
        "frame_index": frame_index,
        "time_difference_ms": abs(float(record.get("time_difference_s", np.nan))) * 1000.0,
        "valid_pixels": gt_count,
        "gt_coverage": gt_count / total_pixels,
        "gt_mean_m": finite_mean(gt[gt_valid]),
        "gt_median_m": float(np.median(gt[gt_valid])) if gt_count else float("nan"),
        "gt_support_mean": finite_mean(gt_support[gt_valid]),
        "bim_pixels": bim_count,
        "bim_coverage": bim_count / total_pixels,
        "bim_gt_overlap_pixels": overlap_count,
        "bim_gt_overlap_fraction": overlap_count / max(gt_count, 1),
        "bim_abs_rel_overlap": bim_abs_rel_sum / max(overlap_count, 1),
        "bim_mean_abs_log_error": bim_log_error_sum / max(overlap_count, 1),
        "bim_win_fraction": bim_wins / max(overlap_count, 1),
        "scale": scale,
        "abs_log_scale": abs(float(np.log(max(scale, 1e-6)))),
        "oracle_scale": oracle_scale,
        "scale_mismatch_abs_log": scale_mismatch_abs_log,
        "bim_scaled_abs_log_disagreement": finite_mean(log_disagreement[valid_bim]),
        "direct_seed_fraction": int(seed_mask.sum()) / max(bim_count, 1),
        "anchor_support_mean": float(support.mean()),
        "anchor_field_abs_mean": float(np.abs(field).mean()),
        "direct_correction_abs_log_mean": float(
            np.abs(np.log(np.maximum(direct, 1e-6) / np.maximum(scaled, 1e-6))).mean()
        ),
        "bim_edge_fraction": float(bim_edge.mean()),
        "base_confidence_gt_mean": finite_mean(confidence[gt_valid]),
        "rgb_luminance_mean": rgb_mean,
        "rgb_luminance_std": rgb_std,
        "rgb_saturation_mean": rgb_saturation,
        "rgb_laplacian_variance": rgb_laplacian,
        "pose_recovery_fitness": pose_fitness,
        "pose_recovery_rmse_m": pose_rmse,
        "base_abs_rel": float(evaluation["base_abs_rel"]),
        "scaled_abs_rel": scaled_abs_rel,
        "direct_abs_rel": direct_abs_rel,
        "refined_abs_rel": refined_abs_rel,
        "direct_gain_over_scaled": scaled_abs_rel - direct_abs_rel,
        "learned_gain_over_direct": direct_abs_rel - refined_abs_rel,
        "learned_relative_gain": (direct_abs_rel - refined_abs_rel) / max(direct_abs_rel, 1e-9),
        "learned_frame_trust": float(evaluation["learned_frame_trust"]),
        "learned_mean_pixel_trust": float(evaluation["learned_mean_pixel_trust"]),
        "learned_mean_variance": float(evaluation["learned_mean_variance"]),
        "learned_mean_abs_log_residual": float(evaluation["learned_mean_abs_log_residual"]),
        **bin_counts,
    }
    sums = {
        "gt_count": gt_count,
        "total_pixels": total_pixels,
        "bim_count": bim_count,
        "overlap_count": overlap_count,
        "bim_abs_rel_sum": bim_abs_rel_sum,
        "bim_log_error_sum": bim_log_error_sum,
        "bim_wins": bim_wins,
        "seed_count": int(seed_mask.sum()),
        "support_sum": float(support.sum()),
        "field_abs_sum": float(np.abs(field).sum()),
        **bin_counts,
    }
    return row, sums


def region_summary(
    frame_df: pd.DataFrame,
    region_sums: dict[str, dict[str, float]],
    cv_root: Path,
) -> pd.DataFrame:
    rows = []
    for region in REGION_ORDER:
        group = frame_df[frame_df.region == region]
        sums = region_sums[region]
        summary_path = next(
            (cv_root / "folds").glob(f"fold_*_{region}/seed_42/evaluation_test/summary.json")
        )
        evaluation = json.loads(summary_path.read_text())
        overall = evaluation["overall"]
        row: dict[str, Any] = {
            "region": region,
            "frames": len(group),
            "valid_pixels": int(sums["gt_count"]),
            "base_abs_rel": overall["base"]["abs_rel"],
            "scaled_abs_rel": overall["global_scale"]["abs_rel"],
            "direct_abs_rel": overall["previous_scale_local"]["abs_rel"],
            "refined_abs_rel": overall["refined"]["abs_rel"],
            "direct_gain_over_scaled_pct": (
                overall["global_scale"]["abs_rel"] - overall["previous_scale_local"]["abs_rel"]
            )
            / overall["global_scale"]["abs_rel"]
            * 100.0,
            "learned_gain_over_direct_pct": (
                overall["previous_scale_local"]["abs_rel"] - overall["refined"]["abs_rel"]
            )
            / overall["previous_scale_local"]["abs_rel"]
            * 100.0,
            "frame_macro_direct_abs_rel": group.direct_abs_rel.mean(),
            "frame_macro_refined_abs_rel": group.refined_abs_rel.mean(),
            "frame_win_fraction": (group.refined_abs_rel < group.direct_abs_rel).mean(),
            "bim_gt_overlap_fraction": sums["overlap_count"] / max(sums["gt_count"], 1),
            "bim_abs_rel_overlap": sums["bim_abs_rel_sum"] / max(sums["overlap_count"], 1),
            "bim_mean_abs_log_error": sums["bim_log_error_sum"] / max(sums["overlap_count"], 1),
            "bim_win_fraction": sums["bim_wins"] / max(sums["overlap_count"], 1),
            "direct_seed_fraction": sums["seed_count"] / max(sums["bim_count"], 1),
            "anchor_support_mean": sums["support_sum"] / max(sums["total_pixels"], 1),
            "anchor_field_abs_mean": sums["field_abs_sum"] / max(sums["total_pixels"], 1),
        }
        for column in (
            "gt_mean_m",
            "gt_median_m",
            "gt_coverage",
            "gt_support_mean",
            "bim_coverage",
            "scale",
            "abs_log_scale",
            "oracle_scale",
            "scale_mismatch_abs_log",
            "bim_scaled_abs_log_disagreement",
            "base_confidence_gt_mean",
            "rgb_luminance_mean",
            "rgb_luminance_std",
            "rgb_saturation_mean",
            "rgb_laplacian_variance",
            "pose_recovery_fitness",
            "pose_recovery_rmse_m",
            "time_difference_ms",
            "learned_mean_abs_log_residual",
        ):
            row[f"{column}_mean"] = group[column].mean()
            row[f"{column}_std"] = group[column].std(ddof=1)
        row["scale_median"] = group.scale.median()
        row["scale_iqr"] = group.scale.quantile(0.75) - group.scale.quantile(0.25)
        row["scale_extreme_fraction"] = ((group.scale < 0.6) | (group.scale > 1.4)).mean()
        row["scale_mismatch_gt_0.45_fraction"] = (group.scale_mismatch_abs_log > 0.45).mean()
        for lower, upper in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:]):
            count = sums[f"gt_{lower:g}_{upper:g}_count"]
            row[f"gt_{lower:g}_{upper:g}_fraction"] = count / max(sums["gt_count"], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(frame_df: pd.DataFrame) -> pd.DataFrame:
    factors = (
        "gt_mean_m",
        "gt_0.2_1_count",
        "gt_3_5_count",
        "gt_coverage",
        "gt_support_mean",
        "bim_gt_overlap_fraction",
        "bim_abs_rel_overlap",
        "bim_mean_abs_log_error",
        "bim_win_fraction",
        "scale",
        "abs_log_scale",
        "scale_mismatch_abs_log",
        "bim_scaled_abs_log_disagreement",
        "direct_seed_fraction",
        "anchor_support_mean",
        "anchor_field_abs_mean",
        "base_confidence_gt_mean",
        "rgb_luminance_mean",
        "rgb_luminance_std",
        "rgb_saturation_mean",
        "rgb_laplacian_variance",
        "pose_recovery_fitness",
        "pose_recovery_rmse_m",
        "time_difference_ms",
    )
    targets = ("direct_abs_rel", "refined_abs_rel", "learned_gain_over_direct")
    rows = []
    working = frame_df.copy()
    working["gt_near_fraction"] = working["gt_0.2_1_count"] / working["valid_pixels"].clip(lower=1)
    working["gt_far_fraction"] = working["gt_3_5_count"] / working["valid_pixels"].clip(lower=1)
    factor_names = [
        "gt_mean_m",
        "gt_near_fraction",
        "gt_far_fraction",
        *factors[3:],
    ]
    for target in targets:
        for factor in factor_names:
            valid = np.isfinite(working[target]) & np.isfinite(working[factor])
            rho, p_value = spearmanr(working.loc[valid, factor], working.loc[valid, target])
            rows.append(
                {
                    "target": target,
                    "factor": factor,
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "frames": int(valid.sum()),
                }
            )
    return pd.DataFrame(rows)


def select_examples(frame_df: pd.DataFrame) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    for region in REGION_ORDER:
        group = frame_df[frame_df.region == region].copy()
        median_direct = group.direct_abs_rel.median()
        representative = group.iloc[(group.direct_abs_rel - median_direct).abs().argmin()]
        best = group.loc[group.learned_gain_over_direct.idxmax()]
        worst = group.loc[group.learned_gain_over_direct.idxmin()]
        selections[region] = {
            "representative": representative.sample_id,
            "best_learned_gain": best.sample_id,
            "worst_learned_gain": worst.sample_id,
            "representative_metrics": {
                "direct_abs_rel": float(representative.direct_abs_rel),
                "refined_abs_rel": float(representative.refined_abs_rel),
            },
            "best_metrics": {
                "direct_abs_rel": float(best.direct_abs_rel),
                "refined_abs_rel": float(best.refined_abs_rel),
            },
            "worst_metrics": {
                "direct_abs_rel": float(worst.direct_abs_rel),
                "refined_abs_rel": float(worst.refined_abs_rel),
            },
        }
    return selections


def plot_overview(region_df: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update({"font.size": 9})
    labels = [region.replace("_Region", "_R") for region in region_df.region]
    colors = ("#8a8f98", "#4e79a7", "#f28e2b", "#59a14f")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    x = np.arange(len(region_df))
    width = 0.19
    for index, (column, label) in enumerate(
        (
            ("base_abs_rel", "Raw DA3"),
            ("scaled_abs_rel", "Global scale"),
            ("direct_abs_rel", "Direct BIM"),
            ("refined_abs_rel", "Learned"),
        )
    ):
        axes[0, 0].bar(
            x + (index - 1.5) * width,
            region_df[column],
            width,
            label=label,
            color=colors[index],
        )
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylabel("Pixel-pooled AbsRel")
    axes[0, 0].set_title("A. Error is region-dependent after identical processing")
    axes[0, 0].legend(ncols=2, frameon=False)

    bottom = np.zeros(len(region_df))
    depth_colors = ("#d7ebf7", "#9ecae1", "#4292c6", "#08519c")
    for color, (lower, upper) in zip(depth_colors, zip(DEPTH_BINS[:-1], DEPTH_BINS[1:])):
        values = region_df[f"gt_{lower:g}_{upper:g}_fraction"].to_numpy()
        axes[0, 1].bar(x, values, bottom=bottom, label=f"{lower:g}-{upper:g} m", color=color)
        bottom += values
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylabel("Fraction of valid GT pixels")
    axes[0, 1].set_title("B. Depth-range composition differs")
    axes[0, 1].legend(ncols=2, frameon=False)

    quality_columns = (
        ("bim_gt_overlap_fraction", "BIM/GT overlap"),
        ("bim_win_fraction", "BIM better than scaled DA3"),
        ("direct_seed_fraction", "Direct-correction seed"),
    )
    qwidth = 0.25
    for index, (column, label) in enumerate(quality_columns):
        axes[1, 0].bar(
            x + (index - 1) * qwidth,
            region_df[column],
            qwidth,
            label=label,
        )
    axes[1, 0].set_xticks(x, labels)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("Fraction")
    axes[1, 0].set_title("C. BIM usefulness differs despite similar RGB")
    axes[1, 0].legend(frameon=False, fontsize=8)

    frame_groups = []
    frame_df = pd.read_csv(output.parent / "frame_factors.csv")
    for region in REGION_ORDER:
        frame_groups.append(frame_df.loc[frame_df.region == region, "scale_mismatch_abs_log"])
    axes[1, 1].boxplot(frame_groups, tick_labels=labels, showfliers=True)
    axes[1, 1].axhline(0.45, color="#d62728", linewidth=0.9, linestyle="--")
    axes[1, 1].set_ylabel("|log(BIM scale / GT-oracle scale)|")
    axes[1, 1].set_title("D. A few trajectory segments have bad scale anchors")

    fig.suptitle(
        "SLABIM seed-42 cross-region error factors (real held-out data)",
        fontsize=13,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_relationships(frame_df: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for index, region in enumerate(REGION_ORDER):
        group = frame_df[frame_df.region == region]
        label = region.replace("_Region", "_R")
        axes[0].scatter(
            group.bim_mean_abs_log_error,
            group.direct_abs_rel,
            s=12,
            alpha=0.55,
            color=palette(index),
            label=label,
        )
        axes[1].scatter(
            group.scale_mismatch_abs_log,
            group.direct_abs_rel,
            s=12,
            alpha=0.55,
            color=palette(index),
        )
        axes[2].scatter(
            group.direct_abs_rel,
            group.refined_abs_rel,
            s=12,
            alpha=0.55,
            color=palette(index),
        )
    axes[0].set_xlabel("Raw BIM mean absolute log error on GT overlap")
    axes[0].set_ylabel("Direct BIM frame AbsRel")
    axes[0].set_title("A. BIM/GT disagreement")
    axes[0].legend(ncols=2, frameon=False, fontsize=8)
    axes[1].set_xlabel("|log(BIM scale / GT-oracle scale)|")
    axes[1].set_ylabel("Direct BIM frame AbsRel")
    axes[1].set_title("B. Extreme scale estimates")
    maximum = float(max(frame_df.direct_abs_rel.max(), frame_df.refined_abs_rel.max()))
    axes[2].plot((0, maximum), (0, maximum), "--", color="black", linewidth=0.8)
    axes[2].set_xlabel("Direct BIM frame AbsRel")
    axes[2].set_ylabel("Learned frame AbsRel")
    axes[2].set_title("C. Learned model helps most hard frames")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest.resolve())
    evaluations = load_evaluation_rows(args.cv_root.resolve())
    diagnostics = pose_diagnostics(args.slabim_root.resolve())
    frame_rows = []
    region_sums: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for index, evaluation in enumerate(evaluations, 1):
        record = manifest[evaluation["sample_id"]]
        row, sums = analyze_frame(evaluation, record, diagnostics)
        frame_rows.append(row)
        for key, value in sums.items():
            region_sums[evaluation["region"]][key] += value
        if index % 50 == 0 or index == len(evaluations):
            print(f"analyzed {index}/{len(evaluations)}", flush=True)

    frame_df = pd.DataFrame(frame_rows)
    frame_df.to_csv(output / "frame_factors.csv", index=False)
    region_df = region_summary(frame_df, region_sums, args.cv_root.resolve())
    region_df.to_csv(output / "region_factor_summary.csv", index=False)
    correlations = correlation_table(frame_df)
    correlations.to_csv(output / "frame_factor_correlations.csv", index=False)
    selections = select_examples(frame_df)
    (output / "example_selections.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plot_overview(region_df, output / "region_error_factors.png")
    plot_relationships(frame_df, output / "frame_error_relationships.png")
    print(region_df.to_string(index=False))
    print("\nStrongest frame-level correlations:")
    for target in ("direct_abs_rel", "refined_abs_rel", "learned_gain_over_direct"):
        view = correlations[correlations.target == target].copy()
        view["magnitude"] = view.spearman_rho.abs()
        print(f"\n{target}")
        print(
            view.sort_values("magnitude", ascending=False)
            .head(8)[["factor", "spearman_rho", "p_value"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
