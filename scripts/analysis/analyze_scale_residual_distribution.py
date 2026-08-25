#!/usr/bin/env python3
"""Diagnose whether post-scale depth residuals are additive or proportional.

This is a validation/train-only model-design diagnostic.  It compares the
learned coarse scale output with the frozen universal scale on the same sampled
GT pixels.  GT is used only after inference to characterize the residual; it is
never passed to the model or scale estimator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from bim_priorda3.checkpoints import (
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_model_config,
)
from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import build_loader, move_batch, seed_everything
from bim_priorda3.models import BIMPriorDA3

SCHEMA_VERSION = 1
DEFAULT_DEPTH_EDGES_M = (0.2, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: float | np.floating[Any]) -> float:
    return float(value)


def _linear_fit(x: np.ndarray, y: np.ndarray, *, through_origin: bool) -> dict[str, float]:
    if x.size < 2 or y.size != x.size:
        raise ValueError("Linear fit needs at least two paired values")
    if through_origin:
        slope = float(np.dot(x, y) / np.dot(x, x))
        intercept = 0.0
    else:
        design = np.column_stack((np.ones_like(x), x))
        intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
        intercept = float(intercept)
        slope = float(slope)
    fitted = intercept + slope * x
    residual = y - fitted
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denominator = float(np.sum(np.square(y - np.mean(y))))
    r_squared = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else 0.0
    return {
        "intercept_m": intercept,
        "slope_m_per_m": slope,
        "rmse_m": rmse,
        "normalized_rmse": rmse / float(np.mean(y)),
        "r_squared": r_squared,
    }


def _power_fit(depth: np.ndarray, absolute_error: np.ndarray) -> dict[str, float]:
    keep = (depth > 0) & (absolute_error > 0)
    if int(keep.sum()) < 2:
        raise ValueError("Power fit needs at least two positive bins")
    log_depth = np.log(depth[keep])
    log_error = np.log(absolute_error[keep])
    design = np.column_stack((np.ones_like(log_depth), log_depth))
    log_coefficient, exponent = np.linalg.lstsq(design, log_error, rcond=None)[0]
    fitted = design @ np.array((log_coefficient, exponent))
    denominator = float(np.sum(np.square(log_error - np.mean(log_error))))
    residual = log_error - fitted
    r_squared = 1.0 - float(np.sum(np.square(residual))) / denominator if denominator > 0 else 0.0
    return {
        "coefficient_m": float(np.exp(log_coefficient)),
        "depth_exponent": float(exponent),
        "log_space_r_squared": r_squared,
        "interpretation_reference": {
            "additive_depth_exponent": 0.0,
            "proportional_depth_exponent": 1.0,
        },
    }


def _bin_statistics(
    gt: np.ndarray,
    prediction: np.ndarray,
    depth_edges: np.ndarray,
) -> list[dict[str, float | int]]:
    correction = gt - prediction
    absolute = np.abs(correction)
    relative = correction / gt
    log_ratio = np.log(gt / prediction)
    rows: list[dict[str, float | int]] = []
    for index, (low, high) in enumerate(zip(depth_edges[:-1], depth_edges[1:], strict=True)):
        if index == len(depth_edges) - 2:
            mask = (gt >= low) & (gt <= high)
        else:
            mask = (gt >= low) & (gt < high)
        count = int(mask.sum())
        if not count:
            continue
        sample_correction = correction[mask]
        sample_absolute = absolute[mask]
        sample_relative = relative[mask]
        sample_log = log_ratio[mask]
        rows.append(
            {
                "depth_low_m": _float(low),
                "depth_high_m": _float(high),
                "count": count,
                "median_gt_depth_m": _float(np.median(gt[mask])),
                "mean_signed_correction_m": _float(np.mean(sample_correction)),
                "median_signed_correction_m": _float(np.median(sample_correction)),
                "mean_absolute_correction_m": _float(np.mean(sample_absolute)),
                "median_absolute_correction_m": _float(np.median(sample_absolute)),
                "p90_absolute_correction_m": _float(np.quantile(sample_absolute, 0.90)),
                "mean_signed_relative_correction": _float(np.mean(sample_relative)),
                "median_signed_relative_correction": _float(np.median(sample_relative)),
                "mean_absolute_relative_correction": _float(np.mean(np.abs(sample_relative))),
                "median_absolute_relative_correction": _float(
                    np.median(np.abs(sample_relative))
                ),
                "mean_signed_log_ratio": _float(np.mean(sample_log)),
                "median_signed_log_ratio": _float(np.median(sample_log)),
                "mean_absolute_log_ratio": _float(np.mean(np.abs(sample_log))),
                "median_absolute_log_ratio": _float(np.median(np.abs(sample_log))),
                "log_ratio_iqr": _float(np.quantile(sample_log, 0.75) - np.quantile(sample_log, 0.25)),
                "additive_correction_iqr_m": _float(
                    np.quantile(sample_correction, 0.75)
                    - np.quantile(sample_correction, 0.25)
                ),
            }
        )
    return rows


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return float(np.std(values) / mean) if mean > 0 else float("nan")


def summarize_residuals(
    gt: np.ndarray,
    prediction: np.ndarray,
    depth_edges: np.ndarray,
) -> dict[str, Any]:
    if gt.ndim != 1 or prediction.shape != gt.shape:
        raise ValueError("gt and prediction must be paired one-dimensional arrays")
    valid = (
        np.isfinite(gt)
        & np.isfinite(prediction)
        & (gt > 0)
        & (prediction > 0)
    )
    if not bool(valid.all()):
        raise ValueError("Residual summary received invalid depth values")
    correction = gt - prediction
    log_ratio = np.log(gt / prediction)
    absolute_correction = np.abs(correction)
    absolute_log_ratio = np.abs(log_ratio)
    correction_quantiles = np.quantile(correction, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    log_quantiles = np.quantile(log_ratio, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    absolute_quantiles = np.quantile(absolute_correction, (0.5, 0.75, 0.9, 0.95, 0.99))
    absolute_log_quantiles = np.quantile(absolute_log_ratio, (0.5, 0.75, 0.9, 0.95, 0.99))
    correction_std = float(np.std(correction))
    log_std = float(np.std(log_ratio))
    correction_centered = correction - np.mean(correction)
    log_centered = log_ratio - np.mean(log_ratio)
    bins = _bin_statistics(gt, prediction, depth_edges)
    if len(bins) < 3:
        raise ValueError("At least three populated depth bins are required")
    bin_depth = np.asarray([row["median_gt_depth_m"] for row in bins], dtype=np.float64)
    bin_mae = np.asarray([row["mean_absolute_correction_m"] for row in bins], dtype=np.float64)
    bin_log_mae = np.asarray([row["mean_absolute_log_ratio"] for row in bins], dtype=np.float64)
    bin_additive_iqr = np.asarray(
        [row["additive_correction_iqr_m"] for row in bins], dtype=np.float64
    )
    bin_log_iqr = np.asarray([row["log_ratio_iqr"] for row in bins], dtype=np.float64)
    additive_constant = float(np.mean(bin_mae))
    additive_rmse = float(np.sqrt(np.mean(np.square(bin_mae - additive_constant))))
    return {
        "sample_count": int(gt.size),
        "global": {
            "mean_signed_correction_m": _float(np.mean(correction)),
            "median_signed_correction_m": _float(np.median(correction)),
            "mae_m": _float(np.mean(absolute_correction)),
            "median_absolute_correction_m": _float(np.median(absolute_correction)),
            "mean_signed_log_ratio": _float(np.mean(log_ratio)),
            "median_signed_log_ratio": _float(np.median(log_ratio)),
            "mean_absolute_log_ratio": _float(np.mean(absolute_log_ratio)),
            "median_absolute_log_ratio": _float(np.median(absolute_log_ratio)),
            "signed_correction_quantiles_m": {
                name: _float(value)
                for name, value in zip(
                    ("q01", "q05", "q25", "q50", "q75", "q95", "q99"),
                    correction_quantiles,
                    strict=True,
                )
            },
            "signed_log_ratio_quantiles": {
                name: _float(value)
                for name, value in zip(
                    ("q01", "q05", "q25", "q50", "q75", "q95", "q99"),
                    log_quantiles,
                    strict=True,
                )
            },
            "absolute_correction_quantiles_m": {
                name: _float(value)
                for name, value in zip(
                    ("q50", "q75", "q90", "q95", "q99"),
                    absolute_quantiles,
                    strict=True,
                )
            },
            "absolute_log_ratio_quantiles": {
                name: _float(value)
                for name, value in zip(
                    ("q50", "q75", "q90", "q95", "q99"),
                    absolute_log_quantiles,
                    strict=True,
                )
            },
            "signed_correction_skewness": _float(
                np.mean(np.power(correction_centered / correction_std, 3))
            ),
            "signed_correction_excess_kurtosis": _float(
                np.mean(np.power(correction_centered / correction_std, 4)) - 3.0
            ),
            "signed_log_ratio_skewness": _float(
                np.mean(np.power(log_centered / log_std, 3))
            ),
            "signed_log_ratio_excess_kurtosis": _float(
                np.mean(np.power(log_centered / log_std, 4)) - 3.0
            ),
        },
        "depth_bins": bins,
        "bin_level_models_for_mean_absolute_meter_error": {
            "additive_constant": {
                "constant_mae_m": additive_constant,
                "rmse_m": additive_rmse,
                "normalized_rmse": additive_rmse / additive_constant,
            },
            "proportional_through_origin": _linear_fit(
                bin_depth,
                bin_mae,
                through_origin=True,
            ),
            "mixed_intercept_plus_depth": _linear_fit(
                bin_depth,
                bin_mae,
                through_origin=False,
            ),
            "power_law": _power_fit(bin_depth, bin_mae),
        },
        "cross_depth_stationarity": {
            "mean_absolute_meter_error_cv": _coefficient_of_variation(bin_mae),
            "mean_absolute_log_ratio_cv": _coefficient_of_variation(bin_log_mae),
            "additive_iqr_cv": _coefficient_of_variation(bin_additive_iqr),
            "log_ratio_iqr_cv": _coefficient_of_variation(bin_log_iqr),
            "lower_cv_is_more_depth_stationary": True,
        },
    }


def _bootstrap_room_exponents(
    room_exponents: list[float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(room_exponents, dtype=np.float64)
    if not values.size:
        raise ValueError("No room exponents to bootstrap")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repetitions, values.size))
    means = values[indices].mean(axis=1)
    return {
        "rooms": int(values.size),
        "mean_depth_exponent": _float(np.mean(values)),
        "median_depth_exponent": _float(np.median(values)),
        "room_cluster_bootstrap_mean_ci95": [
            _float(np.quantile(means, 0.025)),
            _float(np.quantile(means, 0.975)),
        ],
        "rooms_closer_to_proportional_1_than_additive_0": int(
            np.sum(np.abs(values - 1.0) < np.abs(values))
        ),
        "per_room_depth_exponent": [_float(value) for value in values],
    }


def _write_bin_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    rows = []
    for method, summary in summaries.items():
        for row in summary["depth_bins"]:
            rows.append({"method": method, **row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    labels = {
        "universal_scale": "Universal scale",
        "learned_attention_scale": "Learned attention scale",
    }
    colors = {"universal_scale": "#8c8c8c", "learned_attention_scale": "#2171b5"}
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), constrained_layout=True)
    fields = (
        ("mean_absolute_correction_m", "Mean |GT - scaled| (m)"),
        ("mean_absolute_log_ratio", "Mean |log(GT / scaled)|"),
        ("median_signed_correction_m", "Median GT - scaled (m)"),
        ("median_signed_log_ratio", "Median log(GT / scaled)"),
    )
    for axis, (field, ylabel) in zip(axes.flat, fields, strict=True):
        for method, summary in summaries.items():
            bins = summary["depth_bins"]
            x = np.asarray([row["median_gt_depth_m"] for row in bins])
            y = np.asarray([row[field] for row in bins])
            axis.plot(
                x,
                y,
                marker="o",
                linewidth=2,
                markersize=4,
                color=colors.get(method),
                label=labels.get(method, method),
            )
        axis.axhline(0.0, color="#444444", linewidth=0.7, alpha=0.6)
        axis.set_xlabel("GT depth bin median (m)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Area_1 validation: residual after image-level scale correction")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_histograms(
    path: Path,
    gt: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    labels = {
        "universal_scale": "Universal scale",
        "learned_attention_scale": "Learned attention scale",
    }
    colors = {"universal_scale": "#8c8c8c", "learned_attention_scale": "#2171b5"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    corrections = {method: gt - prediction for method, prediction in predictions.items()}
    log_ratios = {method: np.log(gt / prediction) for method, prediction in predictions.items()}
    correction_limit = max(
        float(np.quantile(np.abs(values), 0.99)) for values in corrections.values()
    )
    log_limit = max(float(np.quantile(np.abs(values), 0.99)) for values in log_ratios.values())
    correction_bins = np.linspace(-correction_limit, correction_limit, 181)
    log_bins = np.linspace(-log_limit, log_limit, 181)
    for method in predictions:
        axes[0].hist(
            corrections[method],
            bins=correction_bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=colors.get(method),
            label=labels.get(method, method),
        )
        axes[1].hist(
            log_ratios[method],
            bins=log_bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=colors.get(method),
            label=labels.get(method, method),
        )
    axes[0].set_xlabel("Additive correction GT - scaled (m), central common range")
    axes[0].set_ylabel("Density")
    axes[1].set_xlabel("Multiplicative correction log(GT / scaled), central common range")
    axes[1].set_ylabel("Density")
    for axis in axes:
        axis.axvline(0.0, color="#333333", linewidth=0.8)
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Area_1 validation: sampled post-scale residual distributions")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--pixels-per-frame", type=_positive_int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=_positive_int, default=10_000)
    parser.add_argument("--log-every", type=_positive_int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    cfg = load_config(args.config)
    seed_everything(args.seed)
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    loader = build_loader(
        dataset,
        args.batch_size,
        int(cfg.train.num_workers),
        shuffle=False,
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_evaluation_dataset_provenance(
        state,
        dataset.split_provenance,
        split=args.split,
        allow_cross_dataset=False,
    )
    model_overrides = validate_checkpoint_model_config(state, cfg.model)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    sampled_gt: list[np.ndarray] = []
    sampled_rooms: list[np.ndarray] = []
    sampled_subsets: dict[str, list[np.ndarray]] = defaultdict(list)
    sampled_predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    room_to_index: dict[str, int] = {}
    sampled_frames = 0
    sampled_pixels = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            rooms = [str(value) for value in batch["region"]]
            batch = move_batch(batch, device)
            output = model(batch)
            prediction_tensors = {
                "universal_scale": batch["scaled_depth"],
                "learned_attention_scale": output["coarse_depth"],
            }
            gt_batch = batch["gt_depth"].detach().cpu().numpy()[:, 0]
            valid_batch = (batch["gt_valid"] > 0).detach().cpu().numpy()[:, 0]
            furniture_batch = (batch["furniture_mask"] > 0).detach().cpu().numpy()[:, 0]
            bim_depth_batch = batch["bim_depth"].detach().cpu().numpy()[:, 0]
            bim_valid_batch = (batch["bim_valid"] > 0).detach().cpu().numpy()[:, 0]
            prediction_batch = {
                name: tensor.detach().cpu().numpy()[:, 0]
                for name, tensor in prediction_tensors.items()
            }
            for frame_index, room in enumerate(rooms):
                gt = gt_batch[frame_index]
                mask = (
                    valid_batch[frame_index]
                    & np.isfinite(gt)
                    & (gt >= float(cfg.data.min_depth))
                    & (gt <= float(cfg.data.max_depth))
                )
                for prediction in prediction_batch.values():
                    current = prediction[frame_index]
                    mask &= np.isfinite(current) & (current > 0)
                indices = np.flatnonzero(mask.reshape(-1))
                if not indices.size:
                    continue
                count = min(args.pixels_per_frame, int(indices.size))
                selected = (
                    indices
                    if count == indices.size
                    else rng.choice(indices, size=count, replace=False)
                )
                sampled_gt.append(gt.reshape(-1)[selected].astype(np.float32, copy=False))
                room_index = room_to_index.setdefault(room, len(room_to_index))
                sampled_rooms.append(np.full(count, room_index, dtype=np.int16))
                selected_gt = gt.reshape(-1)[selected]
                selected_bim = bim_depth_batch[frame_index].reshape(-1)[selected]
                selected_bim_valid = bim_valid_batch[frame_index].reshape(-1)[selected] & (
                    selected_bim > 0
                )
                tolerance = np.maximum(0.10, 0.05 * selected_bim)
                sampled_subsets["furniture"].append(
                    furniture_batch[frame_index].reshape(-1)[selected]
                )
                sampled_subsets["bim_foreground_conflict"].append(
                    selected_bim_valid & (selected_gt < selected_bim - tolerance)
                )
                sampled_subsets["bim_consistent"].append(
                    selected_bim_valid & (np.abs(selected_gt - selected_bim) <= tolerance)
                )
                sampled_subsets["bim_no_hit"].append(~selected_bim_valid)
                for name, prediction in prediction_batch.items():
                    sampled_predictions[name].append(
                        prediction[frame_index].reshape(-1)[selected].astype(
                            np.float32,
                            copy=False,
                        )
                    )
                sampled_frames += 1
                sampled_pixels += count
            if (batch_index + 1) % args.log_every == 0:
                print(
                    f"batches={batch_index + 1}/{len(loader)} "
                    f"frames={sampled_frames} sampled_pixels={sampled_pixels}"
                )

    gt = np.concatenate(sampled_gt).astype(np.float64)
    room_indices = np.concatenate(sampled_rooms)
    predictions = {
        name: np.concatenate(values).astype(np.float64)
        for name, values in sampled_predictions.items()
    }
    subset_masks = {
        name: np.concatenate(values).astype(bool)
        for name, values in sampled_subsets.items()
    }
    edges = np.asarray(DEFAULT_DEPTH_EDGES_M, dtype=np.float64)
    summaries = {
        name: summarize_residuals(gt, prediction, edges)
        for name, prediction in predictions.items()
    }
    subset_summaries = {
        method: {
            subset: summarize_residuals(gt[mask], prediction[mask], edges)
            for subset, mask in subset_masks.items()
            if int(mask.sum()) >= 1000
        }
        for method, prediction in predictions.items()
    }
    index_to_room = {value: key for key, value in room_to_index.items()}
    room_results: dict[str, dict[str, Any]] = {}
    room_bootstrap: dict[str, dict[str, Any]] = {}
    for method, prediction in predictions.items():
        exponents = []
        method_rooms = {}
        for room_index in sorted(index_to_room):
            mask = room_indices == room_index
            room_summary = summarize_residuals(gt[mask], prediction[mask], edges)
            exponent = room_summary["bin_level_models_for_mean_absolute_meter_error"][
                "power_law"
            ]["depth_exponent"]
            exponents.append(float(exponent))
            method_rooms[index_to_room[room_index]] = {
                "sample_count": int(mask.sum()),
                "depth_exponent": float(exponent),
                "meter_error_cv": room_summary["cross_depth_stationarity"][
                    "mean_absolute_meter_error_cv"
                ],
                "log_error_cv": room_summary["cross_depth_stationarity"][
                    "mean_absolute_log_ratio_cv"
                ],
            }
        room_results[method] = method_rooms
        room_bootstrap[method] = _bootstrap_room_exponents(
            exponents,
            repetitions=args.bootstrap_repetitions,
            seed=args.seed,
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "depth_bin_statistics.csv"
    plot_path = output / "residual_vs_depth.png"
    histogram_path = output / "residual_histograms.png"
    _write_bin_csv(csv_path, summaries)
    _plot(plot_path, summaries)
    _plot_histograms(histogram_path, gt, predictions)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "validation_or_train_only_model_design_diagnostic",
        "split": args.split,
        "samples": sampled_frames,
        "rooms": len(room_to_index),
        "depth_support_m": [float(cfg.data.min_depth), float(cfg.data.max_depth)],
        "sampling": {
            "type": "deterministic_uniform_without_replacement_within_each_frame",
            "pixels_per_frame_max": args.pixels_per_frame,
            "sampled_pixels": sampled_pixels,
            "seed": args.seed,
            "purpose": "distribution quantiles and model-design diagnosis, not headline metrics",
        },
        "ground_truth_policy": (
            "GT is read only after scale inference to measure residuals; it is not a model "
            "input and is not used for test-time scale fitting"
        ),
        "methods": {
            "universal_scale": "frozen log_upper_cap_v1 output from the prepared batch",
            "learned_attention_scale": "checkpoint coarse_depth before spatial refinement",
        },
        "model_config_overrides": model_overrides,
        "config": str(Path(args.config).expanduser().resolve()),
        "config_sha256": _sha256(Path(args.config).expanduser().resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "summaries": summaries,
        "subset_summaries": subset_summaries,
        "per_room": room_results,
        "room_cluster_bootstrap": room_bootstrap,
        "artifacts": {
            "depth_bin_statistics_csv": csv_path.name,
            "residual_vs_depth_plot": plot_path.name,
            "residual_histograms_plot": histogram_path.name,
        },
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "summary": str(summary_path),
        "sampled_frames": sampled_frames,
        "sampled_pixels": sampled_pixels,
        "learned_power_exponent": summaries["learned_attention_scale"][
            "bin_level_models_for_mean_absolute_meter_error"
        ]["power_law"]["depth_exponent"],
    }, indent=2))


if __name__ == "__main__":
    main()
