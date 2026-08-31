#!/usr/bin/env python3
"""Zero-shot dense BIM early-fusion evaluation on Matterport3D/BIMNet.

The predictor is frozen after Stanford Area_1 training.  A previous benchmark
CSV can be supplied as an immutable frame-selection receipt so that changing
the model cannot silently change the reported Matterport subsets.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from bim_priorda3.config import load_config
from bim_priorda3.models import (
    BIMEarlyFusionDAv2ScaleRegressor,
    BIMEarlyFusionDepthAnythingV2,
    build_bim_condition,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_EVALUATOR = PROJECT_ROOT / "scripts/model/evaluate_matterport_bimnet_full_regression.py"
SPEC = importlib.util.spec_from_file_location("matterport_bimnet_benchmark", LEGACY_EVALUATOR)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - installation guard
    raise RuntimeError(f"Cannot import benchmark helpers from {LEGACY_EVALUATOR}")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)

METRICS = BENCHMARK.METRICS
LINEAR_MICRO_METRICS = BENCHMARK.LINEAR_MICRO_METRICS
PREDICTION_NAMES = ("raw", "learned", "oracle_frame_scale")
DEFAULT_DA3_MODEL = BENCHMARK.DEFAULT_DA3_MODEL
DEFAULT_DA3_REVISION = BENCHMARK.DEFAULT_DA3_REVISION
PROCESS_RES_METHOD = BENCHMARK.PROCESS_RES_METHOD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--bimnet-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--bimnet-scene", default="hxp")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scale-regression",
        action="store_true",
        help=(
            "Evaluate the PriorDA-style DAv2 encoder global-scale regressor "
            "instead of the dense DPT model"
        ),
    )
    parser.add_argument("--benchmark-reference-csv", type=Path)
    parser.add_argument("--da3-model", default=DEFAULT_DA3_MODEL)
    parser.add_argument("--da3-revision", default=DEFAULT_DA3_REVISION)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh-source", choices=("obj", "ifc"), default="obj")
    parser.add_argument("--no-wall-filled", action="store_true")
    parser.add_argument("--gt-min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--bim-min-hit-fraction", type=float, default=0.20)
    parser.add_argument("--bim-min-agree-image-fraction", type=float, default=0.10)
    parser.add_argument("--bim-gt-absolute-tolerance-m", type=float, default=0.10)
    parser.add_argument("--bim-gt-relative-tolerance", type=float, default=0.05)
    parser.add_argument("--bim-aabb-margin-m", type=float, default=0.25)
    parser.add_argument("--ratio-min", type=float, default=0.20)
    parser.add_argument("--ratio-max", type=float, default=5.0)
    parser.add_argument("--ratio-min-support", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    BENCHMARK._validate_args(args)
    if not 0 < args.ratio_min < args.ratio_max:
        raise ValueError("Expected 0 < --ratio-min < --ratio-max")
    if args.ratio_min_support < 1:
        raise ValueError("--ratio-min-support must be positive")
    if args.benchmark_reference_csv and not args.benchmark_reference_csv.is_file():
        raise FileNotFoundError(args.benchmark_reference_csv)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).casefold() == "true"


def _selection(
    recomputed: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    keys = (
        "gt_quality_pass",
        "model_support_pass",
        "bim_applicability_pass",
        "effective_pass",
    )
    if reference is None:
        return dict(recomputed), True
    selected = {key: _as_bool(reference.get(key)) for key in keys}
    selected["filter_reasons"] = str(reference.get("filter_reasons") or "")
    matched = all(bool(recomputed[key]) == bool(selected[key]) for key in keys)
    return selected, matched


def _tensor_batch(
    rgb: np.ndarray,
    base_depth: np.ndarray,
    bim_depth: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32),
        "base_depth": torch.from_numpy(base_depth[None, None].copy()).to(
            device=device, dtype=torch.float32
        ),
        "bim_depth": torch.from_numpy(bim_depth[None, None].copy()).to(
            device=device, dtype=torch.float32
        ),
        "bim_valid": torch.from_numpy((bim_depth > 0)[None, None].copy()).to(
            device=device, dtype=torch.float32
        ),
    }


def evaluate_frame(
    *,
    frame: Any,
    da3_model: Any,
    model: BIMEarlyFusionDepthAnythingV2 | BIMEarlyFusionDAv2ScaleRegressor,
    raycaster: Any,
    bim_stats: Mapping[str, Any],
    reference_row: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    height, width = frame.image_shape
    process_height, process_width, processed_k, focal_scale = BENCHMARK.processed_geometry(
        height,
        width,
        frame.intrinsics,
        args.process_res,
    )
    gt = np.asarray(frame.depth, dtype=np.float32)
    gt_valid = np.isfinite(gt) & (gt > 0)
    base_row: dict[str, Any] = {
        "scene_id": frame.scene_id,
        "panorama_id": frame.panorama_id,
        "frame_id": frame.frame_id,
        "camera_index": frame.camera_index,
        "yaw_index": frame.yaw_index,
        "rgb_path": str(frame.rgb_path),
        "depth_path": str(frame.depth_path),
        "height": height,
        "width": width,
        "process_height": process_height,
        "process_width": process_width,
        "focal_scale": focal_scale,
        "gt_valid_pixels": int(gt_valid.sum()),
        "gt_valid_fraction": float(gt_valid.mean()),
    }
    if not np.any(gt_valid):
        reference_selection, selection_match = _selection(
            {
                "gt_quality_pass": False,
                "model_support_pass": False,
                "bim_applicability_pass": False,
                "effective_pass": False,
                "filter_reasons": "gt_zero_depth",
            },
            reference_row,
        )
        return {
            **base_row,
            **reference_selection,
            "selection_matches_recomputation": selection_match,
            "status": "skipped_bad_gt",
            "error": "Matterport depth contains no finite positive pixels",
        }

    render_start = time.perf_counter()
    bim_depth = raycaster.depth(
        processed_k,
        frame.world_to_camera,
        process_width,
        process_height,
    )
    render_seconds = time.perf_counter() - render_start
    bim_hit = bim_depth > 0
    gt_process = cv2.resize(
        gt,
        (process_width, process_height),
        interpolation=cv2.INTER_NEAREST,
    )
    gt_process_valid = np.isfinite(gt_process) & (gt_process > 0)
    overlap = bim_hit & gt_process_valid
    tolerance = np.maximum(
        float(args.bim_gt_absolute_tolerance_m),
        float(args.bim_gt_relative_tolerance) * gt_process,
    )
    agreement = overlap & (np.abs(bim_depth - gt_process) <= tolerance)
    image_pixels = int(bim_depth.size)
    overlap_count = int(overlap.sum())
    agreement_count = int(agreement.sum())
    gt_process_pixels = int(gt_process_valid.sum())
    camera_position = np.asarray(frame.camera_position, dtype=np.float64)
    camera_in_aabb = raycaster.contains_camera(
        camera_position,
        float(args.bim_aabb_margin_m),
    )
    overlap_ratio = bim_depth[overlap] / gt_process[overlap] if overlap_count else np.array([])

    da3_start = time.perf_counter()
    with torch.inference_mode():
        da3_output = da3_model.inference(
            [str(frame.rgb_path)],
            process_res=args.process_res,
            process_res_method=PROCESS_RES_METHOD,
            export_dir=None,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    da3_seconds = time.perf_counter() - da3_start
    canonical_depth = np.asarray(da3_output.depth[0], dtype=np.float32)
    expected_shape = (process_height, process_width)
    if canonical_depth.shape != expected_shape:
        raise RuntimeError(f"DA3 output {canonical_depth.shape} != expected {expected_shape}")
    base_depth = canonical_depth * focal_scale

    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image: {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (process_width, process_height), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0
    batch = _tensor_batch(rgb, base_depth, bim_depth, next(model.parameters()).device)
    condition = build_bim_condition(
        batch,
        bim_log_mean=float(bim_stats["mean"]),
        bim_log_std=float(bim_stats["std"]),
    )
    dense_start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=next(model.parameters()).device.type,
        dtype=torch.float16,
        enabled=next(model.parameters()).device.type == "cuda",
    ):
        if args.scale_regression:
            if not isinstance(model, BIMEarlyFusionDAv2ScaleRegressor):
                raise TypeError("--scale-regression requires the DAv2 scale model")
            model_output = model(batch["rgb"], condition, batch["base_depth"])
            prediction = model_output["scaled_depth"]
            learned_log_scale = float(model_output["log_scale"].detach().float().item())
            learned_scale = float(model_output["scale"].detach().float().item())
        else:
            if not isinstance(model, BIMEarlyFusionDepthAnythingV2):
                raise TypeError("Dense evaluation requires the DAv2 DPT model")
            prediction = model(batch["rgb"], condition)
            learned_log_scale = float("nan")
            learned_scale = float("nan")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dense_seconds = time.perf_counter() - dense_start
    dense_process = prediction.detach().float().squeeze().cpu().numpy()
    if dense_process.shape != expected_shape:
        raise RuntimeError(f"Dense output {dense_process.shape} != expected {expected_shape}")

    base_full = cv2.resize(base_depth, (width, height), interpolation=cv2.INTER_LINEAR)
    learned_full = cv2.resize(dense_process, (width, height), interpolation=cv2.INTER_LINEAR)
    gt_values = gt[gt_valid].astype(np.float64)
    base_values = base_full[gt_valid].astype(np.float64)
    oracle_scale = float(np.median(gt_values / base_values))
    oracle_full = base_full * oracle_scale

    ratio = np.zeros_like(base_depth, dtype=np.float32)
    positive = bim_hit & np.isfinite(base_depth) & (base_depth > 1e-3)
    ratio[positive] = bim_depth[positive] / base_depth[positive]
    ratio_support = positive & (ratio > args.ratio_min) & (ratio < args.ratio_max)
    ratio_support_pixels = int(ratio_support.sum())
    gt_quality_pass = bool(gt_valid.mean() >= args.gt_min_valid_fraction)
    hit_fraction = float(bim_hit.mean())
    agreement_image_fraction = agreement_count / image_pixels
    model_support_pass = bool(ratio_support_pixels >= args.ratio_min_support)
    bim_applicability_pass = bool(
        camera_in_aabb
        and hit_fraction >= args.bim_min_hit_fraction
        and agreement_image_fraction >= args.bim_min_agree_image_fraction
        and model_support_pass
    )
    reasons = []
    if not gt_quality_pass:
        reasons.append("sparse_gt")
    if not camera_in_aabb:
        reasons.append("camera_outside_bim_aabb")
    if hit_fraction < args.bim_min_hit_fraction:
        reasons.append("low_bim_hit")
    if agreement_image_fraction < args.bim_min_agree_image_fraction:
        reasons.append("low_bim_gt_agreement")
    if not model_support_pass:
        reasons.append("insufficient_bim_da3_ratio_support")
    recomputed_selection = {
        "gt_quality_pass": gt_quality_pass,
        "model_support_pass": model_support_pass,
        "bim_applicability_pass": bim_applicability_pass,
        "effective_pass": bool(gt_quality_pass and bim_applicability_pass),
        "filter_reasons": ";".join(reasons),
    }
    selected, selection_match = _selection(recomputed_selection, reference_row)

    return {
        **base_row,
        "camera_x": float(camera_position[0]),
        "camera_y": float(camera_position[1]),
        "camera_z": float(camera_position[2]),
        "camera_in_bim_aabb": camera_in_aabb,
        "bim_hit_pixels": int(bim_hit.sum()),
        "bim_hit_fraction": hit_fraction,
        "bim_gt_overlap_pixels": overlap_count,
        "bim_gt_overlap_image_fraction": overlap_count / image_pixels,
        "bim_gt_overlap_gt_fraction": overlap_count / max(gt_process_pixels, 1),
        "bim_gt_agree_pixels": agreement_count,
        "bim_gt_agree_image_fraction": agreement_image_fraction,
        "bim_gt_agree_gt_fraction": agreement_count / max(gt_process_pixels, 1),
        "bim_gt_agree_overlap_fraction": agreement_count / max(overlap_count, 1),
        "bim_gt_median_ratio": float(np.median(overlap_ratio)) if overlap_count else "",
        "bim_gt_median_abs_log_error": (
            float(np.median(np.abs(np.log(overlap_ratio)))) if overlap_count else ""
        ),
        "recomputed_gt_quality_pass": gt_quality_pass,
        "recomputed_model_support_pass": model_support_pass,
        "recomputed_bim_applicability_pass": bim_applicability_pass,
        "recomputed_effective_pass": bool(gt_quality_pass and bim_applicability_pass),
        **selected,
        "selection_matches_recomputation": selection_match,
        "ratio_support_pixels": ratio_support_pixels,
        "learned_scale": learned_scale,
        "learned_log_scale": learned_log_scale,
        "oracle_frame_scale": oracle_scale,
        "scale_log_error": (
            abs(learned_log_scale - math.log(oracle_scale))
            if args.scale_regression
            else ""
        ),
        "scale_signed_log_error": (
            learned_log_scale - math.log(oracle_scale)
            if args.scale_regression
            else ""
        ),
        **BENCHMARK._prefixed("raw", BENCHMARK.metric_values(base_full, gt, gt_valid)),
        **BENCHMARK._prefixed("learned", BENCHMARK.metric_values(learned_full, gt, gt_valid)),
        **BENCHMARK._prefixed(
            "oracle_frame_scale",
            BENCHMARK.metric_values(oracle_full, gt, gt_valid),
        ),
        "bim_render_seconds": render_seconds,
        "da3_inference_seconds": da3_seconds,
        "dense_inference_seconds": dense_seconds,
        "status": "ok",
        "error": "",
    }


def _micro_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    weights = np.asarray([row["gt_valid_pixels"] for row in rows], dtype=np.float64)
    output = {
        metric: float(
            np.average(
                np.asarray([row[f"{prefix}_{metric}"] for row in rows], dtype=np.float64),
                weights=weights,
            )
        )
        for metric in LINEAR_MICRO_METRICS
    }
    for metric in ("rmse_m", "rmse_log"):
        output[metric] = float(
            math.sqrt(
                np.average(
                    np.asarray([row[f"{prefix}_{metric}"] ** 2 for row in rows]),
                    weights=weights,
                )
            )
        )
    output["silog_x100"] = float(
        100.0
        * math.sqrt(max(0.0, output["rmse_log"] ** 2 - output["mean_log_error"] ** 2))
    )
    output["valid_pixels"] = int(weights.sum())
    return output


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row.get("status") == "ok"]
    if not selected:
        return {"frames": 0, "valid_pixels": 0}
    predictions = {}
    for prefix in PREDICTION_NAMES:
        predictions[prefix] = {
            "pixel_micro": _micro_metrics(selected, prefix),
            "frame_macro": {
                metric: float(np.mean([row[f"{prefix}_{metric}"] for row in selected]))
                for metric in METRICS
            },
        }
    raw = np.asarray([row["raw_abs_rel"] for row in selected])
    learned = np.asarray([row["learned_abs_rel"] for row in selected])
    raw_micro = predictions["raw"]["pixel_micro"]["abs_rel"]
    learned_micro = predictions["learned"]["pixel_micro"]["abs_rel"]
    return {
        "frames": len(selected),
        "valid_pixels": int(sum(row["gt_valid_pixels"] for row in selected)),
        "predictions": predictions,
        "learned_vs_raw": {
            "pixel_micro_abs_rel_difference": learned_micro - raw_micro,
            "pixel_micro_abs_rel_relative_improvement": (raw_micro - learned_micro) / raw_micro,
            "frame_macro_abs_rel_difference": float(np.mean(learned - raw)),
            "frame_win_fraction": float(np.mean(learned < raw)),
        },
    }


def _reason_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for reason in str(row.get("filter_reasons") or "").split(";"):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def build_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    checkpoint: Mapping[str, Any],
    scene_id: str,
    bimnet_key: str,
    mesh_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    subsets = {
        "all_gt_valid": ok,
        "gt_quality": [row for row in ok if _as_bool(row.get("gt_quality_pass"))],
        "operational_no_gt": [
            row
            for row in ok
            if _as_bool(row.get("gt_quality_pass"))
            and _as_bool(row.get("camera_in_bim_aabb"))
            and float(row.get("bim_hit_fraction", 0.0)) >= args.bim_min_hit_fraction
            and _as_bool(row.get("model_support_pass"))
        ],
        "gt_verified": [row for row in ok if _as_bool(row.get("effective_pass"))],
        "bim_applicable": [row for row in ok if _as_bool(row.get("bim_applicability_pass"))],
        "effective": [row for row in ok if _as_bool(row.get("effective_pass"))],
        "rejected_from_effective": [
            row for row in ok if not _as_bool(row.get("effective_pass"))
        ],
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "zero-shot dense metric BIM early fusion; full Matterport depth",
        "scene": {"matterport_scene_id": scene_id, "bimnet_scene_key": bimnet_key},
        "model": {
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": BENCHMARK._sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "best_epoch": int(checkpoint["best_epoch"]),
            "class": (
                "BIMEarlyFusionDAv2ScaleRegressor"
                if args.scale_regression
                else "BIMEarlyFusionDepthAnythingV2"
            ),
            "dav2": "Depth-Anything-V2-Metric-Indoor-Base-hf ViT-B/14",
            "prediction": (
                "focal-corrected DA3 multiplied by one learned global scale; no alignment"
                if args.scale_regression
                else "raw dense absolute metric depth; no alignment"
            ),
            "condition": [
                "train-normalized BIM log depth",
                "binary BIM hit mask",
                "clipped focal-corrected BIM/DA3 log disagreement",
            ],
            "da3_model": args.da3_model,
            "da3_revision": args.da3_revision,
            "focal_correction": "mean(processed fx, fy) / 300",
            "bim_normalization": dict(checkpoint["bim_log_statistics"]),
        },
        "bim": dict(mesh_metadata),
        "ground_truth": "Matterport undistorted z-depth uint16 / 4000 metres; full positive depth",
        "filter": {
            "selection_source": (
                str(args.benchmark_reference_csv.expanduser().resolve())
                if args.benchmark_reference_csv
                else "recomputed"
            ),
            "warning": "GT-assisted subsets are diagnostic and not deployable selectors",
            "gt_min_valid_fraction": args.gt_min_valid_fraction,
            "camera_aabb_margin_m": args.bim_aabb_margin_m,
            "bim_min_hit_fraction": args.bim_min_hit_fraction,
            "bim_min_agree_image_fraction": args.bim_min_agree_image_fraction,
            "absolute_tolerance_m": args.bim_gt_absolute_tolerance_m,
            "relative_tolerance": args.bim_gt_relative_tolerance,
            "ratio_filter": [args.ratio_min, args.ratio_max],
            "ratio_min_support": args.ratio_min_support,
            "reason_counts_all_rows": _reason_counts(rows),
            "selection_recomputation_mismatches": sum(
                not _as_bool(row.get("selection_matches_recomputation")) for row in rows
            ),
        },
        "row_counts": {
            "unique_frames": len(rows),
            "ok": len(ok),
            "skipped_bad_gt": sum(row.get("status") == "skipped_bad_gt" for row in rows),
            "error": sum(row.get("status") == "error" for row in rows),
        },
        "subsets": {name: aggregate_rows(subset) for name, subset in subsets.items()},
    }


def _csv_columns() -> list[str]:
    identifiers = [
        "scene_id",
        "panorama_id",
        "frame_id",
        "camera_index",
        "yaw_index",
        "rgb_path",
        "depth_path",
        "height",
        "width",
        "process_height",
        "process_width",
        "focal_scale",
        "gt_valid_pixels",
        "gt_valid_fraction",
    ]
    diagnostics = [
        "camera_x",
        "camera_y",
        "camera_z",
        "camera_in_bim_aabb",
        "bim_hit_pixels",
        "bim_hit_fraction",
        "bim_gt_overlap_pixels",
        "bim_gt_overlap_image_fraction",
        "bim_gt_overlap_gt_fraction",
        "bim_gt_agree_pixels",
        "bim_gt_agree_image_fraction",
        "bim_gt_agree_gt_fraction",
        "bim_gt_agree_overlap_fraction",
        "bim_gt_median_ratio",
        "bim_gt_median_abs_log_error",
        "ratio_support_pixels",
        "learned_scale",
        "learned_log_scale",
        "gt_quality_pass",
        "model_support_pass",
        "bim_applicability_pass",
        "effective_pass",
        "filter_reasons",
        "recomputed_gt_quality_pass",
        "recomputed_model_support_pass",
        "recomputed_bim_applicability_pass",
        "recomputed_effective_pass",
        "selection_matches_recomputation",
        "oracle_frame_scale",
        "scale_log_error",
        "scale_signed_log_error",
    ]
    metrics = [f"{prefix}_{metric}" for prefix in PREDICTION_NAMES for metric in METRICS]
    timing = [
        "bim_render_seconds",
        "da3_inference_seconds",
        "dense_inference_seconds",
        "status",
        "error",
    ]
    return [*identifiers, *diagnostics, *metrics, *timing]


def main() -> None:
    args = parse_args()
    validate_args(args)
    toolkit_src = args.toolkit_root.expanduser().resolve() / "src"
    sys.path.insert(0, str(toolkit_src))
    from depth_anything_3.api import DepthAnything3
    from s3dis_sam3d import BIMNetDataset, Matterport3DDataset

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_frame.csv"
    summary_path = output_dir / "summary.json"
    if args.no_resume and csv_path.exists():
        raise FileExistsError(f"--no-resume refuses existing output: {csv_path}")

    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["epoch"]) != int(checkpoint["best_epoch"]):
        raise RuntimeError("Zero-shot evaluation requires the selected best-epoch checkpoint")
    if args.scale_regression:
        scale_cfg = cfg.model.dav2_scale
        model = BIMEarlyFusionDAv2ScaleRegressor.from_pretrained(
            str(cfg.model.dav2.model_id),
            revision=str(cfg.model.dav2.revision),
            local_files_only=bool(cfg.model.dav2.local_files_only),
            regression_hidden_size=int(scale_cfg.regression_hidden_size),
            head_dropout_probability=float(scale_cfg.head_dropout_probability),
            output_weight_std=float(scale_cfg.output_weight_std),
        )
    else:
        model = BIMEarlyFusionDepthAnythingV2.from_pretrained(
            str(cfg.model.dav2.model_id),
            revision=str(cfg.model.dav2.revision),
            local_files_only=bool(cfg.model.dav2.local_files_only),
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    bim_stats = checkpoint["bim_log_statistics"]

    reference_rows = (
        BENCHMARK._read_latest_rows(args.benchmark_reference_csv)
        if args.benchmark_reference_csv
        else []
    )
    reference_by_frame = {str(row["frame_id"]): row for row in reference_rows}

    bim_scene = BIMNetDataset(args.bimnet_root)[args.bimnet_scene]
    matterport_scene = bim_scene.matterport_scene(Matterport3DDataset(args.matterport_root))
    frames = list(matterport_scene.frames)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    if reference_rows and len(reference_rows) != len(list(matterport_scene.frames)):
        raise RuntimeError("Benchmark reference CSV does not cover the complete scene")

    wall_filled = not args.no_wall_filled
    if args.mesh_source == "ifc" and wall_filled:
        raise ValueError("wall-filled meshes are only available with --mesh-source obj")
    mesh_start = time.perf_counter()
    mesh = bim_scene.mesh(
        source=args.mesh_source,
        wall_filled=wall_filled,
        coordinates="point_cloud",
        progress=False,
    )
    raycaster = BENCHMARK.BIMRaycaster(mesh)
    mesh_metadata = {
        "source": args.mesh_source,
        "wall_filled": wall_filled,
        "coordinates": "Matterport/BIMNet point-cloud world coordinates",
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "aabb_min": raycaster.minimum.tolist(),
        "aabb_max": raycaster.maximum.tolist(),
        "load_seconds": time.perf_counter() - mesh_start,
    }
    da3_model = (
        DepthAnything3.from_pretrained(
            args.da3_model,
            revision=args.da3_revision,
            local_files_only=not args.allow_network,
        )
        .to(device)
        .eval()
    )

    existing = BENCHMARK._read_latest_rows(csv_path)
    completed = {str(row["frame_id"]) for row in existing}
    pending = [frame for frame in frames if frame.frame_id not in completed]
    print(
        f"scene={matterport_scene.scene_id} bim={bim_scene.key} frames={len(frames)} "
        f"completed={len(frames) - len(pending)} pending={len(pending)} device={device}",
        flush=True,
    )
    columns = _csv_columns()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    start = time.perf_counter()
    with csv_path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for index, frame in enumerate(pending, start=1):
            try:
                row = evaluate_frame(
                    frame=frame,
                    da3_model=da3_model,
                    model=model,
                    raycaster=raycaster,
                    bim_stats=bim_stats,
                    reference_row=reference_by_frame.get(frame.frame_id),
                    args=args,
                )
            except Exception as error:  # noqa: BLE001 - preserve resumable benchmark progress
                row = {
                    "scene_id": frame.scene_id,
                    "panorama_id": frame.panorama_id,
                    "frame_id": frame.frame_id,
                    "camera_index": frame.camera_index,
                    "yaw_index": frame.yaw_index,
                    "rgb_path": str(frame.rgb_path),
                    "depth_path": str(frame.depth_path),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"ERROR {frame.frame_id}: {row['error']}", flush=True)
            writer.writerow({column: row.get(column, "") for column in columns})
            handle.flush()
            if index == 1 or index % args.progress_every == 0 or index == len(pending):
                elapsed = time.perf_counter() - start
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else float("nan")
                metrics = (
                    f" raw={row['raw_abs_rel']:.5f} learned={row['learned_abs_rel']:.5f}"
                    if row.get("status") == "ok"
                    else ""
                )
                print(
                    f"progress={index}/{len(pending)} frame={frame.frame_id}{metrics} "
                    f"effective={row.get('effective_pass')} rate={rate:.3f}fps "
                    f"eta_min={eta / 60:.1f}",
                    flush=True,
                )

    rows = BENCHMARK._read_latest_rows(csv_path)
    summary = build_summary(
        rows,
        args=args,
        checkpoint=checkpoint,
        scene_id=matterport_scene.scene_id,
        bimnet_key=bim_scene.key,
        mesh_metadata=mesh_metadata,
    )
    summary["artifacts"] = {
        "per_frame_csv": str(csv_path),
        "per_frame_csv_sha256": BENCHMARK._sha256(csv_path),
    }
    temporary = summary_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, summary_path)
    headline = summary["subsets"]["gt_quality"]
    print(
        f"COMPLETE rows={len(rows)}/{len(frames)} gt_quality={headline['frames']} "
        f"raw_abs_rel={headline['predictions']['raw']['pixel_micro']['abs_rel']:.6f} "
        f"learned_abs_rel={headline['predictions']['learned']['pixel_micro']['abs_rel']:.6f} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
