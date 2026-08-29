#!/usr/bin/env python3
"""Zero-shot learned-scale evaluation on one Matterport3D/BIMNet scene.

The learned estimator is evaluated exactly as a scale-only model: frozen DA3
produces focal-corrected metric depth, the registered BIMNet envelope is
raycast into the same processed pinhole camera, and the Area_1-trained scale
head predicts one scalar scale per frame.  Ground truth is never a model
input.  It is optionally used to curate a diagnostic ``effective`` benchmark
that removes corrupt depth and views outside the modeled BIM extent.

The CSV is append-only and resumable.  ``summary.json`` always deduplicates by
``scene_id/frame_id`` before aggregating, so a failed frame can safely be
rerun after its row is removed or the output directory is changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
import open3d as o3d
import torch

from bim_priorda3.checkpoints import validate_checkpoint_model_config
from bim_priorda3.config import load_config
from bim_priorda3.models import BIMPriorDA3

DEFAULT_DA3_MODEL = "depth-anything/da3metric-large"
DEFAULT_DA3_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
PROCESS_RES_METHOD = "upper_bound_resize"
PATCH_SIZE = 14
CANONICAL_FOCAL_PX = 300.0
METRICS = (
    "abs_rel",
    "sq_rel",
    "rmse_m",
    "mae_m",
    "rmse_log",
    "mean_log_error",
    "mean_abs_log_error",
    "log10_error",
    "silog_x100",
    "delta1",
    "delta2",
    "delta3",
)
LINEAR_MICRO_METRICS = (
    "abs_rel",
    "sq_rel",
    "mae_m",
    "mean_log_error",
    "mean_abs_log_error",
    "log10_error",
    "delta1",
    "delta2",
    "delta3",
)
PREDICTION_NAMES = ("raw", "learned", "oracle_frame_scale")
SUPPORTED_SCALE_ESTIMATORS = {
    "full_regression_iterative_v1",
    "pseudo_huber_attention_v1",
}


def _estimator_name(scale_model: BIMPriorDA3) -> str:
    return str(
        scale_model.attention_scale_config.get(
            "estimator",
            "pseudo_huber_attention_v1",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--bimnet-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--bimnet-scene", default="hxp")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.process_res < PATCH_SIZE:
        raise ValueError(f"--process-res must be at least {PATCH_SIZE}")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    for name in (
        "gt_min_valid_fraction",
        "bim_min_hit_fraction",
        "bim_min_agree_image_fraction",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.bim_gt_absolute_tolerance_m < 0 or args.bim_gt_relative_tolerance < 0:
        raise ValueError("BIM/GT tolerances must be non-negative")
    if args.bim_aabb_margin_m < 0:
        raise ValueError("--bim-aabb-margin-m must be non-negative")


def _nearest_multiple(value: int, multiple: int = PATCH_SIZE) -> int:
    down = (value // multiple) * multiple
    up = down + multiple
    return max(multiple, up if abs(up - value) <= abs(value - down) else down)


def processed_geometry(
    height: int,
    width: int,
    intrinsics: np.ndarray,
    process_res: int,
) -> tuple[int, int, np.ndarray, float]:
    """Reproduce DA3 ``upper_bound_resize`` geometry and focal correction."""

    boundary_scale = process_res / float(max(height, width))
    resized_width = max(1, round(width * boundary_scale))
    resized_height = max(1, round(height * boundary_scale))
    process_width = _nearest_multiple(resized_width)
    process_height = _nearest_multiple(resized_height)
    processed_intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    if processed_intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    processed_intrinsics[0] *= process_width / float(width)
    processed_intrinsics[1] *= process_height / float(height)
    focal_px = float((processed_intrinsics[0, 0] + processed_intrinsics[1, 1]) / 2.0)
    focal_scale = focal_px / CANONICAL_FOCAL_PX
    if not np.isfinite(focal_scale) or focal_scale <= 0:
        raise ValueError(f"invalid focal scale {focal_scale}")
    return process_height, process_width, processed_intrinsics, focal_scale


def metric_values(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    support = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or support.shape != target.shape:
        raise ValueError("prediction, target and valid mask shapes must agree")
    if not np.any(support):
        raise ValueError("frame has no valid target pixels")
    pred = np.asarray(prediction[support], dtype=np.float64)
    gt = np.asarray(target[support], dtype=np.float64)
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError("prediction must be finite and positive on target support")
    difference = pred - gt
    absolute = np.abs(difference)
    squared = difference**2
    log_difference = np.log(pred) - np.log(gt)
    ratio = np.maximum(pred / gt, gt / pred)
    log_variance = max(
        0.0,
        float(np.mean(log_difference**2) - np.mean(log_difference) ** 2),
    )
    return {
        "abs_rel": float(np.mean(absolute / gt)),
        "sq_rel": float(np.mean(squared / gt)),
        "rmse_m": float(np.sqrt(np.mean(squared))),
        "mae_m": float(np.mean(absolute)),
        "rmse_log": float(np.sqrt(np.mean(log_difference**2))),
        "mean_log_error": float(np.mean(log_difference)),
        "mean_abs_log_error": float(np.mean(np.abs(log_difference))),
        "log10_error": float(np.mean(np.abs(np.log10(pred) - np.log10(gt)))),
        "silog_x100": float(100.0 * math.sqrt(log_variance)),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def _prefixed(prefix: str, values: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


class BIMRaycaster:
    """One reusable CPU raycasting scene for all cameras in a BIMNet scene."""

    def __init__(self, mesh: o3d.geometry.TriangleMesh) -> None:
        if mesh.is_empty():
            raise ValueError("BIM mesh is empty")
        self.minimum = np.asarray(mesh.get_min_bound(), dtype=np.float64)
        self.maximum = np.asarray(mesh.get_max_bound(), dtype=np.float64)
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(tensor_mesh)

    def depth(
        self,
        intrinsics: np.ndarray,
        world_to_camera: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        rays = self.scene.create_rays_pinhole(
            np.asarray(intrinsics, dtype=np.float64),
            np.asarray(world_to_camera, dtype=np.float64),
            width,
            height,
        )
        depth = self.scene.cast_rays(rays)["t_hit"].numpy().astype(np.float32)
        depth[~np.isfinite(depth) | (depth <= 0)] = 0.0
        return depth

    def contains_camera(self, camera_position: np.ndarray, margin: float) -> bool:
        position = np.asarray(camera_position, dtype=np.float64)
        return bool(
            np.all(position >= self.minimum - margin) and np.all(position <= self.maximum + margin)
        )


def _scale_head_batch(
    rgb: np.ndarray,
    base_depth: np.ndarray,
    bim_depth: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if rgb.shape[:2] != base_depth.shape or bim_depth.shape != base_depth.shape:
        raise ValueError("RGB, DA3 depth and BIM depth geometries differ")
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).unsqueeze(0)
    base_tensor = torch.from_numpy(base_depth[None, None].copy())
    bim_tensor = torch.from_numpy(bim_depth[None, None].copy())
    return {
        "rgb": rgb_tensor.to(device=device, dtype=torch.float32),
        "base_depth": base_tensor.to(device=device, dtype=torch.float32),
        # The pseudo-Huber API expects the cached deterministic branch here.
        # Matched zero-shot configs disable that input, so raw DA3 is an exact
        # no-op placeholder and cannot leak a deterministic BIM scale.
        "scaled_depth": base_tensor.to(device=device, dtype=torch.float32),
        "bim_depth": bim_tensor.to(device=device, dtype=torch.float32),
        "bim_valid": (bim_tensor > 0).to(device=device, dtype=torch.float32),
    }


def predict_scale(
    scale_model: BIMPriorDA3,
    batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    with torch.inference_mode():
        output = scale_model._estimate_attention_scale(batch, batch["base_depth"])
    centers = output["iteration_log_scales"].detach().float().reshape(-1).cpu().numpy()
    if "iteration_log_scale_updates" in output:
        updates = (
            output["iteration_log_scale_updates"]
            .detach()
            .float()
            .reshape(-1)
            .cpu()
            .numpy()
        )
    else:
        # Pseudo-Huber exposes the post-update centers, not their increments.
        # All matched controls start at c0=0, so adjacent differences recover
        # the exact realized updates without changing model inference.
        updates = np.diff(np.concatenate((np.zeros(1, dtype=centers.dtype), centers)))
    if len(centers) != 3 or len(updates) != 3:
        raise RuntimeError(f"Expected a three-round scale estimator, got {len(centers)} rounds")
    log_scale = float(output["log_scale"].detach().float().reshape(-1)[0].cpu())
    scale = float(output["scale"].detach().float().reshape(-1)[0].cpu())
    return {
        "scale": scale,
        "log_scale": log_scale,
        "pixel_support": int(output["pixel_support"].detach().cpu().reshape(-1)[0]),
        "token_support": int(output["token_support"].detach().cpu().reshape(-1)[0]),
        **{f"round_{index + 1}_log_scale": float(value) for index, value in enumerate(centers)},
        **{f"round_{index + 1}_update": float(value) for index, value in enumerate(updates)},
    }


def evaluate_frame(
    *,
    frame: Any,
    da3_model: Any,
    scale_model: BIMPriorDA3,
    raycaster: BIMRaycaster,
    args: argparse.Namespace,
    model_min_support: int,
) -> dict[str, Any]:
    height, width = frame.image_shape
    process_height, process_width, processed_k, focal_scale = processed_geometry(
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
        return {
            **base_row,
            "gt_quality_pass": False,
            "bim_applicability_pass": False,
            "effective_pass": False,
            "filter_reasons": "gt_zero_depth",
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
    overlap_count = int(overlap.sum())
    agreement_count = int(agreement.sum())
    image_pixels = int(bim_depth.size)
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
        raise RuntimeError(
            f"DA3 output {canonical_depth.shape} does not match expected {expected_shape}"
        )
    base_depth = canonical_depth * focal_scale
    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image: {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (process_width, process_height), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0

    scale_start = time.perf_counter()
    batch = _scale_head_batch(rgb, base_depth, bim_depth, next(scale_model.parameters()).device)
    scale_output = predict_scale(scale_model, batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    scale_seconds = time.perf_counter() - scale_start
    learned_scale = float(scale_output["scale"])

    base_full = cv2.resize(base_depth, (width, height), interpolation=cv2.INTER_LINEAR)
    learned_full = base_full * learned_scale
    gt_values = gt[gt_valid].astype(np.float64)
    base_values = base_full[gt_valid].astype(np.float64)
    oracle_scale = float(np.median(gt_values / base_values))
    oracle_full = base_full * oracle_scale

    gt_quality_pass = bool(gt_valid.mean() >= args.gt_min_valid_fraction)
    hit_fraction = float(bim_hit.mean())
    agreement_image_fraction = agreement_count / image_pixels
    model_support_pass = bool(scale_output["pixel_support"] >= model_min_support)
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

    row = {
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
        "bim_gt_median_ratio": (float(np.median(overlap_ratio)) if overlap_count else float("nan")),
        "bim_gt_median_abs_log_error": (
            float(np.median(np.abs(np.log(overlap_ratio)))) if overlap_count else float("nan")
        ),
        "gt_quality_pass": gt_quality_pass,
        "model_support_pass": model_support_pass,
        "bim_applicability_pass": bim_applicability_pass,
        "effective_pass": bool(gt_quality_pass and bim_applicability_pass),
        "filter_reasons": ";".join(reasons),
        "learned_scale": learned_scale,
        "learned_log_scale": float(scale_output["log_scale"]),
        "oracle_frame_scale": oracle_scale,
        "scale_log_error": float(abs(math.log(learned_scale) - math.log(oracle_scale))),
        "scale_signed_log_error": float(math.log(learned_scale) - math.log(oracle_scale)),
        "scale_pixel_support": int(scale_output["pixel_support"]),
        "scale_token_support": int(scale_output["token_support"]),
        **{key: value for key, value in scale_output.items() if key.startswith("round_")},
        **_prefixed("raw", metric_values(base_full, gt, gt_valid)),
        **_prefixed("learned", metric_values(learned_full, gt, gt_valid)),
        **_prefixed(
            "oracle_frame_scale",
            metric_values(oracle_full, gt, gt_valid),
        ),
        "bim_render_seconds": render_seconds,
        "da3_inference_seconds": da3_seconds,
        "scale_inference_seconds": scale_seconds,
        "status": "ok",
        "error": "",
    }
    return row


def _to_number(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    if value is None or value == "":
        return value
    text = str(value)
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return value


def _read_latest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {key: _to_number(value) for key, value in raw.items()}
            latest[f"{row['scene_id']}/{row['frame_id']}"] = row
    return list(latest.values())


def _micro_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    weights = np.asarray([row["gt_valid_pixels"] for row in rows], dtype=np.float64)
    total = float(weights.sum())
    output = {
        metric: float(
            np.average(
                np.asarray([row[f"{prefix}_{metric}"] for row in rows], dtype=np.float64),
                weights=weights,
            )
        )
        for metric in LINEAR_MICRO_METRICS
    }
    output["rmse_m"] = float(
        math.sqrt(
            np.average(
                np.asarray([row[f"{prefix}_rmse_m"] ** 2 for row in rows]),
                weights=weights,
            )
        )
    )
    output["rmse_log"] = float(
        math.sqrt(
            np.average(
                np.asarray([row[f"{prefix}_rmse_log"] ** 2 for row in rows]),
                weights=weights,
            )
        )
    )
    mean_log = output["mean_log_error"]
    output["silog_x100"] = float(100.0 * math.sqrt(max(0.0, output["rmse_log"] ** 2 - mean_log**2)))
    output["valid_pixels"] = int(total)
    return output


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    metric_rows = [row for row in rows if row.get("status") == "ok"]
    if not metric_rows:
        return {"frames": 0, "valid_pixels": 0}
    predictions: dict[str, Any] = {}
    for prefix in PREDICTION_NAMES:
        macro = {
            metric: float(np.mean([row[f"{prefix}_{metric}"] for row in metric_rows]))
            for metric in METRICS
        }
        predictions[prefix] = {
            "pixel_micro": _micro_metrics(metric_rows, prefix),
            "frame_macro": macro,
        }
    raw_abs = np.asarray([row["raw_abs_rel"] for row in metric_rows], dtype=np.float64)
    learned_abs = np.asarray([row["learned_abs_rel"] for row in metric_rows], dtype=np.float64)
    raw_micro = predictions["raw"]["pixel_micro"]["abs_rel"]
    learned_micro = predictions["learned"]["pixel_micro"]["abs_rel"]
    return {
        "frames": len(metric_rows),
        "valid_pixels": int(sum(row["gt_valid_pixels"] for row in metric_rows)),
        "predictions": predictions,
        "learned_vs_raw": {
            "pixel_micro_abs_rel_difference": learned_micro - raw_micro,
            "pixel_micro_abs_rel_relative_improvement": (
                (raw_micro - learned_micro) / raw_micro if raw_micro > 0 else float("nan")
            ),
            "frame_macro_abs_rel_difference": float(np.mean(learned_abs - raw_abs)),
            "frame_win_fraction": float(np.mean(learned_abs < raw_abs)),
        },
        "scale": {
            "mean": float(np.mean([row["learned_scale"] for row in metric_rows])),
            "median": float(np.median([row["learned_scale"] for row in metric_rows])),
            "mean_abs_log_error": float(np.mean([row["scale_log_error"] for row in metric_rows])),
        },
    }


def _reason_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reasons = str(row.get("filter_reasons") or "").split(";")
        for reason in reasons:
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def _sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for hit_threshold in (0.10, 0.20, 0.30, 0.50):
        for agreement_threshold in (0.05, 0.10, 0.20):
            selected = [
                row
                for row in rows
                if row.get("status") == "ok"
                and bool(row.get("gt_quality_pass"))
                and bool(row.get("camera_in_bim_aabb"))
                and float(row.get("bim_hit_fraction", 0)) >= hit_threshold
                and float(row.get("bim_gt_agree_image_fraction", 0)) >= agreement_threshold
                and bool(row.get("model_support_pass"))
            ]
            aggregate = aggregate_rows(selected)
            comparison = aggregate.get("learned_vs_raw", {})
            output.append(
                {
                    "bim_min_hit_fraction": hit_threshold,
                    "bim_min_agree_image_fraction": agreement_threshold,
                    "frames": aggregate.get("frames", 0),
                    "raw_abs_rel": aggregate.get("predictions", {})
                    .get("raw", {})
                    .get("pixel_micro", {})
                    .get("abs_rel"),
                    "learned_abs_rel": aggregate.get("predictions", {})
                    .get("learned", {})
                    .get("pixel_micro", {})
                    .get("abs_rel"),
                    "relative_improvement": comparison.get(
                        "pixel_micro_abs_rel_relative_improvement"
                    ),
                }
            )
    return output


def build_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    scale_model: BIMPriorDA3,
    scene_id: str,
    bimnet_key: str,
    mesh_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    operational_no_gt = [
        row
        for row in ok
        if bool(row.get("gt_quality_pass"))
        and bool(row.get("camera_in_bim_aabb"))
        and float(row.get("bim_hit_fraction", 0.0)) >= args.bim_min_hit_fraction
        and bool(row.get("model_support_pass"))
    ]
    subsets = {
        "all_gt_valid": ok,
        "gt_quality": [row for row in ok if bool(row.get("gt_quality_pass"))],
        "operational_no_gt": operational_no_gt,
        "gt_verified": [
            row
            for row in ok
            if bool(row.get("gt_quality_pass"))
            and bool(row.get("bim_applicability_pass"))
        ],
        # Compatibility names retained for existing analysis consumers. These
        # include GT/BIM agreement and therefore are diagnostic, not deployable.
        "bim_applicable": [row for row in ok if bool(row.get("bim_applicability_pass"))],
        "effective": [row for row in ok if bool(row.get("effective_pass"))],
        "rejected_from_effective": [row for row in ok if not bool(row.get("effective_pass"))],
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "zero-shot single-view learned frame scale",
        "scene": {
            "matterport_scene_id": scene_id,
            "bimnet_scene_key": bimnet_key,
        },
        "model": {
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": _sha256(args.checkpoint),
            "estimator": _estimator_name(scale_model),
            "input_channels": scale_model.attention_scale_input_channels,
            "inputs": [
                "RGB(3)",
                "log focal-corrected raw DA3 depth(1)",
                "masked log registered BIM depth(1)",
                "BIM hit mask(1)",
                "signed BIM/raw-DA3 log disagreement(1)",
                "absolute BIM/raw-DA3 log disagreement(1)",
            ],
            "rounds": scale_model.attention_scale.iterative_updates,
            "initial_log_scale": float(
                getattr(scale_model.attention_scale, "iterative_initial_log_scale", 0.0)
            ),
            "iterative_refresh_attention": getattr(
                scale_model.attention_scale,
                "iterative_refresh_attention",
                None,
            ),
            "use_fallback_gate": getattr(
                scale_model.attention_scale,
                "use_fallback_gate",
                False,
            ),
            "huber_delta": getattr(scale_model.attention_scale, "huber_delta", None),
            "da3_model": args.da3_model,
            "da3_revision": args.da3_revision,
            "focal_correction": "mean(processed fx, fy) / 300",
        },
        "bim": dict(mesh_metadata),
        "ground_truth": "Matterport undistorted z-depth uint16 / 4000 metres",
        "filter": {
            "diagnostic_gt_assisted": True,
            "warning": (
                "effective-pass uses GT/BIM agreement and is benchmark curation, "
                "not a deployable test-time frame selector"
            ),
            "operational_no_gt_definition": (
                "GT-quality scoring frames with camera inside BIM AABB, sufficient "
                "BIM hit fraction, and sufficient BIM/DA3 ratio support; selection "
                "does not use BIM/GT agreement"
            ),
            "gt_min_valid_fraction": args.gt_min_valid_fraction,
            "camera_aabb_margin_m": args.bim_aabb_margin_m,
            "bim_min_hit_fraction": args.bim_min_hit_fraction,
            "bim_min_agree_image_fraction": args.bim_min_agree_image_fraction,
            "bim_gt_agreement": ("abs(BIM-GT) <= max(absolute_tolerance_m, relative_tolerance*GT)"),
            "absolute_tolerance_m": args.bim_gt_absolute_tolerance_m,
            "relative_tolerance": args.bim_gt_relative_tolerance,
            "model_min_ratio_support": scale_model.attention_scale.min_support,
            "reason_counts_all_rows": _reason_counts(rows),
        },
        "row_counts": {
            "unique_frames": len(rows),
            "ok": len(ok),
            "skipped_bad_gt": sum(row.get("status") == "skipped_bad_gt" for row in rows),
            "error": sum(row.get("status") == "error" for row in rows),
        },
        "subsets": {name: aggregate_rows(subset) for name, subset in subsets.items()},
        "threshold_sensitivity": _sensitivity(ok),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


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
    diagnostic = [
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
        "gt_quality_pass",
        "model_support_pass",
        "bim_applicability_pass",
        "effective_pass",
        "filter_reasons",
    ]
    scale = [
        "learned_scale",
        "learned_log_scale",
        "oracle_frame_scale",
        "scale_log_error",
        "scale_signed_log_error",
        "scale_pixel_support",
        "scale_token_support",
        *[f"round_{index}_log_scale" for index in range(1, 4)],
        *[f"round_{index}_update" for index in range(1, 4)],
    ]
    metrics = [f"{prefix}_{metric}" for prefix in PREDICTION_NAMES for metric in METRICS]
    timing = [
        "bim_render_seconds",
        "da3_inference_seconds",
        "scale_inference_seconds",
        "status",
        "error",
    ]
    return [*identifiers, *diagnostic, *scale, *metrics, *timing]


def main() -> None:
    args = parse_args()
    _validate_args(args)
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
    scale_model = BIMPriorDA3(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_model_config(checkpoint, cfg.model)
    scale_model.load_state_dict(checkpoint["model"], strict=True)
    scale_model.to(device).eval()
    if scale_model.attention_scale is None:
        raise RuntimeError("Configured checkpoint has no learned scale estimator")
    estimator_name = _estimator_name(scale_model)
    if estimator_name not in SUPPORTED_SCALE_ESTIMATORS:
        raise RuntimeError(
            f"Unsupported learned scale estimator {estimator_name!r}; expected one of "
            f"{sorted(SUPPORTED_SCALE_ESTIMATORS)}"
        )
    if scale_model.attention_scale_input_channels != 8:
        raise RuntimeError(
            "Expected the current RGB+DA3+BIM 8-channel checkpoint, got "
            f"{scale_model.attention_scale_input_channels} channels"
        )
    if scale_model.attention_scale.iterative_updates != 3:
        raise RuntimeError("This benchmark requires exactly three scale-update rounds")
    if estimator_name == "pseudo_huber_attention_v1":
        if scale_model.attention_scale_use_deterministic_fallback_input:
            raise RuntimeError("Matched pseudo-Huber evaluation forbids deterministic scale input")
        if scale_model.attention_scale.use_fallback_gate:
            raise RuntimeError("Matched pseudo-Huber evaluation forbids the fallback gate")
        if not math.isclose(
            scale_model.attention_scale.iterative_initial_log_scale,
            0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError("Matched pseudo-Huber evaluation requires c0=0")

    bim_dataset = BIMNetDataset(args.bimnet_root)
    bim_scene = bim_dataset[args.bimnet_scene]
    matterport_dataset = Matterport3DDataset(args.matterport_root)
    matterport_scene = bim_scene.matterport_scene(matterport_dataset)
    frames = list(matterport_scene.frames)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
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
    raycaster = BIMRaycaster(mesh)
    mesh_seconds = time.perf_counter() - mesh_start
    mesh_metadata = {
        "source": args.mesh_source,
        "wall_filled": wall_filled,
        "coordinates": "Matterport/BIMNet point-cloud world coordinates",
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.triangles),
        "aabb_min": raycaster.minimum.tolist(),
        "aabb_max": raycaster.maximum.tolist(),
        "load_seconds": mesh_seconds,
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
    prior_rows = _read_latest_rows(csv_path)
    completed = {f"{row['scene_id']}/{row['frame_id']}" for row in prior_rows}
    pending = [frame for frame in frames if f"{frame.scene_id}/{frame.frame_id}" not in completed]
    print(
        f"scene={matterport_scene.scene_id} bim={bim_scene.key} frames={len(frames)} "
        f"completed={len(frames) - len(pending)} pending={len(pending)} device={device}",
        flush=True,
    )
    print(
        f"mesh={args.mesh_source} wall_filled={wall_filled} "
        f"vertices={len(mesh.vertices)} triangles={len(mesh.triangles)}; "
        "GT is used only for diagnostic benchmark filtering and scoring",
        flush=True,
    )

    columns = _csv_columns()
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    run_start = time.perf_counter()
    with csv_path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for index, frame in enumerate(pending, start=1):
            try:
                row = evaluate_frame(
                    frame=frame,
                    da3_model=da3_model,
                    scale_model=scale_model,
                    raycaster=raycaster,
                    args=args,
                    model_min_support=scale_model.attention_scale.min_support,
                )
            except Exception as error:  # noqa: BLE001 - preserve long benchmark progress
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
                elapsed = time.perf_counter() - run_start
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else float("nan")
                metric_text = (
                    f" raw={row['raw_abs_rel']:.5f} learned={row['learned_abs_rel']:.5f}"
                    if row.get("status") == "ok"
                    else ""
                )
                print(
                    f"progress={index}/{len(pending)} frame={frame.frame_id}{metric_text} "
                    f"effective={row.get('effective_pass')} rate={rate:.3f}fps "
                    f"eta_min={eta / 60:.1f}",
                    flush=True,
                )

    rows = _read_latest_rows(csv_path)
    summary = build_summary(
        rows,
        args=args,
        scale_model=scale_model,
        scene_id=matterport_scene.scene_id,
        bimnet_key=bim_scene.key,
        mesh_metadata=mesh_metadata,
    )
    summary["artifacts"] = {
        "per_frame_csv": str(csv_path),
        "per_frame_csv_sha256": _sha256(csv_path),
    }
    _atomic_json(summary_path, summary)
    effective = summary["subsets"]["effective"]
    raw_abs = effective.get("predictions", {}).get("raw", {}).get("pixel_micro", {}).get("abs_rel")
    learned_abs = (
        effective.get("predictions", {}).get("learned", {}).get("pixel_micro", {}).get("abs_rel")
    )
    print(
        f"COMPLETE rows={len(rows)}/{len(frames)} effective={effective.get('frames', 0)} "
        f"effective_raw_abs_rel={raw_abs} effective_learned_abs_rel={learned_abs} "
        f"summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
