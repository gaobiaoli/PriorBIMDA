#!/usr/bin/env python3
"""Zero-shot evaluation of Area_1 scale+refiner models on Matterport/BIMNet.

The Stanford Area_1 checkpoint is frozen.  DA3 Metric-Large supplies focal-
corrected depth and, when configured, frozen intermediate features; the
registered BIMNet envelope supplies hit-only z-depth and optional surface
geometry. Matterport GT is used only for scoring and the frozen three-rule
frame selection, never as a model input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import sys
import time
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from bim_priorda3.checkpoints import validate_checkpoint_model_config
from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.geometry import depth_edges
from bim_priorda3.models import (
    BIMEarlyFusionDAv2JointScaleLow,
    BIMPriorDA3,
    FrozenHuberDAv2LowRefiner,
    FrozenHuberPriorDAV11BIM,
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
PREDICTION_NAMES = ("raw", "scale", "scale_low", "final", "oracle_frame_scale")
EXPECTED_FRAME_SET_SHA256 = "e6639e7bd16eb7b666a6f22f41ee17ec50a2ce8d8b841427ad489126013bd18b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--bimnet-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--bimnet-scene", default="hxp")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--da3-model", default=BENCHMARK.DEFAULT_DA3_MODEL)
    parser.add_argument("--da3-revision", default=BENCHMARK.DEFAULT_DA3_REVISION)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mesh-source", choices=("obj", "ifc"), default="obj")
    parser.add_argument("--no-wall-filled", action="store_true")
    parser.add_argument("--gt-min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--bim-min-hit-fraction", type=float, default=0.20)
    parser.add_argument("--aabb-margin-m", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.process_res < 14 or args.process_res % 14:
        raise ValueError("--process-res must be a positive multiple of DA3 patch size 14")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if not 0 <= args.gt_min_valid_fraction <= 1:
        raise ValueError("--gt-min-valid-fraction must be in [0, 1]")
    if not 0 <= args.bim_min_hit_fraction <= 1:
        raise ValueError("--bim-min-hit-fraction must be in [0, 1]")
    if args.aabb_margin_m < 0:
        raise ValueError("--aabb-margin-m must be non-negative")


def render_bim_geometry(
    raycaster: Any,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render hit-only BIM z-depth and camera-space primitive normals."""

    rays = raycaster.scene.create_rays_pinhole(
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(world_to_camera, dtype=np.float64),
        width,
        height,
    )
    result = raycaster.scene.cast_rays(rays)
    depth = result["t_hit"].numpy().astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    depth[~valid] = 0.0

    normals_world = result["primitive_normals"].numpy().astype(np.float32)
    camera_to_world = np.linalg.inv(np.asarray(world_to_camera, dtype=np.float64))
    normals_camera = normals_world @ camera_to_world[:3, :3]
    normals_camera[~valid] = 0.0
    return depth, normals_camera.transpose(2, 0, 1).astype(np.float32)


def _da3_feature(aux: Mapping[str, Any], layer: int, grid_shape: tuple[int, int]) -> np.ndarray:
    key = f"feat_layer_{layer}"
    if key not in aux:
        raise KeyError(f"DA3 prediction did not return {key}")
    feature = np.asarray(aux[key], dtype=np.float32)
    # OutputProcessor removes the batch dimension: [view, grid_h, grid_w, channels].
    if feature.ndim != 4 or feature.shape[0] != 1:
        raise RuntimeError(f"{key} must have shape [1,H,W,C], got {feature.shape}")
    if tuple(feature.shape[1:3]) != grid_shape:
        raise RuntimeError(f"{key} grid {feature.shape[1:3]} != expected {grid_shape}")
    return feature[0].transpose(2, 0, 1).copy()


def _tensor(value: np.ndarray, device: torch.device, *, channel: bool = False) -> torch.Tensor:
    array = value[None, None] if channel else value[None]
    return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=torch.float32)


def build_batch(
    *,
    model: BIMPriorDA3 | FrozenHuberDAv2LowRefiner | FrozenHuberPriorDAV11BIM | BIMEarlyFusionDAv2JointScaleLow,
    rgb: np.ndarray,
    base_depth: np.ndarray,
    base_confidence: np.ndarray,
    bim_depth: np.ndarray,
    bim_normals: np.ndarray,
    bim_edge: np.ndarray,
    da3_feature_mid: np.ndarray | None,
    da3_feature_deep: np.ndarray | None,
) -> tuple[dict[str, torch.Tensor], float, int]:
    device = next(model.parameters()).device
    batch = {
        "rgb": _tensor(rgb.transpose(2, 0, 1), device),
        "base_depth": _tensor(base_depth, device, channel=True),
        "base_confidence": _tensor(base_confidence, device, channel=True),
        "bim_depth": _tensor(bim_depth, device, channel=True),
        "bim_valid": _tensor((bim_depth > 0).astype(np.float32), device, channel=True),
        "bim_normals": _tensor(bim_normals, device),
        "bim_edge": _tensor(bim_edge, device, channel=True),
    }
    if da3_feature_mid is not None:
        batch["da3_feature_mid"] = _tensor(da3_feature_mid, device)
    if da3_feature_deep is not None:
        batch["da3_feature_deep"] = _tensor(da3_feature_deep, device)
    if isinstance(model, BIMEarlyFusionDAv2JointScaleLow):
        # The joint regressor never consumes an analytic BIM scale. Keep the
        # legacy diagnostic columns neutral rather than instantiating a
        # deterministic estimator outside the model.
        deterministic_scale = torch.ones_like(batch["base_depth"][:, :1, :1, :1])
        support = batch["bim_valid"].flatten(1).sum(dim=1)
    else:
        scale_system = (
            model.scale_system
            if isinstance(model, (FrozenHuberDAv2LowRefiner, FrozenHuberPriorDAV11BIM))
            else model
        )
        deterministic_scale, support, _, _ = scale_system._robust_bim_scale(
            batch["base_depth"], batch["bim_depth"], batch["bim_valid"]
        )
    batch["scaled_depth"] = batch["base_depth"] * deterministic_scale
    return batch, float(deterministic_scale.item()), int(support.item())


def evaluate_frame(
    *,
    frame: Any,
    da3_model: Any,
    model: BIMPriorDA3 | FrozenHuberDAv2LowRefiner | FrozenHuberPriorDAV11BIM | BIMEarlyFusionDAv2JointScaleLow,
    raycaster: Any,
    args: argparse.Namespace,
    feature_layers: tuple[int, ...],
    bim_log_mean: float | None = None,
    bim_log_std: float | None = None,
) -> dict[str, Any]:
    height, width = frame.image_shape
    process_height, process_width, processed_k, focal_scale = BENCHMARK.processed_geometry(
        height, width, frame.intrinsics, args.process_res
    )
    expected_shape = (process_height, process_width)
    gt = np.asarray(frame.depth, dtype=np.float32)
    gt_valid = np.isfinite(gt) & (gt > 0)
    if not np.any(gt_valid):
        return {
            "scene_id": frame.scene_id,
            "panorama_id": frame.panorama_id,
            "frame_id": frame.frame_id,
            "camera_index": frame.camera_index,
            "yaw_index": frame.yaw_index,
            "rgb_path": str(frame.rgb_path),
            "depth_path": str(frame.depth_path),
            "height": height,
            "width": width,
            "gt_valid_pixels": 0,
            "gt_valid_fraction": 0.0,
            "status": "skipped_bad_gt",
            "error": "Matterport depth contains no finite positive pixels",
        }

    render_start = time.perf_counter()
    bim_depth, bim_normals = render_bim_geometry(
        raycaster, processed_k, frame.world_to_camera, process_width, process_height
    )
    bim_valid = bim_depth > 0
    bim_edge = depth_edges(bim_depth, bim_valid, threshold_m=0.08)
    render_seconds = time.perf_counter() - render_start

    da3_start = time.perf_counter()
    with torch.inference_mode():
        da3_output = da3_model.inference(
            [str(frame.rgb_path)],
            process_res=args.process_res,
            process_res_method=BENCHMARK.PROCESS_RES_METHOD,
            export_dir=None,
            export_feat_layers=list(feature_layers),
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    da3_seconds = time.perf_counter() - da3_start
    canonical_depth = np.asarray(da3_output.depth[0], dtype=np.float32)
    if canonical_depth.shape != expected_shape:
        raise RuntimeError(f"DA3 depth {canonical_depth.shape} != expected {expected_shape}")
    base_depth = canonical_depth * focal_scale
    raw_confidence = getattr(da3_output, "conf", None)
    if raw_confidence is None:
        raw_confidence = getattr(da3_output, "depth_conf", None)
    if raw_confidence is None:
        log_depth = np.log(np.maximum(base_depth, 1e-4))
        laplacian = np.abs(cv2.Laplacian(log_depth, cv2.CV_32F))
        confidence_scale = float(np.quantile(laplacian, 0.9)) + 1e-6
        base_confidence = np.exp(-laplacian / confidence_scale).astype(np.float32)
        confidence_source = "log-depth Laplacian fallback"
    else:
        base_confidence = np.asarray(raw_confidence[0], dtype=np.float32)
        if base_confidence.shape != expected_shape:
            base_confidence = cv2.resize(
                base_confidence, (process_width, process_height), interpolation=cv2.INTER_LINEAR
            )
        confidence_source = "DA3 depth_conf"

    da3_feature_mid = None
    da3_feature_deep = None
    if feature_layers:
        grid_shape = (process_height // 14, process_width // 14)
        da3_feature_mid = _da3_feature(da3_output.aux, feature_layers[0], grid_shape)
        da3_feature_deep = _da3_feature(da3_output.aux, feature_layers[1], grid_shape)
    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image: {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (process_width, process_height), interpolation=cv2.INTER_AREA)
    rgb = rgb.astype(np.float32) / 255.0

    batch, deterministic_scale, deterministic_support = build_batch(
        model=model,
        rgb=rgb,
        base_depth=base_depth,
        base_confidence=base_confidence,
        bim_depth=bim_depth,
        bim_normals=bim_normals,
        bim_edge=bim_edge,
        da3_feature_mid=da3_feature_mid,
        da3_feature_deep=da3_feature_deep,
    )
    model_start = time.perf_counter()
    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        if isinstance(model, BIMEarlyFusionDAv2JointScaleLow):
            if bim_log_mean is None or bim_log_std is None:
                raise RuntimeError("Joint DAv2 evaluator lacks BIM normalization statistics")
            condition = build_bim_condition(
                batch,
                bim_log_mean=bim_log_mean,
                bim_log_std=bim_log_std,
            )
            output = model(batch["rgb"], condition, batch["base_depth"])
        else:
            output = model(batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_seconds = time.perf_counter() - model_start

    scale_process = output["scaled_depth"].detach().float().squeeze().cpu().numpy()
    if isinstance(model, BIMEarlyFusionDAv2JointScaleLow):
        scale_low_process = (
            output["scaled_depth"] * torch.exp(output["low1_log_residual"].float())
        ).detach().float().squeeze().cpu().numpy()
    elif isinstance(model, FrozenHuberDAv2LowRefiner):
        # This architecture deliberately has no r_detail; its final prediction
        # is exactly the scale+r_low stage.
        scale_low_process = output["depth"].detach().float().squeeze().cpu().numpy()
    elif isinstance(model, FrozenHuberPriorDAV11BIM):
        # For this adapter, scale_low denotes the non-learned local Huber
        # condition. The final column is the official PriorDA v1.1 fine output.
        scale_low_process = output["local_depth"].detach().float().squeeze().cpu().numpy()
    else:
        scale_low_process = (
            output["refinement_anchor_depth"] * torch.exp(output["low_log_residual"])
        ).clamp(1e-3, model.output_max_depth).detach().float().squeeze().cpu().numpy()
    final_process = output["depth"].detach().float().squeeze().cpu().numpy()
    learned_scale = (
        float(output["scale"].detach().float().item())
        if isinstance(model, BIMEarlyFusionDAv2JointScaleLow)
        else float((scale_process / np.maximum(base_depth, 1e-6)).mean())
    )

    predictions_process = {
        "raw": base_depth,
        "scale": scale_process,
        "scale_low": scale_low_process,
        "final": final_process,
    }
    predictions_full = {
        name: cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
        for name, value in predictions_process.items()
    }
    base_values = predictions_full["raw"][gt_valid].astype(np.float64)
    gt_values = gt[gt_valid].astype(np.float64)
    oracle_scale = float(np.median(gt_values / base_values))
    predictions_full["oracle_frame_scale"] = predictions_full["raw"] * oracle_scale

    camera_position = np.asarray(frame.camera_position, dtype=np.float64)
    camera_in_aabb = raycaster.contains_camera(camera_position, float(args.aabb_margin_m))
    gt_fraction = float(gt_valid.mean())
    bim_hit_fraction = float(bim_valid.mean())
    three_rule_pass = bool(
        gt_fraction > args.gt_min_valid_fraction
        and bim_hit_fraction > args.bim_min_hit_fraction
        and camera_in_aabb
    )
    reasons = []
    if not gt_fraction > args.gt_min_valid_fraction:
        reasons.append("sparse_gt")
    if not bim_hit_fraction > args.bim_min_hit_fraction:
        reasons.append("low_bim_hit")
    if not camera_in_aabb:
        reasons.append("camera_outside_bim_aabb")

    row: dict[str, Any] = {
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
        "gt_valid_fraction": gt_fraction,
        "camera_x": float(camera_position[0]),
        "camera_y": float(camera_position[1]),
        "camera_z": float(camera_position[2]),
        "camera_in_bim_aabb": camera_in_aabb,
        "bim_hit_pixels": int(bim_valid.sum()),
        "bim_hit_fraction": bim_hit_fraction,
        "three_rule_pass": three_rule_pass,
        "filter_reasons": ";".join(reasons),
        "deterministic_scale": deterministic_scale,
        "deterministic_scale_support": deterministic_support,
        "learned_scale": learned_scale,
        "learned_log_scale": math.log(learned_scale),
        "oracle_frame_scale": oracle_scale,
        "scale_abs_log_error": abs(math.log(learned_scale) - math.log(oracle_scale)),
        "confidence_source": confidence_source,
        "bim_render_seconds": render_seconds,
        "da3_inference_seconds": da3_seconds,
        "model_inference_seconds": model_seconds,
        "status": "ok",
        "error": "",
    }
    iteration_scales = output.get("attention_iteration_log_scales")
    if iteration_scales is None:
        iteration_scales = output.get("scale_iteration_log_scales")
    if iteration_scales is not None:
        values = iteration_scales.detach().float().reshape(-1).cpu().numpy()
        for index, value in enumerate(values, start=1):
            row[f"round_{index}_log_scale"] = float(value)
    for name, prediction in predictions_full.items():
        row.update(BENCHMARK._prefixed(name, BENCHMARK.metric_values(prediction, gt, gt_valid)))
    return row


def _micro_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
    weights = np.asarray([int(row["gt_valid_pixels"]) for row in rows], dtype=np.float64)
    output: dict[str, float | int] = {
        metric: float(np.average([float(row[f"{prefix}_{metric}"]) for row in rows], weights=weights))
        for metric in LINEAR_MICRO_METRICS
    }
    for metric in ("rmse_m", "rmse_log"):
        output[metric] = float(
            math.sqrt(np.average([float(row[f"{prefix}_{metric}"]) ** 2 for row in rows], weights=weights))
        )
    output["silog_x100"] = float(
        100.0 * math.sqrt(max(0.0, float(output["rmse_log"]) ** 2 - float(output["mean_log_error"]) ** 2))
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
                metric: float(np.mean([float(row[f"{prefix}_{metric}"]) for row in selected]))
                for metric in METRICS
            },
        }
    raw_abs = float(predictions["raw"]["pixel_micro"]["abs_rel"])
    comparisons = {}
    for prefix in ("scale", "scale_low", "final"):
        value = float(predictions[prefix]["pixel_micro"]["abs_rel"])
        comparisons[f"{prefix}_vs_raw"] = {
            "pixel_micro_abs_rel_difference": value - raw_abs,
            "pixel_micro_abs_rel_relative_improvement": (raw_abs - value) / raw_abs,
            "frame_win_fraction": float(
                np.mean([float(row[f"{prefix}_abs_rel"]) < float(row["raw_abs_rel"]) for row in selected])
            ),
        }
    return {
        "frames": len(selected),
        "valid_pixels": int(sum(int(row["gt_valid_pixels"]) for row in selected)),
        "predictions": predictions,
        "comparisons": comparisons,
    }


def _frame_set_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(sorted(str(row["frame_id"]) for row in rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    model: BIMPriorDA3 | FrozenHuberDAv2LowRefiner | FrozenHuberPriorDAV11BIM | BIMEarlyFusionDAv2JointScaleLow,
    scene_id: str,
    bimnet_key: str,
    mesh_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    valid = [row for row in ok if bool(row.get("three_rule_pass"))]
    frame_sha = _frame_set_sha256(valid)
    # Fifteen Hxp frames have no positive GT and are expected to be marked as
    # skipped_bad_gt.  They still count as successfully audited source frames.
    complete_scene = len(rows) == 792 and not any(row.get("status") == "error" for row in rows)
    if complete_scene and frame_sha != EXPECTED_FRAME_SET_SHA256:
        raise RuntimeError(
            "Three-rule frame set changed: "
            f"{frame_sha} != expected {EXPECTED_FRAME_SET_SHA256}"
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": "frozen Area_1 scale+refiner zero-shot; three-rule valid-frame benchmark",
            "rules": [
                f"GT positive-depth fraction > {args.gt_min_valid_fraction}",
                f"BIM ray-hit fraction > {args.bim_min_hit_fraction}",
                "camera center inside BIM AABB",
            ],
            "excluded_selection_rules": [
                "BIM/GT agreement",
                "BIM/DA3 ratio support",
                "confidence or model prediction error",
            ],
            "depth_support": "all finite positive Matterport GT depth",
            "aggregation": "pixel-micro; no scale/affine alignment",
            "gt_is_model_input": False,
        },
        "scene": {"matterport_scene_id": scene_id, "bimnet_scene_key": bimnet_key},
        "model": {
            "config": str(args.config.expanduser().resolve()),
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": BENCHMARK._sha256(args.checkpoint),
            "architecture": (
                (
                    "single early-fusion DAv2 global scale + native 72x72 r_low"
                    if model.residual_mode == "low72_only"
                    else (
                        "single early-fusion DAv2 global scale + native 36x36 r_low"
                        if model.residual_mode == "low36_only"
                        else "single early-fusion DAv2 global scale + native 18/36 Laplacian r_low"
                    )
                )
                if isinstance(model, BIMEarlyFusionDAv2JointScaleLow)
                else (
                    "frozen 3-round iterative-attention Huber + official PriorDA v1.1 BIM fine stage"
                    if isinstance(model, FrozenHuberPriorDAV11BIM)
                    else (
                        "frozen 3-round iterative-attention Huber + pretrained DINOv2/DPT r_low"
                        if isinstance(model, FrozenHuberDAv2LowRefiner)
                        else "3-round attention scale + r_low + r_detail"
                    )
                )
            ),
            "da3_feature_layers": (
                list(model.da3_feature_layers)
                if isinstance(model, BIMPriorDA3) and model.da3_feature_fusion_enabled
                else []
            ),
            "da3_feature_fusion_enabled": (
                model.da3_feature_fusion_enabled if isinstance(model, BIMPriorDA3) else False
            ),
            "da3_model": args.da3_model,
            "da3_revision": args.da3_revision,
            "focal_correction": "mean(processed fx, fy) / 300",
        },
        "bim": dict(mesh_metadata),
        "selection": {
            "frames": len(valid),
            "frame_ids_sha256": frame_sha,
            "expected_complete_scene_sha256": EXPECTED_FRAME_SET_SHA256,
            "matches_expected": frame_sha == EXPECTED_FRAME_SET_SHA256,
        },
        "row_counts": {
            "unique_frames": len(rows),
            "ok": len(ok),
            "error": sum(row.get("status") == "error" for row in rows),
            "skipped_bad_gt": sum(row.get("status") == "skipped_bad_gt" for row in rows),
        },
        "subsets": {
            "all_gt_valid": aggregate_rows(ok),
            "three_rule_valid": aggregate_rows(valid),
        },
    }


def _csv_columns() -> list[str]:
    identifiers = [
        "scene_id", "panorama_id", "frame_id", "camera_index", "yaw_index",
        "rgb_path", "depth_path", "height", "width", "process_height", "process_width",
        "focal_scale", "gt_valid_pixels", "gt_valid_fraction",
    ]
    diagnostic = [
        "camera_x", "camera_y", "camera_z", "camera_in_bim_aabb", "bim_hit_pixels",
        "bim_hit_fraction", "three_rule_pass", "filter_reasons", "deterministic_scale",
        "deterministic_scale_support", "learned_scale", "learned_log_scale",
        "oracle_frame_scale", "scale_abs_log_error", "confidence_source",
        "round_1_log_scale", "round_2_log_scale", "round_3_log_scale",
    ]
    metrics = [f"{prefix}_{metric}" for prefix in PREDICTION_NAMES for metric in METRICS]
    timing = ["bim_render_seconds", "da3_inference_seconds", "model_inference_seconds", "status", "error"]
    return [*identifiers, *diagnostic, *metrics, *timing]


def _load_frozen_huber_dpt_model(
    cfg: Any,
    checkpoint: Mapping[str, Any],
) -> FrozenHuberDAv2LowRefiner:
    if "trainable_model" not in checkpoint:
        raise KeyError("DPT r_low checkpoint lacks trainable_model")
    expected_scale_sha = str(cfg.model.frozen_scale.checkpoint_sha256)
    if str(checkpoint.get("frozen_scale_checkpoint_sha256")) != expected_scale_sha:
        raise RuntimeError("DPT checkpoint refers to a different frozen scale checkpoint")
    expected_dav2_sha = str(cfg.model.dav2.checkpoint_sha256)
    if str(checkpoint.get("official_dav2_checkpoint_sha256")) != expected_dav2_sha:
        raise RuntimeError("DPT checkpoint refers to a different official DAv2 checkpoint")
    scale_path = resolve_project_path(cfg, cfg.model.frozen_scale.checkpoint)
    if BENCHMARK._sha256(scale_path) != expected_scale_sha:
        raise RuntimeError("Local frozen scale checkpoint SHA256 differs from config")
    scale_checkpoint = torch.load(scale_path, map_location="cpu", weights_only=False)
    model = FrozenHuberDAv2LowRefiner.from_checkpoints(
        cfg,
        scale_checkpoint=scale_checkpoint,
    )
    prefixes = (
        "refiner.dav2.backbone.",
        "refiner.dav2.neck.",
        "refiner.bim_condition_embed.",
        "refiner.low_output.",
    )
    state = checkpoint["trainable_model"]
    expected = {name for name in model.state_dict() if name.startswith(prefixes)}
    if set(state) != expected:
        raise RuntimeError(
            "DPT trainable-state contract changed: "
            f"missing={sorted(expected - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    merged = model.state_dict()
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    return model


def _load_frozen_huber_priorda_model(
    cfg: Any,
    checkpoint: Mapping[str, Any],
) -> FrozenHuberPriorDAV11BIM:
    if checkpoint.get("architecture") != "official_priorda_v1_1_bim_global_local_condition":
        raise RuntimeError("Checkpoint is not the PriorDA v1.1 BIM adapter")
    if "trainable_model" not in checkpoint:
        raise KeyError("PriorDA-BIM checkpoint lacks trainable_model")
    expected_scale_sha = str(cfg.model.frozen_scale.checkpoint_sha256)
    if str(checkpoint.get("frozen_scale_checkpoint_sha256")) != expected_scale_sha:
        raise RuntimeError("PriorDA checkpoint refers to a different frozen scale checkpoint")
    expected_priorda_sha = str(cfg.model.priorda_v11.checkpoint_sha256)
    if str(checkpoint.get("official_priorda_checkpoint_sha256")) != expected_priorda_sha:
        raise RuntimeError("PriorDA checkpoint refers to a different official base checkpoint")
    expected_commit = str(cfg.model.priorda_v11.official_repository_commit)
    if str(checkpoint.get("official_priorda_repository_commit")) != expected_commit:
        raise RuntimeError("PriorDA checkpoint refers to a different official source commit")

    scale_path = resolve_project_path(cfg, cfg.model.frozen_scale.checkpoint)
    if BENCHMARK._sha256(scale_path) != expected_scale_sha:
        raise RuntimeError("Local frozen scale checkpoint SHA256 differs from config")
    prior_cfg = cfg.model.priorda_v11
    priorda_path = Path(
        hf_hub_download(
            repo_id=str(prior_cfg.checkpoint_repo),
            filename=str(prior_cfg.checkpoint_filename),
            revision=str(prior_cfg.checkpoint_revision),
            local_files_only=bool(prior_cfg.local_files_only),
        )
    ).resolve()
    if BENCHMARK._sha256(priorda_path) != expected_priorda_sha:
        raise RuntimeError("Local official PriorDA checkpoint SHA256 differs from config")
    scale_checkpoint = torch.load(scale_path, map_location="cpu", weights_only=False)
    model = FrozenHuberPriorDAV11BIM.from_checkpoints(
        cfg,
        scale_checkpoint=scale_checkpoint,
        priorda_checkpoint_path=priorda_path,
    )
    state = checkpoint["trainable_model"]
    expected = {name for name in model.state_dict() if name.startswith("priorda.")}
    if set(state) != expected:
        raise RuntimeError(
            "PriorDA trainable-state contract changed: "
            f"missing={sorted(expected - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    merged = model.state_dict()
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    return model


def _load_joint_dav2_scale_low_model(
    cfg: Any,
    checkpoint: Mapping[str, Any],
) -> BIMEarlyFusionDAv2JointScaleLow:
    expected_architectures = {
        "dav2_early_fusion_joint_global_scale_laplacian_low18_low36",
        "dav2_early_fusion_joint_global_scale_low36",
        "dav2_early_fusion_joint_global_scale_low72",
    }
    if checkpoint.get("architecture") not in expected_architectures:
        raise RuntimeError("Checkpoint is not the registered joint DAv2 scale+r_low model")
    joint = cfg.model.dav2_joint_scale_low
    dav2 = cfg.model.dav2
    model = BIMEarlyFusionDAv2JointScaleLow.from_pretrained(
        str(dav2.model_id),
        revision=str(dav2.revision),
        local_files_only=bool(dav2.local_files_only),
        regression_hidden_size=int(joint.regression_hidden_size),
        head_dropout_probability=float(joint.head_dropout_probability),
        output_weight_std=float(joint.output_weight_std),
        residual_hidden_channels=int(joint.residual_hidden_channels),
        max_low1_log_residual=float(joint.max_low1_log_residual),
        max_low2_log_residual=float(joint.max_low2_log_residual),
        output_max_depth_m=float(cfg.model.output_max_depth_m),
        residual_mode=str(getattr(joint, "residual_mode", "low18_low36")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


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
    if "dav2_joint_scale_low" in cfg.model:
        model = _load_joint_dav2_scale_low_model(cfg, checkpoint)
    elif "priorda_v11" in cfg.model:
        model = _load_frozen_huber_priorda_model(cfg, checkpoint)
    elif "dav2_low_refiner" in cfg.model:
        model = _load_frozen_huber_dpt_model(cfg, checkpoint)
    else:
        model = BIMPriorDA3(cfg)
        validate_checkpoint_model_config(checkpoint, cfg.model)
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    if isinstance(model, BIMPriorDA3):
        if not model.attention_scale_enabled or model.attention_scale is None:
            raise RuntimeError("Checkpoint does not contain the learned scale head")
        if model.use_frame_residual or not model.use_low_residual:
            raise RuntimeError("Expected released r_low+r_detail refiner with frame residual disabled")
    feature_layers = (
        tuple(int(value) for value in model.da3_feature_layers)
        if isinstance(model, BIMPriorDA3) and model.da3_feature_fusion_enabled
        else ()
    )
    if feature_layers and len(feature_layers) != 2:
        raise RuntimeError(f"Expected two DA3 feature layers, got {feature_layers}")

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

    prior_rows = BENCHMARK._read_latest_rows(csv_path)
    completed = {f"{row['scene_id']}/{row['frame_id']}" for row in prior_rows}
    pending = [frame for frame in frames if f"{frame.scene_id}/{frame.frame_id}" not in completed]
    print(
        f"scene={matterport_scene.scene_id} bim={bim_scene.key} frames={len(frames)} "
        f"completed={len(frames) - len(pending)} pending={len(pending)} device={device}",
        flush=True,
    )
    print(
        f"checkpoint={args.checkpoint} layers={feature_layers} wall_filled={wall_filled}; "
        "GT is used only for scoring and the frozen three-rule selection",
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
                    model=model,
                    raycaster=raycaster,
                    args=args,
                    feature_layers=feature_layers,
                    bim_log_mean=(
                        float(cfg.model.bim_normalization.mean)
                        if isinstance(model, BIMEarlyFusionDAv2JointScaleLow)
                        else None
                    ),
                    bim_log_std=(
                        float(cfg.model.bim_normalization.std)
                        if isinstance(model, BIMEarlyFusionDAv2JointScaleLow)
                        else None
                    ),
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
                elapsed = time.perf_counter() - run_start
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else float("nan")
                metric_text = (
                    f" raw={row['raw_abs_rel']:.5f} scale={row['scale_abs_rel']:.5f} "
                    f"final={row['final_abs_rel']:.5f}"
                    if row.get("status") == "ok" else ""
                )
                print(
                    f"progress={index}/{len(pending)} frame={frame.frame_id}{metric_text} "
                    f"selected={row.get('three_rule_pass')} rate={rate:.3f}fps eta_min={eta / 60:.1f}",
                    flush=True,
                )

    rows = BENCHMARK._read_latest_rows(csv_path)
    summary = build_summary(
        rows,
        args=args,
        model=model,
        scene_id=matterport_scene.scene_id,
        bimnet_key=bim_scene.key,
        mesh_metadata=mesh_metadata,
    )
    summary["artifacts"] = {
        "per_frame_csv": str(csv_path),
        "per_frame_csv_sha256": BENCHMARK._sha256(csv_path),
    }
    BENCHMARK._atomic_json(summary_path, summary)
    valid = summary["subsets"]["three_rule_valid"]
    predictions = valid.get("predictions", {})
    print(
        f"COMPLETE rows={len(rows)}/{len(frames)} selected={valid.get('frames', 0)} "
        + " ".join(
            f"{name}_abs_rel={predictions.get(name, {}).get('pixel_micro', {}).get('abs_rel')}"
            for name in ("raw", "scale", "scale_low", "final")
        )
        + f" summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
