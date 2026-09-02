#!/usr/bin/env python3
"""Evaluate DA3 Nested metric depth on the fixed MP3D three-scene set.

Each RGB frame is inferred independently.  Two zero-shot outputs are scored in
one pass: the model's native metric output, and a calibrated-focal variant that
replaces DA3's predicted focal scale with the known RGB camera focal length.
Neither output uses test depth for scale or affine alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

from evaluate_matterport_metric_foundations import (
    SUMMARY_METRICS,
    atomic_json,
    frame_key,
    metric_values,
    selected_hash,
    selected_manifest,
    sha256,
)
from evaluate_matterport_raw_da3 import processed_geometry


DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DEFAULT_REVISION = "b2359bdf726fb44ef62acca04d629dcf158053e7"
PROCESS_RES_METHOD = "upper_bound_resize"
MODES = ("native", "calibrated_focal")

IDENTITY_COLUMNS = [
    "scene_id", "bimnet_scene_key", "panorama_id", "frame_id", "camera_index",
    "yaw_index", "rgb_path", "depth_path", "height", "width", "process_height",
    "process_width", "true_fx_process_px", "true_fy_process_px",
    "predicted_fx_process_px", "predicted_fy_process_px", "focal_correction_ratio",
    "valid_pixels", "total_pixels", "valid_fraction",
]
MODE_VALUE_COLUMNS = [
    "prediction_clipped_pixels", "prediction_min_m", "prediction_mean_m",
    "prediction_median_m", "prediction_max_m", *SUMMARY_METRICS,
    "mean_log_error", "oracle_median_scale", "oracle_scaled_abs_rel",
]
CSV_COLUMNS = [
    *IDENTITY_COLUMNS,
    *[f"{mode}_{column}" for mode in MODES for column in MODE_VALUE_COLUMNS],
    "inference_seconds", "status", "error",
]
INTEGER_COLUMNS = {
    "camera_index", "yaw_index", "height", "width", "process_height",
    "process_width", "valid_pixels", "total_pixels",
    *[f"{mode}_prediction_clipped_pixels" for mode in MODES],
}
NON_NUMERIC_COLUMNS = {
    "scene_id", "bimnet_scene_key", "panorama_id", "frame_id", "rgb_path",
    "depth_path", "status", "error",
}
FLOAT_COLUMNS = set(CSV_COLUMNS) - INTEGER_COLUMNS - NON_NUMERIC_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--model-code-root", type=Path, default=None)
    parser.add_argument("--model-code-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def coerce_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    for key in INTEGER_COLUMNS:
        value = result.get(key, "")
        result[key] = int(value) if value not in {"", None} else None
    for key in FLOAT_COLUMNS:
        value = result.get(key, "")
        result[key] = float(value) if value not in {"", None} else None
    return result


def completed_rows(path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        if row.get("status") in {"ok", "skipped_bad_gt"}:
            latest[frame_key(row["scene_id"], row["frame_id"])] = row
    return [coerce_row(latest[key]) for key in sorted(latest)]


def aggregate_rows(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if not rows:
        return {"frames": 0, "valid_pixels": 0}
    counts = np.asarray([row["valid_pixels"] for row in rows], dtype=np.float64)
    total = float(counts.sum())
    result: dict[str, Any] = {
        "frames": len(rows), "valid_pixels": int(total), "frame_macro": {},
        "pixel_micro": {},
    }
    for metric in SUMMARY_METRICS:
        key = f"{mode}_{metric}"
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result["frame_macro"][metric] = float(values.mean())
        if metric in {"rmse_m", "rmse_log"}:
            micro = math.sqrt(float(np.sum(values**2 * counts) / total))
        elif metric == "silog_x100":
            mean_log = np.asarray(
                [row[f"{mode}_mean_log_error"] for row in rows], dtype=np.float64
            )
            second = (values / 100.0) ** 2 + mean_log**2
            global_mean = float(np.sum(mean_log * counts) / total)
            global_second = float(np.sum(second * counts) / total)
            micro = 100.0 * math.sqrt(max(0.0, global_second - global_mean**2))
        else:
            micro = float(np.sum(values * counts) / total)
        result["pixel_micro"][metric] = micro
    return result


def mode_values(prediction: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    bad = ~np.isfinite(prediction) | (prediction <= 0)
    clipped_pixels = int(np.count_nonzero(bad & valid))
    stable = np.nan_to_num(prediction, nan=1e-4, posinf=1e4, neginf=1e-4)
    stable = np.clip(stable, 1e-4, 1e4)
    metrics = metric_values(stable, gt, valid)
    pred_values = stable[valid].astype(np.float64)
    gt_values = gt[valid].astype(np.float64)
    oracle_scale = float(np.median(gt_values / pred_values))
    oracle_abs_rel = float(
        np.mean(np.abs(pred_values * oracle_scale - gt_values) / gt_values)
    )
    return {
        "prediction_clipped_pixels": clipped_pixels,
        "prediction_min_m": float(pred_values.min()),
        "prediction_mean_m": float(pred_values.mean()),
        "prediction_median_m": float(np.median(pred_values)),
        "prediction_max_m": float(pred_values.max()),
        **metrics,
        "oracle_median_scale": oracle_scale,
        "oracle_scaled_abs_rel": oracle_abs_rel,
    }


def prefix_values(mode: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{mode}_{key}": value for key, value in values.items()}


def base_row(frame: Any, scene_key: str) -> dict[str, Any]:
    return {
        "scene_id": frame.scene_id,
        "bimnet_scene_key": scene_key,
        "panorama_id": frame.panorama_id,
        "frame_id": frame.frame_id,
        "camera_index": frame.camera_index,
        "yaw_index": frame.yaw_index,
        "rgb_path": str(frame.rgb_path),
        "depth_path": str(frame.depth_path),
    }


def evaluate_frame(model: Any, frame: Any, scene_key: str, process_res: int) -> dict[str, Any]:
    height, width = frame.image_shape
    process_height, process_width, true_processed_k, _ = processed_geometry(
        height, width, np.asarray(frame.intrinsics, dtype=np.float32), process_res
    )
    gt = np.asarray(frame.depth, dtype=np.float32)
    valid = np.isfinite(gt) & (gt > 0)
    common = {
        **base_row(frame, scene_key),
        "height": height,
        "width": width,
        "process_height": process_height,
        "process_width": process_width,
        "true_fx_process_px": float(true_processed_k[0, 0]),
        "true_fy_process_px": float(true_processed_k[1, 1]),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "valid_fraction": float(valid.mean()),
    }
    if not np.any(valid):
        return {**common, "status": "skipped_bad_gt", "error": "no finite positive GT"}

    # DA3 Nested samples pixels while estimating its branch-alignment scale.  A
    # frame-derived seed makes interrupted/resumed evaluation bit-reproducible.
    seed = int.from_bytes(frame_key(frame.scene_id, frame.frame_id).encode("utf-8"), "little") % (2**31)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.inference(
            [str(frame.rgb_path)], process_res=process_res,
            process_res_method=PROCESS_RES_METHOD, export_dir=None,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start

    native = np.asarray(result.depth[0], dtype=np.float32)
    if native.shape != (process_height, process_width):
        raise RuntimeError(
            f"DA3 output shape {native.shape} != expected {(process_height, process_width)}"
        )
    if result.intrinsics is None or np.asarray(result.intrinsics).shape != (1, 3, 3):
        raise RuntimeError(f"DA3 predicted invalid intrinsics shape: {np.shape(result.intrinsics)}")
    predicted_k = np.asarray(result.intrinsics[0], dtype=np.float64)
    predicted_focal = float((predicted_k[0, 0] + predicted_k[1, 1]) / 2.0)
    true_focal = float((true_processed_k[0, 0] + true_processed_k[1, 1]) / 2.0)
    focal_ratio = true_focal / predicted_focal
    if not np.isfinite(focal_ratio) or focal_ratio <= 0:
        raise RuntimeError(f"invalid focal correction ratio: {focal_ratio}")

    native = cv2.resize(native, (width, height), interpolation=cv2.INTER_LINEAR)
    calibrated = native * focal_ratio
    return {
        **common,
        "predicted_fx_process_px": float(predicted_k[0, 0]),
        "predicted_fy_process_px": float(predicted_k[1, 1]),
        "focal_correction_ratio": focal_ratio,
        **prefix_values("native", mode_values(native, gt, valid)),
        **prefix_values("calibrated_focal", mode_values(calibrated, gt, valid)),
        "inference_seconds": inference_seconds,
        "status": "ok",
        "error": "",
    }


def build_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    selection_rows: list[dict[str, str]],
    started_at: str,
) -> dict[str, Any]:
    metric_rows = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["scene_id"]].append(row)
    expected_by_scene: dict[str, list[str]] = defaultdict(list)
    for row in selection_rows:
        expected_by_scene[row["matterport_scene_id"]].append(row["frame_id"])
    scores = {
        mode: {
            "overall": aggregate_rows(metric_rows, mode),
            "by_scene": {
                scene: aggregate_rows(grouped[scene], mode) for scene in sorted(grouped)
            },
        }
        for mode in MODES
    }
    focal_ratios = np.asarray(
        [row["focal_correction_ratio"] for row in metric_rows], dtype=np.float64
    )
    return {
        "schema_version": 1,
        "protocol": "matterport3d-fixed-three-rule-da3nested-single-view-metric-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "model": args.model,
        "revision": args.revision,
        "model_code_root": str(args.model_code_root.resolve()) if args.model_code_root else None,
        "model_code_revision": args.model_code_revision,
        "inference": {
            "frame_mode": "independent single image",
            "process_res": args.process_res,
            "process_res_method": PROCESS_RES_METHOD,
            "native": "official DA3 Nested output in metres; predicted intrinsics",
            "calibrated_focal": (
                "native depth * known processed mean focal / predicted processed mean focal"
            ),
            "camera_information": (
                "native is RGB-only; calibrated_focal uses GT RGB intrinsics but no GT depth"
            ),
            "test_depth_alignment": "none for native and calibrated_focal",
        },
        "ground_truth": "Matterport undistorted z-depth uint16 / 4000 metres; all positive pixels",
        "selection": {
            "csv": str(args.selection_csv.resolve()),
            "csv_sha256": sha256(args.selection_csv),
            "expected_frames": len(selection_rows),
            "by_scene": {
                scene: {"frames": len(ids), "frame_ids_sha256": selected_hash(ids)}
                for scene, ids in sorted(expected_by_scene.items())
            },
        },
        "row_counts": {
            "completed": len(rows),
            "ok": len(metric_rows),
            "skipped_bad_gt": sum(row.get("status") == "skipped_bad_gt" for row in rows),
        },
        "focal_correction_ratio": {
            "mean": float(focal_ratios.mean()) if len(focal_ratios) else None,
            "median": float(np.median(focal_ratios)) if len(focal_ratios) else None,
            "min": float(focal_ratios.min()) if len(focal_ratios) else None,
            "max": float(focal_ratios.max()) if len(focal_ratios) else None,
        },
        "scores": scores,
        "diagnostic_only": (
            "per-frame oracle_median_scale/oracle_scaled_abs_rel use test GT and are not headline metrics"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.process_res < 1 or args.progress_every < 1:
        raise ValueError("process-res and progress-every must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")
    if args.model_code_root:
        sys.path.insert(0, str(args.model_code_root.expanduser().resolve() / "src"))
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")

    selection_rows, scene_keys = selected_manifest(args.selection_csv.resolve())
    selected = {
        frame_key(row["matterport_scene_id"], row["frame_id"]): row
        for row in selection_rows
    }
    sys.path.insert(0, str(args.toolkit_root.expanduser().resolve() / "src"))
    from depth_anything_3.api import DepthAnything3
    from s3dis_sam3d import Matterport3DDataset

    dataset = Matterport3DDataset(args.matterport_root.expanduser().resolve())
    frames = [
        frame for scene_id in scene_keys for frame in dataset[scene_id].frames
        if frame_key(scene_id, frame.frame_id) in selected
    ]
    frames.sort(key=lambda frame: (frame.scene_id, frame.frame_id))
    found = {frame_key(frame.scene_id, frame.frame_id) for frame in frames}
    missing = sorted(set(selected) - found)
    if missing:
        raise RuntimeError(f"selected frames missing from dataset: {missing[:10]}")
    if args.max_frames is not None:
        frames = frames[:args.max_frames]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_frame.csv"
    summary_path = output_dir / "summary.json"
    if args.no_resume and csv_path.exists():
        raise FileExistsError(f"--no-resume refuses existing output: {csv_path}")
    prior = completed_rows(csv_path)
    done = {frame_key(row["scene_id"], row["frame_id"]) for row in prior}
    pending = [frame for frame in frames if frame_key(frame.scene_id, frame.frame_id) not in done]
    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[{started_at}] frames={len(frames)} completed={len(frames)-len(pending)} "
        f"pending={len(pending)} model={args.model}@{args.revision}", flush=True,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = None
    if pending:
        model = DepthAnything3.from_pretrained(
            args.model, revision=args.revision, local_files_only=args.local_files_only
        ).to(device).eval()
    print(f"model_loaded={bool(model)} device={device}", flush=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    errors = 0
    run_start = time.perf_counter()
    with csv_path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for index, frame in enumerate(pending, start=1):
            scene_key = scene_keys[frame.scene_id]
            try:
                row = evaluate_frame(model, frame, scene_key, args.process_res)
            except Exception as error:
                errors += 1
                row = {
                    **base_row(frame, scene_key), "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"ERROR {frame.scene_id}/{frame.frame_id} {row['error']}", flush=True)
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
            handle.flush()
            if index == 1 or index % args.progress_every == 0 or index == len(pending):
                elapsed = time.perf_counter() - run_start
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else float("nan")
                print(
                    f"progress={index}/{len(pending)} frame={frame.scene_id}/{frame.frame_id} "
                    f"native_abs_rel={row.get('native_abs_rel')} "
                    f"calibrated_abs_rel={row.get('calibrated_focal_abs_rel')} "
                    f"rate={rate:.3f}fps eta_hours={eta/3600:.2f} errors={errors}", flush=True,
                )

    rows = completed_rows(csv_path)
    summary = build_summary(args, rows, selection_rows, started_at)
    summary["artifacts"] = {
        "per_frame_csv": str(csv_path), "per_frame_csv_sha256": sha256(csv_path)
    }
    atomic_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), **summary["scores"], "row_counts": summary["row_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
