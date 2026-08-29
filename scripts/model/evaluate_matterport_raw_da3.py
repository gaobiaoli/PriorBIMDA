#!/usr/bin/env python3
"""Evaluate standalone focal-corrected DA3METRIC on Matterport3D RGB-D frames.

The script deliberately runs every perspective frame independently.  It reuses
``Matterport3DDataset`` from S3-SAM3D-ToolKit for calibrated frame discovery and
Matterport depth decoding, while keeping DA3 inference inside PriorBIMDA's pinned
environment.  Successful rows are appended immediately, so a stopped run can be
resumed without repeating completed frames.
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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


DEFAULT_MODEL = "depth-anything/da3metric-large"
DEFAULT_REVISION = "4010e39f3634a45bc60553321fb49fb760bd594e"
CANONICAL_FOCAL_PX = 300.0
PROCESS_RES_METHOD = "upper_bound_resize"
PATCH_SIZE = 14

CSV_COLUMNS = [
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
    "fx_process_px",
    "fy_process_px",
    "focal_scale",
    "valid_pixels",
    "total_pixels",
    "valid_fraction",
    "gt_min_m",
    "gt_mean_m",
    "gt_median_m",
    "gt_max_m",
    "prediction_min_m",
    "prediction_mean_m",
    "prediction_median_m",
    "prediction_max_m",
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
    "canonical_abs_rel",
    "canonical_rmse_m",
    "oracle_median_scale",
    "oracle_scaled_abs_rel",
    "inference_seconds",
    "status",
    "error",
]

INTEGER_COLUMNS = {
    "camera_index",
    "yaw_index",
    "height",
    "width",
    "process_height",
    "process_width",
    "valid_pixels",
    "total_pixels",
}
FLOAT_COLUMNS = set(CSV_COLUMNS) - {
    "scene_id",
    "panorama_id",
    "frame_id",
    "rgb_path",
    "depth_path",
    "status",
    "error",
} - INTEGER_COLUMNS

SUMMARY_METRICS = [
    "abs_rel",
    "sq_rel",
    "rmse_m",
    "mae_m",
    "rmse_log",
    "mean_abs_log_error",
    "log10_error",
    "silog_x100",
    "delta1",
    "delta2",
    "delta3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--excel-every-scenes", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow Hugging Face network access instead of requiring the local pinned model.",
    )
    return parser.parse_args()


def _nearest_multiple(value: int, multiple: int = PATCH_SIZE) -> int:
    down = (value // multiple) * multiple
    up = down + multiple
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def processed_geometry(
    height: int,
    width: int,
    intrinsics: np.ndarray,
    process_res: int,
) -> tuple[int, int, np.ndarray, float]:
    """Reproduce DA3 ``upper_bound_resize`` geometry without decoding RGB twice."""

    if min(height, width, process_res) < 1:
        raise ValueError("image dimensions and process_res must be positive")
    if np.asarray(intrinsics).shape != (3, 3):
        raise ValueError("intrinsics must have shape (3, 3)")
    boundary_scale = process_res / float(max(height, width))
    resized_width = max(1, int(round(width * boundary_scale)))
    resized_height = max(1, int(round(height * boundary_scale)))
    process_width = _nearest_multiple(resized_width)
    process_height = _nearest_multiple(resized_height)
    processed_intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    processed_intrinsics[0] *= process_width / float(width)
    processed_intrinsics[1] *= process_height / float(height)
    focal_px = float((processed_intrinsics[0, 0] + processed_intrinsics[1, 1]) / 2.0)
    focal_scale = focal_px / CANONICAL_FOCAL_PX
    if not np.isfinite(focal_scale) or focal_scale <= 0:
        raise ValueError(f"invalid processed focal scale {focal_scale}")
    return process_height, process_width, processed_intrinsics, focal_scale


def metric_values(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("prediction, target and valid mask shapes must agree")
    support = np.asarray(valid, dtype=bool)
    count = int(np.count_nonzero(support))
    if count == 0:
        raise ValueError("frame has no valid Matterport depth pixels")
    pred = np.asarray(prediction[support], dtype=np.float64)
    gt = np.asarray(target[support], dtype=np.float64)
    if not np.isfinite(pred).all() or np.any(pred <= 0):
        raise ValueError("DA3 prediction must be finite and positive on GT support")
    difference = pred - gt
    absolute = np.abs(difference)
    squared = difference * difference
    log_difference = np.log(pred) - np.log(gt)
    ratio = np.maximum(pred / gt, gt / pred)
    log_variance = max(0.0, float(np.mean(log_difference**2) - np.mean(log_difference) ** 2))
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


def _frame_key(scene_id: str, frame_id: str) -> str:
    return f"{scene_id}/{frame_id}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_keys(rows: Iterable[dict[str, str]]) -> set[str]:
    return {
        _frame_key(row["scene_id"], row["frame_id"])
        for row in rows
        if row.get("status") in {"ok", "skipped_bad_gt"}
    }


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    for key in INTEGER_COLUMNS:
        value = result.get(key, "")
        result[key] = int(value) if value not in {"", None} else None
    for key in FLOAT_COLUMNS:
        value = result.get(key, "")
        result[key] = float(value) if value not in {"", None} else None
    return result


def _latest_completed_rows(csv_path: Path) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, str]] = {}
    for row in _read_rows(csv_path):
        if row.get("status") in {"ok", "skipped_bad_gt"}:
            latest[_frame_key(row["scene_id"], row["frame_id"])] = row
    return [_coerce_row(latest[key]) for key in sorted(latest)]


def _metric_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"frames": 0, "valid_pixels": 0}
    counts = np.asarray([row["valid_pixels"] for row in rows], dtype=np.float64)
    total = float(counts.sum())
    result: dict[str, Any] = {
        "frames": len(rows),
        "valid_pixels": int(total),
        "frame_macro": {},
        "pixel_micro": {},
    }
    for key in SUMMARY_METRICS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result["frame_macro"][key] = float(values.mean())
        if key == "rmse_m":
            result["pixel_micro"][key] = float(np.sqrt(np.sum(values**2 * counts) / total))
        elif key == "rmse_log":
            result["pixel_micro"][key] = float(np.sqrt(np.sum(values**2 * counts) / total))
        elif key == "silog_x100":
            mean_log = np.asarray([row["mean_log_error"] for row in rows], dtype=np.float64)
            second_moment = (values / 100.0) ** 2 + mean_log**2
            global_mean = float(np.sum(mean_log * counts) / total)
            global_second = float(np.sum(second_moment * counts) / total)
            result["pixel_micro"][key] = 100.0 * math.sqrt(
                max(0.0, global_second - global_mean**2)
            )
        else:
            result["pixel_micro"][key] = float(np.sum(values * counts) / total)
    return result


def build_summary(
    rows: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    scene_ids: list[str],
    started_at: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    return {
        "schema_version": 1,
        "protocol": "matterport3d-standalone-da3metric-focal-corrected-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "matterport_root": str(args.matterport_root.expanduser().resolve()),
        "toolkit_root": str(args.toolkit_root.expanduser().resolve()),
        "model": args.model,
        "revision": args.revision,
        "process_res": args.process_res,
        "process_res_method": PROCESS_RES_METHOD,
        "canonical_focal_px": CANONICAL_FOCAL_PX,
        "metric_depth_formula": (
            "DA3 canonical-focal z-depth * mean(fx_processed,fy_processed)/300"
        ),
        "ground_truth": "Matterport undistorted z-depth uint16 / 4000 metres",
        "evaluation_support": "all GT pixels with raw depth > 0; no metric cutoff",
        "frame_inference": "each perspective RGB is inferred independently",
        "requested_scenes": scene_ids,
        "completed_scenes": sorted(grouped),
        "evaluated_frames": len(rows),
        "overall": aggregate_rows(rows),
        "by_scene": {scene: aggregate_rows(grouped[scene]) for scene in sorted(grouped)},
    }


def _summary_table(summary: dict[str, Any]) -> list[list[Any]]:
    header = ["scope", "frames", "valid_pixels", "aggregation", *SUMMARY_METRICS]
    table = [header]
    scopes = [("all", summary["overall"]), *summary["by_scene"].items()]
    for scope, aggregate in scopes:
        for aggregation in ("pixel_micro", "frame_macro"):
            table.append(
                [
                    scope,
                    aggregate.get("frames", 0),
                    aggregate.get("valid_pixels", 0),
                    aggregation,
                    *[aggregate.get(aggregation, {}).get(key) for key in SUMMARY_METRICS],
                ]
            )
    return table


def export_excel(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        from openpyxl import Workbook
        from openpyxl.cell import WriteOnlyCell
        from openpyxl.styles import Font
    except ImportError as error:
        raise RuntimeError("Excel export requires openpyxl") from error

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook(write_only=True)

    def append_table(sheet_name: str, table: Iterable[list[Any]]) -> None:
        sheet = workbook.create_sheet(sheet_name)
        for index, values in enumerate(table):
            if index == 0:
                cells = []
                for value in values:
                    cell = WriteOnlyCell(sheet, value=value)
                    cell.font = Font(bold=True)
                    cells.append(cell)
                sheet.append(cells)
            else:
                sheet.append(values)

    summary_table = _summary_table(summary)
    append_table("overall", [summary_table[0], *summary_table[1:3]])
    append_table("by_scene", [summary_table[0], *summary_table[3:]])
    append_table(
        "per_frame",
        [CSV_COLUMNS, *[[row.get(column) for column in CSV_COLUMNS] for row in rows]],
    )
    protocol_keys = [
        "protocol",
        "created_at",
        "started_at",
        "matterport_root",
        "toolkit_root",
        "model",
        "revision",
        "process_res",
        "process_res_method",
        "canonical_focal_px",
        "metric_depth_formula",
        "ground_truth",
        "evaluation_support",
        "frame_inference",
    ]
    append_table("protocol", [["key", "value"], *[[key, summary[key]] for key in protocol_keys]])
    temporary = path.with_suffix(path.suffix + ".tmp")
    workbook.save(temporary)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _base_row(frame: Any) -> dict[str, Any]:
    return {
        "scene_id": frame.scene_id,
        "panorama_id": frame.panorama_id,
        "frame_id": frame.frame_id,
        "camera_index": frame.camera_index,
        "yaw_index": frame.yaw_index,
        "rgb_path": str(frame.rgb_path),
        "depth_path": str(frame.depth_path),
    }


def evaluate_frame(model: Any, frame: Any, process_res: int) -> dict[str, Any]:
    height, width = frame.image_shape
    process_height, process_width, processed_k, focal_scale = processed_geometry(
        height, width, frame.intrinsics, process_res
    )
    gt = np.asarray(frame.depth, dtype=np.float32)
    valid = np.isfinite(gt) & (gt > 0)
    if not np.any(valid):
        return {
            **_base_row(frame),
            "height": height,
            "width": width,
            "process_height": process_height,
            "process_width": process_width,
            "fx_process_px": float(processed_k[0, 0]),
            "fy_process_px": float(processed_k[1, 1]),
            "focal_scale": focal_scale,
            "valid_pixels": 0,
            "total_pixels": int(valid.size),
            "valid_fraction": 0.0,
            "inference_seconds": 0.0,
            "status": "skipped_bad_gt",
            "error": "Matterport depth contains no finite positive GT pixels",
        }
    start = time.perf_counter()
    with torch.inference_mode():
        result = model.inference(
            [str(frame.rgb_path)],
            process_res=process_res,
            process_res_method=PROCESS_RES_METHOD,
            export_dir=None,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    canonical = np.asarray(result.depth[0], dtype=np.float32)
    if canonical.shape != (process_height, process_width):
        raise RuntimeError(
            f"DA3 output shape {canonical.shape} != expected {(process_height, process_width)}"
        )
    canonical = cv2.resize(canonical, (width, height), interpolation=cv2.INTER_LINEAR)
    prediction = canonical * focal_scale
    metrics = metric_values(prediction, gt, valid)
    canonical_metrics = metric_values(canonical, gt, valid)
    pred_values = prediction[valid].astype(np.float64)
    gt_values = gt[valid].astype(np.float64)
    oracle_scale = float(np.median(gt_values / pred_values))
    oracle_abs_rel = float(np.mean(np.abs(pred_values * oracle_scale - gt_values) / gt_values))
    return {
        **_base_row(frame),
        "height": height,
        "width": width,
        "process_height": process_height,
        "process_width": process_width,
        "fx_process_px": float(processed_k[0, 0]),
        "fy_process_px": float(processed_k[1, 1]),
        "focal_scale": focal_scale,
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
        "valid_fraction": float(valid.mean()),
        "gt_min_m": float(gt_values.min()),
        "gt_mean_m": float(gt_values.mean()),
        "gt_median_m": float(np.median(gt_values)),
        "gt_max_m": float(gt_values.max()),
        "prediction_min_m": float(pred_values.min()),
        "prediction_mean_m": float(pred_values.mean()),
        "prediction_median_m": float(np.median(pred_values)),
        "prediction_max_m": float(pred_values.max()),
        **metrics,
        "canonical_abs_rel": canonical_metrics["abs_rel"],
        "canonical_rmse_m": canonical_metrics["rmse_m"],
        "oracle_median_scale": oracle_scale,
        "oracle_scaled_abs_rel": oracle_abs_rel,
        "inference_seconds": inference_seconds,
        "status": "ok",
        "error": "",
    }


def main() -> None:
    args = parse_args()
    if args.process_res < 1 or args.progress_every < 1 or args.excel_every_scenes < 1:
        raise ValueError("process-res/progress-every/excel-every-scenes must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max-frames must be positive")

    toolkit_src = args.toolkit_root.expanduser().resolve() / "src"
    sys.path.insert(0, str(toolkit_src))
    from s3dis_sam3d import Matterport3DDataset
    from depth_anything_3.api import DepthAnything3

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_frame.csv"
    summary_path = output_dir / "summary.json"
    excel_path = output_dir / "matterport3d_raw_da3_per_frame.xlsx"
    if args.no_resume and csv_path.exists():
        raise FileExistsError(f"--no-resume refuses existing output: {csv_path}")

    dataset = Matterport3DDataset(args.matterport_root)
    scene_ids = list(args.scenes) if args.scenes else dataset.scene_ids
    unknown = sorted(set(scene_ids) - set(dataset.scene_ids))
    if unknown:
        raise ValueError(f"Unknown Matterport scenes: {unknown}")
    frames = [frame for scene_id in scene_ids for frame in dataset[scene_id].frames]
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    prior_rows = _read_rows(csv_path)
    completed = _completed_keys(prior_rows)
    pending = [
        frame for frame in frames if _frame_key(frame.scene_id, frame.frame_id) not in completed
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[{started_at}] scenes={len(scene_ids)} frames={len(frames)} "
        f"completed={len(frames)-len(pending)} pending={len(pending)}",
        flush=True,
    )
    print(
        f"protocol=standalone-single-view model={args.model}@{args.revision} "
        f"process_res={args.process_res} focal_correction=mean(fx,fy)/300",
        flush=True,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = (
        DepthAnything3.from_pretrained(
            args.model,
            revision=args.revision,
            local_files_only=not args.allow_network,
        )
        .to(device)
        .eval()
    )
    print(f"model_loaded device={device}", flush=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    scene_progress: dict[str, int] = defaultdict(int)
    errors = 0
    skipped_bad_gt = 0
    run_start = time.perf_counter()
    with csv_path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
            handle.flush()
        for index, frame in enumerate(pending, start=1):
            try:
                row = evaluate_frame(model, frame, args.process_res)
                scene_progress[frame.scene_id] += 1
                if row["status"] == "skipped_bad_gt":
                    skipped_bad_gt += 1
                    print(
                        f"SKIP_BAD_GT {_frame_key(frame.scene_id, frame.frame_id)}",
                        flush=True,
                    )
            except Exception as error:  # keep a long benchmark alive and make failures visible
                errors += 1
                row = {
                    **_base_row(frame),
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
                print(f"ERROR {_frame_key(frame.scene_id, frame.frame_id)} {row['error']}", flush=True)
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
            handle.flush()
            if index == 1 or index % args.progress_every == 0 or index == len(pending):
                elapsed = time.perf_counter() - run_start
                rate = index / elapsed if elapsed > 0 else 0.0
                remaining = (len(pending) - index) / rate if rate > 0 else float("nan")
                metric = row.get("abs_rel")
                metric_text = f" abs_rel={metric:.6f}" if metric is not None else ""
                print(
                    f"progress={index}/{len(pending)} total_done={len(frames)-len(pending)+index}/"
                    f"{len(frames)} scene={frame.scene_id} frame={frame.frame_id}{metric_text} "
                    f"rate={rate:.3f}fps eta_hours={remaining/3600:.2f} "
                    f"skipped_bad_gt={skipped_bad_gt} errors={errors}",
                    flush=True,
                )

            current_scene_done = index == len(pending) or pending[index].scene_id != frame.scene_id
            scenes_finished = sum(1 for value in scene_progress.values() if value > 0)
            if current_scene_done and scenes_finished % args.excel_every_scenes == 0:
                completed_rows = _latest_completed_rows(csv_path)
                rows = _metric_rows(completed_rows)
                summary = build_summary(
                    rows, args=args, scene_ids=scene_ids, started_at=started_at
                )
                summary["skipped_bad_gt_frames"] = sum(
                    row["status"] == "skipped_bad_gt" for row in completed_rows
                )
                _atomic_json(summary_path, summary)
                export_excel(excel_path, completed_rows, summary)
                print(
                    f"checkpoint_export evaluated={len(rows)} "
                    f"skipped_bad_gt={summary['skipped_bad_gt_frames']} "
                    f"summary={summary_path} excel={excel_path}",
                    flush=True,
                )

    completed_rows = _latest_completed_rows(csv_path)
    rows = _metric_rows(completed_rows)
    summary = build_summary(rows, args=args, scene_ids=scene_ids, started_at=started_at)
    summary["skipped_bad_gt_frames"] = sum(
        row["status"] == "skipped_bad_gt" for row in completed_rows
    )
    summary["artifacts"] = {
        "per_frame_csv": str(csv_path),
        "per_frame_csv_sha256": _sha256(csv_path),
        "excel": str(excel_path),
    }
    _atomic_json(summary_path, summary)
    export_excel(excel_path, completed_rows, summary)
    summary["artifacts"]["excel_sha256"] = _sha256(excel_path)
    _atomic_json(summary_path, summary)
    print(
        f"COMPLETE evaluated={len(rows)} skipped_bad_gt={summary['skipped_bad_gt_frames']} "
        f"completed={len(completed_rows)}/{len(frames)} errors={errors} "
        f"abs_rel={summary['overall'].get('pixel_micro', {}).get('abs_rel')} "
        f"summary={summary_path} excel={excel_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
