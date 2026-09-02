#!/usr/bin/env python3
"""Evaluate metric monocular foundation models on the fixed MP3D three-scene set.

Every RGB frame is inferred independently.  Camera intrinsics are allowed model
inputs, but Matterport depth is used only for scoring.  In particular, the
headline prediction receives no test-time GT scale or affine alignment.
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


BACKEND_DEFAULTS = {
    "metricanything_pointmap": {
        "model": "yjh001/metricanything_student_pointmap",
        "revision": "8f1f08c53c683d4e19864601dfaf78515f29a63b",
    },
    "metricanything_depthmap": {
        "model": "yjh001/metricanything_student_depthmap",
        "revision": "cae9b4eb052e827048c9b385366c6d0dce83fb01",
    },
    "moge3": {
        "model": "Ruicheng/moge-3-vitg",
        "revision": "6ef26c5a4b4148dab5ccaacdb08b72dc66380475",
    },
    "unidepth_v2": {
        "model": "lpiccinelli/unidepth-v2-vitl14",
        "revision": "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9",
    },
}

CSV_COLUMNS = [
    "scene_id", "bimnet_scene_key", "panorama_id", "frame_id", "camera_index",
    "yaw_index", "rgb_path", "depth_path", "height", "width", "fx_px", "fy_px",
    "cx_px", "cy_px", "fov_x_deg", "valid_pixels", "total_pixels", "valid_fraction",
    "prediction_clipped_pixels", "prediction_min_m", "prediction_mean_m",
    "prediction_median_m", "prediction_max_m", "abs_rel", "sq_rel", "rmse_m",
    "mae_m", "rmse_log", "mean_log_error", "mean_abs_log_error", "log10_error",
    "silog_x100", "delta1", "delta2", "delta3", "oracle_median_scale",
    "oracle_scaled_abs_rel", "inference_seconds", "status", "error",
]

INTEGER_COLUMNS = {
    "camera_index", "yaw_index", "height", "width", "valid_pixels", "total_pixels",
    "prediction_clipped_pixels",
}
NON_NUMERIC_COLUMNS = {
    "scene_id", "bimnet_scene_key", "panorama_id", "frame_id", "rgb_path",
    "depth_path", "status", "error",
}
FLOAT_COLUMNS = set(CSV_COLUMNS) - INTEGER_COLUMNS - NON_NUMERIC_COLUMNS
SUMMARY_METRICS = [
    "abs_rel", "sq_rel", "rmse_m", "mae_m", "rmse_log", "mean_abs_log_error",
    "log10_error", "silog_x100", "delta1", "delta2", "delta3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKEND_DEFAULTS), required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--model-code-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = [row for row in read_csv(path) if row.get("selected", "").lower() == "true"]
    if not rows:
        raise ValueError(f"selection contains no selected rows: {path}")
    scene_keys: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["matterport_scene_id"], row["frame_id"])
        if key in seen:
            raise ValueError(f"duplicate selected frame: {key}")
        seen.add(key)
        previous = scene_keys.setdefault(row["matterport_scene_id"], row["bimnet_scene_key"])
        if previous != row["bimnet_scene_key"]:
            raise ValueError(f"scene has multiple BIMNet keys: {row['matterport_scene_id']}")
    return rows, scene_keys


def frame_key(scene_id: str, frame_id: str) -> str:
    return f"{scene_id}/{frame_id}"


def selected_hash(frame_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(frame_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def metric_values(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction[valid], dtype=np.float64)
    gt = np.asarray(target[valid], dtype=np.float64)
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
        if key in {"rmse_m", "rmse_log"}:
            result["pixel_micro"][key] = float(np.sqrt(np.sum(values**2 * counts) / total))
        elif key == "silog_x100":
            mean_log = np.asarray([row["mean_log_error"] for row in rows], dtype=np.float64)
            second_moment = (values / 100.0) ** 2 + mean_log**2
            global_mean = float(np.sum(mean_log * counts) / total)
            global_second = float(np.sum(second_moment * counts) / total)
            result["pixel_micro"][key] = 100.0 * math.sqrt(max(0.0, global_second - global_mean**2))
        else:
            result["pixel_micro"][key] = float(np.sum(values * counts) / total)
    return result


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
    if not path.is_file():
        return []
    latest: dict[str, dict[str, str]] = {}
    for row in read_csv(path):
        if row.get("status") in {"ok", "skipped_bad_gt"}:
            latest[frame_key(row["scene_id"], row["frame_id"])] = row
    return [coerce_row(latest[key]) for key in sorted(latest)]


class Predictor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if args.model_code_root:
            sys.path.insert(0, str(args.model_code_root.expanduser().resolve()))
        self.model = self._load()

    def _load(self) -> Any:
        if self.args.backend == "metricanything_pointmap":
            from moge.model.v2 import MoGeModel
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(
                repo_id=self.args.model, filename="student_pointmap.pt",
                revision=self.args.revision,
                local_files_only=self.args.local_files_only,
            )
            return MoGeModel.from_pretrained(checkpoint).to(self.device).eval()
        if self.args.backend == "metricanything_depthmap":
            from depth_model import MetricAnythingDepthMap

            if self.args.model_code_root is None:
                raise ValueError("metricanything_depthmap requires --model-code-root")
            previous_cwd = Path.cwd()
            try:
                # The official model constructs DINOv3 through torch.hub with the
                # relative local repository name "network".
                os.chdir(self.args.model_code_root.expanduser().resolve())
                model = MetricAnythingDepthMap.from_pretrained(
                    self.args.model, filename="student_depthmap.pt",
                    revision=self.args.revision, local_files_only=self.args.local_files_only,
                )
            finally:
                os.chdir(previous_cwd)
            return model.to(self.device).eval()
        if self.args.backend == "moge3":
            from moge.model.v3 import MoGeModel

            return MoGeModel.from_pretrained(
                self.args.model, revision=self.args.revision,
                local_files_only=self.args.local_files_only,
            ).to(self.device).eval()
        if self.args.backend == "unidepth_v2":
            from unidepth.models import UniDepthV2

            return UniDepthV2.from_pretrained(
                self.args.model, revision=self.args.revision,
                local_files_only=self.args.local_files_only,
            ).to(self.device).eval()
        raise AssertionError(self.args.backend)

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        height, width = rgb.shape[:2]
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).to(self.device)
        if self.args.backend in {"metricanything_pointmap", "moge3"}:
            image = image.float().div_(255.0)
            fov_x = math.degrees(2.0 * math.atan(width / (2.0 * float(intrinsics[0, 0]))))
            kwargs = dict(
                resolution_level=self.args.resolution_level,
                fov_x=fov_x,
                apply_mask=False,
                use_fp16=True,
            )
            if self.args.backend == "moge3":
                kwargs["refine_steps"] = self.args.refine_steps
            prediction = self.model.infer(image, **kwargs)["depth"]
        elif self.args.backend == "metricanything_depthmap":
            image = image.float().div_(255.0)
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
            prediction = self.model.infer((image - mean) / std, f_px=float(intrinsics[0, 0]))["depth"]
        else:
            camera = torch.from_numpy(np.asarray(intrinsics, dtype=np.float32)).to(self.device)
            prediction = self.model.infer(image, camera)["depth"]
        prediction = prediction.detach().float().squeeze().cpu().numpy()
        if prediction.shape != (height, width):
            prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.asarray(prediction, dtype=np.float32)


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


def evaluate_frame(predictor: Predictor, frame: Any, scene_key: str) -> dict[str, Any]:
    height, width = frame.image_shape
    intrinsics = np.asarray(frame.intrinsics, dtype=np.float32)
    fov_x = math.degrees(2.0 * math.atan(width / (2.0 * float(intrinsics[0, 0]))))
    gt = np.asarray(frame.depth, dtype=np.float32)
    valid = np.isfinite(gt) & (gt > 0)
    common = {
        **base_row(frame, scene_key), "height": height, "width": width,
        "fx_px": float(intrinsics[0, 0]), "fy_px": float(intrinsics[1, 1]),
        "cx_px": float(intrinsics[0, 2]), "cy_px": float(intrinsics[1, 2]),
        "fov_x_deg": fov_x, "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size), "valid_fraction": float(valid.mean()),
    }
    if not np.any(valid):
        return {**common, "status": "skipped_bad_gt", "error": "no finite positive GT"}
    rgb_bgr = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(frame.rgb_path)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    prediction = predictor(rgb, intrinsics)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - start
    if prediction.shape != gt.shape:
        raise RuntimeError(f"prediction {prediction.shape} != GT {gt.shape}")
    bad = ~np.isfinite(prediction) | (prediction <= 0)
    clipped_pixels = int(np.count_nonzero(bad & valid))
    prediction = np.nan_to_num(prediction, nan=1e-4, posinf=1e4, neginf=1e-4)
    prediction = np.clip(prediction, 1e-4, 1e4)
    metrics = metric_values(prediction, gt, valid)
    pred_values = prediction[valid].astype(np.float64)
    gt_values = gt[valid].astype(np.float64)
    oracle_scale = float(np.median(gt_values / pred_values))
    oracle_abs_rel = float(np.mean(np.abs(pred_values * oracle_scale - gt_values) / gt_values))
    return {
        **common,
        "prediction_clipped_pixels": clipped_pixels,
        "prediction_min_m": float(pred_values.min()),
        "prediction_mean_m": float(pred_values.mean()),
        "prediction_median_m": float(np.median(pred_values)),
        "prediction_max_m": float(pred_values.max()),
        **metrics,
        "oracle_median_scale": oracle_scale,
        "oracle_scaled_abs_rel": oracle_abs_rel,
        "inference_seconds": inference_seconds,
        "status": "ok",
        "error": "",
    }


def build_summary(args: argparse.Namespace, rows: list[dict[str, Any]], selection_rows: list[dict[str, str]], started_at: str) -> dict[str, Any]:
    metric_rows = [row for row in rows if row.get("status") == "ok"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        grouped[row["scene_id"]].append(row)
    expected_by_scene: dict[str, list[str]] = defaultdict(list)
    for row in selection_rows:
        expected_by_scene[row["matterport_scene_id"]].append(row["frame_id"])
    return {
        "schema_version": 1,
        "protocol": "matterport3d-fixed-three-rule-single-view-metric-foundations-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "backend": args.backend,
        "model": args.model,
        "revision": args.revision,
        "model_code_root": str(args.model_code_root.resolve()) if args.model_code_root else None,
        "inference": {
            "frame_mode": "independent single image",
            "camera_input": "GT RGB camera intrinsics/FoV only",
            "test_depth_alignment": "none",
            "prediction_stabilization": "nonfinite/nonpositive values clipped to [1e-4,1e4] m",
            "resolution_level": args.resolution_level if args.backend in {"metricanything_pointmap", "moge3"} else None,
            "refine_steps": args.refine_steps if args.backend == "moge3" else None,
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
            "completed": len(rows), "ok": len(metric_rows),
            "skipped_bad_gt": sum(row.get("status") == "skipped_bad_gt" for row in rows),
            "prediction_clipped_pixels": int(sum(row.get("prediction_clipped_pixels") or 0 for row in metric_rows)),
        },
        "overall": aggregate_rows(metric_rows),
        "by_scene": {scene: aggregate_rows(grouped[scene]) for scene in sorted(grouped)},
        "diagnostic_only": "oracle_median_scale/oracle_scaled_abs_rel use test GT and are never headline metrics",
    }


def main() -> None:
    args = parse_args()
    defaults = BACKEND_DEFAULTS[args.backend]
    args.model = args.model or defaults["model"]
    args.revision = args.revision or defaults["revision"]
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be positive")
    if not 0 <= args.resolution_level <= 9 or args.refine_steps < 0:
        raise ValueError("resolution-level must be in [0,9] and refine-steps nonnegative")

    selection_rows, scene_keys = selected_manifest(args.selection_csv.resolve())
    selected = {
        frame_key(row["matterport_scene_id"], row["frame_id"]): row
        for row in selection_rows
    }
    sys.path.insert(0, str(args.toolkit_root.expanduser().resolve() / "src"))
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
        frames = frames[: args.max_frames]

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
    print(f"[{started_at}] backend={args.backend} frames={len(frames)} completed={len(frames)-len(pending)} pending={len(pending)}", flush=True)
    predictor = Predictor(args) if pending else None
    print(f"model_loaded={bool(predictor)} model={args.model}@{args.revision} device={args.device}", flush=True)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    run_start = time.perf_counter()
    errors = 0
    with csv_path.open("a", encoding="utf-8", newline="", buffering=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for index, frame in enumerate(pending, start=1):
            scene_key = scene_keys[frame.scene_id]
            try:
                row = evaluate_frame(predictor, frame, scene_key)
            except Exception as error:
                errors += 1
                row = {**base_row(frame, scene_key), "status": "error", "error": f"{type(error).__name__}: {error}"}
                print(f"ERROR {frame.scene_id}/{frame.frame_id} {row['error']}", flush=True)
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
            handle.flush()
            if index == 1 or index % args.progress_every == 0 or index == len(pending):
                elapsed = time.perf_counter() - run_start
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else float("nan")
                print(f"progress={index}/{len(pending)} frame={frame.scene_id}/{frame.frame_id} abs_rel={row.get('abs_rel')} rate={rate:.3f}fps eta_hours={eta/3600:.2f} errors={errors}", flush=True)

    rows = completed_rows(csv_path)
    summary = build_summary(args, rows, selection_rows, started_at)
    atomic_json(summary_path, summary)
    print(json.dumps({"summary": str(summary_path), "overall": summary["overall"], "by_scene": summary["by_scene"], "row_counts": summary["row_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
