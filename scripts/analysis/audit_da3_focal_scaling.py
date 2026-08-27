#!/usr/bin/env python3
"""Audit DA3METRIC canonical-to-metric focal scaling on a prepared split.

The standalone DA3METRIC head predicts depth at a canonical focal length of
300 px.  Metric z-depth is therefore ``depth * mean(fx, fy) / 300``, where the
focal lengths must be expressed at the network processing resolution.  The
prepared PriorBIMDA samples store exactly those resized intrinsics.

This script intentionally does not run a task checkpoint or alter cached DA3
arrays.  It compares the historical cached tensor with its focal-corrected
metric interpretation on exactly the same fixed GT support.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bim_priorda3.config import Config, load_config
from bim_priorda3.data import (
    BIMDepthDataset,
    load_stanford_all_valid_depth,
    official_regular_depth_path,
)

CANONICAL_FOCAL_PX = 300.0


@dataclass
class MetricSums:
    count: int = 0
    abs_rel_sum: float = 0.0
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    delta1_sum: int = 0
    delta2_sum: int = 0
    delta3_sum: int = 0

    def add(self, values: tuple[int, float, float, float, int, int, int]) -> None:
        self.count += values[0]
        self.abs_rel_sum += values[1]
        self.squared_error_sum += values[2]
        self.absolute_error_sum += values[3]
        self.delta1_sum += values[4]
        self.delta2_sum += values[5]
        self.delta3_sum += values[6]

    def compute(self) -> dict[str, float | int]:
        if self.count == 0:
            raise RuntimeError("The selected split has no valid GT pixels")
        return {
            "count": self.count,
            "abs_rel": self.abs_rel_sum / self.count,
            "rmse": (self.squared_error_sum / self.count) ** 0.5,
            "mae": self.absolute_error_sum / self.count,
            "delta1": self.delta1_sum / self.count,
            "delta2": self.delta2_sum / self.count,
            "delta3": self.delta3_sum / self.count,
        }


def _metric_values(prediction: np.ndarray, target: np.ndarray) -> tuple:
    error = prediction - target
    ratio = np.maximum(prediction / target, target / prediction)
    return (
        int(prediction.size),
        float(np.sum(np.abs(error) / target, dtype=np.float64)),
        float(np.sum(error * error, dtype=np.float64)),
        float(np.sum(np.abs(error), dtype=np.float64)),
        int(np.count_nonzero(ratio < 1.25)),
        int(np.count_nonzero(ratio < 1.25**2)),
        int(np.count_nonzero(ratio < 1.25**3)),
    )


def _disable_da3_feature_loading(cfg: Config) -> None:
    feature_cfg = cfg.model.get("da3_feature_fusion")
    if not isinstance(feature_cfg, dict):
        return
    feature_cfg["enabled"] = False
    feature_cfg["scale_enabled"] = False
    feature_cfg["refiner_enabled"] = False


def _load_record(
    record: dict[str, Any],
    *,
    target_shape: tuple[int, int],
    all_valid_stanford: bool,
) -> tuple[tuple, tuple, float, float, float]:
    with np.load(record["sample"]) as item:
        base_depth = item["base_depth"].astype(np.float64)
        intrinsics = item["intrinsic"].astype(np.float64)
        if all_valid_stanford:
            gt_depth, gt_valid = load_stanford_all_valid_depth(
                official_regular_depth_path(record["image"]),
                target_shape,
            )
        else:
            gt_depth = item["gt_depth"].astype(np.float32)
            gt_valid = item["gt_valid"] > 0

    if intrinsics.shape != (3, 3):
        raise ValueError(f"{record['id']}: intrinsic must be 3x3")
    focal_px = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    if not np.isfinite(focal_px) or focal_px <= 0:
        raise ValueError(f"{record['id']}: invalid resized focal length {focal_px}")
    metric_scale = focal_px / CANONICAL_FOCAL_PX
    support = (
        np.asarray(gt_valid, dtype=bool)
        & np.isfinite(gt_depth)
        & (gt_depth > 0)
        & np.isfinite(base_depth)
        & (base_depth > 0)
    )
    canonical = base_depth[support]
    target = gt_depth[support].astype(np.float64)
    canonical_values = _metric_values(canonical, target)
    metric_values = _metric_values(canonical * metric_scale, target)
    canonical_frame_abs_rel = canonical_values[1] / canonical_values[0]
    metric_frame_abs_rel = metric_values[1] / metric_values[0]
    return (
        canonical_values,
        metric_values,
        metric_scale,
        canonical_frame_abs_rel,
        metric_frame_abs_rel,
    )


def _audit_split(cfg: Config, split: str, workers: int) -> dict[str, Any]:
    dataset = BIMDepthDataset(cfg, split, augment=False)
    target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
    all_valid_stanford = str(cfg.data.get("ground_truth_support", "prepared")) == (
        "official_all_valid"
    )
    canonical = MetricSums()
    metric = MetricSums()
    factors: list[float] = []
    canonical_frame_abs_rel: list[float] = []
    metric_frame_abs_rel: list[float] = []

    def load(record: dict[str, Any]) -> tuple:
        return _load_record(
            record,
            target_shape=target_shape,
            all_valid_stanford=all_valid_stanford,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for old, corrected, factor, old_frame, corrected_frame in pool.map(
            load,
            dataset.records,
        ):
            canonical.add(old)
            metric.add(corrected)
            factors.append(factor)
            canonical_frame_abs_rel.append(old_frame)
            metric_frame_abs_rel.append(corrected_frame)

    factor_array = np.asarray(factors, dtype=np.float64)
    old_frames = np.asarray(canonical_frame_abs_rel, dtype=np.float64)
    corrected_frames = np.asarray(metric_frame_abs_rel, dtype=np.float64)
    return {
        "frame_count": len(dataset),
        "pixel_micro": {
            "cached_canonical_output": canonical.compute(),
            "focal_corrected_metric_output": metric.compute(),
        },
        "frame_macro_abs_rel": {
            "cached_canonical_output": float(old_frames.mean()),
            "focal_corrected_metric_output": float(corrected_frames.mean()),
        },
        "frames_improved_by_focal_correction": int(np.count_nonzero(corrected_frames < old_frames)),
        "metric_scale": {
            "min": float(factor_array.min()),
            "mean": float(factor_array.mean()),
            "median": float(np.median(factor_array)),
            "max": float(factor_array.max()),
            "std": float(factor_array.std()),
        },
        "split_provenance": dataset.split_provenance,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "val", "test"),
        dest="splits",
        help="May be repeated; defaults to val and test",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")

    cfg = load_config(args.config)
    _disable_da3_feature_loading(cfg)
    script_path = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "protocol": "da3metric-canonical-focal-audit-v1",
        "formula": "metric_depth = cached_depth * mean(fx, fy) / 300",
        "focal_coordinate_system": "prepared DA3 processing resolution",
        "canonical_focal_px": CANONICAL_FOCAL_PX,
        "config": str(Path(cfg.config_path).resolve()),
        "da3_model": str(cfg.data.da3_model),
        "da3_revision": str(cfg.data.da3_revision),
        "da3_process_res": int(cfg.data.da3_process_res),
        "script_sha256": _sha256(script_path),
        "splits": {
            split: _audit_split(cfg, split, args.workers)
            for split in (args.splits or ["val", "test"])
        },
        "interpretation": {
            "cached_arrays_mutated": False,
            "task_checkpoint_executed": False,
            "gt_used_for_scaling": False,
        },
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(output)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
