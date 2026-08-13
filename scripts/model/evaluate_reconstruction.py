#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from bim_priorda3.baselines import previous_scale_baselines
from bim_priorda3.checkpoints import validate_checkpoint_model_config
from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.geometry import read_poses, read_timestamps
from bim_priorda3.engine import move_batch
from bim_priorda3.models import BIMPriorDA3
from bim_priorda3.reconstruction import (
    depth_to_world_points,
    reconstruction_metrics,
    save_point_cloud,
    voxel_downsample,
)

METHOD_COLORS = {
    "base": (0.20, 0.55, 0.95),
    "global_scale": (0.35, 0.75, 0.35),
    "previous_scale_local": (0.95, 0.65, 0.20),
    "coarse": (0.65, 0.35, 0.85),
    "refined": (0.95, 0.30, 0.25),
    "gt": (0.70, 0.70, 0.70),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse per-frame depth into BIM coordinates and evaluate 3D reconstruction"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(name for name in METHOD_COLORS if name != "gt"),
        default=("base", "previous_scale_local", "refined"),
    )
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--prediction-mask",
        choices=("all", "gt"),
        default="all",
        help="'all' measures reconstructed surface; 'gt' is a matched-pixel diagnostic",
    )
    parser.add_argument("--save-clouds", action="store_true")
    return parser.parse_args()


def _batch(sample: dict, device: torch.device) -> dict:
    batched = {
        key: value[None] if torch.is_tensor(value) else [value] for key, value in sample.items()
    }
    return move_batch(batched, device)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    slabim = resolve_slabim_root(cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    model = BIMPriorDA3(cfg).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_model_config(state, cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    output = args.output or (
        resolve_project_path(cfg, cfg.experiment.output_dir) / f"reconstruction_{args.split}"
    )
    output.mkdir(parents=True, exist_ok=True)

    calibration = np.loadtxt(slabim / "calibration_files/cam_to_lidar.txt")
    pose_cache: dict[str, np.ndarray] = {}
    method_parts: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    reference_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    records = dataset.records[: args.max_frames] if args.max_frames else dataset.records
    with torch.no_grad():
        for index, record in enumerate(records):
            region = str(record["region"])
            if region not in pose_cache:
                region_root = slabim / "sensor_data" / region
                timestamps = read_timestamps(region_root / "points/timestamps.txt")
                pose_cache[region] = read_poses(
                    region_root / "points" / cfg.data.pose_bim_file,
                    timestamps,
                )
            lidar_to_bim = pose_cache[region][int(record["lidar_index"])]
            camera_to_bim = lidar_to_bim @ calibration
            with np.load(record["sample"]) as stored:
                intrinsic = stored["intrinsic"].astype(np.float64)
            batch = _batch(dataset[index], device)
            output_prediction = model(batch)
            base = batch["base_depth"][0, 0].float().cpu().numpy()
            bim = batch["bim_depth"][0, 0].float().cpu().numpy()
            scaled, previous, _ = previous_scale_baselines(base, bim)
            predictions = {
                "base": base,
                "global_scale": scaled,
                "previous_scale_local": previous,
                "coarse": output_prediction["coarse_depth"][0, 0].float().cpu().numpy(),
                "refined": output_prediction["depth"][0, 0].float().cpu().numpy(),
            }
            gt = batch["gt_depth"][0, 0].float().cpu().numpy()
            gt_valid = batch["gt_valid"][0, 0].bool().cpu().numpy()
            common = {
                "intrinsic": intrinsic,
                "camera_to_world": camera_to_bim,
                "pixel_stride": args.pixel_stride,
                "min_depth": float(cfg.data.min_depth),
                "max_depth": float(cfg.data.max_depth),
            }
            reference_parts[region].append(depth_to_world_points(gt, valid_mask=gt_valid, **common))
            prediction_mask = gt_valid if args.prediction_mask == "gt" else None
            for method in args.methods:
                method_parts[region][method].append(
                    depth_to_world_points(
                        predictions[method],
                        valid_mask=prediction_mask,
                        **common,
                    )
                )
            print(f"[{index + 1}/{len(records)}] {record['id']}", flush=True)

    per_region = {}
    for region in sorted(reference_parts):
        reference = voxel_downsample(np.concatenate(reference_parts[region]), args.voxel_size)
        region_metrics = {}
        region_dir = output / region
        region_dir.mkdir(parents=True, exist_ok=True)
        if args.save_clouds:
            save_point_cloud(str(region_dir / "gt_fused.ply"), reference, METHOD_COLORS["gt"])
        for method in args.methods:
            reconstruction = voxel_downsample(
                np.concatenate(method_parts[region][method]), args.voxel_size
            )
            region_metrics[method] = reconstruction_metrics(
                reconstruction, reference, args.thresholds
            )
            if args.save_clouds:
                save_point_cloud(
                    str(region_dir / f"{method}.ply"),
                    reconstruction,
                    METHOD_COLORS[method],
                )
        per_region[region] = region_metrics

    combined_reference = voxel_downsample(
        np.concatenate(
            [part for region in sorted(reference_parts) for part in reference_parts[region]]
        ),
        args.voxel_size,
    )
    aggregate = {}
    save_aggregate_clouds = args.save_clouds and len(reference_parts) > 1
    for method in args.methods:
        reconstruction = voxel_downsample(
            np.concatenate(
                [part for region in sorted(method_parts) for part in method_parts[region][method]]
            ),
            args.voxel_size,
        )
        aggregate[method] = reconstruction_metrics(
            reconstruction, combined_reference, args.thresholds
        )
        if save_aggregate_clouds:
            save_point_cloud(
                str(output / f"all_{method}.ply"),
                reconstruction,
                METHOD_COLORS[method],
            )
    if save_aggregate_clouds:
        save_point_cloud(
            str(output / "all_gt_fused.ply"),
            combined_reference,
            METHOD_COLORS["gt"],
        )
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "regions": sorted(reference_parts),
        "frames": len(records),
        "coordinate_system": "BIM",
        "alignment": "calibrated recovered poses only; no evaluation-time ICP",
        "reference": "occlusion-filtered +/- fusion-radius LiDAR GT depth",
        "prediction_mask": args.prediction_mask,
        "prediction_mask_note": (
            "all: dense predicted surfaces are compared with sparse fused LiDAR reference; "
            "gt: diagnostic using only pixels where LiDAR GT exists"
        ),
        "pixel_stride": args.pixel_stride,
        "voxel_size_m": args.voxel_size,
        "depth_range_m": [float(cfg.data.min_depth), float(cfg.data.max_depth)],
        "thresholds_m": args.thresholds,
        "aggregate": aggregate,
        "per_region": per_region,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
