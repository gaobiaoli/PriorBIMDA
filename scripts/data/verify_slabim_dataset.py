#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bim_priorda3.data.slabim import DEFAULT_REGIONS, region_core_is_complete

POSE_FILES = (
    "lidar_pose_local_to_slam_smoothed.txt",
    "lidar_pose_local_to_bim_from_rosbag.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw SLABIM inputs needed by this project"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--require-rosbag", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    shared = {
        "intrinsics": (root / "calibration_files/cam_intrinsics.txt").is_file(),
        "camera_to_lidar": (root / "calibration_files/cam_to_lidar.txt").is_file(),
        "bim": (root / "BIM").is_dir(),
    }
    regions = {}
    failures = []
    for name in args.regions:
        region = root / "sensor_data" / name
        images = sorted((region / "images/data").glob("*.png"))
        pcds = sorted((region / "points/data").glob("*.pcd"))
        image_times_path = region / "images/timestamps.txt"
        lidar_times_path = region / "points/timestamps.txt"
        image_times = (
            np.atleast_1d(np.loadtxt(image_times_path))
            if image_times_path.exists()
            else np.empty(0)
        )
        lidar_times = (
            np.atleast_1d(np.loadtxt(lidar_times_path))
            if lidar_times_path.exists()
            else np.empty(0)
        )
        pose_status = {}
        for pose_name in POSE_FILES:
            path = region / "points" / pose_name
            if path.exists():
                rows = np.atleast_2d(np.loadtxt(path))
                pose_status[pose_name] = {
                    "exists": True,
                    "shape": list(rows.shape),
                    "timestamp_aligned": bool(
                        rows.shape == (len(lidar_times), 8)
                        and np.allclose(rows[:, 0], lidar_times, atol=1e-4, rtol=0.0)
                    ),
                }
            else:
                pose_status[pose_name] = {"exists": False}
        has_rosbag = any((region / "rosbag").glob("*.bag"))
        valid = (
            region_core_is_complete(region)
            and len(images) == len(image_times)
            and len(pcds) == len(lidar_times)
            and all(item.get("timestamp_aligned", False) for item in pose_status.values())
            and (has_rosbag if args.require_rosbag else True)
        )
        regions[name] = {
            "valid": valid,
            "images": len(images),
            "image_timestamps": len(image_times),
            "pcds": len(pcds),
            "lidar_timestamps": len(lidar_times),
            "has_rosbag": has_rosbag,
            "poses": pose_status,
        }
        if not valid:
            failures.append(name)
    report = {
        "root": str(root),
        "valid": all(shared.values()) and not failures,
        "shared": shared,
        "regions": regions,
        "failed_regions": failures,
    }
    output = args.output or root / "dataset_verification.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
