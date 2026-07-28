#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bim_priorda3.data.pose_recovery import recover_lidar_poses
from bim_priorda3.data.slabim import DEFAULT_REGIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover per-LiDAR-frame SLAM/BIM poses from SLABIM rosbags using ICP"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", default=DEFAULT_REGIONS)
    parser.add_argument("--voxel-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-time-difference", type=float, default=0.01)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--delete-rosbags", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    results = []
    for region in args.regions:
        result = recover_lidar_poses(
            root / "sensor_data" / region,
            voxel_size=args.voxel_size,
            threshold=args.threshold,
            iterations=args.iterations,
            max_time_difference=args.max_time_difference,
            smoothing_window=args.smoothing_window,
            overwrite=args.overwrite,
            delete_rosbags=args.delete_rosbags,
        )
        results.append(result.to_dict())
        print(json.dumps(result.to_dict(), ensure_ascii=False), flush=True)
    report = {
        "slabim_root": str(root),
        "method": "raw /livox/lidar local scan -> official SLAM-global PCD point-to-point ICP",
        "quality_warning": "Inspect per-region diagnostics; pose recovery is not GT trajectory.",
        "regions": results,
    }
    output = args.output or root / "pose_recovery_summary.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pose recovery summary: {output}")


if __name__ == "__main__":
    main()
