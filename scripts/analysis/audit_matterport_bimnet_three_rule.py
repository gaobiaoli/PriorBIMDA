#!/usr/bin/env python3
"""Freeze and diagnose the three-rule Matterport3D/BIMNet frame selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--bimnet-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--bimnet-scenes", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--gt-min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--bim-min-hit-fraction", type=float, default=0.20)
    parser.add_argument("--aabb-margin-m", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def frame_set_sha256(frame_ids: list[str]) -> str:
    payload = "\n".join(sorted(frame_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p05", "p25", "median", "p75", "p95", "max"),
            np.quantile(array, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)),
            strict=True,
        )
    }


def main() -> None:
    args = parse_args()
    if args.process_res < 14 or args.process_res % 14:
        raise ValueError("--process-res must be a positive multiple of 14")
    if not 0 <= args.gt_min_valid_fraction <= 1:
        raise ValueError("--gt-min-valid-fraction must be in [0, 1]")
    if not 0 <= args.bim_min_hit_fraction <= 1:
        raise ValueError("--bim-min-hit-fraction must be in [0, 1]")
    if args.aabb_margin_m < 0:
        raise ValueError("--aabb-margin-m must be non-negative")

    toolkit_src = args.toolkit_root.expanduser().resolve() / "src"
    sys.path.insert(0, str(toolkit_src))
    model_scripts = Path(__file__).resolve().parents[1] / "model"
    sys.path.insert(0, str(model_scripts))
    from evaluate_matterport_bimnet_full_regression import BIMRaycaster, processed_geometry
    from s3dis_sam3d import BIMNetDataset, Matterport3DDataset

    bim_dataset = BIMNetDataset(args.bimnet_root)
    matterport_dataset = Matterport3DDataset(args.matterport_root)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scene_receipts: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []

    for scene_alias in args.bimnet_scenes:
        bim_scene = bim_dataset[scene_alias]
        matterport_scene = bim_scene.matterport_scene(matterport_dataset)
        frames = list(matterport_scene.frames)
        mesh = bim_scene.mesh(
            source="obj", wall_filled=True, coordinates="point_cloud", progress=False
        )
        raycaster = BIMRaycaster(mesh)
        scene_rows = []
        for index, frame in enumerate(frames, start=1):
            height, width = frame.image_shape
            process_height, process_width, intrinsics, _ = processed_geometry(
                height, width, frame.intrinsics, args.process_res
            )
            gt = np.asarray(frame.depth, dtype=np.float32)
            gt_valid_fraction = float((np.isfinite(gt) & (gt > 0)).mean())
            bim_depth = raycaster.depth(
                intrinsics, frame.world_to_camera, process_width, process_height
            )
            bim_hit_fraction = float((bim_depth > 0).mean())
            camera_inside = raycaster.contains_camera(
                np.asarray(frame.camera_position, dtype=np.float64), args.aabb_margin_m
            )
            failures = []
            if not gt_valid_fraction > args.gt_min_valid_fraction:
                failures.append("sparse_gt")
            if not bim_hit_fraction > args.bim_min_hit_fraction:
                failures.append("low_bim_hit")
            if not camera_inside:
                failures.append("camera_outside_bim_aabb")
            row = {
                "bimnet_scene_key": bim_scene.key,
                "matterport_scene_id": matterport_scene.scene_id,
                "frame_id": frame.frame_id,
                "gt_valid_fraction": gt_valid_fraction,
                "bim_hit_fraction": bim_hit_fraction,
                "camera_in_bim_aabb": camera_inside,
                "selected": not failures,
                "filter_reasons": ";".join(failures),
            }
            scene_rows.append(row)
            all_rows.append(row)
            if index % args.progress_every == 0 or index == len(frames):
                print(
                    f"scene={bim_scene.key} audited={index}/{len(frames)} "
                    f"selected={sum(item['selected'] for item in scene_rows)}",
                    flush=True,
                )

        selected_ids = [str(row["frame_id"]) for row in scene_rows if row["selected"]]
        reason_sets = Counter(row["filter_reasons"] or "pass" for row in scene_rows)
        individual_failures = {
            reason: sum(reason in str(row["filter_reasons"]).split(";") for row in scene_rows)
            for reason in ("sparse_gt", "low_bim_hit", "camera_outside_bim_aabb")
        }
        scene_receipts[bim_scene.key] = {
            "matterport_scene_id": matterport_scene.scene_id,
            "source_frames": len(scene_rows),
            "selected_frames": len(selected_ids),
            "selected_fraction": len(selected_ids) / len(scene_rows),
            "selected_frame_ids_sha256": frame_set_sha256(selected_ids),
            "individual_failure_counts": individual_failures,
            "failure_intersections": dict(sorted(reason_sets.items())),
            "gt_valid_fraction_quantiles": quantiles(
                [float(row["gt_valid_fraction"]) for row in scene_rows]
            ),
            "bim_hit_fraction_quantiles": quantiles(
                [float(row["bim_hit_fraction"]) for row in scene_rows]
            ),
            "bim_hit_fraction_inside_aabb_quantiles": quantiles(
                [
                    float(row["bim_hit_fraction"])
                    for row in scene_rows
                    if row["camera_in_bim_aabb"]
                ]
            ),
            "mesh": {
                "wall_filled": True,
                "vertices": len(mesh.vertices),
                "triangles": len(mesh.triangles),
                "aabb_min": raycaster.minimum.tolist(),
                "aabb_max": raycaster.maximum.tolist(),
            },
        }

    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "gt_min_valid_fraction": args.gt_min_valid_fraction,
            "bim_min_hit_fraction": args.bim_min_hit_fraction,
            "aabb_margin_m": args.aabb_margin_m,
            "process_res": args.process_res,
            "selection_uses_model_prediction": False,
            "selection_uses_bim_gt_agreement": False,
        },
        "scenes": scene_receipts,
    }
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {output}", flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
