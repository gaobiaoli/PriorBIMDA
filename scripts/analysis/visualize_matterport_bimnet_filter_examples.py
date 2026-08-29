#!/usr/bin/env python3
"""Create three material panels explaining Matterport/BIMNet frame filtering."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-frame-csv", type=Path, required=True)
    parser.add_argument("--matterport-root", type=Path, required=True)
    parser.add_argument("--bimnet-root", type=Path, required=True)
    parser.add_argument("--toolkit-root", type=Path, required=True)
    parser.add_argument("--bimnet-scene", default="hxp")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--absolute-tolerance-m", type=float, default=0.10)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    return parser.parse_args()


def _bool(value: str) -> bool:
    return str(value).casefold() == "true"


def _load_rows(path: Path) -> list[dict[str, str]]:
    latest = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            latest[f"{row['scene_id']}/{row['frame_id']}"] = row
    return [row for row in latest.values() if row["status"] == "ok"]


def _select_examples(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    effective = [
        row
        for row in rows
        if _bool(row["effective_pass"]) and float(row["gt_valid_fraction"]) >= 0.50
    ]
    if not effective:
        raise RuntimeError("No effective frame is available")
    median_agreement = float(
        np.median([float(row["bim_gt_agree_image_fraction"]) for row in effective])
    )
    representative = min(
        effective,
        key=lambda row: abs(float(row["bim_gt_agree_image_fraction"]) - median_agreement),
    )
    outside = [
        row
        for row in rows
        if not _bool(row["camera_in_bim_aabb"]) and _bool(row["gt_quality_pass"])
    ]
    if not outside:
        raise RuntimeError("No outside-BIM frame is available")
    outside_bim = max(outside, key=lambda row: float(row["gt_valid_fraction"]))
    mismatch = [
        row
        for row in rows
        if _bool(row["camera_in_bim_aabb"])
        and _bool(row["gt_quality_pass"])
        and "low_bim_gt_agreement" in row["filter_reasons"]
    ]
    if not mismatch:
        raise RuntimeError("No inside-AABB BIM/GT mismatch frame is available")
    inside_mismatch = max(
        mismatch,
        key=lambda row: (
            float(row["gt_valid_fraction"]),
            -float(row["bim_gt_agree_image_fraction"]),
        ),
    )
    return {
        "effective": representative,
        "outside_bim": outside_bim,
        "inside_aabb_mismatch": inside_mismatch,
    }


def _render_panel(
    *,
    label: str,
    row: dict[str, str],
    frame: object,
    raycaster: object,
    processed_geometry: object,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    height, width = frame.image_shape
    process_height, process_width, intrinsics, _ = processed_geometry(
        height,
        width,
        frame.intrinsics,
        args.process_res,
    )
    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Cannot read {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (process_width, process_height), interpolation=cv2.INTER_AREA)
    gt = cv2.resize(
        np.asarray(frame.depth, dtype=np.float32),
        (process_width, process_height),
        interpolation=cv2.INTER_NEAREST,
    )
    bim = raycaster.depth(
        intrinsics,
        frame.world_to_camera,
        process_width,
        process_height,
    )
    gt_valid = np.isfinite(gt) & (gt > 0)
    bim_valid = np.isfinite(bim) & (bim > 0)
    overlap = gt_valid & bim_valid
    tolerance = np.maximum(
        args.absolute_tolerance_m,
        args.relative_tolerance * gt,
    )
    agreement = overlap & (np.abs(bim - gt) <= tolerance)
    diagnostic = rgb.astype(np.float32) / 255.0
    diagnostic[bim_valid & ~gt_valid] = 0.55 * diagnostic[bim_valid & ~gt_valid] + np.array(
        [0.65, 0.0, 0.65]
    )
    diagnostic[overlap & ~agreement] = 0.45 * diagnostic[overlap & ~agreement] + np.array(
        [0.55, 0.0, 0.0]
    )
    diagnostic[agreement] = 0.45 * diagnostic[agreement] + np.array([0.0, 0.55, 0.0])
    depths = np.concatenate((gt[gt_valid], bim[bim_valid]))
    minimum, maximum = np.percentile(depths, (2, 98)) if len(depths) else (0.0, 1.0)
    gt_show = np.ma.masked_where(~gt_valid, gt)
    bim_show = np.ma.masked_where(~bim_valid, bim)
    error_show = np.ma.masked_where(~overlap, np.abs(bim - gt))

    figure, axes = plt.subplots(1, 5, figsize=(19, 4.1), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("Matterport RGB")
    depth_image = axes[1].imshow(gt_show, cmap="turbo", vmin=minimum, vmax=maximum)
    axes[1].set_title("Matterport GT depth")
    axes[2].imshow(bim_show, cmap="turbo", vmin=minimum, vmax=maximum)
    axes[2].set_title("Registered BIM depth")
    error_image = axes[3].imshow(error_show, cmap="magma", vmin=0, vmax=0.5)
    axes[3].set_title("|BIM - GT| (m)")
    axes[4].imshow(np.clip(diagnostic, 0, 1))
    axes[4].set_title("Green: agree; red: mismatch")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(depth_image, ax=axes[1:3], fraction=0.025, pad=0.01, label="metres")
    figure.colorbar(error_image, ax=axes[3], fraction=0.05, pad=0.02, label="metres")
    figure.suptitle(
        f"{label} | {row['frame_id']} | effective={row['effective_pass']} | "
        f"hit={float(row['bim_hit_fraction']):.1%}, "
        f"agree/image={float(row['bim_gt_agree_image_fraction']):.1%}\n"
        f"reasons={row['filter_reasons'] or 'none'}",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    toolkit_src = args.toolkit_root.expanduser().resolve() / "src"
    sys.path.insert(0, str(toolkit_src))
    model_script_dir = Path(__file__).parents[1] / "model"
    sys.path.insert(0, str(model_script_dir))
    from evaluate_matterport_bimnet_full_regression import (
        BIMRaycaster,
        processed_geometry,
    )
    from s3dis_sam3d import BIMNetDataset, Matterport3DDataset

    rows = _load_rows(args.per_frame_csv)
    selected = _select_examples(rows)
    bim_scene = BIMNetDataset(args.bimnet_root)[args.bimnet_scene]
    matterport_scene = bim_scene.matterport_scene(Matterport3DDataset(args.matterport_root))
    mesh = bim_scene.mesh(
        source="obj",
        wall_filled=True,
        coordinates="point_cloud",
        progress=False,
    )
    raycaster = BIMRaycaster(mesh)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {}
    for label, row in selected.items():
        frame = matterport_scene.get_frame(row["frame_id"])
        output_path = args.output_dir / f"filter_example_{label}.png"
        _render_panel(
            label=label,
            row=row,
            frame=frame,
            raycaster=raycaster,
            processed_geometry=processed_geometry,
            args=args,
            output_path=output_path,
        )
        receipt[label] = {
            "frame_id": row["frame_id"],
            "effective": _bool(row["effective_pass"]),
            "filter_reasons": row["filter_reasons"],
            "bim_hit_fraction": float(row["bim_hit_fraction"]),
            "bim_gt_agree_image_fraction": float(row["bim_gt_agree_image_fraction"]),
            "asset": str(output_path.resolve()),
        }
    (args.output_dir / "selection.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
