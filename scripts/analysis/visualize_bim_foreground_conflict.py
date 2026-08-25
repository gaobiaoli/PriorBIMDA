#!/usr/bin/env python3
"""Export reproducible Stanford Area_1 BIM/foreground-conflict examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import (
    BIMDepthDataset,
    load_stanford_all_valid_depth,
    official_regular_depth_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/stanford_area1_attentive_scale_da3_features_hit_only_full_depth.yaml",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument(
        "--output-dir",
        default="docs/assets/attention_scale/bim_foreground_conflict",
    )
    return parser.parse_args()


def _overlay(rgb: np.ndarray, masks: list[tuple[np.ndarray, tuple[int, int, int]]]) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    for mask, color in masks:
        color_array = np.asarray(color, dtype=np.float32)
        output[mask] = 0.42 * output[mask] + 0.58 * color_array
    return np.clip(output, 0, 255).astype(np.uint8)


def _safe_stem(sample_id: str) -> str:
    return sample_id.replace("/", "__")


def _export_sample(record: dict, dataset: BIMDepthDataset, output_dir: Path) -> dict:
    target_shape = (dataset.height, dataset.width)
    rgb_bgr = cv2.imread(str(record["image"]), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(record["image"])
    rgb = cv2.cvtColor(
        cv2.resize(rgb_bgr, target_shape[::-1], interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2RGB,
    )
    with np.load(Path(record["sample"]), allow_pickle=False) as sample:
        bim_depth = sample["bim_depth"].astype(np.float32)
        bim_valid = sample["bim_valid"] > 0
        furniture = sample["furniture_mask"] > 0

    gt_depth, gt_valid = load_stanford_all_valid_depth(
        official_regular_depth_path(record["image"]),
        target_shape,
    )
    tolerance = np.maximum(np.full_like(bim_depth, 0.10), 0.05 * bim_depth)
    furniture = gt_valid & furniture
    conflict = gt_valid & bim_valid & (gt_depth < bim_depth - tolerance)
    overlap = furniture & conflict
    furniture_only = furniture & ~conflict
    conflict_only = conflict & ~furniture

    furniture_overlay = _overlay(rgb, [(furniture, (0, 210, 255))])
    conflict_overlay = _overlay(rgb, [(conflict, (245, 50, 60))])
    decomposition = _overlay(
        rgb,
        [
            (conflict_only, (245, 50, 60)),
            (furniture_only, (0, 210, 255)),
            (overlap, (255, 205, 0)),
        ],
    )

    depth_values = np.concatenate((gt_depth[gt_valid], bim_depth[bim_valid]))
    display_max = float(np.percentile(depth_values, 98.0)) if depth_values.size else 5.0
    display_max = max(display_max, 1.0)
    masked_gt = np.ma.masked_where(~gt_valid, gt_depth)
    masked_bim = np.ma.masked_where(~bim_valid, bim_depth)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8), constrained_layout=True)
    panels = (
        (axes[0, 0], rgb, "RGB"),
        (axes[1, 0], furniture_overlay, "Official furniture mask (cyan)"),
        (axes[1, 1], conflict_overlay, "BIM-foreground conflict (red)"),
        (axes[1, 2], decomposition, "Red: conflict only | Yellow: overlap | Cyan: furniture only"),
    )
    for axis, image, title in panels:
        axis.imshow(image)
        axis.set_title(title, fontsize=10)
        axis.axis("off")
    depth_image = axes[0, 1].imshow(masked_gt, cmap="turbo", vmin=0.0, vmax=display_max)
    axes[0, 1].set_title("Official all-valid GT z-depth")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(masked_bim, cmap="turbo", vmin=0.0, vmax=display_max)
    axes[0, 2].set_title("Hit-only BIM z-depth")
    axes[0, 2].axis("off")
    colorbar = fig.colorbar(depth_image, ax=(axes[0, 1], axes[0, 2]), shrink=0.80)
    colorbar.set_label(f"Depth (m), display clipped at p98={display_max:.2f} m")
    fig.suptitle(str(record["id"]), fontsize=11)

    stem = _safe_stem(str(record["id"]))
    panel_path = output_dir / f"{stem}__panel.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    components = {
        "rgb": rgb,
        "furniture_overlay": furniture_overlay,
        "conflict_overlay": conflict_overlay,
        "mask_decomposition": decomposition,
    }
    component_paths = {}
    for name, image in components.items():
        path = output_dir / f"{stem}__{name}.png"
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        component_paths[name] = str(path.relative_to(PROJECT_ROOT))

    valid_count = int(gt_valid.sum())
    furniture_count = int(furniture.sum())
    conflict_count = int(conflict.sum())
    overlap_count = int(overlap.sum())
    return {
        "sample_id": str(record["id"]),
        "split": dataset.split,
        "definition": "GT < BIM - max(0.10 m, 0.05 * BIM), with valid GT and BIM",
        "full_depth_gt_validity": "official uint16 raw != 0 and raw != 65535",
        "counts": {
            "valid": valid_count,
            "furniture": furniture_count,
            "conflict": conflict_count,
            "overlap": overlap_count,
            "furniture_only": int(furniture_only.sum()),
            "conflict_only": int(conflict_only.sum()),
        },
        "fractions": {
            "furniture_of_valid": furniture_count / max(valid_count, 1),
            "conflict_of_valid": conflict_count / max(valid_count, 1),
            "furniture_in_conflict": overlap_count / max(furniture_count, 1),
            "conflict_that_is_furniture": overlap_count / max(conflict_count, 1),
        },
        "depth_display_p98_m": display_max,
        "panel": str(panel_path.relative_to(PROJECT_ROOT)),
        "components": component_paths,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = BIMDepthDataset(
        cfg,
        split=args.split,
        augment=False,
        require_ground_truth=True,
    )
    requested = set(args.sample_id)
    records = {str(record["id"]): record for record in dataset.records}
    missing = sorted(requested - set(records))
    if missing:
        raise KeyError(f"Sample IDs are not in split {args.split!r}: {missing}")

    output_dir = resolve_project_path(cfg, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = [_export_sample(records[sample_id], dataset, output_dir) for sample_id in args.sample_id]
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "samples": metadata}, indent=2))


if __name__ == "__main__":
    main()
