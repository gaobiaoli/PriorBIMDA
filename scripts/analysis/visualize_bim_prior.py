#!/usr/bin/env python3
"""Render the RGB-aligned BIM depth prior stored in a prepared sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    return parser.parse_args()


def find_record(manifest: Path, sample_id: str) -> dict[str, object]:
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record["id"]) == sample_id:
                return record
    raise KeyError(f"Sample {sample_id!r} is absent from {manifest}")


def main() -> None:
    args = parse_args()
    record = find_record(args.manifest, args.sample_id)
    sample = np.load(str(record["sample"]))
    bim_depth = sample["bim_depth"].astype(np.float32)
    bim_valid = sample["bim_valid"].astype(bool)

    image = cv2.imread(str(record["image"]), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {record['image']}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(
        image,
        (bim_depth.shape[1], bim_depth.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    rgb = image.astype(np.float32) / 255.0

    depth_display = bim_depth.copy()
    depth_display[~bim_valid] = np.nan
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    norm = Normalize(vmin=args.min_depth, vmax=args.max_depth)

    depth_rgba = cmap(norm(np.nan_to_num(depth_display, nan=args.min_depth)))
    overlay = rgb.copy()
    overlay[bim_valid] = 0.42 * rgb[bim_valid] + 0.58 * depth_rgba[bim_valid, :3]
    overlay[~bim_valid] *= 0.35

    coverage = float(bim_valid.mean())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB input")
    depth_artist = axes[1].imshow(depth_display, cmap=cmap, norm=norm)
    axes[1].set_title(f"Rendered BIM depth prior\nvalid coverage = {coverage:.1%}")
    axes[2].imshow(overlay)
    axes[2].contour(bim_valid, levels=[0.5], colors=["white"], linewidths=0.65)
    axes[2].set_title("BIM depth aligned with RGB\nwhite = valid-support boundary")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.colorbar(
        depth_artist,
        ax=axes[1:].ravel().tolist(),
        shrink=0.78,
        pad=0.02,
        label="BIM camera-z depth (m)",
    )
    fig.suptitle(str(record["id"]), fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
