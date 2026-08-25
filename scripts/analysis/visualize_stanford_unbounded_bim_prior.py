#!/usr/bin/env python3
"""Render a hit-only, depth-unbounded Area_1 structural BIM prior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from bim_priorda3.data.ifc_envelope import (
    ENVELOPE_CATEGORIES,
    build_global_ifc_envelope_scene,
    render_ifc_envelope,
)
from bim_priorda3.data.stanford2d3ds import pose_matrices
from bim_priorda3.data.stanford_registration import accepted_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--alignment", required=True, type=Path)
    parser.add_argument("--ifc-dir", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=504)
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
    alignment = json.loads(args.alignment.read_text(encoding="utf-8"))
    transforms = accepted_transforms(alignment)
    ifc_paths = {
        room: (args.ifc_dir / f"{room}.ifc").resolve()
        for room in sorted(transforms)
    }
    missing = [str(path) for path in ifc_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing registered IFC files: {missing[:5]}")

    # Keep every fixed envelope class, including doors/windows. Furniture,
    # furnishing proxies and MEP never enter the envelope allow-list.
    scene, geometry = build_global_ifc_envelope_scene(
        ifc_paths,
        transforms,
        included_categories=ENVELOPE_CATEGORIES,
    )

    pose = json.loads(Path(str(record["pose"])).read_text(encoding="utf-8"))
    intrinsic_source, _, camera_to_area = pose_matrices(pose)
    image_bgr = cv2.imread(str(record["image"]), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {record['image']}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
    intrinsic = intrinsic_source.copy()
    intrinsic[0] *= args.size / image_bgr.shape[1]
    intrinsic[1] *= args.size / image_bgr.shape[0]

    # Infinity deliberately disables the old max-depth cutoff. The renderer's
    # remaining validity rule is exactly: finite positive ray hit.
    depth, _, category = render_ifc_envelope(
        scene,
        geometry,
        intrinsic,
        camera_to_area,
        args.size,
        args.size,
        float("inf"),
    )
    valid = np.isfinite(depth) & (depth > 0.0)
    category[~valid] = 255

    prepared = np.load(str(record["sample"]))
    old_valid = prepared["bim_valid"].astype(bool)
    new_only = valid & ~old_valid
    old_coverage = float(old_valid.mean())
    new_coverage = float(valid.mean())
    new_only_fraction = float(new_only.mean())
    recovered_above_5m = new_only & (depth > 5.0)
    recovered_within_old_range = new_only & (depth >= 0.2) & (depth <= 5.0)

    finite_depth = depth[valid]
    if not len(finite_depth):
        raise RuntimeError("The unbounded global BIM produced no positive ray hit")
    minimum = max(float(finite_depth.min()), np.finfo(np.float32).tiny)
    maximum = float(finite_depth.max())
    norm = LogNorm(vmin=minimum, vmax=maximum)
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad("black")
    depth_display = depth.copy()
    depth_display[~valid] = np.nan

    colored = cmap(norm(np.nan_to_num(depth_display, nan=minimum)))[..., :3]
    overlay = 0.45 * (rgb.astype(np.float32) / 255.0) + 0.55 * colored
    overlay[~valid] = 0.25 * (rgb.astype(np.float32) / 255.0)[~valid]

    difference = np.zeros((*valid.shape, 3), dtype=np.float32)
    difference[old_valid] = (0.22, 0.66, 0.36)
    difference[new_only] = (0.96, 0.45, 0.13)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB regular view")
    depth_artist = axes[1].imshow(depth_display, cmap=cmap, norm=norm)
    axes[1].set_title(
        "Unbounded structural BIM depth\n"
        f"hit range = {minimum:.2f}–{maximum:.2f} m"
    )
    axes[2].imshow(difference)
    axes[2].set_title(
        "Hit-only validity mask\n"
        f"green=old valid, orange=new hit ({new_only_fraction:.1%})"
    )
    axes[3].imshow(np.clip(overlay, 0.0, 1.0))
    axes[3].contour(valid, levels=[0.5], colors=["white"], linewidths=0.6)
    axes[3].set_title(
        "Unbounded BIM aligned with RGB\n"
        f"coverage {old_coverage:.1%} → {new_coverage:.1%}"
    )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(
        depth_artist,
        ax=[axes[1], axes[3]],
        shrink=0.76,
        pad=0.02,
        label="BIM camera-z depth (m, log colour scale)",
    )
    fig.suptitle(
        f"{args.sample_id} | all positive BIM hits valid; no 0.2–5.0 m prior filter",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "old_coverage": old_coverage,
                "unbounded_coverage": new_coverage,
                "new_only_fraction": new_only_fraction,
                "new_only_above_5m_fraction": float(recovered_above_5m.mean()),
                "new_only_within_0p2_5m_fraction": float(recovered_within_old_range.mean()),
                "minimum_hit_depth_m": minimum,
                "maximum_hit_depth_m": maximum,
                "included_categories": list(ENVELOPE_CATEGORIES),
                "validity_rule": "isfinite(depth) and depth > 0",
                "uses_gt": False,
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
