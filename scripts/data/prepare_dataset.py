#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bim_priorda3.config import load_config
from bim_priorda3.data.preparation import prepare_region, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SLABIM single-frame training samples")
    parser.add_argument("--config", required=True)
    parser.add_argument("--regions", nargs="*", help="Override configured region list")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replace-regions-in-manifest",
        action="store_true",
        help="Replace selected regions' manifest records so changed stride cannot leave stale samples",
    )
    parser.add_argument(
        "--refresh-gt-only",
        action="store_true",
        help="Preserve RGB/DA3/BIM arrays and only recompute fused LiDAR GT",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Prepare RGB/DA3/BIM inputs without reading PCDs or creating GT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_gt_only and args.inference_only:
        raise ValueError("--refresh-gt-only and --inference-only are mutually exclusive")
    cfg = load_config(args.config)
    regions = args.regions or cfg.data.regions
    all_records = []
    for region in regions:
        all_records.extend(
            prepare_region(
                cfg,
                region,
                max_frames=args.max_frames,
                stride=args.stride,
                overwrite=args.overwrite,
                refresh_gt_only=args.refresh_gt_only,
                inference_only=args.inference_only,
            )
        )
    manifest = write_manifest(
        cfg,
        all_records,
        replace_regions=set(regions) if args.replace_regions_in_manifest else None,
    )
    print(f"Prepared {len(all_records)} samples: {manifest}")


if __name__ == "__main__":
    main()
