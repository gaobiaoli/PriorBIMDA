#!/usr/bin/env python3
from __future__ import annotations

import argparse

from bim_priorda3.config import load_config
from bim_priorda3.data.stanford_preparation import (
    prepare_stanford_area1,
    write_stanford_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Stanford Area_1 + fixed BIMSyn-envelope samples"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--rooms", nargs="+", default=None)
    parser.add_argument("--max-frames-per-room", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_canonical_preparation(
    *,
    rooms: list[str] | None,
    max_frames_per_room: int | None,
    stride: int,
) -> bool:
    """Return whether the invocation covers the canonical full population."""
    return rooms is None and max_frames_per_room is None and stride == 1


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    records, metadata = prepare_stanford_area1(
        cfg,
        rooms=set(args.rooms) if args.rooms is not None else None,
        max_frames_per_room=args.max_frames_per_room,
        stride=args.stride,
        overwrite=args.overwrite,
    )
    if is_canonical_preparation(
        rooms=args.rooms,
        max_frames_per_room=args.max_frames_per_room,
        stride=args.stride,
    ):
        manifest, metadata_path = write_stanford_manifest(cfg, records, metadata)
        print(f"Wrote {len(records)} records to {manifest}")
        print(f"Wrote preparation receipt to {metadata_path}")
    else:
        print(f"Prepared {len(records)} filtered sample artifacts.")
        print(
            "Canonical manifest.jsonl and metadata.json were left unchanged. "
            "Run without --rooms/--max-frames-per-room and with --stride 1 "
            "to publish the complete canonical dataset."
        )


if __name__ == "__main__":
    main()
