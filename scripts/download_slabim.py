#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bim_priorda3.data.slabim import (
    ALL_REGIONS,
    DEFAULT_REGIONS,
    SLABIM_REPOSITORY,
    download_regions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumably download SLABIM BIM, calibration, images, PCDs and optional rosbags"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regions", nargs="+", choices=ALL_REGIONS, default=DEFAULT_REGIONS)
    parser.add_argument("--include-rosbag", action="store_true")
    parser.add_argument(
        "--rosbag-only",
        action="store_true",
        help="Only add rosbags to already downloaded regions",
    )
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--repository", default=SLABIM_REPOSITORY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download_regions(
        args.root,
        args.regions,
        include_rosbag=args.include_rosbag,
        rosbag_only=args.rosbag_only,
        repository=args.repository,
        keep_archives=args.keep_archives,
    )


if __name__ == "__main__":
    main()

