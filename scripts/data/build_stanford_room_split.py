#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bim_priorda3.data.stanford_split import write_room_split_annotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, room-disjoint Area_1 annotation"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--train-rooms", type=int, default=30)
    parser.add_argument("--val-rooms", type=int, default=7)
    parser.add_argument("--test-rooms", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-trials", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = write_room_split_annotation(
        args.manifest,
        args.output,
        args.receipt,
        train_rooms=args.train_rooms,
        val_rooms=args.val_rooms,
        test_rooms=args.test_rooms,
        seed=args.seed,
        search_trials=args.search_trials,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
