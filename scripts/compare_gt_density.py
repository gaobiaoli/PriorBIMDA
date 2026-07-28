#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sample_path(root: Path, record: dict) -> Path:
    """Resolve samples by stable region/name instead of stale absolute manifest paths."""
    manifest_path = Path(record["sample"]).expanduser()
    return root / "samples" / str(record["region"]) / manifest_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare valid GT coverage between two datasets")
    parser.add_argument("--old", type=Path, default=Path("data/processed/slabim_504"))
    parser.add_argument("--new", type=Path, default=Path("data/processed/slabim_504_r50"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gt_density_r10_vs_r50.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(line)
        for line in (args.new / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    stats = defaultdict(lambda: {"samples": 0, "old_pixels": 0, "new_pixels": 0, "unchanged": 0})
    for record in records:
        old_path = sample_path(args.old.resolve(), record)
        new_path = sample_path(args.new.resolve(), record)
        with np.load(old_path) as old, np.load(new_path) as new:
            old_count = int(old["gt_valid"].sum())
            new_count = int(new["gt_valid"].sum())
        region = record["region"]
        stats[region]["samples"] += 1
        stats[region]["old_pixels"] += old_count
        stats[region]["new_pixels"] += new_count
        stats[region]["unchanged"] += int(old_count == new_count)
    for values in stats.values():
        values["old_mean_pixels"] = values["old_pixels"] / values["samples"]
        values["new_mean_pixels"] = values["new_pixels"] / values["samples"]
        values["relative_density_gain"] = (
            values["new_pixels"] - values["old_pixels"]
        ) / values["old_pixels"]
    report = {
        "old": str(args.old.resolve()),
        "new": str(args.new.resolve()),
        "regions": dict(stats),
        "all_samples_refreshed": all(values["unchanged"] == 0 for values in stats.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
