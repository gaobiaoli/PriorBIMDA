#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root
from bim_priorda3.data import relocate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared data and split integrity")
    parser.add_argument("--config", default="configs/slabim_single_frame.yaml")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = resolve_project_path(cfg, cfg.data.processed_root)
    records = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [
        relocate_record(record, root, resolve_slabim_root(cfg))
        for record in records
    ]
    train_regions = set(cfg.data.train_regions)
    val_regions = set(cfg.data.val_regions)
    test_regions = set(cfg.data.test_regions)
    overlap = (
        (train_regions & val_regions)
        | (train_regions & test_regions)
        | (val_regions & test_regions)
    )
    if overlap:
        raise RuntimeError(f"Train/validation region leakage: {sorted(overlap)}")

    region_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "samples": 0,
            "gt_pixels": 0,
            "bim_pixels": 0,
            "overlap_pixels": 0,
            "bim_wins": 0,
            "da3_wins": 0,
            "base_abs_error_sum": 0.0,
            "bim_abs_error_sum": 0.0,
        }
    )
    for record in records:
        sample = np.load(record["sample"])
        base = sample["base_depth"].astype(np.float32)
        bim = sample["bim_depth"].astype(np.float32)
        gt = sample["gt_depth"].astype(np.float32)
        gt_valid = sample["gt_valid"] > 0
        bim_valid = sample["bim_valid"] > 0
        overlap_mask = gt_valid & bim_valid & (base > 0) & (bim > 0) & (gt > 0)
        base_error = np.abs(np.log(np.maximum(base[overlap_mask], 1e-4)) - np.log(gt[overlap_mask]))
        bim_error = np.abs(np.log(np.maximum(bim[overlap_mask], 1e-4)) - np.log(gt[overlap_mask]))
        stats = region_stats[record["region"]]
        stats["samples"] += 1
        stats["gt_pixels"] += int(gt_valid.sum())
        stats["bim_pixels"] += int(bim_valid.sum())
        stats["overlap_pixels"] += int(overlap_mask.sum())
        stats["bim_wins"] += int((bim_error < base_error).sum())
        stats["da3_wins"] += int((base_error <= bim_error).sum())
        stats["base_abs_error_sum"] += float(base_error.sum())
        stats["bim_abs_error_sum"] += float(bim_error.sum())

    for stats in region_stats.values():
        samples = max(stats["samples"], 1)
        overlap_pixels = max(stats["overlap_pixels"], 1)
        stats["mean_gt_pixels"] = stats["gt_pixels"] / samples
        stats["mean_bim_pixels"] = stats["bim_pixels"] / samples
        stats["bim_win_fraction"] = stats["bim_wins"] / overlap_pixels
        stats["mean_base_log_error"] = stats["base_abs_error_sum"] / overlap_pixels
        stats["mean_bim_log_error"] = stats["bim_abs_error_sum"] / overlap_pixels
    report = {
        "manifest": str((root / "manifest.jsonl").resolve()),
        "samples": len(records),
        "train_regions": sorted(train_regions),
        "val_regions": sorted(val_regions),
        "test_regions": sorted(test_regions),
        "region_leakage": False,
        "prepared_train_samples": sum(
            stats["samples"] for name, stats in region_stats.items() if name in train_regions
        ),
        "prepared_val_samples": sum(
            stats["samples"] for name, stats in region_stats.items() if name in val_regions
        ),
        "prepared_test_samples": sum(
            stats["samples"] for name, stats in region_stats.items() if name in test_regions
        ),
        "regions": dict(region_stats),
    }
    output = args.output or root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
