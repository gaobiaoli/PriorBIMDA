#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bim_priorda3.baselines import bim_scale_and_local_features
from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root
from bim_priorda3.data import relocate_record

BASELINE_KEYS = (
    "scaled_depth",
    "anchor_depth",
)
OBSOLETE_KEYS = ("anchor_field", "anchor_support")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache scale-only and direct local BIM baselines in prepared samples"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
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
    records = [relocate_record(record, root, resolve_slabim_root(cfg)) for record in records]
    updated = 0
    for index, record in enumerate(records, 1):
        path = Path(record["sample"])
        with np.load(path) as item:
            if not args.overwrite and set(BASELINE_KEYS).issubset(item.files):
                continue
            payload = {
                key: item[key] for key in item.files if key not in (*BASELINE_KEYS, *OBSOLETE_KEYS)
            }
        scaled, anchor, _, _, _ = bim_scale_and_local_features(
            payload["base_depth"].astype(np.float32),
            payload["bim_depth"].astype(np.float32),
        )
        payload.update(
            scaled_depth=scaled.astype(np.float16),
            anchor_depth=anchor.astype(np.float16),
        )
        np.savez_compressed(path, **payload)
        updated += 1
        if index % 50 == 0 or index == len(records):
            print(f"{index}/{len(records)} scanned, {updated} updated", flush=True)
    print(f"Finished: {updated}/{len(records)} samples updated")


if __name__ == "__main__":
    main()
