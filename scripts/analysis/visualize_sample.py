#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bim_priorda3.baselines import configured_scale_and_local_features
from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root
from bim_priorda3.data import relocate_record


def colorize(depth: np.ndarray, valid: np.ndarray, maximum: float) -> np.ndarray:
    normalized = np.clip(depth / maximum, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def label(panel: np.ndarray, text: str) -> np.ndarray:
    output = panel.copy()
    cv2.putText(
        output,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one prepared training sample")
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-id", default="5F_Region2/000000")
    parser.add_argument("--output", type=Path, default=Path("outputs/sample_audit.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = resolve_project_path(cfg, cfg.data.processed_root)
    records = {
        record["id"]: record
        for record in (
            json.loads(line)
            for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    record = relocate_record(
        records[args.sample_id],
        root,
        resolve_slabim_root(cfg),
    )
    sample = np.load(record["sample"])
    height, width = int(cfg.data.target_height), int(cfg.data.target_width)
    rgb = cv2.resize(cv2.imread(record["image"]), (width, height), interpolation=cv2.INTER_AREA)
    base = sample["base_depth"].astype(np.float32)
    bim = sample["bim_depth"].astype(np.float32)
    scaled = configured_scale_and_local_features(
        base,
        bim,
        cfg.model.get("scale_estimator"),
    )[0]
    gt = sample["gt_depth"].astype(np.float32)
    base_valid = base > 0
    scaled_valid = scaled > 0
    bim_valid = sample["bim_valid"] > 0
    gt_valid = sample["gt_valid"] > 0
    overlap = bim_valid & gt_valid & scaled_valid
    scaled_error = np.abs(np.log(np.maximum(scaled, 1e-4)) - np.log(np.maximum(gt, 1e-4)))
    bim_error = np.abs(np.log(np.maximum(bim, 1e-4)) - np.log(np.maximum(gt, 1e-4)))
    trust = np.zeros_like(base)
    trust[overlap] = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (scaled_error[overlap] - bim_error[overlap] - float(cfg.loss.trust_margin))
                / float(cfg.loss.trust_temperature),
                -30,
                30,
            )
        )
    )
    trust_color = cv2.applyColorMap((trust * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    trust_color[~overlap] = 0
    maximum = float(cfg.data.max_depth)
    panels = (
        label(rgb, "RGB"),
        label(colorize(base, base_valid, maximum), "Raw DA3"),
        label(colorize(scaled, scaled_valid, maximum), "Scale-corrected DA3"),
        label(colorize(bim, bim_valid, maximum), "BIM"),
        label(colorize(gt, gt_valid, maximum), "Fused LiDAR GT"),
        label(trust_color, "BIM trust target"),
    )
    canvas = np.concatenate(panels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"{args.sample_id}: {args.output.resolve()}")


if __name__ == "__main__":
    main()
