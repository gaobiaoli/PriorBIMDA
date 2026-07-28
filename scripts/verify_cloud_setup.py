#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import cv2
import numpy as np
import torch

from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.models import BIMPriorDA3


REQUIRED_SAMPLE_KEYS = {
    "base_depth",
    "base_confidence",
    "bim_depth",
    "bim_valid",
    "gt_depth",
    "gt_valid",
    "scaled_depth",
    "anchor_depth",
    "anchor_field",
    "anchor_support",
    "candidate_depth",
    "candidate_log_variance",
    "candidate_trust",
    "candidate_frame_trust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a relocated cloud workspace")
    parser.add_argument(
        "--config",
        default="configs/slabim_single_frame_r50_v3.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/slabim_single_frame_r50_v3/best.pt"),
    )
    return parser.parse_args()


def add_batch_dimension(
    item: dict[str, object], device: torch.device
) -> dict[str, object]:
    return {
        key: value[None].to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    processed = resolve_project_path(cfg, cfg.data.processed_root)
    slabim = resolve_slabim_root(cfg)
    checkpoint = args.checkpoint.expanduser().resolve()
    print(f"python={platform.python_version()}")
    print(f"torch={torch.__version__}, torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"opencv={cv2.__version__}, numpy={np.__version__}")
    print(f"processed_root={processed}")
    print(f"slabim_root={slabim}")
    print(f"checkpoint={checkpoint}")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    expected_counts = {"train": 565, "val": 82, "test": 164}
    datasets: dict[str, BIMDepthDataset] = {}
    for split, expected in expected_counts.items():
        dataset = BIMDepthDataset(cfg, split, augment=False)
        datasets[split] = dataset
        if len(dataset) != expected:
            raise RuntimeError(
                f"{split}: expected {expected} samples, found {len(dataset)}"
            )
        record = dataset.records[0]
        if not Path(record["sample"]).exists():
            raise FileNotFoundError(record["sample"])
        if not Path(record["image"]).exists():
            raise FileNotFoundError(record["image"])
        with np.load(record["sample"]) as sample:
            missing = REQUIRED_SAMPLE_KEYS - set(sample.files)
        if missing:
            raise RuntimeError(f"{split}: missing cached keys {sorted(missing)}")
        print(f"{split}: {len(dataset)} samples OK")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg).to(device).eval()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    item = datasets["test"][0]
    batch = add_batch_dimension(item, device)
    with torch.no_grad():
        original = model(batch)["depth"]
    randomized = dict(batch)
    for key in ("gt_depth", "gt_valid", "gt_weight", "trust_target", "trust_mask"):
        randomized[key] = torch.randn_like(randomized[key]) * 1000.0
    with torch.no_grad():
        changed = model(randomized)["depth"]
    difference = float((original - changed).abs().max())
    if difference != 0.0:
        raise RuntimeError(f"GT leakage check failed: max output change={difference}")
    if not torch.isfinite(original).all():
        raise RuntimeError("Inference produced non-finite depth")
    print(
        f"inference OK: shape={tuple(original.shape)}, "
        f"min={float(original.min()):.4f}, max={float(original.max()):.4f}"
    )
    print("GT independence OK: max output change=0.0")
    print("CLOUD SETUP VERIFIED")


if __name__ == "__main__":
    main()
