#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.models import BIMPriorDA3


def colorize(depth: np.ndarray, max_depth: float) -> np.ndarray:
    normalized = np.clip(depth / max_depth, 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one prepared BIM-PriorDA3 sample")
    parser.add_argument("--config", default="configs/slabim_single_frame.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/inference.png"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    batch = dataset[args.index]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    tensor_batch = {
        key: value[None].to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        output = model(tensor_batch)
    rgb = (batch["rgb"].permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8)
    base = batch["base_depth"][0].numpy()
    bim = batch["bim_depth"][0].numpy()
    refined = output["depth"][0, 0].cpu().numpy()
    trust = output["trust_probability"][0, 0].cpu().numpy()
    maximum = float(cfg.data.max_depth)
    trust_color = cv2.applyColorMap((trust * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    canvas = np.concatenate(
        (rgb, colorize(base, maximum), colorize(bim, maximum), colorize(refined, maximum), trust_color),
        axis=1,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"RGB | DA3 | BIM | refined | learned trust -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
