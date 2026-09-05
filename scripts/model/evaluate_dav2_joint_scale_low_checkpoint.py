#!/usr/bin/env python3
"""Evaluate a saved joint DAv2 scale+r_low checkpoint on Stanford val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import build_loader
from evaluate_matterport_bimnet_scale_refiner import _load_joint_dav2_scale_low_model
from train_bim_early_fusion_scale import atomic_json
from train_dav2_joint_scale_low import evaluate, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.experiment.seed)
    seed_everything(seed, deterministic=args.deterministic)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = _load_joint_dav2_scale_low_model(cfg, checkpoint).to(device).eval()
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    generator_offset = 17 if args.split == "val" else 19
    loader = build_loader(
        dataset,
        int(cfg.train.val_batch_size),
        int(cfg.train.num_workers),
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + generator_offset),
        persistent_workers=False,
    )
    stats = cfg.model.bim_normalization
    summary = evaluate(
        model,
        loader,
        device=device,
        bim_stats={
            "mean": float(stats.mean),
            "std": float(stats.std),
            "valid_pixels": int(stats.valid_pixels),
            "train_records": int(checkpoint["config"]["data"]["split_provenance"]["split_counts"]["train"])
            if "split_provenance" in checkpoint["config"]["data"]
            else 0,
        },
        amp=bool(cfg.train.amp) and device.type == "cuda",
        oracle_min_support=int(cfg.loss.attention_scale_oracle_min_support),
    )
    summary.update(
        {
            "selected_checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "best_epoch": int(checkpoint["best_epoch"]),
        }
    )
    atomic_json(args.output, summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
