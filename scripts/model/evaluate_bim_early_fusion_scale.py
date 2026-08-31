#!/usr/bin/env python3
"""Evaluate a trained DAv2 early-fusion single-scale checkpoint on Area_1."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import build_loader
from bim_priorda3.models import BIMEarlyFusionDAv2ScaleRegressor
from train_bim_early_fusion_scale import (
    atomic_json,
    evaluate,
    resolve_checkpoint,
    seed_everything,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--confirm-test", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_test:
        parser.error("--split test requires --confirm-test")
    return args


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if str(cfg.data.ground_truth_support) != "official_all_valid":
        raise ValueError("Evaluation requires data.ground_truth_support=official_all_valid")

    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    loader = build_loader(
        dataset,
        int(cfg.train.val_batch_size),
        int(cfg.train.num_workers),
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + (19 if args.split == "test" else 17)),
        persistent_workers=False,
    )

    pretrained_path, model_id, revision = resolve_checkpoint(cfg)
    scale_cfg = cfg.model.dav2_scale
    model = BIMEarlyFusionDAv2ScaleRegressor.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=bool(cfg.model.dav2.local_files_only),
        regression_hidden_size=int(scale_cfg.regression_hidden_size),
        head_dropout_probability=float(scale_cfg.head_dropout_probability),
        output_weight_std=float(scale_cfg.output_weight_std),
    ).to(device)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    normalization = cfg.model.bim_normalization
    bim_stats = {
        "mean": float(normalization.mean),
        "std": float(normalization.std),
    }
    checkpoint_stats = state.get("bim_log_statistics", {})
    if (
        float(checkpoint_stats.get("mean", float("nan"))) != bim_stats["mean"]
        or float(checkpoint_stats.get("std", float("nan"))) != bim_stats["std"]
    ):
        raise RuntimeError("Checkpoint and evaluation BIM normalization differ")

    summary = evaluate(
        model,
        loader,
        device=device,
        bim_stats=bim_stats,
        amp=bool(cfg.train.amp) and device.type == "cuda",
        oracle_min_support=int(cfg.loss.attention_scale_oracle_min_support),
    )
    summary.update(
        {
            "split": args.split,
            "selected_checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(state["epoch"]),
            "best_epoch": int(state["best_epoch"]),
            "pretrained_checkpoint_sha256": sha256_file(pretrained_path),
            "dataset_split_provenance": dataset.split_provenance,
        }
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else resolve_project_path(cfg, cfg.experiment.results_dir) / f"{args.split}_summary.json"
    )
    atomic_json(output, summary)
    learned = summary["dav2_early_fusion_scale"]
    raw = summary["raw_da3_focal_corrected"]
    print(
        f"split={args.split} frames={summary['frames']} "
        f"raw_abs_rel={float(raw['abs_rel']):.6f} "
        f"learned_abs_rel={float(learned['abs_rel']):.6f} "
        f"scale_log_mae={float(summary['scale']['oracle_log_scale_mae']):.6f} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
