#!/usr/bin/env python3
"""Evaluate a frozen Huber scale plus PriorDA v1.1 condition on Area_1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.early_fusion import DenseDepthMetricAccumulator
from bim_priorda3.engine import build_loader, seed_everything
from bim_priorda3.models import FrozenHuberPriorDAV11BIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--priorda-state",
        type=Path,
        help="Optional project checkpoint containing trainable_model; default is official v1.1.",
    )
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--all-bim-control",
        action="store_true",
        help="Disable the configured effective-attention prior for a matched control.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        help="Override effective_attention_prior.top_fraction for a matched audit.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_batch(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "rgb",
        "base_depth",
        "bim_depth",
        "bim_valid",
        "scaled_depth",
        "gt_depth",
        "gt_valid",
    )
    return {key: raw[key].to(device=device, non_blocking=True) for key in keys}


def load_priorda_state(
    model: FrozenHuberPriorDAV11BIM,
    checkpoint_path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("trainable_model")
    if not isinstance(state, dict):
        raise TypeError("PriorDA project checkpoint lacks trainable_model")
    expected = {name for name in model.state_dict() if name.startswith("priorda.")}
    if set(state) != expected:
        raise RuntimeError(
            f"PriorDA state mismatch: missing={len(expected - set(state))}, "
            f"unexpected={len(set(state) - expected)}"
        )
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid loader settings")
    cfg = load_config(args.config)
    if args.all_bim_control:
        cfg.model.effective_attention_prior = None
    elif args.top_fraction is not None:
        if not 0 < args.top_fraction <= 1:
            raise ValueError("--top-fraction must lie in (0, 1]")
        cfg.model.effective_attention_prior.top_fraction = args.top_fraction
    seed_everything(int(cfg.experiment.seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    scale_path = resolve_project_path(cfg, cfg.model.frozen_scale.checkpoint)
    scale_sha = sha256_file(scale_path)
    if scale_sha != str(cfg.model.frozen_scale.checkpoint_sha256):
        raise RuntimeError("Frozen scale checkpoint SHA256 mismatch")
    prior_cfg = cfg.model.priorda_v11
    priorda_path = Path(
        hf_hub_download(
            repo_id=str(prior_cfg.checkpoint_repo),
            filename=str(prior_cfg.checkpoint_filename),
            revision=str(prior_cfg.checkpoint_revision),
            local_files_only=bool(prior_cfg.local_files_only),
        )
    ).resolve()
    if sha256_file(priorda_path) != str(prior_cfg.checkpoint_sha256):
        raise RuntimeError("Official PriorDA checkpoint SHA256 mismatch")
    scale_checkpoint = torch.load(scale_path, map_location="cpu", weights_only=False)
    model = FrozenHuberPriorDAV11BIM.from_checkpoints(
        cfg,
        scale_checkpoint=scale_checkpoint,
        priorda_checkpoint_path=priorda_path,
    )
    weights_source = "official PriorDA v1.1 pretrained checkpoint"
    state_metadata = None
    if args.priorda_state is not None:
        state_path = args.priorda_state.expanduser().resolve()
        state_metadata = load_priorda_state(model, state_path)
        weights_source = str(state_path)
    model.to(device).eval()

    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset.records = dataset.records[: args.max_samples]
    loader = build_loader(
        dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )
    metrics = {
        name: DenseDepthMetricAccumulator()
        for name in ("raw_da3", "fixed_attention_huber_scale", "trusted_local", "priorda")
    }
    trusted_pixels = 0
    bim_pixels = 0
    local_support_sum = 0.0
    frames = 0
    amp = bool(cfg.train.amp) and device.type == "cuda"
    with torch.inference_mode():
        for raw_batch in loader:
            batch = selected_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(batch)
            support = (
                (batch["gt_valid"] > 0)
                & torch.isfinite(batch["gt_depth"])
                & (batch["gt_depth"] > 0)
            )
            predictions = {
                "raw_da3": batch["base_depth"],
                "fixed_attention_huber_scale": output["scaled_depth"],
                "trusted_local": output["local_depth"],
                "priorda": output["depth"],
            }
            for name, prediction in predictions.items():
                metrics[name].update(prediction, batch["gt_depth"], support)
            trusted_pixels += int((output["condition_bim_valid"] > 0).sum().item())
            bim_pixels += int((batch["bim_valid"] > 0).sum().item())
            local_support_sum += float(output["local_support"].mean()) * len(batch["rgb"])
            frames += len(batch["rgb"])
            if frames == len(dataset) or frames % args.log_every < len(batch["rgb"]):
                print(f"[{frames}/{len(dataset)}]", flush=True)

    computed = {name: accumulator.compute() for name, accumulator in metrics.items()}
    result = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "frames": frames,
        "support": "official all positive non-sentinel GT depth",
        "alignment": "none",
        "scale_checkpoint": str(scale_path),
        "scale_checkpoint_sha256": scale_sha,
        "priorda_weights_source": weights_source,
        "priorda_state_epoch": (
            None if state_metadata is None else int(state_metadata.get("epoch", -1))
        ),
        "condition_semantics": output["condition_semantics"],
        "effective_attention_prior": (
            None
            if cfg.model.effective_attention_prior is None
            else dict(cfg.model.effective_attention_prior)
        ),
        "trusted_bim_fraction_of_original_bim_hits": (
            trusted_pixels / bim_pixels if bim_pixels else float("nan")
        ),
        "mean_local_support_fraction": local_support_sum / max(frames, 1),
        "metrics": computed,
        "priorda_relative_improvement_over_scale": (
            (computed["fixed_attention_huber_scale"]["abs_rel"] - computed["priorda"]["abs_rel"])
            / computed["fixed_attention_huber_scale"]["abs_rel"]
        ),
        "trusted_local_relative_improvement_over_scale": (
            (
                computed["fixed_attention_huber_scale"]["abs_rel"]
                - computed["trusted_local"]["abs_rel"]
            )
            / computed["fixed_attention_huber_scale"]["abs_rel"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
