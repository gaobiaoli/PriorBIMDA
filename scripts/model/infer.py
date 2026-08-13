#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from bim_priorda3.checkpoints import validate_checkpoint_model_config
from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.models import BIMPriorDA3


def colorize(depth: np.ndarray, max_depth: float) -> np.ndarray:
    normalized = np.clip(depth / max_depth, 0.0, 1.0)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V5 on prepared RGB/DA3/BIM inputs without loading GT"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--regions", nargs="+")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inference"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-previews", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=False,
        regions=args.regions,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg).to(device)
    checkpoint = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_model_config(state, cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    limit = len(dataset)
    if args.max_samples is not None:
        limit = min(limit, args.max_samples)

    records = []
    with torch.inference_mode():
        for index in range(limit):
            item = dataset[index]
            tensor_batch = {
                key: value[None].to(device) if torch.is_tensor(value) else value
                for key, value in item.items()
            }
            output = model(tensor_batch)
            sample_id = str(item["sample_id"])
            relative = Path(sample_id)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe sample id in manifest: {sample_id!r}")
            depth_path = (output_dir / relative).with_suffix(".npz")
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            refined = output["depth"][0, 0].float().cpu().numpy()
            scaled = item["scaled_depth"][0].numpy()
            reliability = output["bim_reliability"][0, 0].float().cpu().numpy()
            np.savez_compressed(
                depth_path,
                depth=refined.astype(np.float32),
                scaled_depth=scaled.astype(np.float32),
                bim_reliability=reliability.astype(np.float32),
                log_variance=output["log_variance"][0, 0].float().cpu().numpy().astype(np.float32),
                log_residual=output["log_residual"][0, 0].float().cpu().numpy().astype(np.float32),
            )

            preview_path = None
            if args.save_previews:
                rgb = (item["rgb"].permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8)
                base = item["base_depth"][0].numpy()
                bim = item["bim_depth"][0].numpy()
                maximum = float(cfg.data.max_depth)
                reliability_color = cv2.applyColorMap(
                    (reliability * 255).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS,
                )
                canvas = np.concatenate(
                    (
                        rgb,
                        colorize(base, maximum),
                        colorize(scaled, maximum),
                        colorize(bim, maximum),
                        colorize(refined, maximum),
                        reliability_color,
                    ),
                    axis=1,
                )
                preview_path = depth_path.with_suffix(".png")
                cv2.imwrite(str(preview_path), canvas)

            records.append(
                {
                    "sample_id": sample_id,
                    "depth": str(depth_path),
                    "preview": str(preview_path) if preview_path else None,
                    "mean_depth": float(refined.mean()),
                    "mean_abs_log_residual": float(
                        np.abs(output["log_residual"][0, 0].float().cpu().numpy()).mean()
                    ),
                }
            )
            print(f"[{index + 1}/{limit}] {sample_id} -> {depth_path}", flush=True)

    summary = {
        "config": str(Path(cfg.config_path).resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "device": str(device),
        "regions": sorted({str(record["region"]) for record in dataset.records[:limit]}),
        "samples": len(records),
        "ground_truth_loaded": False,
        "records": records,
    }
    summary_path = output_dir / "inference.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Inference complete: {summary_path}")


if __name__ == "__main__":
    main()
