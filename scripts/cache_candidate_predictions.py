#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import move_batch
from bim_priorda3.models import BIMPriorDA3


CANDIDATE_KEYS = (
    "candidate_depth",
    "candidate_log_variance",
    "candidate_trust",
    "candidate_frame_trust",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache predictions from a frozen learned candidate model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    updated = 0
    total = 0
    for split in ("train", "val", "test"):
        dataset = BIMDepthDataset(cfg, split, augment=False)
        for index, record in enumerate(dataset.records):
            total += 1
            path = Path(record["sample"])
            with np.load(path) as item:
                if not args.overwrite and set(CANDIDATE_KEYS).issubset(item.files):
                    continue
            batch = dataset[index]
            batch = {
                key: value[None] if torch.is_tensor(value) else [value]
                for key, value in batch.items()
            }
            batch = move_batch(batch, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                output = model(batch)
            payload_additions = {
                "candidate_depth": output["depth"][0, 0].float().cpu().numpy(),
                "candidate_log_variance": output["log_variance"][
                    0, 0
                ].float().cpu().numpy(),
                "candidate_trust": output["trust_probability"][
                    0, 0
                ].float().cpu().numpy(),
                "candidate_frame_trust": np.asarray(
                    torch.sigmoid(output["frame_trust_logits"])[0]
                    .float()
                    .cpu()
                    .item(),
                    dtype=np.float32,
                ),
            }
            with np.load(path) as item:
                payload = {key: item[key] for key in item.files if key not in CANDIDATE_KEYS}
            payload.update(
                {
                    key: value.astype(np.float16)
                    for key, value in payload_additions.items()
                }
            )
            np.savez_compressed(path, **payload)
            updated += 1
            if updated % 50 == 0:
                print(f"{updated} candidate predictions cached", flush=True)
    print(f"Finished: {updated}/{total} samples updated")


if __name__ == "__main__":
    main()
