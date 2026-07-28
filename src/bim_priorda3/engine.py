from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from bim_priorda3.losses import BIMPriorLoss
from bim_priorda3.metrics import depth_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: BIMPriorLoss,
    device: torch.device,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    base_predictions, refined_predictions, targets, masks = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(batch)
            loss = criterion(output, batch)["total"]
        losses.append(float(loss))
        base_predictions.append(batch["base_depth"].cpu())
        refined_predictions.append(output["depth"].cpu())
        targets.append(batch["gt_depth"].cpu())
        masks.append(batch["gt_valid"].cpu())
    target = torch.cat(targets)
    mask = torch.cat(masks)
    base = depth_metrics(torch.cat(base_predictions), target, mask)
    refined = depth_metrics(torch.cat(refined_predictions), target, mask)
    return {
        "loss": float(np.mean(losses)),
        **{f"base_{key}": value for key, value in base.items()},
        **{f"refined_{key}": value for key, value in refined.items()},
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_metric: float,
    cfg: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": dict(cfg),
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)

