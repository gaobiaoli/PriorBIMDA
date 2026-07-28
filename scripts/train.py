#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import (
    build_loader,
    move_batch,
    save_checkpoint,
    seed_everything,
    validate,
    write_history,
)
from bim_priorda3.losses import BIMPriorLoss
from bim_priorda3.models import BIMPriorDA3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-frame BIM-PriorDA3 refiner")
    parser.add_argument("--config", default="configs/slabim_single_frame.yaml")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load model weights only and start a fresh optimizer/schedule",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override configured learning rate for a fresh optimizer",
    )
    parser.add_argument("--output-dir", type=Path, help="Override experiment output directory")
    parser.add_argument("--max-train-samples", type=int, help="Smoke-test subset size")
    parser.add_argument("--max-val-samples", type=int, help="Smoke-test subset size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    cfg = load_config(args.config)
    seed_everything(int(cfg.experiment.seed))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_project_path(cfg, cfg.experiment.output_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = BIMDepthDataset(cfg, "train")
    val_set = BIMDepthDataset(cfg, "val", augment=False)
    if args.max_train_samples:
        train_set = Subset(train_set, range(min(args.max_train_samples, len(train_set))))
    if args.max_val_samples:
        val_set = Subset(val_set, range(min(args.max_val_samples, len(val_set))))
    train_loader = build_loader(
        train_set,
        int(cfg.train.batch_size),
        int(cfg.train.num_workers),
        shuffle=True,
    )
    val_loader = build_loader(
        val_set,
        1,
        int(cfg.train.num_workers),
        shuffle=False,
    )
    model = BIMPriorDA3(cfg).to(device)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        print(f"Initialized model weights from {args.init_checkpoint}")
    criterion = BIMPriorLoss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            float(args.learning_rate)
            if args.learning_rate is not None
            else float(cfg.train.learning_rate)
        ),
        weight_decay=float(cfg.train.weight_decay),
    )
    total_epochs = args.epochs or int(cfg.train.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    start_epoch, best_metric = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state["best_metric"])

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    accumulation = int(cfg.train.gradient_accumulation)
    history_path = output_dir / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if args.resume and history_path.exists()
        else []
    )
    if history:
        best_history_index = min(
            range(len(history)),
            key=lambda index: history[index]["val_refined_abs_rel"],
        )
        stale_epochs = len(history) - 1 - best_history_index
    else:
        stale_epochs = 0
    print(
        f"device={device}, train={len(train_set)}, val={len(val_set)}, "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    try:
        for epoch in range(start_epoch, total_epochs):
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            for step, batch in enumerate(train_loader, 1):
                batch = move_batch(batch, device)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    output = model(batch)
                    losses = criterion(output, batch)
                    scaled_loss = losses["total"] / accumulation
                scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.train.gradient_clip_norm)
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                if step % int(cfg.train.log_every) == 0:
                    print(
                        f"epoch={epoch + 1}/{total_epochs} step={step}/{len(train_loader)} "
                        f"loss={sums['total'] / step:.5f}",
                        flush=True,
                    )
            scheduler.step()
            validation = validate(model, val_loader, criterion, device, use_amp)
            row = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "peak_cuda_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                ),
                **{f"train_{key}": value / len(train_loader) for key, value in sums.items()},
                **{f"val_{key}": value for key, value in validation.items()},
            }
            history.append(row)
            write_history(history_path, history)
            improved = validation["refined_abs_rel"] < best_metric
            if improved:
                best_metric = validation["refined_abs_rel"]
                stale_epochs = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metric,
                    cfg,
                )
            else:
                stale_epochs += 1
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                cfg,
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if stale_epochs >= int(cfg.train.early_stopping_patience):
                print("Early stopping: validation AbsRel stopped improving")
                break
    except torch.cuda.OutOfMemoryError:
        emergency = output_dir / "oom_state.pt"
        torch.save({"model": model.state_dict(), "config": dict(cfg)}, emergency)
        message = (
            f"CUDA out of memory. State saved to {emergency}. "
            "Reduce target resolution/channels or move this project to the cloud server."
        )
        (output_dir / "OOM_README.txt").write_text(message + "\n", encoding="utf-8")
        print(message)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
