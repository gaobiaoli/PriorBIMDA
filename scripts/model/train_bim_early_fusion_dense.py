#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset, Subset

from bim_priorda3.config import Config, load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.early_fusion import (
    DenseDepthMetricAccumulator,
    compute_train_bim_log_statistics,
    dense_metric_depth_loss,
    fixed_depth_support,
)
from bim_priorda3.models import BIMEarlyFusionDepthAnythingV2, build_bim_condition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train zero-initialized BIM early fusion in DAv2 Metric Indoor Base"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        generator=generator,
        drop_last=False,
    )


def selected_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    required = ("rgb", "base_depth", "bim_depth", "bim_valid", "gt_depth", "gt_valid")
    output = {
        key: batch[key].to(device=device, non_blocking=True)
        for key in required
    }
    shapes = {key: tuple(value.shape[-2:]) for key, value in output.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Early-fusion modalities are not pixel-aligned: {shapes}")
    return output


def resolve_checkpoint(cfg: Config) -> tuple[Path, str, str]:
    model_id = str(cfg.model.dav2.model_id)
    revision = str(cfg.model.dav2.revision)
    checkpoint = Path(
        hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            revision=revision,
            local_files_only=bool(cfg.model.dav2.local_files_only),
        )
    ).resolve()
    actual_sha = sha256_file(checkpoint)
    expected_sha = str(cfg.model.dav2.checkpoint_sha256)
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"DAv2 checkpoint SHA256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )
    return checkpoint, model_id, revision


def evaluate(
    model: BIMEarlyFusionDepthAnythingV2,
    loader: DataLoader,
    *,
    device: torch.device,
    bim_stats: dict[str, Any],
    min_depth: float,
    max_depth: float,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    learned = DenseDepthMetricAccumulator()
    raw_da3 = DenseDepthMetricAccumulator()
    total_seconds = 0.0
    frames = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = selected_batch(raw_batch, device)
            condition = build_bim_condition(
                batch,
                bim_log_mean=float(bim_stats["mean"]),
                bim_log_std=float(bim_stats["std"]),
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(batch["rgb"], condition)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - start
            support = fixed_depth_support(
                batch["gt_depth"],
                batch["gt_valid"],
                min_depth=min_depth,
                max_depth=max_depth,
            )
            learned.update(prediction, batch["gt_depth"], support)
            raw_da3.update(batch["base_depth"], batch["gt_depth"], support)
            frames += batch["rgb"].shape[0]
    learned_metrics = learned.compute()
    raw_metrics = raw_da3.compute()
    return {
        "frames": frames,
        "support": f"fixed GT support {min_depth:g}-{max_depth:g} m",
        "alignment": "none",
        "bim_early_fusion_dense": learned_metrics,
        "raw_da3_focal_corrected": raw_metrics,
        "relative_abs_rel_improvement": (
            (float(raw_metrics["abs_rel"]) - float(learned_metrics["abs_rel"]))
            / float(raw_metrics["abs_rel"])
        ),
        "inference_seconds": total_seconds,
    }


def checkpoint_payload(
    *,
    model: BIMEarlyFusionDepthAnythingV2,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_epoch: int,
    best_abs_rel: float,
    history: list[dict[str, Any]],
    optimizer_steps: int,
    skipped_steps: int,
    bim_stats: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_validation_abs_rel": best_abs_rel,
        "history": history,
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "bim_log_statistics": bim_stats,
        "config": config,
    }


def git_receipt(project_root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "dirty": bool(command("status", "--porcelain")),
    }


def main() -> None:
    args = parse_args()
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    for name in ("max_train_samples", "max_val_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    cfg = load_config(args.config)
    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else resolve_project_path(cfg, cfg.experiment.output_dir)
    )
    results_dir = resolve_project_path(cfg, cfg.experiment.results_dir)
    if args.resume is None and any(
        (output_dir / name).exists() for name in ("best.pt", "latest.pt", "training_history.csv")
    ):
        raise FileExistsError(f"Fresh run refuses existing training artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BIMDepthDataset(cfg, "train", augment=True)
    val_dataset = BIMDepthDataset(cfg, "val", augment=False)
    test_dataset = BIMDepthDataset(cfg, "test", augment=False)
    normalization_config = cfg.model.get("bim_normalization", {})
    configured_mean = normalization_config.get("mean")
    configured_std = normalization_config.get("std")
    if configured_mean is None or configured_std is None:
        bim_stats = compute_train_bim_log_statistics(train_dataset.records)
    else:
        bim_stats = {
            "mean": float(configured_mean),
            "std": float(configured_std),
            "valid_pixels": int(normalization_config["valid_pixels"]),
            "train_records": len(train_dataset.records),
            "definition": "configured train-only BIM log-depth moments",
        }
    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset,
            range(min(args.max_train_samples, len(train_dataset))),
        )
    if args.max_val_samples is not None:
        val_dataset = Subset(
            val_dataset,
            range(min(args.max_val_samples, len(val_dataset))),
        )

    checkpoint_path, model_id, revision = resolve_checkpoint(cfg)
    model = BIMEarlyFusionDepthAnythingV2.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=bool(cfg.model.dav2.local_files_only),
    ).to(device)
    initialization = model.initialization_audit(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    atomic_json(output_dir / "initialization_verification.json", initialization)
    if not bool(initialization["all_pass"]):
        raise RuntimeError("Pretrained initialization verification failed; training is forbidden")
    if bool(cfg.train.get("gradient_checkpointing", True)):
        model.enable_gradient_checkpointing()

    encoder_lr = float(cfg.train.encoder_learning_rate)
    decoder_lr = float(cfg.train.decoder_learning_rate)
    condition_lr = float(cfg.train.bim_condition_learning_rate)
    parameter_groups = model.optimizer_parameter_groups(
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        condition_lr=condition_lr,
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(args.epochs or cfg.train.epochs)
    batch_size = int(cfg.train.batch_size)
    gradient_accumulation = int(cfg.train.get("gradient_accumulation", 1))
    if gradient_accumulation < 1:
        raise ValueError("train.gradient_accumulation must be positive")
    train_loader = build_loader(
        train_dataset,
        batch_size=batch_size,
        workers=int(cfg.train.num_workers),
        shuffle=True,
        seed=seed,
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=int(cfg.train.val_batch_size),
        workers=int(cfg.train.num_workers),
        shuffle=False,
        seed=seed + 1,
    )
    steps_per_epoch = math.ceil(len(train_loader) / gradient_accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps_per_epoch * epochs),
        eta_min=0.0,
    )
    amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp,
        init_scale=float(cfg.train.get("amp_initial_scale", 1024.0)),
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_abs_rel = float("inf")
    optimizer_steps = 0
    skipped_steps = 0
    materialized_config = plain(cfg)
    materialized_config["model"]["bim_normalization"] = dict(bim_stats)
    materialized_config["runtime"] = {
        "epochs": epochs,
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
    }
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        history = list(state["history"])
        best_epoch = int(state["best_epoch"])
        best_abs_rel = float(state["best_validation_abs_rel"])
        optimizer_steps = int(state["optimizer_steps"])
        skipped_steps = int(state["skipped_steps"])
        saved_stats = state["bim_log_statistics"]
        if saved_stats != bim_stats:
            raise RuntimeError("Resume BIM normalization statistics differ")

    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(materialized_config, handle, sort_keys=False)
    shutil.copy2(output_dir / "config.yaml", results_dir / "config.yaml")
    min_depth = float(cfg.data.min_depth)
    max_depth = float(cfg.data.max_depth)
    gradient_weight = float(cfg.loss.gradient)
    training_start = time.perf_counter()
    log_every = int(cfg.train.log_every)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"total": 0.0, "log_depth": 0.0, "gradient": 0.0}
        samples = 0
        epoch_start = time.perf_counter()
        accumulation_count = 0
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = selected_batch(raw_batch, device)
            condition = build_bim_condition(
                batch,
                bim_log_mean=float(bim_stats["mean"]),
                bim_log_std=float(bim_stats["std"]),
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(batch["rgb"], condition)
                losses = dense_metric_depth_loss(
                    prediction,
                    batch["gt_depth"],
                    batch["gt_valid"],
                    min_depth=min_depth,
                    max_depth=max_depth,
                    gradient_weight=gradient_weight,
                )
                scaled_loss = losses["total"] / gradient_accumulation
            batch_samples = batch["rgb"].shape[0]
            for key in totals:
                totals[key] += float(losses[key].detach()) * batch_samples
            samples += batch_samples
            if not bool(torch.isfinite(scaled_loss)):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
                continue
            scaler.scale(scaled_loss).backward()
            accumulation_count += 1
            should_step = accumulation_count == gradient_accumulation or batch_index == len(train_loader)
            if should_step:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                step_skipped = scaler.get_scale() < scale_before
                if step_skipped:
                    skipped_steps += 1
                else:
                    scheduler.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
            if batch_index == 1 or batch_index % log_every == 0:
                elapsed = time.perf_counter() - epoch_start
                print(
                    f"epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={float(losses['total'].detach()):.5f} "
                    f"samples_per_s={samples / elapsed:.2f}",
                    flush=True,
                )

        validation = evaluate(
            model,
            val_loader,
            device=device,
            bim_stats=bim_stats,
            min_depth=min_depth,
            max_depth=max_depth,
            amp=amp,
        )
        learned_val = validation["bim_early_fusion_dense"]
        row = {
            "epoch": epoch,
            "train_total_loss": totals["total"] / samples,
            "train_log_depth_loss": totals["log_depth"] / samples,
            "train_gradient_loss": totals["gradient"] / samples,
            "val_abs_rel": learned_val["abs_rel"],
            "val_rmse": learned_val["rmse"],
            "val_mae": learned_val["mae"],
            "val_delta1": learned_val["delta1"],
            "val_delta2": learned_val["delta2"],
            "val_rmse_log": learned_val["rmse_log"],
            "encoder_lr": optimizer.param_groups[0]["lr"],
            "decoder_lr": optimizer.param_groups[1]["lr"],
            "bim_condition_lr": optimizer.param_groups[2]["lr"],
            "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped_steps,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        improved = float(learned_val["abs_rel"]) < best_abs_rel
        if improved:
            best_abs_rel = float(learned_val["abs_rel"])
            best_epoch = epoch
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_epoch=best_epoch,
            best_abs_rel=best_abs_rel,
            history=history,
            optimizer_steps=optimizer_steps,
            skipped_steps=skipped_steps,
            bim_stats=bim_stats,
            config=materialized_config,
        )
        atomic_torch_save(output_dir / "latest.pt", payload)
        if improved:
            atomic_torch_save(output_dir / "best.pt", payload)
        write_history(output_dir / "training_history.csv", history)
        shutil.copy2(output_dir / "training_history.csv", results_dir / "training_history.csv")
        print(
            f"epoch={epoch} val_abs_rel={float(learned_val['abs_rel']):.6f} "
            f"raw_da3={float(validation['raw_da3_focal_corrected']['abs_rel']):.6f} "
            f"best_epoch={best_epoch} best_abs_rel={best_abs_rel:.6f}",
            flush=True,
        )

    best_state = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model"], strict=True)
    val_summary = evaluate(
        model,
        val_loader,
        device=device,
        bim_stats=bim_stats,
        min_depth=min_depth,
        max_depth=max_depth,
        amp=amp,
    )
    val_summary.update({"selected_checkpoint": "best.pt", "best_epoch": best_epoch})
    atomic_json(output_dir / "val_summary.json", val_summary)
    atomic_json(results_dir / "val_summary.json", val_summary)

    test_summary = None
    if not args.skip_test:
        test_loader = build_loader(
            test_dataset,
            batch_size=int(cfg.train.val_batch_size),
            workers=int(cfg.train.num_workers),
            shuffle=False,
            seed=seed + 2,
        )
        test_summary = evaluate(
            model,
            test_loader,
            device=device,
            bim_stats=bim_stats,
            min_depth=min_depth,
            max_depth=max_depth,
            amp=amp,
        )
        test_summary.update({"selected_checkpoint": "best.pt", "best_epoch": best_epoch})
        atomic_json(output_dir / "test_summary.json", test_summary)
        atomic_json(results_dir / "test_summary.json", test_summary)

    weight_norm = float(model.bim_condition_embed.weight.detach().float().norm().cpu())
    bias_norm = float(model.bim_condition_embed.bias.detach().float().norm().cpu())
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    project_root = Path(cfg.project_root)
    provenance = {
        "schema_version": 1,
        "git": git_receipt(project_root),
        "random_seed": seed,
        "dataset_split": {
            "annotation": str(resolve_project_path(cfg, cfg.data.split_annotation)),
            "annotation_sha256": str(cfg.data.split_annotation_sha256),
            "fingerprint_sha256": str(cfg.data.split_fingerprint_sha256),
            "train_frames": len(BIMDepthDataset(cfg, "train", augment=False)),
            "val_frames": len(BIMDepthDataset(cfg, "val", augment=False)),
            "test_frames": len(test_dataset),
        },
        "dav2": {
            "model_id": model_id,
            "revision": revision,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "encoder": "DINOv2 ViT-B/14",
            "decoder": "official DAv2 Metric Indoor DPT",
            "max_depth_m": 20.0,
        },
        "da3_condition_source": {
            "model": str(cfg.data.da3_model),
            "revision": str(cfg.data.da3_revision),
            "quantity": "cached canonical depth * mean(processed fx,fy)/300",
            "uses_confidence": False,
            "uses_latent_features": False,
        },
        "bim_normalization": bim_stats,
        "input_resolution": [int(cfg.data.target_height), int(cfg.data.target_width)],
        "optimizer": "AdamW",
        "encoder_lr": encoder_lr,
        "decoder_lr": decoder_lr,
        "bim_condition_lr": condition_lr,
        "weight_decay": float(cfg.train.weight_decay),
        "scheduler": "cosine decay",
        "epoch_count": epochs,
        "best_epoch": best_epoch,
        "best_validation_abs_rel": best_abs_rel,
        "final_epoch_validation_abs_rel": float(history[-1]["val_abs_rel"]),
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "bim_condition_projection_weight_norm": weight_norm,
        "bim_condition_projection_bias_norm": bias_norm,
        "initialization_verification": initialization,
        "training_seconds": time.perf_counter() - training_start,
    }
    atomic_json(output_dir / "provenance.json", provenance)
    atomic_json(results_dir / "provenance.json", provenance)
    summary = {
        "best_epoch": best_epoch,
        "best_validation_abs_rel": best_abs_rel,
        "final_epoch_validation_abs_rel": float(history[-1]["val_abs_rel"]),
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "bim_condition_projection_weight_norm": weight_norm,
        "bim_condition_projection_bias_norm": bias_norm,
        "validation": val_summary,
        "test": test_summary,
    }
    atomic_json(output_dir / "training_summary.json", summary)
    atomic_json(results_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
