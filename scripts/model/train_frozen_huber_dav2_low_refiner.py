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
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download
from torch.nn import functional
from torch.utils.data import Subset

from bim_priorda3.config import Config, load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.early_fusion import DenseDepthMetricAccumulator
from bim_priorda3.engine import build_loader
from bim_priorda3.losses import build_depth_supervision_weight
from bim_priorda3.models import FrozenHuberDAv2LowRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen iterative-Huber scale + DAv2-DPT r_low refiner"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
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
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def selected_batch(raw: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    required = (
        "rgb",
        "base_depth",
        "bim_depth",
        "bim_valid",
        "scaled_depth",
        "gt_depth",
        "gt_valid",
        "gt_weight",
        "furniture_mask",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"Frozen-Huber DPT batch lacks fields: {missing}")
    batch = {key: raw[key].to(device=device, non_blocking=True) for key in required}
    shapes = {key: tuple(value.shape[-2:]) for key, value in batch.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"Input modalities are not pixel-aligned: {shapes}")
    return batch


def fixed_all_valid_support(target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    support = valid > 0
    invalid = support & (~torch.isfinite(target) | (target <= 0))
    if bool(invalid.any()):
        raise RuntimeError(f"Invalid official GT depths on support: {int(invalid.sum())}")
    return support


def masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    effective = mask.float()
    if weight is not None:
        effective = effective * weight
    return (value * effective).sum() / effective.sum().clamp_min(1.0)


def masked_per_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    effective = mask.float()
    if weight is not None:
        effective = effective * weight
    dimensions = tuple(range(1, value.ndim))
    sample = (value * effective).sum(dim=dimensions) / effective.sum(
        dim=dimensions
    ).clamp_min(1.0)
    return sample.mean()


def low_refiner_loss(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    cfg: Config,
) -> dict[str, torch.Tensor]:
    prediction = output["depth"].float()
    target = batch["gt_depth"].float()
    valid = fixed_all_valid_support(target, batch["gt_valid"])
    pixel_weight = build_depth_supervision_weight(batch, cfg.loss).float()
    log_target = torch.log(target.clamp_min(1e-6))
    log_prediction = torch.log(prediction.clamp_min(1e-6))
    absolute_log_error = (log_prediction - log_target).abs()
    depth = 0.5 * masked_mean(absolute_log_error, valid, pixel_weight) + 0.5 * (
        masked_per_sample_mean(absolute_log_error, valid, pixel_weight)
    )

    horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    pred_dx = log_prediction[..., :, 1:] - log_prediction[..., :, :-1]
    gt_dx = log_target[..., :, 1:] - log_target[..., :, :-1]
    pred_dy = log_prediction[..., 1:, :] - log_prediction[..., :-1, :]
    gt_dy = log_target[..., 1:, :] - log_target[..., :-1, :]
    gradient = 0.5 * (
        masked_mean((pred_dx - gt_dx).abs(), horizontal_valid)
        + masked_mean((pred_dy - gt_dy).abs(), vertical_valid)
    )

    residual_target = (
        log_target - torch.log(output["scaled_depth"].detach().float().clamp_min(1e-6))
    ).clamp(-0.45, 0.45)
    residual_teacher = masked_per_sample_mean(
        functional.smooth_l1_loss(
            output["low_log_residual"].float(),
            residual_target,
            reduction="none",
            beta=0.02,
        ),
        valid,
        pixel_weight,
    )
    low_native = output["low_log_residual_native"].float()
    low_smoothness = 0.5 * (
        (low_native[..., :, 1:] - low_native[..., :, :-1]).abs().mean()
        + (low_native[..., 1:, :] - low_native[..., :-1, :]).abs().mean()
    )
    total = (
        float(cfg.loss.depth) * depth
        + float(cfg.loss.gradient) * gradient
        + (
            float(cfg.loss.residual_teacher)
            + float(cfg.loss.local_residual_teacher)
        )
        * residual_teacher
        + float(cfg.loss.low_smoothness) * low_smoothness
    )
    return {
        "total": total,
        "depth": depth,
        "gradient": gradient,
        "residual_teacher": residual_teacher,
        "low_smoothness": low_smoothness,
    }


def trainable_state(model: FrozenHuberDAv2LowRefiner) -> dict[str, torch.Tensor]:
    prefixes = (
        "refiner.dav2.backbone.",
        "refiner.dav2.neck.",
        "refiner.bim_condition_embed.",
        "refiner.low_output.",
    )
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }


def load_trainable_state(
    model: FrozenHuberDAv2LowRefiner,
    state: dict[str, torch.Tensor],
) -> None:
    prefixes = (
        "refiner.dav2.backbone.",
        "refiner.dav2.neck.",
        "refiner.bim_condition_embed.",
        "refiner.low_output.",
    )
    expected = {name for name in model.state_dict() if name.startswith(prefixes)}
    if set(state) != expected:
        raise RuntimeError(
            "Trainable checkpoint contract changed: "
            f"missing={sorted(expected - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def evaluate(
    model: FrozenHuberDAv2LowRefiner,
    loader,
    *,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    model.eval()
    refined = DenseDepthMetricAccumulator()
    scale_only = DenseDepthMetricAccumulator()
    raw = DenseDepthMetricAccumulator()
    frames = 0
    seconds = 0.0
    low_abs_sum = 0.0
    low_pixels = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = selected_batch(raw_batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds += time.perf_counter() - started
            support = fixed_all_valid_support(batch["gt_depth"], batch["gt_valid"])
            refined.update(output["depth"], batch["gt_depth"], support)
            scale_only.update(output["scaled_depth"], batch["gt_depth"], support)
            raw.update(batch["base_depth"], batch["gt_depth"], support)
            low = output["low_log_residual"].float()
            low_abs_sum += float(low.abs().sum())
            low_pixels += low.numel()
            frames += batch["rgb"].shape[0]
    refined_metrics = refined.compute()
    scale_metrics = scale_only.compute()
    raw_metrics = raw.compute()
    return {
        "frames": frames,
        "support": "official all positive non-sentinel GT depth",
        "alignment": "none",
        "frozen_huber_dav2_dpt_rlow": refined_metrics,
        "frozen_huber_scale_only": scale_metrics,
        "raw_da3_focal_corrected": raw_metrics,
        "relative_improvement_over_scale_only": (
            (float(scale_metrics["abs_rel"]) - float(refined_metrics["abs_rel"]))
            / float(scale_metrics["abs_rel"])
        ),
        "mean_abs_r_low": low_abs_sum / max(low_pixels, 1),
        "inference_seconds": seconds,
    }


def main() -> None:
    args = parse_args()
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
    results_dir = (
        args.results_dir.expanduser().resolve()
        if args.results_dir
        else resolve_project_path(cfg, cfg.experiment.results_dir)
    )
    if args.resume is None and any(
        (output_dir / name).exists()
        for name in ("best.pt", "latest.pt", "training_history.csv")
    ):
        raise FileExistsError(f"Fresh run refuses existing artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    scale_path = resolve_project_path(cfg, cfg.model.frozen_scale.checkpoint)
    actual_scale_sha = sha256_file(scale_path)
    expected_scale_sha = str(cfg.model.frozen_scale.checkpoint_sha256)
    if actual_scale_sha != expected_scale_sha:
        raise RuntimeError(
            f"Frozen scale checkpoint SHA mismatch: {actual_scale_sha} != {expected_scale_sha}"
        )
    dav2_checkpoint = Path(
        hf_hub_download(
            repo_id=str(cfg.model.dav2.model_id),
            filename="model.safetensors",
            revision=str(cfg.model.dav2.revision),
            local_files_only=bool(cfg.model.dav2.local_files_only),
        )
    ).resolve()
    actual_dav2_sha = sha256_file(dav2_checkpoint)
    expected_dav2_sha = str(cfg.model.dav2.checkpoint_sha256)
    if actual_dav2_sha != expected_dav2_sha:
        raise RuntimeError(
            f"DAv2 checkpoint SHA mismatch: {actual_dav2_sha} != {expected_dav2_sha}"
        )
    scale_checkpoint = torch.load(scale_path, map_location="cpu", weights_only=False)
    model = FrozenHuberDAv2LowRefiner.from_checkpoints(
        cfg,
        scale_checkpoint=scale_checkpoint,
    ).to(device)
    if bool(cfg.train.gradient_checkpointing):
        model.refiner.enable_gradient_checkpointing()

    initialization = {
        "scale_checkpoint": str(scale_path),
        "scale_checkpoint_sha256": actual_scale_sha,
        "scale_checkpoint_epoch": int(scale_checkpoint["epoch"]),
        "scale_checkpoint_best_validation_abs_rel": float(scale_checkpoint["best_metric"]),
        "dav2_checkpoint": str(dav2_checkpoint),
        "dav2_checkpoint_sha256": actual_dav2_sha,
        "scale_all_frozen": all(
            not parameter.requires_grad for parameter in model.scale_system.parameters()
        ),
        "bim_patch_projection_zero": bool(
            torch.count_nonzero(model.refiner.bim_condition_embed.weight).item() == 0
            and torch.count_nonzero(model.refiner.bim_condition_embed.bias).item() == 0
        ),
        "low_output_zero": bool(
            torch.count_nonzero(model.refiner.low_output[-1].weight).item() == 0
            and torch.count_nonzero(model.refiner.low_output[-1].bias).item() == 0
        ),
        "low_feature_source": "dav2.dpt_neck.top_down_fusion[1]",
    }
    initialization["all_pass"] = all(
        initialization[key]
        for key in ("scale_all_frozen", "bim_patch_projection_zero", "low_output_zero")
    )
    atomic_json(output_dir / "initialization_verification.json", initialization)
    if not initialization["all_pass"]:
        raise RuntimeError("Initialization verification failed; training is forbidden")

    train_dataset = BIMDepthDataset(cfg, "train", augment=True)
    val_dataset = BIMDepthDataset(cfg, "val", augment=False)
    test_dataset = BIMDepthDataset(cfg, "test", augment=False)
    if args.max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(args.max_train_samples, len(train_dataset))))
    if args.max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))
    train_loader = build_loader(
        train_dataset,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle=True,
        region_balanced=bool(cfg.train.region_balanced_sampling),
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = build_loader(
        val_dataset,
        batch_size=int(cfg.train.val_batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1),
    )

    # A real-sample functional audit complements the parameter-level checks:
    # before any optimizer step, the zero r_low head must make the composite
    # numerically identical to its frozen learned-scale anchor.
    model.eval()
    with torch.inference_mode():
        audit_batch = selected_batch(next(iter(val_loader)), device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(cfg.train.amp) and device.type == "cuda",
        ):
            audit_output = model(audit_batch)
        initial_difference = (
            audit_output["depth"].float() - audit_output["scaled_depth"].float()
        ).abs()
    initialization["initial_depth_equals_frozen_scale"] = {
        "pass": bool(torch.count_nonzero(initial_difference).item() == 0),
        "max_abs_difference_m": float(initial_difference.max()),
    }
    initialization["observed_native_r_low_grid"] = list(
        audit_output["low_log_residual_native"].shape[-2:]
    )
    initialization["native_grid_matches_config"] = bool(
        initialization["observed_native_r_low_grid"]
        == list(cfg.model.dav2_low_refiner.native_grid_at_504)
    )
    initialization["all_pass"] = bool(
        initialization["all_pass"]
        and initialization["initial_depth_equals_frozen_scale"]["pass"]
        and initialization["native_grid_matches_config"]
    )
    atomic_json(output_dir / "initialization_verification.json", initialization)
    if not initialization["all_pass"]:
        raise RuntimeError("Functional initialization verification failed; training is forbidden")

    parameter_groups = model.refiner.optimizer_parameter_groups(
        encoder_lr=float(cfg.train.encoder_learning_rate),
        decoder_lr=float(cfg.train.decoder_learning_rate),
        condition_lr=float(cfg.train.bim_condition_learning_rate),
        low_head_lr=float(cfg.train.low_head_learning_rate),
    )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(args.epochs or cfg.train.epochs)
    accumulation = int(cfg.train.gradient_accumulation)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, steps_per_epoch * epochs),
    )
    amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp,
        init_scale=float(cfg.train.amp_initial_scale),
    )
    start_epoch = 1
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_abs_rel = float("inf")
    optimizer_steps = 0
    skipped_steps = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_trainable_state(model, state["trainable_model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        history = list(state["history"])
        best_epoch = int(state["best_epoch"])
        best_abs_rel = float(state["best_validation_abs_rel"])
        optimizer_steps = int(state["optimizer_steps"])
        skipped_steps = int(state["skipped_steps"])

    materialized = plain(cfg)
    materialized["runtime"] = {
        "epochs": epochs,
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(materialized, handle, sort_keys=False)
    shutil.copy2(output_dir / "config.yaml", results_dir / "config.yaml")
    training_started = time.perf_counter()
    log_every = int(cfg.train.log_every)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if model.scale_system.training:
            raise RuntimeError("Frozen scale system entered training mode")
        optimizer.zero_grad(set_to_none=True)
        totals = {key: 0.0 for key in ("total", "depth", "gradient", "residual_teacher", "low_smoothness")}
        samples = 0
        accumulation_count = 0
        epoch_started = time.perf_counter()
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = selected_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(batch)
                losses = low_refiner_loss(output, batch, cfg)
                scaled_loss = losses["total"] / accumulation
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
            should_step = accumulation_count == accumulation or batch_index == len(train_loader)
            if should_step:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    skipped_steps += 1
                else:
                    scheduler.step()
                    optimizer_steps += 1
                optimizer.zero_grad(set_to_none=True)
                accumulation_count = 0
            if batch_index == 1 or batch_index % log_every == 0:
                print(
                    f"epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={float(losses['total'].detach()):.6f} "
                    f"samples_per_s={samples / (time.perf_counter() - epoch_started):.2f}",
                    flush=True,
                )

        if any(parameter.grad is not None for parameter in model.scale_system.parameters()):
            raise RuntimeError("Frozen scale system unexpectedly received gradients")
        validation = evaluate(model, val_loader, device=device, amp=amp)
        learned = validation["frozen_huber_dav2_dpt_rlow"]
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / samples for key, value in totals.items()},
            "val_abs_rel": learned["abs_rel"],
            "val_rmse": learned["rmse"],
            "val_mae": learned["mae"],
            "val_delta1": learned["delta1"],
            "val_scale_only_abs_rel": validation["frozen_huber_scale_only"]["abs_rel"],
            "val_raw_da3_abs_rel": validation["raw_da3_focal_corrected"]["abs_rel"],
            "val_mean_abs_r_low": validation["mean_abs_r_low"],
            "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped_steps,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        improved = float(learned["abs_rel"]) < best_abs_rel
        if improved:
            best_abs_rel = float(learned["abs_rel"])
            best_epoch = epoch
        payload = {
            "schema_version": 1,
            "trainable_model": trainable_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_abs_rel": best_abs_rel,
            "history": history,
            "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped_steps,
            "frozen_scale_checkpoint_sha256": actual_scale_sha,
            "official_dav2_checkpoint_sha256": actual_dav2_sha,
            "config": materialized,
        }
        atomic_torch_save(output_dir / "latest.pt", payload)
        if improved:
            # Inference checkpoint excludes Adam moments but contains every
            # trained DINO/DPT/BIM/r_low tensor needed atop the two pinned bases.
            atomic_torch_save(
                output_dir / "best.pt",
                {
                    key: payload[key]
                    for key in (
                        "schema_version",
                        "trainable_model",
                        "epoch",
                        "best_epoch",
                        "best_validation_abs_rel",
                        "frozen_scale_checkpoint_sha256",
                        "official_dav2_checkpoint_sha256",
                        "config",
                    )
                },
            )
        write_history(output_dir / "training_history.csv", history)
        shutil.copy2(output_dir / "training_history.csv", results_dir / "training_history.csv")
        print(
            f"epoch={epoch} val_abs_rel={float(learned['abs_rel']):.6f} "
            f"scale_only={float(validation['frozen_huber_scale_only']['abs_rel']):.6f} "
            f"best_epoch={best_epoch} best_abs_rel={best_abs_rel:.6f}",
            flush=True,
        )

    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    load_trainable_state(model, best["trainable_model"])
    val_summary = evaluate(model, val_loader, device=device, amp=amp)
    val_summary.update({"selected_checkpoint": "best.pt", "best_epoch": best_epoch})
    atomic_json(output_dir / "val_summary.json", val_summary)
    atomic_json(results_dir / "val_summary.json", val_summary)
    test_summary = None
    if not args.skip_test:
        test_loader = build_loader(
            test_dataset,
            batch_size=int(cfg.train.val_batch_size),
            num_workers=int(cfg.train.num_workers),
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 2),
        )
        test_summary = evaluate(model, test_loader, device=device, amp=amp)
        test_summary.update({"selected_checkpoint": "best.pt", "best_epoch": best_epoch})
        atomic_json(output_dir / "test_summary.json", test_summary)
        atomic_json(results_dir / "test_summary.json", test_summary)

    summary = {
        "best_epoch": best_epoch,
        "best_validation_abs_rel": best_abs_rel,
        "final_epoch_validation_abs_rel": float(history[-1]["val_abs_rel"]),
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "training_seconds": time.perf_counter() - training_started,
        "initialization": initialization,
        "validation": val_summary,
        "test": test_summary,
    }
    atomic_json(output_dir / "training_summary.json", summary)
    atomic_json(results_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
