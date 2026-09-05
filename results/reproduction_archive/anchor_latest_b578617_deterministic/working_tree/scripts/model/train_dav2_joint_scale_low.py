#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Subset

# Reuse the scale baseline's audited experiment I/O and checkpoint resolver.
from train_bim_early_fusion_scale import (
    atomic_json,
    atomic_torch_save,
    fixed_all_valid_support,
    plain,
    resolve_checkpoint,
    selected_batch,
    write_history,
)

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import (
    BIMDepthDataset,
    apply_bim_condition_dropout,
    apply_da3_global_scale_perturbation,
)
from bim_priorda3.early_fusion import DenseDepthMetricAccumulator
from bim_priorda3.engine import build_loader
from bim_priorda3.losses import absrel_optimal_log_scale, build_depth_supervision_weight
from bim_priorda3.models import (
    BIMEarlyFusionDAv2JointScaleLow,
    build_bim_condition,
    joint_scale_low_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one DAv2 encoder for global scale + native 18/36 r_low"
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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: BIMEarlyFusionDAv2JointScaleLow,
    loader,
    *,
    device: torch.device,
    bim_stats: dict[str, Any],
    amp: bool,
    oracle_min_support: int,
) -> dict[str, Any]:
    model.eval()
    accumulators = {
        "raw_da3_focal_corrected": DenseDepthMetricAccumulator(),
        "joint_global_scale": DenseDepthMetricAccumulator(),
        "joint_scale_low1": DenseDepthMetricAccumulator(),
        "joint_scale_low1_low2": DenseDepthMetricAccumulator(),
    }
    frames = 0
    seconds = 0.0
    scale_abs_error = 0.0
    scale_signed_error = 0.0
    scale_relative_error = 0.0
    scale_frames = 0
    low1_mean_abs = 0.0
    low12_mean_abs = 0.0
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
            started = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(
                    batch["rgb"],
                    condition,
                    batch["base_depth"],
                    bim_depth=batch["bim_depth"],
                    bim_valid=batch["bim_valid"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds += time.perf_counter() - started

            support = fixed_all_valid_support(batch["gt_depth"], batch["gt_valid"])
            low1_depth = output["scaled_depth"] * torch.exp(output["low1_log_residual"].float())
            predictions = {
                "raw_da3_focal_corrected": batch["base_depth"],
                "joint_global_scale": output["scaled_depth"],
                "joint_scale_low1": low1_depth,
                "joint_scale_low1_low2": output["depth"],
            }
            for name, prediction in predictions.items():
                accumulators[name].update(prediction, batch["gt_depth"], support)

            oracle, oracle_supported = absrel_optimal_log_scale(
                batch["base_depth"],
                batch["gt_depth"],
                batch["gt_valid"],
                min_support=oracle_min_support,
            )
            if model.residual_mode == "direct_low18":
                residual = output["low1_log_residual"].float()
                valid = batch["gt_valid"].float()
                predicted = (residual * valid).flatten(1).sum(dim=1) / valid.flatten(1).sum(
                    dim=1
                ).clamp_min(1.0)
            else:
                predicted = output["log_scale"].float().flatten(1).mean(dim=1)
            oracle_vector = oracle.flatten(1).mean(dim=1)
            available = oracle_supported.bool()
            if bool(available.any()):
                error = predicted[available] - oracle_vector[available]
                scale_abs_error += float(error.abs().sum())
                scale_signed_error += float(error.sum())
                scale_relative_error += float((error.exp() - 1.0).abs().sum())
                scale_frames += int(available.sum())
            low1_mean_abs += float(
                output["low1_log_residual_native"].float().mean(dim=(1, 2, 3)).abs().sum()
            )
            combined36 = (
                torch.nn.functional.interpolate(
                    output["low1_log_residual_native"].float(),
                    size=output["low2_log_residual_native"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                + output["low2_log_residual_native"].float()
            )
            low12_mean_abs += float(combined36.mean(dim=(1, 2, 3)).abs().sum())
            frames += batch["rgb"].shape[0]

    metrics = {name: accumulator.compute() for name, accumulator in accumulators.items()}
    final_abs_rel = float(metrics["joint_scale_low1_low2"]["abs_rel"])
    scale_abs_rel = float(metrics["joint_global_scale"]["abs_rel"])
    return {
        "frames": frames,
        "support": "official all positive non-sentinel GT depth",
        "alignment": "none",
        **metrics,
        "scale": {
            "prediction_source": (
                "GT-support mean of direct r18"
                if model.residual_mode == "direct_low18"
                else "global scale head"
            ),
            "oracle_supported_frames": scale_frames,
            "oracle_log_scale_mae": scale_abs_error / max(scale_frames, 1),
            "oracle_log_scale_bias": scale_signed_error / max(scale_frames, 1),
            "oracle_scale_relative_error": scale_relative_error / max(scale_frames, 1),
        },
        "residual": {
            "mean_abs_low1_native_mean": low1_mean_abs / max(frames, 1),
            "mean_abs_combined36_mean": low12_mean_abs / max(frames, 1),
        },
        "relative_improvement_over_scale_only": ((scale_abs_rel - final_abs_rel) / scale_abs_rel),
        "inference_seconds": seconds,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")

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
        (output_dir / name).exists() for name in ("best.pt", "latest.pt", "training_history.csv")
    ):
        raise FileExistsError(f"Fresh run refuses existing artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_dataset: Dataset = BIMDepthDataset(cfg, "train", augment=True)
    val_dataset: Dataset = BIMDepthDataset(cfg, "val", augment=False)
    test_dataset: Dataset = BIMDepthDataset(cfg, "test", augment=False)
    stats = cfg.model.bim_normalization
    bim_stats = {
        "mean": float(stats.mean),
        "std": float(stats.std),
        "valid_pixels": int(stats.valid_pixels),
        "train_records": len(train_dataset),
    }
    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
    if args.max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(args.max_val_samples, len(val_dataset))))

    official_path, model_id, revision = resolve_checkpoint(cfg)
    joint = cfg.model.dav2_joint_scale_low
    disagreement_adapter = joint.get("calibrated_disagreement_adapter", {})
    iterative_geometry = joint.get("iterative_geometry_adapters", {})
    model = BIMEarlyFusionDAv2JointScaleLow.from_pretrained(
        model_id,
        revision=revision,
        local_files_only=bool(cfg.model.dav2.local_files_only),
        regression_hidden_size=int(joint.regression_hidden_size),
        head_dropout_probability=float(joint.head_dropout_probability),
        output_weight_std=float(joint.output_weight_std),
        residual_hidden_channels=int(joint.residual_hidden_channels),
        max_low1_log_residual=float(joint.max_low1_log_residual),
        max_low2_log_residual=float(joint.max_low2_log_residual),
        output_max_depth_m=float(cfg.model.output_max_depth_m),
        residual_mode=str(getattr(joint, "residual_mode", "low18_low36")),
        calibrated_disagreement_adapter_enabled=bool(disagreement_adapter.get("enabled", False)),
        calibrated_disagreement_adapter_hidden_channels=int(
            disagreement_adapter.get("hidden_channels", 32)
        ),
        calibrated_disagreement_adapter_residual_blocks=int(
            disagreement_adapter.get("residual_blocks", 0)
        ),
        calibrated_disagreement_adapter_expansion_channels=(
            int(disagreement_adapter.expansion_channels)
            if disagreement_adapter.get("expansion_channels") is not None
            else None
        ),
        calibrated_disagreement_adapter_injection=str(
            disagreement_adapter.get("injection", "fused_f36")
        ),
        calibrated_disagreement_adapter_include_rgb=bool(
            disagreement_adapter.get("include_rgb", False)
        ),
        iterative_geometry_adapters_enabled=bool(
            iterative_geometry.get("enabled", False)
        ),
        iterative_geometry_adapters_hidden_channels=int(
            iterative_geometry.get("hidden_channels", 32)
        ),
        iterative_geometry_adapters_residual_blocks=int(
            iterative_geometry.get("residual_blocks", 0)
        ),
        iterative_geometry_adapters_expansion_channels=(
            int(iterative_geometry.expansion_channels)
            if iterative_geometry.get("expansion_channels") is not None
            else None
        ),
        iterative_geometry_adapters_weight_sharing=str(
            iterative_geometry.get("weight_sharing", "independent")
        ),
        low1_decoder_hidden_channels=joint.get("low1_decoder_hidden_channels"),
        low2_decoder_hidden_channels=joint.get("low2_decoder_hidden_channels"),
        detached_scale_second_pass_dino_adapter_enabled=bool(
            joint.get("detached_scale_second_pass_dino_adapter", {}).get("enabled", False)
        ),
        detached_scale_second_pass_dino_adapter_hidden_channels=int(
            joint.get("detached_scale_second_pass_dino_adapter", {}).get(
                "hidden_channels", 64
            )
        ),
        detached_scale_second_pass_dino_adapter_scope=str(
            joint.get("detached_scale_second_pass_dino_adapter", {}).get(
                "scope", "all_dino_tokens"
            )
        ),
    ).to(device)
    initialization = model.initialization_audit(official_path)
    if bool(cfg.train.gradient_checkpointing):
        model.enable_gradient_checkpointing()

    train_generator = torch.Generator()
    train_loader = build_loader(
        train_dataset,
        int(cfg.train.batch_size),
        int(cfg.train.num_workers),
        shuffle=True,
        region_balanced=bool(cfg.train.region_balanced_sampling),
        region_balance_exponent=float(cfg.train.region_balance_exponent),
        samples_per_epoch=cfg.train.samples_per_epoch,
        generator=train_generator,
        persistent_workers=False,
    )
    val_loader = build_loader(
        val_dataset,
        int(cfg.train.val_batch_size),
        int(cfg.train.num_workers),
        shuffle=False,
        generator=torch.Generator().manual_seed(seed + 17),
        persistent_workers=False,
    )
    model.eval()
    with torch.inference_mode():
        audit_batch = selected_batch(next(iter(val_loader)), device)
        audit_condition = build_bim_condition(
            audit_batch,
            bim_log_mean=bim_stats["mean"],
            bim_log_std=bim_stats["std"],
        )
        audit_output = model(
            audit_batch["rgb"],
            audit_condition,
            audit_batch["base_depth"],
            bim_depth=audit_batch["bim_depth"],
            bim_valid=audit_batch["bim_valid"],
        )
    initialization["native_feature_shapes"] = audit_output["native_feature_shapes"]
    initialization["native_shapes_match"] = initialization["native_feature_shapes"] == [
        [18, 18],
        [36, 36],
        [72, 72],
    ]
    initialization["active_residual_shape"] = audit_output["active_residual_shape"]
    expected_active_shape = (
        [18, 18]
        if model.residual_mode == "direct_low18"
        else ([72, 72] if model.residual_mode == "low72_only" else [36, 36])
    )
    initialization["active_residual_shape_match"] = (
        initialization["active_residual_shape"] == expected_active_shape
    )
    initialization["initial_residual_exact_zero"] = bool(
        torch.count_nonzero(audit_output["low_log_residual"]).item() == 0
    )
    initialization["all_pass"] = bool(
        initialization["all_pass"]
        and initialization["native_shapes_match"]
        and initialization["active_residual_shape_match"]
        and initialization["initial_residual_exact_zero"]
    )
    atomic_json(output_dir / "initialization_verification.json", initialization)
    if not initialization["all_pass"]:
        raise RuntimeError("Joint-model initialization audit failed")

    optimizer = torch.optim.AdamW(
        model.optimizer_parameter_groups(
            encoder_lr=float(cfg.train.encoder_learning_rate),
            decoder_lr=float(cfg.train.decoder_learning_rate),
            condition_lr=float(cfg.train.bim_condition_learning_rate),
            scale_head_lr=float(cfg.train.scale_head_learning_rate),
            residual_head_lr=float(cfg.train.residual_head_learning_rate),
        ),
        weight_decay=float(cfg.train.weight_decay),
    )
    epochs = int(args.epochs or cfg.train.epochs)
    accumulation = int(cfg.train.gradient_accumulation)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
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
    reset_training_state = bool(getattr(cfg.train, "reset_optimizer_scheduler_on_resume", False))
    resume_state: dict[str, Any] | None = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["model"], strict=True)
        start_epoch = int(resume_state["epoch"]) + 1
        history = list(resume_state["history"])
        best_epoch = int(resume_state["best_epoch"])
        best_abs_rel = float(resume_state["best_validation_abs_rel"])
        optimizer_steps = int(resume_state["optimizer_steps"])
        skipped_steps = int(resume_state["skipped_steps"])
        if not reset_training_state:
            optimizer.load_state_dict(resume_state["optimizer"])
            scaler.load_state_dict(resume_state["scaler"])

    remaining_epochs = epochs - start_epoch + 1
    if remaining_epochs < 1:
        raise ValueError(
            f"No epochs remain: checkpoint starts at {start_epoch}, requested total is {epochs}"
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(
            1,
            steps_per_epoch * (remaining_epochs if reset_training_state else epochs),
        ),
    )
    if resume_state is not None and not reset_training_state:
        scheduler.load_state_dict(resume_state["scheduler"])
    if resume_state is not None and reset_training_state:
        # Preserve the parent validation winner in the independent continuation
        # directory. A later epoch replaces it only after a genuine improvement.
        shutil.copy2(args.resume, output_dir / "best.pt")

    materialized = plain(cfg)
    materialized["runtime"] = {
        "epochs": epochs,
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
        "reset_optimizer_scheduler_on_resume": reset_training_state,
        "continuation_start_epoch": start_epoch,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(materialized, handle, sort_keys=False)
    shutil.copy2(output_dir / "config.yaml", results_dir / "config.yaml")

    loss_cfg = cfg.loss
    oracle_min_support = int(loss_cfg.attention_scale_oracle_min_support)
    equivariance_probability = float(joint.equivariance_probability)
    equivariance_log_range = float(joint.equivariance_log_range)
    perturb_cfg = cfg.train.augment.get("da3_global_scale_perturbation", {})
    perturb_enabled = bool(perturb_cfg.get("enabled", False))
    perturb_probability = float(perturb_cfg.get("probability", 1.0)) if perturb_enabled else 0.0
    perturb_log_range = float(perturb_cfg.get("log_range", 0.0)) if perturb_enabled else 0.0
    condition_dropout_cfg = cfg.train.augment.get("bim_condition_dropout", {})
    condition_dropout_enabled = bool(condition_dropout_cfg.get("enabled", False))
    condition_dropout_probability = (
        float(condition_dropout_cfg.get("probability", 0.15)) if condition_dropout_enabled else 0.0
    )
    # Validate once before training rather than discovering a malformed config
    # after the first expensive model forward.
    apply_da3_global_scale_perturbation(
        {"base_depth": torch.ones((1, 1, 1, 1), device=device)},
        probability=perturb_probability,
        log_range=perturb_log_range,
    )
    apply_bim_condition_dropout(
        torch.ones((1, 3, 1, 1), device=device),
        probability=condition_dropout_probability,
    )
    training_started = time.perf_counter()
    log_every = int(cfg.train.log_every)
    for epoch in range(start_epoch, epochs + 1):
        epoch_seed = (seed + (epoch - 1) * 1_000_003) % 2**32
        seed_everything(epoch_seed)
        train_generator.manual_seed(epoch_seed)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        keys = (
            "total",
            "depth",
            "scale_teacher",
            "low1_teacher",
            "low2_teacher",
            "zero_mean",
            "equivariance",
            "predicted_residual_mean",
        )
        totals = {key: 0.0 for key in keys}
        perturb_applied = 0
        perturb_abs_log_q = 0.0
        perturb_signed_log_q = 0.0
        condition_dropout_applied = 0
        samples = 0
        accumulation_count = 0
        epoch_started = time.perf_counter()
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = selected_batch(raw_batch, device)
            batch, augmentation_log_q, augmentation_applied = apply_da3_global_scale_perturbation(
                batch,
                probability=perturb_probability,
                log_range=perturb_log_range,
            )
            condition = build_bim_condition(
                batch,
                bim_log_mean=bim_stats["mean"],
                bim_log_std=bim_stats["std"],
            )
            condition, condition_dropout_mask = apply_bim_condition_dropout(
                condition,
                probability=condition_dropout_probability,
            )
            oracle_log_scale, oracle_supported = absrel_optimal_log_scale(
                batch["base_depth"],
                batch["gt_depth"],
                batch["gt_valid"],
                min_support=oracle_min_support,
            )
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                adapter_bim_valid = batch["bim_valid"]
                if condition_dropout_enabled:
                    available = (~condition_dropout_mask).view(-1, 1, 1, 1)
                    adapter_bim_valid = adapter_bim_valid * available.to(
                        dtype=adapter_bim_valid.dtype
                    )
                output = model(
                    batch["rgb"],
                    condition,
                    batch["base_depth"],
                    bim_depth=batch["bim_depth"],
                    bim_valid=adapter_bim_valid,
                )
                equivariance_error = None
                if equivariance_probability > 0 and equivariance_log_range > 0:
                    selected = (
                        torch.rand((batch["rgb"].shape[0], 1, 1, 1), device=device)
                        < equivariance_probability
                    )
                    log_factor = torch.empty_like(output["log_scale"]).uniform_(
                        -equivariance_log_range,
                        equivariance_log_range,
                    )
                    log_factor = torch.where(selected, log_factor, torch.zeros_like(log_factor))
                    perturbed_base = batch["base_depth"] * log_factor.exp()
                    perturbed_condition = build_bim_condition(
                        {**batch, "base_depth": perturbed_base},
                        bim_log_mean=bim_stats["mean"],
                        bim_log_std=bim_stats["std"],
                    )
                    if condition_dropout_enabled:
                        perturbed_condition, _ = apply_bim_condition_dropout(
                            perturbed_condition,
                            applied=condition_dropout_mask,
                        )
                    perturbed_log_scale = model.predict_log_scale(
                        batch["rgb"],
                        perturbed_condition,
                    )
                    equivariance_error = perturbed_log_scale + log_factor - output["log_scale"]
                losses = joint_scale_low_loss(
                    output,
                    batch,
                    pixel_weight=build_depth_supervision_weight(batch, loss_cfg),
                    oracle_log_scale=oracle_log_scale,
                    oracle_supported=oracle_supported,
                    depth_weight=float(loss_cfg.depth),
                    scale_teacher_weight=float(loss_cfg.attention_scale_oracle),
                    low1_teacher_weight=float(loss_cfg.low1_residual_teacher),
                    low2_teacher_weight=float(loss_cfg.low2_residual_teacher),
                    zero_mean_weight=float(loss_cfg.residual_zero_mean),
                    teacher_beta=float(loss_cfg.native_teacher_beta),
                    residual_mode=model.residual_mode,
                    equivariance_error=equivariance_error,
                    equivariance_weight=float(loss_cfg.attention_scale_equivariance),
                )
                scaled_loss = losses["total"] / accumulation
            batch_samples = batch["rgb"].shape[0]
            perturb_applied += int(augmentation_applied.sum())
            perturb_abs_log_q += float(augmentation_log_q.abs().sum())
            perturb_signed_log_q += float(augmentation_log_q.sum())
            condition_dropout_applied += int(condition_dropout_mask.sum())
            for key in keys:
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
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.train.gradient_clip_norm)
                )
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
                rate = samples / max(time.perf_counter() - epoch_started, 1e-6)
                print(
                    f"epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={float(losses['total'].detach()):.6f} samples_per_s={rate:.2f}",
                    flush=True,
                )

        validation = evaluate(
            model,
            val_loader,
            device=device,
            bim_stats=bim_stats,
            amp=amp,
            oracle_min_support=oracle_min_support,
        )
        learned = validation["joint_scale_low1_low2"]
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / samples for key, value in totals.items()},
            "train_da3_scale_perturbation_fraction": perturb_applied / samples,
            "train_da3_scale_perturbation_abs_log_q": perturb_abs_log_q / samples,
            "train_da3_scale_perturbation_mean_log_q": perturb_signed_log_q / samples,
            "train_bim_condition_dropout_fraction": condition_dropout_applied / samples,
            "val_abs_rel": learned["abs_rel"],
            "val_rmse": learned["rmse"],
            "val_delta1": learned["delta1"],
            "val_scale_abs_rel": validation["joint_global_scale"]["abs_rel"],
            "val_low1_abs_rel": validation["joint_scale_low1"]["abs_rel"],
            "val_scale_log_mae": validation["scale"]["oracle_log_scale_mae"],
            "val_combined_residual_mean_abs": validation["residual"]["mean_abs_combined36_mean"],
            "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped_steps,
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        improved = float(learned["abs_rel"]) < best_abs_rel
        if improved:
            best_abs_rel = float(learned["abs_rel"])
            best_epoch = epoch
        if model.residual_mode == "direct_low18":
            architecture = "dav2_early_fusion_direct_low18_no_global_scale"
        elif model.residual_mode == "low72_only":
            architecture = "dav2_early_fusion_joint_global_scale_low72"
        elif model.iterative_geometry_adapters_enabled:
            architecture = "dav2_early_fusion_joint_global_scale_iterative_"
            architecture += (
                "shared_geometry_trunk_separate_r18_r36_heads_low18_low36"
                if model.iterative_geometry_adapters_weight_sharing
                == "shared_trunk_separate_heads"
                else "independent_geometry18_geometry36_low18_low36"
            )
        elif model.residual_mode != "low36_only":
            architecture = "dav2_early_fusion_joint_global_scale_laplacian_low18_low36"
        elif model.detached_scale_second_pass_dino_adapter_enabled:
            architecture = (
                "dav2_early_fusion_joint_global_scale_low36_"
                "calibrated_disagreement_adapter_detached_second_pass_dino_"
                f"{model.detached_scale_second_pass_dino_adapter_scope}_adapter"
            )
        elif model.calibrated_disagreement_adapter_enabled:
            architecture = "dav2_early_fusion_joint_global_scale_low36_"
            architecture += (
                "calibrated_disagreement_adapter_projected_p36"
                if model.calibrated_disagreement_adapter_injection == "projected_p36"
                else (
                    "calibrated_disagreement_adapter_rgb6"
                    if model.calibrated_disagreement_adapter_include_rgb
                    else "calibrated_disagreement_adapter"
                )
            )
        else:
            architecture = "dav2_early_fusion_joint_global_scale_low36"
        payload = {
            "schema_version": 1,
            "architecture": architecture,
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
            "config": materialized,
        }
        atomic_torch_save(output_dir / "latest.pt", payload)
        if improved:
            atomic_torch_save(output_dir / "best.pt", payload)
        write_history(output_dir / "training_history.csv", history)
        shutil.copy2(output_dir / "training_history.csv", results_dir / "training_history.csv")
        print(
            f"epoch={epoch} val_final={float(learned['abs_rel']):.6f} "
            f"scale={float(validation['joint_global_scale']['abs_rel']):.6f} "
            f"low1={float(validation['joint_scale_low1']['abs_rel']):.6f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )

    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    val_summary = evaluate(
        model,
        val_loader,
        device=device,
        bim_stats=bim_stats,
        amp=amp,
        oracle_min_support=oracle_min_support,
    )
    val_summary.update({"selected_checkpoint": "best.pt", "best_epoch": best_epoch})
    atomic_json(output_dir / "val_summary.json", val_summary)
    atomic_json(results_dir / "val_summary.json", val_summary)
    test_summary = None
    if not args.skip_test:
        test_loader = build_loader(
            test_dataset,
            int(cfg.train.val_batch_size),
            int(cfg.train.num_workers),
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 19),
            persistent_workers=False,
        )
        test_summary = evaluate(
            model,
            test_loader,
            device=device,
            bim_stats=bim_stats,
            amp=amp,
            oracle_min_support=oracle_min_support,
        )
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
