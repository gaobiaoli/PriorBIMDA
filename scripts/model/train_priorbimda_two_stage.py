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
from collections.abc import Mapping
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
from bim_priorda3.early_fusion import DenseDepthMetricAccumulator, fixed_depth_support
from bim_priorda3.engine import build_loader
from bim_priorda3.losses import build_depth_supervision_weight
from bim_priorda3.models import (
    DAV2_METRIC_DEPTH_OUTPUT,
    DAV2_RELATIVE_DISPARITY_OUTPUT,
    FIXED_TRAIN_LOG_NORMALIZATION,
    PER_FRAME_BIM_MINMAX_NORMALIZATION,
    PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
    BIMEarlyFusionDepthAnythingV2,
    BIMPriorDA3,
    PriorBIMDAConditionStatistics,
    PriorBIMDATwoStage,
    run_fixed_attention_stage1,
)

ARCHITECTURE_PREFIX = "priorbimda_fixed_attention_huber_then_dav2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PriorBIMDA Stage 2 with a frozen validation-best Stage 1"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument(
        "--max-stat-samples",
        type=int,
        help="Smoke-test only; production statistics must use the full train split",
    )
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def selected_batch(raw: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
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
        raise KeyError(f"PriorBIMDA batch lacks fields: {missing}")
    batch = {key: raw[key].to(device=device, non_blocking=True) for key in required}
    shapes = {key: tuple(value.shape[-2:]) for key, value in batch.items()}
    if len(set(shapes.values())) != 1:
        raise RuntimeError(f"PriorBIMDA inputs are not pixel-aligned: {shapes}")
    return batch


def _masked_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    effective = mask.float() * weight
    return (value * effective).sum() / effective.sum().clamp_min(1.0)


def _masked_per_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    effective = mask.float() * weight
    dimensions = tuple(range(1, value.ndim))
    sample = (value * effective).sum(dim=dimensions) / effective.sum(dim=dimensions).clamp_min(1.0)
    return sample.mean()


def epoch_loss_trend_audit(
    losses: list[float],
    *,
    comparison_fraction: float = 0.25,
    minimum_relative_decrease: float = 0.05,
) -> dict[str, Any]:
    """Compare broad first/last windows instead of noisy individual batches."""

    if not 0 < comparison_fraction <= 0.5:
        raise ValueError("Loss-trend comparison fraction must lie in (0, 0.5]")
    if not 0 <= minimum_relative_decrease < 1:
        raise ValueError("Minimum relative loss decrease must lie in [0, 1)")
    finite = [float(value) for value in losses if math.isfinite(float(value))]
    if len(finite) < 4:
        raise RuntimeError("At least four finite batches are required for a loss-trend audit")
    window = max(1, int(len(finite) * comparison_fraction))
    first_mean = sum(finite[:window]) / window
    last_mean = sum(finite[-window:]) / window
    relative_decrease = (first_mean - last_mean) / max(abs(first_mean), 1e-12)
    x_mean = 0.5 * (len(finite) - 1)
    y_mean = sum(finite) / len(finite)
    numerator = sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(finite)
    )
    denominator = sum((index - x_mean) ** 2 for index in range(len(finite)))
    slope_per_batch = numerator / max(denominator, 1e-12)
    passed = bool(
        last_mean < first_mean
        and relative_decrease >= minimum_relative_decrease
        and slope_per_batch < 0
    )
    return {
        "pass": passed,
        "finite_batches": len(finite),
        "comparison_fraction": comparison_fraction,
        "window_batches": window,
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "relative_decrease": relative_decrease,
        "minimum_relative_decrease": minimum_relative_decrease,
        "least_squares_slope_per_batch": slope_per_batch,
    }


def priorbimda_refinement_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    cfg: Config,
) -> dict[str, torch.Tensor]:
    """PriorDA-style robust primary loss plus metric and gradient auxiliaries."""

    prediction = output["depth"].float()
    target = batch["gt_depth"].float()
    valid = fixed_depth_support(
        target,
        batch["gt_valid"],
        min_depth=float(cfg.data.min_depth),
        max_depth=float(cfg.data.max_depth),
    )
    valid &= torch.isfinite(prediction) & (prediction > 0)
    if not bool(valid.any()):
        raise RuntimeError("PriorBIMDA training batch has no valid depth supervision")
    pixel_weight = build_depth_supervision_weight(batch, cfg.loss).float()
    log_error = torch.log(prediction.clamp_min(1e-6)) - torch.log(target.clamp_min(1e-6))
    absolute_log_error = log_error.abs()
    log_depth = 0.5 * _masked_mean(absolute_log_error, valid, pixel_weight) + 0.5 * (
        _masked_per_sample_mean(absolute_log_error, valid, pixel_weight)
    )
    robust_error = functional.smooth_l1_loss(
        log_error,
        torch.zeros_like(log_error),
        beta=float(cfg.loss.robust_log_beta),
        reduction="none",
    )
    robust_log_depth = 0.5 * _masked_mean(robust_error, valid, pixel_weight) + 0.5 * (
        _masked_per_sample_mean(robust_error, valid, pixel_weight)
    )
    horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
    horizontal = log_error[..., :, 1:] - log_error[..., :, :-1]
    vertical = log_error[..., 1:, :] - log_error[..., :-1, :]
    zero = prediction.sum() * 0.0
    gradient_terms = []
    if bool(horizontal_valid.any()):
        gradient_terms.append(horizontal.abs()[horizontal_valid].mean())
    if bool(vertical_valid.any()):
        gradient_terms.append(vertical.abs()[vertical_valid].mean())
    gradient = torch.stack(gradient_terms).mean() if gradient_terms else zero
    silog = zero
    silog_weight = float(cfg.loss.get("silog", 0.0))
    if silog_weight > 0:
        # Match ZoeDepth's SILog formulation, which PriorDA names as its
        # pixel-level objective. Compute it in FP32 because the variance and
        # square root are numerically fragile under autocast.
        silog_error = log_error[valid].float()
        silog_mean = silog_error.mean()
        silog_variance = torch.var(silog_error)
        silog = 10.0 * torch.sqrt(
            silog_variance
            + float(cfg.loss.get("silog_beta", 0.15)) * silog_mean.square()
            + 1e-12
        )
    normalized_disparity = zero
    normalized_disparity_weight = float(cfg.loss.get("normalized_disparity", 0.0))
    if normalized_disparity_weight > 0:
        predicted_disparity = output.get("normalized_disparity")
        prior_minimum = output.get("prior_minimum")
        prior_range = output.get("prior_range")
        if not all(
            isinstance(value, torch.Tensor)
            for value in (predicted_disparity, prior_minimum, prior_range)
        ):
            raise RuntimeError(
                "Normalized-disparity supervision requires relative DAv2 and prior range"
            )
        assert isinstance(predicted_disparity, torch.Tensor)
        assert isinstance(prior_minimum, torch.Tensor)
        assert isinstance(prior_range, torch.Tensor)
        normalized_target_depth = (target - prior_minimum.float()) / prior_range.float()
        disparity_valid = valid & (normalized_target_depth > 0)
        target_disparity = torch.where(
            disparity_valid,
            normalized_target_depth.clamp_min(1e-6).reciprocal(),
            torch.zeros_like(normalized_target_depth),
        ).clamp_max(float(cfg.loss.get("normalized_disparity_max", 1000.0)))
        disparity_error = functional.smooth_l1_loss(
            torch.log1p(predicted_disparity.float().clamp_min(0.0)),
            torch.log1p(target_disparity),
            beta=float(cfg.loss.get("normalized_disparity_beta", 0.02)),
            reduction="none",
        )
        normalized_disparity = 0.5 * _masked_mean(
            disparity_error, disparity_valid, pixel_weight
        ) + 0.5 * _masked_per_sample_mean(
            disparity_error, disparity_valid, pixel_weight
        )
    total = (
        silog_weight * silog
        + normalized_disparity_weight * normalized_disparity
        + float(cfg.loss.robust_log_depth) * robust_log_depth
        + float(cfg.loss.depth) * log_depth
        + float(cfg.loss.gradient) * gradient
    )
    return {
        "total": total,
        "silog": silog,
        "normalized_disparity": normalized_disparity,
        "robust_log_depth": robust_log_depth,
        "depth": log_depth,
        "gradient": gradient,
    }


def trainable_state(model: PriorBIMDATwoStage) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.startswith("refiner.")
    }


def load_trainable_state(
    model: PriorBIMDATwoStage,
    state: Mapping[str, torch.Tensor],
) -> None:
    expected = {name for name in model.state_dict() if name.startswith("refiner.")}
    if set(state) != expected:
        raise RuntimeError(
            "Stage-2 checkpoint contract changed: "
            f"missing={sorted(expected - set(state))[:5]}, "
            f"unexpected={sorted(set(state) - expected)[:5]}"
        )
    current = model.state_dict()
    current.update(state)
    model.load_state_dict(current, strict=True)


def configured_condition_statistics(cfg: Config) -> dict[str, Any] | None:
    configured = cfg.model.priorbimda_condition.get("train_statistics", {})
    names = ("depth_log_mean", "depth_log_std", "effective_reliability_mean")
    present = [configured.get(name) is not None for name in names]
    if any(present) and not all(present):
        raise ValueError(f"Configure all or none of the condition statistics: {names}")
    if not any(present):
        return None
    output = {name: float(configured[name]) for name in names}
    output.update(
        {
            "depth_valid_pixels": int(configured.get("depth_valid_pixels", 0)),
            "effective_reliability_valid_pixels": int(
                configured.get("effective_reliability_valid_pixels", 0)
            ),
            "definition": "configured immutable full-train-split statistics",
        }
    )
    return output


def compute_condition_statistics(
    scale_system: BIMPriorDA3,
    loader,
    *,
    device: torch.device,
    amp: bool,
) -> dict[str, Any]:
    accumulator = PriorBIMDAConditionStatistics()
    scale_system.eval()
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader, start=1):
            batch = selected_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                stage1 = run_fixed_attention_stage1(scale_system, batch)
            accumulator.update(
                stage1["scaled_depth"],
                stage1["effective_reliability"],
                stage1["ratio_valid"],
            )
            if batch_index == 1 or batch_index % 200 == 0:
                print(f"condition_statistics batch={batch_index}/{len(loader)}", flush=True)
    return accumulator.compute()


def evaluate(
    model: PriorBIMDATwoStage,
    loader,
    *,
    device: torch.device,
    amp: bool,
    cfg: Config,
) -> dict[str, Any]:
    model.eval()
    refined = DenseDepthMetricAccumulator()
    scale_only = DenseDepthMetricAccumulator()
    raw = DenseDepthMetricAccumulator()
    frames = 0
    seconds = 0.0
    c2_sum = 0.0
    c2_count = 0
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
            support = fixed_depth_support(
                batch["gt_depth"],
                batch["gt_valid"],
                min_depth=float(cfg.data.min_depth),
                max_depth=float(cfg.data.max_depth),
            )
            refined.update(output["depth"], batch["gt_depth"], support)
            scale_only.update(output["scaled_depth"], batch["gt_depth"], support)
            raw.update(batch["base_depth"], batch["gt_depth"], support)
            c2_valid = output["ratio_valid"]
            c2_sum += float(output["condition_effective_reliability"][c2_valid].sum())
            c2_count += int(c2_valid.sum())
            frames += batch["rgb"].shape[0]
    learned = refined.compute()
    scale_metrics = scale_only.compute()
    return {
        "frames": frames,
        "support": f"fixed GT support {cfg.data.min_depth:g}-{cfg.data.max_depth:g} m",
        "alignment": "none",
        "priorbimda_two_stage": learned,
        "frozen_stage1_scale": scale_metrics,
        "raw_da3_focal_corrected": raw.compute(),
        "relative_improvement_over_stage1": (
            (float(scale_metrics["abs_rel"]) - float(learned["abs_rel"]))
            / float(scale_metrics["abs_rel"])
        ),
        "mean_normalized_effective_reliability": c2_sum / max(c2_count, 1),
        "inference_seconds": seconds,
    }


def main() -> None:
    args = parse_args()
    for name in ("epochs", "max_train_samples", "max_val_samples", "max_stat_samples"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    cfg = load_config(args.config)
    seed = int(cfg.experiment.seed)
    seed_everything(seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    amp = bool(cfg.train.amp) and device.type == "cuda"
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

    frozen_cfg = cfg.model.frozen_scale
    scale_path = resolve_project_path(cfg, frozen_cfg.checkpoint)
    if str(frozen_cfg.selection) != "validation_best" or scale_path.name != "best.pt":
        raise RuntimeError("Stage 2 requires the Stage-1 validation-best checkpoint named best.pt")
    actual_scale_sha = sha256_file(scale_path)
    expected_scale_sha = frozen_cfg.get("checkpoint_sha256")
    if expected_scale_sha and actual_scale_sha != str(expected_scale_sha):
        raise RuntimeError("Frozen Stage-1 checkpoint SHA256 mismatch")
    scale_checkpoint = torch.load(scale_path, map_location="cpu", weights_only=False)
    if not isinstance(scale_checkpoint.get("model"), Mapping):
        raise TypeError("Stage-1 checkpoint lacks model state")
    checkpoint_train = scale_checkpoint.get("config", {}).get("train", {})
    if int(checkpoint_train.get("epochs", 0)) != 3 or not bool(
        checkpoint_train.get("scale_only_experiment", False)
    ):
        raise RuntimeError("Stage 1 must be a three-epoch scale-only experiment")
    if int(scale_checkpoint.get("epoch", 0)) not in range(1, 4):
        raise RuntimeError("Stage-1 best checkpoint must have been selected from epochs 1..3")
    checkpoint_attention = (
        scale_checkpoint.get("config", {}).get("model", {}).get("attention_scale", {})
    )
    if checkpoint_attention.get("estimator") != "pseudo_huber_attention_v1":
        raise RuntimeError("Stage 1 is not a pseudo-Huber attention estimator")
    if bool(checkpoint_attention.get("iterative_refresh_attention", True)):
        raise RuntimeError("Stage 1 checkpoint does not use fixed attention")
    scale_system = BIMPriorDA3(cfg)
    scale_system.load_state_dict(scale_checkpoint["model"], strict=True)
    scale_system.to(device).eval()
    for parameter in scale_system.parameters():
        parameter.requires_grad_(False)

    dav2_cfg = cfg.model.dav2
    refiner_output_domain = str(
        dav2_cfg.get("output_domain", DAV2_METRIC_DEPTH_OUTPUT)
    )
    if refiner_output_domain not in {
        DAV2_METRIC_DEPTH_OUTPUT,
        DAV2_RELATIVE_DISPARITY_OUTPUT,
    }:
        raise ValueError(f"Unsupported configured DAv2 output domain: {refiner_output_domain}")
    architecture = f"{ARCHITECTURE_PREFIX}_{refiner_output_domain}_conditioned"
    dav2_checkpoint = Path(
        hf_hub_download(
            repo_id=str(dav2_cfg.model_id),
            filename="model.safetensors",
            revision=str(dav2_cfg.revision),
            local_files_only=bool(dav2_cfg.local_files_only),
        )
    ).resolve()
    actual_dav2_sha = sha256_file(dav2_checkpoint)
    if actual_dav2_sha != str(dav2_cfg.checkpoint_sha256):
        raise RuntimeError("Official DAv2 checkpoint SHA256 mismatch")

    train_records = BIMDepthDataset(cfg, "train", augment=False)
    val_dataset = BIMDepthDataset(cfg, "val", augment=False)
    test_dataset = BIMDepthDataset(cfg, "test", augment=False)
    resume_state = (
        torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    )
    condition_cfg = cfg.model.priorbimda_condition
    condition_normalization = str(condition_cfg.get("normalization", FIXED_TRAIN_LOG_NORMALIZATION))
    if condition_normalization not in {
        FIXED_TRAIN_LOG_NORMALIZATION,
        PER_FRAME_BIM_MINMAX_NORMALIZATION,
        PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
    }:
        raise ValueError(
            f"Unsupported PriorBIMDA condition normalization: {condition_normalization}"
        )
    if condition_normalization in {
        PER_FRAME_BIM_MINMAX_NORMALIZATION,
        PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
    }:
        anchor_description = (
            "per-frame valid-BIM and frozen Stage-1 union minimum/range"
            if condition_normalization
            == PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION
            else "per-frame valid-BIM metric minimum/range with frozen Stage-1 fallback"
        )
        statistics = {
            "normalization": condition_normalization,
            "definition": (
                f"{anchor_description}; "
                "per-frame maximum normalization of W_eff; no train-split moments"
            ),
            "train_records": len(train_records),
            "full_train_split": True,
        }
        if resume_state is not None and dict(resume_state["condition_statistics"]) != statistics:
            raise RuntimeError("Resume checkpoint uses different condition normalization")
    else:
        statistics = (
            dict(resume_state["condition_statistics"])
            if resume_state is not None
            else configured_condition_statistics(cfg)
        )
    if condition_normalization == FIXED_TRAIN_LOG_NORMALIZATION and statistics is None:
        statistics_dataset = train_records
        if args.max_stat_samples is not None:
            statistics_dataset = Subset(
                statistics_dataset,
                range(min(args.max_stat_samples, len(statistics_dataset))),
            )
        statistics_loader = build_loader(
            statistics_dataset,
            batch_size=int(cfg.train.val_batch_size),
            num_workers=int(cfg.train.num_workers),
            shuffle=False,
            generator=torch.Generator().manual_seed(seed + 11),
        )
        statistics = compute_condition_statistics(
            scale_system,
            statistics_loader,
            device=device,
            amp=amp,
        )
        statistics["train_records"] = len(statistics_dataset)
        statistics["full_train_split"] = args.max_stat_samples is None
    atomic_json(output_dir / "condition_statistics.json", statistics)
    atomic_json(results_dir / "condition_statistics.json", statistics)

    refiner = BIMEarlyFusionDepthAnythingV2.from_pretrained(
        str(dav2_cfg.model_id),
        revision=str(dav2_cfg.revision),
        local_files_only=bool(dav2_cfg.local_files_only),
    ).to(device)
    initialization = refiner.initialization_audit(
        checkpoint_path=dav2_checkpoint,
        device=device,
    )
    model = PriorBIMDATwoStage(
        scale_system,
        refiner,
        depth_log_mean=(
            float(statistics["depth_log_mean"]) if "depth_log_mean" in statistics else None
        ),
        depth_log_std=(
            float(statistics["depth_log_std"]) if "depth_log_std" in statistics else None
        ),
        effective_reliability_mean=(
            float(statistics["effective_reliability_mean"])
            if "effective_reliability_mean" in statistics
            else None
        ),
        disagreement_clip=float(condition_cfg.disagreement_clip),
        output_max_depth_m=float(dav2_cfg.max_depth_m),
        condition_normalization=condition_normalization,
        refiner_output_domain=refiner_output_domain,
    ).to(device)
    initialization.update(
        {
            "stage1_checkpoint": str(scale_path),
            "stage1_checkpoint_sha256": actual_scale_sha,
            "stage1_epoch": int(scale_checkpoint["epoch"]),
            "stage1_validation_metric": float(scale_checkpoint["best_metric"]),
            "stage1_all_frozen": all(
                not parameter.requires_grad for parameter in model.scale_system.parameters()
            ),
            "stage1_eval": not model.scale_system.training,
            "condition_statistics": statistics,
            "condition_normalization": condition_normalization,
            "refiner_output_domain": refiner_output_domain,
        }
    )
    initialization["all_pass"] = bool(
        initialization["all_pass"]
        and initialization["stage1_all_frozen"]
        and initialization["stage1_eval"]
    )
    atomic_json(output_dir / "initialization_verification.json", initialization)
    if not initialization["all_pass"]:
        raise RuntimeError("PriorBIMDA initialization verification failed")
    if bool(cfg.train.gradient_checkpointing):
        model.refiner.enable_gradient_checkpointing()

    train_dataset = BIMDepthDataset(cfg, "train", augment=True)
    if args.max_train_samples is not None:
        train_dataset = Subset(
            train_dataset, range(min(args.max_train_samples, len(train_dataset)))
        )
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
    groups = model.optimizer_parameter_groups(
        encoder_lr=float(cfg.train.encoder_learning_rate),
        decoder_lr=float(cfg.train.decoder_learning_rate),
        condition_lr=float(cfg.train.prior_condition_learning_rate),
    )
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.train.weight_decay))
    epochs = int(args.epochs or cfg.train.epochs)
    accumulation = int(cfg.train.gradient_accumulation)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, steps_per_epoch * epochs)
    )
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
    if resume_state is not None:
        if str(resume_state["frozen_scale_checkpoint_sha256"]) != actual_scale_sha:
            raise RuntimeError("Resume checkpoint refers to a different Stage 1")
        if dict(resume_state["condition_statistics"]) != statistics:
            raise RuntimeError("Resume condition statistics differ")
        load_trainable_state(model, resume_state["trainable_model"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        history = list(resume_state["history"])
        best_epoch = int(resume_state["best_epoch"])
        best_abs_rel = float(resume_state["best_validation_abs_rel"])
        optimizer_steps = int(resume_state["optimizer_steps"])
        skipped_steps = int(resume_state["skipped_steps"])

    materialized = plain(cfg)
    materialized["model"]["priorbimda_condition"]["train_statistics"] = (
        dict(statistics) if condition_normalization == FIXED_TRAIN_LOG_NORMALIZATION else None
    )
    materialized["runtime"] = {
        "epochs": epochs,
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_stat_samples": args.max_stat_samples,
    }
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(materialized, handle, sort_keys=False)
    shutil.copy2(output_dir / "config.yaml", results_dir / "config.yaml")

    training_started = time.perf_counter()
    stopped_early_reason = None
    epoch1_loss_trend = None
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        if model.scale_system.training:
            raise RuntimeError("Frozen Stage 1 entered training mode")
        optimizer.zero_grad(set_to_none=True)
        totals = {
            key: 0.0
            for key in (
                "total",
                "silog",
                "normalized_disparity",
                "robust_log_depth",
                "depth",
                "gradient",
            )
        }
        samples = 0
        accumulation_count = 0
        batch_total_losses: list[float] = []
        window_loss_sum = 0.0
        window_loss_count = 0
        epoch_started = time.perf_counter()
        for batch_index, raw_batch in enumerate(train_loader, start=1):
            batch = selected_batch(raw_batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model(batch)
                losses = priorbimda_refinement_loss(output, batch, cfg)
                scaled_loss = losses["total"] / accumulation
            batch_samples = batch["rgb"].shape[0]
            batch_total_loss = float(losses["total"].detach())
            batch_total_losses.append(batch_total_loss)
            window_loss_sum += batch_total_loss
            window_loss_count += 1
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
            if accumulation_count == accumulation or batch_index == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in groups for parameter in group["params"]],
                    float(cfg.train.gradient_clip_norm),
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
            if batch_index == 1 or batch_index % int(cfg.train.log_every) == 0:
                print(
                    f"epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"loss={batch_total_loss:.6f} "
                    f"window_loss={window_loss_sum / max(window_loss_count, 1):.6f} "
                    f"samples_per_s={samples / (time.perf_counter() - epoch_started):.2f}",
                    flush=True,
                )
                window_loss_sum = 0.0
                window_loss_count = 0
        if any(parameter.grad is not None for parameter in model.scale_system.parameters()):
            raise RuntimeError("Frozen Stage 1 unexpectedly received gradients")
        trend_cfg = cfg.train.get("epoch1_loss_trend_gate", {})
        if epoch == 1 and bool(trend_cfg.get("enabled", False)):
            epoch1_loss_trend = epoch_loss_trend_audit(
                batch_total_losses,
                comparison_fraction=float(trend_cfg.get("comparison_fraction", 0.25)),
                minimum_relative_decrease=float(
                    trend_cfg.get("minimum_relative_decrease", 0.05)
                ),
            )
            atomic_json(output_dir / "epoch1_loss_trend.json", epoch1_loss_trend)
            atomic_json(results_dir / "epoch1_loss_trend.json", epoch1_loss_trend)
            print(f"epoch=1 loss_trend={json.dumps(epoch1_loss_trend)}", flush=True)
        validation = evaluate(model, val_loader, device=device, amp=amp, cfg=cfg)
        learned = validation["priorbimda_two_stage"]
        row = {
            "epoch": epoch,
            **{f"train_{key}": value / samples for key, value in totals.items()},
            "val_abs_rel": learned["abs_rel"],
            "val_rmse": learned["rmse"],
            "val_mae": learned["mae"],
            "val_delta1": learned["delta1"],
            "val_stage1_abs_rel": validation["frozen_stage1_scale"]["abs_rel"],
            "val_raw_da3_abs_rel": validation["raw_da3_focal_corrected"]["abs_rel"],
            "optimizer_steps": optimizer_steps,
            "skipped_steps": skipped_steps,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "loss_trend_pass": (
                epoch1_loss_trend["pass"] if epoch == 1 and epoch1_loss_trend else None
            ),
            "loss_first_window_mean": (
                epoch1_loss_trend["first_window_mean"]
                if epoch == 1 and epoch1_loss_trend
                else None
            ),
            "loss_last_window_mean": (
                epoch1_loss_trend["last_window_mean"]
                if epoch == 1 and epoch1_loss_trend
                else None
            ),
            "loss_relative_decrease": (
                epoch1_loss_trend["relative_decrease"]
                if epoch == 1 and epoch1_loss_trend
                else None
            ),
        }
        history.append(row)
        improved = float(learned["abs_rel"]) < best_abs_rel
        if improved:
            best_abs_rel = float(learned["abs_rel"])
            best_epoch = epoch
        payload = {
            "schema_version": 1,
            "architecture": architecture,
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
            "condition_statistics": statistics,
            "config": materialized,
        }
        atomic_torch_save(output_dir / "latest.pt", payload)
        if improved:
            atomic_torch_save(output_dir / "best.pt", payload)
        write_history(output_dir / "training_history.csv", history)
        shutil.copy2(output_dir / "training_history.csv", results_dir / "training_history.csv")
        print(
            f"epoch={epoch} val_abs_rel={float(learned['abs_rel']):.6f} "
            f"stage1={float(validation['frozen_stage1_scale']['abs_rel']):.6f} "
            f"best_epoch={best_epoch} best_abs_rel={best_abs_rel:.6f}",
            flush=True,
        )
        if epoch == 1 and epoch1_loss_trend and not epoch1_loss_trend["pass"]:
            stopped_early_reason = "epoch1_loss_did_not_decrease"
            print(
                "early_stop=epoch1_loss_did_not_decrease; no epoch 2 will be trained",
                flush=True,
            )
            break

    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    load_trainable_state(model, best["trainable_model"])
    val_summary = evaluate(model, val_loader, device=device, amp=amp, cfg=cfg)
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
        test_summary = evaluate(model, test_loader, device=device, amp=amp, cfg=cfg)
        test_summary.update({"selected_checkpoint": "best.pt", "best_epoch": best_epoch})
        atomic_json(output_dir / "test_summary.json", test_summary)
        atomic_json(results_dir / "test_summary.json", test_summary)
    summary = {
        "architecture": architecture,
        "best_epoch": best_epoch,
        "best_validation_abs_rel": best_abs_rel,
        "final_epoch_validation_abs_rel": float(history[-1]["val_abs_rel"]),
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "completed_epochs": len(history),
        "stopped_early_reason": stopped_early_reason,
        "epoch1_loss_trend": epoch1_loss_trend,
        "condition_statistics": statistics,
        "initialization": initialization,
        "validation": val_summary,
        "test": test_summary,
        "training_seconds": time.perf_counter() - training_started,
    }
    atomic_json(output_dir / "training_summary.json", summary)
    atomic_json(results_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
