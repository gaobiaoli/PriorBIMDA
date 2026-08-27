from __future__ import annotations

import hashlib
import json
import os
import random
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from bim_priorda3.baselines import ROBUST_LOG_CAP_SCALE_ESTIMATOR
from bim_priorda3.losses import BIMPriorLoss, absrel_optimal_log_scale
from bim_priorda3.metrics import depth_metrics


def resolved_config_sha256(cfg: dict) -> str:
    payload = json.dumps(
        dict(cfg),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def semantic_config_sha256(cfg: dict) -> str:
    """Hash behavior while ignoring the config file's machine-specific location."""
    payload = {
        key: value for key, value in dict(cfg).items() if key not in {"config_path", "project_root"}
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_source_sha256(project_root: Path) -> str:
    project_root = project_root.resolve()
    paths = sorted((project_root / "src").rglob("*.py"))
    paths.append(project_root / "scripts" / "model" / "train.py")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def uses_robust_scale_estimator(model: torch.nn.Module) -> bool:
    """Return whether the model uses the universal robust scale estimator."""

    unwrapped = getattr(model, "module", model)
    parameters = getattr(unwrapped, "scale_estimator_config", {})
    return isinstance(parameters, dict) and parameters.get("name") == ROBUST_LOG_CAP_SCALE_ESTIMATOR


def _fixed_gt_support(
    target: torch.Tensor,
    gt_valid: torch.Tensor,
) -> torch.Tensor:
    """Resolve one immutable validation support from declared GT validity."""

    if target.shape != gt_valid.shape:
        raise ValueError(
            "Validation GT depth and validity shapes differ: "
            f"{tuple(target.shape)} != {tuple(gt_valid.shape)}"
        )
    if not bool(torch.isfinite(gt_valid).all()):
        raise RuntimeError("Validation gt_valid contains non-finite values")
    support = gt_valid > 0
    invalid_target = support & (~torch.isfinite(target) | (target <= 0))
    if bool(invalid_target.any()):
        raise RuntimeError(
            "Validation GT depth is non-finite or non-positive on declared "
            f"GT support; violations={int(invalid_target.sum())}"
        )
    return support


def _assert_predictions_on_fixed_gt_support(
    predictions: dict[str, torch.Tensor],
    target: torch.Tensor,
    support: torch.Tensor,
) -> int:
    """Fail rather than let individual methods silently shrink support."""

    support_count = int(support.sum())
    for name, prediction in predictions.items():
        if prediction.shape != target.shape:
            raise ValueError(
                f"Validation {name} shape differs from GT: "
                f"{tuple(prediction.shape)} != {tuple(target.shape)}"
            )
        invalid = support & (~torch.isfinite(prediction) | (prediction <= 0))
        if bool(invalid.any()):
            raise RuntimeError(
                f"Validation {name} is non-finite or non-positive on the "
                "fixed GT support; "
                f"violations={int(invalid.sum())}, "
                f"support_count={support_count}"
            )
    return support_count


def _assert_metric_counts(
    metrics: dict[str, dict[str, float]],
    *,
    expected: int,
    support_name: str,
) -> None:
    counts = {name: int(values["count"]) for name, values in metrics.items()}
    mismatched = {name: count for name, count in counts.items() if count != expected}
    if mismatched:
        raise RuntimeError(
            f"Validation metric counts differ from {support_name}={expected}: {mismatched}"
        )


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    region_balanced: bool = False,
    region_balance_exponent: float = 1.0,
    samples_per_epoch: int | None = None,
    generator: torch.Generator | None = None,
    persistent_workers: bool = True,
) -> DataLoader:
    if samples_per_epoch is not None and (
        isinstance(samples_per_epoch, bool)
        or not isinstance(samples_per_epoch, Integral)
        or samples_per_epoch < 1
    ):
        raise ValueError("samples_per_epoch must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= float(region_balance_exponent) <= 1.0:
        raise ValueError("region_balance_exponent must be in [0, 1]")
    sampler = None
    if shuffle and region_balanced:
        if isinstance(dataset, Subset):
            records = [dataset.dataset.records[index] for index in dataset.indices]
        else:
            records = dataset.records
        region_counts: dict[str, int] = {}
        for record in records:
            region = str(record["region"])
            region_counts[region] = region_counts.get(region, 0) + 1
        sample_weights = [
            region_counts[str(record["region"])] ** (-float(region_balance_exponent))
            for record in records
        ]
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=samples_per_epoch or len(sample_weights),
            replacement=True,
            generator=generator,
        )
    elif shuffle and samples_per_epoch is not None:
        sampler = torch.utils.data.RandomSampler(
            dataset,
            replacement=True,
            num_samples=samples_per_epoch,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        generator=generator,
        drop_last=shuffle and (samples_per_epoch or len(dataset)) >= batch_size,
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
    base_predictions, scaled_predictions, anchor_predictions = [], [], []
    live_da3_predictions, live_scale_predictions = [], []
    live_bim_direct_predictions = []
    learned_scale_predictions = []
    proportional_predictions = []
    refined_predictions, targets, masks = [], [], []
    uses_live_da3: bool | None = None
    refined_frame_abs_rel: list[float] = []
    anchor_frame_abs_rel: list[float] = []
    residual_rms: list[float] = []
    iteration_log_scale_predictions: list[torch.Tensor] = []
    iteration_oracle_log_scales: list[torch.Tensor] = []
    iteration_oracle_supported: list[torch.Tensor] = []
    robust_scale_enabled = uses_robust_scale_estimator(model)
    live_direct_output_key = "live_robust_bim_direct" if robust_scale_enabled else "live_bim_direct"
    model_output_max_depth = float(
        getattr(
            model,
            "output_max_depth",
            float(getattr(model, "max_depth", 5.0)) * 2.0,
        )
    )
    for batch in loader:
        batch = move_batch(batch, device)
        # E2E inference skips the CPU OpenCV comparator by default. Validation
        # explicitly opts in because both its loss and acceptance metrics use it.
        batch["request_live_bim_direct"] = True
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(batch)
            loss_output = criterion(output, batch)
            loss = loss_output["total"]
        batch_uses_live_da3 = bool(output.get("uses_live_da3", False))
        if uses_live_da3 is None:
            uses_live_da3 = batch_uses_live_da3
        elif batch_uses_live_da3 != uses_live_da3:
            raise RuntimeError("Model changed uses_live_da3 within one validation pass")
        support = _fixed_gt_support(batch["gt_depth"], batch["gt_valid"])
        support_predictions = {
            "base": batch["base_depth"],
            "scaled": batch["scaled_depth"],
            "anchor": batch["anchor_depth"],
            "refined": output["depth"],
        }
        if bool(getattr(model, "additive_residual_enabled", False)):
            support_predictions["proportional_refined"] = output[
                "proportional_depth"
            ].clamp(
                1e-3,
                model_output_max_depth,
            )
        if "attention_scale" in output:
            support_predictions["learned_scale"] = output["scaled_depth"]
        if batch_uses_live_da3:
            if live_direct_output_key not in output:
                raise KeyError(
                    "E2E validation requires "
                    f"output[{live_direct_output_key!r}] for its configured "
                    "BIM-direct comparator"
                )
            support_predictions.update(
                {
                    "live_da3": output["base_depth"],
                    "live_scale": output["scaled_depth"],
                    live_direct_output_key: output[live_direct_output_key],
                }
            )
        _assert_predictions_on_fixed_gt_support(
            support_predictions,
            batch["gt_depth"],
            support,
        )
        losses.append(float(loss))
        base_predictions.append(batch["base_depth"].cpu())
        scaled_predictions.append(batch["scaled_depth"].cpu())
        anchor_predictions.append(batch["anchor_depth"].cpu())
        if "attention_scale" in output:
            learned_scale_predictions.append(output["scaled_depth"].cpu())
            iteration_prediction = output.get("attention_iteration_log_scales")
            if iteration_prediction is not None:
                if iteration_prediction.ndim != 4 or iteration_prediction.shape[-2:] != (
                    1,
                    1,
                ):
                    raise ValueError(
                        "Validation iterative log scales must have shape [B,T,1,1]"
                    )
                oracle_log_scale, oracle_supported = absrel_optimal_log_scale(
                    output["base_depth"],
                    batch["gt_depth"],
                    batch["gt_valid"],
                    min_support=criterion.attention_scale_oracle_min_support,
                )
                iteration_log_scale_predictions.append(
                    iteration_prediction.detach().flatten(2).mean(dim=-1).cpu()
                )
                iteration_oracle_log_scales.append(
                    oracle_log_scale.detach().flatten(1).mean(dim=-1).cpu()
                )
                iteration_oracle_supported.append(oracle_supported.detach().cpu())
        if batch_uses_live_da3:
            live_da3_predictions.append(output["base_depth"].cpu())
            live_scale_predictions.append(output["scaled_depth"].cpu())
            live_bim_direct_predictions.append(output[live_direct_output_key].cpu())
        refined_predictions.append(output["depth"].cpu())
        if bool(getattr(model, "additive_residual_enabled", False)):
            proportional_predictions.append(
                output["proportional_depth"]
                .clamp(
                    1e-3,
                    model_output_max_depth,
                )
                .cpu()
            )
        targets.append(batch["gt_depth"].cpu())
        masks.append(support.cpu())
        valid = support
        dimensions = tuple(range(1, batch["gt_depth"].ndim))
        counts = valid.sum(dim=dimensions).clamp_min(1)
        refined_error = (
            torch.where(
                valid,
                ((output["depth"] - batch["gt_depth"]).abs() / batch["gt_depth"].clamp_min(1e-3)),
                torch.zeros_like(batch["gt_depth"]),
            ).sum(dim=dimensions)
            / counts
        )
        anchor_error = (
            torch.where(
                valid,
                (
                    (batch["anchor_depth"] - batch["gt_depth"]).abs()
                    / batch["gt_depth"].clamp_min(1e-3)
                ),
                torch.zeros_like(batch["gt_depth"]),
            ).sum(dim=dimensions)
            / counts
        )
        refined_frame_abs_rel.extend(refined_error.float().cpu().tolist())
        anchor_frame_abs_rel.extend(anchor_error.float().cpu().tolist())
        residual_rms.append(float(output["log_residual"].float().square().mean().sqrt()))
    target = torch.cat(targets)
    mask = torch.cat(masks)
    fixed_support_count = int(mask.sum())
    base_tensor = torch.cat(base_predictions)
    scaled_tensor = torch.cat(scaled_predictions)
    anchor_tensor = torch.cat(anchor_predictions)
    refined_tensor = torch.cat(refined_predictions)
    base = depth_metrics(base_tensor, target, mask)
    scaled = depth_metrics(scaled_tensor, target, mask)
    anchor = depth_metrics(anchor_tensor, target, mask)
    refined = depth_metrics(refined_tensor, target, mask)
    _assert_metric_counts(
        {
            "base": base,
            "scaled": scaled,
            "anchor": anchor,
            "refined": refined,
        },
        expected=fixed_support_count,
        support_name="fixed_gt_support_count",
    )
    near_mask = mask.bool() & (target >= 0.2) & (target < 1.0)
    near_support_count = int(near_mask.sum())
    anchor_near = depth_metrics(anchor_tensor, target, near_mask)
    refined_near = depth_metrics(refined_tensor, target, near_mask)
    _assert_metric_counts(
        {"anchor_near": anchor_near, "refined_near": refined_near},
        expected=near_support_count,
        support_name="near_fixed_gt_support_count",
    )
    frame_wins = np.asarray(refined_frame_abs_rel) < np.asarray(anchor_frame_abs_rel)
    metrics = {
        "loss": float(np.mean(losses)),
        **{f"base_{key}": value for key, value in base.items()},
        **{f"scaled_{key}": value for key, value in scaled.items()},
        **{f"anchor_{key}": value for key, value in anchor.items()},
        **{f"refined_{key}": value for key, value in refined.items()},
        "refined_minus_anchor_abs_rel": refined["abs_rel"] - anchor["abs_rel"],
        "refined_frame_abs_rel": float(np.mean(refined_frame_abs_rel)),
        "anchor_frame_abs_rel": float(np.mean(anchor_frame_abs_rel)),
        "frame_win_rate": float(np.mean(frame_wins)),
        "refined_near_abs_rel": refined_near["abs_rel"],
        "refined_near_count": refined_near["count"],
        "anchor_near_abs_rel": anchor_near["abs_rel"],
        "anchor_near_count": anchor_near["count"],
        "residual_rms": float(np.mean(residual_rms)),
        "fixed_gt_support_count": fixed_support_count,
        "near_fixed_gt_support_count": near_support_count,
    }
    if proportional_predictions:
        proportional_tensor = torch.cat(proportional_predictions)
        proportional = depth_metrics(proportional_tensor, target, mask)
        proportional_near = depth_metrics(proportional_tensor, target, near_mask)
        _assert_metric_counts(
            {"proportional_refined": proportional},
            expected=fixed_support_count,
            support_name="fixed_gt_support_count",
        )
        metrics.update(
            {
                **{f"proportional_refined_{key}": value for key, value in proportional.items()},
                "proportional_refined_near_abs_rel": proportional_near["abs_rel"],
                "refined_minus_proportional_abs_rel": (
                    refined["abs_rel"] - proportional["abs_rel"]
                ),
            }
        )
    if learned_scale_predictions:
        learned_scale_tensor = torch.cat(learned_scale_predictions)
        learned_scale = depth_metrics(learned_scale_tensor, target, mask)
        learned_scale_near = depth_metrics(learned_scale_tensor, target, near_mask)
        _assert_metric_counts(
            {"learned_scale": learned_scale},
            expected=fixed_support_count,
            support_name="fixed_gt_support_count",
        )
        _assert_metric_counts(
            {"learned_scale_near": learned_scale_near},
            expected=near_support_count,
            support_name="near_fixed_gt_support_count",
        )
        metrics.update(
            {
                **{f"learned_scale_{key}": value for key, value in learned_scale.items()},
                "learned_scale_near_abs_rel": learned_scale_near["abs_rel"],
                "learned_scale_near_count": learned_scale_near["count"],
                "learned_scale_minus_universal_scale_abs_rel": (
                    learned_scale["abs_rel"] - scaled["abs_rel"]
                ),
            }
        )
    if iteration_log_scale_predictions:
        iteration_prediction = torch.cat(iteration_log_scale_predictions).float()
        oracle_log_scale = torch.cat(iteration_oracle_log_scales).float()
        oracle_supported = torch.cat(iteration_oracle_supported).bool()
        if not bool(oracle_supported.any()):
            raise RuntimeError("Iterative scale validation has no oracle-supported frames")
        prediction = iteration_prediction[oracle_supported]
        oracle = oracle_log_scale[oracle_supported]
        previous_prediction = torch.zeros_like(oracle)
        previous_update: torch.Tensor | None = None
        for iteration_index in range(prediction.shape[1]):
            current_prediction = prediction[:, iteration_index]
            error = current_prediction - oracle
            previous_error = previous_prediction - oracle
            update = current_prediction - previous_prediction
            prefix = f"attention_iteration_{iteration_index + 1}"
            metrics.update(
                {
                    f"{prefix}_log_scale_mae": float(error.abs().mean()),
                    f"{prefix}_log_scale_rmse": float(error.square().mean().sqrt()),
                    f"{prefix}_log_scale_bias": float(error.mean()),
                    f"{prefix}_scale_relative_error": float(
                        (error.exp() - 1.0).abs().mean()
                    ),
                    f"{prefix}_prediction_mean": float(current_prediction.mean()),
                    f"{prefix}_prediction_std": float(
                        current_prediction.std(unbiased=False)
                    ),
                    f"{prefix}_update_abs_mean": float(update.abs().mean()),
                    f"{prefix}_improved_rate": float(
                        (error.abs() < previous_error.abs()).float().mean()
                    ),
                    f"{prefix}_regressed_rate": float(
                        (error.abs() > previous_error.abs()).float().mean()
                    ),
                    f"{prefix}_overshoot_rate": float(
                        ((error * previous_error) < 0).float().mean()
                    ),
                    f"{prefix}_oscillation_rate": (
                        float(((update * previous_update) < 0).float().mean())
                        if previous_update is not None
                        else 0.0
                    ),
                }
            )
            previous_prediction = current_prediction
            previous_update = update
        metrics.update(
            {
                "attention_oracle_supported_frames": int(oracle_supported.sum()),
                "attention_oracle_log_scale_mean": float(oracle.mean()),
                "attention_oracle_log_scale_std": float(oracle.std(unbiased=False)),
                "attention_final_log_scale_mae": metrics[
                    f"attention_iteration_{prediction.shape[1]}_log_scale_mae"
                ],
                "attention_final_scale_relative_error": metrics[
                    f"attention_iteration_{prediction.shape[1]}_scale_relative_error"
                ],
            }
        )
    if robust_scale_enabled:
        metrics.update(
            {
                **{f"robust_global_scale_{key}": value for key, value in scaled.items()},
                **{f"robust_bim_direct_{key}": value for key, value in anchor.items()},
                "robust_bim_direct_near_abs_rel": anchor_near["abs_rel"],
                "robust_bim_direct_near_count": anchor_near["count"],
                "refined_minus_robust_bim_direct_abs_rel": (refined["abs_rel"] - anchor["abs_rel"]),
                "refined_beats_robust_bim_direct_frame_win_rate": float(np.mean(frame_wins)),
            }
        )
    if uses_live_da3:
        live_da3_tensor = torch.cat(live_da3_predictions)
        live_scale_tensor = torch.cat(live_scale_predictions)
        live_da3 = depth_metrics(
            live_da3_tensor,
            target,
            mask,
        )
        live_scale = depth_metrics(
            live_scale_tensor,
            target,
            mask,
        )
        live_bim_direct_tensor = torch.cat(live_bim_direct_predictions)
        live_bim_direct = depth_metrics(
            live_bim_direct_tensor,
            target,
            mask,
        )
        live_bim_direct_near = depth_metrics(
            live_bim_direct_tensor,
            target,
            near_mask,
        )
        _assert_metric_counts(
            {
                "live_da3": live_da3,
                "live_scale": live_scale,
                live_direct_output_key: live_bim_direct,
            },
            expected=fixed_support_count,
            support_name="fixed_gt_support_count",
        )
        _assert_metric_counts(
            {f"{live_direct_output_key}_near": live_bim_direct_near},
            expected=near_support_count,
            support_name="near_fixed_gt_support_count",
        )
        live_direct_prefix = "live_robust_bim_direct" if robust_scale_enabled else "live_bim_direct"
        live_scale_prefix = "live_robust_global_scale" if robust_scale_enabled else "live_scale"
        metrics.update(
            {
                **{f"live_da3_{key}": value for key, value in live_da3.items()},
                **{f"{live_scale_prefix}_{key}": value for key, value in live_scale.items()},
                **{f"{live_direct_prefix}_{key}": value for key, value in live_bim_direct.items()},
                f"{live_direct_prefix}_near_abs_rel": live_bim_direct_near["abs_rel"],
                f"{live_direct_prefix}_near_count": live_bim_direct_near["count"],
                f"refined_minus_{live_direct_prefix}_abs_rel": (
                    refined["abs_rel"] - live_bim_direct["abs_rel"]
                ),
            }
        )
    return metrics


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_metric: float,
    cfg: dict,
    provenance: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "config": dict(cfg),
            "provenance": provenance or {},
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, ensure_ascii=False)
