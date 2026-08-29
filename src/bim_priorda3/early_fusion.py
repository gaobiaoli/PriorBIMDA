from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def fixed_depth_support(
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    if target.shape != valid.shape:
        raise ValueError("Target depth and valid mask shapes must agree")
    return (
        valid.bool()
        & torch.isfinite(target)
        & (target >= float(min_depth))
        & (target <= float(max_depth))
    )


def dense_metric_depth_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    gradient_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Log-depth L1 plus valid-neighbour log-error gradients."""

    if prediction.ndim == 3:
        prediction = prediction[:, None]
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Prediction, target, and valid mask shapes must agree")
    # Keep the numerically sensitive logarithms and finite-value tests out of
    # autocast.  In particular, the derivative of log(depth) can overflow in
    # fp16 for small positive predictions even though the scalar loss itself
    # is finite.  The cast preserves the gradient path back to the model.
    prediction = prediction.float()
    target = target.float()
    support = fixed_depth_support(
        target,
        valid,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    support &= torch.isfinite(prediction) & (prediction > 0)
    if not bool(support.any()):
        raise RuntimeError("Training batch has no valid metric-depth supervision")
    log_error = torch.log(prediction.clamp_min(1e-6)) - torch.log(target.clamp_min(1e-6))
    log_depth = log_error[support].abs().mean()

    horizontal_support = support[..., :, 1:] & support[..., :, :-1]
    vertical_support = support[..., 1:, :] & support[..., :-1, :]
    horizontal_error = (log_error[..., :, 1:] - log_error[..., :, :-1]).abs()
    vertical_error = (log_error[..., 1:, :] - log_error[..., :-1, :]).abs()
    gradient_terms: list[torch.Tensor] = []
    if bool(horizontal_support.any()):
        gradient_terms.append(horizontal_error[horizontal_support].mean())
    if bool(vertical_support.any()):
        gradient_terms.append(vertical_error[vertical_support].mean())
    gradient = (
        torch.stack(gradient_terms).sum()
        if gradient_terms
        else prediction.new_zeros(())
    )
    total = log_depth + float(gradient_weight) * gradient
    return {
        "total": total,
        "log_depth": log_depth,
        "gradient": gradient,
        "valid_pixels": support.sum(),
    }


class DenseDepthMetricAccumulator:
    """Pixel-micro absolute metric-depth metrics on one immutable support."""

    def __init__(self) -> None:
        self.count = 0
        self.abs_rel_sum = 0.0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.squared_log_error_sum = 0.0
        self.delta1_count = 0
        self.delta2_count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        support: torch.Tensor,
    ) -> None:
        if prediction.ndim == 3:
            prediction = prediction[:, None]
        if prediction.shape != target.shape or support.shape != target.shape:
            raise ValueError("Metric tensors must have identical shapes")
        mask = support.bool() & torch.isfinite(prediction) & (prediction > 0)
        pred = prediction[mask].float()
        gt = target[mask].float()
        if not pred.numel():
            return
        difference = pred - gt
        log_difference = torch.log(pred) - torch.log(gt)
        ratio = torch.maximum(pred / gt, gt / pred)
        self.count += pred.numel()
        self.abs_rel_sum += float((difference.abs() / gt).sum())
        self.squared_error_sum += float(difference.square().sum())
        self.absolute_error_sum += float(difference.abs().sum())
        self.squared_log_error_sum += float(log_difference.square().sum())
        self.delta1_count += int((ratio < 1.25).sum())
        self.delta2_count += int((ratio < 1.25**2).sum())

    def compute(self) -> dict[str, float | int]:
        if self.count < 1:
            raise RuntimeError("Cannot compute depth metrics without valid pixels")
        return {
            "abs_rel": self.abs_rel_sum / self.count,
            "rmse": math.sqrt(self.squared_error_sum / self.count),
            "mae": self.absolute_error_sum / self.count,
            "delta1": self.delta1_count / self.count,
            "delta2": self.delta2_count / self.count,
            "rmse_log": math.sqrt(self.squared_log_error_sum / self.count),
            "count": self.count,
        }


def compute_train_bim_log_statistics(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Compute BIM log-depth moments from train records and valid hits only."""

    count = 0
    value_sum = 0.0
    squared_sum = 0.0
    record_count = 0
    for record in records:
        sample_path = Path(str(record["sample"]))
        with np.load(sample_path) as sample:
            bim = sample["bim_depth"].astype(np.float64, copy=False)
            valid = sample["bim_valid"] > 0.5
            support = valid & np.isfinite(bim) & (bim > 1e-3)
            values = np.log(bim[support])
        if values.size:
            count += int(values.size)
            value_sum += float(values.sum(dtype=np.float64))
            squared_sum += float(np.square(values).sum(dtype=np.float64))
        record_count += 1
    if count < 2:
        raise RuntimeError("Train split has insufficient valid BIM pixels for normalization")
    mean = value_sum / count
    variance = max(0.0, squared_sum / count - mean**2)
    std = math.sqrt(variance)
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError(f"Invalid train BIM normalization moments: mean={mean}, std={std}")
    return {
        "mean": mean,
        "std": std,
        "valid_pixels": count,
        "train_records": record_count,
        "definition": "log(max(D_BIM,1e-3)) on train-split bim_valid pixels",
    }
