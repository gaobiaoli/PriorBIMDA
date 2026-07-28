from __future__ import annotations

import torch


METRIC_NAMES = ("abs_rel", "rmse", "mae", "delta1", "delta2", "delta3")


def depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    mask = valid.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    mask &= (prediction > 0) & (target > 0)
    pred, gt = prediction[mask].double(), target[mask].double()
    if not pred.numel():
        return {name: float("nan") for name in METRIC_NAMES} | {"count": 0}
    difference = pred - gt
    ratio = torch.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float((difference.abs() / gt).mean()),
        "rmse": float(difference.square().mean().sqrt()),
        "mae": float(difference.abs().mean()),
        "delta1": float((ratio < 1.25).double().mean()),
        "delta2": float((ratio < 1.25**2).double().mean()),
        "delta3": float((ratio < 1.25**3).double().mean()),
        "count": int(pred.numel()),
    }

