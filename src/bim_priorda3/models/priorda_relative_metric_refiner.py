"""PriorDA-style metric refinement with a DAv2-relative DINOv2--DPT.

The metric coordinate frame is defined only by valid BIM depth. DA3 metric
depth and BIM depth are transformed into that shared frame and represented as
normalized disparity. The DPT output is one dense normalized-disparity map;
there is no scale head, coarse stage, residual head, or auxiliary prediction.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .bim_early_fusion_dav2 import BIMEarlyFusionDepthAnythingV2


def depth2disparity(depth: torch.Tensor) -> torch.Tensor:
    """Exact PriorDA convention: reciprocal where depth > 0, otherwise zero."""

    disparity = torch.zeros_like(depth)
    positive = depth > 0
    disparity[positive] = depth[positive].reciprocal()
    return disparity


def bim_affine_frame(
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return valid BIM mask, per-frame minimum, and per-frame range."""

    if bim_depth.ndim != 4 or bim_depth.shape[1] != 1 or bim_depth.shape != bim_valid.shape:
        raise ValueError("BIM depth/mask must have matching [B,1,H,W] shapes")
    valid = (bim_valid > 0.5) & torch.isfinite(bim_depth) & (bim_depth > 0)
    support = valid.flatten(1).sum(dim=1)
    if bool((support == 0).any()):
        raise ValueError("PriorDA shared affine normalization requires valid BIM support in every frame")
    minimum = bim_depth.masked_fill(~valid, torch.inf).amin(dim=(-2, -1), keepdim=True)
    maximum = bim_depth.masked_fill(~valid, -torch.inf).amax(dim=(-2, -1), keepdim=True)
    value_range = maximum - minimum
    value_range = torch.where(value_range == 0, torch.ones_like(value_range), value_range)
    if not bool((torch.isfinite(minimum) & torch.isfinite(value_range) & (value_range > 0)).all()):
        raise FloatingPointError("Invalid BIM-only affine normalization frame")
    return valid, minimum, value_range


def build_priorda_relative_condition(
    batch: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [q_DA3, q_BIM, M_BIM] in the one shared BIM affine frame."""

    base = batch["base_depth"].float()
    bim = batch["bim_depth"].float()
    mask = batch["bim_valid"].float()
    if base.ndim != 4 or base.shape[1] != 1 or base.shape != bim.shape:
        raise ValueError("DA3 and BIM depth must have matching [B,1,H,W] shapes")
    if not bool((torch.isfinite(base) & (base > 0)).all()):
        raise ValueError("Metric DA3 must be dense, positive and finite")
    valid, minimum, value_range = bim_affine_frame(bim, mask)
    normalized_base = (base - minimum) / value_range
    normalized_bim = torch.where(valid, (bim - minimum) / value_range, torch.zeros_like(bim))
    q_base = depth2disparity(normalized_base)
    q_bim = depth2disparity(normalized_bim)
    condition = torch.cat((q_base, q_bim, valid.float()), dim=1)
    if not bool(torch.isfinite(condition).all()):
        raise FloatingPointError("Nonfinite PriorDA normalized-disparity condition")
    return condition, minimum, value_range


class PriorDARelativeMetricRefiner(BIMEarlyFusionDepthAnythingV2):
    """DAv2-relative conditioned DPT whose sole output is metric depth."""

    CONDITION_CHANNELS = 3
    ARCHITECTURE = "dav2_relative_priorda_bim_affine_metric_refiner"

    def __init__(self, pretrained_model) -> None:
        super().__init__(pretrained_model)
        if self.depth_estimation_type != "relative":
            raise ValueError("PriorDA metric refiner requires the official DAv2-relative checkpoint")

    def forward(self, rgb, bim_condition=None):
        # Tensor interface is retained for the inherited zero-init audit.
        if not isinstance(rgb, Mapping):
            return super().forward(rgb, bim_condition)
        batch = rgb
        condition, minimum, value_range = build_priorda_relative_condition(batch)
        normalized_disparity = super().forward(batch["rgb"], condition)[:, None].float()
        normalized_depth = depth2disparity(normalized_disparity)
        depth = normalized_depth * value_range + minimum
        if not bool((torch.isfinite(depth) & (depth > 0)).all()):
            raise FloatingPointError("Invalid de-normalized metric depth")
        return {
            "depth": depth,
            "normalized_disparity": normalized_disparity,
            "normalized_depth": normalized_depth,
            "bim_minimum": minimum,
            "bim_range": value_range,
        }
