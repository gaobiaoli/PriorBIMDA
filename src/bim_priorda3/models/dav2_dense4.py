"""Direct metric DINOv2--DPT with a four-channel zero-initialized input."""
from __future__ import annotations

from collections.abc import Mapping
import torch

from .bim_early_fusion_dav2 import BIMEarlyFusionDepthAnythingV2


def build_dense4_condition(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """[log(metric DA3), masked log(metric BIM), masked log(BIM/DA3), hit].

    Natural logs of depth in metres, without per-image normalization or
    clipping. Invalid BIM is replaced BEFORE log to prevent NaN propagation.
    Hit is availability, not correctness/confidence. No GT is accessed.
    """
    base, bim, mask = (batch[k].float() for k in ("base_depth", "bim_depth", "bim_valid"))
    if base.ndim != 4 or base.shape[1] != 1 or base.shape != bim.shape or bim.shape != mask.shape:
        raise ValueError("Dense4 depth/mask must have matching [B,1,H,W] shapes")
    if not bool((torch.isfinite(base) & (base > 0)).all()):
        raise ValueError("Metric DA3 must be dense, positive and finite")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("Nonfinite BIM mask")
    hit = (mask > 0.5) & torch.isfinite(bim) & (bim > 0)
    log_base = base.log()
    log_bim = torch.where(hit, bim, torch.ones_like(bim)).log()
    disagreement = torch.where(hit, log_bim - log_base, torch.zeros_like(log_base))
    return torch.cat((log_base, log_bim, disagreement, hit.float()), dim=1)


class DAv2Dense4(BIMEarlyFusionDepthAnythingV2):
    """Full official metric decoder; no learned/analytic scale or residual head."""

    CONDITION_CHANNELS = 4
    ARCHITECTURE = "dav2_metric_dense4_log_da3_log_bim_disagreement_mask"

    def forward(self, rgb, bim_condition=None):
        # Tensor interface also permits the inherited official-init audit.
        if isinstance(rgb, Mapping):
            batch = rgb
            depth = super().forward(batch["rgb"], build_dense4_condition(batch))
            return {"depth": depth[:, None].float()}
        return super().forward(rgb, bim_condition)


def dense_silog_loss(prediction, target, valid, *, variance_focus=0.85):
    """ZoeDepth/PriorDA-style SILog: 10 sqrt(var(g)+(1-lambda)mean(g)^2).

    g=log(pred+1e-7)-log(GT+1e-7), torch.var correction=1 (official default).
    Computed across valid pixels of the physical microbatch, in float32.
    A 1e-12 floor only guards the sqrt derivative at an exactly perfect fit.
    GT validity is immutable; nonfinite predictions fail rather than disappear.
    """
    if prediction.ndim == 3:
        prediction = prediction[:, None]
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("SILog inputs must have equal shapes")
    support = valid.bool() & torch.isfinite(target) & (target > 0)
    if int(support.sum()) < 2:
        raise ValueError("SILog needs at least two GT-valid pixels")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred, gt = prediction.float()[support], target.float()[support]
        if not bool((torch.isfinite(pred) & (pred >= 0)).all()):
            raise FloatingPointError("Invalid dense prediction on fixed GT support")
        g = (pred + 1e-7).log() - (gt + 1e-7).log()
        variance = g.var(correction=1)
        mean = g.mean()
        loss = 10 * (variance + (1 - float(variance_focus)) * mean.square()).clamp_min(1e-12).sqrt()
    return {"total": loss, "log_error_mean": mean, "log_error_var": variance}
