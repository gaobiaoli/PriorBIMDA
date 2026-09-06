from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .bim_early_fusion_dav2 import BIMEarlyFusionDepthAnythingV2
from .system import BIMPriorDA3

FIXED_TRAIN_LOG_NORMALIZATION = "fixed_full_train_split_no_per_frame_statistics"
PER_FRAME_BIM_MINMAX_NORMALIZATION = "per_frame_bim_minmax_0_1"
PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION = (
    "per_frame_stage1_bim_union_minmax_0_1"
)
DAV2_METRIC_DEPTH_OUTPUT = "metric_depth"
DAV2_RELATIVE_DISPARITY_OUTPUT = "relative_disparity"


def priorbimda_prior_min_range(
    *,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    fallback_depth: torch.Tensor,
    include_fallback_in_range: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-frame metric minimum/range for reversible normalization.

    By default this follows PriorDA's metric anchoring rule: valid metric-prior
    values determine the normalization. Full BIM dropout falls back to the
    frozen Stage-1 prediction. ``include_fallback_in_range`` instead uses the
    union of BIM and dense Stage-1 ranges, which avoids treating a BIM surface
    behind visible furniture as a hard minimum-depth bound.
    """

    if not (bim_depth.shape == bim_valid.shape == fallback_depth.shape):
        raise ValueError("BIM and fallback depth tensors must have identical shapes")
    if bim_depth.ndim != 4 or bim_depth.shape[1] != 1:
        raise ValueError("Prior normalization depths must have shape [B,1,H,W]")

    bim = bim_depth.float()
    fallback = fallback_depth.float()
    support = (bim_valid > 0) & torch.isfinite(bim) & (bim > 0)
    fallback_support = torch.isfinite(fallback) & (fallback > 0)
    if bool((fallback_support.flatten(1).sum(dim=1) < 1).any()):
        raise RuntimeError("Stage-1 normalization fallback contains no valid depth")

    flat_bim = bim.flatten(1)
    flat_support = support.flatten(1)
    flat_fallback = fallback.flatten(1)
    flat_fallback_support = fallback_support.flatten(1)
    prior_minimum = flat_bim.masked_fill(~flat_support, torch.inf).min(dim=1).values
    prior_maximum = flat_bim.masked_fill(~flat_support, -torch.inf).max(dim=1).values
    fallback_minimum = (
        flat_fallback.masked_fill(~flat_fallback_support, torch.inf).min(dim=1).values
    )
    fallback_maximum = (
        flat_fallback.masked_fill(~flat_fallback_support, -torch.inf).max(dim=1).values
    )
    unsupported = flat_support.sum(dim=1) < 1
    prior_minimum = torch.where(unsupported, fallback_minimum, prior_minimum)
    prior_maximum = torch.where(unsupported, fallback_maximum, prior_maximum)
    if include_fallback_in_range:
        prior_minimum = torch.minimum(prior_minimum, fallback_minimum)
        prior_maximum = torch.maximum(prior_maximum, fallback_maximum)

    # A constant prior has no usable range.  Prefer the Stage-1 range before
    # falling back to one metre so normalization always remains reversible.
    prior_range = prior_maximum - prior_minimum
    fallback_range = fallback_maximum - fallback_minimum
    degenerate = ~torch.isfinite(prior_range) | (prior_range <= 1e-6)
    prior_minimum = torch.where(degenerate, fallback_minimum, prior_minimum)
    prior_range = torch.where(degenerate, fallback_range, prior_range)
    prior_range = torch.where(
        torch.isfinite(prior_range) & (prior_range > 1e-6),
        prior_range,
        torch.ones_like(prior_range),
    )
    return prior_minimum.view(-1, 1, 1, 1), prior_range.view(-1, 1, 1, 1)


def fixed_attention_effective_reliability(
    *,
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    scale_output: Mapping[str, torch.Tensor],
    huber_delta: float,
    ratio_min: float,
    ratio_max: float,
) -> dict[str, torch.Tensor]:
    """Recover ``A`` and the final pseudo-Huber ``W_eff`` from Stage 1.

    Stage 1 estimates one center per attention head.  The exported maps use
    the learned head mixture, so at token resolution they are exactly

    ``A = sum_h pi_h A_h`` and
    ``W_eff = sum_h pi_h A_h / sqrt(1 + (r-c_h)^2 / delta^2)``.

    No spatial or per-frame renormalization is applied here.  Consequently
    ``W_eff`` preserves both the learned attention mass and the final robust
    influence used by the scale estimator.
    """

    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    if not 0 < ratio_min < ratio_max:
        raise ValueError("ratio bounds must satisfy 0 < ratio_min < ratio_max")
    if not (base_depth.shape == bim_depth.shape == bim_valid.shape):
        raise ValueError("Stage-1 depth and BIM mask tensors must have identical shapes")
    if base_depth.ndim != 4 or base_depth.shape[1] != 1:
        raise ValueError("Stage-1 depths must have shape [B,1,H,W]")

    attention = scale_output["attention_token_distribution"].float()
    token_valid = scale_output["attention_token_valid"].bool()
    head_mixture = scale_output["head_mixture"].float()
    head_center = scale_output["head_log_scale"].float()
    if attention.ndim != 4:
        raise ValueError("attention_token_distribution must have shape [B,K,h,w]")
    batch_size, heads, token_height, token_width = attention.shape
    if token_valid.shape != (batch_size, 1, token_height, token_width):
        raise ValueError("attention_token_valid shape does not match attention tokens")
    if head_mixture.shape != (batch_size, heads):
        raise ValueError("head_mixture shape does not match attention heads")
    if head_center.shape != (batch_size, heads):
        raise ValueError("head_log_scale shape does not match attention heads")

    base = base_depth.float()
    bim = bim_depth.float()
    ratio = bim / base.clamp_min(1e-6)
    ratio_valid = (
        (bim_valid > 0)
        & torch.isfinite(base)
        & torch.isfinite(bim)
        & torch.isfinite(ratio)
        & (base > 0)
        & (bim > 0)
        & (ratio > float(ratio_min))
        & (ratio < float(ratio_max))
    )
    valid_float = ratio_valid.float()
    token_fraction = functional.adaptive_avg_pool2d(valid_float, (token_height, token_width))
    token_numerator = functional.adaptive_avg_pool2d(
        torch.where(ratio_valid, ratio.clamp_min(1e-6).log(), 0.0),
        (token_height, token_width),
    )
    token_log_ratio = token_numerator / token_fraction.clamp_min(1e-6)
    residual = token_log_ratio - head_center[:, :, None, None]
    robust_weight = torch.rsqrt(1.0 + (residual / float(huber_delta)).square())
    valid_tokens = token_valid & (token_fraction > 0)
    robust_weight = robust_weight * valid_tokens.float()

    mixture = head_mixture[:, :, None, None]
    attention_token_map = (mixture * attention).sum(dim=1, keepdim=True)
    effective_token_map = (mixture * attention * robust_weight).sum(dim=1, keepdim=True)
    output_size = base.shape[-2:]
    attention_map = functional.interpolate(
        attention_token_map,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    effective_map = functional.interpolate(
        effective_token_map,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    # A token may straddle a BIM boundary.  The pixel-domain contract is
    # explicit: invalid BIM locations carry neither attention nor reliability.
    attention_map = attention_map * ratio_valid.float()
    effective_map = effective_map * ratio_valid.float()
    return {
        "attention_map": attention_map,
        "effective_reliability": effective_map,
        "attention_token_map": attention_token_map,
        "effective_reliability_token_map": effective_token_map,
        "final_huber_robust_weight": robust_weight,
        "ratio_valid": ratio_valid,
    }


def build_priorbimda_condition(
    *,
    scaled_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    effective_reliability: torch.Tensor,
    depth_log_mean: float | None = None,
    depth_log_std: float | None = None,
    effective_reliability_mean: float | None = None,
    disagreement_clip: float = 1.5,
    normalization: str = FIXED_TRAIN_LOG_NORMALIZATION,
    prior_minimum: torch.Tensor | None = None,
    prior_range: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build the Stage-2 condition ``[C1,C2,C3]``.

    ``fixed_full_train_split_no_per_frame_statistics`` preserves the original
    immutable train-statistics contract. Both per-frame modes put every input
    channel in ``[0,1]``; their metric depth channel uses the supplied
    minimum/range that is also used to de-normalize the network output.
    """

    if not (scaled_depth.shape == bim_depth.shape == bim_valid.shape):
        raise ValueError("Scaled depth, BIM depth, and BIM validity shapes must match")
    if effective_reliability.shape != scaled_depth.shape:
        raise ValueError("Effective reliability must match scaled depth shape")
    if scaled_depth.ndim != 4 or scaled_depth.shape[1] != 1:
        raise ValueError("Condition depths must have shape [B,1,H,W]")
    if disagreement_clip <= 0 or not math.isfinite(float(disagreement_clip)):
        raise ValueError("disagreement_clip must be positive and finite")
    if normalization not in {
        FIXED_TRAIN_LOG_NORMALIZATION,
        PER_FRAME_BIM_MINMAX_NORMALIZATION,
        PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
    }:
        raise ValueError(f"Unsupported PriorBIMDA condition normalization: {normalization}")

    scaled = scaled_depth.float()
    bim = bim_depth.float()
    if bool((~torch.isfinite(scaled) | (scaled <= 0)).any()):
        raise ValueError("Scaled DA3 depth must be positive and finite everywhere")
    support = (
        (bim_valid > 0)
        & torch.isfinite(bim)
        & (bim > 0)
        & torch.isfinite(effective_reliability)
        & (effective_reliability >= 0)
    )
    zeros = torch.zeros_like(scaled)
    disagreement = (bim.clamp_min(1e-6).log() - scaled.log()).clamp(
        -float(disagreement_clip), float(disagreement_clip)
    ) / float(disagreement_clip)

    if normalization == FIXED_TRAIN_LOG_NORMALIZATION:
        scalars = (depth_log_mean, depth_log_std, effective_reliability_mean)
        if not all(value is not None and math.isfinite(float(value)) for value in scalars):
            raise ValueError("Fixed condition normalization statistics must be finite")
        assert depth_log_mean is not None
        assert depth_log_std is not None
        assert effective_reliability_mean is not None
        if depth_log_std <= 0:
            raise ValueError("depth_log_std must be positive")
        if effective_reliability_mean <= 0:
            raise ValueError("effective_reliability_mean must be positive")
        c1 = (scaled.log() - float(depth_log_mean)) / float(depth_log_std)
        c2 = torch.where(
            support,
            effective_reliability.float() / float(effective_reliability_mean),
            zeros,
        )
        c3 = torch.where(support, disagreement, zeros)
    else:
        if prior_minimum is None or prior_range is None:
            raise ValueError("Per-frame min/max normalization requires prior minimum and range")
        expected_shape = (scaled.shape[0], 1, 1, 1)
        if prior_minimum.shape != expected_shape or prior_range.shape != expected_shape:
            raise ValueError("Prior minimum/range must have shape [B,1,1,1]")
        if bool((~torch.isfinite(prior_minimum) | ~torch.isfinite(prior_range)).any()):
            raise ValueError("Prior minimum/range must be finite")
        if bool((prior_range <= 0).any()):
            raise ValueError("Prior range must be positive")
        c1 = ((scaled - prior_minimum.float()) / prior_range.float()).clamp(0.0, 1.0)
        reliability = torch.where(
            support,
            effective_reliability.float().clamp_min(0),
            zeros,
        )
        reliability_maximum = reliability.flatten(1).max(dim=1).values.view(-1, 1, 1, 1)
        c2 = torch.where(
            support & (reliability_maximum > 0),
            reliability / reliability_maximum.clamp_min(1e-8),
            zeros,
        ).clamp(0.0, 1.0)
        c3 = torch.where(support, 0.5 * (disagreement + 1.0), zeros).clamp(0.0, 1.0)
    condition = torch.cat((c1, c2, c3), dim=1)
    if not bool(torch.isfinite(condition).all()):
        raise RuntimeError("PriorBIMDA Stage-2 condition contains non-finite values")
    return condition


def run_fixed_attention_stage1(
    scale_system: BIMPriorDA3,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run the frozen Stage-1 estimator and materialize its four outputs."""

    head = scale_system.attention_scale
    if head is None:
        raise ValueError("Stage 1 must contain an attention scale estimator")
    if bool(head.iterative_refresh_attention):
        raise ValueError("Stage 1 must use fixed attention across pseudo-Huber updates")
    base = batch["base_depth"]
    with torch.no_grad():
        scale_output = scale_system._estimate_attention_scale(dict(batch), base)
        scale = scale_output["scale"].to(dtype=base.dtype)
        scaled_depth = base * scale
        maps = fixed_attention_effective_reliability(
            base_depth=base,
            bim_depth=batch["bim_depth"],
            bim_valid=batch["bim_valid"],
            scale_output=scale_output,
            huber_delta=float(head.huber_delta),
            ratio_min=math.exp(float(head.log_ratio_min)),
            ratio_max=math.exp(float(head.log_ratio_max)),
        )
    return {
        "scale": scale,
        "log_scale": scale_output["log_scale"],
        "scaled_depth": scaled_depth,
        **maps,
    }


class PriorBIMDAConditionStatistics:
    """Streaming train-split moments for ``log(D_s)`` and ``W_eff``."""

    def __init__(self) -> None:
        self.depth_count = 0
        self.depth_sum = 0.0
        self.depth_squared_sum = 0.0
        self.reliability_count = 0
        self.reliability_sum = 0.0

    def update(
        self,
        scaled_depth: torch.Tensor,
        effective_reliability: torch.Tensor,
        reliability_valid: torch.Tensor,
    ) -> None:
        depth_support = torch.isfinite(scaled_depth) & (scaled_depth > 0)
        log_depth = scaled_depth.float().log()
        depth_values = log_depth[depth_support].double()
        reliability_support = (
            reliability_valid.bool()
            & torch.isfinite(effective_reliability)
            & (effective_reliability >= 0)
        )
        reliability_values = effective_reliability[reliability_support].double()
        self.depth_count += depth_values.numel()
        self.depth_sum += float(depth_values.sum())
        self.depth_squared_sum += float(depth_values.square().sum())
        self.reliability_count += reliability_values.numel()
        self.reliability_sum += float(reliability_values.sum())

    def compute(self) -> dict[str, float | int | str]:
        if self.depth_count < 2:
            raise RuntimeError("Train split has insufficient scaled-depth pixels")
        if self.reliability_count < 1:
            raise RuntimeError("Train split has no valid effective-reliability pixels")
        mean = self.depth_sum / self.depth_count
        variance = max(0.0, self.depth_squared_sum / self.depth_count - mean**2)
        std = math.sqrt(variance)
        reliability_mean = self.reliability_sum / self.reliability_count
        if not all(math.isfinite(value) for value in (mean, std, reliability_mean)):
            raise RuntimeError("Computed PriorBIMDA condition statistics are non-finite")
        if std <= 0 or reliability_mean <= 0:
            raise RuntimeError("Computed PriorBIMDA condition scales must be positive")
        return {
            "depth_log_mean": mean,
            "depth_log_std": std,
            "depth_valid_pixels": self.depth_count,
            "effective_reliability_mean": reliability_mean,
            "effective_reliability_valid_pixels": self.reliability_count,
            "definition": (
                "pixel-micro train-split moments of frozen Stage-1 log(D_s); "
                "train-split mean of unnormalized W_eff on valid BIM/ratio pixels"
            ),
        }


class PriorBIMDATwoStage(nn.Module):
    """Frozen fixed-attention scale estimation plus conditioned DAv2 refinement."""

    def __init__(
        self,
        scale_system: BIMPriorDA3,
        refiner: BIMEarlyFusionDepthAnythingV2,
        *,
        depth_log_mean: float | None = None,
        depth_log_std: float | None = None,
        effective_reliability_mean: float | None = None,
        disagreement_clip: float = 1.5,
        output_max_depth_m: float = 20.0,
        condition_normalization: str = FIXED_TRAIN_LOG_NORMALIZATION,
        refiner_output_domain: str = DAV2_METRIC_DEPTH_OUTPUT,
    ) -> None:
        super().__init__()
        head = scale_system.attention_scale
        if head is None:
            raise ValueError("Stage 1 must contain an attention scale estimator")
        if bool(head.iterative_refresh_attention):
            raise ValueError("Stage 1 must use fixed attention across pseudo-Huber updates")
        if output_max_depth_m <= 0:
            raise ValueError("output_max_depth_m must be positive")
        self.scale_system = scale_system
        self.refiner = refiner
        self.depth_log_mean = None if depth_log_mean is None else float(depth_log_mean)
        self.depth_log_std = None if depth_log_std is None else float(depth_log_std)
        self.effective_reliability_mean = (
            None if effective_reliability_mean is None else float(effective_reliability_mean)
        )
        self.disagreement_clip = float(disagreement_clip)
        self.output_max_depth_m = float(output_max_depth_m)
        self.condition_normalization = str(condition_normalization)
        self.refiner_output_domain = str(refiner_output_domain)
        if self.condition_normalization not in {
            FIXED_TRAIN_LOG_NORMALIZATION,
            PER_FRAME_BIM_MINMAX_NORMALIZATION,
            PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
        }:
            raise ValueError(
                f"Unsupported PriorBIMDA condition normalization: {self.condition_normalization}"
            )
        if self.condition_normalization == FIXED_TRAIN_LOG_NORMALIZATION:
            values = (
                self.depth_log_mean,
                self.depth_log_std,
                self.effective_reliability_mean,
            )
            if not all(value is not None and math.isfinite(value) for value in values):
                raise ValueError("Fixed normalization requires finite train statistics")
            assert self.depth_log_std is not None
            assert self.effective_reliability_mean is not None
            if self.depth_log_std <= 0 or self.effective_reliability_mean <= 0:
                raise ValueError("Fixed normalization scales must be positive")
        if self.refiner_output_domain not in {
            DAV2_METRIC_DEPTH_OUTPUT,
            DAV2_RELATIVE_DISPARITY_OUTPUT,
        }:
            raise ValueError(f"Unsupported DAv2 output domain: {self.refiner_output_domain}")
        if (
            self.refiner_output_domain == DAV2_RELATIVE_DISPARITY_OUTPUT
            and self.condition_normalization
            not in {
                PER_FRAME_BIM_MINMAX_NORMALIZATION,
                PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
            }
        ):
            raise ValueError("Relative DAv2 output requires reversible per-frame normalization")
        checkpoint_domain = getattr(refiner, "depth_estimation_type", None)
        expected_checkpoint_domain = {
            DAV2_METRIC_DEPTH_OUTPUT: "metric",
            DAV2_RELATIVE_DISPARITY_OUTPUT: "relative",
        }[self.refiner_output_domain]
        if checkpoint_domain is not None and checkpoint_domain != expected_checkpoint_domain:
            raise ValueError(
                "DAv2 checkpoint/output-domain mismatch: "
                f"checkpoint={checkpoint_domain}, configured={self.refiner_output_domain}"
            )
        for parameter in self.scale_system.parameters():
            parameter.requires_grad_(False)
        self.scale_system.eval()

    def train(self, mode: bool = True) -> PriorBIMDATwoStage:
        super().train(mode)
        # This also disables Stage-1 token dropout.  Freezing gradients alone
        # would not make the global scale deterministic during Stage-2 train.
        self.scale_system.eval()
        return self

    def stage1(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return run_fixed_attention_stage1(self.scale_system, batch)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        stage1 = self.stage1(batch)
        prior_minimum = None
        prior_range = None
        if self.condition_normalization in {
            PER_FRAME_BIM_MINMAX_NORMALIZATION,
            PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
        }:
            prior_minimum, prior_range = priorbimda_prior_min_range(
                bim_depth=batch["bim_depth"],
                bim_valid=batch["bim_valid"],
                fallback_depth=stage1["scaled_depth"],
                include_fallback_in_range=(
                    self.condition_normalization
                    == PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION
                ),
            )
        condition = build_priorbimda_condition(
            scaled_depth=stage1["scaled_depth"],
            bim_depth=batch["bim_depth"],
            bim_valid=batch["bim_valid"],
            effective_reliability=stage1["effective_reliability"],
            depth_log_mean=self.depth_log_mean,
            depth_log_std=self.depth_log_std,
            effective_reliability_mean=self.effective_reliability_mean,
            disagreement_clip=self.disagreement_clip,
            normalization=self.condition_normalization,
            prior_minimum=prior_minimum,
            prior_range=prior_range,
        )
        prediction = self.refiner(batch["rgb"], condition)
        if prediction.ndim == 3:
            prediction = prediction[:, None]
        if self.condition_normalization in {
            PER_FRAME_BIM_MINMAX_NORMALIZATION,
            PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
        }:
            assert prior_minimum is not None and prior_range is not None
            if self.refiner_output_domain == DAV2_RELATIVE_DISPARITY_OUTPUT:
                normalized_disparity = prediction.float().clamp_min(0.0)
                normalized_depth = torch.where(
                    normalized_disparity > 0,
                    normalized_disparity.clamp_min(1e-6).reciprocal(),
                    torch.zeros_like(normalized_disparity),
                )
                output_description = (
                    "relative disparity inverted to normalized depth and de-normalized"
                )
            else:
                normalized_disparity = None
                normalized_depth = (prediction.float() / self.output_max_depth_m).clamp(
                    0.0, 1.0
                )
                output_description = (
                    "metric depth divided by its checkpoint maximum and de-normalized"
                )
            depth = (normalized_depth * prior_range + prior_minimum).clamp(
                1e-3, self.output_max_depth_m
            )
            condition_semantics = (
                "[BIM-minmax normalized s*D_DA3, per-frame-max normalized W_eff, "
                "0.5*(clipped log(D_BIM/(s*D_DA3))/tau+1)]; output is "
                f"{output_description} by the same BIM minimum/range"
            )
        else:
            normalized_disparity = None
            normalized_depth = None
            depth = prediction.float().clamp(1e-3, self.output_max_depth_m)
            condition_semantics = (
                "[train-normalized log(s*D_DA3), train-mean-normalized W_eff, "
                "clipped log(D_BIM/(s*D_DA3))/tau]"
            )
        return {
            **stage1,
            "depth": depth,
            "condition": condition,
            "condition_scaled_log_depth": condition[:, 0:1],
            "condition_effective_reliability": condition[:, 1:2],
            "condition_bim_disagreement": condition[:, 2:3],
            "condition_semantics": condition_semantics,
            "normalized_disparity": normalized_disparity,
            "normalized_depth": normalized_depth,
            "prior_minimum": prior_minimum,
            "prior_range": prior_range,
        }

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        decoder_lr: float,
        condition_lr: float,
    ) -> list[dict[str, Any]]:
        groups = self.refiner.optimizer_parameter_groups(
            encoder_lr=encoder_lr,
            decoder_lr=decoder_lr,
            condition_lr=condition_lr,
        )
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        actual = {id(parameter) for group in groups for parameter in group["params"]}
        if actual != expected:
            raise RuntimeError("Stage-2 optimizer groups must exclude every Stage-1 parameter")
        return groups
