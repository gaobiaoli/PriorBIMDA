from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from bim_priorda3.baselines import (
    ROBUST_LOG_CAP_SCALE_ESTIMATOR,
    resolve_scale_estimator_config,
)
from bim_priorda3.config import Config


@torch.no_grad()
def absrel_optimal_log_scale(
    base_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    min_support: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact per-sample scalar minimizing pixel-mean AbsRel.

    For positive base depth ``D`` and target ``G``, minimizing
    ``sum(|s D - G| / G)`` is a weighted-median problem over ``G / D``
    with weights ``D / G``.  The target is training-only supervision: it
    must never be used to set scale during validation or inference.

    Returns ``(log_scale, supported)`` with shapes ``[B, 1, 1, 1]`` and
    ``[B]``. Unsupported samples receive a neutral zero log-scale and are
    excluded from the loss by the caller.
    """

    if base_depth.shape != gt_depth.shape or gt_valid.shape != gt_depth.shape:
        raise ValueError(
            "Oracle-scale tensors must have identical shapes: "
            f"base={tuple(base_depth.shape)}, gt={tuple(gt_depth.shape)}, "
            f"valid={tuple(gt_valid.shape)}"
        )
    if base_depth.ndim != 4 or base_depth.shape[1] != 1:
        raise ValueError("Oracle-scale tensors must have shape [B, 1, H, W]")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")

    # Keep the target deterministic and outside autocast. Float32 exactly
    # matches the prepared model inputs while avoiding a large float64 sort.
    base = base_depth.detach().float()
    target = gt_depth.detach().float()
    valid = (
        (gt_valid.detach() > 0)
        & torch.isfinite(base)
        & torch.isfinite(target)
        & (base > 0)
        & (target > 0)
    )
    support = valid.flatten(1).sum(dim=1)
    supported = support >= min_support

    ratios = torch.where(
        valid,
        target / base.clamp_min(1e-8),
        torch.full_like(base, torch.inf),
    ).flatten(1)
    weights = torch.where(
        valid,
        base / target.clamp_min(1e-8),
        torch.zeros_like(base),
    ).flatten(1)
    sorted_ratios, order = torch.sort(ratios, dim=1)
    sorted_weights = torch.gather(weights, 1, order)
    cumulative = sorted_weights.cumsum(dim=1)
    threshold = 0.5 * sorted_weights.sum(dim=1, keepdim=True)
    median_index = (cumulative >= threshold).to(torch.int8).argmax(dim=1, keepdim=True)
    scale = torch.gather(sorted_ratios, 1, median_index).squeeze(1)
    supported = supported & torch.isfinite(scale) & (scale > 0)
    log_scale = torch.where(supported, scale.clamp_min(1e-8).log(), torch.zeros_like(scale))
    return log_scale.view(-1, 1, 1, 1), supported


def attention_scale_distribution_target_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    oracle_log_scale: torch.Tensor,
    oracle_supported: torch.Tensor,
    *,
    temperature: float,
    ratio_min: float,
    ratio_max: float,
) -> torch.Tensor:
    """Supervise scale attention with GT-derived train-only ratio reliability.

    A BIM/DA3 token is reliable for global scale when its measured log ratio is
    close to the frame's AbsRel-optimal log scale.  The target is normalized on
    the exact post-dropout token support used by the estimator.  It is never
    needed at validation/inference time.
    """

    if temperature <= 0:
        raise ValueError("attention scale target temperature must be positive")
    if not 0 < ratio_min < ratio_max:
        raise ValueError("attention scale target ratio bounds are invalid")
    for key in ("attention_token_distribution", "attention_token_valid"):
        if key not in output:
            raise KeyError(f"Direct attention supervision requires output[{key!r}]")
    attention = output["attention_token_distribution"].float()
    token_valid = output["attention_token_valid"] > 0
    if attention.ndim != 4 or token_valid.ndim != 4 or token_valid.shape[1] != 1:
        raise ValueError("Attention token tensors must be [B,H,h,w] and [B,1,h,w]")
    if attention.shape[0] != token_valid.shape[0] or attention.shape[-2:] != token_valid.shape[-2:]:
        raise ValueError("Attention distribution and token-valid shapes differ")

    base = output["base_depth"].detach().float()
    bim = batch["bim_depth"].detach().float()
    ratio = bim / base.clamp_min(1e-8)
    pixel_valid = (
        (batch["gt_valid"] > 0)
        & (batch["bim_valid"] > 0)
        & torch.isfinite(base)
        & torch.isfinite(bim)
        & torch.isfinite(ratio)
        & (base > 0)
        & (bim > 0)
        & (ratio > ratio_min)
        & (ratio < ratio_max)
    )
    log_ratio = ratio.clamp_min(1e-8).log()
    confidence = torch.exp(
        -(log_ratio - oracle_log_scale.detach().float()).abs() / temperature
    ) * pixel_valid.float()
    token_fraction = functional.adaptive_avg_pool2d(
        pixel_valid.float(),
        attention.shape[-2:],
    )
    token_score = functional.adaptive_avg_pool2d(
        confidence,
        attention.shape[-2:],
    ) / token_fraction.clamp_min(1e-6)
    active = token_valid & (token_fraction > 0)
    token_score = token_score * active.float()
    target_sum = token_score.sum(dim=(-2, -1), keepdim=True)
    target = token_score / target_sum.clamp_min(1e-8)

    target_flat = target.flatten(2)
    attention_flat = attention.flatten(2).clamp_min(1e-12)
    target_log = target_flat.clamp_min(1e-12).log()
    kl_per_head = (
        target_flat * (target_log - attention_flat.log())
    ).sum(dim=-1)
    available = oracle_supported & (target_sum.flatten(1).squeeze(1) > 0)
    if not bool(available.any()):
        return attention.sum() * 0.0
    return kl_per_head[available].mean()


@torch.no_grad()
def build_live_trust_target(
    scaled_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    gt_depth: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build BIM-reliability supervision from the current online DA3 depth."""
    if temperature <= 0:
        raise ValueError("trust temperature must be positive")
    trust_mask = (
        (gt_valid > 0)
        & (bim_valid > 0)
        & torch.isfinite(scaled_depth)
        & torch.isfinite(bim_depth)
        & torch.isfinite(gt_depth)
        & (scaled_depth > 0)
        & (bim_depth > 0)
        & (gt_depth > 0)
    )
    base_error = (
        torch.log(scaled_depth.clamp_min(1e-4)) - torch.log(gt_depth.clamp_min(1e-4))
    ).abs()
    bim_error = (torch.log(bim_depth.clamp_min(1e-4)) - torch.log(gt_depth.clamp_min(1e-4))).abs()
    trust_logit = ((base_error - bim_error - margin) / temperature).clamp(-30.0, 30.0)
    trust_target = torch.sigmoid(trust_logit)
    trust_target = torch.where(
        trust_mask,
        trust_target,
        torch.zeros_like(trust_target),
    )
    return trust_target, trust_mask.float()


def _masked_mean(
    value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    effective = mask.float()
    if weight is not None:
        effective = effective * weight
    return (value * effective).sum() / effective.sum().clamp_min(1.0)


def _masked_per_sample_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    effective = mask.float()
    if weight is not None:
        effective = effective * weight
    dimensions = tuple(range(1, value.ndim))
    numerator = (value * effective).sum(dim=dimensions)
    denominator = effective.sum(dim=dimensions)
    available = denominator > 0
    if not torch.any(available):
        return value.sum() * 0.0
    return (numerator[available] / denominator[available]).mean()


def _per_sample_masked_mean_vector(
    value: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    effective = mask.float()
    if weight is not None:
        effective = effective * weight
    dimensions = tuple(range(1, value.ndim))
    numerator = (value * effective).sum(dim=dimensions)
    denominator = effective.sum(dim=dimensions)
    return numerator / denominator.clamp_min(1.0)


def build_depth_supervision_weight(
    batch: dict[str, torch.Tensor],
    weights: Config,
) -> torch.Tensor:
    """Build train-only emphasis for near, furniture and BIM-conflict pixels.

    Semantic labels and ground-truth disagreement are used only to shape the
    supervised objective.  They are not model inputs and are therefore not
    required at inference time.  Contributions are additive so an overlapping
    near/furniture/conflict pixel cannot receive an accidental multiplicative
    explosion in weight.
    """

    pixel_weight = batch["gt_weight"].clamp_min(0.05)
    emphasis = torch.ones_like(pixel_weight)

    near_boost = float(weights.get("near_range_boost", 0.0))
    if near_boost < 0:
        raise ValueError("loss.near_range_boost must be non-negative")
    emphasis = emphasis + near_boost * (batch["gt_depth"] < 1.0).float()

    furniture_multiplier = float(weights.get("furniture_multiplier", 1.0))
    if furniture_multiplier < 1.0:
        raise ValueError("loss.furniture_multiplier must be at least 1")
    if furniture_multiplier > 1.0:
        if "furniture_mask" not in batch:
            raise KeyError("loss.furniture_multiplier > 1 requires furniture_mask in the dataset")
        emphasis = emphasis + (furniture_multiplier - 1.0) * (batch["furniture_mask"] > 0).float()

    conflict_multiplier = float(weights.get("bim_foreground_conflict_multiplier", 1.0))
    if conflict_multiplier < 1.0:
        raise ValueError("loss.bim_foreground_conflict_multiplier must be at least 1")
    if conflict_multiplier > 1.0:
        tolerance = torch.maximum(
            torch.full_like(batch["bim_depth"], 0.10),
            0.05 * batch["bim_depth"],
        )
        conflict = (
            (batch["gt_valid"] > 0)
            & (batch["bim_valid"] > 0)
            & (batch["gt_depth"] > 0)
            & (batch["bim_depth"] > 0)
            & (batch["gt_depth"] < batch["bim_depth"] - tolerance)
        )
        emphasis = emphasis + (conflict_multiplier - 1.0) * conflict.float()

    return pixel_weight * emphasis


class BIMPriorLoss(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        variant = str(cfg.model.get("variant", ""))
        if variant != "prior_conditioned_v4":
            raise ValueError(
                f"Unsupported model variant {variant!r}; expected 'prior_conditioned_v4'"
            )
        self.weights = cfg.loss
        self.max_total_log_residual = float(cfg.model.get("max_total_log_residual", 0.45))
        self.use_frame_residual = bool(cfg.model.get("use_frame_residual", True))
        self.attention_scale_enabled = bool(
            cfg.model.get("attention_scale", {}).get("enabled", False)
        )
        self.additive_residual_enabled = bool(
            cfg.model.get("additive_residual", {}).get("enabled", False)
        )
        self.max_additive_residual_m = float(
            cfg.model.get("additive_residual", {}).get("max_residual_m", 0.0)
        )
        self.additive_residual_beta_m = float(
            cfg.loss.get("additive_residual_beta_m", 0.02)
        )
        if self.additive_residual_enabled and self.max_additive_residual_m <= 0:
            raise ValueError("Enabled additive residual requires a positive maximum")
        if self.additive_residual_beta_m <= 0:
            raise ValueError("loss.additive_residual_beta_m must be positive")
        self.attention_min_normalized_entropy = float(
            cfg.model.get("attention_scale", {}).get("min_normalized_entropy", 0.35)
        )
        if not 0.0 <= self.attention_min_normalized_entropy <= 1.0:
            raise ValueError("model.attention_scale.min_normalized_entropy must be in [0, 1]")
        self.attention_scale_oracle_weight = float(cfg.loss.get("attention_scale_oracle", 0.0))
        self.attention_scale_oracle_beta = float(cfg.loss.get("attention_scale_oracle_beta", 0.02))
        self.attention_scale_oracle_min_support = int(
            cfg.loss.get("attention_scale_oracle_min_support", 100)
        )
        self.attention_weight_target_weight = float(
            cfg.loss.get("attention_weight_target", 0.0)
        )
        self.attention_weight_target_temperature = float(
            cfg.loss.get("attention_weight_target_temperature", 0.15)
        )
        self.attention_entropy_decay_epochs = int(
            cfg.loss.get("attention_entropy_decay_epochs", 0)
        )
        attention_cfg = cfg.model.get("attention_scale", {})
        self.attention_ratio_min = float(attention_cfg.get("ratio_min", 0.2))
        self.attention_ratio_max = float(attention_cfg.get("ratio_max", 5.0))
        if self.attention_scale_oracle_weight < 0:
            raise ValueError("loss.attention_scale_oracle must be non-negative")
        if self.attention_scale_oracle_weight > 0 and not self.attention_scale_enabled:
            raise ValueError(
                "loss.attention_scale_oracle requires model.attention_scale.enabled=true"
            )
        if self.attention_scale_oracle_beta <= 0:
            raise ValueError("loss.attention_scale_oracle_beta must be positive")
        if self.attention_scale_oracle_min_support < 1:
            raise ValueError("loss.attention_scale_oracle_min_support must be positive")
        if self.attention_weight_target_weight < 0:
            raise ValueError("loss.attention_weight_target must be non-negative")
        if self.attention_weight_target_weight > 0 and not self.attention_scale_enabled:
            raise ValueError(
                "loss.attention_weight_target requires model.attention_scale.enabled=true"
            )
        if self.attention_weight_target_temperature <= 0:
            raise ValueError("loss.attention_weight_target_temperature must be positive")
        if self.attention_entropy_decay_epochs < 0:
            raise ValueError("loss.attention_entropy_decay_epochs must be non-negative")
        if not 0 < self.attention_ratio_min < self.attention_ratio_max:
            raise ValueError("model.attention_scale ratio bounds are invalid")
        self.warmup_epochs = int(cfg.loss.get("warmup_epochs", 0))
        self.trust_margin = float(cfg.loss.get("trust_margin", 0.005))
        self.trust_temperature = float(cfg.loss.get("trust_temperature", 0.03))
        self.robust_scale_enabled = (
            resolve_scale_estimator_config(cfg.model.get("scale_estimator"))["name"]
            == ROBUST_LOG_CAP_SCALE_ESTIMATOR
        )
        self.current_epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def attention_entropy_factor(self) -> float:
        if self.attention_entropy_decay_epochs == 0:
            return 1.0
        return max(
            0.0,
            1.0 - self.current_epoch / float(self.attention_entropy_decay_epochs),
        )

    def forward(
        self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self._scale_refinement_loss(output, batch)

    def _scale_refinement_loss(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        valid = (batch["gt_valid"] > 0) & (batch["gt_depth"] > 0)
        pixel_weight = build_depth_supervision_weight(batch, self.weights)

        log_target = torch.log(batch["gt_depth"].clamp_min(1e-3))
        log_prediction = torch.log(output["depth"].clamp_min(1e-3))
        uses_live_da3 = bool(output.get("uses_live_da3", False))
        scaled_depth = output["scaled_depth"]
        residual_anchor_depth = output.get(
            "refinement_anchor_depth",
            scaled_depth,
        )
        if residual_anchor_depth.shape != scaled_depth.shape:
            raise ValueError(
                "Loss residual anchor shape differs from scaled depth: "
                f"{tuple(residual_anchor_depth.shape)} != "
                f"{tuple(scaled_depth.shape)}"
            )
        log_residual_anchor = torch.log(residual_anchor_depth.clamp_min(1e-3))
        if uses_live_da3:
            live_direct_key = (
                "live_robust_bim_direct" if self.robust_scale_enabled else "live_bim_direct"
            )
            if live_direct_key not in output:
                raise KeyError(
                    f"E2E loss requires output[{live_direct_key!r}] for the "
                    "configured degradation anchor"
                )
            degradation_anchor = output[live_direct_key]
        else:
            degradation_anchor = batch["anchor_depth"]
        log_anchor = torch.log(degradation_anchor.clamp_min(1e-3))
        log_error = log_prediction - log_target
        prediction_error = log_error.abs()
        anchor_error = (log_anchor - log_target).abs()

        depth_loss = 0.5 * _masked_mean(
            prediction_error,
            valid,
            pixel_weight,
        ) + 0.5 * _masked_per_sample_mean(
            prediction_error,
            valid,
            pixel_weight,
        )
        coarse_error = (torch.log(output["coarse_depth"].clamp_min(1e-3)) - log_target).abs()
        coarse_depth_loss = 0.5 * _masked_mean(
            coarse_error,
            valid,
            pixel_weight,
        ) + 0.5 * _masked_per_sample_mean(
            coarse_error,
            valid,
            pixel_weight,
        )

        horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
        vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
        pred_dx = log_prediction[..., :, 1:] - log_prediction[..., :, :-1]
        gt_dx = log_target[..., :, 1:] - log_target[..., :, :-1]
        pred_dy = log_prediction[..., 1:, :] - log_prediction[..., :-1, :]
        gt_dy = log_target[..., 1:, :] - log_target[..., :-1, :]
        gradient_loss = 0.5 * (
            _masked_mean((pred_dx - gt_dx).abs(), horizontal_valid)
            + _masked_mean((pred_dy - gt_dy).abs(), vertical_valid)
        )

        residual_target = (log_target - log_residual_anchor).clamp(
            -self.max_total_log_residual,
            self.max_total_log_residual,
        )
        residual_teacher = _masked_per_sample_mean(
            functional.smooth_l1_loss(
                output["log_residual"],
                residual_target,
                reduction="none",
                beta=0.02,
            ),
            valid,
            pixel_weight,
        )

        if self.use_frame_residual:
            frame_targets = []
            for sample_index in range(residual_target.shape[0]):
                sample_mask = valid[sample_index]
                values = residual_target[sample_index][sample_mask]
                frame_targets.append(
                    values.median() if values.numel() else residual_target.new_tensor(0.0)
                )
            frame_target = torch.stack(frame_targets).view(-1, 1, 1, 1)
            frame_residual_teacher = functional.smooth_l1_loss(
                output.get("raw_frame_log_residual", output["frame_log_residual"]),
                frame_target,
                beta=0.02,
            )
            routed_frame_target = frame_target.expand_as(residual_target) * output.get(
                "residual_routing_gate",
                torch.ones_like(residual_target),
            )
            local_target = residual_target - routed_frame_target
        else:
            frame_residual_teacher = log_prediction.sum() * 0.0
            local_target = residual_target
        local_prediction = output["low_log_residual"] + output["detail_log_residual"]
        local_residual_teacher = _masked_per_sample_mean(
            functional.smooth_l1_loss(
                local_prediction,
                local_target,
                reduction="none",
                beta=0.02,
            ),
            valid,
            pixel_weight,
        )

        low = output.get("raw_low_log_residual", output["low_log_residual"])
        low_smoothness = 0.5 * (
            (low[..., :, 1:] - low[..., :, :-1]).abs().mean()
            + (low[..., 1:, :] - low[..., :-1, :]).abs().mean()
        )
        detail_regularization = output.get(
            "raw_detail_log_residual",
            output["detail_log_residual"],
        ).abs().mean()

        if self.additive_residual_enabled:
            for key in ("proportional_depth", "additive_metric_residual"):
                if key not in output:
                    raise KeyError(f"Enabled additive residual requires output[{key!r}]")
            proportional_depth = output["proportional_depth"]
            additive_residual = output["additive_metric_residual"]
            additive_target = (batch["gt_depth"] - proportional_depth.detach()).clamp(
                -self.max_additive_residual_m,
                self.max_additive_residual_m,
            )
            additive_residual_teacher = _masked_per_sample_mean(
                functional.smooth_l1_loss(
                    additive_residual,
                    additive_target,
                    reduction="none",
                    beta=self.additive_residual_beta_m,
                ),
                valid,
                pixel_weight,
            )
            additive_regularization = additive_residual.abs().mean()
            valid_float = valid.float()
            proportional_vector = (proportional_depth.detach() * valid_float).flatten(1)
            additive_vector = (additive_residual * valid_float).flatten(1)
            dot = (proportional_vector * additive_vector).sum(dim=1)
            denominator = (
                proportional_vector.square().sum(dim=1).clamp_min(1e-8).sqrt()
                * additive_vector.square().sum(dim=1).clamp_min(1e-8).sqrt()
            )
            additive_scale_orthogonality = (dot / denominator).square().mean()
        else:
            additive_residual_teacher = log_prediction.sum() * 0.0
            additive_regularization = log_prediction.sum() * 0.0
            additive_scale_orthogonality = log_prediction.sum() * 0.0

        if self.attention_scale_enabled:
            if "spatial_log_residual" not in output:
                raise KeyError("Attentive scale training requires spatial_log_residual")
            spatial_mean = _per_sample_masked_mean_vector(
                output["spatial_log_residual"],
                valid,
            )
            spatial_mean_regularization = spatial_mean.square().mean()
            entropy = output["attention_scale_normalized_entropy"]
            attention_entropy_regularization = (
                functional.relu(self.attention_min_normalized_entropy - entropy).square().mean()
            ) * self.attention_entropy_factor()
            equivariance_error = output.get("attention_scale_equivariance_error")
            attention_scale_equivariance = (
                equivariance_error.square().mean()
                if equivariance_error is not None
                else log_prediction.sum() * 0.0
            )
            attention_scale_residual = (
                output["attention_bounded_log_scale_residual"].square().mean()
            )
            oracle_required = (
                self.attention_scale_oracle_weight > 0
                or self.attention_weight_target_weight > 0
            )
            if oracle_required:
                if "attention_log_scale" not in output:
                    raise KeyError(
                        "Oracle scale supervision requires output['attention_log_scale']"
                    )
                oracle_log_scale, oracle_supported = absrel_optimal_log_scale(
                    output["base_depth"],
                    batch["gt_depth"],
                    batch["gt_valid"],
                    min_support=self.attention_scale_oracle_min_support,
                )
            if self.attention_scale_oracle_weight > 0:
                predicted_log_scale = output["attention_log_scale"].float()
                if predicted_log_scale.shape != oracle_log_scale.shape:
                    raise ValueError(
                        "Predicted and oracle log-scale shapes differ: "
                        f"{tuple(predicted_log_scale.shape)} != "
                        f"{tuple(oracle_log_scale.shape)}"
                    )
                oracle_raw = (
                    functional.smooth_l1_loss(
                        predicted_log_scale,
                        oracle_log_scale,
                        reduction="none",
                        beta=self.attention_scale_oracle_beta,
                    )
                    .flatten(1)
                    .mean(dim=1)
                )
                attention_scale_oracle = (
                    oracle_raw[oracle_supported].mean()
                    if bool(oracle_supported.any())
                    else log_prediction.sum() * 0.0
                )
            else:
                attention_scale_oracle = log_prediction.sum() * 0.0
            if self.attention_weight_target_weight > 0:
                attention_weight_target = attention_scale_distribution_target_loss(
                    output,
                    batch,
                    oracle_log_scale,
                    oracle_supported,
                    temperature=self.attention_weight_target_temperature,
                    ratio_min=self.attention_ratio_min,
                    ratio_max=self.attention_ratio_max,
                )
            else:
                attention_weight_target = log_prediction.sum() * 0.0
        else:
            spatial_mean_regularization = log_prediction.sum() * 0.0
            attention_entropy_regularization = log_prediction.sum() * 0.0
            attention_scale_equivariance = log_prediction.sum() * 0.0
            attention_scale_residual = log_prediction.sum() * 0.0
            attention_scale_oracle = log_prediction.sum() * 0.0
            attention_weight_target = log_prediction.sum() * 0.0

        warm = self.current_epoch >= self.warmup_epochs
        zero = log_prediction.sum() * 0.0
        if uses_live_da3 or self.attention_scale_enabled:
            trust_target, trust_mask = build_live_trust_target(
                scaled_depth,
                batch["bim_depth"],
                batch["bim_valid"],
                batch["gt_depth"],
                batch["gt_valid"],
                margin=self.trust_margin,
                temperature=self.trust_temperature,
            )
        else:
            trust_target = batch["trust_target"]
            trust_mask = batch["trust_mask"]
        if warm:
            uncertainty = (
                torch.exp(-output["log_variance"]) * prediction_error + output["log_variance"]
            )
            uncertainty_loss = _masked_mean(uncertainty, valid, pixel_weight)

            trust_raw = functional.binary_cross_entropy_with_logits(
                output["bim_reliability_logits"],
                trust_target,
                reduction="none",
            )
            trust_loss = _masked_mean(trust_raw, trust_mask > 0)
            trust_count = trust_mask.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
            frame_target_trust = (trust_target * trust_mask).sum(
                dim=(-2, -1),
                keepdim=True,
            ) / trust_count
            frame_trust_loss = functional.binary_cross_entropy_with_logits(
                output["frame_trust_logits"],
                frame_target_trust,
            )

            prediction_frame_error = _per_sample_masked_mean_vector(
                prediction_error,
                valid,
                pixel_weight,
            )
            anchor_frame_error = _per_sample_masked_mean_vector(
                anchor_error,
                valid,
                pixel_weight,
            )
            degradation_loss = functional.relu(prediction_frame_error - anchor_frame_error).mean()
        else:
            uncertainty_loss = zero
            trust_loss = zero
            frame_trust_loss = zero
            degradation_loss = zero
        if "bim_adapter_gate_logits" in output:
            gate_logits_pyramid = output.get(
                "bim_adapter_gate_logits_pyramid",
                (output["bim_adapter_gate_logits"],),
            )
            gate_losses = []
            for gate_logits in gate_logits_pyramid:
                pooled_mask = functional.interpolate(
                    trust_mask,
                    size=gate_logits.shape[-2:],
                    mode="area",
                )
                pooled_target = functional.interpolate(
                    trust_target * trust_mask,
                    size=gate_logits.shape[-2:],
                    mode="area",
                ) / pooled_mask.clamp_min(1e-6)
                adapter_gate_raw = functional.binary_cross_entropy_with_logits(
                    gate_logits,
                    pooled_target,
                    reduction="none",
                )
                gate_losses.append(
                    _masked_mean(
                        adapter_gate_raw,
                        pooled_mask > 0,
                    )
                )
            adapter_gate_loss = torch.stack(gate_losses).mean()
        else:
            adapter_gate_loss = zero

        total = (
            float(self.weights.depth) * depth_loss
            + float(self.weights.get("coarse_depth", 0.0)) * coarse_depth_loss
            + float(self.weights.gradient) * gradient_loss
            + float(self.weights.get("residual_teacher", 0.0)) * residual_teacher
            + float(self.weights.get("frame_residual_teacher", 0.0)) * frame_residual_teacher
            + float(self.weights.get("local_residual_teacher", 0.0)) * local_residual_teacher
            + float(self.weights.get("low_smoothness", 0.0)) * low_smoothness
            + float(self.weights.get("detail_regularization", 0.0)) * detail_regularization
            + float(self.weights.get("additive_residual_teacher", 0.0))
            * additive_residual_teacher
            + float(self.weights.get("additive_regularization", 0.0))
            * additive_regularization
            + float(self.weights.get("additive_scale_orthogonality", 0.0))
            * additive_scale_orthogonality
            + float(self.weights.get("trust", 0.0)) * trust_loss
            + float(self.weights.get("frame_trust", 0.0)) * frame_trust_loss
            + float(self.weights.get("uncertainty", 0.0)) * uncertainty_loss
            + float(self.weights.get("degradation", 0.0)) * degradation_loss
            + float(self.weights.get("adapter_gate", 0.0)) * adapter_gate_loss
            + float(self.weights.get("spatial_mean", 0.0)) * spatial_mean_regularization
            + float(self.weights.get("attention_entropy", 0.0)) * attention_entropy_regularization
            + float(self.weights.get("attention_scale_equivariance", 0.0))
            * attention_scale_equivariance
            + float(self.weights.get("attention_scale_residual", 0.0)) * attention_scale_residual
            + self.attention_scale_oracle_weight * attention_scale_oracle
            + self.attention_weight_target_weight * attention_weight_target
        )
        return {
            "total": total,
            "depth": depth_loss.detach(),
            "coarse_depth": coarse_depth_loss.detach(),
            "gradient": gradient_loss.detach(),
            "residual_teacher": residual_teacher.detach(),
            "frame_residual_teacher": frame_residual_teacher.detach(),
            "local_residual_teacher": local_residual_teacher.detach(),
            "low_smoothness": low_smoothness.detach(),
            "detail_regularization": detail_regularization.detach(),
            "additive_residual_teacher": additive_residual_teacher.detach(),
            "additive_regularization": additive_regularization.detach(),
            "additive_scale_orthogonality": additive_scale_orthogonality.detach(),
            "trust": trust_loss.detach(),
            "frame_trust": frame_trust_loss.detach(),
            "uncertainty": uncertainty_loss.detach(),
            "degradation": degradation_loss.detach(),
            "adapter_gate": adapter_gate_loss.detach(),
            "spatial_mean": spatial_mean_regularization.detach(),
            "attention_entropy": attention_entropy_regularization.detach(),
            "attention_scale_equivariance": attention_scale_equivariance.detach(),
            "attention_scale_residual": attention_scale_residual.detach(),
            "attention_scale_oracle": attention_scale_oracle.detach(),
            "attention_weight_target": attention_weight_target.detach(),
        }
