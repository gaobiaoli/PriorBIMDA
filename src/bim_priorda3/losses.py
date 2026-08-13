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
        scaled_depth = output["scaled_depth"] if uses_live_da3 else batch["scaled_depth"]
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
        detail_regularization = output["detail_log_residual"].abs().mean()

        warm = self.current_epoch >= self.warmup_epochs
        zero = log_prediction.sum() * 0.0
        if uses_live_da3:
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
            + float(self.weights.gradient) * gradient_loss
            + float(self.weights.get("residual_teacher", 0.0)) * residual_teacher
            + float(self.weights.get("frame_residual_teacher", 0.0)) * frame_residual_teacher
            + float(self.weights.get("local_residual_teacher", 0.0)) * local_residual_teacher
            + float(self.weights.get("low_smoothness", 0.0)) * low_smoothness
            + float(self.weights.get("detail_regularization", 0.0)) * detail_regularization
            + float(self.weights.get("trust", 0.0)) * trust_loss
            + float(self.weights.get("frame_trust", 0.0)) * frame_trust_loss
            + float(self.weights.get("uncertainty", 0.0)) * uncertainty_loss
            + float(self.weights.get("degradation", 0.0)) * degradation_loss
            + float(self.weights.get("adapter_gate", 0.0)) * adapter_gate_loss
        )
        return {
            "total": total,
            "depth": depth_loss.detach(),
            "gradient": gradient_loss.detach(),
            "residual_teacher": residual_teacher.detach(),
            "frame_residual_teacher": frame_residual_teacher.detach(),
            "local_residual_teacher": local_residual_teacher.detach(),
            "low_smoothness": low_smoothness.detach(),
            "detail_regularization": detail_regularization.detach(),
            "trust": trust_loss.detach(),
            "frame_trust": frame_trust_loss.detach(),
            "uncertainty": uncertainty_loss.detach(),
            "degradation": degradation_loss.detach(),
            "adapter_gate": adapter_gate_loss.detach(),
        }
