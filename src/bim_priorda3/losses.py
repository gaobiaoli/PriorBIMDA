from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from bim_priorda3.config import Config


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
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


class BIMPriorLoss(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.weights = cfg.loss
        self.candidate_fusion = bool(cfg.model.get("candidate_fusion", False))
        self.strong_anchor = bool(cfg.model.get("strong_anchor", False))
        self.max_log_residual = float(cfg.model.get("max_log_residual", 0.2))

    def forward(
        self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.candidate_fusion:
            return self._candidate_fusion_loss(output, batch)
        valid = (batch["gt_valid"] > 0) & (batch["gt_depth"] > 0)
        pixel_weight = batch["gt_weight"].clamp_min(0.05)
        if self.strong_anchor:
            near_weight = float(self.weights.get("near_range_boost", 0.0))
            pixel_weight = pixel_weight * (
                1.0 + near_weight * (batch["gt_depth"] < 1.0).float()
            )
        log_prediction = torch.log(output["depth"].clamp_min(1e-3))
        log_target = torch.log(batch["gt_depth"].clamp_min(1e-3))
        log_error = log_prediction - log_target

        pooled_depth_loss = _masked_mean(log_error.abs(), valid, pixel_weight)
        frame_depth_loss = _masked_per_sample_mean(
            log_error.abs(), valid, pixel_weight
        )
        depth_loss = (
            0.5 * pooled_depth_loss + 0.5 * frame_depth_loss
            if self.strong_anchor
            else pooled_depth_loss
        )
        uncertainty = torch.exp(-output["log_variance"]) * log_error.abs()
        uncertainty = uncertainty + output["log_variance"]
        uncertainty_loss = _masked_mean(uncertainty, valid, pixel_weight)

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

        trust_loss_raw = functional.binary_cross_entropy_with_logits(
            output["trust_logits"], batch["trust_target"], reduction="none"
        )
        trust_loss = _masked_mean(trust_loss_raw, batch["trust_mask"] > 0)
        trust_count = batch["trust_mask"].sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
        frame_target = (
            (batch["trust_target"] * batch["trust_mask"]).sum(
                dim=(-2, -1), keepdim=True
            )
            / trust_count
        )
        frame_trust_loss = functional.binary_cross_entropy_with_logits(
            output["frame_trust_logits"], frame_target
        )

        anchor_mask = (batch["trust_target"] > 0.7) & (batch["trust_mask"] > 0)
        bim_log = torch.log(batch["bim_depth"].clamp_min(1e-3))
        anchor_loss = _masked_mean((log_prediction - bim_log).abs(), anchor_mask)

        total = (
            float(self.weights.depth) * depth_loss
            + float(self.weights.gradient) * gradient_loss
            + float(self.weights.trust) * trust_loss
            + float(self.weights.frame_trust) * frame_trust_loss
            + float(self.weights.uncertainty) * uncertainty_loss
            + float(self.weights.bim_anchor) * anchor_loss
        )
        result = {
            "total": total,
            "depth": depth_loss.detach(),
            "frame_depth": frame_depth_loss.detach(),
            "gradient": gradient_loss.detach(),
            "trust": trust_loss.detach(),
            "frame_trust": frame_trust_loss.detach(),
            "uncertainty": uncertainty_loss.detach(),
            "bim_anchor": anchor_loss.detach(),
        }
        if self.strong_anchor:
            log_anchor = torch.log(batch["anchor_depth"].clamp_min(1e-3))
            anchor_error = (log_anchor - log_target).abs()
            predicted_error = log_error.abs()
            degradation_margin = float(
                self.weights.get("degradation_log_margin", 0.005)
            )
            degradation_loss = _masked_mean(
                functional.relu(
                    predicted_error - anchor_error - degradation_margin
                ),
                valid,
                pixel_weight,
            )
            preservation_threshold = float(
                self.weights.get("preservation_log_threshold", 0.04)
            )
            preservation_mask = valid & (anchor_error <= preservation_threshold)
            preservation_loss = _masked_mean(
                (log_prediction - log_anchor).abs(),
                preservation_mask,
                pixel_weight,
            )
            gate_temperature = float(self.weights.get("update_gate_temperature", 0.02))
            update_target = torch.sigmoid(
                (anchor_error - preservation_threshold) / gate_temperature
            )
            update_gate_raw = functional.binary_cross_entropy_with_logits(
                output["update_gate_logits"],
                update_target,
                reduction="none",
            )
            update_gate_loss = _masked_mean(
                update_gate_raw,
                valid,
                pixel_weight,
            )
            residual_target = (log_target - log_anchor).clamp(
                -self.max_log_residual,
                self.max_log_residual,
            )
            residual_teacher_loss = _masked_per_sample_mean(
                functional.smooth_l1_loss(
                    output["residual_proposal"],
                    residual_target,
                    reduction="none",
                    beta=0.02,
                ),
                valid,
                pixel_weight,
            )
            total = (
                total
                + float(self.weights.get("degradation", 0.0)) * degradation_loss
                + float(self.weights.get("preservation", 0.0)) * preservation_loss
                + float(self.weights.get("update_gate", 0.0)) * update_gate_loss
                + float(self.weights.get("residual_teacher", 0.0))
                * residual_teacher_loss
            )
            result.update(
                {
                    "total": total,
                    "degradation": degradation_loss.detach(),
                    "preservation": preservation_loss.detach(),
                    "update_gate": update_gate_loss.detach(),
                    "residual_teacher": residual_teacher_loss.detach(),
                }
            )
        return result

    def _candidate_fusion_loss(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        valid = (batch["gt_valid"] > 0) & (batch["gt_depth"] > 0)
        pixel_weight = batch["gt_weight"].clamp_min(0.05)
        near_weight = float(self.weights.get("near_range_boost", 0.0))
        pixel_weight = pixel_weight * (
            1.0 + near_weight * (batch["gt_depth"] < 1.0).float()
        )
        log_target = torch.log(batch["gt_depth"].clamp_min(1e-3))
        log_prediction = torch.log(output["depth"].clamp_min(1e-3))
        log_anchor = torch.log(batch["anchor_depth"].clamp_min(1e-3))
        log_candidate = torch.log(batch["candidate_depth"].clamp_min(1e-3))
        prediction_error = (log_prediction - log_target).abs()
        anchor_error = (log_anchor - log_target).abs()
        candidate_error = (log_candidate - log_target).abs()

        depth_loss = 0.5 * _masked_mean(
            prediction_error, valid, pixel_weight
        ) + 0.5 * _masked_per_sample_mean(
            prediction_error, valid, pixel_weight
        )
        horizontal_valid = valid[..., :, 1:] & valid[..., :, :-1]
        vertical_valid = valid[..., 1:, :] & valid[..., :-1, :]
        gradient_loss = 0.5 * (
            _masked_mean(
                (
                    (log_prediction[..., :, 1:] - log_prediction[..., :, :-1])
                    - (log_target[..., :, 1:] - log_target[..., :, :-1])
                ).abs(),
                horizontal_valid,
            )
            + _masked_mean(
                (
                    (log_prediction[..., 1:, :] - log_prediction[..., :-1, :])
                    - (log_target[..., 1:, :] - log_target[..., :-1, :])
                ).abs(),
                vertical_valid,
            )
        )

        margin = float(self.weights.get("candidate_margin", 0.005))
        temperature = float(self.weights.get("candidate_temperature", 0.02))
        candidate_target = torch.sigmoid(
            (anchor_error - candidate_error - margin) / temperature
        )
        gate_raw = functional.binary_cross_entropy_with_logits(
            output["update_gate_logits"],
            candidate_target,
            reduction="none",
        )
        gate_loss = _masked_mean(gate_raw, valid, pixel_weight)

        dimensions = tuple(range(1, anchor_error.ndim))
        effective = valid.float() * pixel_weight
        denominator = effective.sum(dim=dimensions).clamp_min(1.0)
        frame_advantage = (
            ((anchor_error - candidate_error) * effective).sum(dim=dimensions)
            / denominator
        )
        frame_target = torch.sigmoid(frame_advantage / temperature).view(
            -1, 1, 1, 1
        )
        frame_gate_loss = functional.binary_cross_entropy_with_logits(
            output["frame_trust_logits"],
            frame_target,
        )

        degradation_margin = float(
            self.weights.get("degradation_log_margin", 0.01)
        )
        degradation_loss = _masked_mean(
            functional.relu(
                prediction_error - anchor_error - degradation_margin
            ),
            valid,
            pixel_weight,
        )
        anchor_wins = valid & (anchor_error + margin < candidate_error)
        preservation_loss = _masked_mean(
            (log_prediction - log_anchor).abs(),
            anchor_wins,
            pixel_weight,
        )
        total = (
            float(self.weights.depth) * depth_loss
            + float(self.weights.gradient) * gradient_loss
            + float(self.weights.get("candidate_gate", 1.0)) * gate_loss
            + float(self.weights.get("frame_candidate_gate", 0.1))
            * frame_gate_loss
            + float(self.weights.get("degradation", 0.5)) * degradation_loss
            + float(self.weights.get("preservation", 0.25)) * preservation_loss
        )
        return {
            "total": total,
            "depth": depth_loss.detach(),
            "gradient": gradient_loss.detach(),
            "candidate_gate": gate_loss.detach(),
            "frame_candidate_gate": frame_gate_loss.detach(),
            "degradation": degradation_loss.detach(),
            "preservation": preservation_loss.detach(),
        }
