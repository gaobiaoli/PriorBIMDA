from __future__ import annotations

import torch
from torch import nn

from bim_priorda3.config import Config

from .alignment import RobustLocalAffineAlignment
from .refiner import ConditionalDepthRefiner
from .trust import BIMTrustNet


def safe_log(depth: torch.Tensor) -> torch.Tensor:
    return torch.log(depth.clamp_min(1e-3))


class BIMPriorDA3(nn.Module):
    """Single-frame DA3 refinement with learned BIM reliability."""

    TRUST_CHANNELS = 12
    REFINER_CHANNELS = 15
    STRONG_TRUST_CHANNELS = 16
    STRONG_REFINER_CHANNELS = 17
    CANDIDATE_FUSION_CHANNELS = 13

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        model = cfg.model
        self.max_depth = float(cfg.data.max_depth)
        self.candidate_fusion = bool(model.get("candidate_fusion", False))
        self.frame_safety_threshold = model.get("frame_safety_threshold")
        self.strong_anchor = bool(model.get("strong_anchor", False))
        self.use_frame_residual = bool(model.get("frame_residual", False))
        if self.candidate_fusion:
            self.trust = BIMTrustNet(
                self.CANDIDATE_FUSION_CHANNELS,
                int(model.trust_channels),
                initial_bias=float(model.get("initial_gate_bias", -2.0)),
            )
            self.alignment = None
            self.refiner = None
        elif self.strong_anchor:
            self.trust = BIMTrustNet(
                self.STRONG_TRUST_CHANNELS, int(model.trust_channels)
            )
            self.alignment = None
            self.refiner = ConditionalDepthRefiner(
                self.STRONG_REFINER_CHANNELS,
                int(model.base_channels),
                float(model.max_log_residual),
                gated=True,
                initial_gate_bias=float(model.get("initial_gate_bias", -1.5)),
                frame_residual=self.use_frame_residual,
            )
        else:
            self.trust = BIMTrustNet(self.TRUST_CHANNELS, int(model.trust_channels))
            self.alignment = RobustLocalAffineAlignment(
                kernel_size=int(model.alignment_kernel),
                downsample=int(model.alignment_downsample),
                huber_delta=float(model.alignment_huber_delta),
                min_support=float(model.alignment_min_support),
                scale_range=(
                    float(model.alignment_scale_min),
                    float(model.alignment_scale_max),
                ),
            )
            self.refiner = ConditionalDepthRefiner(
                self.REFINER_CHANNELS,
                int(model.base_channels),
                float(model.max_log_residual),
            )

    def _common_features(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        base = batch["base_depth"]
        bim = batch["bim_depth"]
        valid = batch["bim_valid"]
        log_base = safe_log(base) / 3.0
        log_bim = safe_log(torch.where(valid > 0, bim, base)) / 3.0
        disagreement = torch.abs(log_bim - log_base) * valid
        return [
            batch["rgb"],
            log_base,
            batch["base_confidence"].clamp(0.0, 1.0),
            log_bim * valid,
            valid,
            batch["bim_normals"],
            batch["bim_edge"],
            disagreement,
        ]

    def _strong_features(self, batch: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        base = batch["base_depth"]
        scaled = batch["scaled_depth"]
        anchor = batch["anchor_depth"]
        bim = batch["bim_depth"]
        valid = batch["bim_valid"]
        log_base = safe_log(base) / 3.0
        log_scaled = safe_log(scaled) / 3.0
        log_anchor = safe_log(anchor) / 3.0
        log_bim = safe_log(torch.where(valid > 0, bim, scaled)) / 3.0
        disagreement = torch.abs(safe_log(torch.where(valid > 0, bim, scaled)) - safe_log(scaled))
        return [
            batch["rgb"],
            log_base,
            batch["base_confidence"].clamp(0.0, 1.0),
            log_scaled,
            log_anchor,
            log_bim * valid,
            valid,
            batch["bim_normals"],
            batch["bim_edge"],
            disagreement * valid,
            batch["anchor_field"],
            batch["anchor_support"].clamp(0.0, 1.0),
        ]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.candidate_fusion:
            return self._forward_candidate_fusion(batch)
        if self.strong_anchor:
            return self._forward_strong_anchor(batch)
        common = self._common_features(batch)
        pixel_trust_logits, frame_trust_logits = self.trust(torch.cat(common, dim=1))
        trust_logits = pixel_trust_logits + frame_trust_logits
        trust_probability = torch.sigmoid(trust_logits) * batch["bim_valid"]
        coarse, support, scale, shift = self.alignment(
            batch["base_depth"],
            batch["bim_depth"],
            batch["bim_valid"],
            trust_probability,
            batch["bim_edge"],
        )
        coarse_ratio = safe_log(coarse) - safe_log(batch["base_depth"])
        refiner_input = torch.cat(
            common + [trust_probability, coarse_ratio, support.clamp(0.0, 1.0)],
            dim=1,
        )
        residual, log_variance = self.refiner(refiner_input)
        refined = batch["base_depth"] * torch.exp(residual)
        refined = refined.clamp(1e-3, self.max_depth * 2.0)
        return {
            "depth": refined,
            "base_depth": batch["base_depth"],
            "coarse_depth": coarse,
            "trust_logits": trust_logits,
            "pixel_trust_logits": pixel_trust_logits,
            "frame_trust_logits": frame_trust_logits,
            "trust_probability": trust_probability,
            "support": support,
            "local_scale": scale,
            "local_shift": shift,
            "log_residual": residual,
            "log_variance": log_variance,
        }

    def _forward_candidate_fusion(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        anchor = batch["anchor_depth"]
        candidate = batch["candidate_depth"]
        valid = batch["bim_valid"]
        log_anchor = safe_log(anchor)
        log_candidate = safe_log(candidate)
        log_bim = safe_log(torch.where(valid > 0, batch["bim_depth"], anchor))
        frame_candidate_trust = batch["candidate_frame_trust"].view(-1, 1, 1, 1)
        frame_candidate_map = frame_candidate_trust.expand_as(anchor)
        features = [
            batch["rgb"],
            log_anchor / 3.0,
            log_candidate / 3.0,
            torch.abs(log_candidate - log_anchor),
            batch["candidate_log_variance"].clamp(-3.0, 3.0) / 3.0,
            batch["candidate_trust"].clamp(0.0, 1.0),
            frame_candidate_map.clamp(0.0, 1.0),
            batch["anchor_support"].clamp(0.0, 1.0),
            (log_bim / 3.0) * valid,
            valid,
            batch["bim_edge"],
        ]
        pixel_logits, frame_logits = self.trust(torch.cat(features, dim=1))
        fusion_logits = pixel_logits + frame_logits
        fusion_gate = torch.sigmoid(fusion_logits)
        fused_log_depth = log_anchor + fusion_gate * (log_candidate - log_anchor)
        pixel_fused = torch.exp(fused_log_depth).clamp(
            1e-3, self.max_depth * 2.0
        )
        frame_probability = torch.sigmoid(frame_logits)
        if not self.training and self.frame_safety_threshold is not None:
            use_fusion = frame_probability >= float(self.frame_safety_threshold)
            refined = torch.where(use_fusion, pixel_fused, anchor)
        else:
            refined = pixel_fused
        local_scale = batch["scaled_depth"] / batch["base_depth"].clamp_min(1e-3)
        return {
            "depth": refined,
            "base_depth": batch["base_depth"],
            "scaled_depth": batch["scaled_depth"],
            "anchor_depth": anchor,
            "candidate_depth": candidate,
            "pixel_fused_depth": pixel_fused,
            "coarse_depth": anchor,
            "trust_logits": fusion_logits,
            "pixel_trust_logits": pixel_logits,
            "frame_trust_logits": frame_logits,
            "trust_probability": fusion_gate,
            "support": batch["anchor_support"],
            "local_scale": local_scale,
            "local_shift": torch.zeros_like(local_scale),
            "log_residual": safe_log(refined) - log_anchor,
            "update_gate_logits": fusion_logits,
            "update_gate": fusion_gate,
            "log_variance": batch["candidate_log_variance"],
        }

    def _forward_strong_anchor(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        common = self._strong_features(batch)
        pixel_trust_logits, frame_trust_logits = self.trust(torch.cat(common, dim=1))
        trust_logits = pixel_trust_logits + frame_trust_logits
        trust_probability = torch.sigmoid(trust_logits) * batch["bim_valid"]
        refiner_input = torch.cat(common + [trust_probability], dim=1)
        refiner_output = self.refiner(refiner_input)
        if self.use_frame_residual:
            proposal, log_variance, update_gate_logits, frame_residual = refiner_output
        else:
            proposal, log_variance, update_gate_logits = refiner_output
            frame_residual = torch.zeros_like(proposal[..., :1, :1])
        update_gate = torch.sigmoid(update_gate_logits)
        combined_proposal = (proposal + frame_residual).clamp(
            -float(self.refiner.max_log_residual),
            float(self.refiner.max_log_residual),
        )
        effective_residual = combined_proposal * update_gate
        anchor = batch["anchor_depth"]
        refined = anchor * torch.exp(effective_residual)
        refined = refined.clamp(1e-3, self.max_depth * 2.0)
        local_scale = batch["scaled_depth"] / batch["base_depth"].clamp_min(1e-3)
        return {
            "depth": refined,
            "base_depth": batch["base_depth"],
            "scaled_depth": batch["scaled_depth"],
            "anchor_depth": anchor,
            "coarse_depth": anchor,
            "trust_logits": trust_logits,
            "pixel_trust_logits": pixel_trust_logits,
            "frame_trust_logits": frame_trust_logits,
            "trust_probability": trust_probability,
            "support": batch["anchor_support"],
            "local_scale": local_scale,
            "local_shift": torch.zeros_like(local_scale),
            "log_residual": effective_residual,
            "residual_proposal": combined_proposal,
            "pixel_residual_proposal": proposal,
            "frame_residual_proposal": frame_residual,
            "update_gate_logits": update_gate_logits,
            "update_gate": update_gate,
            "log_variance": log_variance,
        }
