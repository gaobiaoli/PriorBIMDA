from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from .blocks import ConvNormAct, ResidualBlock


class RGBDINOFullRegressionIterativeScaleHead(nn.Module):
    """Three-round scale regression from separate appearance and geometry paths.

    The DINOv2 tensor is a frozen, cached last-layer patch grid.  This module
    contains only its trainable 1x1 projection; it never owns or calls a DINO
    backbone.  DA3 contributes its metric depth and BIM/DA3 log ratio only.
    """

    ESTIMATOR_NAME = "full_regression_rgb_dinov2_iterative_v1"

    def __init__(
        self,
        *,
        geometry_channels: int = 7,
        rgb_base_channels: int = 24,
        fusion_channels: int = 96,
        dinov2_channels: int = 768,
        attention_heads: int = 4,
        min_support: int = 100,
        ratio_min: float = 0.2,
        ratio_max: float = 5.0,
        token_dropout_probability: float = 0.10,
        iterative_updates: int = 3,
        iterative_hidden_channels: int = 32,
        delta_hidden_channels: int = 64,
        iterative_max_log_update: float = 0.15,
    ) -> None:
        super().__init__()
        if geometry_channels < 1:
            raise ValueError("geometry_channels must be positive")
        if rgb_base_channels < 4:
            raise ValueError("rgb_base_channels must be at least 4")
        if fusion_channels < 4:
            raise ValueError("fusion_channels must be at least 4")
        if dinov2_channels < 1:
            raise ValueError("dinov2_channels must be positive")
        if attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if min_support < 1:
            raise ValueError("min_support must be positive")
        if not 0 < ratio_min < ratio_max:
            raise ValueError("ratio bounds must satisfy 0 < ratio_min < ratio_max")
        if not 0 <= token_dropout_probability < 1:
            raise ValueError("token_dropout_probability must be in [0, 1)")
        if iterative_updates < 1:
            raise ValueError("full regression requires at least one iterative update")
        if iterative_hidden_channels < 1 or delta_hidden_channels < 1:
            raise ValueError("regression hidden channels must be positive")
        if iterative_max_log_update <= 0:
            raise ValueError("iterative_max_log_update must be positive")

        self.geometry_channels = int(geometry_channels)
        self.rgb_base_channels = int(rgb_base_channels)
        self.fusion_channels = int(fusion_channels)
        self.dinov2_channels = int(dinov2_channels)
        self.attention_heads = int(attention_heads)
        self.min_support = int(min_support)
        self.log_ratio_min = math.log(float(ratio_min))
        self.log_ratio_max = math.log(float(ratio_max))
        self.token_dropout_probability = float(token_dropout_probability)
        self.iterative_updates = int(iterative_updates)
        self.iterative_max_log_update = float(iterative_max_log_update)

        rgb_channels = self.rgb_base_channels
        self.rgb_encoder = nn.Sequential(
            ConvNormAct(3, rgb_channels, stride=2),
            ResidualBlock(rgb_channels),
            ConvNormAct(rgb_channels, rgb_channels * 2, stride=2),
            ResidualBlock(rgb_channels * 2),
            ConvNormAct(rgb_channels * 2, self.fusion_channels, stride=2),
            ResidualBlock(self.fusion_channels),
        )
        self.geometry_encoder = nn.Sequential(
            ConvNormAct(self.geometry_channels, rgb_channels, stride=2),
            ResidualBlock(rgb_channels),
            ConvNormAct(rgb_channels, rgb_channels * 2, stride=2),
            ResidualBlock(rgb_channels * 2),
            ConvNormAct(rgb_channels * 2, self.fusion_channels, stride=2),
            ResidualBlock(self.fusion_channels),
        )
        self.dinov2_projection = nn.Sequential(
            nn.Conv2d(self.dinov2_channels, self.fusion_channels, kernel_size=1),
            nn.GroupNorm(min(8, self.fusion_channels), self.fusion_channels),
            nn.SiLU(inplace=True),
        )
        self.static_fusion = nn.Sequential(
            ConvNormAct(self.fusion_channels * 2, self.fusion_channels),
            ResidualBlock(self.fusion_channels),
        )

        # Static RGB+DINO, metric geometry, then the five recurrent fields:
        # ratio residual, |residual|, ratio, valid fraction and current center.
        token_input_channels = self.fusion_channels * 2 + 5
        updater_channels = int(iterative_hidden_channels)
        self.shared_token_updater = nn.Sequential(
            nn.Conv2d(token_input_channels, updater_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(updater_channels, updater_channels, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        self.shared_pool_logits = nn.Conv2d(
            updater_channels,
            self.attention_heads,
            kernel_size=1,
        )
        # DINO cannot predict scale directly: its global descriptor only enters
        # after RGB+DINO spatial fusion and alongside ratio-dependent tokens.
        delta_input_channels = updater_channels * self.attention_heads + self.fusion_channels + 2
        self.shared_delta_head = nn.Sequential(
            nn.Linear(delta_input_channels, int(delta_hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(int(delta_hidden_channels), 1),
        )

        nn.init.zeros_(self.shared_pool_logits.weight)
        nn.init.zeros_(self.shared_pool_logits.bias)
        final_delta = self.shared_delta_head[-1]
        if not isinstance(final_delta, nn.Linear):
            raise TypeError("shared_delta_head must end with nn.Linear")
        nn.init.normal_(final_delta.weight, std=1e-3)
        nn.init.zeros_(final_delta.bias)

    def forward(
        self,
        rgb: torch.Tensor,
        geometry: torch.Tensor,
        log_ratio: torch.Tensor,
        ratio_valid: torch.Tensor,
        dinov2_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [B,3,H,W]")
        if geometry.ndim != 4 or geometry.shape[1] != self.geometry_channels:
            raise ValueError(f"geometry must have shape [B,{self.geometry_channels},H,W]")
        if log_ratio.shape != ratio_valid.shape or log_ratio.ndim != 4:
            raise ValueError("log_ratio and ratio_valid must have equal [B,1,H,W] shapes")
        if log_ratio.shape[1] != 1:
            raise ValueError("full regression expects one log-ratio channel")
        if dinov2_feature.ndim != 4 or dinov2_feature.shape[1] != self.dinov2_channels:
            raise ValueError(
                f"dinov2_feature must have shape [B,{self.dinov2_channels},H_tokens,W_tokens]"
            )
        batch_size = rgb.shape[0]
        if any(value.shape[0] != batch_size for value in (geometry, log_ratio, dinov2_feature)):
            raise ValueError("RGB, geometry, ratio and DINO batch sizes differ")
        if geometry.shape[-2:] != rgb.shape[-2:] or log_ratio.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB, geometry and ratio spatial shapes differ")

        valid_pixels = (
            (ratio_valid > 0)
            & torch.isfinite(log_ratio)
            & (log_ratio > self.log_ratio_min)
            & (log_ratio < self.log_ratio_max)
        )
        pixel_support = valid_pixels.sum(dim=(-3, -2, -1))
        supported = pixel_support >= self.min_support

        rgb_feature = self.rgb_encoder(rgb)
        geometry_feature = self.geometry_encoder(geometry)
        projected_dino = self.dinov2_projection(dinov2_feature.to(rgb_feature.dtype))
        projected_dino = functional.interpolate(
            projected_dino,
            size=rgb_feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        static_feature = self.static_fusion(torch.cat((rgb_feature, projected_dino), dim=1))
        if geometry_feature.shape[-2:] != static_feature.shape[-2:]:
            raise RuntimeError("RGB and geometry encoders produced different spatial shapes")

        token_size = static_feature.shape[-2:]
        valid_float = valid_pixels.to(log_ratio.dtype)
        token_fraction = functional.adaptive_avg_pool2d(valid_float, token_size)
        token_numerator = functional.adaptive_avg_pool2d(
            torch.where(valid_pixels, log_ratio, torch.zeros_like(log_ratio)),
            token_size,
        )
        token_log_ratio = token_numerator / token_fraction.clamp_min(1e-6)
        token_mask = (token_fraction > 0).flatten(2).squeeze(1)
        token_count = token_size[0] * token_size[1]
        if self.training and self.token_dropout_probability > 0:
            retained = token_mask & (
                torch.rand((batch_size, token_count), device=rgb.device)
                >= self.token_dropout_probability
            )
            token_mask = torch.where(retained.any(dim=1, keepdim=True), retained, token_mask)
        no_valid_token = ~token_mask.any(dim=1)

        def normalize_pool_logits(logits: torch.Tensor) -> torch.Tensor:
            masked = logits.masked_fill(~token_mask[:, None, :], -torch.inf)
            if bool(no_valid_token.any()):
                masked = masked.clone()
                masked[no_valid_token, :, 0] = 0.0
            weights = torch.softmax(masked.float(), dim=-1)
            weights = weights * token_mask[:, None, :].float()
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Per the experiment contract, global condition is RGB+DINO only.
        static_condition = static_feature.mean(dim=(-2, -1))
        support_fraction = valid_pixels.float().mean(dim=(-3, -2, -1))[:, None]
        center = token_log_ratio.new_zeros((batch_size, 1), dtype=torch.float32)
        iteration_centers: list[torch.Tensor] = []
        iteration_updates: list[torch.Tensor] = []
        iteration_entropies: list[torch.Tensor] = []
        final_attention: torch.Tensor | None = None
        for _ in range(self.iterative_updates):
            center_map = (
                center.to(static_feature.dtype)
                .view(batch_size, 1, 1, 1)
                .expand(-1, -1, *token_size)
            )
            residual = token_log_ratio.to(static_feature.dtype) - center_map
            token_inputs = torch.cat(
                (
                    static_feature,
                    geometry_feature,
                    residual,
                    residual.abs(),
                    token_log_ratio.to(static_feature.dtype),
                    token_fraction.to(static_feature.dtype),
                    center_map,
                ),
                dim=1,
            )
            token_hidden = self.shared_token_updater(token_inputs)
            pool_logits = self.shared_pool_logits(token_hidden).flatten(2)
            attention = normalize_pool_logits(pool_logits)
            pooled_heads = torch.einsum(
                "bhn,bcn->bhc",
                attention,
                token_hidden.flatten(2).float(),
            ).flatten(1)
            raw_delta = self.shared_delta_head(
                torch.cat(
                    (
                        pooled_heads,
                        static_condition.float(),
                        center,
                        support_fraction.float(),
                    ),
                    dim=1,
                )
            ).float()
            bounded_delta = self.iterative_max_log_update * torch.tanh(raw_delta)
            bounded_delta = bounded_delta * supported[:, None].float()
            center = center + bounded_delta
            iteration_centers.append(center)
            iteration_updates.append(bounded_delta)
            positive_attention = attention.clamp_min(1e-12)
            iteration_entropies.append(-(attention * positive_attention.log()).sum(dim=-1))
            final_attention = attention

        assert final_attention is not None
        iteration_stack = torch.stack(iteration_centers, dim=1)
        update_stack = torch.stack(iteration_updates, dim=1)
        token_support = token_mask.sum(dim=-1)
        entropy_denominator = token_support.float().clamp_min(2.0).log()[:, None]
        entropy = iteration_entropies[-1]
        normalized_entropy = torch.where(
            token_support[:, None] > 1,
            entropy / entropy_denominator,
            torch.ones_like(entropy),
        )
        iteration_entropy = torch.stack(iteration_entropies, dim=1)
        normalized_iteration_entropy = torch.where(
            token_support[:, None, None] > 1,
            iteration_entropy / entropy_denominator[:, None, :],
            torch.ones_like(iteration_entropy),
        )
        attention_map = functional.interpolate(
            final_attention.mean(dim=1).reshape(batch_size, 1, *token_size),
            size=rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        zero = torch.zeros_like(center)
        uniform_head_mixture = center.new_full(
            (batch_size, self.attention_heads),
            1.0 / self.attention_heads,
        )
        return {
            "scale": center.exp().view(-1, 1, 1, 1),
            "log_scale": center.view(-1, 1, 1, 1),
            "attentive_log_scale": center.view(-1, 1, 1, 1),
            "raw_attentive_log_scale": center.view(-1, 1, 1, 1),
            "bounded_log_scale_residual": zero.view(-1, 1, 1, 1),
            "pixel_support": pixel_support,
            "token_support": token_support,
            "head_log_scale": center.expand(-1, self.attention_heads),
            "head_mixture": uniform_head_mixture,
            "normalized_attention_entropy": normalized_entropy.mean(dim=-1),
            "attention_map": attention_map,
            "attention_token_distribution": final_attention.reshape(
                batch_size, self.attention_heads, *token_size
            ),
            "attention_token_valid": token_mask.reshape(batch_size, 1, *token_size),
            "iteration_log_scales": iteration_stack.unsqueeze(-1),
            "iteration_raw_log_scales": iteration_stack.unsqueeze(-1),
            "iteration_log_scale_updates": update_stack.unsqueeze(-1),
            "iteration_head_log_scales": iteration_stack.expand(-1, -1, self.attention_heads),
            "iteration_normalized_attention_entropy": (normalized_iteration_entropy.mean(dim=-1)),
        }
