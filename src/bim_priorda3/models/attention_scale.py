from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from .blocks import ConvNormAct, ResidualBlock


class AttentiveBIMScaleHead(nn.Module):
    """Estimate one metric scale per image by attending to BIM/DA3 ratios.

    The network predicts *where* to trust the prior, but it never predicts an
    unconstrained scale value from appearance.  Attention weights are applied
    to measured ``log(BIM / DA3)`` values.  A deterministic scale is retained
    only as a low-support/low-confidence fallback.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 24,
        attention_heads: int = 4,
        min_support: int = 100,
        ratio_min: float = 0.2,
        ratio_max: float = 5.0,
        huber_delta: float = 0.15,
        token_dropout_probability: float = 0.10,
        fallback_gate_bias: float = -1.5,
        bounded_log_scale_residual: float = 0.0,
        residual_hidden_channels: int = 32,
        da3_feature_channels: int = 0,
    ) -> None:
        super().__init__()
        if hidden_channels < 4:
            raise ValueError("hidden_channels must be at least 4")
        if attention_heads < 1:
            raise ValueError("attention_heads must be positive")
        if min_support < 1:
            raise ValueError("min_support must be positive")
        if not 0 < ratio_min < ratio_max:
            raise ValueError("ratio bounds must satisfy 0 < ratio_min < ratio_max")
        if huber_delta <= 0:
            raise ValueError("huber_delta must be positive")
        if not 0 <= token_dropout_probability < 1:
            raise ValueError("token_dropout_probability must be in [0, 1)")
        if bounded_log_scale_residual < 0:
            raise ValueError("bounded_log_scale_residual must be non-negative")
        if residual_hidden_channels < 1:
            raise ValueError("residual_hidden_channels must be positive")

        self.attention_heads = int(attention_heads)
        self.min_support = int(min_support)
        self.log_ratio_min = math.log(float(ratio_min))
        self.log_ratio_max = math.log(float(ratio_max))
        self.huber_delta = float(huber_delta)
        self.token_dropout_probability = float(token_dropout_probability)
        self.bounded_log_scale_residual = float(bounded_log_scale_residual)
        self.da3_feature_channels = int(da3_feature_channels)
        if self.da3_feature_channels < 0:
            raise ValueError("da3_feature_channels must be non-negative")

        channels = int(hidden_channels)
        self.encoder = nn.Sequential(
            ConvNormAct(in_channels, channels, stride=2),
            ResidualBlock(channels),
            ConvNormAct(channels, channels * 2, stride=2),
            ResidualBlock(channels * 2),
            ConvNormAct(channels * 2, channels * 4, stride=2),
            ResidualBlock(channels * 4),
        )
        embedding_channels = channels * 4
        self.da3_mid_projection: nn.Sequential | None = None
        self.da3_deep_projection: nn.Sequential | None = None
        self.da3_spatial_fusion: nn.Sequential | None = None
        condition_channels = embedding_channels
        if self.da3_feature_channels:
            def projection() -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(self.da3_feature_channels, embedding_channels, kernel_size=1),
                    nn.GroupNorm(min(8, embedding_channels), embedding_channels),
                    nn.SiLU(inplace=True),
                )

            self.da3_mid_projection = projection()
            self.da3_deep_projection = projection()
            self.da3_spatial_fusion = nn.Sequential(
                ConvNormAct(embedding_channels * 2, embedding_channels),
                ResidualBlock(embedding_channels),
            )
            condition_channels = embedding_channels * 2
        self.key_logits = nn.Conv2d(
            embedding_channels,
            self.attention_heads,
            kernel_size=1,
        )
        self.head_mixer = nn.Linear(condition_channels, self.attention_heads)
        self.fallback_gate = nn.Linear(condition_channels + 2, 1)
        self.scale_residual_mlp: nn.Sequential | None = None
        if self.bounded_log_scale_residual > 0:
            self.scale_residual_mlp = nn.Sequential(
                nn.Linear(condition_channels, int(residual_hidden_channels)),
                nn.SiLU(),
                nn.Linear(int(residual_hidden_channels), 1),
            )

        # Start close to a uniform spatial estimator and mostly retain the
        # deterministic fallback.  Training can then specialize attention
        # without an unstable scale jump on the first update.
        nn.init.zeros_(self.key_logits.weight)
        nn.init.zeros_(self.key_logits.bias)
        nn.init.zeros_(self.head_mixer.weight)
        nn.init.zeros_(self.head_mixer.bias)
        nn.init.zeros_(self.fallback_gate.weight)
        nn.init.constant_(self.fallback_gate.bias, float(fallback_gate_bias))
        if self.scale_residual_mlp is not None:
            # The new branch is an exact no-op at initialization.  This keeps
            # the scale estimator anchored to the measured BIM/DA3 ratios
            # while allowing a small appearance-conditioned correction later.
            final = self.scale_residual_mlp[-1]
            if not isinstance(final, nn.Linear):
                raise TypeError("scale_residual_mlp must end with nn.Linear")
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        if self.da3_feature_channels:
            # DA3 features are a native input in this variant, not a late
            # zero-gated adapter. Small logits retain a stable near-uniform
            # initial estimator while allowing encoder gradients immediately.
            nn.init.normal_(self.key_logits.weight, std=1e-3)
            nn.init.normal_(self.head_mixer.weight, std=1e-3)

    @staticmethod
    def _masked_spatial_mean(
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        numerator = (values * valid).sum(dim=(-2, -1))
        denominator = valid.sum(dim=(-2, -1)).clamp_min(1.0)
        return numerator / denominator

    def forward(
        self,
        features: torch.Tensor,
        log_ratio: torch.Tensor,
        ratio_valid: torch.Tensor,
        fallback_log_scale: torch.Tensor,
        da3_feature_mid: torch.Tensor | None = None,
        da3_feature_deep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("attention scale features must have shape [B, C, H, W]")
        if log_ratio.shape != ratio_valid.shape or log_ratio.ndim != 4:
            raise ValueError("log_ratio and ratio_valid must have equal [B, 1, H, W] shapes")
        if log_ratio.shape[1] != 1:
            raise ValueError("attention scale expects one log-ratio channel")
        if features.shape[0] != log_ratio.shape[0] or features.shape[-2:] != log_ratio.shape[-2:]:
            raise ValueError("attention scale feature and ratio spatial shapes differ")
        expected_fallback_shape = (features.shape[0], 1, 1, 1)
        if fallback_log_scale.shape != expected_fallback_shape:
            raise ValueError(
                "fallback_log_scale must have shape "
                f"{expected_fallback_shape}; got {tuple(fallback_log_scale.shape)}"
            )

        valid_pixels = (
            (ratio_valid > 0)
            & torch.isfinite(log_ratio)
            & (log_ratio > self.log_ratio_min)
            & (log_ratio < self.log_ratio_max)
        )
        pixel_support = valid_pixels.sum(dim=(-3, -2, -1))

        encoded = self.encoder(features)
        pooled_deep: torch.Tensor | None = None
        if self.da3_feature_channels:
            if da3_feature_mid is None or da3_feature_deep is None:
                raise ValueError("DA3-enabled attention scale requires mid and deep features")
            expected_prefix = (features.shape[0], self.da3_feature_channels)
            if da3_feature_mid.shape[:2] != expected_prefix:
                raise ValueError(
                    "DA3 mid feature must start with shape "
                    f"{expected_prefix}; got {tuple(da3_feature_mid.shape)}"
                )
            if da3_feature_deep.shape[:2] != expected_prefix:
                raise ValueError(
                    "DA3 deep feature must start with shape "
                    f"{expected_prefix}; got {tuple(da3_feature_deep.shape)}"
                )
            assert self.da3_mid_projection is not None
            assert self.da3_deep_projection is not None
            assert self.da3_spatial_fusion is not None
            projected_mid = self.da3_mid_projection(da3_feature_mid.to(dtype=encoded.dtype))
            projected_mid = functional.interpolate(
                projected_mid,
                size=encoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            encoded = self.da3_spatial_fusion(torch.cat((encoded, projected_mid), dim=1))
            projected_deep = self.da3_deep_projection(
                da3_feature_deep.to(dtype=encoded.dtype)
            )
            pooled_deep = projected_deep.mean(dim=(-2, -1))
        token_size = encoded.shape[-2:]
        valid_float = valid_pixels.to(dtype=log_ratio.dtype)
        token_fraction = functional.adaptive_avg_pool2d(valid_float, token_size)
        token_numerator = functional.adaptive_avg_pool2d(
            torch.where(valid_pixels, log_ratio, torch.zeros_like(log_ratio)),
            token_size,
        )
        token_log_ratio = token_numerator / token_fraction.clamp_min(1e-6)
        token_valid = token_fraction > 0

        batch_size = features.shape[0]
        token_count = token_size[0] * token_size[1]
        token_values = token_log_ratio.flatten(2).squeeze(1).float()
        token_mask = token_valid.flatten(2).squeeze(1)
        if self.training and self.token_dropout_probability > 0:
            retained = token_mask & (
                torch.rand(
                    (batch_size, token_count),
                    device=features.device,
                )
                >= self.token_dropout_probability
            )
            # Token dropout is a robustness augmentation, not a second failure
            # mode.  If it happens to remove everything, retain the original
            # valid-token set for that sample.
            has_retained = retained.any(dim=1, keepdim=True)
            token_mask = torch.where(has_retained, retained, token_mask)

        logits = self.key_logits(encoded).flatten(2).float()
        masked_logits = logits.masked_fill(~token_mask[:, None, :], -torch.inf)
        no_valid_token = ~token_mask.any(dim=1)
        if bool(no_valid_token.any()):
            # Softmax cannot consume an all--inf row.  The corresponding sample
            # is forced to the deterministic fallback below, so this temporary
            # token only keeps the arithmetic finite.
            masked_logits = masked_logits.clone()
            masked_logits[no_valid_token, :, 0] = 0.0
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * token_mask[:, None, :].float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Two deterministic IRLS-style Huber updates.  The values are measured
        # log ratios; only their spatial weights and the confidence fallback
        # are learned.
        center = fallback_log_scale.flatten(1).float().expand(-1, self.attention_heads)
        expanded_values = token_values[:, None, :]
        for _ in range(2):
            residual = expanded_values - center[:, :, None]
            robust_weight = torch.rsqrt(1.0 + (residual / self.huber_delta).square())
            effective_weight = attention * robust_weight
            center = (effective_weight * expanded_values).sum(dim=-1) / effective_weight.sum(
                dim=-1
            ).clamp_min(1e-8)
        head_log_scale = center

        valid_tokens_float = token_mask[:, None, :].to(dtype=encoded.dtype)
        pooled_embedding = (encoded.flatten(2) * valid_tokens_float).sum(
            dim=-1
        ) / valid_tokens_float.sum(dim=-1).clamp_min(1.0)
        condition_embedding = (
            torch.cat((pooled_embedding, pooled_deep), dim=1)
            if pooled_deep is not None
            else pooled_embedding
        )
        head_mixture = torch.softmax(self.head_mixer(condition_embedding).float(), dim=-1)
        raw_attentive_log_scale = (head_mixture * head_log_scale).sum(dim=-1, keepdim=True)
        if self.scale_residual_mlp is None:
            log_scale_residual = torch.zeros_like(raw_attentive_log_scale)
        else:
            log_scale_residual = self.bounded_log_scale_residual * torch.tanh(
                self.scale_residual_mlp(condition_embedding).float()
            )
        attentive_log_scale = raw_attentive_log_scale + log_scale_residual

        support_fraction = valid_pixels.float().mean(dim=(-3, -2, -1), keepdim=False)[:, None]
        fallback_flat = fallback_log_scale.flatten(1).float()
        scale_disagreement = (attentive_log_scale - fallback_flat).abs()
        gate_features = torch.cat(
            (condition_embedding.float(), support_fraction, scale_disagreement),
            dim=1,
        )
        learned_gate = torch.sigmoid(self.fallback_gate(gate_features).float())
        sufficient = (pixel_support >= self.min_support).float()[:, None]
        fallback_gate = learned_gate * sufficient
        log_scale = fallback_gate * attentive_log_scale + (1.0 - fallback_gate) * fallback_flat

        positive_attention = attention.clamp_min(1e-12)
        entropy = -(attention * positive_attention.log()).sum(dim=-1)
        valid_token_count = token_mask.sum(dim=-1).float()
        entropy_denominator = valid_token_count.clamp_min(2.0).log()[:, None]
        normalized_entropy = torch.where(
            valid_token_count[:, None] > 1,
            entropy / entropy_denominator,
            torch.ones_like(entropy),
        )
        attention_map = functional.interpolate(
            attention.mean(dim=1).reshape(batch_size, 1, *token_size),
            size=features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention_token_distribution = attention.reshape(
            batch_size,
            self.attention_heads,
            *token_size,
        )
        attention_token_valid = token_mask.reshape(batch_size, 1, *token_size)

        return {
            "scale": log_scale.exp().view(-1, 1, 1, 1),
            "log_scale": log_scale.view(-1, 1, 1, 1),
            "attentive_log_scale": attentive_log_scale.view(-1, 1, 1, 1),
            "raw_attentive_log_scale": raw_attentive_log_scale.view(-1, 1, 1, 1),
            "bounded_log_scale_residual": log_scale_residual.view(-1, 1, 1, 1),
            "fallback_log_scale": fallback_flat.view(-1, 1, 1, 1),
            "fallback_gate": fallback_gate.view(-1, 1, 1, 1),
            "pixel_support": pixel_support,
            "token_support": token_mask.sum(dim=-1),
            "head_log_scale": head_log_scale,
            "head_mixture": head_mixture,
            "normalized_attention_entropy": normalized_entropy.mean(dim=-1),
            "attention_map": attention_map,
            # Expose the normalized token distribution and the exact
            # post-dropout support used by the scale estimator.  Train-only
            # direct attention supervision can therefore target the weights
            # without approximating them from an interpolated display map.
            "attention_token_distribution": attention_token_distribution,
            "attention_token_valid": attention_token_valid,
        }
