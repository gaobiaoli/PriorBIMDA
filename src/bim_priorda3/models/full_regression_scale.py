from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional

from .blocks import ConvNormAct, ResidualBlock


class FullRegressionIterativeScaleHead(nn.Module):
    """Regress bounded log-scale residual updates without an M-estimator.

    BIM/DA3 log ratios and their residuals to the current estimate are neural
    input features only.  No ratio value is converted into a scale through a
    Huber weight, IRLS update, weighted ratio mean, or deterministic center.
    One shared updater predicts ``delta c`` in each recurrent round.
    """

    ESTIMATOR_NAME = "full_regression_iterative_v1"

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int = 24,
        attention_heads: int = 4,
        min_support: int = 100,
        ratio_min: float = 0.2,
        ratio_max: float = 5.0,
        token_dropout_probability: float = 0.10,
        iterative_updates: int = 3,
        iterative_hidden_channels: int = 32,
        delta_hidden_channels: int = 64,
        iterative_max_log_update: float = 0.15,
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
        if not 0 <= token_dropout_probability < 1:
            raise ValueError("token_dropout_probability must be in [0, 1)")
        if iterative_updates < 1:
            raise ValueError("full regression requires at least one iterative update")
        if iterative_hidden_channels < 1 or delta_hidden_channels < 1:
            raise ValueError("regression hidden channels must be positive")
        if iterative_max_log_update <= 0:
            raise ValueError("iterative_max_log_update must be positive")
        if da3_feature_channels < 0:
            raise ValueError("da3_feature_channels must be non-negative")

        self.attention_heads = int(attention_heads)
        self.min_support = int(min_support)
        self.log_ratio_min = math.log(float(ratio_min))
        self.log_ratio_max = math.log(float(ratio_max))
        self.token_dropout_probability = float(token_dropout_probability)
        self.iterative_updates = int(iterative_updates)
        self.iterative_max_log_update = float(iterative_max_log_update)
        self.da3_feature_channels = int(da3_feature_channels)

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
        condition_channels = embedding_channels
        self.da3_mid_projection: nn.Sequential | None = None
        self.da3_deep_projection: nn.Sequential | None = None
        self.da3_spatial_fusion: nn.Sequential | None = None
        if self.da3_feature_channels:

            def projection() -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(
                        self.da3_feature_channels,
                        embedding_channels,
                        kernel_size=1,
                    ),
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

        # Per-token inputs: encoded image/geometry feature, log-ratio residual,
        # its magnitude, the measured log ratio, valid-pixel fraction, and the
        # current scalar center.  The module is instantiated once and reused.
        token_input_channels = embedding_channels + 5
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
        delta_input_channels = (
            updater_channels * self.attention_heads + condition_channels + 2
        )
        self.shared_delta_head = nn.Sequential(
            nn.Linear(delta_input_channels, int(delta_hidden_channels)),
            nn.SiLU(inplace=True),
            nn.Linear(int(delta_hidden_channels), 1),
        )

        # Near-zero but non-degenerate initialization: c0=0 remains an almost
        # exact raw-DA3 predictor, while gradients reach the entire updater on
        # the first optimizer step.
        nn.init.zeros_(self.shared_pool_logits.weight)
        nn.init.zeros_(self.shared_pool_logits.bias)
        final_delta = self.shared_delta_head[-1]
        if not isinstance(final_delta, nn.Linear):
            raise TypeError("shared_delta_head must end with nn.Linear")
        nn.init.normal_(final_delta.weight, std=1e-3)
        nn.init.zeros_(final_delta.bias)

    def forward(
        self,
        features: torch.Tensor,
        log_ratio: torch.Tensor,
        ratio_valid: torch.Tensor,
        da3_feature_mid: torch.Tensor | None = None,
        da3_feature_deep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError("full-regression features must have shape [B,C,H,W]")
        if log_ratio.shape != ratio_valid.shape or log_ratio.ndim != 4:
            raise ValueError("log_ratio and ratio_valid must have equal [B,1,H,W] shapes")
        if log_ratio.shape[1] != 1:
            raise ValueError("full regression expects one log-ratio channel")
        if features.shape[0] != log_ratio.shape[0] or features.shape[-2:] != log_ratio.shape[-2:]:
            raise ValueError("feature and ratio spatial shapes differ")

        valid_pixels = (
            (ratio_valid > 0)
            & torch.isfinite(log_ratio)
            & (log_ratio > self.log_ratio_min)
            & (log_ratio < self.log_ratio_max)
        )
        pixel_support = valid_pixels.sum(dim=(-3, -2, -1))
        supported = pixel_support >= self.min_support

        encoded = self.encoder(features)
        pooled_deep: torch.Tensor | None = None
        if self.da3_feature_channels:
            if da3_feature_mid is None or da3_feature_deep is None:
                raise ValueError("DA3-enabled full regression requires mid and deep features")
            expected_prefix = (features.shape[0], self.da3_feature_channels)
            if da3_feature_mid.shape[:2] != expected_prefix:
                raise ValueError("DA3 mid feature shape is incompatible")
            if da3_feature_deep.shape[:2] != expected_prefix:
                raise ValueError("DA3 deep feature shape is incompatible")
            assert self.da3_mid_projection is not None
            assert self.da3_deep_projection is not None
            assert self.da3_spatial_fusion is not None
            projected_mid = self.da3_mid_projection(da3_feature_mid.to(encoded.dtype))
            projected_mid = functional.interpolate(
                projected_mid,
                size=encoded.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            encoded = self.da3_spatial_fusion(torch.cat((encoded, projected_mid), dim=1))
            projected_deep = self.da3_deep_projection(da3_feature_deep.to(encoded.dtype))
            pooled_deep = projected_deep.mean(dim=(-2, -1))

        token_size = encoded.shape[-2:]
        valid_float = valid_pixels.to(log_ratio.dtype)
        token_fraction = functional.adaptive_avg_pool2d(valid_float, token_size)
        token_numerator = functional.adaptive_avg_pool2d(
            torch.where(valid_pixels, log_ratio, torch.zeros_like(log_ratio)),
            token_size,
        )
        token_log_ratio = token_numerator / token_fraction.clamp_min(1e-6)
        token_mask = (token_fraction > 0).flatten(2).squeeze(1)
        batch_size = features.shape[0]
        token_count = token_size[0] * token_size[1]
        if self.training and self.token_dropout_probability > 0:
            retained = token_mask & (
                torch.rand((batch_size, token_count), device=features.device)
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

        valid_tokens = token_mask[:, None, :].to(encoded.dtype)
        pooled_embedding = (encoded.flatten(2) * valid_tokens).sum(dim=-1) / (
            valid_tokens.sum(dim=-1).clamp_min(1.0)
        )
        condition_embedding = (
            torch.cat((pooled_embedding, pooled_deep), dim=1)
            if pooled_deep is not None
            else pooled_embedding
        )
        support_fraction = valid_pixels.float().mean(dim=(-3, -2, -1))[:, None]

        # c0 is exactly log(1)=0. No deterministic BIM scale enters recurrence.
        center = token_log_ratio.new_zeros((batch_size, 1), dtype=torch.float32)
        iteration_centers: list[torch.Tensor] = []
        iteration_updates: list[torch.Tensor] = []
        iteration_entropies: list[torch.Tensor] = []
        final_attention: torch.Tensor | None = None
        for _ in range(self.iterative_updates):
            center_map = center.to(encoded.dtype).view(batch_size, 1, 1, 1).expand(
                -1, -1, *token_size
            )
            residual = token_log_ratio.to(encoded.dtype) - center_map
            token_inputs = torch.cat(
                (
                    encoded,
                    residual,
                    residual.abs(),
                    token_log_ratio.to(encoded.dtype),
                    token_fraction.to(encoded.dtype),
                    center_map,
                ),
                dim=1,
            )
            token_hidden = self.shared_token_updater(token_inputs)
            pool_logits = self.shared_pool_logits(token_hidden).flatten(2)
            attention = normalize_pool_logits(pool_logits)
            hidden_flat = token_hidden.flatten(2).float()
            pooled_heads = torch.einsum("bhn,bcn->bhc", attention, hidden_flat).flatten(1)
            delta_inputs = torch.cat(
                (
                    pooled_heads,
                    condition_embedding.float(),
                    center,
                    support_fraction.float(),
                ),
                dim=1,
            )
            raw_delta = self.shared_delta_head(delta_inputs).float()
            bounded_delta = self.iterative_max_log_update * torch.tanh(raw_delta)
            # Hard raw-DA3 fallback is a validity rule, not a learned gate.
            bounded_delta = bounded_delta * supported[:, None].float()
            center = center + bounded_delta
            iteration_centers.append(center)
            iteration_updates.append(bounded_delta)
            positive_attention = attention.clamp_min(1e-12)
            iteration_entropies.append(
                -(attention * positive_attention.log()).sum(dim=-1)
            )
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
            size=features.shape[-2:],
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
            "iteration_head_log_scales": iteration_stack.expand(
                -1, -1, self.attention_heads
            ),
            "iteration_normalized_attention_entropy": (
                normalized_iteration_entropy.mean(dim=-1)
            ),
        }
