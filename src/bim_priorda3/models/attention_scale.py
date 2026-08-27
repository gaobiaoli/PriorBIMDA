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
        iterative_updates: int = 0,
        iterative_hidden_channels: int = 32,
        iterative_initial_log_scale: float = 0.0,
        iterative_damping: tuple[float, ...] | list[float] | None = None,
        iterative_max_log_update: float = 0.15,
        iterative_refresh_attention: bool = True,
        use_fallback_gate: bool = True,
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
        if iterative_updates < 0:
            raise ValueError("iterative_updates must be non-negative")
        if iterative_hidden_channels < 1:
            raise ValueError("iterative_hidden_channels must be positive")
        if not math.isfinite(iterative_initial_log_scale):
            raise ValueError("iterative_initial_log_scale must be finite")
        if iterative_max_log_update <= 0:
            raise ValueError("iterative_max_log_update must be positive")

        self.attention_heads = int(attention_heads)
        self.min_support = int(min_support)
        self.log_ratio_min = math.log(float(ratio_min))
        self.log_ratio_max = math.log(float(ratio_max))
        self.huber_delta = float(huber_delta)
        self.token_dropout_probability = float(token_dropout_probability)
        self.bounded_log_scale_residual = float(bounded_log_scale_residual)
        self.da3_feature_channels = int(da3_feature_channels)
        self.iterative_updates = int(iterative_updates)
        self.iterative_initial_log_scale = float(iterative_initial_log_scale)
        self.iterative_max_log_update = float(iterative_max_log_update)
        self.iterative_refresh_attention = bool(iterative_refresh_attention)
        self.use_fallback_gate = bool(use_fallback_gate)
        if self.da3_feature_channels < 0:
            raise ValueError("da3_feature_channels must be non-negative")
        if iterative_damping is None:
            iterative_damping = [0.5] * self.iterative_updates
        damping = tuple(float(value) for value in iterative_damping)
        if len(damping) != self.iterative_updates:
            raise ValueError(
                "iterative_damping must contain one value per iterative update"
            )
        if any(not 0.0 < value < 1.0 for value in damping):
            raise ValueError("iterative damping values must lie strictly in (0, 1)")

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
        self.fallback_gate: nn.Linear | None = (
            nn.Linear(condition_channels + 2, 1)
            if self.use_fallback_gate
            else None
        )
        self.iterative_reliability: nn.Sequential | None = None
        self.iterative_step_logits: nn.Parameter | None = None
        if self.iterative_updates:
            # The same compact reliability function is reused at every round
            # and for every attention head. Its inputs are the encoded token,
            # the signed/absolute ratio residual to the current scale, and the
            # corresponding analytic Huber confidence.
            self.iterative_reliability = nn.Sequential(
                nn.Conv2d(
                    embedding_channels + 3,
                    int(iterative_hidden_channels),
                    kernel_size=1,
                ),
                nn.SiLU(inplace=True),
                nn.Conv2d(int(iterative_hidden_channels), 1, kernel_size=1),
            )
            damping_tensor = torch.tensor(damping, dtype=torch.float32)
            self.iterative_step_logits = nn.Parameter(torch.logit(damping_tensor))
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
        if self.fallback_gate is not None:
            nn.init.zeros_(self.fallback_gate.weight)
            nn.init.constant_(self.fallback_gate.bias, float(fallback_gate_bias))
        if self.iterative_reliability is not None:
            iterative_output = self.iterative_reliability[-1]
            if not isinstance(iterative_output, nn.Conv2d):
                raise TypeError("iterative_reliability must end with nn.Conv2d")
            # A very small non-zero output keeps the initial estimator near the
            # static one while allowing both MLP layers to receive gradients
            # from the first optimizer step.
            nn.init.normal_(iterative_output.weight, std=1e-3)
            nn.init.zeros_(iterative_output.bias)
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

        static_logits = self.key_logits(encoded).flatten(2).float()
        no_valid_token = ~token_mask.any(dim=1)

        def normalized_attention(candidate_logits: torch.Tensor) -> torch.Tensor:
            masked_logits = candidate_logits.masked_fill(
                ~token_mask[:, None, :],
                -torch.inf,
            )
            if bool(no_valid_token.any()):
                # Softmax cannot consume an all--inf row. The corresponding
                # sample is forced to its fallback below, so this temporary
                # token only keeps the arithmetic finite.
                masked_logits = masked_logits.clone()
                masked_logits[no_valid_token, :, 0] = 0.0
            normalized = torch.softmax(masked_logits, dim=-1)
            normalized = normalized * token_mask[:, None, :].float()
            return normalized / normalized.sum(dim=-1, keepdim=True).clamp_min(1e-8)

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

        expanded_values = token_values[:, None, :]
        iteration_head_log_scales: list[torch.Tensor] = []
        iteration_raw_log_scales: list[torch.Tensor] = []
        iteration_entropies: list[torch.Tensor] = []
        if self.iterative_updates:
            assert self.iterative_reliability is not None
            assert self.iterative_step_logits is not None
            encoded_per_head = (
                encoded[:, None, :, :, :]
                .expand(-1, self.attention_heads, -1, -1, -1)
                .reshape(
                    batch_size * self.attention_heads,
                    encoded.shape[1],
                    *token_size,
                )
            )
            center = token_values.new_full(
                (batch_size, self.attention_heads),
                self.iterative_initial_log_scale,
            )
            step_sizes = torch.sigmoid(self.iterative_step_logits.float())
            frozen_attention: torch.Tensor | None = None
            for iteration_index in range(self.iterative_updates):
                # Detaching the current center only on the reliability path
                # avoids a self-reinforcing shortcut. The robust center update
                # itself remains differentiable through all three rounds.
                reliability_residual = expanded_values - center.detach()[:, :, None]
                analytic_confidence = torch.rsqrt(
                    1.0 + (reliability_residual / self.huber_delta).square()
                )
                dynamic_features = torch.cat(
                    (
                        encoded_per_head,
                        reliability_residual.reshape(
                            batch_size * self.attention_heads,
                            1,
                            *token_size,
                        ),
                        reliability_residual.abs().reshape(
                            batch_size * self.attention_heads,
                            1,
                            *token_size,
                        ),
                        analytic_confidence.reshape(
                            batch_size * self.attention_heads,
                            1,
                            *token_size,
                        ),
                    ),
                    dim=1,
                )
                if frozen_attention is None or self.iterative_refresh_attention:
                    dynamic_logits = self.iterative_reliability(dynamic_features).reshape(
                        batch_size,
                        self.attention_heads,
                        token_count,
                    )
                    attention = normalized_attention(
                        static_logits + dynamic_logits.float()
                    )
                    if not self.iterative_refresh_attention:
                        # Matched static-attention control: compute reliability
                        # once at z^(0), then reuse the exact normalized weights
                        # for every subsequent robust center update.  Parameter
                        # count, initialization, update count, damping, and
                        # supervision stay identical to the refreshed variant.
                        frozen_attention = attention
                else:
                    attention = frozen_attention

                residual = expanded_values - center[:, :, None]
                robust_weight = torch.rsqrt(
                    1.0 + (residual / self.huber_delta).square()
                )
                effective_weight = attention * robust_weight
                proposed_center = (
                    (effective_weight * expanded_values).sum(dim=-1)
                    / effective_weight.sum(dim=-1).clamp_min(1e-8)
                )
                center_delta = self.iterative_max_log_update * torch.tanh(
                    (proposed_center - center) / self.iterative_max_log_update
                )
                center = center + step_sizes[iteration_index] * center_delta
                iteration_head_log_scales.append(center)
                iteration_raw_log_scales.append(
                    (head_mixture * center).sum(dim=-1, keepdim=True)
                )
                positive_iteration_attention = attention.clamp_min(1e-12)
                iteration_entropies.append(
                    -(attention * positive_iteration_attention.log()).sum(dim=-1)
                )
        else:
            attention = normalized_attention(static_logits)
            # Legacy fixed-attention path retained bit-for-bit for published
            # checkpoints. Only new configs enable the recurrent estimator.
            center = fallback_log_scale.flatten(1).float().expand(
                -1,
                self.attention_heads,
            )
            for _ in range(2):
                residual = expanded_values - center[:, :, None]
                robust_weight = torch.rsqrt(1.0 + (residual / self.huber_delta).square())
                effective_weight = attention * robust_weight
                center = (
                    (effective_weight * expanded_values).sum(dim=-1)
                    / effective_weight.sum(dim=-1).clamp_min(1e-8)
                )
            iteration_head_log_scales.append(center)
            iteration_raw_log_scales.append(
                (head_mixture * center).sum(dim=-1, keepdim=True)
            )

        head_log_scale = iteration_head_log_scales[-1]
        raw_attentive_log_scale = iteration_raw_log_scales[-1]
        if self.scale_residual_mlp is None:
            log_scale_residual = torch.zeros_like(raw_attentive_log_scale)
        else:
            log_scale_residual = self.bounded_log_scale_residual * torch.tanh(
                self.scale_residual_mlp(condition_embedding).float()
            )
        support_fraction = valid_pixels.float().mean(dim=(-3, -2, -1), keepdim=False)[:, None]
        deterministic_fallback_flat = fallback_log_scale.flatten(1).float()
        fallback_flat = (
            torch.full_like(
                deterministic_fallback_flat,
                self.iterative_initial_log_scale,
            )
            if self.iterative_updates
            else deterministic_fallback_flat
        )
        sufficient = (pixel_support >= self.min_support).float()[:, None]
        raw_iteration_stack = torch.stack(iteration_raw_log_scales, dim=1)
        attentive_iteration_stack = raw_iteration_stack + log_scale_residual[:, None, :]
        iteration_count = attentive_iteration_stack.shape[1]
        iteration_gate_features = torch.cat(
            (
                condition_embedding.float()[:, None, :].expand(-1, iteration_count, -1),
                support_fraction[:, None, :].expand(-1, iteration_count, -1),
                (
                    attentive_iteration_stack
                    - fallback_flat[:, None, :]
                ).abs(),
            ),
            dim=-1,
        )
        if self.fallback_gate is None:
            # With z^(0)=0, an unsupported recurrence stays exactly at raw DA3
            # scale. No learned interpolation is needed, and supported frames
            # expose the estimator output without attenuation.
            iteration_fallback_gate = torch.ones_like(attentive_iteration_stack)
            iteration_log_scale = attentive_iteration_stack
        else:
            learned_iteration_gate = torch.sigmoid(
                self.fallback_gate(iteration_gate_features).float()
            )
            iteration_fallback_gate = learned_iteration_gate * sufficient[:, None, :]
            iteration_log_scale = (
                iteration_fallback_gate * attentive_iteration_stack
                + (1.0 - iteration_fallback_gate) * fallback_flat[:, None, :]
            )
        attentive_log_scale = attentive_iteration_stack[:, -1]
        fallback_gate = iteration_fallback_gate[:, -1]
        log_scale = iteration_log_scale[:, -1]

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

        result = {
            "scale": log_scale.exp().view(-1, 1, 1, 1),
            "log_scale": log_scale.view(-1, 1, 1, 1),
            "attentive_log_scale": attentive_log_scale.view(-1, 1, 1, 1),
            "raw_attentive_log_scale": raw_attentive_log_scale.view(-1, 1, 1, 1),
            "bounded_log_scale_residual": log_scale_residual.view(-1, 1, 1, 1),
            "fallback_log_scale": fallback_flat.view(-1, 1, 1, 1),
            "deterministic_fallback_log_scale": deterministic_fallback_flat.view(
                -1, 1, 1, 1
            ),
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
        if self.iterative_updates:
            valid_token_count = token_mask.sum(dim=-1).float()
            entropy_denominator = valid_token_count.clamp_min(2.0).log()[:, None, None]
            iteration_entropy = torch.stack(iteration_entropies, dim=1)
            normalized_iteration_entropy = torch.where(
                valid_token_count[:, None, None] > 1,
                iteration_entropy / entropy_denominator,
                torch.ones_like(iteration_entropy),
            )
            result.update(
                {
                    "iteration_log_scales": iteration_log_scale.unsqueeze(-1),
                    "iteration_raw_log_scales": raw_iteration_stack.unsqueeze(-1),
                    "iteration_head_log_scales": torch.stack(
                        iteration_head_log_scales,
                        dim=1,
                    ),
                    "iteration_fallback_gates": iteration_fallback_gate.unsqueeze(-1),
                    "iteration_step_sizes": torch.sigmoid(
                        self.iterative_step_logits.float()
                    ),
                    "iteration_normalized_attention_entropy": (
                        normalized_iteration_entropy.mean(dim=-1)
                    ),
                }
            )
        return result
