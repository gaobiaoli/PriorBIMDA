from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .blocks import ConvNormAct, ResidualBlock, UpBlock


class WindowedBIMCrossAttention(nn.Module):
    """Inject locally matched BIM/DA3 context into an aligned image query.

    Keys are masked by the downsampled BIM ray-hit map. A window without any
    valid BIM key returns an exact zero delta, preserving the CNN fallback.
    """

    def __init__(
        self,
        query_channels: int,
        context_channels: int,
        attention_channels: int,
        attention_heads: int,
        window_size: int,
        layer_scale_init: float,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if query_channels < 1 or context_channels < 1 or attention_channels < 1:
            raise ValueError("Cross-attention channel counts must be positive")
        if attention_heads < 1 or attention_channels % attention_heads:
            raise ValueError("attention_channels must be divisible by positive attention_heads")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 0.0 <= attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if layer_scale_init < 0:
            raise ValueError("layer_scale_init must be non-negative")

        self.query_channels = int(query_channels)
        self.context_channels = int(context_channels)
        self.attention_channels = int(attention_channels)
        self.attention_heads = int(attention_heads)
        self.window_size = int(window_size)
        self.head_channels = self.attention_channels // self.attention_heads
        self.scale = self.head_channels**-0.5

        self.query_norm = nn.LayerNorm(self.query_channels)
        self.context_norm = nn.LayerNorm(self.context_channels)
        self.query_projection = nn.Linear(self.query_channels, self.attention_channels)
        self.key_value_projection = nn.Linear(
            self.context_channels,
            self.attention_channels * 2,
        )
        self.output_projection = nn.Linear(self.attention_channels, self.query_channels)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.layer_scale = nn.Parameter(torch.full((self.query_channels,), float(layer_scale_init)))

        relative_axis = torch.arange(self.window_size)
        coordinates = torch.stack(
            torch.meshgrid(relative_axis, relative_axis, indexing="ij")
        ).flatten(1)
        relative = coordinates[:, :, None] - coordinates[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[..., 0] += self.window_size - 1
        relative[..., 1] += self.window_size - 1
        relative[..., 0] *= 2 * self.window_size - 1
        relative_position_index = relative.sum(dim=-1)
        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=False,
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(
                self.attention_heads,
                (2 * self.window_size - 1) ** 2,
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

    def _partition(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = tensor.shape
        window = self.window_size
        return (
            tensor.view(
                batch,
                height // window,
                window,
                width // window,
                window,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, window * window, channels)
        )

    def _reverse(
        self,
        windows: torch.Tensor,
        *,
        batch: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        window = self.window_size
        channels = windows.shape[-1]
        return (
            windows.view(
                batch,
                height // window,
                width // window,
                window,
                window,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, height, width, channels)
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        bim_valid: torch.Tensor,
    ) -> torch.Tensor:
        if query.ndim != 4 or context.ndim != 4 or bim_valid.ndim != 4:
            raise ValueError("Cross-attention inputs must have shape [B, C, H, W]")
        if query.shape[0] != context.shape[0] or query.shape[-2:] != context.shape[-2:]:
            raise ValueError("Cross-attention query/context geometry must match")
        if bim_valid.shape != (query.shape[0], 1, *query.shape[-2:]):
            raise ValueError("bim_valid must have shape [B, 1, H, W]")
        if query.shape[1] != self.query_channels:
            raise ValueError("Unexpected cross-attention query channel count")
        if context.shape[1] != self.context_channels:
            raise ValueError("Unexpected cross-attention context channel count")

        batch, _, original_height, original_width = query.shape
        window = self.window_size
        pad_height = (-original_height) % window
        pad_width = (-original_width) % window
        query = functional.pad(query, (0, pad_width, 0, pad_height))
        context = functional.pad(context, (0, pad_width, 0, pad_height))
        bim_valid = functional.pad(bim_valid, (0, pad_width, 0, pad_height))
        height, width = query.shape[-2:]

        query_windows = self._partition(query.permute(0, 2, 3, 1).contiguous())
        context_windows = self._partition(context.permute(0, 2, 3, 1).contiguous())
        valid_windows = self._partition((bim_valid > 0).permute(0, 2, 3, 1).contiguous())[..., 0]
        window_has_valid = valid_windows.any(dim=1, keepdim=True)

        query_tokens = self.query_projection(self.query_norm(query_windows))
        key_value = self.key_value_projection(self.context_norm(context_windows))
        key_tokens, value_tokens = key_value.chunk(2, dim=-1)
        token_count = window * window
        query_tokens = query_tokens.view(
            -1,
            token_count,
            self.attention_heads,
            self.head_channels,
        ).transpose(1, 2)
        key_tokens = key_tokens.view(
            -1,
            token_count,
            self.attention_heads,
            self.head_channels,
        ).transpose(1, 2)
        value_tokens = value_tokens.view(
            -1,
            token_count,
            self.attention_heads,
            self.head_channels,
        ).transpose(1, 2)

        scores = (query_tokens * self.scale) @ key_tokens.transpose(-2, -1)
        relative_bias = self.relative_position_bias[
            :, self.relative_position_index.reshape(-1)
        ].view(self.attention_heads, token_count, token_count)
        scores = scores + relative_bias.unsqueeze(0).to(dtype=scores.dtype)
        scores = scores.masked_fill(
            ~valid_windows[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        attention = functional.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attention = self.attention_dropout(attention)
        attended = (
            (attention @ value_tokens)
            .transpose(1, 2)
            .reshape(
                -1,
                token_count,
                self.attention_channels,
            )
        )
        attended = self.output_projection(attended)
        attended = attended * window_has_valid[..., None].to(dtype=attended.dtype)
        delta = self._reverse(
            attended,
            batch=batch,
            height=height,
            width=width,
        )
        delta = delta[:, :original_height, :original_width]
        delta = delta.permute(0, 3, 1, 2).contiguous()
        return delta * self.layer_scale.view(1, -1, 1, 1)


class _ConditionPyramid(nn.Module):
    """Encode one condition source without mixing its input statistics."""

    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        channels = base_channels
        self.level0 = nn.Sequential(
            ConvNormAct(in_channels, channels),
            ResidualBlock(channels),
        )
        self.level1 = nn.Sequential(
            ConvNormAct(channels, channels * 2, stride=2),
            ResidualBlock(channels * 2),
        )
        self.level2 = nn.Sequential(
            ConvNormAct(channels * 2, channels * 4, stride=2),
            ResidualBlock(channels * 4),
        )
        self.level3 = nn.Sequential(
            ConvNormAct(channels * 4, channels * 8, stride=2),
            ResidualBlock(channels * 8),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        level0 = self.level0(inputs)
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        level3 = self.level3(level2)
        return level0, level1, level2, level3


class ScaleAnchoredDepthRefiner(nn.Module):
    """Multi-condition refiner anchored to scale-corrected DA3.

    RGB, DA3 geometry, and raw BIM priors are encoded independently. BIM
    features enter through zero-initialized adapters so the initial model is an
    exact scale-only baseline. The three residual heads are deliberately not
    multiplied by a trust gate: BIM reliability is an auxiliary task rather
    than a second gradient bottleneck.
    """

    def __init__(
        self,
        rgb_channels: int,
        geometry_channels: int,
        bim_channels: int,
        base_channels: int,
        max_frame_log_residual: float,
        max_low_log_residual: float,
        max_detail_log_residual: float,
        max_total_log_residual: float,
        gate_bim_adapters: bool = False,
        bim_adapter_gate_floor: float = 0.25,
        bim_adapter_gate_use_rgb: bool = False,
        da3_feature_channels: int = 0,
        additive_residual_enabled: bool = False,
        max_additive_residual_m: float = 0.0,
        additive_detach_shared_features: bool = True,
        bim_cross_attention_enabled: bool = False,
        bim_cross_attention_channels: int = 128,
        bim_cross_attention_heads: int = 4,
        bim_cross_attention_window_size: int = 7,
        bim_cross_attention_layer_scale_init: float = 1e-3,
        bim_cross_attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = base_channels
        self.max_frame_log_residual = max_frame_log_residual
        self.max_low_log_residual = max_low_log_residual
        self.max_detail_log_residual = max_detail_log_residual
        self.max_total_log_residual = max_total_log_residual
        self.gate_bim_adapters = gate_bim_adapters
        self.bim_adapter_gate_floor = bim_adapter_gate_floor
        self.bim_adapter_gate_use_rgb = bool(bim_adapter_gate_use_rgb)
        self.da3_feature_channels = int(da3_feature_channels)
        self.additive_residual_enabled = bool(additive_residual_enabled)
        self.max_additive_residual_m = float(max_additive_residual_m)
        self.additive_detach_shared_features = bool(additive_detach_shared_features)
        self.bim_cross_attention_enabled = bool(bim_cross_attention_enabled)
        if self.da3_feature_channels < 0:
            raise ValueError("da3_feature_channels must be non-negative")
        if not 0.0 <= self.bim_adapter_gate_floor < 1.0:
            raise ValueError("bim_adapter_gate_floor must be in [0, 1)")
        if self.additive_residual_enabled and self.max_additive_residual_m <= 0:
            raise ValueError(
                "max_additive_residual_m must be positive when additive residual is enabled"
            )
        if self.bim_cross_attention_enabled and self.da3_feature_channels < 1:
            raise ValueError("BIM cross-attention requires cached DA3 refiner features")

        self.rgb_encoder = _ConditionPyramid(rgb_channels, channels)
        self.geometry_encoder = _ConditionPyramid(geometry_channels, channels)
        self.bim_encoder = _ConditionPyramid(bim_channels, channels)

        widths = (channels, channels * 2, channels * 4, channels * 8)
        self.main_fusion = nn.ModuleList(
            [
                nn.Sequential(
                    ConvNormAct(width * 2, width),
                    ResidualBlock(width),
                )
                for width in widths
            ]
        )
        self.bim_adapters = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for width in widths]
        )
        for adapter in self.bim_adapters:
            nn.init.zeros_(adapter.weight)
            nn.init.zeros_(adapter.bias)
        self.da3_mid_projection: nn.Sequential | None = None
        self.da3_deep_projection: nn.Sequential | None = None
        self.da3_level2_fusion: nn.Sequential | None = None
        self.da3_level3_fusion: nn.Sequential | None = None
        if self.da3_feature_channels:

            def projection(width: int) -> nn.Sequential:
                return nn.Sequential(
                    nn.Conv2d(self.da3_feature_channels, width, kernel_size=1),
                    nn.GroupNorm(min(8, width), width),
                    nn.SiLU(inplace=True),
                )

            self.da3_mid_projection = projection(widths[2])
            self.da3_deep_projection = projection(widths[3])
            self.da3_level2_fusion = nn.Sequential(
                ConvNormAct(widths[2] * 2, widths[2]),
                ResidualBlock(widths[2]),
            )
            self.da3_level3_fusion = nn.Sequential(
                ConvNormAct(widths[3] * 2, widths[3]),
                ResidualBlock(widths[3]),
            )
        self.up2 = UpBlock(channels * 8, channels * 4, channels * 4)
        self.up1 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.up0 = UpBlock(channels * 2, channels, channels)

        self.low_output = nn.Conv2d(channels * 8, 1, kernel_size=1)
        self.detail_output = nn.Conv2d(channels, 3, kernel_size=1)
        self.frame_output = nn.Linear(channels * 8, 2)
        for head in (self.low_output, self.detail_output, self.frame_output):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.additive_body: nn.Sequential | None = None
        self.additive_output: nn.Conv2d | None = None
        if self.additive_residual_enabled:
            # The head sees the decoded image feature, current metric anchor,
            # and already predicted proportional residual.  Its final layer is
            # zero initialized, so enabling the branch is an exact no-op before
            # its dedicated training stage.
            self.additive_body = nn.Sequential(
                ConvNormAct(channels + 2, channels),
                ResidualBlock(channels),
            )
            self.additive_output = nn.Conv2d(channels, 1, kernel_size=1)
            nn.init.zeros_(self.additive_output.weight)
            nn.init.zeros_(self.additive_output.bias)
        # Construct optional ablation-only modules after every shared module so
        # V5 and V6 receive identical common weights under the same seed.
        if self.gate_bim_adapters:
            gate_input_multiplier = 3 if self.bim_adapter_gate_use_rgb else 2
            self.bim_adapter_gates = nn.ModuleList(
                [nn.Conv2d(width * gate_input_multiplier, 1, kernel_size=1) for width in widths]
            )
            for gate in self.bim_adapter_gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)
        # Construct the ablation module after every shared tensor so enabling
        # it cannot shift the seeded initialization of the baseline network.
        self.bim_cross_attention: WindowedBIMCrossAttention | None = None
        if self.bim_cross_attention_enabled:
            self.bim_cross_attention = WindowedBIMCrossAttention(
                query_channels=widths[3],
                context_channels=widths[3] * 2,
                attention_channels=int(bim_cross_attention_channels),
                attention_heads=int(bim_cross_attention_heads),
                window_size=int(bim_cross_attention_window_size),
                layer_scale_init=float(bim_cross_attention_layer_scale_init),
                attention_dropout=float(bim_cross_attention_dropout),
            )

    @torch.no_grad()
    def zero_multiplicative_residual_heads(self) -> dict[str, object]:
        """Zero only output slices that contribute to the depth residual.

        The auxiliary uncertainty and BIM-reliability channels share final
        layers with the detail/frame residuals.  This method deliberately
        addresses exact module attributes and channel slices so a target-domain
        reset cannot silently erase those auxiliary source heads.
        """

        if self.low_output.out_channels != 1:
            raise RuntimeError("low_output must expose exactly one residual channel")
        if self.detail_output.out_channels != 3:
            raise RuntimeError("detail_output must expose residual/variance/reliability channels")
        if self.frame_output.out_features != 2:
            raise RuntimeError("frame_output must expose residual and trust channels")
        targets = (
            ("refiner.low_output.weight", self.low_output.weight, "all"),
            ("refiner.low_output.bias", self.low_output.bias, "all"),
            (
                "refiner.detail_output.weight",
                self.detail_output.weight[0:1],
                "output_channel_0",
            ),
            (
                "refiner.detail_output.bias",
                self.detail_output.bias[0:1],
                "output_channel_0",
            ),
            (
                "refiner.frame_output.weight",
                self.frame_output.weight[0:1],
                "output_feature_0",
            ),
            (
                "refiner.frame_output.bias",
                self.frame_output.bias[0:1],
                "output_feature_0",
            ),
        )
        receipts = []
        for name, values, selected_slice in targets:
            before_nonzero = int(torch.count_nonzero(values).item())
            elements = int(values.numel())
            values.zero_()
            after_nonzero = int(torch.count_nonzero(values).item())
            if after_nonzero:
                raise RuntimeError(f"Failed to zero residual output slice {name}")
            receipts.append(
                {
                    "parameter": name,
                    "slice": selected_slice,
                    "elements": elements,
                    "nonzero_before": before_nonzero,
                    "nonzero_after": after_nonzero,
                }
            )
        return {
            "operation": "zero_multiplicative_residual_heads",
            "targets": receipts,
            "zeroed_elements": sum(int(receipt["elements"]) for receipt in receipts),
        }

    def forward(
        self,
        rgb: torch.Tensor,
        geometry: torch.Tensor,
        bim: torch.Tensor,
        da3_feature_mid: torch.Tensor | None = None,
        da3_feature_deep: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rgb_levels = self.rgb_encoder(rgb)
        geometry_levels = self.geometry_encoder(geometry)
        bim_levels = self.bim_encoder(bim)
        main_levels = tuple(
            main(torch.cat((rgb_level, geometry_level), dim=1))
            for main, rgb_level, geometry_level in zip(
                self.main_fusion,
                rgb_levels,
                geometry_levels,
            )
        )
        adapter_gate_logits = None
        if self.gate_bim_adapters:
            valid = bim[:, 1:2].clamp(0.0, 1.0)
            fused = []
            gate_logits = []
            for main_level, adapter, gate, rgb_level, geometry_level, bim_level in zip(
                main_levels,
                self.bim_adapters,
                self.bim_adapter_gates,
                rgb_levels,
                geometry_levels,
                bim_levels,
            ):
                gate_features = (
                    torch.cat((rgb_level, geometry_level, bim_level), dim=1)
                    if self.bim_adapter_gate_use_rgb
                    else torch.cat((geometry_level, bim_level), dim=1)
                )
                logits = gate(gate_features)
                valid_level = nn.functional.interpolate(
                    valid,
                    size=logits.shape[-2:],
                    mode="nearest",
                )
                probability = self.bim_adapter_gate_floor + (
                    1.0 - self.bim_adapter_gate_floor
                ) * torch.sigmoid(logits)
                fused.append(main_level + valid_level * probability * adapter(bim_level))
                gate_logits.append(logits)
            fused_levels = tuple(fused)
            adapter_gate_logits = gate_logits[0]
        else:
            fused_levels = tuple(
                main_level + adapter(bim_level)
                for main_level, adapter, bim_level in zip(
                    main_levels,
                    self.bim_adapters,
                    bim_levels,
                )
            )

        level0, level1, level2, bottleneck = fused_levels
        deep: torch.Tensor | None = None
        if self.da3_feature_channels:
            if da3_feature_mid is None or da3_feature_deep is None:
                raise ValueError("DA3-enabled refiner requires mid and deep features")
            expected_prefix = (rgb.shape[0], self.da3_feature_channels)
            if da3_feature_mid.shape[:2] != expected_prefix:
                raise ValueError(
                    f"DA3 mid feature must start with {expected_prefix}; "
                    f"got {tuple(da3_feature_mid.shape)}"
                )
            if da3_feature_deep.shape[:2] != expected_prefix:
                raise ValueError(
                    f"DA3 deep feature must start with {expected_prefix}; "
                    f"got {tuple(da3_feature_deep.shape)}"
                )
            assert self.da3_mid_projection is not None
            assert self.da3_deep_projection is not None
            assert self.da3_level2_fusion is not None
            assert self.da3_level3_fusion is not None
            mid = self.da3_mid_projection(da3_feature_mid.to(dtype=level2.dtype))
            mid = nn.functional.interpolate(
                mid,
                size=level2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            deep = self.da3_deep_projection(da3_feature_deep.to(dtype=bottleneck.dtype))
            deep = nn.functional.interpolate(
                deep,
                size=bottleneck.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            level2 = self.da3_level2_fusion(torch.cat((level2, mid), dim=1))
            bottleneck = self.da3_level3_fusion(torch.cat((bottleneck, deep), dim=1))
        if self.bim_cross_attention is not None:
            if deep is None:
                raise RuntimeError("BIM cross-attention did not receive DA3 deep features")
            bim_valid_bottleneck = functional.adaptive_max_pool2d(
                bim[:, 1:2].clamp(0.0, 1.0),
                bottleneck.shape[-2:],
            )
            cross_context = torch.cat((bim_levels[3], deep), dim=1)
            cross_delta = self.bim_cross_attention(
                main_levels[3],
                cross_context,
                bim_valid_bottleneck,
            )
            bottleneck = bottleneck + cross_delta
        decoded = self.up0(
            self.up1(self.up2(bottleneck, level2), level1),
            level0,
        )
        detail_prediction = self.detail_output(decoded)

        detail_residual = torch.tanh(detail_prediction[:, :1]) * self.max_detail_log_residual
        low_residual = torch.tanh(self.low_output(bottleneck))
        low_residual = nn.functional.interpolate(
            low_residual,
            size=detail_residual.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_residual = low_residual * self.max_low_log_residual

        pooled = torch.mean(bottleneck, dim=(-2, -1))
        frame_prediction = self.frame_output(pooled)
        frame_residual = (
            torch.tanh(frame_prediction[:, :1]).view(-1, 1, 1, 1) * self.max_frame_log_residual
        )
        frame_trust_logits = frame_prediction[:, 1:2].view(-1, 1, 1, 1)

        log_residual = torch.clamp(
            frame_residual + low_residual + detail_residual,
            -self.max_total_log_residual,
            self.max_total_log_residual,
        )
        output = {
            "log_residual": log_residual,
            "frame_log_residual": frame_residual,
            "low_log_residual": low_residual,
            "detail_log_residual": detail_residual,
            "log_variance": detail_prediction[:, 1:2].clamp(-6.0, 4.0),
            "bim_reliability_logits": detail_prediction[:, 2:3],
            "frame_trust_logits": frame_trust_logits,
        }
        if adapter_gate_logits is not None:
            output["bim_adapter_gate_logits"] = adapter_gate_logits
            output["bim_adapter_gate_logits_pyramid"] = tuple(gate_logits)
        if self.additive_residual_enabled:
            output["decoded_features_for_additive"] = decoded
        return output

    def predict_additive_metric_residual(
        self,
        decoded: torch.Tensor,
        metric_anchor: torch.Tensor,
        proportional_log_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Predict a bounded pixel-level correction in metres.

        Detaching the shared inputs keeps additive losses from rewriting the
        proportional backbone during the dedicated/additive part of joint
        training.  The additive head itself remains fully differentiable.
        """

        if not self.additive_residual_enabled:
            return torch.zeros_like(metric_anchor)
        if decoded.ndim != 4 or metric_anchor.shape != proportional_log_residual.shape:
            raise ValueError("Additive residual inputs have incompatible shapes")
        if (
            decoded.shape[0] != metric_anchor.shape[0]
            or decoded.shape[-2:] != metric_anchor.shape[-2:]
        ):
            raise ValueError("Decoded/additive metric tensors must share batch and spatial shape")
        assert self.additive_body is not None
        assert self.additive_output is not None
        if self.additive_detach_shared_features:
            decoded = decoded.detach()
            metric_anchor = metric_anchor.detach()
            proportional_log_residual = proportional_log_residual.detach()
        log_anchor = torch.log(metric_anchor.clamp_min(1e-3)) / 3.0
        inputs = torch.cat((decoded, log_anchor, proportional_log_residual), dim=1)
        logits = self.additive_output(self.additive_body(inputs))
        return torch.tanh(logits) * self.max_additive_residual_m
