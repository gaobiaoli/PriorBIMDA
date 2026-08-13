from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, ResidualBlock, UpBlock


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
    ) -> None:
        super().__init__()
        channels = base_channels
        self.max_frame_log_residual = max_frame_log_residual
        self.max_low_log_residual = max_low_log_residual
        self.max_detail_log_residual = max_detail_log_residual
        self.max_total_log_residual = max_total_log_residual
        self.gate_bim_adapters = gate_bim_adapters
        self.bim_adapter_gate_floor = bim_adapter_gate_floor

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
        self.up2 = UpBlock(channels * 8, channels * 4, channels * 4)
        self.up1 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.up0 = UpBlock(channels * 2, channels, channels)

        self.low_output = nn.Conv2d(channels * 8, 1, kernel_size=1)
        self.detail_output = nn.Conv2d(channels, 3, kernel_size=1)
        self.frame_output = nn.Linear(channels * 8, 2)
        for head in (self.low_output, self.detail_output, self.frame_output):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Construct optional ablation-only modules after every shared module so
        # V5 and V6 receive identical common weights under the same seed.
        if self.gate_bim_adapters:
            self.bim_adapter_gates = nn.ModuleList(
                [nn.Conv2d(width * 2, 1, kernel_size=1) for width in widths]
            )
            for gate in self.bim_adapter_gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

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
    ) -> dict[str, torch.Tensor]:
        rgb_levels = self.rgb_encoder(rgb)
        geometry_levels = self.geometry_encoder(geometry)
        bim_levels = self.bim_encoder(bim)
        adapter_gate_logits = None
        if self.gate_bim_adapters:
            valid = bim[:, 1:2].clamp(0.0, 1.0)
            fused = []
            gate_logits = []
            for main, adapter, gate, rgb_level, geometry_level, bim_level in zip(
                self.main_fusion,
                self.bim_adapters,
                self.bim_adapter_gates,
                rgb_levels,
                geometry_levels,
                bim_levels,
            ):
                logits = gate(torch.cat((geometry_level, bim_level), dim=1))
                valid_level = nn.functional.interpolate(
                    valid,
                    size=logits.shape[-2:],
                    mode="nearest",
                )
                probability = self.bim_adapter_gate_floor + (
                    1.0 - self.bim_adapter_gate_floor
                ) * torch.sigmoid(logits)
                fused.append(
                    main(torch.cat((rgb_level, geometry_level), dim=1))
                    + valid_level * probability * adapter(bim_level)
                )
                gate_logits.append(logits)
            fused_levels = tuple(fused)
            adapter_gate_logits = gate_logits[0]
        else:
            fused_levels = tuple(
                main(torch.cat((rgb_level, geometry_level), dim=1)) + adapter(bim_level)
                for main, adapter, rgb_level, geometry_level, bim_level in zip(
                    self.main_fusion,
                    self.bim_adapters,
                    rgb_levels,
                    geometry_levels,
                    bim_levels,
                )
            )

        level0, level1, level2, bottleneck = fused_levels
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
        return output
