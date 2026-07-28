from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, ResidualBlock, UpBlock


class ConditionalDepthRefiner(nn.Module):
    """Lightweight zero-initialized U-Net that predicts log-depth residual and variance."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        max_log_residual: float,
        gated: bool = False,
        initial_gate_bias: float = -2.5,
        frame_residual: bool = False,
    ) -> None:
        super().__init__()
        self.max_log_residual = max_log_residual
        self.gated = gated
        self.frame_residual = frame_residual
        channels = base_channels
        self.stem = nn.Sequential(ConvNormAct(in_channels, channels), ResidualBlock(channels))
        self.down1 = nn.Sequential(
            ConvNormAct(channels, channels * 2, stride=2), ResidualBlock(channels * 2)
        )
        self.down2 = nn.Sequential(
            ConvNormAct(channels * 2, channels * 4, stride=2),
            ResidualBlock(channels * 4),
        )
        self.down3 = nn.Sequential(
            ConvNormAct(channels * 4, channels * 8, stride=2),
            ResidualBlock(channels * 8),
        )
        self.up2 = UpBlock(channels * 8, channels * 4, channels * 4)
        self.up1 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.up0 = UpBlock(channels * 2, channels, channels)
        self.output = nn.Conv2d(channels, 3 if gated else 2, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if gated:
            with torch.no_grad():
                self.output.bias[2] = initial_gate_bias
        if frame_residual:
            self.frame_output = nn.Linear(channels * 8, 1)
            nn.init.zeros_(self.frame_output.weight)
            nn.init.zeros_(self.frame_output.bias)

    def forward(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        level0 = self.stem(inputs)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        bottleneck = self.down3(level2)
        decoded = self.up0(self.up1(self.up2(bottleneck, level2), level1), level0)
        prediction = self.output(decoded)
        residual = torch.tanh(prediction[:, :1]) * self.max_log_residual
        log_variance = prediction[:, 1:2].clamp(-6.0, 4.0)
        if self.gated:
            if self.frame_residual:
                pooled = torch.mean(bottleneck, dim=(-2, -1))
                frame = (
                    torch.tanh(self.frame_output(pooled)).view(-1, 1, 1, 1)
                    * self.max_log_residual
                )
                return residual, log_variance, prediction[:, 2:3], frame
            return residual, log_variance, prediction[:, 2:3]
        return residual, log_variance
