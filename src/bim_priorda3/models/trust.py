from __future__ import annotations

import torch
from torch import nn

from .blocks import ConvNormAct, ResidualBlock, UpBlock


class BIMTrustNet(nn.Module):
    """Learn when BIM is safer than the base DA3 prediction at each pixel."""

    def __init__(
        self,
        in_channels: int = 12,
        channels: int = 32,
        initial_bias: float | None = None,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(ConvNormAct(in_channels, channels), ResidualBlock(channels))
        self.down1 = nn.Sequential(
            ConvNormAct(channels, channels * 2, stride=2), ResidualBlock(channels * 2)
        )
        self.down2 = nn.Sequential(
            ConvNormAct(channels * 2, channels * 4, stride=2),
            ResidualBlock(channels * 4),
        )
        self.up1 = UpBlock(channels * 4, channels * 2, channels * 2)
        self.up2 = UpBlock(channels * 2, channels, channels)
        self.output = nn.Conv2d(channels, 1, 1)
        self.frame_output = nn.Linear(channels * 4, 1)
        if initial_bias is not None:
            nn.init.zeros_(self.output.weight)
            nn.init.constant_(self.output.bias, initial_bias * 0.5)
            nn.init.zeros_(self.frame_output.weight)
            nn.init.constant_(self.frame_output.bias, initial_bias * 0.5)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        level0 = self.stem(inputs)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        pixel_logits = self.output(self.up2(self.up1(level2, level1), level0))
        pooled = torch.mean(level2, dim=(-2, -1))
        frame_logits = self.frame_output(pooled).view(-1, 1, 1, 1)
        return pixel_logits, frame_logits
