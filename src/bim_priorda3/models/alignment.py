from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn


class RobustLocalAffineAlignment(nn.Module):
    """Differentiable local weighted affine fit inspired by Prior Depth Anything."""

    def __init__(
        self,
        kernel_size: int = 31,
        downsample: int = 4,
        huber_delta: float = 0.08,
        min_support: float = 0.02,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("alignment kernel_size must be odd")
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.huber_delta = huber_delta
        self.min_support = min_support
        self.scale_range = scale_range

    def _pool(self, value: torch.Tensor) -> torch.Tensor:
        return functional.avg_pool2d(
            value,
            self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
        )

    def _fit(
        self, base: torch.Tensor, bim: torch.Tensor, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sum_w = self._pool(weight)
        mean_x = self._pool(weight * base) / (sum_w + 1e-6)
        mean_y = self._pool(weight * bim) / (sum_w + 1e-6)
        covariance = self._pool(weight * base * bim) / (sum_w + 1e-6) - mean_x * mean_y
        variance = self._pool(weight * base.square()) / (sum_w + 1e-6) - mean_x.square()
        scale = covariance / (variance + 1e-5)
        scale = scale.clamp(*self.scale_range)
        shift = mean_y - scale * mean_x
        return scale, shift, sum_w

    def forward(
        self,
        base_depth: torch.Tensor,
        bim_depth: torch.Tensor,
        bim_valid: torch.Tensor,
        trust: torch.Tensor,
        bim_edge: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = base_depth.shape[-2:]
        small_shape = tuple(max(1, size // self.downsample) for size in shape)
        resize = lambda value: functional.interpolate(
            value, size=small_shape, mode="bilinear", align_corners=False
        )
        base = resize(base_depth)
        bim = resize(bim_depth)
        valid = functional.interpolate(bim_valid, size=small_shape, mode="nearest")
        edge = functional.interpolate(bim_edge, size=small_shape, mode="nearest")
        probability = resize(trust)
        weight = probability * valid * (1.0 - edge).clamp(0.0, 1.0)

        scale, shift, support = self._fit(base, bim, weight)
        residual = torch.abs(bim - (scale * base + shift))
        robust_weight = torch.clamp(self.huber_delta / (residual + 1e-6), max=1.0)
        scale, shift, support = self._fit(base, bim, weight * robust_weight)

        scale = functional.interpolate(scale, size=shape, mode="bilinear", align_corners=False)
        shift = functional.interpolate(shift, size=shape, mode="bilinear", align_corners=False)
        support = functional.interpolate(support, size=shape, mode="bilinear", align_corners=False)
        has_support = support >= self.min_support
        coarse = torch.where(has_support, scale * base_depth + shift, base_depth)
        coarse = coarse.clamp_min(1e-3)
        return coarse, support, scale, shift

