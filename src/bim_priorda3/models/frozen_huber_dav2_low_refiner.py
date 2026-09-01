from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from bim_priorda3.config import Config

from .bim_early_fusion_dav2 import (
    BIMEarlyFusionDepthAnythingV2,
    build_bim_condition,
)
from .system import BIMPriorDA3


class BIMEarlyFusionDAv2LowRefiner(BIMEarlyFusionDepthAnythingV2):
    """Decode only ``r_low`` from a top-down fused pretrained DPT feature.

    DPT neck output index 1 is approximately 1/7 input resolution for a
    ViT-B/14 backbone (72x72 at the project's 504x504 input). It is the
    closest fused feature to the historical CNN refiner's 1/8 ``r_low`` grid.
    The metric-depth head is retained solely as part of the pinned official
    checkpoint and is never used by this model.
    """

    LOW_FUSION_INDEX = 1

    def __init__(
        self,
        pretrained_model: nn.Module,
        *,
        max_low_log_residual: float = 0.25,
        output_max_depth_m: float = 128.0,
        low_hidden_channels: int = 64,
    ) -> None:
        super().__init__(pretrained_model)
        if max_low_log_residual <= 0:
            raise ValueError("max_low_log_residual must be positive")
        if output_max_depth_m <= 0:
            raise ValueError("output_max_depth_m must be positive")
        if low_hidden_channels < 1:
            raise ValueError("low_hidden_channels must be positive")
        self.max_low_log_residual = float(max_low_log_residual)
        self.output_max_depth_m = float(output_max_depth_m)
        self.low_output = nn.Sequential(
            nn.Conv2d(128, low_hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(low_hidden_channels, 1, kernel_size=1),
        )
        nn.init.kaiming_normal_(self.low_output[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.low_output[0].bias)
        nn.init.zeros_(self.low_output[-1].weight)
        nn.init.zeros_(self.low_output[-1].bias)
        for parameter in self.dav2.head.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        local_files_only: bool = True,
        max_low_log_residual: float = 0.25,
        output_max_depth_m: float = 128.0,
        low_hidden_channels: int = 64,
    ) -> BIMEarlyFusionDAv2LowRefiner:
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("DAv2 low refiner requires the optional 'dav2' dependency") from error
        pretrained = AutoModelForDepthEstimation.from_pretrained(
            model_name_or_path,
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(
            pretrained,
            max_low_log_residual=max_low_log_residual,
            output_max_depth_m=output_max_depth_m,
            low_hidden_channels=low_hidden_channels,
        )

    def forward(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
        scaled_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor | str | int]:
        if scaled_depth.ndim != 4 or scaled_depth.shape[1] != 1:
            raise ValueError("scaled_depth must have shape [B, 1, H, W]")
        if scaled_depth.shape[0] != rgb.shape[0] or scaled_depth.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB and scaled depth batch/spatial shapes must agree")
        normalized = self.normalized_rgb(rgb)
        embeddings = self._early_embeddings(normalized, bim_condition)
        neck_output, _, _ = self._neck_embeddings(
            embeddings,
            height=normalized.shape[-2],
            width=normalized.shape[-1],
        )
        low_feature = neck_output[self.LOW_FUSION_INDEX]
        raw_low_native = self.low_output(low_feature)
        low_native = self.max_low_log_residual * torch.tanh(raw_low_native)
        low_full = functional.interpolate(
            low_native,
            size=scaled_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        depth = (scaled_depth * torch.exp(low_full.float())).clamp(
            1e-3,
            self.output_max_depth_m,
        )
        return {
            "depth": depth,
            "scaled_depth": scaled_depth,
            "low_log_residual": low_full,
            "low_log_residual_native": low_native,
            "raw_low_log_residual_native": raw_low_native,
            "low_feature": low_feature,
            "low_feature_source": "dav2.dpt_neck.top_down_fusion[1]",
            "low_fusion_index": self.LOW_FUSION_INDEX,
        }

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        decoder_lr: float,
        condition_lr: float,
        low_head_lr: float,
    ) -> list[dict[str, Any]]:
        groups = [
            {
                "name": "dinov2_encoder",
                "params": [p for p in self.dav2.backbone.parameters() if p.requires_grad],
                "lr": encoder_lr,
            },
            {
                "name": "dpt_top_down_decoder",
                "params": [p for p in self.dav2.neck.parameters() if p.requires_grad],
                "lr": decoder_lr,
            },
            {
                "name": "bim_condition_projection",
                "params": list(self.bim_condition_embed.parameters()),
                "lr": condition_lr,
            },
            {
                "name": "low_residual_head",
                "params": list(self.low_output.parameters()),
                "lr": low_head_lr,
            },
        ]
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("Optimizer parameter groups overlap")
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if set(ids) != expected:
            raise RuntimeError("Optimizer groups do not cover exactly the trainable refiner")
        return groups


class FrozenHuberDAv2LowRefiner(nn.Module):
    """Frozen three-round Huber scale followed by a DAv2-DPT ``r_low`` refiner."""

    def __init__(
        self,
        scale_system: BIMPriorDA3,
        refiner: BIMEarlyFusionDAv2LowRefiner,
        *,
        bim_log_mean: float,
        bim_log_std: float,
        disagreement_clip: float = 1.5,
    ) -> None:
        super().__init__()
        if scale_system.attention_scale is None:
            raise ValueError("Frozen scale system must contain an attention scale head")
        self.scale_system = scale_system
        self.refiner = refiner
        self.bim_log_mean = float(bim_log_mean)
        self.bim_log_std = float(bim_log_std)
        self.disagreement_clip = float(disagreement_clip)
        for parameter in self.scale_system.parameters():
            parameter.requires_grad_(False)
        self.scale_system.eval()

    @classmethod
    def from_checkpoints(
        cls,
        cfg: Config,
        *,
        scale_checkpoint: Mapping[str, Any],
    ) -> FrozenHuberDAv2LowRefiner:
        scale_system = BIMPriorDA3(cfg)
        state = scale_checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise TypeError("Scale checkpoint lacks a model state mapping")
        scale_system.load_state_dict(state, strict=True)
        dav2 = cfg.model.dav2
        low = cfg.model.dav2_low_refiner
        refiner = BIMEarlyFusionDAv2LowRefiner.from_pretrained(
            str(dav2.model_id),
            revision=str(dav2.revision),
            local_files_only=bool(dav2.local_files_only),
            max_low_log_residual=float(low.max_low_log_residual),
            output_max_depth_m=float(cfg.model.output_max_depth_m),
            low_hidden_channels=int(low.hidden_channels),
        )
        stats = cfg.model.bim_normalization
        condition = cfg.model.bim_condition
        return cls(
            scale_system,
            refiner,
            bim_log_mean=float(stats.mean),
            bim_log_std=float(stats.std),
            disagreement_clip=float(condition.disagreement_clip),
        )

    def train(self, mode: bool = True) -> FrozenHuberDAv2LowRefiner:
        super().train(mode)
        # The scale head has token dropout during training; forcing eval is
        # essential for an exactly frozen, deterministic scale anchor.
        self.scale_system.eval()
        return self

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
        base = batch["base_depth"]
        with torch.no_grad():
            scale_output = self.scale_system._estimate_attention_scale(dict(batch), base)
            scale = scale_output["scale"].to(dtype=base.dtype)
            scaled = base * scale
        condition_batch = dict(batch)
        condition_batch["base_depth"] = scaled.detach()
        condition = build_bim_condition(
            condition_batch,
            bim_log_mean=self.bim_log_mean,
            bim_log_std=self.bim_log_std,
            disagreement_clip=self.disagreement_clip,
        )
        output = self.refiner(batch["rgb"], condition, scaled.detach())
        output.update(
            {
                "scale": scale,
                "log_scale": scale_output["log_scale"],
                "scale_iteration_log_scales": scale_output.get("iteration_log_scales"),
                "bim_condition": condition,
            }
        )
        return output

