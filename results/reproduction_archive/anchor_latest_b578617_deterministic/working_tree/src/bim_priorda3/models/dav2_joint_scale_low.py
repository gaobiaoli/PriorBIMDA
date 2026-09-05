from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional

from .bim_early_fusion_dav2 import (
    BIMEarlyFusionDepthAnythingV2,
    deterministic_interpolate_2d,
)


class AdapterResidualBlock(nn.Module):
    """Spatial residual block in the calibrated-disagreement bottleneck."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("Residual block channels must be positive")
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        for convolution in (self.conv1, self.conv2):
            nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
            nn.init.zeros_(convolution.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.conv2(self.activation(self.conv1(values)))


class CalibratedDisagreementAdapter(nn.Module):
    """Zero-initialized residual adapter from three condition maps to DPT features."""

    def __init__(
        self,
        output_channels: int,
        *,
        input_channels: int = 3,
        hidden_channels: int = 32,
        residual_blocks: int = 0,
        expansion_channels: int | None = None,
    ) -> None:
        super().__init__()
        if (
            output_channels < 1
            or input_channels < 1
            or hidden_channels < 1
            or residual_blocks < 0
            or (expansion_channels is not None and expansion_channels < 1)
        ):
            raise ValueError("Adapter channel counts must be positive")
        self.input_projection = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.GELU()
        self.residual_blocks = nn.ModuleList(
            AdapterResidualBlock(hidden_channels) for _ in range(residual_blocks)
        )
        self.expansion_projection = (
            nn.Conv2d(hidden_channels, expansion_channels, kernel_size=3, padding=1)
            if expansion_channels is not None
            else None
        )
        self.expansion_activation = nn.GELU()
        output_input_channels = (
            int(expansion_channels) if expansion_channels is not None else hidden_channels
        )
        self.output_projection = nn.Conv2d(output_input_channels, output_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        if self.expansion_projection is not None:
            nn.init.kaiming_normal_(self.expansion_projection.weight, nonlinearity="relu")
            nn.init.zeros_(self.expansion_projection.bias)
        # This is the identity-preserving part of the adapter: at initialization
        # F36 + A(C36) is bitwise F36 for every possible condition.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if (
            condition.ndim != 4
            or condition.shape[1] != self.input_projection.in_channels
        ):
            raise ValueError("Calibrated disagreement condition channel count changed")
        values = self.activation(self.input_projection(condition))
        for block in self.residual_blocks:
            values = block(values)
        if self.expansion_projection is not None:
            values = self.expansion_activation(self.expansion_projection(values))
        return self.output_projection(values)


class SharedGeometryAdapterWithStageHeads(nn.Module):
    """Shared convolutional geometry trunk with zero-init r18/r36 heads."""

    STAGES = ("r18", "r36")

    def __init__(
        self,
        output_channels: int,
        *,
        input_channels: int = 3,
        hidden_channels: int = 32,
        residual_blocks: int = 0,
        expansion_channels: int | None = None,
    ) -> None:
        super().__init__()
        if (
            output_channels < 1
            or input_channels < 1
            or hidden_channels < 1
            or residual_blocks < 0
            or (expansion_channels is not None and expansion_channels < 1)
        ):
            raise ValueError("Shared geometry adapter channel counts must be positive")
        self.input_projection = nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.GELU()
        self.residual_blocks = nn.ModuleList(
            AdapterResidualBlock(hidden_channels) for _ in range(residual_blocks)
        )
        self.expansion_projection = (
            nn.Conv2d(hidden_channels, expansion_channels, kernel_size=3, padding=1)
            if expansion_channels is not None
            else None
        )
        self.expansion_activation = nn.GELU()
        head_input_channels = (
            int(expansion_channels) if expansion_channels is not None else hidden_channels
        )
        self.stage_heads = nn.ModuleDict(
            {
                stage: nn.Conv2d(head_input_channels, output_channels, kernel_size=1)
                for stage in self.STAGES
            }
        )
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        if self.expansion_projection is not None:
            nn.init.kaiming_normal_(self.expansion_projection.weight, nonlinearity="relu")
            nn.init.zeros_(self.expansion_projection.bias)
        for head in self.stage_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, condition: torch.Tensor, *, stage: str) -> torch.Tensor:
        if stage not in self.stage_heads:
            raise ValueError(f"Unsupported iterative geometry stage: {stage}")
        if condition.ndim != 4 or condition.shape[1] != self.input_projection.in_channels:
            raise ValueError("Iterative geometry condition channel count changed")
        values = self.activation(self.input_projection(condition))
        for block in self.residual_blocks:
            values = block(values)
        if self.expansion_projection is not None:
            values = self.expansion_activation(self.expansion_projection(values))
        return self.stage_heads[stage](values)


class ZeroInitDINOFeatureAdapter(nn.Module):
    """Shared bottleneck adapter for detached second-pass DINO tokens."""

    def __init__(self, channels: int, *, hidden_channels: int = 64) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("DINO feature adapter channel counts must be positive")
        self.input_projection = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_channels, channels)
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_projection.in_features:
            raise ValueError("DINO tokens must have shape [B,N,C]")
        return self.output_projection(self.activation(self.input_projection(tokens)))


class ZeroInitDPTShortcutAdapter(nn.Module):
    """Zero-init spatial adapter for one detached DPT lateral shortcut."""

    def __init__(self, channels: int, *, hidden_channels: int = 64) -> None:
        super().__init__()
        if channels < 1 or hidden_channels < 1:
            raise ValueError("DPT shortcut adapter channel counts must be positive")
        self.input_projection = nn.Conv2d(
            channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        nn.init.kaiming_normal_(self.input_projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_projection.bias)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 4 or feature.shape[1] != self.input_projection.in_channels:
            raise ValueError("DPT shortcut feature must have shape [B,C,H,W]")
        return self.output_projection(self.activation(self.input_projection(feature)))


def build_native_residual_head(
    input_channels: int,
    hidden_channels: Sequence[int],
) -> nn.Sequential:
    """Build a native-grid residual decoder with a zero-output initialization."""

    widths = tuple(int(channels) for channels in hidden_channels)
    if input_channels < 1 or not widths or min(widths) < 1:
        raise ValueError("Residual decoder channel counts must be positive")
    layers: list[nn.Module] = []
    current_channels = input_channels
    for channels in widths:
        convolution = nn.Conv2d(current_channels, channels, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
        nn.init.zeros_(convolution.bias)
        layers.extend((convolution, nn.GELU()))
        current_channels = channels
    output = nn.Conv2d(current_channels, 1, kernel_size=1)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


def build_calibrated_disagreement_condition(
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    log_scale: torch.Tensor,
    output_size: tuple[int, int],
    *,
    rgb: torch.Tensor | None = None,
    log_residual: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build C=[z_s, |z_s|, M, optional RGB] at a DPT feature resolution.

    ``z_s = log(D_BIM) - log(D_DA3) - stop_gradient(c + r)`` is evaluated at
    full image resolution, where optional ``r`` is the preceding iterative
    residual. Before mask-aware area pooling, signed disagreement is
    normalized by 1.5 and clipped to [-1,1], while its magnitude is normalized
    by 1.5 and clipped to [0,1]. The mask channel is the BIM hit fraction in
    each output cell, rather than a nearest-neighbour binary mask.
    """

    if base_depth.ndim != 4 or base_depth.shape[1] != 1:
        raise ValueError("base_depth must have shape [B,1,H,W]")
    if bim_depth.shape != base_depth.shape or bim_valid.shape != base_depth.shape:
        raise ValueError("DA3 depth, BIM depth, and BIM mask must have identical shapes")
    if log_scale.shape != (base_depth.shape[0], 1, 1, 1):
        raise ValueError("log_scale must have shape [B,1,1,1]")
    if log_residual is not None and log_residual.shape != base_depth.shape:
        raise ValueError("log_residual must match the full-resolution base depth")
    if len(output_size) != 2 or min(output_size) < 1:
        raise ValueError("output_size must contain two positive dimensions")

    base = base_depth.float()
    bim = bim_depth.float()
    valid = (
        (bim_valid > 0.5)
        & torch.isfinite(base)
        & torch.isfinite(bim)
        & (base > 1e-3)
        & (bim > 1e-3)
    )
    support = valid.float()
    calibrated_log_scale = log_scale.detach().float()
    disagreement = bim.clamp_min(1e-3).log() - base.clamp_min(1e-3).log() - calibrated_log_scale
    if log_residual is not None:
        disagreement = disagreement - log_residual.detach().float()
    normalized_disagreement = (disagreement / 1.5).clamp(-1.0, 1.0)
    normalized_magnitude = (disagreement.abs() / 1.5).clamp(0.0, 1.0)
    pooled_support = functional.adaptive_avg_pool2d(support, output_size)
    denominator = pooled_support.clamp_min(torch.finfo(disagreement.dtype).eps)

    def masked_pool(value: torch.Tensor) -> torch.Tensor:
        pooled = functional.adaptive_avg_pool2d(value * support, output_size)
        return torch.where(pooled_support > 0, pooled / denominator, torch.zeros_like(pooled))

    channels = [
        masked_pool(normalized_disagreement),
        masked_pool(normalized_magnitude),
        pooled_support,
    ]
    if rgb is not None:
        if rgb.ndim != 4 or rgb.shape[:2] != (base_depth.shape[0], 3):
            raise ValueError("RGB must have shape [B,3,H,W]")
        if rgb.shape[-2:] != base_depth.shape[-2:]:
            raise ValueError("RGB and depth tensors must be pixel-aligned")
        channels.append(
            functional.adaptive_avg_pool2d(rgb.float().clamp(0.0, 1.0), output_size)
        )
    return torch.cat(
        channels,
        dim=1,
    )


def rebuild_bim_condition_with_scaled_prediction(
    bim_condition: torch.Tensor,
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    log_scale: torch.Tensor,
    *,
    disagreement_clip: float = 1.5,
) -> torch.Tensor:
    """Rebuild the early-fusion disagreement using detached scaled DA3 depth.

    The normalized BIM log-depth and BIM hit-mask channels are copied exactly
    from the first-pass condition. Only the third channel is recomputed as
    ``clip((log(D_BIM) - log(D_DA3) - stop_gradient(c)) / clip, -1, 1)``.
    """

    if bim_condition.ndim != 4 or bim_condition.shape[1] != 3:
        raise ValueError("bim_condition must have shape [B,3,H,W]")
    if base_depth.ndim != 4 or base_depth.shape[1] != 1:
        raise ValueError("base_depth must have shape [B,1,H,W]")
    if bim_depth.shape != base_depth.shape:
        raise ValueError("DA3 depth and BIM depth must have identical shapes")
    if bim_condition.shape[0] != base_depth.shape[0] or bim_condition.shape[-2:] != (
        base_depth.shape[-2:]
    ):
        raise ValueError("BIM condition and depth tensors must be pixel-aligned")
    if log_scale.shape != (base_depth.shape[0], 1, 1, 1):
        raise ValueError("log_scale must have shape [B,1,1,1]")
    if disagreement_clip <= 0:
        raise ValueError("disagreement_clip must be positive")

    base = base_depth.float()
    bim = bim_depth.float()
    valid = (
        (bim_condition[:, 1:2] > 0.5)
        & torch.isfinite(base)
        & torch.isfinite(bim)
        & (base > 1e-3)
        & (bim > 1e-3)
    )
    detached_log_scale = log_scale.detach().float()
    disagreement = (
        bim.clamp_min(1e-3).log()
        - base.clamp_min(1e-3).log()
        - detached_log_scale
    ).clamp(-float(disagreement_clip), float(disagreement_clip)) / float(
        disagreement_clip
    )
    disagreement = torch.where(valid, disagreement, torch.zeros_like(disagreement))
    return torch.cat(
        (
            bim_condition[:, :2],
            disagreement.to(dtype=bim_condition.dtype),
        ),
        dim=1,
    )


class BIMEarlyFusionDAv2JointScaleLow(BIMEarlyFusionDepthAnythingV2):
    """One early-fusion DAv2 encoder for global scale and/or native residuals.

    The final DINOv2 CLS/mean-patch descriptor predicts one global log-scale.
    The two deepest pretrained DPT fusion stages are tapped immediately before
    their terminal upsampling, producing native top-down features at 18x18 and
    36x36. They predict a Laplacian pair: ``r_low1`` is the coarse field and
    ``r_low2`` is the additional 36x36 band. Only the one-channel predictions
    are resized for composing a full-resolution metric-depth map.
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        *,
        regression_hidden_size: int = 256,
        head_dropout_probability: float = 0.0,
        output_weight_std: float = 1e-3,
        residual_hidden_channels: int = 64,
        max_low1_log_residual: float = 0.20,
        max_low2_log_residual: float = 0.10,
        output_max_depth_m: float = 128.0,
        residual_mode: str = "low18_low36",
        calibrated_disagreement_adapter_enabled: bool = False,
        calibrated_disagreement_adapter_hidden_channels: int = 32,
        calibrated_disagreement_adapter_residual_blocks: int = 0,
        calibrated_disagreement_adapter_expansion_channels: int | None = None,
        calibrated_disagreement_adapter_injection: str = "fused_f36",
        calibrated_disagreement_adapter_include_rgb: bool = False,
        iterative_geometry_adapters_enabled: bool = False,
        iterative_geometry_adapters_hidden_channels: int = 32,
        iterative_geometry_adapters_residual_blocks: int = 0,
        iterative_geometry_adapters_expansion_channels: int | None = None,
        iterative_geometry_adapters_weight_sharing: str = "independent",
        low1_decoder_hidden_channels: Sequence[int] | None = None,
        low2_decoder_hidden_channels: Sequence[int] | None = None,
        detached_scale_second_pass_dino_adapter_enabled: bool = False,
        detached_scale_second_pass_dino_adapter_hidden_channels: int = 64,
        detached_scale_second_pass_dino_adapter_scope: str = "all_dino_tokens",
    ) -> None:
        super().__init__(pretrained_model)
        if regression_hidden_size < 1 or residual_hidden_channels < 1:
            raise ValueError("Head hidden sizes must be positive")
        if not 0.0 <= head_dropout_probability < 1.0:
            raise ValueError("head_dropout_probability must be in [0,1)")
        if output_weight_std <= 0:
            raise ValueError("output_weight_std must be positive")
        if min(max_low1_log_residual, max_low2_log_residual, output_max_depth_m) <= 0:
            raise ValueError("Residual bounds and maximum depth must be positive")
        if residual_mode not in {
            "low18_low36",
            "low36_only",
            "low72_only",
            "direct_low18",
        }:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")

        hidden_size = int(self.dav2.config.backbone_config.hidden_size)
        fusion_channels = int(self.dav2.config.fusion_hidden_size)
        self.max_low1_log_residual = float(max_low1_log_residual)
        self.max_low2_log_residual = float(max_low2_log_residual)
        self.output_max_depth_m = float(output_max_depth_m)
        self.residual_mode = str(residual_mode)
        self.detached_scale_second_pass_dino_adapter_enabled = bool(
            detached_scale_second_pass_dino_adapter_enabled
        )
        self.detached_scale_second_pass_dino_adapter_scope = str(
            detached_scale_second_pass_dino_adapter_scope
        )
        if self.detached_scale_second_pass_dino_adapter_scope not in {
            "all_dino_tokens",
            "r36_shortcut",
        }:
            raise ValueError("Unsupported detached second-pass DINO adapter scope")
        self.calibrated_disagreement_adapter_enabled = bool(calibrated_disagreement_adapter_enabled)
        self.calibrated_disagreement_adapter_injection = str(
            calibrated_disagreement_adapter_injection
        )
        self.calibrated_disagreement_adapter_include_rgb = bool(
            calibrated_disagreement_adapter_include_rgb
        )
        self.iterative_geometry_adapters_enabled = bool(
            iterative_geometry_adapters_enabled
        )
        self.iterative_geometry_adapters_weight_sharing = str(
            iterative_geometry_adapters_weight_sharing
        )
        if self.iterative_geometry_adapters_weight_sharing not in {
            "independent",
            "shared_trunk_separate_heads",
        }:
            raise ValueError("Unsupported iterative geometry weight-sharing mode")
        if self.calibrated_disagreement_adapter_injection not in {
            "fused_f36",
            "projected_p36",
        }:
            raise ValueError("Unsupported calibrated-disagreement adapter injection point")
        if self.calibrated_disagreement_adapter_enabled and self.residual_mode != "low36_only":
            raise ValueError(
                "The calibrated disagreement adapter requires a native 36x36 residual head"
            )
        if self.iterative_geometry_adapters_enabled and self.residual_mode != "low18_low36":
            raise ValueError(
                "Iterative geometry adapters require residual_mode=low18_low36"
            )
        if (
            self.iterative_geometry_adapters_enabled
            and self.calibrated_disagreement_adapter_enabled
        ):
            raise ValueError(
                "The iterative geometry path replaces the single r36 disagreement adapter"
            )
        self.scale_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, int(regression_hidden_size)),
            nn.GELU(),
            nn.Dropout(float(head_dropout_probability)),
            nn.Linear(int(regression_hidden_size), 1),
        )
        for module in self.scale_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.scale_head[-1].weight, mean=0.0, std=float(output_weight_std))

        low1_widths = (
            (residual_hidden_channels,)
            if low1_decoder_hidden_channels is None
            else tuple(int(channels) for channels in low1_decoder_hidden_channels)
        )
        low2_widths = (
            (residual_hidden_channels,)
            if low2_decoder_hidden_channels is None
            else tuple(int(channels) for channels in low2_decoder_hidden_channels)
        )
        self.low1_head = build_native_residual_head(
            fusion_channels,
            low1_widths,
        )
        self.low2_head = build_native_residual_head(fusion_channels, low2_widths)
        self.calibrated_disagreement_adapter = (
            CalibratedDisagreementAdapter(
                fusion_channels,
                input_channels=(6 if self.calibrated_disagreement_adapter_include_rgb else 3),
                hidden_channels=int(calibrated_disagreement_adapter_hidden_channels),
                residual_blocks=int(calibrated_disagreement_adapter_residual_blocks),
                expansion_channels=calibrated_disagreement_adapter_expansion_channels,
            )
            if self.calibrated_disagreement_adapter_enabled
            else None
        )
        shared_iterative_geometry = (
            self.iterative_geometry_adapters_enabled
            and self.iterative_geometry_adapters_weight_sharing
            == "shared_trunk_separate_heads"
        )
        self.shared_iterative_geometry_adapter = (
            SharedGeometryAdapterWithStageHeads(
                fusion_channels,
                hidden_channels=int(iterative_geometry_adapters_hidden_channels),
                residual_blocks=int(iterative_geometry_adapters_residual_blocks),
                expansion_channels=iterative_geometry_adapters_expansion_channels,
            )
            if shared_iterative_geometry
            else None
        )
        self.iterative_geometry18_adapter = (
            CalibratedDisagreementAdapter(
                fusion_channels,
                hidden_channels=int(iterative_geometry_adapters_hidden_channels),
                residual_blocks=int(iterative_geometry_adapters_residual_blocks),
                expansion_channels=iterative_geometry_adapters_expansion_channels,
            )
            if self.iterative_geometry_adapters_enabled and not shared_iterative_geometry
            else None
        )
        self.iterative_geometry36_adapter = (
            CalibratedDisagreementAdapter(
                fusion_channels,
                hidden_channels=int(iterative_geometry_adapters_hidden_channels),
                residual_blocks=int(iterative_geometry_adapters_residual_blocks),
                expansion_channels=iterative_geometry_adapters_expansion_channels,
            )
            if self.iterative_geometry_adapters_enabled and not shared_iterative_geometry
            else None
        )
        if not self.detached_scale_second_pass_dino_adapter_enabled:
            self.detached_scale_second_pass_dino_adapter = None
        elif self.detached_scale_second_pass_dino_adapter_scope == "all_dino_tokens":
            self.detached_scale_second_pass_dino_adapter = ZeroInitDINOFeatureAdapter(
                hidden_size,
                hidden_channels=int(detached_scale_second_pass_dino_adapter_hidden_channels),
            )
        else:
            if self.residual_mode != "low36_only":
                raise ValueError("The r36 shortcut adapter requires residual_mode=low36_only")
            self.detached_scale_second_pass_dino_adapter = ZeroInitDPTShortcutAdapter(
                fusion_channels,
                hidden_channels=int(detached_scale_second_pass_dino_adapter_hidden_channels),
            )
        if self.residual_mode in {"low36_only", "low72_only"}:
            for parameter in self.low1_head.parameters():
                parameter.requires_grad_(False)
        elif self.residual_mode == "direct_low18":
            # The DC component of r18 is the scale estimate in this ablation.
            # No independent global regression path is trainable or applied.
            for parameter in self.scale_head.parameters():
                parameter.requires_grad_(False)
            for parameter in self.low2_head.parameters():
                parameter.requires_grad_(False)
        # The official metric-depth output head is not part of this task. It is
        # retained only so the pinned checkpoint contract remains auditable.
        for parameter in self.dav2.head.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> BIMEarlyFusionDAv2JointScaleLow:
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Joint DAv2 scale+r_low requires the dav2 dependency") from error
        pretrained = AutoModelForDepthEstimation.from_pretrained(
            str(model_name_or_path),
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(pretrained, **kwargs)

    def _encode_dino(
        self,
        normalized_rgb: torch.Tensor,
        bim_condition: torch.Tensor,
        *,
        return_feature_maps: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        embeddings = self._early_embeddings(normalized_rgb, bim_condition)
        backbone = self.dav2.backbone
        outputs = backbone.encoder(
            embeddings,
            output_hidden_states=return_feature_maps,
            output_attentions=False,
            return_dict=True,
        )
        final_tokens = backbone.layernorm(outputs.last_hidden_state)
        if not return_feature_maps:
            return final_tokens, ()
        feature_maps: tuple[torch.Tensor, ...] = ()
        for stage, hidden_state in zip(
            backbone.stage_names,
            outputs.hidden_states,
            strict=True,
        ):
            if stage not in backbone.out_features:
                continue
            if backbone.config.apply_layernorm:
                hidden_state = backbone.layernorm(hidden_state)
            if backbone.config.reshape_hidden_states:
                raise RuntimeError("DAv2 unexpectedly reshapes backbone hidden states")
            feature_maps += (hidden_state,)
        if len(feature_maps) != 4:
            raise RuntimeError(f"Expected four DINO feature maps, got {len(feature_maps)}")
        return final_tokens, feature_maps

    def _decode_dpt_native_features(
        self,
        feature_maps: tuple[torch.Tensor, ...],
        *,
        height: int,
        width: int,
        shortcut36_delta: torch.Tensor | None = None,
        projected36_delta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(feature_maps) != 4:
            raise ValueError("DPT decoding requires exactly four DINO feature maps")
        patch_height = height // self.PATCH_SIZE
        patch_width = width // self.PATCH_SIZE
        dpt_neck = self.dav2.neck
        reassembled = dpt_neck.reassemble_stage(
            feature_maps,
            patch_height,
            patch_width,
        )
        projected = [
            convolution(feature)
            for convolution, feature in zip(dpt_neck.convs, reassembled, strict=True)
        ]
        if len(projected) != 4:
            raise RuntimeError("Expected four projected DPT features")
        fusion18 = dpt_neck.fusion_stage.layers[0]
        feature18 = fusion18.projection(fusion18.residual_layer2(projected[3]))
        # Projection is 1x1, so moving it before the official terminal bilinear
        # resize preserves the fused values while exposing the native grid.
        top_down36 = deterministic_interpolate_2d(
            feature18,
            size=projected[2].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        fusion36 = dpt_neck.fusion_stage.layers[1]
        projected36 = projected[2]
        if projected36_delta is not None:
            if projected36_delta.shape != projected36.shape:
                raise ValueError("P36 adapter output shape changed")
            projected36 = projected36 + projected36_delta.to(dtype=projected36.dtype)
        shortcut36 = fusion36.residual_layer1(projected36)
        if shortcut36_delta is not None:
            if shortcut36_delta.shape != shortcut36.shape:
                raise ValueError("r36 shortcut adapter output shape changed")
            shortcut36 = shortcut36 + shortcut36_delta.to(dtype=shortcut36.dtype)
        feature36 = top_down36 + shortcut36
        feature36 = fusion36.projection(fusion36.residual_layer2(feature36))
        top_down72 = deterministic_interpolate_2d(
            feature36,
            size=projected[1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        fusion72 = dpt_neck.fusion_stage.layers[2]
        feature72 = top_down72 + fusion72.residual_layer1(projected[1])
        feature72 = fusion72.projection(fusion72.residual_layer2(feature72))
        return feature18, feature36, feature72

    def _detached_second_pass_r36_shortcut(
        self,
        feature_map: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Map only DINO stage 2 to the native 36x36 DPT shortcut."""

        patch_height = height // self.PATCH_SIZE
        patch_width = width // self.PATCH_SIZE
        patch_tokens = feature_map[:, 1:]
        batch_size, token_count, channels = patch_tokens.shape
        if token_count != patch_height * patch_width:
            raise RuntimeError("Second-pass DINO token grid changed")
        spatial = patch_tokens.reshape(
            batch_size,
            patch_height,
            patch_width,
            channels,
        ).permute(0, 3, 1, 2).contiguous()
        dpt_neck = self.dav2.neck
        reassembled36 = dpt_neck.reassemble_stage.layers[2](spatial)
        projected36 = dpt_neck.convs[2](reassembled36)
        return dpt_neck.fusion_stage.layers[1].residual_layer1(projected36)

    def predict_log_scale(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-only auxiliary path used by DA3 scale equivariance."""
        if self.residual_mode == "direct_low18":
            return rgb.new_zeros((rgb.shape[0], 1, 1, 1))
        normalized = self.normalized_rgb(rgb)
        embeddings = self._early_embeddings(normalized, bim_condition)
        outputs = self.dav2.backbone.encoder(
            embeddings,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        tokens = self.dav2.backbone.layernorm(outputs.last_hidden_state)
        descriptor = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
        return self.scale_head(descriptor.float()).view(-1, 1, 1, 1)

    def _iterative_geometry_delta(
        self,
        condition: torch.Tensor,
        *,
        stage: str,
    ) -> torch.Tensor:
        if self.shared_iterative_geometry_adapter is not None:
            adapter = self.shared_iterative_geometry_adapter
            dtype = adapter.input_projection.weight.dtype
            return adapter(condition.to(dtype=dtype), stage=stage)
        adapter = (
            self.iterative_geometry18_adapter
            if stage == "r18"
            else self.iterative_geometry36_adapter
        )
        if adapter is None:
            raise RuntimeError("Iterative geometry adapter is disabled")
        dtype = adapter.input_projection.weight.dtype
        return adapter(condition.to(dtype=dtype))

    def forward(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
        base_depth: torch.Tensor,
        *,
        bim_depth: torch.Tensor | None = None,
        bim_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise ValueError("base_depth must have shape [B,1,H,W]")
        if base_depth.shape[0] != rgb.shape[0] or base_depth.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB and base-depth shapes must agree")
        if not bool(torch.isfinite(base_depth).all()) or bool((base_depth <= 0).any()):
            raise ValueError("base_depth must be finite and positive")

        normalized_rgb = self.normalized_rgb(rgb)
        tokens, first_pass_feature_maps = self._encode_dino(
            normalized_rgb,
            bim_condition,
            return_feature_maps=True,
        )
        descriptor = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
        log_scale = (
            descriptor.new_zeros((descriptor.shape[0], 1, 1, 1))
            if self.residual_mode == "direct_low18"
            else self.scale_head(descriptor.float()).view(-1, 1, 1, 1)
        )
        scale = log_scale.exp()
        scaled_depth = base_depth.float() * scale

        second_pass_condition = None
        second_pass_adapter_deltas: tuple[torch.Tensor, ...] = ()
        shortcut36_delta = None
        if self.detached_scale_second_pass_dino_adapter is not None:
            if bim_depth is None:
                raise ValueError(
                    "Detached-scale second-pass DINO adapter requires BIM depth"
                )
            second_pass_condition = rebuild_bim_condition_with_scaled_prediction(
                bim_condition,
                base_depth,
                bim_depth,
                log_scale,
            )
            # Pass 2 reuses the same RGB+BIM early-fusion encoder. Its graph is
            # fully detached; only the zero-init adapter remains trainable.
            with torch.no_grad():
                _, second_pass_feature_maps = self._encode_dino(
                    normalized_rgb,
                    second_pass_condition,
                    return_feature_maps=True,
                )
            adapter_dtype = (
                self.detached_scale_second_pass_dino_adapter.input_projection.weight.dtype
            )
            if self.detached_scale_second_pass_dino_adapter_scope == "all_dino_tokens":
                second_pass_adapter_deltas = tuple(
                    self.detached_scale_second_pass_dino_adapter(
                        second_feature.detach().to(dtype=adapter_dtype)
                    ).to(dtype=first_feature.dtype)
                    for first_feature, second_feature in zip(
                        first_pass_feature_maps,
                        second_pass_feature_maps,
                        strict=True,
                    )
                )
                dpt_feature_maps = tuple(
                    first_feature + delta
                    for first_feature, delta in zip(
                        first_pass_feature_maps,
                        second_pass_adapter_deltas,
                        strict=True,
                    )
                )
            else:
                with torch.no_grad():
                    detached_shortcut36 = self._detached_second_pass_r36_shortcut(
                        second_pass_feature_maps[2].detach(),
                        height=normalized_rgb.shape[-2],
                        width=normalized_rgb.shape[-1],
                    ).detach()
                shortcut36_delta = self.detached_scale_second_pass_dino_adapter(
                    detached_shortcut36.to(dtype=adapter_dtype)
                ).to(dtype=detached_shortcut36.dtype)
                second_pass_adapter_deltas = (shortcut36_delta,)
                dpt_feature_maps = first_pass_feature_maps
        else:
            dpt_feature_maps = first_pass_feature_maps

        calibrated_condition = None
        calibrated_delta = None
        if self.calibrated_disagreement_adapter is not None:
            if bim_depth is None or bim_valid is None:
                raise ValueError(
                    "The calibrated disagreement adapter requires BIM depth and BIM mask"
                )
            native36_size = (
                normalized_rgb.shape[-2] // self.PATCH_SIZE,
                normalized_rgb.shape[-1] // self.PATCH_SIZE,
            )
            calibrated_condition = build_calibrated_disagreement_condition(
                base_depth,
                bim_depth,
                bim_valid,
                log_scale,
                native36_size,
                rgb=(rgb if self.calibrated_disagreement_adapter_include_rgb else None),
            )
            adapter_dtype = self.calibrated_disagreement_adapter.input_projection.weight.dtype
            calibrated_delta = self.calibrated_disagreement_adapter(
                calibrated_condition.to(dtype=adapter_dtype)
            )
        projected36_delta = (
            calibrated_delta
            if self.calibrated_disagreement_adapter_injection == "projected_p36"
            else None
        )
        feature18, feature36, feature72 = self._decode_dpt_native_features(
            dpt_feature_maps,
            height=normalized_rgb.shape[-2],
            width=normalized_rgb.shape[-1],
            shortcut36_delta=shortcut36_delta,
            projected36_delta=projected36_delta,
        )

        iterative_condition18 = None
        iterative_condition36 = None
        iterative_delta18 = None
        iterative_delta36 = None
        low1_feature = feature18
        if self.iterative_geometry_adapters_enabled:
            if bim_depth is None or bim_valid is None:
                raise ValueError(
                    "Iterative geometry adapters require BIM depth and BIM mask"
                )
            iterative_condition18 = build_calibrated_disagreement_condition(
                base_depth,
                bim_depth,
                bim_valid,
                log_scale,
                tuple(feature18.shape[-2:]),
            )
            iterative_delta18 = self._iterative_geometry_delta(
                iterative_condition18,
                stage="r18",
            )
            low1_feature = feature18 + iterative_delta18.to(dtype=feature18.dtype)

        low1_native = (
            self.max_low1_log_residual * torch.tanh(self.low1_head(low1_feature))
            if self.residual_mode in {"low18_low36", "direct_low18"}
            else torch.zeros_like(feature18[:, :1])
        )
        low1_full = deterministic_interpolate_2d(
            low1_native,
            size=base_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.residual_mode == "direct_low18":
            low2_native = torch.zeros_like(low1_native)
        else:
            low2_feature = feature72 if self.residual_mode == "low72_only" else feature36
            if self.iterative_geometry_adapters_enabled:
                assert bim_depth is not None and bim_valid is not None
                # Stage 2 observes the prediction already corrected by r18.
                # Stop-gradient preserves the explicit coarse/fine decomposition:
                # r36 losses still train r18 through final depth, but not through
                # the geometry condition used to compute the r36 adapter feature.
                iterative_condition36 = build_calibrated_disagreement_condition(
                    base_depth,
                    bim_depth,
                    bim_valid,
                    log_scale,
                    tuple(feature36.shape[-2:]),
                    log_residual=low1_full,
                )
                iterative_delta36 = self._iterative_geometry_delta(
                    iterative_condition36,
                    stage="r36",
                )
                low2_feature = feature36 + iterative_delta36.to(dtype=feature36.dtype)
            if (
                calibrated_delta is not None
                and self.calibrated_disagreement_adapter_injection == "fused_f36"
            ):
                low2_feature = feature36 + calibrated_delta.to(dtype=feature36.dtype)
            low2_native = self.max_low2_log_residual * torch.tanh(self.low2_head(low2_feature))
        low2_full = deterministic_interpolate_2d(
            low2_native,
            size=base_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_full = low1_full.float() + low2_full.float()
        depth = (scaled_depth * torch.exp(low_full)).clamp(1e-3, self.output_max_depth_m)
        result: dict[str, torch.Tensor | list[list[int]]] = {
            "depth": depth,
            "scaled_depth": scaled_depth,
            "scale": scale,
            "log_scale": log_scale,
            "descriptor": descriptor,
            "low1_log_residual_native": low1_native,
            "low2_log_residual_native": low2_native,
            "low1_log_residual": low1_full,
            "low2_log_residual": low2_full,
            "low_log_residual": low_full,
            "native_feature_shapes": [
                list(feature18.shape[-2:]),
                list(feature36.shape[-2:]),
                list(feature72.shape[-2:]),
            ],
            "active_residual_shape": list(
                (low1_native if self.residual_mode == "direct_low18" else low2_native).shape[-2:]
            ),
        }
        if self.calibrated_disagreement_adapter is not None:
            assert calibrated_condition is not None and calibrated_delta is not None
            result["calibrated_disagreement_condition_native"] = calibrated_condition
            result["calibrated_disagreement_adapter_delta_native"] = calibrated_delta
        if self.iterative_geometry_adapters_enabled:
            assert (
                iterative_condition18 is not None
                and iterative_condition36 is not None
                and iterative_delta18 is not None
                and iterative_delta36 is not None
            )
            result["iterative_geometry_condition18_native"] = iterative_condition18
            result["iterative_geometry_condition36_native"] = iterative_condition36
            result["iterative_geometry_delta18_native"] = iterative_delta18
            result["iterative_geometry_delta36_native"] = iterative_delta36
        if second_pass_condition is not None:
            result["second_pass_bim_condition"] = second_pass_condition
            result["second_pass_adapter_delta_mean_abs"] = torch.stack(
                [delta.float().abs().mean() for delta in second_pass_adapter_deltas]
            )
        return result

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        decoder_lr: float,
        condition_lr: float,
        scale_head_lr: float,
        residual_head_lr: float,
    ) -> list[dict[str, Any]]:
        def trainable(parameters: Any) -> list[nn.Parameter]:
            return [parameter for parameter in parameters if parameter.requires_grad]

        groups = [
            {
                "name": "dinov2_encoder",
                "params": trainable(self.dav2.backbone.parameters()),
                "lr": encoder_lr,
            },
            {
                "name": "dpt_top_down_decoder",
                "params": trainable(self.dav2.neck.parameters()),
                "lr": decoder_lr,
            },
            {
                "name": "bim_condition_projection",
                "params": trainable(self.bim_condition_embed.parameters()),
                "lr": condition_lr,
            },
            {
                "name": "scale_regression_head",
                "params": trainable(self.scale_head.parameters()),
                "lr": scale_head_lr,
            },
            {
                "name": "native_residual_heads",
                "params": [
                    parameter
                    for head in (self.low1_head, self.low2_head)
                    for parameter in head.parameters()
                    if parameter.requires_grad
                ],
                "lr": residual_head_lr,
            },
        ]
        if self.calibrated_disagreement_adapter is not None:
            groups.append(
                {
                    "name": "calibrated_disagreement_adapter",
                    "params": trainable(self.calibrated_disagreement_adapter.parameters()),
                    "lr": residual_head_lr,
                }
            )
        if self.shared_iterative_geometry_adapter is not None:
            groups.append(
                {
                    "name": "shared_iterative_geometry_adapter",
                    "params": trainable(
                        self.shared_iterative_geometry_adapter.parameters()
                    ),
                    "lr": residual_head_lr,
                }
            )
        elif self.iterative_geometry18_adapter is not None:
            assert self.iterative_geometry36_adapter is not None
            groups.append(
                {
                    "name": "iterative_geometry_adapters",
                    "params": trainable(
                        list(self.iterative_geometry18_adapter.parameters())
                        + list(self.iterative_geometry36_adapter.parameters())
                    ),
                    "lr": residual_head_lr,
                }
            )
        if self.detached_scale_second_pass_dino_adapter is not None:
            groups.append(
                {
                    "name": "detached_scale_second_pass_dino_adapter",
                    "params": trainable(
                        self.detached_scale_second_pass_dino_adapter.parameters()
                    ),
                    "lr": residual_head_lr,
                }
            )
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("Optimizer parameter groups overlap")
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if set(parameter_ids) != expected:
            raise RuntimeError("Optimizer groups do not exactly cover trainable parameters")
        return groups

    def initialization_audit(self, checkpoint_path: Path) -> dict[str, Any]:
        maximum_difference = 0.0
        checked_values = 0
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            for prefix, module in (
                ("backbone", self.dav2.backbone),
                ("neck", self.dav2.neck),
            ):
                for key, actual_value in module.state_dict().items():
                    checkpoint_key = f"{prefix}.{key}"
                    if checkpoint_key not in available:
                        raise RuntimeError(f"Official checkpoint lacks {checkpoint_key}")
                    actual = actual_value.detach().float().cpu()
                    expected = checkpoint.get_tensor(checkpoint_key).float()
                    difference = (actual - expected).abs()
                    maximum_difference = max(
                        maximum_difference,
                        float(difference.max()) if difference.numel() else 0.0,
                    )
                    checked_values += actual.numel()
        if self.shared_iterative_geometry_adapter is not None:
            iterative_geometry_adapters_zero = all(
                torch.count_nonzero(parameter).item() == 0
                for head in self.shared_iterative_geometry_adapter.stage_heads.values()
                for parameter in (head.weight, head.bias)
            )
        elif self.iterative_geometry18_adapter is not None:
            assert self.iterative_geometry36_adapter is not None
            iterative_geometry_adapters_zero = all(
                torch.count_nonzero(parameter).item() == 0
                for adapter in (
                    self.iterative_geometry18_adapter,
                    self.iterative_geometry36_adapter,
                )
                for parameter in (
                    adapter.output_projection.weight,
                    adapter.output_projection.bias,
                )
            )
        else:
            iterative_geometry_adapters_zero = True
        result = {
            "official_encoder_dpt_exact": maximum_difference == 0.0,
            "official_parameter_values": checked_values,
            "official_max_abs_difference": maximum_difference,
            "bim_projection_zero": bool(
                torch.count_nonzero(self.bim_condition_embed.weight).item() == 0
                and torch.count_nonzero(self.bim_condition_embed.bias).item() == 0
            ),
            "low1_output_zero": bool(
                torch.count_nonzero(self.low1_head[-1].weight).item() == 0
                and torch.count_nonzero(self.low1_head[-1].bias).item() == 0
            ),
            "low2_output_zero": bool(
                torch.count_nonzero(self.low2_head[-1].weight).item() == 0
                and torch.count_nonzero(self.low2_head[-1].bias).item() == 0
            ),
            "calibrated_disagreement_adapter_zero": bool(
                self.calibrated_disagreement_adapter is None
                or (
                    torch.count_nonzero(
                        self.calibrated_disagreement_adapter.output_projection.weight
                    ).item()
                    == 0
                    and torch.count_nonzero(
                        self.calibrated_disagreement_adapter.output_projection.bias
                    ).item()
                    == 0
                )
            ),
            "iterative_geometry_adapters_zero": iterative_geometry_adapters_zero,
            "detached_scale_second_pass_dino_adapter_zero": bool(
                self.detached_scale_second_pass_dino_adapter is None
                or (
                    torch.count_nonzero(
                        self.detached_scale_second_pass_dino_adapter.output_projection.weight
                    ).item()
                    == 0
                    and torch.count_nonzero(
                        self.detached_scale_second_pass_dino_adapter.output_projection.bias
                    ).item()
                    == 0
                )
            ),
        }
        result["all_pass"] = all(
            bool(result[key])
            for key in (
                "official_encoder_dpt_exact",
                "bim_projection_zero",
                "low1_output_zero",
                "low2_output_zero",
                "calibrated_disagreement_adapter_zero",
                "iterative_geometry_adapters_zero",
                "detached_scale_second_pass_dino_adapter_zero",
            )
        )
        return result


def masked_area_downsample(
    value: torch.Tensor,
    valid: torch.Tensor,
    output_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if value.shape != valid.shape:
        raise ValueError("value and valid must have identical shapes")
    support = valid.to(dtype=value.dtype)
    pooled_support = functional.adaptive_avg_pool2d(support, output_size)
    pooled_value = functional.adaptive_avg_pool2d(value * support, output_size)
    low_valid = pooled_support > 0
    low_value = pooled_value / pooled_support.clamp_min(torch.finfo(value.dtype).eps)
    return torch.where(low_valid, low_value, torch.zeros_like(low_value)), low_valid


def _masked_per_sample_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    effective = mask.to(dtype=value.dtype)
    dimensions = tuple(range(1, value.ndim))
    numerator = (value * effective).sum(dim=dimensions)
    denominator = effective.sum(dim=dimensions).clamp_min(1.0)
    return (numerator / denominator).mean()


def joint_scale_low_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    pixel_weight: torch.Tensor,
    oracle_log_scale: torch.Tensor,
    oracle_supported: torch.Tensor,
    depth_weight: float,
    scale_teacher_weight: float,
    low1_teacher_weight: float,
    low2_teacher_weight: float,
    zero_mean_weight: float,
    teacher_beta: float,
    residual_mode: str = "low18_low36",
    equivariance_error: torch.Tensor | None = None,
    equivariance_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    target = batch["gt_depth"].float()
    prediction = output["depth"].float()
    valid = (
        (batch["gt_valid"] > 0)
        & torch.isfinite(target)
        & torch.isfinite(prediction)
        & (target > 0)
        & (prediction > 0)
    )
    effective = valid.float() * pixel_weight.float()
    log_error = (prediction.clamp_min(1e-6).log() - target.clamp_min(1e-6).log()).abs()
    pixel_micro = (log_error * effective).sum() / effective.sum().clamp_min(1.0)
    per_denominator = effective.flatten(1).sum(dim=1)
    per_numerator = (log_error * effective).flatten(1).sum(dim=1)
    available = per_denominator > 0
    frame_macro = (per_numerator[available] / per_denominator[available]).mean()
    depth = 0.5 * (pixel_micro + frame_macro)

    if residual_mode == "direct_low18":
        scale_teacher = prediction.sum() * 0.0
        # Direct r18 contains both its DC/global-scale component and spatial
        # low-frequency correction. Do not oracle-scale or mean-center it.
        spatial_target = (
            target.clamp_min(1e-6).log()
            - batch["base_depth"].detach().float().clamp_min(1e-6).log()
        )
    else:
        predicted_scale = output["log_scale"].float().flatten(1).mean(dim=1)
        oracle_scale = oracle_log_scale.detach().float().flatten(1).mean(dim=1)
        scale_raw = functional.smooth_l1_loss(
            predicted_scale,
            oracle_scale,
            reduction="none",
            beta=float(teacher_beta),
        )
        scale_teacher = (
            scale_raw[oracle_supported.bool()].mean()
            if bool(oracle_supported.any())
            else prediction.sum() * 0.0
        )
        oracle_scaled = (
            batch["base_depth"].detach().float() * oracle_log_scale.detach().float().exp()
        )
        spatial_target = target.clamp_min(1e-6).log() - oracle_scaled.clamp_min(1e-6).log()
        dimensions = tuple(range(1, spatial_target.ndim))
        target_mean = (spatial_target * valid).sum(dim=dimensions) / valid.sum(
            dim=dimensions
        ).clamp_min(1)
        spatial_target = spatial_target - target_mean.view(-1, 1, 1, 1)

    low1 = output["low1_log_residual_native"].float()
    low2 = output["low2_log_residual_native"].float()
    target18, valid18 = masked_area_downsample(
        spatial_target,
        valid,
        tuple(low1.shape[-2:]),
    )
    target36, valid36 = masked_area_downsample(
        spatial_target,
        valid,
        tuple(low2.shape[-2:]),
    )
    if residual_mode == "low18_low36":
        target_low2 = target36 - functional.interpolate(
            target18,
            size=target36.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low1_teacher = _masked_per_sample_mean(
            functional.smooth_l1_loss(
                low1,
                target18,
                reduction="none",
                beta=float(teacher_beta),
            ),
            valid18,
        )
    elif residual_mode == "direct_low18":
        target_low2 = torch.zeros_like(low2)
        low1_teacher = _masked_per_sample_mean(
            functional.smooth_l1_loss(
                low1,
                target18,
                reduction="none",
                beta=float(teacher_beta),
            ),
            valid18,
        )
    elif residual_mode in {"low36_only", "low72_only"}:
        target_low2 = target36
        low1_teacher = prediction.sum() * 0.0
    else:
        raise ValueError(f"Unsupported residual_mode: {residual_mode}")
    low2_teacher = _masked_per_sample_mean(
        functional.smooth_l1_loss(
            low2,
            target_low2,
            reduction="none",
            beta=float(teacher_beta),
        ),
        valid36,
    )

    combined36 = (
        functional.interpolate(
            low1,
            size=low2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        + low2
    )
    zero_mean = (
        prediction.sum() * 0.0
        if residual_mode == "direct_low18"
        else (
            0.5
            * (low1.mean(dim=(1, 2, 3)).abs().mean() + combined36.mean(dim=(1, 2, 3)).abs().mean())
            if residual_mode == "low18_low36"
            else low2.mean(dim=(1, 2, 3)).abs().mean()
        )
    )
    equivariance = (
        equivariance_error.float().square().mean()
        if equivariance_error is not None
        else prediction.sum() * 0.0
    )
    total = (
        float(depth_weight) * depth
        + float(scale_teacher_weight) * scale_teacher
        + float(low1_teacher_weight) * low1_teacher
        + float(low2_teacher_weight) * low2_teacher
        + float(zero_mean_weight) * zero_mean
        + float(equivariance_weight) * equivariance
    )
    return {
        "total": total,
        "depth": depth,
        "scale_teacher": scale_teacher,
        "low1_teacher": low1_teacher,
        "low2_teacher": low2_teacher,
        "zero_mean": zero_mean,
        "equivariance": equivariance,
        "predicted_residual_mean": combined36.mean(dim=(1, 2, 3)).abs().mean(),
    }
