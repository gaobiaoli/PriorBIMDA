from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional

from .bim_early_fusion_dav2 import BIMEarlyFusionDepthAnythingV2


class BIMEarlyFusionDAv2JointScaleLow(BIMEarlyFusionDepthAnythingV2):
    """One early-fusion DAv2 encoder for global scale and two native residuals.

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
        if residual_mode not in {"low18_low36", "low36_only", "low72_only"}:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")

        hidden_size = int(self.dav2.config.backbone_config.hidden_size)
        fusion_channels = int(self.dav2.config.fusion_hidden_size)
        self.max_low1_log_residual = float(max_low1_log_residual)
        self.max_low2_log_residual = float(max_low2_log_residual)
        self.output_max_depth_m = float(output_max_depth_m)
        self.residual_mode = str(residual_mode)
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

        def residual_head() -> nn.Sequential:
            head = nn.Sequential(
                nn.Conv2d(fusion_channels, residual_hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(residual_hidden_channels, 1, kernel_size=1),
            )
            nn.init.kaiming_normal_(head[0].weight, nonlinearity="relu")
            nn.init.zeros_(head[0].bias)
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            return head

        self.low1_head = residual_head()
        self.low2_head = residual_head()
        if self.residual_mode in {"low36_only", "low72_only"}:
            for parameter in self.low1_head.parameters():
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

    def _encoded_neck(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.normalized_rgb(rgb)
        embeddings = self._early_embeddings(normalized, bim_condition)
        backbone = self.dav2.backbone
        outputs = backbone.encoder(
            embeddings,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        final_tokens = backbone.layernorm(outputs.last_hidden_state)
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
        height, width = normalized.shape[-2:]
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
        top_down36 = functional.interpolate(
            feature18,
            size=projected[2].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        fusion36 = dpt_neck.fusion_stage.layers[1]
        feature36 = top_down36 + fusion36.residual_layer1(projected[2])
        feature36 = fusion36.projection(fusion36.residual_layer2(feature36))
        top_down72 = functional.interpolate(
            feature36,
            size=projected[1].shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        fusion72 = dpt_neck.fusion_stage.layers[2]
        feature72 = top_down72 + fusion72.residual_layer1(projected[1])
        feature72 = fusion72.projection(fusion72.residual_layer2(feature72))
        return final_tokens, feature18, feature36, feature72

    def predict_log_scale(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Scale-only auxiliary path used by DA3 scale equivariance."""
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

    def forward(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
        base_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[list[int]]]:
        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise ValueError("base_depth must have shape [B,1,H,W]")
        if base_depth.shape[0] != rgb.shape[0] or base_depth.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB and base-depth shapes must agree")
        if not bool(torch.isfinite(base_depth).all()) or bool((base_depth <= 0).any()):
            raise ValueError("base_depth must be finite and positive")

        tokens, feature18, feature36, feature72 = self._encoded_neck(rgb, bim_condition)
        descriptor = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
        log_scale = self.scale_head(descriptor.float()).view(-1, 1, 1, 1)
        scale = log_scale.exp()
        scaled_depth = base_depth.float() * scale

        low1_native = (
            self.max_low1_log_residual * torch.tanh(self.low1_head(feature18))
            if self.residual_mode == "low18_low36"
            else torch.zeros_like(feature18[:, :1])
        )
        low2_feature = feature72 if self.residual_mode == "low72_only" else feature36
        low2_native = self.max_low2_log_residual * torch.tanh(self.low2_head(low2_feature))
        low1_full = functional.interpolate(
            low1_native,
            size=base_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low2_full = functional.interpolate(
            low2_native,
            size=base_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        low_full = low1_full.float() + low2_full.float()
        depth = (scaled_depth * torch.exp(low_full)).clamp(1e-3, self.output_max_depth_m)
        return {
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
            "active_residual_shape": list(low2_native.shape[-2:]),
        }

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        decoder_lr: float,
        condition_lr: float,
        scale_head_lr: float,
        residual_head_lr: float,
    ) -> list[dict[str, Any]]:
        groups = [
            {"name": "dinov2_encoder", "params": list(self.dav2.backbone.parameters()), "lr": encoder_lr},
            {"name": "dpt_top_down_decoder", "params": list(self.dav2.neck.parameters()), "lr": decoder_lr},
            {"name": "bim_condition_projection", "params": list(self.bim_condition_embed.parameters()), "lr": condition_lr},
            {"name": "scale_regression_head", "params": list(self.scale_head.parameters()), "lr": scale_head_lr},
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
        }
        result["all_pass"] = all(
            bool(result[key])
            for key in (
                "official_encoder_dpt_exact",
                "bim_projection_zero",
                "low1_output_zero",
                "low2_output_zero",
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

    oracle_scaled = batch["base_depth"].detach().float() * oracle_log_scale.detach().float().exp()
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

    combined36 = functional.interpolate(
        low1,
        size=low2.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ) + low2
    zero_mean = (
        0.5
        * (
            low1.mean(dim=(1, 2, 3)).abs().mean()
            + combined36.mean(dim=(1, 2, 3)).abs().mean()
        )
        if residual_mode == "low18_low36"
        else low2.mean(dim=(1, 2, 3)).abs().mean()
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
