"""DA3-anchored dense DINOv2--DPT scale and residual model."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .bim_early_fusion_dav2 import BIMEarlyFusionDepthAnythingV2


def build_dense3_scale_residual_condition(
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Build ``[masked log(BIM), log(DA3), BIM availability]``.

    Depth is represented in natural-log metres without per-frame
    normalization or clipping. Invalid BIM values are replaced before the
    logarithm. No explicit disagreement channel and no GT are used.
    """

    base, bim, mask = (
        batch[key].float() for key in ("base_depth", "bim_depth", "bim_valid")
    )
    if (
        base.ndim != 4
        or base.shape[1] != 1
        or base.shape != bim.shape
        or bim.shape != mask.shape
    ):
        raise ValueError("Dense3 depth/mask must have matching [B,1,H,W] shapes")
    if not bool((torch.isfinite(base) & (base > 0)).all()):
        raise ValueError("Metric DA3 must be dense, positive and finite")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("Nonfinite BIM mask")
    hit = (mask > 0.5) & torch.isfinite(bim) & (bim > 0)
    log_bim = torch.where(hit, bim, torch.ones_like(bim)).log()
    return torch.cat((log_bim, base.log(), hit.float()), dim=1)


class DenseResidualOutputHead(nn.Module):
    """Reuse the pretrained metric head trunk but output a signed residual."""

    def __init__(self, pretrained_head: nn.Module) -> None:
        super().__init__()
        self.feature_projection = copy.deepcopy(pretrained_head.conv1)
        self.output_features = copy.deepcopy(pretrained_head.conv2)
        self.activation = nn.ReLU()
        self.output_projection = nn.Conv2d(
            self.output_features.out_channels, 1, kernel_size=1
        )
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, feature: torch.Tensor, *, output_size: tuple[int, int]
    ) -> torch.Tensor:
        value = self.feature_projection(feature)
        value = functional.interpolate(
            value, size=output_size, mode="bilinear", align_corners=True
        )
        value = self.activation(self.output_features(value))
        return self.output_projection(value)


class DAv2Dense3ScaleResidual(BIMEarlyFusionDepthAnythingV2):
    """One-pass DA3-anchored dense model with global and spatial log updates.

    ``D_s = D_DA3 exp(z_s)`` and ``D_final = D_s exp(r(x))``. The dense
    residual is hard mean-centered so the global/spatial decomposition is
    identifiable. Neither component has an oracle/teacher loss.
    """

    CONDITION_CHANNELS = 3
    ARCHITECTURE = "dav2_dense3_log_bim_log_da3_mask_global_scale_dense_residual"

    def __init__(
        self,
        pretrained_model: nn.Module,
        *,
        regression_hidden_size: int = 256,
        head_dropout_probability: float = 0.0,
        max_dense_log_residual: float = 0.45,
    ) -> None:
        super().__init__(pretrained_model)
        if regression_hidden_size < 1:
            raise ValueError("regression_hidden_size must be positive")
        if not 0.0 <= head_dropout_probability < 1.0:
            raise ValueError("head_dropout_probability must be in [0,1)")
        if max_dense_log_residual <= 0:
            raise ValueError("max_dense_log_residual must be positive")
        hidden_size = int(self.dav2.config.backbone_config.hidden_size)
        self.max_dense_log_residual = float(max_dense_log_residual)
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
        # Exact identity initialization: initially z_s=0 for every frame.
        nn.init.zeros_(self.scale_head[-1].weight)
        self.residual_head = DenseResidualOutputHead(self.dav2.head)
        # Official sigmoid*20 head is retained only for provenance/auditing.
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
    ) -> "DAv2Dense3ScaleResidual":
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Dense3 scale+residual requires the dav2 dependency") from error
        pretrained = AutoModelForDepthEstimation.from_pretrained(
            str(model_name_or_path), revision=revision, local_files_only=local_files_only
        )
        return cls(pretrained, **kwargs)

    def _encode_once(
        self, normalized_rgb: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        embeddings = self._early_embeddings(normalized_rgb, condition)
        backbone = self.dav2.backbone
        outputs = backbone.encoder(
            embeddings,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        tokens = backbone.layernorm(outputs.last_hidden_state)
        feature_maps: tuple[torch.Tensor, ...] = ()
        for stage, hidden_state in zip(
            backbone.stage_names, outputs.hidden_states, strict=True
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
        return tokens, feature_maps

    def forward(self, rgb, bim_condition=None):
        # Tensor interface is kept for the inherited official-init audit.
        if not isinstance(rgb, Mapping):
            return super().forward(rgb, bim_condition)
        batch = rgb
        base = batch["base_depth"].float()
        if base.ndim != 4 or base.shape[1] != 1:
            raise ValueError("base_depth must have shape [B,1,H,W]")
        condition = build_dense3_scale_residual_condition(batch)
        normalized = self.normalized_rgb(batch["rgb"])
        tokens, feature_maps = self._encode_once(normalized, condition)
        descriptor = torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)
        log_scale = self.scale_head(descriptor.float()).view(-1, 1, 1, 1)
        scale = log_scale.exp()
        scaled_depth = base * scale
        patch_height = normalized.shape[-2] // self.PATCH_SIZE
        patch_width = normalized.shape[-1] // self.PATCH_SIZE
        dpt_features = self.dav2.neck(feature_maps, patch_height, patch_width)
        raw_residual = self.residual_head(
            dpt_features[-1], output_size=tuple(base.shape[-2:])
        ).float()
        residual = self.max_dense_log_residual * torch.tanh(raw_residual)
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
        depth = scaled_depth * residual.exp()
        return {
            "depth": depth,
            "scaled_depth": scaled_depth,
            "scale": scale,
            "log_scale": log_scale,
            "dense_log_residual": residual,
            "raw_dense_log_residual": raw_residual,
            "descriptor": descriptor,
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
            {"name": "dpt_decoder", "params": list(self.dav2.neck.parameters()), "lr": decoder_lr},
            {"name": "bim_condition_projection", "params": list(self.bim_condition_embed.parameters()), "lr": condition_lr},
            {"name": "global_scale_head", "params": list(self.scale_head.parameters()), "lr": scale_head_lr},
            {"name": "dense_residual_head", "params": list(self.residual_head.parameters()), "lr": residual_head_lr},
        ]
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("Optimizer parameter groups overlap")
        trainable_ids = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if set(parameter_ids) != trainable_ids:
            raise RuntimeError("Optimizer groups do not cover trainable model parameters")
        return groups

    def initialization_audit(self, **kwargs: Any) -> dict[str, Any]:
        result = super().initialization_audit(**kwargs)
        result["global_scale_output_zero_init"] = {
            "pass": bool(torch.count_nonzero(self.scale_head[-1].weight).item() == 0)
        }
        result["dense_residual_output_zero_init"] = {
            "pass": bool(
                torch.count_nonzero(self.residual_head.output_projection.weight).item() == 0
                and torch.count_nonzero(self.residual_head.output_projection.bias).item() == 0
            )
        }
        result["all_pass"] = all(
            bool(item["pass"])
            for key, item in result.items()
            if key != "all_pass"
        )
        return result


def dense_log_depth_loss(prediction, target, valid):
    """Absolute metric log-depth loss, balanced micro/macro at 0.5/0.5."""

    if prediction.ndim == 3:
        prediction = prediction[:, None]
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Log-depth inputs must have equal shapes")
    support = valid.bool() & torch.isfinite(target) & (target > 0)
    if not bool(support.flatten(1).any(dim=1).all()):
        raise ValueError("Every training frame needs at least one valid GT pixel")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        pred = prediction.float()
        gt = target.float()
        if not bool((torch.isfinite(pred[support]) & (pred[support] > 0)).all()):
            raise FloatingPointError("Invalid dense prediction on fixed GT support")
        error = (pred.clamp_min(1e-6).log() - gt.clamp_min(1e-6).log()).abs()
        weight = support.float()
        pixel_micro = (error * weight).sum() / weight.sum()
        per_frame = (error * weight).flatten(1).sum(dim=1) / weight.flatten(1).sum(dim=1)
        frame_macro = per_frame.mean()
        total = 0.5 * (pixel_micro + frame_macro)
    return {
        "total": total,
        "pixel_micro_log_depth": pixel_micro.detach(),
        "frame_macro_log_depth": frame_macro.detach(),
    }
