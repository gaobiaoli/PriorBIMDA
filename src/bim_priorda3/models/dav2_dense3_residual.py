"""Single-output DA3-anchored dense DINOv2--DPT residual model."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .bim_early_fusion_dav2 import BIMEarlyFusionDepthAnythingV2


def build_dense3_condition(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Build [masked log(BIM), log(DA3), BIM availability]."""
    base, bim, mask = (
        batch[key].float() for key in ("base_depth", "bim_depth", "bim_valid")
    )
    if base.ndim != 4 or base.shape[1] != 1 or base.shape != bim.shape or bim.shape != mask.shape:
        raise ValueError("Dense3 depth/mask must have matching [B,1,H,W] shapes")
    if not bool((torch.isfinite(base) & (base > 0)).all()):
        raise ValueError("Metric DA3 must be dense, positive and finite")
    if not bool(torch.isfinite(mask).all()):
        raise ValueError("Nonfinite BIM mask")
    hit = (mask > 0.5) & torch.isfinite(bim) & (bim > 0)
    log_bim = torch.where(hit, bim, torch.ones_like(bim)).log()
    return torch.cat((log_bim, base.log(), hit.float()), dim=1)


class DenseResidualOutputHead(nn.Module):
    """Official metric-head trunk followed by a zero-init signed output."""
    def __init__(self, pretrained_head: nn.Module) -> None:
        super().__init__()
        self.feature_projection = copy.deepcopy(pretrained_head.conv1)
        self.output_features = copy.deepcopy(pretrained_head.conv2)
        self.activation = nn.ReLU()
        self.output_projection = nn.Conv2d(self.output_features.out_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, feature: torch.Tensor, *, output_size: tuple[int, int]) -> torch.Tensor:
        value = self.feature_projection(feature)
        value = functional.interpolate(value, size=output_size, mode="bilinear", align_corners=True)
        return self.output_projection(self.activation(self.output_features(value)))


class DAv2Dense3Residual(BIMEarlyFusionDepthAnythingV2):
    """Predict one uncentered dense log-residual and no separate scale."""
    CONDITION_CHANNELS = 3
    ARCHITECTURE = "dav2_dense3_log_bim_log_da3_mask_single_dense_residual"

    def __init__(self, pretrained_model: nn.Module) -> None:
        super().__init__(pretrained_model)
        self.residual_head = DenseResidualOutputHead(self.dav2.head)
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
    ) -> "DAv2Dense3Residual":
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("Dense3 residual requires the dav2 dependency") from error
        pretrained = AutoModelForDepthEstimation.from_pretrained(
            str(model_name_or_path), revision=revision, local_files_only=local_files_only
        )
        return cls(pretrained, **kwargs)

    def forward(self, rgb, bim_condition=None):
        if not isinstance(rgb, Mapping):
            return super().forward(rgb, bim_condition)
        batch = rgb
        base = batch["base_depth"].float()
        condition = build_dense3_condition(batch)
        normalized = self.normalized_rgb(batch["rgb"])
        embeddings = self._early_embeddings(normalized, condition)
        neck, _, _ = self._neck_embeddings(
            embeddings, height=normalized.shape[-2], width=normalized.shape[-1]
        )
        raw_residual = self.residual_head(neck[-1], output_size=tuple(base.shape[-2:])).float()
        # The sole dense output owns both global/DC and spatial correction.
        # Do not tanh-bound or mean-center it.
        residual = raw_residual
        depth = base * residual.exp()
        return {
            "depth": depth,
            "dense_log_residual": residual,
            "raw_dense_log_residual": raw_residual,
        }

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        decoder_lr: float,
        condition_lr: float,
        residual_head_lr: float,
    ) -> list[dict[str, Any]]:
        groups = [
            {"name": "dinov2_encoder", "params": list(self.dav2.backbone.parameters()), "lr": encoder_lr},
            {"name": "dpt_decoder", "params": list(self.dav2.neck.parameters()), "lr": decoder_lr},
            {"name": "bim_condition_projection", "params": list(self.bim_condition_embed.parameters()), "lr": condition_lr},
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
        result["dense_residual_output_zero_init"] = {
            "pass": bool(
                torch.count_nonzero(self.residual_head.output_projection.weight).item() == 0
                and torch.count_nonzero(self.residual_head.output_projection.bias).item() == 0
            )
        }
        result["all_pass"] = all(
            bool(item["pass"]) for key, item in result.items() if key != "all_pass"
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
        pred, gt = prediction.float(), target.float()
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
