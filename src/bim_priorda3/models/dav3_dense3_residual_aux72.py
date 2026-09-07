"""DA3Metric-Large dense residual model with native-F72 auxiliary supervision."""
from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .dav2_dense3_residual import build_dense3_condition
from .dav2_dense3_residual_aux72 import build_native_auxiliary_head


class DA3DenseResidualOutputHead(nn.Module):
    """Copy the pretrained DA3 metric head stem and zero-init a signed output."""

    def __init__(self, pretrained_head: nn.Module) -> None:
        super().__init__()
        self.feature_projection = copy.deepcopy(pretrained_head.scratch.output_conv1)
        self.output_features = copy.deepcopy(pretrained_head.scratch.output_conv2[0])
        self.activation = nn.ReLU()
        self.output_projection = nn.Conv2d(self.output_features.out_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, feature: torch.Tensor, *, output_size: tuple[int, int]) -> torch.Tensor:
        value = self.feature_projection(feature)
        value = functional.interpolate(
            value, size=output_size, mode="bilinear", align_corners=True
        )
        return self.output_projection(self.activation(self.output_features(value)))


class DA3MetricDense3ResidualAux72(nn.Module):
    """Replace the DAv2-B foundation with official DA3Metric-Large.

    RGB and the zero-initialized three-channel BIM/DA3 condition are summed at
    the ViT-L/14 patch-token input. The pretrained DA3 DPT pyramid is retained.
    A single unbounded r504 and a bounded native r72 both correct the externally
    focal-scaled DA3 metric anchor; there is no separate scale head.
    """

    PATCH_SIZE = 14
    CONDITION_CHANNELS = 3
    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD = (0.229, 0.224, 0.225)
    ARCHITECTURE = "dav3metric_large_dense3_single_r504_plus_native_r72_auxiliary"

    def __init__(
        self,
        pretrained_api: nn.Module,
        *,
        auxiliary_hidden_channels: int = 64,
        auxiliary_max_log_residual: float = 0.45,
    ) -> None:
        super().__init__()
        self.da3 = pretrained_api.model
        vision = self.backbone.pretrained
        patch_projection = vision.patch_embed.proj
        hidden_size = int(vision.embed_dim)
        patch_size = tuple(int(value) for value in vision.patch_embed.patch_size)
        if patch_size != (self.PATCH_SIZE, self.PATCH_SIZE):
            raise ValueError(f"Expected DA3 patch size 14, got {patch_size}")
        if hidden_size != 1024 or str(self.backbone.name) != "vitl":
            raise ValueError(
                f"Expected official DA3Metric ViT-L/14 (1024), got {self.backbone.name}/{hidden_size}"
            )
        if patch_projection.in_channels != 3 or patch_projection.out_channels != hidden_size:
            raise ValueError("Official DA3Metric RGB patch projection contract changed")
        if tuple(int(value) for value in self.backbone.out_layers) != (4, 11, 17, 23):
            raise ValueError(f"Unexpected DA3Metric feature layers: {self.backbone.out_layers}")
        if int(self.dpt.patch_size) != self.PATCH_SIZE:
            raise ValueError("DA3Metric DPT patch size differs from the backbone")
        if int(self.dpt.projects[0].in_channels) != hidden_size:
            raise ValueError("DA3Metric DPT hidden size differs from the backbone")
        fusion_channels = int(self.dpt.scratch.layer1_rn.out_channels)
        if fusion_channels != 256:
            raise ValueError(f"Expected DA3Metric DPT width 256, got {fusion_channels}")

        self.bim_condition_embed = nn.Conv2d(
            self.CONDITION_CHANNELS,
            hidden_size,
            kernel_size=self.PATCH_SIZE,
            stride=self.PATCH_SIZE,
        )
        nn.init.zeros_(self.bim_condition_embed.weight)
        nn.init.zeros_(self.bim_condition_embed.bias)
        self.residual_head = DA3DenseResidualOutputHead(self.dpt)
        self.auxiliary_max_log_residual = float(auxiliary_max_log_residual)
        self.auxiliary_r72_head = build_native_auxiliary_head(
            fusion_channels, int(auxiliary_hidden_channels)
        )
        if self.auxiliary_max_log_residual <= 0:
            raise ValueError("Auxiliary residual bound must be positive")

        # The official prediction branches are replaced by residual heads.
        for module_name in ("output_conv1", "output_conv2", "sky_output_conv2"):
            module = getattr(self.dpt.scratch, module_name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

        self.register_buffer(
            "rgb_mean",
            torch.tensor(self.RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor(self.RGB_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self._active_condition: torch.Tensor | None = None
        self._condition_hook = vision.patch_embed.register_forward_hook(
            self._add_condition_patch_tokens
        )
        self._gradient_checkpointing_enabled = False

    @property
    def backbone(self) -> nn.Module:
        return self.da3.backbone

    @property
    def dpt(self) -> nn.Module:
        return self.da3.head

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = True,
        **kwargs: Any,
    ) -> "DA3MetricDense3ResidualAux72":
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("DA3 metric residual model requires depth_anything_3") from error
        pretrained = DepthAnything3.from_pretrained(
            str(model_name_or_path), revision=revision, local_files_only=local_files_only
        )
        return cls(pretrained, **kwargs)

    def normalized_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB must have shape [B,3,H,W]")
        return (rgb.float().clamp(0.0, 1.0) - self.rgb_mean) / self.rgb_std

    def _add_condition_patch_tokens(
        self,
        _module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        rgb_tokens: torch.Tensor,
    ) -> torch.Tensor:
        condition = self._active_condition
        if condition is None:
            raise RuntimeError("DA3 patch embedding ran without an active dense3 condition")
        if len(inputs) != 1 or inputs[0].shape[0] != condition.shape[0]:
            raise RuntimeError("DA3 single-view RGB/condition batch contract changed")
        condition_tokens = self.bim_condition_embed(
            condition.to(dtype=self.bim_condition_embed.weight.dtype)
        ).flatten(2).transpose(1, 2)
        if condition_tokens.shape != rgb_tokens.shape:
            raise RuntimeError(
                f"DA3 RGB/condition token shapes differ: {rgb_tokens.shape}/{condition_tokens.shape}"
            )
        return rgb_tokens + condition_tokens.to(dtype=rgb_tokens.dtype)

    def enable_gradient_checkpointing(self) -> None:
        """Checkpoint each ViT block while preserving the official state dict."""
        if self._gradient_checkpointing_enabled:
            return
        import torch.utils.checkpoint

        for block in self.backbone.pretrained.blocks:
            original_forward = block.forward

            def checkpointed_forward(
                value: torch.Tensor,
                pos: torch.Tensor | None = None,
                attn_mask: torch.Tensor | None = None,
                *,
                _original=original_forward,
            ) -> torch.Tensor:
                if self.training and torch.is_grad_enabled():
                    def run(item: torch.Tensor) -> torch.Tensor:
                        return _original(item, pos=pos, attn_mask=attn_mask)
                    return torch.utils.checkpoint.checkpoint(run, value, use_reentrant=False)
                return _original(value, pos=pos, attn_mask=attn_mask)

            block.forward = checkpointed_forward
        self._gradient_checkpointing_enabled = True

    def _encode(self, normalized_rgb: torch.Tensor, condition: torch.Tensor):
        self._active_condition = condition
        try:
            features, _ = self.backbone(normalized_rgb[:, None])
        finally:
            self._active_condition = None
        if len(features) != 4:
            raise RuntimeError(f"Expected four DA3 DINO features, got {len(features)}")
        return features

    def _decode_with_native_f72(
        self,
        features,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run official DA3 DPT and expose P72-fused state before upsampling."""
        batch, views, tokens, channels = features[0][0].shape
        if views != 1:
            raise ValueError("This experiment supports only monocular DA3 input")
        patch_height, patch_width = height // self.PATCH_SIZE, width // self.PATCH_SIZE
        resized = []
        for stage, feature in enumerate(features):
            value = feature[0].reshape(batch * views, tokens, channels)
            # DA3 get_intermediate_layers has already removed CLS/register
            # tokens.  Its official DPT is therefore called with
            # patch_start_idx=0, unlike the Hugging Face DAv2 interface.
            value = value[:, 0:]
            value = self.dpt.norm(value)
            value = value.permute(0, 2, 1).contiguous().reshape(
                batch * views, channels, patch_height, patch_width
            )
            value = self.dpt.projects[stage](value)
            if self.dpt.pos_embed:
                value = self.dpt._add_pos_embed(value, width, height)
            resized.append(self.dpt.resize_layers[stage](value))

        scratch = self.dpt.scratch
        l1 = scratch.layer1_rn(resized[0])
        l2 = scratch.layer2_rn(resized[1])
        l3 = scratch.layer3_rn(resized[2])
        l4 = scratch.layer4_rn(resized[3])
        out36 = scratch.refinenet4(l4, size=l3.shape[2:])
        top_down72 = scratch.refinenet3(out36, l3, size=l2.shape[2:])

        fusion72 = scratch.refinenet2
        feature72 = fusion72.skip_add.add(top_down72, fusion72.resConfUnit1(l2))
        feature72 = fusion72.resConfUnit2(feature72)
        out144 = functional.interpolate(
            feature72,
            size=l1.shape[2:],
            mode="bilinear",
            align_corners=fusion72.align_corners,
        )
        out144 = fusion72.out_conv(out144)
        out288 = scratch.refinenet1(out144, l1)
        return out288, feature72

    def forward(self, batch: Mapping[str, torch.Tensor]):
        base = batch["base_depth"].float()
        condition = build_dense3_condition(batch)
        normalized = self.normalized_rgb(batch["rgb"])
        features = self._encode(normalized, condition)
        feature288, feature72 = self._decode_with_native_f72(
            features, height=normalized.shape[-2], width=normalized.shape[-1]
        )
        raw_r504 = self.residual_head(
            feature288, output_size=tuple(base.shape[-2:])
        ).float()
        final_depth = base * raw_r504.exp()
        raw_r72 = self.auxiliary_r72_head(feature72).float()
        r72 = self.auxiliary_max_log_residual * torch.tanh(raw_r72)
        r72_full = functional.interpolate(
            r72, size=tuple(base.shape[-2:]), mode="bilinear", align_corners=False
        )
        return {
            "depth": final_depth,
            "dense_log_residual": raw_r504,
            "raw_dense_log_residual": raw_r504,
            "auxiliary_depth72": base * r72_full.exp(),
            "auxiliary_log_residual72_native": r72,
            "auxiliary_log_residual72": r72_full,
            "native_feature72_shape": list(feature72.shape[-2:]),
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
            {"name": "dinov2_encoder", "params": [p for p in self.backbone.parameters() if p.requires_grad], "lr": encoder_lr},
            {"name": "dpt_decoder", "params": [p for p in self.dpt.parameters() if p.requires_grad], "lr": decoder_lr},
            {"name": "bim_condition_projection", "params": list(self.bim_condition_embed.parameters()), "lr": condition_lr},
            {"name": "dense_r504_head", "params": list(self.residual_head.parameters()), "lr": residual_head_lr},
            {"name": "auxiliary_r72_head", "params": list(self.auxiliary_r72_head.parameters()), "lr": residual_head_lr},
        ]
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(ids) != len(set(ids)):
            raise RuntimeError("DA3 optimizer parameter groups overlap")
        trainable = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if set(ids) != trainable:
            raise RuntimeError("DA3 optimizer groups do not cover all trainable parameters")
        return groups

    def initialization_audit(self, **_kwargs: Any) -> dict[str, Any]:
        result = {
            "foundation": {
                "pass": True,
                "backbone": "DA3Metric-Large DINOv2-L/14",
                "hidden_size": 1024,
                "dpt_fusion_channels": 256,
            },
            "condition_zero_init": {
                "pass": bool(
                    torch.count_nonzero(self.bim_condition_embed.weight).item() == 0
                    and torch.count_nonzero(self.bim_condition_embed.bias).item() == 0
                )
            },
            "dense_residual_output_zero_init": {
                "pass": bool(
                    torch.count_nonzero(self.residual_head.output_projection.weight).item() == 0
                    and torch.count_nonzero(self.residual_head.output_projection.bias).item() == 0
                )
            },
            "auxiliary_r72_output_zero_init": {
                "pass": bool(
                    torch.count_nonzero(self.auxiliary_r72_head[-1].weight).item() == 0
                    and torch.count_nonzero(self.auxiliary_r72_head[-1].bias).item() == 0
                )
            },
        }
        result["all_pass"] = all(bool(value["pass"]) for value in result.values())
        return result
