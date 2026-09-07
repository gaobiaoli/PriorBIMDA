"""Single dense r504 model with a native-F72 auxiliary residual head."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .dav2_dense3_residual import DAv2Dense3Residual, build_dense3_condition


def build_native_auxiliary_head(input_channels: int, hidden_channels: int) -> nn.Sequential:
    """Match the r18-only native residual-head structure."""
    hidden = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
    output = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    nn.init.kaiming_normal_(hidden.weight, nonlinearity="relu")
    nn.init.zeros_(hidden.bias)
    nn.init.zeros_(output.weight)
    nn.init.zeros_(output.bias)
    return nn.Sequential(hidden, nn.GELU(), output)


class DAv2Dense3ResidualAux72(DAv2Dense3Residual):
    """Preserve r504 and add full-GT deep supervision through native r72."""

    ARCHITECTURE = "dav2_dense3_single_r504_plus_native_r72_auxiliary"

    def __init__(
        self,
        pretrained_model: nn.Module,
        *,
        auxiliary_hidden_channels: int = 64,
        auxiliary_max_log_residual: float = 0.45,
    ) -> None:
        super().__init__(pretrained_model)
        if auxiliary_hidden_channels < 1 or auxiliary_max_log_residual <= 0:
            raise ValueError("Auxiliary r72 head settings must be positive")
        fusion_channels = int(self.dav2.config.fusion_hidden_size)
        self.auxiliary_max_log_residual = float(auxiliary_max_log_residual)
        self.auxiliary_r72_head = build_native_auxiliary_head(
            fusion_channels, int(auxiliary_hidden_channels)
        )

    def _encode_feature_maps(
        self,
        normalized_rgb: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        embeddings = self._early_embeddings(normalized_rgb, condition)
        backbone = self.dav2.backbone
        outputs = backbone.encoder(
            embeddings,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
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
        return feature_maps

    def _native_f72(
        self,
        feature_maps: tuple[torch.Tensor, ...],
        neck_output: list[torch.Tensor],
        *,
        patch_height: int,
        patch_width: int,
    ) -> torch.Tensor:
        """Recover fused native F72 before its terminal upsampling to F144.

        ``neck_output[1]`` is the official F36 top-down state after it has
        reached the 72 grid. We add the reassembled P72 shortcut and execute
        the shared fusion72 residual/projection weights, stopping immediately
        before the official terminal resize.
        """
        neck = self.dav2.neck
        reassembled = neck.reassemble_stage(feature_maps, patch_height, patch_width)
        projected_p72 = neck.convs[1](reassembled[1])
        fusion72 = neck.fusion_stage.layers[2]
        top_down72 = neck_output[1]
        if top_down72.shape != projected_p72.shape:
            raise RuntimeError(
                f"F72/P72 shapes differ: {top_down72.shape} != {projected_p72.shape}"
            )
        fused = top_down72 + fusion72.residual_layer1(projected_p72)
        return fusion72.projection(fusion72.residual_layer2(fused))

    def forward(self, rgb, bim_condition=None):
        if not isinstance(rgb, Mapping):
            return super().forward(rgb, bim_condition)
        batch = rgb
        base = batch["base_depth"].float()
        condition = build_dense3_condition(batch)
        normalized = self.normalized_rgb(batch["rgb"])
        feature_maps = self._encode_feature_maps(normalized, condition)
        patch_height = normalized.shape[-2] // self.PATCH_SIZE
        patch_width = normalized.shape[-1] // self.PATCH_SIZE

        # This is the unchanged primary r504 path from DAv2Dense3Residual.
        neck_output = self.dav2.neck(feature_maps, patch_height, patch_width)
        raw_r504 = self.residual_head(
            neck_output[-1], output_size=tuple(base.shape[-2:])
        ).float()
        final_depth = base * raw_r504.exp()

        feature72 = self._native_f72(
            feature_maps,
            neck_output,
            patch_height=patch_height,
            patch_width=patch_width,
        )
        raw_r72 = self.auxiliary_r72_head(feature72).float()
        r72 = self.auxiliary_max_log_residual * torch.tanh(raw_r72)
        r72_full = functional.interpolate(
            r72,
            size=tuple(base.shape[-2:]),
            mode="bilinear",
            align_corners=False,
        )
        auxiliary_depth = base * r72_full.exp()
        return {
            "depth": final_depth,
            "dense_log_residual": raw_r504,
            "raw_dense_log_residual": raw_r504,
            "auxiliary_depth72": auxiliary_depth,
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
            {"name": "dinov2_encoder", "params": list(self.dav2.backbone.parameters()), "lr": encoder_lr},
            {"name": "dpt_decoder", "params": list(self.dav2.neck.parameters()), "lr": decoder_lr},
            {"name": "bim_condition_projection", "params": list(self.bim_condition_embed.parameters()), "lr": condition_lr},
            {"name": "dense_r504_head", "params": list(self.residual_head.parameters()), "lr": residual_head_lr},
            {"name": "auxiliary_r72_head", "params": list(self.auxiliary_r72_head.parameters()), "lr": residual_head_lr},
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
        output = self.auxiliary_r72_head[-1]
        result["auxiliary_r72_output_zero_init"] = {
            "pass": bool(
                torch.count_nonzero(output.weight).item() == 0
                and torch.count_nonzero(output.bias).item() == 0
            )
        }
        result["all_pass"] = all(
            bool(item["pass"]) for key, item in result.items() if key != "all_pass"
        )
        return result
