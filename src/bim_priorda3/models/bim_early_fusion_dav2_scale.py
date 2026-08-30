from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn


class BIMEarlyFusionDAv2ScaleRegressor(nn.Module):
    """Predict one global DA3 log-scale from PriorDA-style DAv2 tokens.

    RGB uses the pretrained Depth Anything V2 ViT-B/14 patch embedding.  A
    zero-initialized, three-channel BIM projection is added before the DINOv2
    transformer.  The DPT decoder is intentionally discarded: the final CLS
    token and mean patch token are pooled and a small MLP directly regresses a
    single log-scale.  No analytic BIM scale, Huber aggregation, refiner, or
    cached DA3 feature participates in the prediction.
    """

    PATCH_SIZE = 14
    CONDITION_CHANNELS = 3
    HIDDEN_SIZE = 768
    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        pretrained_model: nn.Module,
        *,
        regression_hidden_size: int = 256,
        head_dropout_probability: float = 0.0,
        output_weight_std: float = 1e-3,
    ) -> None:
        super().__init__()
        if regression_hidden_size < 1:
            raise ValueError("regression_hidden_size must be positive")
        if not 0.0 <= head_dropout_probability < 1.0:
            raise ValueError("head_dropout_probability must be in [0, 1)")
        if output_weight_std <= 0:
            raise ValueError("output_weight_std must be positive")

        config = pretrained_model.config
        backbone_config = config.backbone_config
        patch_size = int(backbone_config.patch_size)
        hidden_size = int(backbone_config.hidden_size)
        if patch_size != self.PATCH_SIZE:
            raise ValueError(f"Expected DAv2 patch size 14, got {patch_size}")
        if hidden_size != self.HIDDEN_SIZE:
            raise ValueError(f"Expected DAv2 ViT-B hidden size 768, got {hidden_size}")
        if str(config.depth_estimation_type) != "metric":
            raise ValueError("DAv2 scale regression requires the metric-depth checkpoint")

        # Keep only the pretrained encoder.  The official DPT neck/head are not
        # registered in this scale-only model and therefore cannot be trained
        # or accidentally used to predict dense depth.
        self.backbone = pretrained_model.backbone
        projection = self.backbone.embeddings.patch_embeddings.projection
        if (
            projection.in_channels != 3
            or projection.out_channels != hidden_size
            or projection.kernel_size != (patch_size, patch_size)
            or projection.stride != (patch_size, patch_size)
        ):
            raise ValueError("Official DAv2 RGB patch embedding contract changed")

        self.bim_condition_embed = nn.Conv2d(
            self.CONDITION_CHANNELS,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        nn.init.zeros_(self.bim_condition_embed.weight)
        nn.init.zeros_(self.bim_condition_embed.bias)

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
        output_layer = self.scale_head[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("scale_head must end in nn.Linear")
        nn.init.normal_(output_layer.weight, mean=0.0, std=float(output_weight_std))

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

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        revision: str | None = None,
        local_files_only: bool = True,
        regression_hidden_size: int = 256,
        head_dropout_probability: float = 0.0,
        output_weight_std: float = 1e-3,
    ) -> BIMEarlyFusionDAv2ScaleRegressor:
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "DAv2 early-fusion scale regression requires the optional 'dav2' dependency"
            ) from error
        pretrained = AutoModelForDepthEstimation.from_pretrained(
            str(model_name_or_path),
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(
            pretrained,
            regression_hidden_size=regression_hidden_size,
            head_dropout_probability=head_dropout_probability,
            output_weight_std=output_weight_std,
        )

    def enable_gradient_checkpointing(self) -> None:
        self.backbone.gradient_checkpointing_enable()

    def normalized_rgb(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("RGB must have shape [B, 3, H, W]")
        return (rgb.float().clamp(0.0, 1.0) - self.rgb_mean) / self.rgb_std

    def _early_embeddings(
        self,
        normalized_rgb: torch.Tensor,
        bim_condition: torch.Tensor,
    ) -> torch.Tensor:
        if bim_condition.ndim != 4 or bim_condition.shape[1] != self.CONDITION_CHANNELS:
            raise ValueError("BIM condition must have shape [B, 3, H, W]")
        if normalized_rgb.shape[0] != bim_condition.shape[0] or normalized_rgb.shape[-2:] != (
            bim_condition.shape[-2:]
        ):
            raise ValueError("RGB and BIM condition batch/spatial shapes must agree")
        height, width = normalized_rgb.shape[-2:]
        if height % self.PATCH_SIZE or width % self.PATCH_SIZE:
            raise ValueError("DAv2 input dimensions must be divisible by patch size 14")

        embeddings = self.backbone.embeddings
        target_dtype = embeddings.patch_embeddings.projection.weight.dtype
        rgb_tokens = embeddings.patch_embeddings(normalized_rgb.to(dtype=target_dtype))
        bim_tokens = self.bim_condition_embed(
            bim_condition.to(dtype=self.bim_condition_embed.weight.dtype)
        ).flatten(2).transpose(1, 2)
        if rgb_tokens.shape != bim_tokens.shape:
            raise RuntimeError(
                f"RGB and BIM patch tokens differ: {rgb_tokens.shape} != {bim_tokens.shape}"
            )
        tokens = rgb_tokens + bim_tokens.to(dtype=rgb_tokens.dtype)
        cls_tokens = embeddings.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        tokens = tokens + embeddings.interpolate_pos_encoding(tokens, height, width)
        return embeddings.dropout(tokens)

    def encode_tokens(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.normalized_rgb(rgb)
        embeddings = self._early_embeddings(normalized, bim_condition)
        outputs = self.backbone.encoder(
            embeddings,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return self.backbone.layernorm(outputs.last_hidden_state)

    def forward(
        self,
        rgb: torch.Tensor,
        bim_condition: torch.Tensor,
        base_depth: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if base_depth.ndim != 4 or base_depth.shape[1] != 1:
            raise ValueError("Focal-corrected DA3 depth must have shape [B,1,H,W]")
        if base_depth.shape[0] != rgb.shape[0] or base_depth.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB and focal-corrected DA3 shapes differ")
        if not bool(torch.isfinite(base_depth).all()) or bool((base_depth <= 0).any()):
            raise ValueError("Focal-corrected DA3 depth must be finite and positive")

        tokens = self.encode_tokens(rgb, bim_condition)
        cls_descriptor = tokens[:, 0]
        patch_descriptor = tokens[:, 1:].mean(dim=1)
        descriptor = torch.cat((cls_descriptor, patch_descriptor), dim=1)
        log_scale = self.scale_head(descriptor.float()).view(-1, 1, 1, 1)
        scale = log_scale.exp()
        scaled_depth = base_depth.float() * scale
        return {
            "log_scale": log_scale,
            "scale": scale,
            "scaled_depth": scaled_depth,
            "descriptor": descriptor,
        }

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        condition_lr: float,
        scale_head_lr: float,
    ) -> list[dict[str, Any]]:
        groups = [
            {
                "name": "dinov2_encoder",
                "params": list(self.backbone.parameters()),
                "lr": float(encoder_lr),
            },
            {
                "name": "bim_condition_projection",
                "params": list(self.bim_condition_embed.parameters()),
                "lr": float(condition_lr),
            },
            {
                "name": "scale_regression_head",
                "params": list(self.scale_head.parameters()),
                "lr": float(scale_head_lr),
            },
        ]
        parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise RuntimeError("Optimizer parameter groups overlap")
        if set(parameter_ids) != {id(parameter) for parameter in self.parameters()}:
            raise RuntimeError("Optimizer parameter groups do not cover the full model")
        return groups

    def initialization_audit(
        self,
        *,
        checkpoint_path: Path,
        device: torch.device,
        height: int = 56,
        width: int = 56,
    ) -> dict[str, Any]:
        """Verify zero BIM projection and exact pretrained encoder loading."""

        was_training = self.training
        self.eval()
        generator = torch.Generator(device=device).manual_seed(42)
        condition = torch.randn((1, 3, height, width), generator=generator, device=device)
        with torch.inference_mode():
            projected = self.bim_condition_embed(condition)

        encoder_values = 0
        encoder_max_difference = 0.0
        state = self.backbone.state_dict()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            missing: list[str] = []
            for key, actual_value in state.items():
                checkpoint_key = f"backbone.{key}"
                if checkpoint_key not in available:
                    missing.append(checkpoint_key)
                    continue
                actual = actual_value.detach().float().cpu()
                expected = checkpoint.get_tensor(checkpoint_key).float()
                difference = (actual - expected).abs()
                encoder_max_difference = max(
                    encoder_max_difference,
                    float(difference.max()) if difference.numel() else 0.0,
                )
                encoder_values += actual.numel()
        if missing:
            raise RuntimeError(f"Official checkpoint lacks encoder parameters: {missing[:5]}")
        result = {
            "bim_projection_zero_init": {
                "pass": bool(torch.count_nonzero(projected).item() == 0),
                "max_abs_output": float(projected.abs().max()),
            },
            "dav2_encoder_pretrained_loading": {
                "pass": encoder_max_difference == 0.0,
                "parameter_values": encoder_values,
                "max_abs_diff": encoder_max_difference,
            },
            "scale_head_output_initialization": {
                "type": "near_zero_normal_weight_zero_bias",
                "output_weight_std": float(self.scale_head[-1].weight.detach().float().std()),
                "output_bias": float(self.scale_head[-1].bias.detach().float().item()),
            },
        }
        result["all_pass"] = bool(
            result["bim_projection_zero_init"]["pass"]
            and result["dav2_encoder_pretrained_loading"]["pass"]
        )
        self.train(was_training)
        return result


def scale_regression_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    pixel_weight: torch.Tensor,
    oracle_log_scale: torch.Tensor,
    oracle_supported: torch.Tensor,
    depth_weight: float,
    coarse_depth_weight: float,
    oracle_weight: float,
    oracle_beta: float,
    equivariance_error: torch.Tensor | None = None,
    equivariance_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Matched scale-only objective used by the Area_1 three-epoch screen."""

    prediction = output["scaled_depth"].float()
    target = batch["gt_depth"].float()
    valid = (
        (batch["gt_valid"] > 0)
        & torch.isfinite(target)
        & torch.isfinite(prediction)
        & (target > 0)
        & (prediction > 0)
    )
    effective = valid.float() * pixel_weight.float()
    log_error = (prediction.clamp_min(1e-6).log() - target.clamp_min(1e-6).log()).abs()
    denominator = effective.sum()
    if not bool(denominator > 0):
        raise RuntimeError("Training batch has no valid scale supervision")
    pixel_micro = (log_error * effective).sum() / denominator
    per_denominator = effective.flatten(1).sum(dim=1)
    per_numerator = (log_error * effective).flatten(1).sum(dim=1)
    per_available = per_denominator > 0
    frame_macro = (per_numerator[per_available] / per_denominator[per_available]).mean()
    depth = 0.5 * pixel_micro + 0.5 * frame_macro

    predicted_log_scale = output["log_scale"].float().flatten(1).mean(dim=1)
    oracle_vector = oracle_log_scale.float().flatten(1).mean(dim=1)
    oracle_raw = torch.nn.functional.smooth_l1_loss(
        predicted_log_scale,
        oracle_vector,
        reduction="none",
        beta=float(oracle_beta),
    )
    oracle = (
        oracle_raw[oracle_supported.bool()].mean()
        if bool(oracle_supported.any())
        else prediction.sum() * 0.0
    )
    equivariance = (
        equivariance_error.float().square().mean()
        if equivariance_error is not None
        else prediction.sum() * 0.0
    )
    total = (
        (float(depth_weight) + float(coarse_depth_weight)) * depth
        + float(oracle_weight) * oracle
        + float(equivariance_weight) * equivariance
    )
    return {
        "total": total,
        "depth": depth.detach(),
        "pixel_micro_log_depth": pixel_micro.detach(),
        "frame_macro_log_depth": frame_macro.detach(),
        "oracle": oracle.detach(),
        "equivariance": equivariance.detach(),
    }
