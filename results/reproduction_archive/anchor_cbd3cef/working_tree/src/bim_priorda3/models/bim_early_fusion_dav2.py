from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional


class _DeterministicResize2D(torch.autograd.Function):
    """Keep the CUDA resize forward but evaluate its linear backward on CPU."""

    @staticmethod
    def forward(
        ctx: Any,
        values: torch.Tensor,
        output_height: int,
        output_width: int,
        mode: str,
        align_corners: bool,
    ) -> torch.Tensor:
        ctx.input_shape = tuple(values.shape)
        ctx.input_device = values.device
        ctx.input_dtype = values.dtype
        ctx.output_size = (int(output_height), int(output_width))
        ctx.mode = str(mode)
        ctx.align_corners = bool(align_corners)
        return functional.interpolate(
            values,
            size=ctx.output_size,
            mode=ctx.mode,
            align_corners=ctx.align_corners,
        )

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        # Bilinear/bicubic resize is linear, so its Jacobian is independent of
        # the forward input. A zero CPU proxy therefore gives the exact same
        # mathematical gradient without CUDA atomic accumulation.
        with torch.enable_grad():
            proxy = torch.zeros(
                ctx.input_shape,
                dtype=torch.float32,
                device="cpu",
                requires_grad=True,
            )
            resized = functional.interpolate(
                proxy,
                size=ctx.output_size,
                mode=ctx.mode,
                align_corners=ctx.align_corners,
            )
            grad_input = torch.autograd.grad(
                resized,
                proxy,
                grad_outputs=grad_output.detach().to(device="cpu", dtype=torch.float32),
            )[0]
        return (
            grad_input.to(device=ctx.input_device, dtype=ctx.input_dtype),
            None,
            None,
            None,
            None,
        )


def deterministic_interpolate_2d(
    values: torch.Tensor,
    *,
    size: tuple[int, int] | torch.Size,
    mode: str,
    align_corners: bool,
) -> torch.Tensor:
    """Use a deterministic backward for CUDA bilinear/bicubic interpolation."""
    output_size = (int(size[0]), int(size[1]))
    if (
        torch.are_deterministic_algorithms_enabled()
        and values.is_cuda
        and values.requires_grad
        and mode in {"bilinear", "bicubic"}
    ):
        return _DeterministicResize2D.apply(
            values,
            output_size[0],
            output_size[1],
            mode,
            align_corners,
        )
    return functional.interpolate(
        values,
        size=output_size,
        mode=mode,
        align_corners=align_corners,
    )


def _deterministic_position_encoding(
    embeddings_module: nn.Module,
    embeddings: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Mirror the Hugging Face DINOv2 positional interpolation exactly."""
    num_patches = embeddings.shape[1] - 1
    num_positions = embeddings_module.position_embeddings.shape[1] - 1
    if num_patches == num_positions and height == width:
        return embeddings_module.position_embeddings

    class_position = embeddings_module.position_embeddings[:, :1]
    patch_position = embeddings_module.position_embeddings[:, 1:]
    channels = embeddings.shape[-1]
    side = int(num_positions**0.5)
    if side * side != num_positions:
        raise RuntimeError("DINOv2 positional embedding grid is not square")
    patch_position = patch_position.reshape(1, side, side, channels)
    patch_position = patch_position.permute(0, 3, 1, 2)
    target_dtype = patch_position.dtype
    patch_position = deterministic_interpolate_2d(
        patch_position.float(),
        size=(height // embeddings_module.patch_size, width // embeddings_module.patch_size),
        mode="bicubic",
        align_corners=False,
    ).to(dtype=target_dtype)
    patch_position = patch_position.permute(0, 2, 3, 1).view(1, -1, channels)
    return torch.cat((class_position, patch_position), dim=1)


class BIMEarlyFusionDepthAnythingV2(nn.Module):
    """Zero-initialized BIM conditioning before the DINOv2 transformer.

    The wrapped model is the official Hugging Face conversion of Depth
    Anything V2 Metric Indoor Base. RGB follows the checkpoint's native patch
    embedding. A separate three-channel BIM projection is added to the RGB
    patch tokens before the class token, positional encoding, and transformer.
    """

    PATCH_SIZE = 14
    CONDITION_CHANNELS = 3
    RGB_MEAN = (0.485, 0.456, 0.406)
    RGB_STD = (0.229, 0.224, 0.225)

    def __init__(self, pretrained_model: nn.Module) -> None:
        super().__init__()
        self.dav2 = pretrained_model
        config = self.dav2.config
        backbone_config = config.backbone_config
        patch_size = int(backbone_config.patch_size)
        hidden_size = int(backbone_config.hidden_size)
        if patch_size != self.PATCH_SIZE:
            raise ValueError(f"Expected DAv2 patch size 14, got {patch_size}")
        if hidden_size != 768:
            raise ValueError(f"Expected DAv2 ViT-B hidden size 768, got {hidden_size}")
        if str(config.depth_estimation_type) != "metric":
            raise ValueError("BIM early fusion requires the official metric-depth checkpoint")
        if float(config.max_depth) != 20.0:
            raise ValueError(f"Expected indoor max depth 20 m, got {config.max_depth}")
        projection = self.dav2.backbone.embeddings.patch_embeddings.projection
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
    ) -> BIMEarlyFusionDepthAnythingV2:
        try:
            from transformers import AutoModelForDepthEstimation
        except ImportError as error:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "Depth Anything V2 early fusion requires the optional 'dav2' dependency"
            ) from error
        pretrained = AutoModelForDepthEstimation.from_pretrained(
            str(model_name_or_path),
            revision=revision,
            local_files_only=local_files_only,
        )
        return cls(pretrained)

    def enable_gradient_checkpointing(self) -> None:
        self.dav2.gradient_checkpointing_enable()

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

        embeddings_module = self.dav2.backbone.embeddings
        target_dtype = embeddings_module.patch_embeddings.projection.weight.dtype
        rgb_tokens = embeddings_module.patch_embeddings(normalized_rgb.to(dtype=target_dtype))
        bim_tokens = self.bim_condition_embed(
            bim_condition.to(dtype=self.bim_condition_embed.weight.dtype)
        )
        bim_tokens = bim_tokens.flatten(2).transpose(1, 2)
        if rgb_tokens.shape != bim_tokens.shape:
            raise RuntimeError(
                f"RGB and BIM patch tokens differ: {rgb_tokens.shape} != {bim_tokens.shape}"
            )
        tokens = rgb_tokens + bim_tokens.to(dtype=rgb_tokens.dtype)
        cls_tokens = embeddings_module.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1)
        position_encoding = (
            _deterministic_position_encoding(embeddings_module, tokens, height, width)
            if torch.are_deterministic_algorithms_enabled()
            else embeddings_module.interpolate_pos_encoding(tokens, height, width)
        )
        tokens = tokens + position_encoding
        return embeddings_module.dropout(tokens)

    def _decode_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        neck_output, patch_height, patch_width = self._neck_embeddings(
            embeddings,
            height=height,
            width=width,
        )
        return self.dav2.head(neck_output, patch_height, patch_width)

    def _neck_embeddings(
        self,
        embeddings: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[list[torch.Tensor], int, int]:
        """Return the four features after DPT reassembly and top-down fusion."""

        backbone = self.dav2.backbone
        outputs = backbone.encoder(
            embeddings,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        feature_maps: tuple[torch.Tensor, ...] = ()
        for stage, hidden_state in zip(backbone.stage_names, outputs.hidden_states, strict=True):
            if stage not in backbone.out_features:
                continue
            if backbone.config.apply_layernorm:
                hidden_state = backbone.layernorm(hidden_state)
            if backbone.config.reshape_hidden_states:
                raise RuntimeError("DAv2 checkpoint unexpectedly reshapes backbone hidden states")
            feature_maps += (hidden_state,)
        if len(feature_maps) != 4:
            raise RuntimeError(f"Expected four DAv2 feature maps, got {len(feature_maps)}")
        patch_height, patch_width = height // self.PATCH_SIZE, width // self.PATCH_SIZE
        neck_output = self.dav2.neck(feature_maps, patch_height, patch_width)
        if len(neck_output) != 4:
            raise RuntimeError(f"Expected four fused DPT neck features, got {len(neck_output)}")
        return neck_output, patch_height, patch_width

    def forward(self, rgb: torch.Tensor, bim_condition: torch.Tensor) -> torch.Tensor:
        normalized = self.normalized_rgb(rgb)
        embeddings = self._early_embeddings(normalized, bim_condition)
        return self._decode_embeddings(
            embeddings,
            height=normalized.shape[-2],
            width=normalized.shape[-1],
        )

    def pretrained_reference(self, rgb: torch.Tensor) -> torch.Tensor:
        """Run the untouched official path for initialization verification."""

        return self.dav2(pixel_values=self.normalized_rgb(rgb)).predicted_depth

    def optimizer_parameter_groups(
        self,
        *,
        encoder_lr: float,
        decoder_lr: float,
        condition_lr: float,
    ) -> list[dict[str, Any]]:
        groups = [
            {"name": "dinov2_encoder", "params": list(self.dav2.backbone.parameters()), "lr": encoder_lr},
            {
                "name": "dpt_decoder",
                "params": [*self.dav2.neck.parameters(), *self.dav2.head.parameters()],
                "lr": decoder_lr,
            },
            {
                "name": "bim_condition_projection",
                "params": list(self.bim_condition_embed.parameters()),
                "lr": condition_lr,
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
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> dict[str, Any]:
        """Run the four mandatory checks and return a serializable receipt."""

        if height % self.PATCH_SIZE or width % self.PATCH_SIZE:
            raise ValueError("Initialization audit dimensions must be divisible by 14")
        was_training = self.training
        self.eval()
        generator = torch.Generator(device=device).manual_seed(42)
        rgb = torch.rand((1, 3, height, width), generator=generator, device=device)
        condition_a = torch.randn((1, 3, height, width), generator=generator, device=device)
        condition_b = torch.randn((1, 3, height, width), generator=generator, device=device)
        with torch.inference_mode():
            projection = self.bim_condition_embed(condition_a)
            early_a = self(rgb, condition_a)
            early_b = self(rgb, condition_b)
            reference = self.pretrained_reference(rgb)

        invariance_difference = (early_a - early_b).abs().float()
        reference_difference = (early_a - reference).abs().float()
        projection_pass = bool(torch.count_nonzero(projection).item() == 0)
        invariance_pass = bool(torch.allclose(early_a, early_b, atol=atol, rtol=rtol))
        reference_pass = bool(torch.allclose(early_a, reference, atol=atol, rtol=rtol))

        decoder_max_difference = 0.0
        decoder_mean_difference_numerator = 0.0
        decoder_value_count = 0
        decoder_parameters = 0
        model_state = self.dav2.state_dict()
        with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
            checkpoint_keys = set(checkpoint.keys())
            decoder_keys = sorted(
                key for key in model_state if key.startswith(("neck.", "head."))
            )
            missing = sorted(set(decoder_keys) - checkpoint_keys)
            if missing:
                raise RuntimeError(f"Official checkpoint lacks DPT parameters: {missing[:5]}")
            for key in decoder_keys:
                actual = model_state[key].detach().float().cpu()
                expected = checkpoint.get_tensor(key).float()
                difference = (actual - expected).abs()
                decoder_max_difference = max(decoder_max_difference, float(difference.max()))
                decoder_mean_difference_numerator += float(difference.sum())
                decoder_value_count += difference.numel()
                decoder_parameters += actual.numel()
        decoder_mean_difference = decoder_mean_difference_numerator / max(decoder_value_count, 1)
        decoder_pass = decoder_max_difference == 0.0
        result = {
            "bim_projection_zero_init": {
                "pass": projection_pass,
                "max_abs_output": float(projection.abs().max()),
            },
            "bim_invariance": {
                "pass": invariance_pass,
                "max_abs_diff": float(invariance_difference.max()),
                "mean_abs_diff": float(invariance_difference.mean()),
                "atol": atol,
                "rtol": rtol,
            },
            "early_equals_pretrained_dav2": {
                "pass": reference_pass,
                "max_abs_diff": float(reference_difference.max()),
                "mean_abs_diff": float(reference_difference.mean()),
                "atol": atol,
                "rtol": rtol,
            },
            "dpt_pretrained_loading": {
                "pass": decoder_pass,
                "parameter_values": decoder_parameters,
                "max_abs_diff": decoder_max_difference,
                "mean_abs_diff": decoder_mean_difference,
            },
        }
        result["all_pass"] = all(bool(value["pass"]) for value in result.values())
        self.train(was_training)
        return result


def build_bim_condition(
    batch: Mapping[str, torch.Tensor],
    *,
    bim_log_mean: float,
    bim_log_std: float,
    disagreement_clip: float = 1.5,
) -> torch.Tensor:
    """Construct [normalized BIM log-depth, hit mask, BIM/DA3 disagreement]."""

    rgb = batch["rgb"]
    base = batch["base_depth"]
    bim = batch["bim_depth"]
    hit = batch["bim_valid"] > 0.5
    if rgb.ndim != 4 or base.ndim != 4 or bim.ndim != 4 or hit.ndim != 4:
        raise ValueError("RGB/depth/mask inputs must be batched image tensors")
    spatial_shape = rgb.shape[-2:]
    if any(tensor.shape[-2:] != spatial_shape for tensor in (base, bim, hit)):
        raise ValueError("RGB, focal-corrected DA3, BIM, and BIM mask must be pixel-aligned")
    if base.shape[1] != 1 or bim.shape[1] != 1 or hit.shape[1] != 1:
        raise ValueError("DA3/BIM/mask inputs must be single-channel")
    if bim_log_std <= 0 or not torch.isfinite(torch.tensor(bim_log_std)):
        raise ValueError("BIM log-depth normalization std must be positive and finite")
    if disagreement_clip <= 0:
        raise ValueError("BIM/DA3 disagreement clip must be positive")

    valid = hit & torch.isfinite(bim) & torch.isfinite(base) & (bim > 1e-3) & (base > 1e-3)
    invalid_da3 = ~torch.isfinite(base) | (base <= 1e-3)
    if bool(invalid_da3.any()):
        raise ValueError("Focal-corrected DA3 depth must be finite and greater than 1e-3")
    log_bim = torch.log(bim.clamp_min(1e-3))
    normalized_bim = (log_bim - float(bim_log_mean)) / float(bim_log_std)
    disagreement = (log_bim - torch.log(base)).clamp(
        -float(disagreement_clip),
        float(disagreement_clip),
    ) / float(disagreement_clip)
    zeros = torch.zeros_like(bim)
    return torch.cat(
        (
            torch.where(valid, normalized_bim, zeros),
            valid.to(dtype=bim.dtype),
            torch.where(valid, disagreement, zeros),
        ),
        dim=1,
    )
