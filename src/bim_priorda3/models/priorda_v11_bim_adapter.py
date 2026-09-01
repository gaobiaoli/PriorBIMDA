from __future__ import annotations

import math
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from bim_priorda3.config import Config, resolve_project_path

from .system import BIMPriorDA3


def _load_official_priorda_v11(
    *,
    repository: Path,
    checkpoint_path: Path,
) -> nn.Module:
    """Build the unmodified official PriorDA v1.1 ViT-B fine network."""

    repository = repository.expanduser().resolve()
    if not (repository / "prior_depth_anything/depth_anything_v2/dpt.py").is_file():
        raise FileNotFoundError(f"Official PriorDA source is unavailable: {repository}")
    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)
    # Importing the official top-level package eagerly imports its KNN coarse
    # stage. This adapter replaces that stage, so torch_cluster is neither
    # used nor required; a namespace placeholder is sufficient for import.
    if "torch_cluster" not in sys.modules:
        sys.modules["torch_cluster"] = types.ModuleType("torch_cluster")
    from prior_depth_anything.depth_anything_v2 import build_backbone
    from prior_depth_anything.depth_anything_v2.dinov2_layers import attention

    # The workspace can import xFormers but its binary has no compatible CUDA
    # kernel for the RTX 5000. Official MemEffAttention already defines the
    # mathematically equivalent PyTorch fallback; explicitly select it rather
    # than letting xFormers fail at the first forward pass.
    attention.XFORMERS_AVAILABLE = False

    model = build_backbone(depth_size="vitb", encoder_cond_dim=3)
    model.construct_aux_layers()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise TypeError("Official PriorDA v1.1 checkpoint lacks model state")
    state = {
        str(name).removeprefix("module."): value
        for name, value in payload["model"].items()
    }
    model.load_state_dict(state, strict=True)
    return model


def _separable_gaussian(
    value: torch.Tensor,
    *,
    kernel: torch.Tensor,
) -> torch.Tensor:
    radius = kernel.numel() // 2
    horizontal = functional.conv2d(value, kernel.view(1, 1, 1, -1), padding=(0, radius))
    return functional.conv2d(horizontal, kernel.view(1, 1, -1, 1), padding=(radius, 0))


def local_huber_log_scale_field(
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    global_log_scale: torch.Tensor,
    *,
    kernel_size: int = 31,
    sigma: float = 7.0,
    huber_delta: float = 0.15,
    iterations: int = 3,
    ratio_min: float = 0.2,
    ratio_max: float = 5.0,
    min_support_fraction: float = 0.01,
    spatial_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense local Huber field over measured BIM/DA3 log ratios.

    Gaussian normalized convolution replaces PriorDA's sparse-point KNN. Each
    IRLS round computes the exact Huber influence weight around the current
    local center. Pixels without sufficient local BIM support fall back to the
    frozen learned global scale rather than to an arbitrary zero scale.
    """

    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer at least 3")
    if sigma <= 0 or huber_delta <= 0 or iterations < 1:
        raise ValueError("sigma, huber_delta and iterations must be positive")
    if not 0 < ratio_min < ratio_max:
        raise ValueError("local ratio bounds are invalid")
    if not 0 <= min_support_fraction <= 1:
        raise ValueError("min_support_fraction must be in [0, 1]")
    if base_depth.shape != bim_depth.shape or bim_depth.shape != bim_valid.shape:
        raise ValueError("Local field depth/mask tensors must have identical shapes")
    if spatial_weight is not None and spatial_weight.shape != base_depth.shape:
        raise ValueError("Local spatial weights must have the same shape as depth")
    if global_log_scale.shape != (base_depth.shape[0], 1, 1, 1):
        raise ValueError("global_log_scale must have shape [B,1,1,1]")

    base = base_depth.float()
    bim = bim_depth.float()
    ratio = bim / base.clamp_min(1e-6)
    valid = (
        (bim_valid > 0)
        & torch.isfinite(base)
        & torch.isfinite(bim)
        & torch.isfinite(ratio)
        & (base > 0)
        & (bim > 0)
        & (ratio > ratio_min)
        & (ratio < ratio_max)
    )
    log_ratio = torch.where(valid, ratio.clamp_min(1e-6).log(), torch.zeros_like(ratio))
    coordinate = torch.arange(kernel_size, device=base.device, dtype=torch.float32)
    coordinate = coordinate - kernel_size // 2
    kernel = torch.exp(-0.5 * (coordinate / float(sigma)).square())
    kernel = kernel / kernel.sum()
    valid_float = valid.float()
    geometric_support = _separable_gaussian(valid_float, kernel=kernel)
    if spatial_weight is None:
        normalized_spatial_weight = torch.ones_like(valid_float)
    else:
        finite_weight = torch.where(
            torch.isfinite(spatial_weight),
            spatial_weight.float().clamp_min(0),
            torch.zeros_like(spatial_weight, dtype=torch.float32),
        )
        dimensions = tuple(range(1, finite_weight.ndim))
        mean_weight = (finite_weight * valid_float).sum(
            dim=dimensions, keepdim=True
        ) / valid_float.sum(dim=dimensions, keepdim=True).clamp_min(1.0)
        normalized_spatial_weight = finite_weight / mean_weight.clamp_min(1e-8)
    center = global_log_scale.float().expand_as(base).clone()
    supported = geometric_support >= float(min_support_fraction)
    for _ in range(iterations):
        absolute_residual = (log_ratio - center).abs()
        huber_weight = torch.where(
            absolute_residual <= float(huber_delta),
            torch.ones_like(absolute_residual),
            float(huber_delta) / absolute_residual.clamp_min(1e-6),
        )
        effective = valid_float * normalized_spatial_weight * huber_weight
        denominator = _separable_gaussian(effective, kernel=kernel)
        numerator = _separable_gaussian(effective * log_ratio, kernel=kernel)
        update_supported = supported & (denominator > 1e-6)
        estimate = numerator / denominator.clamp_min(1e-6)
        center = torch.where(update_supported, estimate, global_log_scale.expand_as(base))
    return center, supported.float()


def build_priorda_v11_bim_condition(
    *,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    global_depth: torch.Tensor,
    local_depth: torch.Tensor,
    max_disparity: float | None = None,
    normalization_fallback_depth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct PriorDA v1.1's exact [mask, global, local] input domain.

    Official v1.1 first uses the valid metric prior's per-frame min/max to
    normalize both dense depth estimates, then maps positive normalized depth
    to disparity. This normalization is essential to reuse pretrained
    ``alpha_proj`` rather than merely matching its three-channel shape.
    """

    if not (
        bim_depth.shape == bim_valid.shape == global_depth.shape == local_depth.shape
    ):
        raise ValueError("PriorDA condition tensors must have identical shapes")
    if max_disparity is not None and max_disparity <= 0:
        raise ValueError("max_disparity must be positive when provided")
    support = (
        (bim_valid > 0)
        & torch.isfinite(bim_depth)
        & (bim_depth > 0)
    )
    flattened = bim_depth.float().flatten(1)
    flat_support = support.flatten(1)
    support_count = flat_support.sum(dim=1)
    minimum = flattened.masked_fill(~flat_support, torch.inf).min(dim=1).values.view(-1, 1, 1, 1)
    maximum = flattened.masked_fill(~flat_support, -torch.inf).max(dim=1).values.view(-1, 1, 1, 1)
    unsupported = support_count < 1
    if bool(unsupported.any()):
        if normalization_fallback_depth is None:
            raise RuntimeError("PriorDA v1.1 condition requires BIM support")
        if normalization_fallback_depth.shape != bim_depth.shape:
            raise ValueError("PriorDA normalization fallback must match BIM depth shape")
        fallback = normalization_fallback_depth.float().flatten(1)
        fallback_support = torch.isfinite(fallback) & (fallback > 0)
        if bool((fallback_support.sum(dim=1) < 1).any()):
            raise RuntimeError("PriorDA normalization fallback contains no valid depth")
        fallback_minimum = fallback.masked_fill(~fallback_support, torch.inf).min(
            dim=1
        ).values.view(-1, 1, 1, 1)
        fallback_maximum = fallback.masked_fill(~fallback_support, -torch.inf).max(
            dim=1
        ).values.view(-1, 1, 1, 1)
        selector = unsupported.view(-1, 1, 1, 1)
        minimum = torch.where(selector, fallback_minimum, minimum)
        maximum = torch.where(selector, fallback_maximum, maximum)
    denominator = maximum - minimum
    denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)

    def normalized_disparity(depth: torch.Tensor) -> torch.Tensor:
        normalized_depth = (depth.float() - minimum) / denominator
        positive = normalized_depth > 0
        disparity = torch.where(
            positive,
            normalized_depth.clamp_min(1e-6).reciprocal(),
            torch.zeros_like(normalized_depth),
        )
        # This preserves the official min/max-plus-inverse domain while
        # preventing a dense estimate infinitesimally above the prior minimum
        # from producing arbitrarily large patch-embedding activations. The
        # official sparse/KNN pipeline rarely creates this BIM-specific case.
        if max_disparity is not None:
            disparity = disparity.clamp_max(float(max_disparity))
        return disparity

    condition = torch.cat(
        (
            support.float(),
            normalized_disparity(global_depth),
            normalized_disparity(local_depth),
        ),
        dim=1,
    )
    return condition, minimum, denominator


def effective_attention_top_prior(
    *,
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    bim_valid: torch.Tensor,
    scale_output: Mapping[str, torch.Tensor],
    huber_delta: float,
    top_fraction: float,
    residual_threshold: float,
    ratio_min: float,
    ratio_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a GT-free top fraction of final attention-times-Huber tokens."""

    if not 0 < top_fraction <= 1:
        raise ValueError("Effective-attention top fraction must lie in (0, 1]")
    if residual_threshold <= 0:
        raise ValueError("Effective-attention residual threshold must be positive")
    attention = scale_output["attention_token_distribution"].float()
    head_mixture = scale_output["head_mixture"].float()
    head_center = scale_output["head_log_scale"].float()
    token_valid = scale_output["attention_token_valid"].squeeze(1).bool()

    base = base_depth.float()
    bim = bim_depth.float()
    ratio = bim / base.clamp_min(1e-6)
    ratio_valid = (
        (bim_valid > 0)
        & torch.isfinite(base)
        & torch.isfinite(bim)
        & torch.isfinite(ratio)
        & (base > 0)
        & (bim > 0)
        & (ratio > float(ratio_min))
        & (ratio < float(ratio_max))
    )
    valid_float = ratio_valid.float()
    token_size = attention.shape[-2:]
    token_fraction = functional.adaptive_avg_pool2d(valid_float, token_size)
    token_numerator = functional.adaptive_avg_pool2d(
        torch.where(ratio_valid, ratio.clamp_min(1e-6).log(), 0.0),
        token_size,
    )
    token_log_ratio = (token_numerator / token_fraction.clamp_min(1e-6)).squeeze(1)
    residual = token_log_ratio[:, None] - head_center[:, :, None, None]
    robust_weight = torch.rsqrt(1.0 + (residual / float(huber_delta)).square())
    effective_per_head = attention * robust_weight
    effective_per_head = effective_per_head / effective_per_head.flatten(2).sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8).unsqueeze(-1)
    distribution = (effective_per_head * head_mixture[:, :, None, None]).sum(dim=1)

    selected_tokens = torch.zeros_like(token_valid)
    for batch_index in range(distribution.shape[0]):
        flat_valid = token_valid[batch_index].flatten()
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        if not valid_indices.numel():
            continue
        count = max(1, math.ceil(float(top_fraction) * int(valid_indices.numel())))
        ranked = torch.topk(distribution[batch_index].flatten()[valid_indices], count).indices
        selected_tokens[batch_index].view(-1)[valid_indices[ranked]] = True

    output_size = base.shape[-2:]
    selected_pixels = functional.interpolate(
        selected_tokens[:, None].float(), size=output_size, mode="nearest"
    ).bool()
    pixel_weight = functional.interpolate(
        distribution[:, None],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    global_log_scale = scale_output["log_scale"].float()
    ratio_residual = (ratio.clamp_min(1e-6).log() - global_log_scale).abs()
    trusted = ratio_valid & selected_pixels & (ratio_residual < float(residual_threshold))
    return trusted.float(), pixel_weight * trusted.float(), distribution[:, None]


class FrozenHuberPriorDAV11BIM(nn.Module):
    """Frozen learned Huber scale with the unmodified PriorDA v1.1 fine net."""

    def __init__(
        self,
        scale_system: BIMPriorDA3,
        priorda: nn.Module,
        *,
        local_config: Mapping[str, Any],
        output_max_depth_m: float = 128.0,
        input_size: int = 518,
        condition_max_disparity: float | None = None,
        effective_attention_prior_config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.scale_system = scale_system
        self.priorda = priorda
        self.local_config = dict(local_config)
        self.output_max_depth_m = float(output_max_depth_m)
        self.input_size = int(input_size)
        self.condition_max_disparity = (
            None if condition_max_disparity is None else float(condition_max_disparity)
        )
        self.effective_attention_prior_config = (
            None
            if effective_attention_prior_config is None
            else dict(effective_attention_prior_config)
        )
        for parameter in self.scale_system.parameters():
            parameter.requires_grad_(False)
        self.scale_system.eval()

    @classmethod
    def from_checkpoints(
        cls,
        cfg: Config,
        *,
        scale_checkpoint: Mapping[str, Any],
        priorda_checkpoint_path: Path,
    ) -> FrozenHuberPriorDAV11BIM:
        scale_system = BIMPriorDA3(cfg)
        scale_system.load_state_dict(scale_checkpoint["model"], strict=True)
        prior_cfg = cfg.model.priorda_v11
        repository = resolve_project_path(cfg, prior_cfg.official_repository)
        priorda = _load_official_priorda_v11(
            repository=repository,
            checkpoint_path=priorda_checkpoint_path,
        )
        return cls(
            scale_system,
            priorda,
            local_config=cfg.model.local_huber_field,
            output_max_depth_m=float(cfg.model.output_max_depth_m),
            input_size=int(prior_cfg.input_size),
            condition_max_disparity=float(prior_cfg.condition_max_disparity),
            effective_attention_prior_config=(
                cfg.model.get("effective_attention_prior")
                if hasattr(cfg.model, "get")
                else None
            ),
        )

    def train(self, mode: bool = True) -> FrozenHuberPriorDAV11BIM:
        super().train(mode)
        self.scale_system.eval()
        return self

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor | str]:
        base = batch["base_depth"]
        with torch.no_grad():
            scale_output = self.scale_system._estimate_attention_scale(dict(batch), base)
            global_log_scale = scale_output["log_scale"].float()
            global_depth = base * scale_output["scale"].to(dtype=base.dtype)
            condition_bim_valid = batch["bim_valid"]
            local_spatial_weight = None
            effective_attention_distribution = None
            if self.effective_attention_prior_config is not None:
                prior_cfg = self.effective_attention_prior_config
                head = self.scale_system.attention_scale
                if head is None or bool(head.iterative_refresh_attention):
                    raise RuntimeError(
                        "Effective-attention PriorDA input requires fixed attention"
                    )
                (
                    condition_bim_valid,
                    local_spatial_weight,
                    effective_attention_distribution,
                ) = effective_attention_top_prior(
                    base_depth=base,
                    bim_depth=batch["bim_depth"],
                    bim_valid=batch["bim_valid"],
                    scale_output=scale_output,
                    huber_delta=float(head.huber_delta),
                    top_fraction=float(prior_cfg.get("top_fraction", 0.25)),
                    residual_threshold=float(
                        prior_cfg.get("residual_threshold", 2.0 * float(head.huber_delta))
                    ),
                    ratio_min=math.exp(float(head.log_ratio_min)),
                    ratio_max=math.exp(float(head.log_ratio_max)),
                )
            local_log_scale, local_support = local_huber_log_scale_field(
                base,
                batch["bim_depth"],
                condition_bim_valid,
                global_log_scale,
                spatial_weight=local_spatial_weight,
                **self.local_config,
            )
            local_depth = base * torch.exp(local_log_scale).to(dtype=base.dtype)
            condition, prior_minimum, prior_range = build_priorda_v11_bim_condition(
                bim_depth=batch["bim_depth"],
                bim_valid=condition_bim_valid,
                global_depth=global_depth,
                local_depth=local_depth,
                max_disparity=self.condition_max_disparity,
                normalization_fallback_depth=global_depth,
            )
        rgb_uint8 = (batch["rgb"].float().clamp(0, 1) * 255.0).round().to(torch.uint8)
        normalized_disparity = self.priorda(
            rgb_uint8,
            self.input_size,
            condition=condition,
            device=str(rgb_uint8.device),
        )
        normalized_depth = torch.where(
            normalized_disparity > 0,
            normalized_disparity.clamp_min(1e-6).reciprocal(),
            torch.zeros_like(normalized_disparity),
        )
        depth = (normalized_depth.float() * prior_range + prior_minimum).clamp(
            1e-3,
            self.output_max_depth_m,
        )
        return {
            "depth": depth,
            "normalized_disparity": normalized_disparity,
            "scaled_depth": global_depth,
            "local_depth": local_depth,
            "scale": scale_output["scale"],
            "log_scale": global_log_scale,
            "scale_iteration_log_scales": scale_output.get("iteration_log_scales"),
            "condition": condition,
            "local_log_scale": local_log_scale,
            "local_support": local_support,
            "condition_bim_valid": condition_bim_valid,
            "effective_attention_distribution": effective_attention_distribution,
            "prior_minimum": prior_minimum,
            "prior_range": prior_range,
            "condition_semantics": "[BIM support mask, normalized global disparity, normalized local disparity]",
        }

    def optimizer_parameter_groups(
        self,
        *,
        backbone_lr: float,
        alpha_projection_lr: float,
    ) -> list[dict[str, Any]]:
        alpha = list(self.priorda.pretrained.patch_embed.alpha_proj.parameters())
        alpha_ids = {id(parameter) for parameter in alpha}
        backbone = [
            parameter
            for parameter in self.priorda.parameters()
            if parameter.requires_grad and id(parameter) not in alpha_ids
        ]
        groups = [
            {"name": "priorda_v11_pretrained", "params": backbone, "lr": backbone_lr},
            {"name": "priorda_v11_alpha_proj", "params": alpha, "lr": alpha_projection_lr},
        ]
        ids = [id(parameter) for group in groups for parameter in group["params"]]
        expected = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise RuntimeError("PriorDA optimizer groups do not match trainable parameters")
        return groups
