from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from bim_priorda3.baselines import (
    BIM_INVALID_DEPTH_ATOL,
    ROBUST_LOG_CAP_SCALE_ESTIMATOR,
    resolve_scale_estimator_config,
    robust_scale_and_local_features,
)
from bim_priorda3.config import Config

from .attention_scale import AttentiveBIMScaleHead
from .full_regression_scale import FullRegressionIterativeScaleHead
from .refiner import ScaleAnchoredDepthRefiner
from .rgb_dinov2_full_regression_scale import (
    RGBDINOFullRegressionIterativeScaleHead,
)


def safe_log(depth: torch.Tensor) -> torch.Tensor:
    return torch.log(depth.clamp_min(1e-3))


class BIMPriorDA3(nn.Module):
    """Refine scale-corrected DA3 depth with RGB and raw BIM conditions."""

    # Kept as a serialized identifier so existing V5 checkpoints remain
    # configuration-compatible. It no longer selects between implementations.
    SUPPORTED_VARIANT = "prior_conditioned_v4"
    GEOMETRY_CHANNELS = 4
    BIM_CHANNELS = 8
    ATTENTION_SCALE_CHANNELS = 13
    FULL_REGRESSION_INPUT_JOINT = "joint"
    FULL_REGRESSION_INPUT_RGB_DA3_UNIT_BIM = "rgb_da3_unit_bim"
    RESIDUAL_ANCHOR_SCALED = "scaled_depth"
    RESIDUAL_ROUTING_FRAME_AND_LOW = "frame_and_low"
    RESIDUAL_ROUTING_FRAME_ONLY = "frame_only"

    def __init__(
        self,
        cfg: Config,
        *,
        da3_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        model = cfg.model
        self.max_depth = float(cfg.data.max_depth)
        self.output_max_depth = float(
            model.get("output_max_depth_m", self.max_depth * 2.0)
        )
        if not math.isfinite(self.output_max_depth) or self.output_max_depth <= 0:
            raise ValueError("model.output_max_depth_m must be finite and positive")
        self.variant = str(model.get("variant", ""))
        if self.variant != self.SUPPORTED_VARIANT:
            raise ValueError(
                f"Unsupported model variant {self.variant!r}; expected {self.SUPPORTED_VARIANT!r}"
            )
        self.use_rgb_condition = bool(model.get("use_rgb_condition", True))
        self.use_bim_condition = bool(model.get("use_bim_condition", True))
        self.use_frame_residual = bool(model.get("use_frame_residual", True))
        self.use_low_residual = bool(model.get("use_low_residual", True))
        self.depth_aware_residual_routing = bool(model.get("depth_aware_residual_routing", False))
        self.residual_routing_depth = float(model.get("residual_routing_depth", 1.0))
        self.residual_routing_temperature = float(model.get("residual_routing_temperature", 0.1))
        self.residual_routing_scope = str(
            model.get(
                "residual_routing_scope",
                self.RESIDUAL_ROUTING_FRAME_AND_LOW,
            )
        )
        if self.residual_routing_scope not in {
            self.RESIDUAL_ROUTING_FRAME_AND_LOW,
            self.RESIDUAL_ROUTING_FRAME_ONLY,
        }:
            raise ValueError("model.residual_routing_scope must be 'frame_and_low' or 'frame_only'")
        self.e2e_da3_config = Config(model.get("e2e_da3", {}))
        self.e2e_da3_enabled = bool(self.e2e_da3_config.get("enabled", False))
        self.da3_feature_fusion_config = Config(model.get("da3_feature_fusion", {}))
        shared_da3_feature_fusion = bool(
            self.da3_feature_fusion_config.get("enabled", False)
        )
        self.da3_feature_scale_enabled = bool(
            self.da3_feature_fusion_config.get(
                "scale_enabled",
                shared_da3_feature_fusion,
            )
        )
        self.da3_feature_refiner_enabled = bool(
            self.da3_feature_fusion_config.get(
                "refiner_enabled",
                shared_da3_feature_fusion,
            )
        )
        self.da3_feature_fusion_enabled = (
            self.da3_feature_scale_enabled or self.da3_feature_refiner_enabled
        )
        self.da3_feature_channels = (
            int(self.da3_feature_fusion_config.get("channels", 1024))
            if self.da3_feature_fusion_enabled
            else 0
        )
        self.da3_feature_layers = tuple(
            int(value)
            for value in self.da3_feature_fusion_config.get("layers", (11, 23))
        )
        if self.da3_feature_fusion_enabled:
            if self.e2e_da3_enabled:
                raise ValueError(
                    "Cached DA3 feature fusion and end-to-end DA3 are mutually exclusive"
                )
            if self.da3_feature_channels < 1:
                raise ValueError("model.da3_feature_fusion.channels must be positive")
            if len(self.da3_feature_layers) != 2:
                raise ValueError("model.da3_feature_fusion.layers must contain two layers")
        self.dinov2_feature_config = Config(model.get("dinov2_feature_fusion", {}))
        self.dinov2_feature_fusion_enabled = bool(
            self.dinov2_feature_config.get("enabled", False)
        )
        self.dinov2_feature_channels = int(
            self.dinov2_feature_config.get("channels", 768)
        )
        if self.dinov2_feature_fusion_enabled and self.dinov2_feature_channels < 1:
            raise ValueError("model.dinov2_feature_fusion.channels must be positive")
        self.additive_residual_config = Config(model.get("additive_residual", {}))
        self.additive_residual_enabled = bool(
            self.additive_residual_config.get("enabled", False)
        )
        self.max_additive_residual_m = float(
            self.additive_residual_config.get("max_residual_m", 0.0)
        )
        if self.additive_residual_enabled and self.max_additive_residual_m <= 0:
            raise ValueError(
                "model.additive_residual.max_residual_m must be positive when enabled"
            )
        self.detail_reliability_gate_config = Config(
            model.get("detail_reliability_gate", {})
        )
        self.detail_reliability_gate_enabled = bool(
            self.detail_reliability_gate_config.get("enabled", False)
        )
        self.detail_reliability_gate_floor = float(
            self.detail_reliability_gate_config.get("floor", 0.0)
        )
        self.detail_reliability_gate_detach = bool(
            self.detail_reliability_gate_config.get("detach", True)
        )
        if not 0.0 <= self.detail_reliability_gate_floor < 1.0:
            raise ValueError("model.detail_reliability_gate.floor must be in [0, 1)")
        self.da3: nn.Module | None = None
        self._da3_trainable_module_names: tuple[str, ...] = ()
        configured_scale = model.get("scale_estimator")
        deprecated_scale_fields = {
            "scale_quantile",
            "scale_ratio_min",
            "scale_ratio_max",
            "scale_min_samples",
        }.intersection(self.e2e_da3_config)
        if deprecated_scale_fields:
            raise ValueError(
                "Deprecated model.e2e_da3 scale fields were removed; configure the "
                "single shared model.scale_estimator instead: "
                f"{sorted(deprecated_scale_fields)}"
            )
        self.scale_estimator_config = resolve_scale_estimator_config(configured_scale)
        if self.scale_estimator_config["name"] != ROBUST_LOG_CAP_SCALE_ESTIMATOR:
            raise ValueError(
                "The public model uses one universal scale estimator; "
                "model.scale_estimator.name must be log_upper_cap_v1"
            )
        if "residual_anchor_mode" in model:
            raise ValueError(
                "model.residual_anchor_mode was removed: learned residuals always "
                "refine the universally scaled DA3 depth; BIM-direct is a baseline"
            )
        self.residual_anchor_mode = self.RESIDUAL_ANCHOR_SCALED
        self.da3_scale_ratio_min = float(self.scale_estimator_config["ratio_min"])
        self.da3_scale_ratio_max = float(self.scale_estimator_config["ratio_max"])
        self.da3_scale_min_samples = int(self.scale_estimator_config["min_samples"])
        self.attention_scale_config = Config(model.get("attention_scale", {}))
        self.attention_scale_enabled = bool(self.attention_scale_config.get("enabled", False))
        self.attention_scale_use_base_confidence = bool(
            self.attention_scale_config.get("use_base_confidence", True)
        )
        self.attention_scale_use_bim_normals = bool(
            self.attention_scale_config.get("use_bim_normals", True)
        )
        self.attention_scale_use_bim_edge = bool(
            self.attention_scale_config.get("use_bim_edge", True)
        )
        self.attention_scale_use_deterministic_fallback_input = bool(
            self.attention_scale_config.get("use_deterministic_fallback_input", True)
        )
        self.full_regression_input_mode = self.FULL_REGRESSION_INPUT_JOINT
        self.attention_scale_input_channels = self.ATTENTION_SCALE_CHANNELS
        if self.attention_scale_enabled and self.e2e_da3_enabled:
            raise ValueError(
                "The attentive-scale candidate currently requires frozen/cached DA3; "
                "disable model.e2e_da3"
            )
        self.attention_scale: (
            AttentiveBIMScaleHead
            | FullRegressionIterativeScaleHead
            | RGBDINOFullRegressionIterativeScaleHead
            | None
        ) = None
        if self.attention_scale_enabled:
            attention_estimator = str(
                self.attention_scale_config.get("estimator", "pseudo_huber_attention_v1")
            )
            if (
                attention_estimator
                != RGBDINOFullRegressionIterativeScaleHead.ESTIMATOR_NAME
            ):
                if not self.attention_scale_use_base_confidence:
                    self.attention_scale_input_channels -= 1
                if not self.attention_scale_use_bim_normals:
                    self.attention_scale_input_channels -= 3
                if not self.attention_scale_use_bim_edge:
                    self.attention_scale_input_channels -= 1
            if attention_estimator == FullRegressionIterativeScaleHead.ESTIMATOR_NAME:
                self.full_regression_input_mode = str(
                    self.attention_scale_config.get(
                        "input_mode",
                        self.FULL_REGRESSION_INPUT_JOINT,
                    )
                )
                supported_input_modes = {
                    self.FULL_REGRESSION_INPUT_JOINT,
                    self.FULL_REGRESSION_INPUT_RGB_DA3_UNIT_BIM,
                }
                if self.full_regression_input_mode not in supported_input_modes:
                    raise ValueError(
                        "model.attention_scale.input_mode must be one of "
                        f"{sorted(supported_input_modes)}"
                    )
                if (
                    self.full_regression_input_mode
                    == self.FULL_REGRESSION_INPUT_RGB_DA3_UNIT_BIM
                ):
                    if self.attention_scale_use_base_confidence:
                        raise ValueError(
                            "rgb_da3_unit_bim input mode requires "
                            "model.attention_scale.use_base_confidence=false"
                        )
                    if self.da3_feature_scale_enabled:
                        raise ValueError(
                            "rgb_da3_unit_bim input mode forbids cached DA3 latent "
                            "features in the scale estimator"
                        )
                    self.attention_scale_input_channels = 4
            shared_arguments = {
                "in_channels": self.attention_scale_input_channels,
                "hidden_channels": int(
                    self.attention_scale_config.get("hidden_channels", 24)
                ),
                "attention_heads": int(
                    self.attention_scale_config.get("attention_heads", 4)
                ),
                "min_support": int(
                    self.attention_scale_config.get(
                        "min_support",
                        self.da3_scale_min_samples,
                    )
                ),
                "ratio_min": float(
                    self.attention_scale_config.get(
                        "ratio_min",
                        self.da3_scale_ratio_min,
                    )
                ),
                "ratio_max": float(
                    self.attention_scale_config.get(
                        "ratio_max",
                        self.da3_scale_ratio_max,
                    )
                ),
                "token_dropout_probability": float(
                    self.attention_scale_config.get(
                        "token_dropout_probability",
                        0.10,
                    )
                ),
                "da3_feature_channels": (
                    self.da3_feature_channels if self.da3_feature_scale_enabled else 0
                ),
            }
            if (
                attention_estimator
                == RGBDINOFullRegressionIterativeScaleHead.ESTIMATOR_NAME
            ):
                if not self.dinov2_feature_fusion_enabled:
                    raise ValueError(
                        "RGB+DINO full regression requires "
                        "model.dinov2_feature_fusion.enabled=true"
                    )
                if self.da3_feature_scale_enabled:
                    raise ValueError(
                        "RGB+DINO full regression must not enable cached DA3 features "
                        "for the scale estimator"
                    )
                self.attention_scale = RGBDINOFullRegressionIterativeScaleHead(
                    geometry_channels=int(
                        self.attention_scale_config.get("geometry_channels", 7)
                    ),
                    rgb_base_channels=int(
                        self.attention_scale_config.get("rgb_base_channels", 24)
                    ),
                    fusion_channels=int(
                        self.attention_scale_config.get("fusion_channels", 96)
                    ),
                    dinov2_channels=self.dinov2_feature_channels,
                    attention_heads=int(
                        self.attention_scale_config.get("attention_heads", 4)
                    ),
                    min_support=int(
                        self.attention_scale_config.get(
                            "min_support", self.da3_scale_min_samples
                        )
                    ),
                    ratio_min=float(
                        self.attention_scale_config.get(
                            "ratio_min", self.da3_scale_ratio_min
                        )
                    ),
                    ratio_max=float(
                        self.attention_scale_config.get(
                            "ratio_max", self.da3_scale_ratio_max
                        )
                    ),
                    token_dropout_probability=float(
                        self.attention_scale_config.get(
                            "token_dropout_probability", 0.10
                        )
                    ),
                    iterative_updates=int(
                        self.attention_scale_config.get("iterative_updates", 3)
                    ),
                    iterative_hidden_channels=int(
                        self.attention_scale_config.get(
                            "iterative_hidden_channels", 32
                        )
                    ),
                    delta_hidden_channels=int(
                        self.attention_scale_config.get(
                            "delta_hidden_channels", 64
                        )
                    ),
                    iterative_max_log_update=float(
                        self.attention_scale_config.get(
                            "iterative_max_log_update", 0.15
                        )
                    ),
                )
            elif attention_estimator == FullRegressionIterativeScaleHead.ESTIMATOR_NAME:
                self.attention_scale = FullRegressionIterativeScaleHead(
                    **shared_arguments,
                    iterative_updates=int(
                        self.attention_scale_config.get("iterative_updates", 3)
                    ),
                    iterative_hidden_channels=int(
                        self.attention_scale_config.get(
                            "iterative_hidden_channels", 32
                        )
                    ),
                    delta_hidden_channels=int(
                        self.attention_scale_config.get(
                            "delta_hidden_channels", 64
                        )
                    ),
                    iterative_max_log_update=float(
                        self.attention_scale_config.get(
                            "iterative_max_log_update", 0.15
                        )
                    ),
                )
            elif attention_estimator == "pseudo_huber_attention_v1":
                self.attention_scale = AttentiveBIMScaleHead(
                    **shared_arguments,
                    huber_delta=float(
                        self.attention_scale_config.get("huber_delta", 0.15)
                    ),
                    fallback_gate_bias=float(
                        self.attention_scale_config.get("fallback_gate_bias", -1.5)
                    ),
                    bounded_log_scale_residual=float(
                        self.attention_scale_config.get(
                            "bounded_log_scale_residual", 0.0
                        )
                    ),
                    residual_hidden_channels=int(
                        self.attention_scale_config.get("residual_hidden_channels", 32)
                    ),
                    iterative_updates=int(
                        self.attention_scale_config.get("iterative_updates", 0)
                    ),
                    iterative_hidden_channels=int(
                        self.attention_scale_config.get(
                            "iterative_hidden_channels", 32
                        )
                    ),
                    iterative_initial_log_scale=float(
                        self.attention_scale_config.get(
                            "iterative_initial_log_scale",
                            0.0,
                        )
                    ),
                    iterative_damping=list(
                        self.attention_scale_config.get("iterative_damping", [])
                    )
                    or None,
                    iterative_max_log_update=float(
                        self.attention_scale_config.get(
                            "iterative_max_log_update",
                            0.15,
                        )
                    ),
                    iterative_refresh_attention=bool(
                        self.attention_scale_config.get(
                            "iterative_refresh_attention",
                            True,
                        )
                    ),
                    use_fallback_gate=bool(
                        self.attention_scale_config.get("use_fallback_gate", True)
                    ),
                )
            else:
                raise ValueError(
                    "Unknown model.attention_scale.estimator "
                    f"{attention_estimator!r}"
                )
        self.attention_scale_equivariance_probability = float(
            self.attention_scale_config.get("equivariance_probability", 0.0)
        )
        self.attention_scale_equivariance_log_range = float(
            self.attention_scale_config.get("equivariance_log_range", 0.20)
        )
        if not 0.0 <= self.attention_scale_equivariance_probability <= 1.0:
            raise ValueError("model.attention_scale.equivariance_probability must be in [0, 1]")
        if self.attention_scale_equivariance_log_range < 0:
            raise ValueError("model.attention_scale.equivariance_log_range must be non-negative")
        self.register_buffer(
            "_da3_rgb_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_da3_rgb_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.refiner = ScaleAnchoredDepthRefiner(
            rgb_channels=3,
            geometry_channels=self.GEOMETRY_CHANNELS,
            bim_channels=self.BIM_CHANNELS,
            base_channels=int(model.base_channels),
            max_frame_log_residual=float(model.get("max_frame_log_residual", 0.20)),
            max_low_log_residual=float(model.get("max_low_log_residual", 0.25)),
            max_detail_log_residual=float(model.get("max_detail_log_residual", 0.15)),
            max_total_log_residual=float(model.get("max_total_log_residual", 0.45)),
            gate_bim_adapters=bool(model.get("gate_bim_adapters", False)),
            bim_adapter_gate_floor=float(model.get("bim_adapter_gate_floor", 0.25)),
            bim_adapter_gate_use_rgb=bool(
                model.get("bim_adapter_gate_use_rgb", False)
            ),
            da3_feature_channels=(
                self.da3_feature_channels if self.da3_feature_refiner_enabled else 0
            ),
            additive_residual_enabled=self.additive_residual_enabled,
            max_additive_residual_m=self.max_additive_residual_m,
            additive_detach_shared_features=bool(
                self.additive_residual_config.get("detach_shared_features", True)
            ),
        )
        if self.e2e_da3_enabled:
            self.da3 = self._load_da3_model(cfg, da3_model)
            self._configure_da3_trainable_scope()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self._validate_bim_depth_mask_contract(batch)
        if not self.e2e_da3_enabled:
            return self._forward_scale_refinement(batch)
        request_live_bim_direct = batch.get(
            "request_live_bim_direct",
            False,
        )
        if not isinstance(request_live_bim_direct, bool):
            raise TypeError("batch['request_live_bim_direct'] must be a non-tensor bool")
        live_batch, da3_receipt = self._build_live_da3_batch(batch)
        output = self._forward_scale_refinement(live_batch)
        output.update(da3_receipt)
        if self.training or request_live_bim_direct:
            live_direct = self._configured_fixed_bim_direct(
                output["base_depth"],
                batch["bim_depth"],
                batch["bim_valid"],
            )
            output["live_bim_direct"] = live_direct
            if self.scale_estimator_config["name"] == ROBUST_LOG_CAP_SCALE_ESTIMATOR:
                output["live_robust_bim_direct"] = live_direct
            else:
                output["live_legacy_bim_direct_q45"] = live_direct
        output["uses_live_da3"] = True
        return output

    @staticmethod
    def _validate_bim_depth_mask_contract(
        batch: dict[str, torch.Tensor],
    ) -> None:
        """Require one BIM support definition at every model entry point."""

        bim_depth = batch["bim_depth"]
        bim_valid = batch["bim_valid"]
        if not torch.is_tensor(bim_depth) or not torch.is_tensor(bim_valid):
            raise TypeError("bim_depth and bim_valid must be tensors")
        if bim_depth.shape != bim_valid.shape:
            raise ValueError(
                "bim_depth and bim_valid shapes differ: "
                f"{tuple(bim_depth.shape)} != {tuple(bim_valid.shape)}"
            )
        if not bool(torch.isfinite(bim_valid).all()):
            raise ValueError("bim_valid contains non-finite values")
        invalid = bim_valid <= 0
        violations = invalid & (
            ~torch.isfinite(bim_depth) | (bim_depth.abs() > BIM_INVALID_DEPTH_ATOL)
        )
        if bool(violations.any()):
            finite_invalid = bim_depth[invalid & torch.isfinite(bim_depth)].abs()
            maximum = float(finite_invalid.max()) if finite_invalid.numel() else float("nan")
            raise ValueError(
                "bim_depth must be zero within "
                f"atol={BIM_INVALID_DEPTH_ATOL:g} wherever bim_valid <= 0; "
                f"violations={int(violations.sum())}, "
                f"max_abs_finite={maximum:g}"
            )

    def train(self, mode: bool = True) -> BIMPriorDA3:
        """Keep the frozen DA3 encoder deterministic during refiner training."""
        super().train(mode)
        if self.da3 is None:
            return self
        self.da3.eval()
        if mode:
            for name in self._da3_trainable_module_names:
                self.da3.get_submodule(name).train()
        return self

    def trainable_parameter_groups(
        self,
    ) -> dict[str, list[nn.Parameter]]:
        """Return independently optimizable DA3 and non-DA3 parameters."""
        groups = {"da3": [], "non_da3": []}
        if self.attention_scale_enabled:
            groups["attention_scale"] = []
        if self.additive_residual_enabled:
            groups["additive_residual"] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("da3."):
                key = "da3"
            elif self.attention_scale_enabled and name.startswith("attention_scale."):
                key = "attention_scale"
            elif self.additive_residual_enabled and name.startswith("refiner.additive_"):
                key = "additive_residual"
            else:
                key = "non_da3"
            groups[key].append(parameter)
        return groups

    def trainable_parameter_names(self) -> dict[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {"da3": [], "non_da3": []}
        if self.attention_scale_enabled:
            groups["attention_scale"] = []
        if self.additive_residual_enabled:
            groups["additive_residual"] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("da3."):
                key = "da3"
            elif self.attention_scale_enabled and name.startswith("attention_scale."):
                key = "attention_scale"
            elif self.additive_residual_enabled and name.startswith("refiner.additive_"):
                key = "additive_residual"
            else:
                key = "non_da3"
            groups[key].append(name)
        return {key: tuple(names) for key, names in groups.items()}

    def _load_da3_model(
        self,
        cfg: Config,
        injected_model: nn.Module | None,
    ) -> nn.Module:
        if injected_model is not None:
            candidate = injected_model
        else:
            try:
                from depth_anything_3.api import DepthAnything3
            except ImportError as exc:
                raise RuntimeError(
                    "End-to-end DA3 is enabled, but depth-anything-3 is not "
                    "installed. Install the optional 'da3' dependency."
                ) from exc
            source_value = self.e2e_da3_config.get(
                "local_model_path",
                self.e2e_da3_config.get(
                    "model_name",
                    cfg.data.get(
                        "da3_model",
                        "depth-anything/da3metric-large",
                    ),
                ),
            )
            source = Path(str(source_value)).expanduser()
            if not source.is_absolute():
                project_source = Path(cfg.project_root) / source
                if project_source.exists():
                    source = project_source
            source_arg = str(source.resolve()) if source.exists() else str(source_value)
            load_kwargs: dict[str, Any] = {
                "local_files_only": bool(self.e2e_da3_config.get("local_files_only", True))
            }
            cache_dir = self.e2e_da3_config.get("cache_dir")
            if cache_dir:
                load_kwargs["cache_dir"] = str(Path(str(cache_dir)).expanduser())
            revision = self.e2e_da3_config.get("revision")
            if revision:
                load_kwargs["revision"] = str(revision)
            candidate = DepthAnything3.from_pretrained(
                source_arg,
                **load_kwargs,
            )
        if hasattr(candidate, "model"):
            candidate = candidate.model
        if not isinstance(candidate, nn.Module):
            raise TypeError("The configured DA3 model is not a torch module")
        required = ("backbone", "head", "_process_depth_head")
        missing = [name for name in required if not hasattr(candidate, name)]
        if missing:
            raise TypeError("DA3 model lacks required Metric-Large modules: " + ", ".join(missing))
        return candidate

    def _configure_da3_trainable_scope(self) -> None:
        assert self.da3 is not None
        for parameter in self.da3.parameters():
            parameter.requires_grad_(False)
        scope = str(self.e2e_da3_config.get("trainable_scope", "last_stage"))
        if scope == "frozen":
            names: tuple[str, ...] = ()
        elif scope == "last_stage":
            names = (
                "head.scratch.refinenet1",
                "head.scratch.output_conv1",
                "head.scratch.output_conv2",
            )
        elif scope == "full_head":
            names = ("head",)
        else:
            raise ValueError(
                "model.e2e_da3.trainable_scope must be one of "
                "'frozen', 'last_stage', or 'full_head'"
            )
        for name in names:
            try:
                module = self.da3.get_submodule(name)
            except AttributeError as exc:
                raise TypeError(f"DA3 Metric-Large model lacks trainable module {name!r}") from exc
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if scope == "full_head" and not bool(self.e2e_da3_config.get("train_sky_head", False)):
            sky_name = "head.scratch.sky_output_conv2"
            try:
                sky_head = self.da3.get_submodule(sky_name)
            except AttributeError:
                sky_head = None
            if sky_head is not None:
                for parameter in sky_head.parameters():
                    parameter.requires_grad_(False)
        self._da3_trainable_module_names = names
        self.da3.eval()

    def _forward_da3_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        assert self.da3 is not None
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("DA3 RGB input must have shape [B, 3, H, W]")
        height, width = rgb.shape[-2:]
        patch_size = int(getattr(self.da3, "PATCH_SIZE", 14))
        if height % patch_size or width % patch_size:
            raise ValueError(
                f"DA3 input {height}x{width} must be divisible by patch size {patch_size}"
            )
        normalized = (rgb.float().clamp(0.0, 1.0) - self._da3_rgb_mean) / self._da3_rgb_std
        images = normalized.unsqueeze(1)
        backbone_amp = rgb.device.type == "cuda"
        backbone_dtype = (
            torch.bfloat16 if backbone_amp and torch.cuda.is_bf16_supported() else torch.float16
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=rgb.device.type,
                dtype=backbone_dtype,
                enabled=backbone_amp,
            ),
        ):
            features, _ = self.da3.backbone(
                images,
                cam_token=None,
                export_feat_layers=[],
                ref_view_strategy="saddle_balanced",
            )
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            raw_output = self.da3._process_depth_head(
                features,
                height,
                width,
            )
            if hasattr(self.da3, "_process_mono_sky_estimation"):
                raw_output = self.da3._process_mono_sky_estimation(raw_output)
        depth = raw_output["depth"] if isinstance(raw_output, dict) else raw_output.depth
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise RuntimeError("DA3 Metric-Large depth must have shape [B, 1, H, W]")
        return depth.clamp_min(1e-3)

    @staticmethod
    def _depth_confidence(depth: torch.Tensor) -> torch.Tensor:
        """Differentiable-operator replica of the cached confidence proxy.

        The proxy is intentionally detached: it conditions the BIM refiner but
        is not a second gradient path into the DA3 decoder.
        """
        with (
            torch.no_grad(),
            torch.autocast(
                device_type=depth.device.type,
                enabled=False,
            ),
        ):
            log_depth = safe_log(depth.detach()).float()
            kernel = log_depth.new_tensor(
                (
                    (0.0, 1.0, 0.0),
                    (1.0, -4.0, 1.0),
                    (0.0, 1.0, 0.0),
                )
            ).view(1, 1, 3, 3)
            padded = functional.pad(
                log_depth,
                (1, 1, 1, 1),
                mode="reflect",
            )
            laplacian = functional.conv2d(padded, kernel).abs()
            scale = torch.quantile(
                laplacian.flatten(1),
                0.9,
                dim=1,
            ).view(-1, 1, 1, 1)
            confidence = torch.exp(-laplacian / (scale + 1e-6))
        return confidence.to(dtype=depth.dtype)

    def _robust_bim_scale(
        self,
        depth: torch.Tensor,
        bim_depth: torch.Tensor,
        bim_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the canonical detached model-input scale estimator."""
        scales = []
        support_counts = []
        quantile_receipts = []
        cap_receipts = []
        with torch.no_grad():
            reference = depth.detach()
            valid = (
                torch.isfinite(reference)
                & torch.isfinite(bim_depth)
                & (reference > 0)
                & (bim_depth > 0)
                & (bim_valid > 0)
            )
            ratios = bim_depth / reference.clamp_min(1e-6)
            valid &= (ratios > self.da3_scale_ratio_min) & (ratios < self.da3_scale_ratio_max)
            for sample_index in range(reference.shape[0]):
                values = ratios[sample_index][valid[sample_index]]
                support_counts.append(values.new_tensor(values.numel()))
                if values.numel() < self.da3_scale_min_samples:
                    scale = values.new_tensor(1.0)
                    quantiles = values.new_full((3,), float("nan"))
                    cap_flags = torch.zeros(
                        2,
                        dtype=torch.bool,
                        device=values.device,
                    )
                elif self.scale_estimator_config["name"] == ROBUST_LOG_CAP_SCALE_ESTIMATOR:
                    log_quantiles = torch.quantile(
                        values.float().log(),
                        values.new_tensor((0.10, 0.25, 0.45)).float(),
                    )
                    q10, q25, q45 = log_quantiles.unbind()
                    q10_bound = q10 + float(self.scale_estimator_config["q10_log_cap"])
                    q25_bound = q25 + float(self.scale_estimator_config["q25_log_cap"])
                    robust_log_scale = torch.minimum(
                        q45,
                        torch.minimum(q10_bound, q25_bound),
                    )
                    scale = robust_log_scale.exp().to(values.dtype)
                    quantiles = log_quantiles.exp().to(values.dtype)
                    cap_flags = torch.stack(
                        (
                            (q10_bound < q45 - 1e-12) & (q10_bound <= q25_bound),
                            (q25_bound < q45 - 1e-12) & (q25_bound <= q10_bound),
                        )
                    )
                else:
                    scale = torch.quantile(
                        values.float(),
                        0.45,
                    ).to(values.dtype)
                    quantiles = torch.stack(
                        (
                            values.new_tensor(float("nan")),
                            values.new_tensor(float("nan")),
                            scale,
                        )
                    )
                    cap_flags = torch.zeros(
                        2,
                        dtype=torch.bool,
                        device=values.device,
                    )
                scales.append(scale)
                quantile_receipts.append(quantiles)
                cap_receipts.append(cap_flags)
        scale_tensor = torch.stack(scales).view(-1, 1, 1, 1)
        support_tensor = torch.stack(support_counts).view(-1)
        quantile_tensor = torch.stack(quantile_receipts)
        cap_tensor = torch.stack(cap_receipts)
        return scale_tensor, support_tensor, quantile_tensor, cap_tensor

    @torch.no_grad()
    def _configured_fixed_bim_direct(
        self,
        base_depth: torch.Tensor,
        bim_depth: torch.Tensor,
        bim_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the configured authoritative CPU BIM-direct comparator."""

        if bim_valid is not None:
            if bim_valid.shape != bim_depth.shape:
                raise ValueError(
                    "Configured BIM-direct BIM mask shape differs from depth: "
                    f"{tuple(bim_valid.shape)} != {tuple(bim_depth.shape)}"
                )
            bim_depth = torch.where(
                bim_valid > 0,
                bim_depth,
                torch.zeros_like(bim_depth),
            )
        if base_depth.shape != bim_depth.shape or base_depth.ndim != 4:
            raise ValueError(
                "Configured BIM-direct expects equal [B, 1, H, W] tensors; "
                f"base={tuple(base_depth.shape)}, bim={tuple(bim_depth.shape)}"
            )
        if base_depth.shape[1] != 1:
            raise ValueError("Configured BIM-direct requires one depth channel")
        base_cpu = base_depth.detach().float().cpu().contiguous()
        bim_cpu = bim_depth.detach().float().cpu().contiguous()
        parameters = self.scale_estimator_config

        def compute_sample(sample_index: int) -> torch.Tensor:
            _, direct, _, _, _ = robust_scale_and_local_features(
                base_cpu[sample_index, 0].numpy(),
                bim_cpu[sample_index, 0].numpy(),
                q10_log_cap=float(parameters["q10_log_cap"]),
                q25_log_cap=float(parameters["q25_log_cap"]),
                ratio_min=float(parameters["ratio_min"]),
                ratio_max=float(parameters["ratio_max"]),
                min_samples=int(parameters["min_samples"]),
            )
            return torch.from_numpy(direct)

        sample_count = base_cpu.shape[0]
        if sample_count > 1:
            with ThreadPoolExecutor(max_workers=min(sample_count, 8)) as executor:
                direct_samples = list(executor.map(compute_sample, range(sample_count)))
        else:
            direct_samples = [compute_sample(0)] if sample_count else []
        if not direct_samples:
            return torch.empty_like(base_depth)
        return (
            torch.stack(direct_samples, dim=0)
            .unsqueeze(1)
            .to(
                device=base_depth.device,
                dtype=base_depth.dtype,
            )
        )

    def _build_live_da3_batch(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        rgb = batch.get("da3_rgb", batch["rgb"])
        base = self._forward_da3_depth(rgb)
        confidence = self._depth_confidence(base)
        scale, scale_support, scale_quantiles, scale_cap_flags = self._robust_bim_scale(
            base,
            batch["bim_depth"],
            batch["bim_valid"],
        )
        scaled = base * scale
        live_batch = dict(batch)
        live_batch.update(
            {
                "base_depth": base,
                "base_confidence": confidence,
                "scaled_depth": scaled,
            }
        )
        receipt = {
            "da3_scale": scale,
            "da3_scale_support": scale_support,
            "da3_scale_quantiles_q10_q25_q45": scale_quantiles,
            "da3_scale_cap_flags_q10_q25": scale_cap_flags,
        }
        return live_batch, receipt

    def _attention_scale_inputs(
        self,
        batch: dict[str, torch.Tensor],
        base: torch.Tensor,
        deterministic_scaled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build GT-free attention keys and measured BIM/DA3 ratio values."""

        valid = batch["bim_valid"].clamp(0.0, 1.0)
        bim = batch["bim_depth"]
        log_base = safe_log(base)
        if self.attention_scale_use_deterministic_fallback_input:
            fallback_log_scale_map = safe_log(deterministic_scaled) - log_base
            fallback_valid = (
                torch.isfinite(base)
                & torch.isfinite(deterministic_scaled)
                & (base > 0)
                & (deterministic_scaled > 0)
            )
            fallback_numerator = torch.where(
                fallback_valid,
                fallback_log_scale_map,
                torch.zeros_like(fallback_log_scale_map),
            ).sum(dim=(-2, -1), keepdim=True)
            fallback_denominator = fallback_valid.sum(
                dim=(-2, -1), keepdim=True
            ).clamp_min(1)
            fallback_log_scale = fallback_numerator / fallback_denominator
            disagreement_reference = deterministic_scaled
        else:
            # Matched no-deterministic-prior experiments start at raw DA3 and
            # must not receive the cached robust BIM scale through either the
            # fallback scalar or the spatial disagreement channels.
            fallback_log_scale = base.new_zeros((base.shape[0], 1, 1, 1))
            disagreement_reference = base

        ratio = bim / base.clamp_min(1e-6)
        ratio_valid = (
            (valid > 0)
            & torch.isfinite(base)
            & torch.isfinite(bim)
            & torch.isfinite(ratio)
            & (base > 0)
            & (bim > 0)
            & (ratio > self.da3_scale_ratio_min)
            & (ratio < self.da3_scale_ratio_max)
        )
        log_ratio = torch.where(
            ratio_valid,
            ratio.clamp_min(1e-6).log(),
            torch.zeros_like(ratio),
        )
        safe_bim = torch.where(valid > 0, bim, disagreement_reference)
        log_bim = safe_log(safe_bim)
        log_disagreement_reference = safe_log(disagreement_reference)
        signed_disagreement = (log_bim - log_disagreement_reference) * valid
        rgb = batch["rgb"] if self.use_rgb_condition else torch.zeros_like(batch["rgb"])
        feature_parts = [rgb, log_base / 3.0]
        if self.attention_scale_use_base_confidence:
            feature_parts.append(batch["base_confidence"].clamp(0.0, 1.0))
        feature_parts.extend(((log_bim / 3.0) * valid, valid))
        if self.attention_scale_use_bim_normals:
            feature_parts.append(batch["bim_normals"])
        if self.attention_scale_use_bim_edge:
            feature_parts.append(batch["bim_edge"].clamp(0.0, 1.0))
        feature_parts.extend(
            (
                signed_disagreement.clamp(-1.0, 1.0),
                signed_disagreement.abs().clamp(0.0, 1.0),
            )
        )
        features = torch.cat(feature_parts, dim=1)
        if features.shape[1] != self.attention_scale_input_channels:
            raise RuntimeError(
                "Attentive scale feature contract changed: "
                f"expected {self.attention_scale_input_channels}, got {features.shape[1]}"
            )
        return features, log_ratio, ratio_valid.float(), fallback_log_scale

    def _full_regression_scale_inputs(
        self,
        batch: dict[str, torch.Tensor],
        base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build scale-regression inputs without any deterministic scale.

        The final two spatial channels are measured BIM/raw-DA3 log-ratio
        disagreement and its magnitude.  Neither the cached robust scale nor
        BIM-direct/fallback depth is read by this path.
        """

        log_base = safe_log(base)
        rgb = batch["rgb"] if self.use_rgb_condition else torch.zeros_like(batch["rgb"])
        if (
            self.full_regression_input_mode
            == self.FULL_REGRESSION_INPUT_RGB_DA3_UNIT_BIM
        ):
            # Deliberately contains no measured/rendered BIM information. The
            # unit constant follows the registered extreme control protocol;
            # both the resulting ratio and its validity depend only on DA3.
            ratio = torch.ones_like(base) / base.clamp_min(1e-6)
            ratio_valid = (
                torch.isfinite(base)
                & torch.isfinite(ratio)
                & (base > 0)
                & (ratio > self.da3_scale_ratio_min)
                & (ratio < self.da3_scale_ratio_max)
            )
            log_ratio = torch.where(
                ratio_valid,
                ratio.clamp_min(1e-6).log(),
                torch.zeros_like(ratio),
            )
            features = torch.cat((rgb, log_base / 3.0), dim=1)
            if features.shape[1] != self.attention_scale_input_channels:
                raise RuntimeError(
                    "RGB+DA3 unit-BIM feature contract changed: "
                    f"expected {self.attention_scale_input_channels}, "
                    f"got {features.shape[1]}"
                )
            return features, log_ratio, ratio_valid.float()

        valid = batch["bim_valid"].clamp(0.0, 1.0)
        bim = batch["bim_depth"]
        ratio = bim / base.clamp_min(1e-6)
        ratio_valid = (
            (valid > 0)
            & torch.isfinite(base)
            & torch.isfinite(bim)
            & torch.isfinite(ratio)
            & (base > 0)
            & (bim > 0)
            & (ratio > self.da3_scale_ratio_min)
            & (ratio < self.da3_scale_ratio_max)
        )
        log_ratio = torch.where(
            ratio_valid,
            ratio.clamp_min(1e-6).log(),
            torch.zeros_like(ratio),
        )
        safe_bim = torch.where(valid > 0, bim, torch.ones_like(bim))
        log_bim = safe_log(safe_bim)
        raw_disagreement = (log_bim - log_base) * valid
        feature_parts = [rgb, log_base / 3.0]
        if self.attention_scale_use_base_confidence:
            feature_parts.append(batch["base_confidence"].clamp(0.0, 1.0))
        feature_parts.extend(((log_bim / 3.0) * valid, valid))
        if self.attention_scale_use_bim_normals:
            feature_parts.append(batch["bim_normals"])
        if self.attention_scale_use_bim_edge:
            feature_parts.append(batch["bim_edge"].clamp(0.0, 1.0))
        feature_parts.extend(
            (
                raw_disagreement.clamp(-1.0, 1.0),
                raw_disagreement.abs().clamp(0.0, 1.0),
            )
        )
        features = torch.cat(feature_parts, dim=1)
        if features.shape[1] != self.attention_scale_input_channels:
            raise RuntimeError(
                "Full-regression scale feature contract changed: "
                f"expected {self.attention_scale_input_channels}, got {features.shape[1]}"
            )
        return features, log_ratio, ratio_valid.float()

    def _rgb_dinov2_full_regression_scale_inputs(
        self,
        batch: dict[str, torch.Tensor],
        base: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build RGB+DINO scale inputs without confidence, DA3 features or fallback."""

        valid = batch["bim_valid"].clamp(0.0, 1.0)
        bim = batch["bim_depth"]
        log_base = safe_log(base)
        ratio = bim / base.clamp_min(1e-6)
        ratio_valid = (
            (valid > 0)
            & torch.isfinite(base)
            & torch.isfinite(bim)
            & torch.isfinite(ratio)
            & (base > 0)
            & (bim > 0)
            & (ratio > self.da3_scale_ratio_min)
            & (ratio < self.da3_scale_ratio_max)
        )
        log_ratio = torch.where(
            ratio_valid,
            ratio.clamp_min(1e-6).log(),
            torch.zeros_like(ratio),
        )
        safe_bim = torch.where(valid > 0, bim, torch.ones_like(bim))
        log_bim = safe_log(safe_bim)
        geometry = torch.cat(
            (
                log_base / 3.0,
                (log_bim / 3.0) * valid,
                valid,
                batch["bim_normals"],
                batch["bim_edge"].clamp(0.0, 1.0),
            ),
            dim=1,
        )
        expected_geometry_channels = 7
        if geometry.shape[1] != expected_geometry_channels:
            raise RuntimeError(
                "RGB+DINO geometry feature contract changed: "
                f"expected {expected_geometry_channels}, got {geometry.shape[1]}"
            )
        rgb = batch["rgb"] if self.use_rgb_condition else torch.zeros_like(batch["rgb"])
        return (
            rgb,
            geometry,
            log_ratio,
            ratio_valid.float(),
            batch["dinov2_feature"],
        )

    def _estimate_attention_scale(
        self,
        batch: dict[str, torch.Tensor],
        base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.attention_scale is None:
            raise RuntimeError("Attention scale estimation requested while disabled")
        if isinstance(
            self.attention_scale,
            RGBDINOFullRegressionIterativeScaleHead,
        ):
            inputs = self._rgb_dinov2_full_regression_scale_inputs(batch, base)
            return self.attention_scale(*inputs)
        if isinstance(self.attention_scale, FullRegressionIterativeScaleHead):
            inputs = self._full_regression_scale_inputs(batch, base)
        else:
            inputs = self._attention_scale_inputs(batch, base, batch["scaled_depth"])
        return self.attention_scale(
            *inputs,
            da3_feature_mid=(
                batch.get("da3_feature_mid") if self.da3_feature_scale_enabled else None
            ),
            da3_feature_deep=(
                batch.get("da3_feature_deep") if self.da3_feature_scale_enabled else None
            ),
        )

    def _forward_scale_refinement(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        base = batch["base_depth"]
        attention_scale_output: dict[str, torch.Tensor] | None = None
        if self.attention_scale_enabled:
            attention_scale_output = self._estimate_attention_scale(
                batch,
                base,
            )
            scaled = base * attention_scale_output["scale"].to(dtype=base.dtype)
        else:
            deterministic_scaled = batch["scaled_depth"]
            scaled = deterministic_scaled
        bim = batch["bim_depth"]
        valid = batch["bim_valid"].clamp(0.0, 1.0)
        residual_anchor = scaled

        log_base = safe_log(base)
        log_scaled = safe_log(scaled)
        geometry_scale_channel = ((log_scaled - log_base) / 0.5).clamp(-2.0, 2.0)
        geometry_scale_channel_semantics = "log(scaled/base)/0.5"
        safe_bim = torch.where(valid > 0, bim, scaled)
        log_bim = safe_log(safe_bim)
        signed_disagreement = (log_bim - log_scaled) * valid

        rgb = batch["rgb"] if self.use_rgb_condition else torch.zeros_like(batch["rgb"])
        geometry = torch.cat(
            [
                log_base / 3.0,
                log_scaled / 3.0,
                batch["base_confidence"].clamp(0.0, 1.0),
                geometry_scale_channel,
            ],
            dim=1,
        )
        bim_features = torch.cat(
            [
                (log_bim / 3.0) * valid,
                valid,
                batch["bim_normals"],
                batch["bim_edge"].clamp(0.0, 1.0),
                signed_disagreement.clamp(-1.0, 1.0),
                signed_disagreement.abs().clamp(0.0, 1.0),
            ],
            dim=1,
        )
        if not self.use_bim_condition:
            bim_features = torch.zeros_like(bim_features)

        prediction = self.refiner(
            rgb,
            geometry,
            bim_features,
            da3_feature_mid=(
                batch.get("da3_feature_mid") if self.da3_feature_refiner_enabled else None
            ),
            da3_feature_deep=(
                batch.get("da3_feature_deep") if self.da3_feature_refiner_enabled else None
            ),
        )
        raw_frame_residual = prediction["frame_log_residual"]
        raw_low_residual = prediction["low_log_residual"]
        if not self.use_frame_residual:
            raw_frame_residual = torch.zeros_like(raw_frame_residual)
        if not self.use_low_residual:
            raw_low_residual = torch.zeros_like(raw_low_residual)
        residual_routing_depth = scaled
        if self.depth_aware_residual_routing:
            residual_routing_gate = torch.sigmoid(
                (residual_routing_depth - self.residual_routing_depth)
                / self.residual_routing_temperature
            )
        else:
            residual_routing_gate = torch.ones_like(scaled)
        frame_residual = raw_frame_residual * residual_routing_gate
        low_residual = (
            raw_low_residual * residual_routing_gate
            if self.residual_routing_scope == self.RESIDUAL_ROUTING_FRAME_AND_LOW
            else raw_low_residual
        )
        raw_detail_residual = prediction["detail_log_residual"]
        detail_reliability_gate = torch.ones_like(raw_detail_residual)
        if self.detail_reliability_gate_enabled:
            reliability_logits_for_gate = prediction["bim_reliability_logits"]
            if self.detail_reliability_gate_detach:
                reliability_logits_for_gate = reliability_logits_for_gate.detach()
            detail_reliability_gate = self.detail_reliability_gate_floor + (
                1.0 - self.detail_reliability_gate_floor
            ) * torch.sigmoid(reliability_logits_for_gate)
        detail_residual = raw_detail_residual * detail_reliability_gate
        log_residual = torch.clamp(
            frame_residual + low_residual + detail_residual,
            -float(self.refiner.max_total_log_residual),
            float(self.refiner.max_total_log_residual),
        )
        proportional_refined = residual_anchor * torch.exp(log_residual)
        additive_residual = torch.zeros_like(proportional_refined)
        if self.additive_residual_enabled:
            decoded = prediction.get("decoded_features_for_additive")
            if decoded is None:
                raise KeyError("Additive refiner did not return its decoded feature tensor")
            additive_residual = self.refiner.predict_additive_metric_residual(
                decoded,
                residual_anchor,
                log_residual,
            )
        refined = proportional_refined + additive_residual
        refined = refined.clamp(1e-3, self.output_max_depth)
        reliability_logits = prediction["bim_reliability_logits"]
        reliability = torch.sigmoid(reliability_logits) * valid
        local_scale = scaled / base.clamp_min(1e-3)

        output = {
            "depth": refined,
            "proportional_depth": proportional_refined,
            "additive_metric_residual": additive_residual,
            "base_depth": base,
            "base_confidence": batch["base_confidence"],
            "scaled_depth": scaled,
            "coarse_depth": scaled,
            "refinement_anchor_depth": residual_anchor,
            "geometry_scale_channel": geometry_scale_channel,
            "geometry_scale_channel_semantics": (geometry_scale_channel_semantics),
            "residual_anchor_mode": self.residual_anchor_mode,
            "trust_logits": reliability_logits,
            "pixel_trust_logits": reliability_logits,
            "frame_trust_logits": prediction["frame_trust_logits"],
            "trust_probability": reliability,
            "bim_reliability_logits": reliability_logits,
            "bim_reliability": reliability,
            "support": valid * (1.0 - batch["bim_edge"]).clamp(0.0, 1.0),
            "local_scale": local_scale,
            "local_shift": torch.zeros_like(local_scale),
            "log_residual": log_residual,
            "raw_frame_log_residual": raw_frame_residual,
            "raw_low_log_residual": raw_low_residual,
            "frame_log_residual": frame_residual,
            "low_log_residual": low_residual,
            "detail_log_residual": detail_residual,
            "raw_detail_log_residual": raw_detail_residual,
            "detail_reliability_gate": detail_reliability_gate,
            "residual_routing_gate": residual_routing_gate,
            "residual_routing_depth": residual_routing_depth,
            "residual_routing_depth_semantics": "scaled_depth",
            "residual_routing_scope": self.residual_routing_scope,
            "log_variance": prediction["log_variance"],
        }
        if not isinstance(
            self.attention_scale,
            (
                FullRegressionIterativeScaleHead,
                RGBDINOFullRegressionIterativeScaleHead,
            ),
        ):
            output["deterministic_scaled_depth"] = batch["scaled_depth"]
        if attention_scale_output is not None:
            output.update(
                {
                    "attention_scale": attention_scale_output["scale"],
                    "attention_log_scale": attention_scale_output["log_scale"],
                    "attention_direct_log_scale": attention_scale_output["attentive_log_scale"],
                    "attention_raw_log_scale": attention_scale_output["raw_attentive_log_scale"],
                    "attention_bounded_log_scale_residual": attention_scale_output[
                        "bounded_log_scale_residual"
                    ],
                    "attention_scale_pixel_support": attention_scale_output["pixel_support"],
                    "attention_scale_token_support": attention_scale_output["token_support"],
                    "attention_scale_head_log_scale": attention_scale_output["head_log_scale"],
                    "attention_scale_head_mixture": attention_scale_output["head_mixture"],
                    "attention_scale_normalized_entropy": attention_scale_output[
                        "normalized_attention_entropy"
                    ],
                    "attention_scale_map": attention_scale_output["attention_map"],
                    "attention_token_distribution": attention_scale_output[
                        "attention_token_distribution"
                    ],
                    "attention_token_valid": attention_scale_output[
                        "attention_token_valid"
                    ],
                }
            )
            if "fallback_log_scale" in attention_scale_output:
                output["attention_fallback_log_scale"] = attention_scale_output[
                    "fallback_log_scale"
                ]
            if "fallback_gate" in attention_scale_output:
                output["attention_fallback_gate"] = attention_scale_output[
                    "fallback_gate"
                ]
            if "deterministic_fallback_log_scale" in attention_scale_output:
                output["attention_deterministic_fallback_log_scale"] = (
                    attention_scale_output["deterministic_fallback_log_scale"]
                )
            if "iteration_log_scales" in attention_scale_output:
                output.update(
                    {
                        "attention_iteration_log_scales": attention_scale_output[
                            "iteration_log_scales"
                        ],
                        "attention_iteration_raw_log_scales": attention_scale_output[
                            "iteration_raw_log_scales"
                        ],
                        "attention_iteration_head_log_scales": attention_scale_output[
                            "iteration_head_log_scales"
                        ],
                        "attention_iteration_normalized_entropy": attention_scale_output[
                            "iteration_normalized_attention_entropy"
                        ],
                    }
                )
                if "iteration_fallback_gates" in attention_scale_output:
                    output["attention_iteration_fallback_gates"] = (
                        attention_scale_output["iteration_fallback_gates"]
                    )
                if "iteration_step_sizes" in attention_scale_output:
                    output["attention_iteration_step_sizes"] = attention_scale_output[
                        "iteration_step_sizes"
                    ]
                if "iteration_log_scale_updates" in attention_scale_output:
                    output["attention_iteration_log_scale_updates"] = (
                        attention_scale_output["iteration_log_scale_updates"]
                    )
            spatial_residual = low_residual + detail_residual
            output["spatial_log_residual"] = spatial_residual
            if (
                self.training
                and self.attention_scale_equivariance_probability > 0
                and self.attention_scale_equivariance_log_range > 0
            ):
                selected = (
                    torch.rand(
                        (base.shape[0], 1, 1, 1),
                        device=base.device,
                    )
                    < self.attention_scale_equivariance_probability
                )
                log_factor = torch.empty(
                    (base.shape[0], 1, 1, 1),
                    device=base.device,
                    dtype=base.dtype,
                ).uniform_(
                    -self.attention_scale_equivariance_log_range,
                    self.attention_scale_equivariance_log_range,
                )
                log_factor = torch.where(selected, log_factor, torch.zeros_like(log_factor))
                factor = log_factor.exp()
                perturbed = self._estimate_attention_scale(
                    batch,
                    base * factor,
                )
                output["attention_scale_equivariance_error"] = (
                    perturbed["log_scale"] + log_factor - attention_scale_output["log_scale"]
                )
        if "bim_adapter_gate_logits" in prediction:
            output["bim_adapter_gate_logits"] = prediction["bim_adapter_gate_logits"]
            output["bim_adapter_gate_logits_pyramid"] = prediction[
                "bim_adapter_gate_logits_pyramid"
            ]
        return output
