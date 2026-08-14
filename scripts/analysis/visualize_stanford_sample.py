#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm

from bim_priorda3.baselines import configured_scale_and_local_features
from bim_priorda3.checkpoints import (
    dataset_split_identity,
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_model_config,
)
from bim_priorda3.config import Config, load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.models import BIMPriorDA3

PROTOCOL_MIN_DEPTH_M = 0.2
PROTOCOL_MAX_DEPTH_M = 5.0
SUBSET_NAMES = (
    "all",
    "furniture",
    "non_structural",
    "bim_foreground_conflict",
    "bim_consistent",
    "bim_no_hit",
)
_DOMAIN_SUFFIX = re.compile(r"_domain_*(?:rgb|depth|semantic|pose)$", re.IGNORECASE)


class FixedSupportMetricSums:
    """Single-frame equivalent of the authoritative Stanford evaluator metrics."""

    def __init__(self) -> None:
        self.count = 0
        self.abs_rel_sum = 0.0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.delta1_sum = 0
        self.delta2_sum = 0
        self.delta3_sum = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        support: torch.Tensor,
        *,
        prediction_name: str,
    ) -> None:
        mask = support.bool()
        if prediction.shape != target.shape or support.shape != target.shape:
            raise ValueError(
                "metric tensors must have identical shapes; "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}, "
                f"support={tuple(support.shape)}"
            )
        invalid_target = mask & (~torch.isfinite(target) | (target <= 0))
        if invalid_target.any():
            raise RuntimeError("Fixed metric support contains an invalid target value")
        invalid_prediction = mask & (~torch.isfinite(prediction) | (prediction <= 0))
        invalid_prediction_count = int(invalid_prediction.sum().item())
        if invalid_prediction_count:
            raise RuntimeError(
                f"{prediction_name} has {invalid_prediction_count} non-finite or "
                "non-positive values on the fixed metric support"
            )
        pred = prediction[mask].double()
        gt = target[mask].double()
        if not pred.numel():
            return
        difference = pred - gt
        ratio = torch.maximum(pred / gt, gt / pred)
        self.count += int(pred.numel())
        self.abs_rel_sum += float((difference.abs() / gt).sum())
        self.squared_error_sum += float(difference.square().sum())
        self.absolute_error_sum += float(difference.abs().sum())
        self.delta1_sum += int((ratio < 1.25).sum())
        self.delta2_sum += int((ratio < 1.25**2).sum())
        self.delta3_sum += int((ratio < 1.25**3).sum())

    def compute(self) -> dict[str, float | int]:
        if not self.count:
            return {
                "abs_rel": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
                "count": 0,
            }
        return {
            "abs_rel": self.abs_rel_sum / self.count,
            "rmse": (self.squared_error_sum / self.count) ** 0.5,
            "mae": self.absolute_error_sum / self.count,
            "delta1": self.delta1_sum / self.count,
            "delta2": self.delta2_sum / self.count,
            "delta3": self.delta3_sum / self.count,
            "count": self.count,
        }


def assert_comparable_counts(
    metrics_by_method: Mapping[str, Mapping[str, float | int]],
    expected_count: int,
    *,
    context: str,
) -> None:
    actual = {method: int(metrics["count"]) for method, metrics in metrics_by_method.items()}
    if any(count != expected_count for count in actual.values()):
        raise RuntimeError(
            f"{context}: comparable methods do not share the fixed support; "
            f"expected={expected_count}, actual={actual}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize one explicitly named Stanford 2D-3D-S Area_1 frame with "
            "the fixed global-core BIM prior and target-trained frozen/E2E models."
        )
    )
    parser.add_argument(
        "--sample-id",
        required=True,
        help=(
            "Full annotation ID (room/camera_..._frame_N), its long basename, "
            "or the corresponding Stanford modality filename"
        ),
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("val", "test"),
        help="Annotation split containing the explicitly chosen sample",
    )
    parser.add_argument("--frozen-config", required=True, type=Path)
    parser.add_argument("--frozen-checkpoint", required=True, type=Path)
    parser.add_argument("--e2e-config", required=True, type=Path)
    parser.add_argument("--e2e-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/stanford_area1"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-live-direct",
        action="store_true",
        help=(
            "Explicitly request the CPU/OpenCV BIM-direct comparator from the E2E "
            "model; ordinary eval inference leaves this branch disabled"
        ),
    )
    return parser.parse_args(argv)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_basename(query: str) -> str:
    value = Path(query.strip()).name
    for suffix in (".png", ".json", ".npz"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return _DOMAIN_SUFFIX.sub("", value)


def resolve_sample_index(records: list[dict[str, Any]], query: str) -> tuple[int, str]:
    """Resolve long Stanford IDs without guessing a frame or scanning another split."""
    stripped = query.strip().strip("/")
    if not stripped:
        raise ValueError("--sample-id cannot be empty")
    direct = [index for index, record in enumerate(records) if str(record["id"]) == stripped]
    if len(direct) == 1:
        return direct[0], str(records[direct[0]]["id"])

    basename = _query_basename(stripped)
    matches = [
        index
        for index, record in enumerate(records)
        if str(record["id"]).rsplit("/", 1)[-1] == basename
    ]
    if not matches:
        raise KeyError(
            f"Sample {query!r} is absent from the selected annotation split; "
            "the visualizer never substitutes or selects another frame"
        )
    if len(matches) > 1:
        identifiers = [str(records[index]["id"]) for index in matches]
        raise ValueError(f"Ambiguous Stanford sample basename {basename!r}: {identifiers}")
    return matches[0], str(records[matches[0]]["id"])


def _require_protocol_config(cfg: Config, *, label: str) -> None:
    limits = (float(cfg.data.min_depth), float(cfg.data.max_depth))
    expected = (PROTOCOL_MIN_DEPTH_M, PROTOCOL_MAX_DEPTH_M)
    if limits != expected:
        raise ValueError(
            f"{label} must use the locked Stanford depth protocol {expected} m, got {limits}"
        )
    if not cfg.data.get("split_annotation"):
        raise ValueError(f"{label} must use data.split_annotation")


def _resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    return device


def _batched_item(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }


def _output_hw(value: torch.Tensor, *, name: str) -> np.ndarray:
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [1, 1, H, W], got {tuple(value.shape)}")
    return value[0, 0].detach().float().cpu().numpy()


def infer_validated_checkpoint(
    *,
    cfg: Config,
    checkpoint_path: Path,
    split_provenance: Mapping[str, Any],
    split: str,
    item: dict[str, Any],
    device: torch.device,
    expected_e2e: bool,
    request_live_direct: bool,
) -> tuple[dict[str, np.ndarray | float | bool], dict[str, Any]]:
    """Validate target provenance/config before running one eval-mode forward pass."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    dataset_validation = validate_checkpoint_evaluation_dataset_provenance(
        state,
        split_provenance,
        split=split,
        allow_cross_dataset=False,
    )
    if dataset_validation.get("verified") is not True:
        raise RuntimeError(
            f"Target checkpoint dataset provenance is not verified: {checkpoint_path}; "
            f"status={dataset_validation.get('status')!r}"
        )
    model_differences = validate_checkpoint_model_config(state, cfg.model)
    model = BIMPriorDA3(cfg)
    if bool(model.e2e_da3_enabled) != expected_e2e:
        branch = "E2E" if expected_e2e else "frozen-DA3"
        raise ValueError(f"{branch} checkpoint/config was supplied to the wrong visualizer slot")
    model_state = state.get("model")
    if not isinstance(model_state, Mapping):
        raise TypeError(f"Checkpoint lacks a model state mapping: {checkpoint_path}")
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    batch = _batched_item(item, device)
    if request_live_direct:
        if not expected_e2e:
            raise ValueError("live BIM-direct can only be requested from the E2E model")
        batch["request_live_bim_direct"] = True
    with torch.inference_mode():
        output = model(batch)

    if expected_e2e and output.get("uses_live_da3") is not True:
        raise RuntimeError("The E2E model did not execute its live DA3 branch")
    if request_live_direct and "live_bim_direct" not in output:
        raise RuntimeError("The explicitly requested live_bim_direct output is missing")
    if not request_live_direct and "live_bim_direct" in output:
        raise RuntimeError("Eval-mode E2E emitted live_bim_direct without an explicit request")

    arrays: dict[str, np.ndarray | float | bool] = {
        "depth": _output_hw(output["depth"], name="depth"),
        "base_depth": _output_hw(output["base_depth"], name="base_depth"),
        "scaled_depth": _output_hw(output["scaled_depth"], name="scaled_depth"),
        "reliability": _output_hw(output["bim_reliability"], name="bim_reliability"),
        "log_residual": _output_hw(output["log_residual"], name="log_residual"),
        "routing_gate": _output_hw(
            output["residual_routing_gate"],
            name="residual_routing_gate",
        ),
        "uses_live_da3": bool(output.get("uses_live_da3", False)),
    }
    if "da3_scale" in output:
        arrays["da3_scale"] = float(output["da3_scale"].detach().float().mean().cpu())
    if "live_bim_direct" in output:
        arrays["live_bim_direct"] = _output_hw(
            output["live_bim_direct"],
            name="live_bim_direct",
        )

    checkpoint_cfg = state.get("config")
    provenance_receipt = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "training_config": (
            checkpoint_cfg.get("config_path") if isinstance(checkpoint_cfg, Mapping) else None
        ),
        "checkpoint_epoch": state.get("epoch"),
        "model_config_validation": {
            "verified": not model_differences,
            "differences": model_differences,
        },
        "dataset_provenance_validation": dataset_validation,
    }
    del batch, output, model, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, provenance_receipt


def item_hw(item: dict[str, Any], key: str) -> np.ndarray:
    value = item.get(key)
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] != 1:
        shape = tuple(value.shape) if torch.is_tensor(value) else None
        raise ValueError(f"Dataset item {key!r} must have shape [1, H, W], got {shape}")
    return value[0].detach().float().cpu().numpy()


def build_fixed_support_and_subsets(
    item: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    required = (
        "gt_depth",
        "gt_valid",
        "bim_depth",
        "bim_valid",
        "furniture_mask",
        "non_structural_mask",
        "semantic_valid",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise RuntimeError(f"Prepared Stanford sample lacks required fields: {missing}")
    gt = item_hw(item, "gt_depth")
    gt_valid = item_hw(item, "gt_valid") > 0
    fixed_support = (
        gt_valid & np.isfinite(gt) & (gt >= PROTOCOL_MIN_DEPTH_M) & (gt <= PROTOCOL_MAX_DEPTH_M)
    )
    if not fixed_support.any():
        raise RuntimeError("The selected frame has no valid official z-depth in 0.2-5.0 m")

    bim = item_hw(item, "bim_depth")
    bim_valid = (item_hw(item, "bim_valid") > 0) & np.isfinite(bim) & (bim > 0)
    furniture = item_hw(item, "furniture_mask") > 0
    non_structural = item_hw(item, "non_structural_mask") > 0
    semantic_valid = item_hw(item, "semantic_valid") > 0
    if np.any(furniture & ~non_structural):
        raise RuntimeError("furniture_mask is not a subset of non_structural_mask")
    if np.any((furniture | non_structural) & ~semantic_valid):
        raise RuntimeError("semantic subset masks include unknown semantic pixels")
    tolerance = np.maximum(0.10, 0.05 * bim)
    subsets = {
        "all": fixed_support,
        "furniture": fixed_support & furniture,
        "non_structural": fixed_support & non_structural,
        "bim_foreground_conflict": (fixed_support & bim_valid & (gt < bim - tolerance)),
        "bim_consistent": fixed_support & bim_valid & (np.abs(gt - bim) <= tolerance),
        "bim_no_hit": fixed_support & ~bim_valid,
    }
    return fixed_support, bim_valid, subsets


def fixed_support_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    *,
    name: str,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or support.shape != target.shape:
        raise ValueError(
            f"{name}: prediction, target and support shapes differ: "
            f"{prediction.shape}, {target.shape}, {support.shape}"
        )
    sums = FixedSupportMetricSums()
    sums.update(
        torch.from_numpy(prediction)[None, None],
        torch.from_numpy(target)[None, None],
        torch.from_numpy(support)[None, None],
        prediction_name=name,
    )
    return sums.compute()


def evaluate_single_frame(
    *,
    predictions: Mapping[str, np.ndarray],
    gt: np.ndarray,
    bim: np.ndarray,
    bim_valid: np.ndarray,
    subsets: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, dict[str, float | int]]], dict[str, Any]]:
    methods: dict[str, dict[str, dict[str, float | int]]] = {}
    for method, prediction in predictions.items():
        methods[method] = {
            subset: fixed_support_metrics(
                prediction,
                gt,
                support,
                name=f"{method}/{subset}",
            )
            for subset, support in subsets.items()
        }
    for subset, support in subsets.items():
        expected = int(support.sum())
        assert_comparable_counts(
            {method: values[subset] for method, values in methods.items()},
            expected,
            context=f"single-frame/{subset}",
        )

    fixed_support = subsets["all"]
    hit_support = fixed_support & bim_valid
    hit_count = int(hit_support.sum())
    gt_count = int(fixed_support.sum())
    envelope = {
        "support_definition": "official_z_depth_0.2_5.0m & global_core_bim_hit",
        "coverage": {
            "fraction": float(hit_count / gt_count),
            "hit_pixels": hit_count,
            "gt_pixels": gt_count,
            "no_hit_pixels": gt_count - hit_count,
        },
        "metrics_on_hit_support": fixed_support_metrics(
            bim,
            gt,
            hit_support,
            name="global_core_bim/hit_support",
        ),
    }
    return methods, envelope


def relative_error_map(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    result = np.full(target.shape, np.nan, dtype=np.float32)
    result[support] = np.abs(prediction[support] - target[support]) / target[support]
    return result


def _masked_depth(depth: np.ndarray, support: np.ndarray | None = None) -> np.ndarray:
    result = depth.astype(np.float32, copy=True)
    valid = np.isfinite(result) & (result > 0)
    if support is not None:
        valid &= support
    result[~valid] = np.nan
    return result


def _metric_title(label: str, metrics: Mapping[str, float | int]) -> str:
    return f"{label}\nAbsRel={float(metrics['abs_rel']):.4f}  MAE={float(metrics['mae']):.3f}m"


def render_figure(
    *,
    sample_id: str,
    split: str,
    rgb: np.ndarray,
    gt: np.ndarray,
    bim: np.ndarray,
    bim_valid: np.ndarray,
    subsets: Mapping[str, np.ndarray],
    predictions: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    e2e_diagnostics: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    fixed_support = subsets["all"]
    depth_cmap = plt.get_cmap("turbo").with_extremes(bad="black")
    error_cmap = plt.get_cmap("magma").with_extremes(bad="black")
    delta_cmap = plt.get_cmap("coolwarm").with_extremes(bad="black")
    depth_norm = Normalize(PROTOCOL_MIN_DEPTH_M, PROTOCOL_MAX_DEPTH_M)
    error_norm = Normalize(0.0, 0.5)
    delta_norm = TwoSlopeNorm(vmin=-0.25, vcenter=0.0, vmax=0.25)

    ordered_methods = [
        name
        for name in ("raw_da3", "global_scale", "bim_direct", "frozen_learned", "e2e")
        if name in predictions
    ]
    if "live_bim_direct" in predictions:
        ordered_methods.append("live_bim_direct")
    labels = {
        "raw_da3": "Raw cached DA3",
        "global_scale": "BIM global scale",
        "bim_direct": "Fixed BIM direct",
        "frozen_learned": "Frozen-DA3 learned",
        "e2e": "Partial E2E learned",
        "live_bim_direct": "E2E live BIM direct",
    }
    errors = {
        name: relative_error_map(predictions[name], gt, fixed_support) for name in ordered_methods
    }

    fig, axes = plt.subplots(4, 6, figsize=(25, 17), constrained_layout=True)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    axes[0, 0].imshow(np.clip(rgb, 0.0, 1.0))
    axes[0, 0].set_title("RGB")
    depth_image = axes[0, 1].imshow(
        _masked_depth(gt, fixed_support), cmap=depth_cmap, norm=depth_norm
    )
    axes[0, 1].set_title("Official camera-z depth GT")
    axes[0, 2].imshow(_masked_depth(bim, bim_valid), cmap=depth_cmap, norm=depth_norm)
    axes[0, 2].set_title("Fixed Area_1 global-core BIM")

    coverage = np.full(gt.shape, np.nan, dtype=np.float32)
    coverage[fixed_support & ~bim_valid] = 0.0
    coverage[fixed_support & bim_valid] = 1.0
    coverage_cmap = ListedColormap(("#f28e2b", "#59a14f")).with_extremes(bad="black")
    axes[0, 3].imshow(
        coverage,
        cmap=coverage_cmap,
        norm=BoundaryNorm((-0.5, 0.5, 1.5), coverage_cmap.N),
    )
    coverage_fraction = float((fixed_support & bim_valid).sum() / fixed_support.sum())
    axes[0, 3].set_title(f"BIM coverage (green=hit)\n{coverage_fraction:.1%}")
    axes[0, 4].imshow(subsets["furniture"], cmap="gray", vmin=0, vmax=1)
    axes[0, 4].set_title(f"Furniture support\n{subsets['furniture'].sum():,} px")
    axes[0, 5].imshow(subsets["bim_foreground_conflict"], cmap="gray", vmin=0, vmax=1)
    axes[0, 5].set_title(
        f"BIM foreground conflict\n{subsets['bim_foreground_conflict'].sum():,} px"
    )

    for column in range(6):
        if column >= len(ordered_methods):
            axes[1, column].axis("off")
            axes[2, column].axis("off")
            continue
        method = ordered_methods[column]
        axes[1, column].imshow(
            _masked_depth(predictions[method]),
            cmap=depth_cmap,
            norm=depth_norm,
        )
        axes[1, column].set_title(_metric_title(labels[method], metrics[method]["all"]))
        axes[2, column].imshow(errors[method], cmap=error_cmap, norm=error_norm)
        axes[2, column].set_title(f"{labels[method]} |relative error|")

    delta_specs = (
        ("frozen_learned", "bim_direct", "Frozen learned - direct error"),
        ("e2e", "bim_direct", "E2E - direct error"),
        ("e2e", "frozen_learned", "E2E - frozen error"),
    )
    delta_image = None
    for column, (candidate, reference, title) in enumerate(delta_specs):
        delta_image = axes[3, column].imshow(
            errors[candidate] - errors[reference],
            cmap=delta_cmap,
            norm=delta_norm,
        )
        axes[3, column].set_title(f"{title}\nblue = candidate better")
    diagnostic_specs = (
        ("reliability", "E2E BIM reliability", "unit"),
        ("log_residual", "E2E log-depth residual", "signed"),
        ("routing_gate", "E2E residual routing gate", "unit"),
    )
    for column, (name, title, kind) in enumerate(diagnostic_specs, start=3):
        value = e2e_diagnostics[name]
        if kind == "unit":
            image = axes[3, column].imshow(value, cmap="viridis", vmin=0.0, vmax=1.0)
        else:
            finite = np.abs(value[np.isfinite(value)])
            limit = max(0.05, float(finite.max(initial=0.0)))
            image = axes[3, column].imshow(
                value,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
        fig.colorbar(image, ax=axes[3, column], shrink=0.68)
        axes[3, column].set_title(title)

    fig.colorbar(
        depth_image,
        ax=axes[:2, :].ravel().tolist(),
        shrink=0.54,
        label="Depth (m), fixed display range 0.2-5.0",
    )
    error_image = axes[2, 0].images[0]
    fig.colorbar(
        error_image,
        ax=axes[2, :].ravel().tolist(),
        shrink=0.70,
        label="Absolute relative error (clipped at 0.5 for display)",
    )
    if delta_image is not None:
        fig.colorbar(
            delta_image,
            ax=axes[3, :3].ravel().tolist(),
            shrink=0.70,
            label="Difference in per-pixel absolute relative error",
        )
    fig.suptitle(
        f"{sample_id} — annotation split: {split}\n"
        "Official Stanford perspective z-depth; every comparable metric uses the same "
        "0.2-5.0 m GT support",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _same_sample_record(
    frozen_record: Mapping[str, Any],
    e2e_record: Mapping[str, Any],
) -> None:
    keys = ("id", "region", "camera_uuid", "preparation_fingerprint_sha256")
    differences = {
        key: (frozen_record.get(key), e2e_record.get(key))
        for key in keys
        if frozen_record.get(key) != e2e_record.get(key)
    }
    if differences:
        raise ValueError(f"Frozen/E2E configs resolve different prepared samples: {differences}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    frozen_cfg = load_config(args.frozen_config)
    e2e_cfg = load_config(args.e2e_config)
    _require_protocol_config(frozen_cfg, label="frozen config")
    _require_protocol_config(e2e_cfg, label="E2E config")
    dimensions = {
        (int(frozen_cfg.data.target_height), int(frozen_cfg.data.target_width)),
        (int(e2e_cfg.data.target_height), int(e2e_cfg.data.target_width)),
    }
    if len(dimensions) != 1:
        raise ValueError(f"Frozen/E2E target dimensions differ: {sorted(dimensions)}")

    frozen_dataset = BIMDepthDataset(frozen_cfg, args.split, augment=False)
    e2e_dataset = BIMDepthDataset(e2e_cfg, args.split, augment=False)
    frozen_identity = dataset_split_identity(frozen_dataset.split_provenance)
    e2e_identity = dataset_split_identity(e2e_dataset.split_provenance)
    if frozen_identity != e2e_identity:
        raise ValueError("Frozen/E2E configs resolve different annotation populations")

    frozen_index, sample_id = resolve_sample_index(frozen_dataset.records, args.sample_id)
    e2e_index, e2e_sample_id = resolve_sample_index(e2e_dataset.records, sample_id)
    if e2e_sample_id != sample_id:
        raise RuntimeError("Frozen/E2E sample IDs differ after exact resolution")
    frozen_record = frozen_dataset.records[frozen_index]
    e2e_record = e2e_dataset.records[e2e_index]
    _same_sample_record(frozen_record, e2e_record)
    item = frozen_dataset[frozen_index]
    device = _resolve_device(args.device)

    frozen_output, frozen_checkpoint_receipt = infer_validated_checkpoint(
        cfg=frozen_cfg,
        checkpoint_path=args.frozen_checkpoint,
        split_provenance=frozen_dataset.split_provenance,
        split=args.split,
        item=item,
        device=device,
        expected_e2e=False,
        request_live_direct=False,
    )
    e2e_output, e2e_checkpoint_receipt = infer_validated_checkpoint(
        cfg=e2e_cfg,
        checkpoint_path=args.e2e_checkpoint,
        split_provenance=e2e_dataset.split_provenance,
        split=args.split,
        item=item,
        device=device,
        expected_e2e=True,
        request_live_direct=bool(args.include_live_direct),
    )

    gt = item_hw(item, "gt_depth")
    bim = item_hw(item, "bim_depth")
    raw = item_hw(item, "base_depth")
    fixed_support, bim_valid, subsets = build_fixed_support_and_subsets(item)
    global_scale, bim_direct, _, _, scale_estimate = configured_scale_and_local_features(
        raw,
        bim,
        frozen_cfg.model.get("scale_estimator"),
    )
    predictions: dict[str, np.ndarray] = {
        "raw_da3": raw,
        "global_scale": global_scale,
        "bim_direct": bim_direct,
        "frozen_learned": np.asarray(frozen_output["depth"]),
        "e2e": np.asarray(e2e_output["depth"]),
    }
    if args.include_live_direct:
        predictions.update(
            {
                "live_da3": np.asarray(e2e_output["base_depth"]),
                "live_scale": np.asarray(e2e_output["scaled_depth"]),
                "live_bim_direct": np.asarray(e2e_output["live_bim_direct"]),
            }
        )
    method_metrics, envelope = evaluate_single_frame(
        predictions=predictions,
        gt=gt,
        bim=bim,
        bim_valid=bim_valid,
        subsets=subsets,
    )

    rgb = item["rgb"].detach().float().cpu().numpy().transpose(1, 2, 0)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", sample_id)
    figure_path = output_dir / f"{safe_name}_comparison.png"
    receipt_path = output_dir / f"{safe_name}_metrics.json"
    render_figure(
        sample_id=sample_id,
        split=args.split,
        rgb=rgb,
        gt=gt,
        bim=bim,
        bim_valid=bim_valid,
        subsets=subsets,
        predictions=predictions,
        metrics=method_metrics,
        e2e_diagnostics={
            "reliability": np.asarray(e2e_output["reliability"]),
            "log_residual": np.asarray(e2e_output["log_residual"]),
            "routing_gate": np.asarray(e2e_output["routing_gate"]),
        },
        output_path=figure_path,
    )

    subset_receipt = {
        "all": {
            "definition": "prepared_gt_valid & finite official z-depth & 0.2m <= z <= 5.0m",
        },
        "furniture": {"definition": "all & table/chair/sofa/bookcase semantic pixels"},
        "non_structural": {"definition": "all & known non-envelope semantic pixels"},
        "bim_foreground_conflict": {
            "definition": "all & BIM hit & GT < BIM - max(0.10m, 5% of BIM depth)",
        },
        "bim_consistent": {
            "definition": "all & BIM hit & |GT-BIM| <= max(0.10m, 5% of BIM depth)",
        },
        "bim_no_hit": {"definition": "all & no global-core BIM hit"},
    }
    for name in SUBSET_NAMES:
        subset_receipt[name]["pixels"] = int(subsets[name].sum())

    frozen_config_path = Path(frozen_cfg.config_path).resolve()
    e2e_config_path = Path(e2e_cfg.config_path).resolve()
    receipt = {
        "schema_version": 1,
        "protocol": "stanford-area1-single-frame-global-core-fixed-support-v1",
        "sample": {
            "id": sample_id,
            "annotation_split": args.split,
            "region": str(frozen_record["region"]),
            "camera_uuid": str(frozen_record.get("camera_uuid", "")),
            "frame_number": frozen_record.get("frame_number"),
            "image": str(frozen_record["image"]),
            "prepared_sample": str(frozen_record["sample"]),
            "preparation_fingerprint_sha256": frozen_record.get("preparation_fingerprint_sha256"),
        },
        "ground_truth": {
            "source": "official Stanford 2D-3D-S perspective depth PNG",
            "geometry": "camera-plane z-depth (not LiDAR range or fused LiDAR)",
            "released_encoding": "uint16 / 512 metres; uint16(65535) invalid",
            "fixed_support_depth_m": [PROTOCOL_MIN_DEPTH_M, PROTOCOL_MAX_DEPTH_M],
            "fixed_support_pixels": int(fixed_support.sum()),
        },
        "bim_prior": {
            "source": "one fixed Area_1 global-core BIM envelope",
            "included_categories": ["wall", "floor", "ceiling", "column", "beam"],
            "excluded_categories": ["door", "window", "furniture", "proxy", "MEP"],
            "standalone_global_core_envelope": envelope,
            "fixed_bim_scale": float(scale_estimate.scale),
        },
        "subsets": subset_receipt,
        "methods": method_metrics,
        "comparability": {
            "all_methods_share_identical_subset_support": True,
            "raw_bim_envelope_excluded_from_fixed_support_method_comparison": True,
            "prediction_values_are_not_clipped_for_metrics": True,
        },
        "live_direct_requested": bool(args.include_live_direct),
        "live_e2e_scale": e2e_output.get("da3_scale"),
        "checkpoints": {
            "frozen_learned": frozen_checkpoint_receipt,
            "e2e": e2e_checkpoint_receipt,
        },
        "configs": {
            "frozen": {"path": str(frozen_config_path), "sha256": file_sha256(frozen_config_path)},
            "e2e": {"path": str(e2e_config_path), "sha256": file_sha256(e2e_config_path)},
        },
        "annotation_population_identity": frozen_identity,
        "artifacts": {
            "figure": str(figure_path),
            "receipt": str(receipt_path),
        },
        "visualizer": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    safe_receipt = _json_safe(receipt)
    receipt_path.write_text(
        json.dumps(safe_receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe_receipt, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
