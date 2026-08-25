#!/usr/bin/env python3
"""Validation-only upper bound for semantic-aware deterministic BIM correction.

The released Area_1 semantic PNG is privileged target annotation.  This study
therefore deliberately accepts only the validation split and labels every
result as an oracle upper bound, not a deployable inference method.  Prediction
construction receives cached DA3, the registered BIM render and semantic class
IDs.  Official depth is opened only after all variants for a frame are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.baselines import (
    PREVIOUS_FIXED_PARAMETERS,
    configured_scale_and_local_features,
    estimate_robust_bim_scale,
    resolve_scale_estimator_config,
)
from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.ifc_envelope import ENVELOPE_CATEGORIES
from bim_priorda3.data.stanford2d3ds import STANFORD_SEMANTIC_CLASSES
from bim_priorda3.scale_protocol import validate_universal_scale_protocol

CORE_CLASS_NAMES = ("ceiling", "floor", "wall", "beam", "column")
CORE_CLASS_IDS = tuple(STANFORD_SEMANTIC_CLASSES.index(name) for name in CORE_CLASS_NAMES)
DOOR_WINDOW_IDS = tuple(STANFORD_SEMANTIC_CLASSES.index(name) for name in ("window", "door"))
FURNITURE_IDS = tuple(
    STANFORD_SEMANTIC_CLASSES.index(name) for name in ("table", "chair", "sofa", "bookcase")
)
SEMANTIC_TO_BIM_CATEGORY = {
    STANFORD_SEMANTIC_CLASSES.index(name): ENVELOPE_CATEGORIES.index(name)
    for name in CORE_CLASS_NAMES
}

SCALE_VARIANTS = (
    "scale_all",
    "scale_core",
    "scale_category_match",
    "scale_category_match_bim_nonedge",
    "scale_category_match_interior",
    "scale_category_balanced",
    "scale_category_match_q40",
    "scale_category_match_q45",
    "scale_category_match_q50",
    "scale_category_match_q55",
    "scale_category_match_q60",
    "scale_category_match_q65",
    "scale_category_match_q70",
    "scale_category_match_q75",
    "scale_category_match_q80",
    "scale_category_match_q85",
)

PIXEL_CORRECTION_VARIANTS = (
    "local_all",
    "semantic_apply_gate",
    "semantic_source_gate",
    "semantic_classwise",
    "semantic_core_scale_classwise",
    "semantic_bim_replace_nonedge",
    "semantic_category_replace_nonedge",
    "semantic_category_replace_asymmetric",
    "semantic_category_replace_consistent",
    "semantic_category_soft_blend",
)

VARIANTS = SCALE_VARIANTS + PIXEL_CORRECTION_VARIANTS

VARIANT_DEFINITIONS = {
    "scale_all": "universal robust scale estimated from all valid BIM hits",
    "scale_core": (
        "one global robust scale estimated from semantic core pixels with any valid BIM hit"
    ),
    "scale_category_match": (
        "one global robust scale estimated only from image/BIM category-matched core pixels"
    ),
    "scale_category_match_bim_nonedge": (
        "category-matched global scale excluding rendered BIM depth edges"
    ),
    "scale_category_match_interior": (
        "category-matched global scale excluding both BIM edges and image semantic boundaries"
    ),
    "scale_category_balanced": (
        "one global scale equal to the median log scale across available matched interior classes"
    ),
    "scale_category_match_q40": "category-matched one-global-scale log-ratio q=.40",
    "scale_category_match_q45": "category-matched one-global-scale log-ratio q=.45",
    "scale_category_match_q50": "category-matched one-global-scale log-ratio median",
    "scale_category_match_q55": "category-matched one-global-scale log-ratio q=.55",
    "scale_category_match_q60": "category-matched one-global-scale log-ratio q=.60",
    "scale_category_match_q65": "category-matched one-global-scale log-ratio q=.65",
    "scale_category_match_q70": "category-matched one-global-scale log-ratio q=.70",
    "scale_category_match_q75": "category-matched one-global-scale log-ratio q=.75",
    "scale_category_match_q80": "category-matched one-global-scale log-ratio q=.80",
    "scale_category_match_q85": "category-matched one-global-scale log-ratio q=.85",
    "local_all": "current geometry-only consistency/edge/Gaussian BIM-direct",
    "semantic_apply_gate": (
        "geometry-only local field, applied only on oracle BIM-supported semantic classes"
    ),
    "semantic_source_gate": (
        "local field sourced and applied only on oracle BIM-supported semantic classes"
    ),
    "semantic_classwise": (
        "per-semantic-class normalized Gaussian fields with all-hit robust scale"
    ),
    "semantic_core_scale_classwise": (
        "oracle core-only robust scale plus per-semantic-class normalized Gaussian fields"
    ),
    "semantic_bim_replace_nonedge": (
        "all-hit robust scale, but replace oracle core/non-edge pixels by rendered BIM depth"
    ),
    "semantic_category_replace_nonedge": (
        "replace only where oracle image class equals the rendered BIM component class"
    ),
    "semantic_category_replace_asymmetric": (
        "category-matched replacement rejected only when BIM is >0.10 log-depth behind scaled DA3"
    ),
    "semantic_category_replace_consistent": (
        "category-matched replacement additionally requiring the frozen 0.10 log-consistency gate"
    ),
    "semantic_category_soft_blend": (
        "category-matched, non-edge log-depth blend with a frozen Gaussian consistency weight"
    ),
}

SUBSET_DEFINITIONS = {
    "all": "all official valid z-depth pixels in [0.2, 5.0] m",
    "core_structure": "all & semantic in ceiling/floor/wall/beam/column",
    "core_structure_bim_hit": "core_structure & valid rendered BIM ray hit",
    "core_structure_consistent": ("core_structure_bim_hit & abs(GT-BIM)<=max(0.10m, 5% BIM)"),
    "furniture": "all & semantic in table/chair/sofa/bookcase",
    "door_window": "all & semantic in door/window, which the fixed core BIM excludes",
    "non_structural": "all & known semantic outside the five BIM-supported core classes",
    "bim_foreground_conflict": "all & BIM hit & GT < BIM-max(0.10m, 5% BIM)",
    "semantic_boundary": "all & 3x3 morphological boundary of the semantic class map",
}

SCALE_DIAGNOSTIC_FIELDS = (
    "all_scale",
    "core_scale",
    "category_match_scale",
    "category_bim_nonedge_scale",
    "category_interior_scale",
    "category_balanced_scale",
    "category_match_q40_scale",
    "category_match_q45_scale",
    "category_match_q50_scale",
    "category_match_q55_scale",
    "category_match_q60_scale",
    "category_match_q65_scale",
    "category_match_q70_scale",
    "category_match_q75_scale",
    "category_match_q80_scale",
    "category_match_q85_scale",
    "all_scale_support",
    "core_scale_support",
    "category_match_scale_support",
    "category_bim_nonedge_scale_support",
    "category_interior_scale_support",
    "category_balanced_available_classes",
    "category_balanced_fallback_to_all",
)

PIXEL_DIAGNOSTIC_FIELDS = (
    "all_local_source_pixels",
    "core_local_source_pixels",
    "semantic_core_pixels",
    "semantic_replacement_pixels",
    "semantic_category_match_pixels",
    "semantic_category_nonedge_pixels",
    "semantic_category_asymmetric_pixels",
    "semantic_category_consistent_pixels",
)

DIAGNOSTIC_FIELDS = SCALE_DIAGNOSTIC_FIELDS + PIXEL_DIAGNOSTIC_FIELDS


@dataclass
class MetricSums:
    count: int = 0
    abs_rel_sum: float = 0.0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    delta1_count: int = 0

    def add(self, other: MetricSums) -> None:
        self.count += other.count
        self.abs_rel_sum += other.abs_rel_sum
        self.abs_error_sum += other.abs_error_sum
        self.squared_error_sum += other.squared_error_sum
        self.delta1_count += other.delta1_count

    def metrics(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "abs_rel": None,
                "mae": None,
                "rmse": None,
                "delta1": None,
            }
        return {
            "count": self.count,
            "abs_rel": self.abs_rel_sum / self.count,
            "mae": self.abs_error_sum / self.count,
            "rmse": math.sqrt(self.squared_error_sum / self.count),
            "delta1": self.delta1_count / self.count,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/validation-only oracle semantic Area_1 scale study"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stanford_area1.yaml"))
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--study",
        choices=("semantic-scale", "full-historical"),
        default="semantic-scale",
        help="Scale-only is the primary study; full-historical retains the superseded pixel gates.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Train-only fast path: evaluate only the all-pixel subset for quantile selection.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")
    if args.selection_only and (args.split != "train" or args.study != "semantic-scale"):
        parser.error("--selection-only requires --split train --study semantic-scale")
    if args.output_dir is None:
        args.output_dir = Path(f"results/stanford_area1/oracle_semantic_scale_{args.split}")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.expanduser().resolve()
    repository = Path.cwd().resolve()
    try:
        return resolved.relative_to(repository).as_posix()
    except ValueError as error:
        raise ValueError(
            f"Public oracle artifact must stay under the repository: {resolved}"
        ) from error


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN is not permitted in an oracle semantic receipt")
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def _semantic_masks(semantic_class: np.ndarray) -> dict[str, np.ndarray]:
    core = np.isin(semantic_class, np.asarray(CORE_CLASS_IDS, dtype=np.uint8))
    furniture = np.isin(semantic_class, np.asarray(FURNITURE_IDS, dtype=np.uint8))
    door_window = np.isin(semantic_class, np.asarray(DOOR_WINDOW_IDS, dtype=np.uint8))
    known = semantic_class != 255
    semantic_u16 = semantic_class.astype(np.uint16)
    eroded = cv2.erode(semantic_u16, np.ones((3, 3), dtype=np.uint8))
    dilated = cv2.dilate(semantic_u16, np.ones((3, 3), dtype=np.uint8))
    boundary = known & (eroded != dilated)
    return {
        "core": core,
        "furniture": furniture,
        "door_window": door_window,
        "known": known,
        "non_structural": known & ~core,
        "boundary": boundary,
    }


def _scale_from_mask(
    base: np.ndarray,
    bim: np.ndarray,
    support: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, Any]:
    masked_bim = np.where(support, bim, 0.0).astype(np.float32)
    estimate = estimate_robust_bim_scale(
        base,
        masked_bim,
        q10_log_cap=float(parameters["q10_log_cap"]),
        q25_log_cap=float(parameters["q25_log_cap"]),
        ratio_min=float(parameters["ratio_min"]),
        ratio_max=float(parameters["ratio_max"]),
        min_samples=int(parameters["min_samples"]),
    )
    return (base * estimate.scale).astype(np.float32), estimate


def _balanced_category_scale(
    base: np.ndarray,
    bim: np.ndarray,
    semantic_class: np.ndarray,
    bim_category: np.ndarray,
    support: np.ndarray,
    parameters: dict[str, Any],
    fallback_estimate: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one global scale after giving each matched structure class equal weight."""

    class_estimates: dict[str, Any] = {}
    available_scales: list[float] = []
    for semantic_id, category_id in SEMANTIC_TO_BIM_CATEGORY.items():
        class_support = support & (semantic_class == semantic_id) & (bim_category == category_id)
        _, estimate = _scale_from_mask(base, bim, class_support, parameters)
        class_name = STANFORD_SEMANTIC_CLASSES[semantic_id]
        class_estimates[class_name] = estimate
        if not estimate.fallback:
            available_scales.append(float(estimate.scale))
    if available_scales:
        scale = float(np.exp(np.median(np.log(np.asarray(available_scales, dtype=np.float64)))))
        used_fallback = False
    else:
        scale = float(fallback_estimate.scale)
        used_fallback = True
    prediction = (base * scale).astype(np.float32)
    return prediction, {
        "scale": scale,
        "fallback_to_all": used_fallback,
        "available_class_count": len(available_scales),
        "class_estimates": class_estimates,
    }


def _quantile_scale_from_mask(
    base: np.ndarray,
    bim: np.ndarray,
    support: np.ndarray,
    parameters: dict[str, Any],
    quantile: float,
) -> tuple[np.ndarray, float, int, bool]:
    valid = support & np.isfinite(base) & (base > 0) & np.isfinite(bim) & (bim > 0)
    ratios = (bim[valid] / base[valid]).astype(np.float32, copy=False)
    ratios = ratios[
        (ratios > float(parameters["ratio_min"])) & (ratios < float(parameters["ratio_max"]))
    ]
    support_count = int(ratios.size)
    fallback = support_count < int(parameters["min_samples"])
    scale = (
        1.0 if fallback else float(np.exp(np.quantile(np.log(ratios.astype(np.float64)), quantile)))
    )
    return (base * scale).astype(np.float32), scale, support_count, fallback


def _bim_nonedge(bim: np.ndarray) -> np.ndarray:
    safe_bim = np.nan_to_num(bim, nan=0.0).astype(np.float32, copy=False)
    gradient_x = cv2.Sobel(safe_bim, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(safe_bim, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gradient_x, gradient_y) < 0.25


def _normalized_gaussian_field(
    scaled: np.ndarray,
    bim: np.ndarray,
    source_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    consistency = float(PREVIOUS_FIXED_PARAMETERS["consistency_log_threshold"])
    sigma = float(PREVIOUS_FIXED_PARAMETERS["smoothing_sigma"])
    residual = np.log(np.maximum(bim, 1e-6)) - np.log(np.maximum(scaled, 1e-6))
    valid = (
        np.isfinite(bim)
        & (bim > 0)
        & np.isfinite(residual)
        & (np.abs(residual) <= consistency)
        & _bim_nonedge(bim)
    )
    if source_mask is not None:
        valid &= source_mask
    numerator = cv2.GaussianBlur(
        np.where(valid, residual, 0.0).astype(np.float32),
        (0, 0),
        sigma,
    )
    denominator = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    field = numerator / np.maximum(denominator, 1e-4)
    field[denominator < 0.05] = 0.0
    return (
        np.clip(field, -consistency, consistency).astype(np.float32),
        denominator.astype(np.float32),
        valid,
    )


def _apply_field(
    scaled: np.ndarray,
    field: np.ndarray,
    application_mask: np.ndarray,
) -> np.ndarray:
    alpha = float(PREVIOUS_FIXED_PARAMETERS["local_correction_alpha"])
    gated_field = np.where(application_mask, field, 0.0).astype(np.float32)
    return (scaled * np.exp(alpha * gated_field)).astype(np.float32)


def _classwise_prediction(
    scaled: np.ndarray,
    bim: np.ndarray,
    semantic_class: np.ndarray,
) -> tuple[np.ndarray, dict[int, int]]:
    combined = np.zeros_like(scaled, dtype=np.float32)
    source_counts: dict[int, int] = {}
    for class_id in CORE_CLASS_IDS:
        class_mask = semantic_class == class_id
        field, _, valid = _normalized_gaussian_field(scaled, bim, class_mask)
        combined[class_mask] = field[class_mask]
        source_counts[class_id] = int(valid.sum())
    return _apply_field(scaled, combined, np.isin(semantic_class, CORE_CLASS_IDS)), source_counts


def oracle_semantic_predictions(
    base: np.ndarray,
    bim: np.ndarray,
    bim_valid: np.ndarray,
    bim_category: np.ndarray,
    semantic_class: np.ndarray,
    scale_parameters: dict[str, Any],
    *,
    include_pixel_corrections: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build all predictions without accepting official depth as an argument."""

    if not (
        base.shape == bim.shape == bim_valid.shape == bim_category.shape == semantic_class.shape
    ):
        raise ValueError("base/BIM/BIM-valid/BIM-category/semantic arrays must have equal shapes")
    clean_bim = np.where(bim_valid, bim, 0.0).astype(np.float32)
    masks = _semantic_masks(semantic_class)
    if include_pixel_corrections:
        scaled_all, local_all, _, _, all_estimate = configured_scale_and_local_features(
            base,
            clean_bim,
            scale_parameters,
        )
    else:
        scaled_all, all_estimate = _scale_from_mask(
            base,
            clean_bim,
            bim_valid,
            scale_parameters,
        )
        local_all = None
    scaled_core, core_estimate = _scale_from_mask(
        base,
        clean_bim,
        bim_valid & masks["core"],
        scale_parameters,
    )

    expected_bim_category = np.full(semantic_class.shape, 255, dtype=np.uint8)
    for semantic_id, category_id in SEMANTIC_TO_BIM_CATEGORY.items():
        expected_bim_category[semantic_class == semantic_id] = category_id
    category_match = (
        bim_valid & (expected_bim_category != 255) & (bim_category == expected_bim_category)
    )
    bim_nonedge = _bim_nonedge(clean_bim)
    category_bim_nonedge = category_match & bim_nonedge
    category_interior = category_bim_nonedge & ~masks["boundary"]
    scaled_category_match, category_match_estimate = _scale_from_mask(
        base,
        clean_bim,
        category_match,
        scale_parameters,
    )
    scaled_category_bim_nonedge, category_bim_nonedge_estimate = _scale_from_mask(
        base,
        clean_bim,
        category_bim_nonedge,
        scale_parameters,
    )
    scaled_category_interior, category_interior_estimate = _scale_from_mask(
        base,
        clean_bim,
        category_interior,
        scale_parameters,
    )
    scaled_category_balanced, balanced_diagnostics = _balanced_category_scale(
        base,
        clean_bim,
        semantic_class,
        bim_category,
        category_interior,
        scale_parameters,
        all_estimate,
    )
    quantile_predictions: dict[str, np.ndarray] = {}
    quantile_diagnostics: dict[str, tuple[float, int, bool]] = {}
    for quantile in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
        suffix = f"q{round(quantile * 100):02d}"
        prediction, scale, support_count, fallback = _quantile_scale_from_mask(
            base,
            clean_bim,
            category_match,
            scale_parameters,
            quantile,
        )
        quantile_predictions[f"scale_category_match_{suffix}"] = prediction
        quantile_diagnostics[suffix] = (scale, support_count, fallback)

    scale_predictions = {
        "scale_all": scaled_all,
        "scale_core": scaled_core,
        "scale_category_match": scaled_category_match,
        "scale_category_match_bim_nonedge": scaled_category_bim_nonedge,
        "scale_category_match_interior": scaled_category_interior,
        "scale_category_balanced": scaled_category_balanced,
        **quantile_predictions,
    }
    scale_diagnostics = {
        "all_scale": float(all_estimate.scale),
        "core_scale": float(core_estimate.scale),
        "category_match_scale": float(category_match_estimate.scale),
        "category_bim_nonedge_scale": float(category_bim_nonedge_estimate.scale),
        "category_interior_scale": float(category_interior_estimate.scale),
        "category_balanced_scale": float(balanced_diagnostics["scale"]),
        **{
            f"category_match_{suffix}_scale": float(values[0])
            for suffix, values in quantile_diagnostics.items()
        },
        "all_scale_support": int(all_estimate.support_count),
        "core_scale_support": int(core_estimate.support_count),
        "category_match_scale_support": int(category_match_estimate.support_count),
        "category_bim_nonedge_scale_support": int(category_bim_nonedge_estimate.support_count),
        "category_interior_scale_support": int(category_interior_estimate.support_count),
        "category_balanced_available_classes": int(balanced_diagnostics["available_class_count"]),
        "category_balanced_fallback_to_all": bool(balanced_diagnostics["fallback_to_all"]),
        "all_scale_fallback": bool(all_estimate.fallback),
        "core_scale_fallback": bool(core_estimate.fallback),
    }
    if not include_pixel_corrections:
        if tuple(scale_predictions) != SCALE_VARIANTS:
            raise RuntimeError("Internal scale variant order changed")
        return scale_predictions, scale_diagnostics

    all_field, _, all_source = _normalized_gaussian_field(scaled_all, clean_bim)
    core_field, _, core_source = _normalized_gaussian_field(
        scaled_all,
        clean_bim,
        masks["core"],
    )
    semantic_apply = _apply_field(scaled_all, all_field, masks["core"])
    semantic_source = _apply_field(scaled_all, core_field, masks["core"])
    classwise_all, class_counts_all = _classwise_prediction(
        scaled_all,
        clean_bim,
        semantic_class,
    )
    classwise_core, class_counts_core = _classwise_prediction(
        scaled_core,
        clean_bim,
        semantic_class,
    )
    replacement = scaled_all.copy()
    replacement_mask = masks["core"] & bim_valid & _bim_nonedge(clean_bim)
    replacement[replacement_mask] = clean_bim[replacement_mask]

    category_nonedge = category_bim_nonedge
    log_residual = np.log(np.maximum(clean_bim, 1e-6)) - np.log(np.maximum(scaled_all, 1e-6))
    consistency = float(PREVIOUS_FIXED_PARAMETERS["consistency_log_threshold"])
    category_consistent = category_nonedge & (np.abs(log_residual) <= consistency)
    category_replacement = scaled_all.copy()
    category_replacement[category_nonedge] = clean_bim[category_nonedge]
    category_asymmetric = category_nonedge & (log_residual <= consistency)
    category_asymmetric_replacement = scaled_all.copy()
    category_asymmetric_replacement[category_asymmetric] = clean_bim[category_asymmetric]
    category_consistent_replacement = scaled_all.copy()
    category_consistent_replacement[category_consistent] = clean_bim[category_consistent]
    soft_weight = np.zeros_like(scaled_all, dtype=np.float32)
    soft_weight[category_nonedge] = np.exp(
        -0.5 * np.square(log_residual[category_nonedge] / consistency)
    )
    category_soft_blend = (
        scaled_all * np.exp(soft_weight * np.clip(log_residual, -1.0, 1.0))
    ).astype(np.float32)

    assert local_all is not None
    predictions = {
        **scale_predictions,
        "local_all": local_all,
        "semantic_apply_gate": semantic_apply,
        "semantic_source_gate": semantic_source,
        "semantic_classwise": classwise_all,
        "semantic_core_scale_classwise": classwise_core,
        "semantic_bim_replace_nonedge": replacement,
        "semantic_category_replace_nonedge": category_replacement,
        "semantic_category_replace_asymmetric": category_asymmetric_replacement,
        "semantic_category_replace_consistent": category_consistent_replacement,
        "semantic_category_soft_blend": category_soft_blend,
    }
    if tuple(predictions) != VARIANTS:
        raise RuntimeError("Internal variant order changed")
    for name, prediction in predictions.items():
        if prediction.shape != base.shape or not np.isfinite(prediction).all():
            raise RuntimeError(f"{name} produced invalid prediction values")
        if np.any(prediction <= 0):
            raise RuntimeError(f"{name} produced non-positive prediction values")
    return predictions, {
        **scale_diagnostics,
        "all_local_source_pixels": int(all_source.sum()),
        "core_local_source_pixels": int(core_source.sum()),
        "classwise_source_pixels_all_scale": class_counts_all,
        "classwise_source_pixels_core_scale": class_counts_core,
        "semantic_core_pixels": int(masks["core"].sum()),
        "semantic_replacement_pixels": int(replacement_mask.sum()),
        "semantic_category_match_pixels": int(category_match.sum()),
        "semantic_category_nonedge_pixels": int(category_nonedge.sum()),
        "semantic_category_asymmetric_pixels": int(category_asymmetric.sum()),
        "semantic_category_consistent_pixels": int(category_consistent.sum()),
    }


def _metric_sums(prediction: np.ndarray, target: np.ndarray, support: np.ndarray) -> MetricSums:
    count = int(support.sum())
    if count == 0:
        return MetricSums()
    selected = prediction[support].astype(np.float64)
    gt = target[support].astype(np.float64)
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise RuntimeError("Invalid prediction on fixed evaluation support")
    error = np.abs(selected - gt)
    ratio = np.maximum(selected / gt, gt / selected)
    return MetricSums(
        count=count,
        abs_rel_sum=float(np.sum(error / gt, dtype=np.float64)),
        abs_error_sum=float(np.sum(error, dtype=np.float64)),
        squared_error_sum=float(np.sum(error * error, dtype=np.float64)),
        delta1_count=int(np.sum(ratio < 1.25)),
    )


def _evaluation_subsets(
    gt: np.ndarray,
    gt_valid: np.ndarray,
    bim: np.ndarray,
    bim_valid: np.ndarray,
    semantic_class: np.ndarray,
    *,
    minimum_depth: float,
    maximum_depth: float,
) -> dict[str, np.ndarray]:
    masks = _semantic_masks(semantic_class)
    fixed = gt_valid & np.isfinite(gt) & (gt >= minimum_depth) & (gt <= maximum_depth)
    tolerance = np.maximum(0.10, 0.05 * bim)
    core_hit = masks["core"] & bim_valid
    return {
        "all": fixed,
        "core_structure": fixed & masks["core"],
        "core_structure_bim_hit": fixed & core_hit,
        "core_structure_consistent": fixed & core_hit & (np.abs(gt - bim) <= tolerance),
        "furniture": fixed & masks["furniture"],
        "door_window": fixed & masks["door_window"],
        "non_structural": fixed & masks["non_structural"],
        "bim_foreground_conflict": fixed & bim_valid & (gt < bim - tolerance),
        "semantic_boundary": fixed & masks["boundary"],
    }


def _evaluate_record(
    record: dict[str, Any],
    scale_parameters: dict[str, Any],
    minimum_depth: float,
    maximum_depth: float,
    include_pixel_corrections: bool,
    selection_only: bool,
) -> dict[str, Any]:
    with np.load(record["sample"]) as sample:
        required = {
            "base_depth",
            "bim_depth",
            "bim_valid",
            "bim_category",
            "semantic_class",
            "gt_depth",
            "gt_valid",
        }
        missing = sorted(required - set(sample.files))
        if missing:
            raise RuntimeError(f"{record['id']}: prepared sample is missing {missing}")
        base = sample["base_depth"].astype(np.float32)
        bim = sample["bim_depth"].astype(np.float32)
        bim_valid = (sample["bim_valid"] > 0) & np.isfinite(bim) & (bim > 0)
        bim_category = sample["bim_category"].astype(np.uint8)
        semantic_class = sample["semantic_class"].astype(np.uint8)

        # Oracle semantic is intentionally allowed, but official depth is not
        # accessed until every prediction has been completely constructed.
        predictions, diagnostics = oracle_semantic_predictions(
            base,
            bim,
            bim_valid,
            bim_category,
            semantic_class,
            scale_parameters,
            include_pixel_corrections=include_pixel_corrections,
        )

        gt = sample["gt_depth"].astype(np.float32)
        gt_valid = sample["gt_valid"] > 0
        subsets = _evaluation_subsets(
            gt,
            gt_valid,
            np.where(bim_valid, bim, 0.0),
            bim_valid,
            semantic_class,
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
        )
        if selection_only:
            subsets = {"all": subsets["all"]}
        metrics = {
            subset: {
                variant: _metric_sums(prediction, gt, support)
                for variant, prediction in predictions.items()
            }
            for subset, support in subsets.items()
        }
    return {
        "sample_id": str(record["id"]),
        "room": str(record["region"]),
        "camera_uuid": str(record.get("camera_uuid", "")),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def _mean_metrics(rows: list[dict[str, float | int | None]]) -> dict[str, Any]:
    available = [row for row in rows if int(row["count"] or 0) > 0]
    return {
        "count": len(available),
        **{
            metric: (
                float(np.mean([float(row[metric]) for row in available])) if available else None
            )
            for metric in ("abs_rel", "mae", "rmse", "delta1")
        },
    }


def _paired_room_bootstrap(
    per_room: dict[str, Any],
    subset: str,
    candidate: str,
    reference: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    rooms = [
        room
        for room in sorted(per_room)
        if int(per_room[room][subset][candidate]["count"]) > 0
        and int(per_room[room][subset][reference]["count"]) > 0
    ]
    differences = np.asarray(
        [
            float(per_room[room][subset][candidate]["abs_rel"])
            - float(per_room[room][subset][reference]["abs_rel"])
            for room in rooms
        ],
        dtype=np.float64,
    )
    if not rooms:
        return {
            "rooms": 0,
            "mean_difference": None,
            "confidence_interval_95": [None, None],
        }
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rooms), size=(repetitions, len(rooms)))
    bootstrap = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "rooms": len(rooms),
        "room_ids": rooms,
        "candidate": candidate,
        "reference": reference,
        "difference_definition": "candidate-reference; negative is better",
        "mean_difference": float(differences.mean()),
        "confidence_interval_95": [float(lower), float(upper)],
        "candidate_better_rooms": int(np.sum(differences < 0)),
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }


def _aggregate(
    rows: list[dict[str, Any]],
    variants: tuple[str, ...],
    subsets: tuple[str, ...],
    repetitions: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pixel: dict[str, dict[str, MetricSums]] = defaultdict(lambda: defaultdict(MetricSums))
    frames: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    room_sums: dict[str, dict[str, dict[str, MetricSums]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(MetricSums))
    )
    for row in rows:
        for subset, variant_metrics in row["metrics"].items():
            for variant, sums in variant_metrics.items():
                pixel[subset][variant].add(sums)
                room_sums[row["room"]][subset][variant].add(sums)
                frames[subset][variant].append(sums.metrics())

    per_room = {
        room: {
            subset: {variant: sums.metrics() for variant, sums in variants.items()}
            for subset, variants in subsets.items()
        }
        for room, subsets in room_sums.items()
    }
    aggregates: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for subset in subsets:
        aggregates[subset] = {}
        bootstrap[subset] = {}
        for variant in variants:
            room_rows = [
                per_room[room][subset][variant]
                for room in sorted(per_room)
                if int(per_room[room][subset][variant]["count"]) > 0
            ]
            aggregates[subset][variant] = {
                "pixel_micro": pixel[subset][variant].metrics(),
                "frame_macro": _mean_metrics(frames[subset][variant]),
                "room_macro": _mean_metrics(room_rows),
            }
            for reference in ("scale_all", "local_all"):
                if reference not in variants or variant == reference:
                    continue
                bootstrap[subset][f"{variant}_vs_{reference}"] = _paired_room_bootstrap(
                    per_room,
                    subset,
                    variant,
                    reference,
                    repetitions,
                    seed,
                )
    return aggregates, per_room, bootstrap


def _per_frame_csv(
    rows: list[dict[str, Any]],
    variants: tuple[str, ...],
    diagnostic_fields: tuple[str, ...],
    subsets: tuple[str, ...],
) -> tuple[list[str], list[dict[str, Any]]]:
    fieldnames = ["sample_id", "room", "camera_uuid"]
    fieldnames.extend(diagnostic_fields)
    for subset in subsets:
        for variant in variants:
            for metric in ("count", "abs_rel", "mae", "rmse", "delta1"):
                fieldnames.append(f"{subset}__{variant}__{metric}")
    output: list[dict[str, Any]] = []
    for row in rows:
        flattened: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "room": row["room"],
            "camera_uuid": row["camera_uuid"],
        }
        for key in diagnostic_fields:
            flattened[key] = row["diagnostics"][key]
        for subset, variant_metrics in row["metrics"].items():
            for variant, sums in variant_metrics.items():
                for metric, value in sums.metrics().items():
                    flattened[f"{subset}__{variant}__{metric}"] = value
        output.append(flattened)
    return fieldnames, output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite oracle semantic result directory: {output_dir}"
        )
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    if "stanford" not in str(cfg.experiment.name).lower():
        raise ValueError("Oracle semantic ablation requires a Stanford Area_1 config")
    universal_protocol = dict(validate_universal_scale_protocol(cfg))
    universal_protocol["path"] = _repo_relative(Path(str(universal_protocol["path"])))
    scale_parameters = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    if scale_parameters["name"] != "log_upper_cap_v1":
        raise ValueError("Oracle semantic ablation requires the universal robust scale estimator")
    include_pixel_corrections = args.study == "full-historical"
    variants = VARIANTS if include_pixel_corrections else SCALE_VARIANTS
    subsets = ("all",) if args.selection_only else tuple(SUBSET_DEFINITIONS)
    diagnostic_fields = DIAGNOSTIC_FIELDS if include_pixel_corrections else SCALE_DIAGNOSTIC_FIELDS
    dataset = BIMDepthDataset(cfg, args.split, augment=False, require_ground_truth=True)
    records = list(dataset.records)
    evaluate = lambda record: _evaluate_record(
        record,
        scale_parameters,
        float(cfg.data.min_depth),
        float(cfg.data.max_depth),
        include_pixel_corrections,
        args.selection_only,
    )
    if args.workers == 1:
        rows = [evaluate(record) for record in records]
    else:
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                rows = list(executor.map(evaluate, records))
        finally:
            cv2.setNumThreads(previous_threads)

    aggregates, per_room, bootstrap = _aggregate(
        rows,
        variants,
        subsets,
        args.bootstrap_repetitions,
        args.bootstrap_seed,
    )
    fieldnames, per_frame_rows = _per_frame_csv(
        rows,
        variants,
        diagnostic_fields,
        subsets,
    )
    output_dir.mkdir(parents=True)
    per_frame_path = output_dir / "per_frame.csv"
    temporary_csv = output_dir / ".per_frame.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame_rows)
    os.replace(temporary_csv, per_frame_path)

    split_provenance = dict(dataset.split_provenance)
    split_provenance["annotation_file"] = "data/annotations/stanford_area1_room_v1.jsonl"
    summary = {
        "schema_version": 2,
        "protocol": "stanford-area1-oracle-semantic-global-scale-v2",
        "status": "complete",
        "publication_status": "exploratory_validation_only_oracle_upper_bound",
        "split": args.split,
        "test_samples_or_artifacts_accessed": False,
        "sample_count": len(rows),
        "room_count": len(per_room),
        "method_scope": {
            "training_free": True,
            "learned_checkpoint": False,
            "official_semantic_used_at_prediction_time": True,
            "official_depth_used_at_prediction_time": False,
            "deployable": False,
            "study": args.study,
            "selection_only": args.selection_only,
            "semantic_role": (
                "select BIM/DA3 ratio samples for one global scale applied to every pixel"
            ),
            "pixelwise_semantic_output_gate": include_pixel_corrections,
            "purpose": "estimate semantic global-scale value before pinning a semantic predictor",
        },
        "semantic_protocol": {
            "classes": list(STANFORD_SEMANTIC_CLASSES),
            "bim_supported_core_classes": list(CORE_CLASS_NAMES),
            "explicitly_not_core": [
                "window",
                "door",
                "table",
                "chair",
                "sofa",
                "bookcase",
                "board",
                "clutter",
            ],
            "official_semantic_is_privileged_annotation": True,
        },
        "variant_order": list(variants),
        "variant_definitions": {name: VARIANT_DEFINITIONS[name] for name in variants},
        "subset_definitions": {name: SUBSET_DEFINITIONS[name] for name in subsets},
        "fixed_parameters": {
            **PREVIOUS_FIXED_PARAMETERS,
            "scale_estimator": scale_parameters,
        },
        "universal_scale_protocol": universal_protocol,
        "split_provenance": split_provenance,
        "aggregates": aggregates,
        "per_room": per_room,
        "paired_room_bootstrap_abs_rel": bootstrap,
        "artifacts": {
            "per_frame_csv": _repo_relative(per_frame_path),
            "per_frame_csv_sha256": _sha256(per_frame_path),
        },
    }
    summary_path = output_dir / "summary.json"
    _atomic_text(
        summary_path,
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    provenance = {
        "schema_version": 1,
        "protocol": summary["protocol"],
        "config": _repo_relative(config_path),
        "config_sha256": _sha256(config_path),
        "script": "scripts/analysis/ablate_oracle_semantic_bim.py",
        "script_sha256": _sha256(Path(__file__).resolve()),
        "summary_sha256": _sha256(summary_path),
        "per_frame_csv_sha256": _sha256(per_frame_path),
        "sample_count": len(rows),
        "ordered_split_ids_sha256": split_provenance["ordered_ids_sha256"][args.split],
        "annotation_raw_sha256": split_provenance["annotation_raw_sha256"],
        "split_fingerprint_sha256": split_provenance["fingerprint_sha256"],
        "test_accessed": False,
    }
    _atomic_text(
        output_dir / "provenance.json",
        json.dumps(_json_safe(provenance), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "samples": len(rows),
                "rooms": len(per_room),
                "all_pixel_micro_abs_rel": {
                    variant: aggregates["all"][variant]["pixel_micro"]["abs_rel"]
                    for variant in variants
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
