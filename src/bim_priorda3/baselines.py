from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

PREVIOUS_FIXED_PARAMETERS = {
    "scale_quantile": 0.45,
    "consistency_log_threshold": 0.10,
    "smoothing_sigma": 64.0,
    "local_correction_alpha": 1.25,
}

LEGACY_SCALE_ESTIMATOR = "legacy_q45"
ROBUST_LOG_CAP_SCALE_ESTIMATOR = "log_upper_cap_v1"
BIM_INVALID_DEPTH_ATOL = 1e-6
LEGACY_SCALE_ESTIMATOR_DEFAULTS = {
    "ratio_min": 0.2,
    "ratio_max": 5.0,
    "min_samples": 100,
}


@dataclass(frozen=True)
class ScaleEstimate:
    """Auditable result of a deterministic BIM/monocular scale estimate."""

    scale: float
    support_count: int
    quantiles: tuple[tuple[float, float], ...]
    fallback: bool
    q10_cap_triggered: bool
    q25_cap_triggered: bool
    estimator: str


def _valid_scale_ratios(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
    *,
    ratio_min: float,
    ratio_max: float,
) -> np.ndarray:
    if prediction.shape != bim_depth.shape:
        raise ValueError(
            "Scale estimation expects prediction and BIM depth with equal shapes; "
            f"prediction={prediction.shape}, bim={bim_depth.shape}"
        )
    if not 0.0 < ratio_min < ratio_max:
        raise ValueError("Scale ratio bounds must satisfy 0 < ratio_min < ratio_max")
    valid = np.isfinite(prediction) & np.isfinite(bim_depth) & (prediction > 0) & (bim_depth > 0)
    ratios = bim_depth[valid] / prediction[valid]
    return ratios[(ratios > ratio_min) & (ratios < ratio_max)]


def _validated_log_cap(value: float, *, name: str) -> float:
    cap = float(value)
    if np.isnan(cap) or cap < 0:
        raise ValueError(f"{name} must be non-negative and not NaN")
    return cap


def resolve_scale_estimator_config(
    parameters: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a validated canonical model-input scale configuration.

    Missing configuration deliberately means the historical q=.45 estimator,
    which keeps old configs and checkpoints behavior-compatible.
    """

    raw = {} if parameters is None else dict(parameters)
    name = str(raw.get("name", LEGACY_SCALE_ESTIMATOR))
    common = {
        "name": name,
        "ratio_min": float(raw.get("ratio_min", LEGACY_SCALE_ESTIMATOR_DEFAULTS["ratio_min"])),
        "ratio_max": float(raw.get("ratio_max", LEGACY_SCALE_ESTIMATOR_DEFAULTS["ratio_max"])),
        "min_samples": int(
            raw.get(
                "min_samples",
                LEGACY_SCALE_ESTIMATOR_DEFAULTS["min_samples"],
            )
        ),
    }
    if not 0.0 < common["ratio_min"] < common["ratio_max"]:
        raise ValueError(
            "model.scale_estimator ratio bounds must satisfy 0 < ratio_min < ratio_max"
        )
    if common["min_samples"] < 1:
        raise ValueError("model.scale_estimator.min_samples must be positive")
    if name == LEGACY_SCALE_ESTIMATOR:
        unsupported = set(raw) - {"name", "ratio_min", "ratio_max", "min_samples"}
        if unsupported:
            raise ValueError(
                f"legacy_q45 scale estimator received unsupported fields: {sorted(unsupported)}"
            )
        non_default = {
            key: raw[key]
            for key, expected in LEGACY_SCALE_ESTIMATOR_DEFAULTS.items()
            if key in raw and float(raw[key]) != float(expected)
        }
        if non_default:
            raise ValueError(
                "model.scale_estimator legacy_q45 is the immutable historical "
                "baseline and only accepts ratio_min=0.2, ratio_max=5.0, "
                f"min_samples=100; non_default={non_default}"
            )
        return common
    if name != ROBUST_LOG_CAP_SCALE_ESTIMATOR:
        raise ValueError("model.scale_estimator.name must be 'legacy_q45' or 'log_upper_cap_v1'")
    required = {"q10_log_cap", "q25_log_cap"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"log_upper_cap_v1 scale estimator is missing required fields: {missing}")
    unsupported = set(raw) - {
        "name",
        "ratio_min",
        "ratio_max",
        "min_samples",
        *required,
    }
    if unsupported:
        raise ValueError(
            f"log_upper_cap_v1 scale estimator received unsupported fields: {sorted(unsupported)}"
        )
    common.update(
        {
            "q10_log_cap": _validated_log_cap(
                float(raw["q10_log_cap"]),
                name="model.scale_estimator.q10_log_cap",
            ),
            "q25_log_cap": _validated_log_cap(
                float(raw["q25_log_cap"]),
                name="model.scale_estimator.q25_log_cap",
            ),
        }
    )
    return common


def estimate_bim_scale(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
    quantile: float = 0.45,
) -> float:
    valid = np.isfinite(prediction) & np.isfinite(bim_depth) & (prediction > 0) & (bim_depth > 0)
    ratios = bim_depth[valid] / prediction[valid]
    ratios = ratios[(ratios > 0.2) & (ratios < 5.0)]
    return float(np.quantile(ratios, quantile)) if len(ratios) >= 100 else 1.0


def estimate_robust_bim_scale(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
    *,
    q10_log_cap: float,
    q25_log_cap: float,
    ratio_min: float = 0.2,
    ratio_max: float = 5.0,
    min_samples: int = 100,
) -> ScaleEstimate:
    """Estimate scale under one-sided positive contamination from occluders.

    A furniture-free envelope is behind furniture in the observed image, so
    ``BIM / prediction`` receives a positive tail.  The historical q=.45 value
    is retained when the distribution is compact and is otherwise capped by
    fixed offsets from q=.25 and q=.10 in log-scale.  The cap parameters must be
    selected before validation/test evaluation and are never inferred from GT
    at runtime.
    """

    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    q10_cap = _validated_log_cap(q10_log_cap, name="q10_log_cap")
    q25_cap = _validated_log_cap(q25_log_cap, name="q25_log_cap")
    ratios = _valid_scale_ratios(
        prediction,
        bim_depth,
        ratio_min=float(ratio_min),
        ratio_max=float(ratio_max),
    )
    support_count = int(ratios.size)
    if support_count < min_samples:
        return ScaleEstimate(
            scale=1.0,
            support_count=support_count,
            quantiles=(),
            fallback=True,
            q10_cap_triggered=False,
            q25_cap_triggered=False,
            estimator=ROBUST_LOG_CAP_SCALE_ESTIMATOR,
        )
    log_quantiles = np.quantile(
        np.log(ratios.astype(np.float64, copy=False)),
        (0.10, 0.25, 0.45),
    )
    q10, q25, q45 = (float(value) for value in log_quantiles)
    q10_bound = q10 + q10_cap
    q25_bound = q25 + q25_cap
    robust_log_scale = min(q45, q25_bound, q10_bound)
    tolerance = 1e-12
    return ScaleEstimate(
        scale=float(np.exp(robust_log_scale)),
        support_count=support_count,
        quantiles=tuple(
            (quantile, float(np.exp(value)))
            for quantile, value in zip((0.10, 0.25, 0.45), log_quantiles)
        ),
        fallback=False,
        q10_cap_triggered=(q10_bound < q45 - tolerance and q10_bound <= q25_bound),
        q25_cap_triggered=(q25_bound < q45 - tolerance and q25_bound <= q10_bound),
        estimator=ROBUST_LOG_CAP_SCALE_ESTIMATOR,
    )


def previous_local_correction(
    scaled_prediction: np.ndarray,
    bim_depth: np.ndarray,
    consistency: float = 0.10,
    sigma: float = 64.0,
) -> np.ndarray:
    field, _ = previous_local_correction_features(
        scaled_prediction,
        bim_depth,
        consistency,
        sigma,
    )
    return field


def previous_local_correction_features(
    scaled_prediction: np.ndarray,
    bim_depth: np.ndarray,
    consistency: float = 0.10,
    sigma: float = 64.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the correction field and its normalized spatial support."""
    residual = np.log(np.maximum(bim_depth, 1e-6)) - np.log(np.maximum(scaled_prediction, 1e-6))
    valid = (
        np.isfinite(bim_depth)
        & (bim_depth > 0)
        & np.isfinite(residual)
        & (np.abs(residual) <= consistency)
    )
    safe_bim = np.nan_to_num(bim_depth, nan=0.0)
    gradient_x = cv2.Sobel(safe_bim, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(safe_bim, cv2.CV_32F, 0, 1, ksize=3)
    valid &= np.hypot(gradient_x, gradient_y) < 0.25
    numerator = cv2.GaussianBlur(
        np.where(valid, residual, 0.0).astype(np.float32),
        (0, 0),
        sigma,
    )
    denominator = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
    field = numerator / np.maximum(denominator, 1e-4)
    field[denominator < 0.05] = 0.0
    support = np.clip(denominator / 0.20, 0.0, 1.0)
    return (
        np.clip(field, -consistency, consistency).astype(np.float32),
        support.astype(np.float32),
    )


def bim_scale_and_local_features(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return global scale, direct local correction, field, support, and scale.

    The learned V5 stage starts from ``scaled``. ``local`` remains the
    deterministic BIM-direct baseline used for acceptance and comparison.
    """
    parameters = PREVIOUS_FIXED_PARAMETERS
    scale = estimate_bim_scale(
        prediction,
        bim_depth,
        float(parameters["scale_quantile"]),
    )
    scaled = prediction * scale
    field, support = previous_local_correction_features(
        scaled,
        bim_depth,
        float(parameters["consistency_log_threshold"]),
        float(parameters["smoothing_sigma"]),
    )
    local = scaled * np.exp(float(parameters["local_correction_alpha"]) * field)
    return (
        scaled.astype(np.float32),
        local.astype(np.float32),
        field,
        support,
        scale,
    )


def robust_scale_and_local_features(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
    *,
    q10_log_cap: float,
    q25_log_cap: float,
    ratio_min: float = 0.2,
    ratio_max: float = 5.0,
    min_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ScaleEstimate]:
    """Return robust scale and the unchanged deterministic local correction."""

    estimate = estimate_robust_bim_scale(
        prediction,
        bim_depth,
        q10_log_cap=q10_log_cap,
        q25_log_cap=q25_log_cap,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        min_samples=min_samples,
    )
    scaled = prediction * estimate.scale
    parameters = PREVIOUS_FIXED_PARAMETERS
    field, support = previous_local_correction_features(
        scaled,
        bim_depth,
        float(parameters["consistency_log_threshold"]),
        float(parameters["smoothing_sigma"]),
    )
    local = scaled * np.exp(float(parameters["local_correction_alpha"]) * field)
    return (
        scaled.astype(np.float32),
        local.astype(np.float32),
        field,
        support,
        estimate,
    )


def configured_scale_and_local_features(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
    parameters: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ScaleEstimate]:
    """Apply the configured model-input scale without changing legacy APIs."""

    resolved = resolve_scale_estimator_config(parameters)
    if resolved["name"] == ROBUST_LOG_CAP_SCALE_ESTIMATOR:
        return robust_scale_and_local_features(
            prediction,
            bim_depth,
            q10_log_cap=float(resolved["q10_log_cap"]),
            q25_log_cap=float(resolved["q25_log_cap"]),
            ratio_min=float(resolved["ratio_min"]),
            ratio_max=float(resolved["ratio_max"]),
            min_samples=int(resolved["min_samples"]),
        )
    scaled, local, field, support, scale = bim_scale_and_local_features(
        prediction,
        bim_depth,
    )
    valid_ratios = _valid_scale_ratios(
        prediction,
        bim_depth,
        ratio_min=0.2,
        ratio_max=5.0,
    )
    support_count = int(valid_ratios.size)
    return (
        scaled,
        local,
        field,
        support,
        ScaleEstimate(
            scale=float(scale),
            support_count=support_count,
            quantiles=((0.45, float(scale)),) if support_count >= 100 else (),
            fallback=support_count < 100,
            q10_cap_triggered=False,
            q25_cap_triggered=False,
            estimator=LEGACY_SCALE_ESTIMATOR,
        ),
    )


def previous_scale_baselines(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return scale-only and the previously used scale+local BIM refinement."""
    scaled, local, _, _, scale = bim_scale_and_local_features(
        prediction,
        bim_depth,
    )
    return scaled, local, scale
