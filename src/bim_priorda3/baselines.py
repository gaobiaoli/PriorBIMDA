from __future__ import annotations

import cv2
import numpy as np


PREVIOUS_FIXED_PARAMETERS = {
    "scale_quantile": 0.45,
    "consistency_log_threshold": 0.10,
    "smoothing_sigma": 64.0,
    "local_correction_alpha": 1.25,
}


def estimate_bim_scale(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
    quantile: float = 0.45,
) -> float:
    valid = (
        np.isfinite(prediction)
        & np.isfinite(bim_depth)
        & (prediction > 0)
        & (bim_depth > 0)
    )
    ratios = bim_depth[valid] / prediction[valid]
    ratios = ratios[(ratios > 0.2) & (ratios < 5.0)]
    return float(np.quantile(ratios, quantile)) if len(ratios) >= 100 else 1.0


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
    residual = np.log(np.maximum(bim_depth, 1e-6)) - np.log(
        np.maximum(scaled_prediction, 1e-6)
    )
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


def strong_anchor_features(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return global scale, strong anchor, correction, support, and scale.

    The anchor is numerically identical to ``previous_scale_baselines``.  The
    additional maps expose how it was produced so the learned stage can refine
    the strong solution rather than rediscovering it.
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


def previous_scale_baselines(
    prediction: np.ndarray,
    bim_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return scale-only and the previously used scale+local BIM refinement."""
    scaled, local, _, _, scale = strong_anchor_features(prediction, bim_depth)
    return scaled, local, scale
