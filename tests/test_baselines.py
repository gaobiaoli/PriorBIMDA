from __future__ import annotations

import numpy as np
import pytest

from bim_priorda3.baselines import (
    LEGACY_SCALE_ESTIMATOR,
    ROBUST_LOG_CAP_SCALE_ESTIMATOR,
    configured_scale_and_local_features,
    estimate_bim_scale,
    estimate_robust_bim_scale,
    previous_scale_baselines,
    resolve_scale_estimator_config,
    robust_scale_and_local_features,
)


def _one_sided_ratio_field() -> tuple[np.ndarray, np.ndarray]:
    prediction = np.ones((20, 20), dtype=np.float32)
    ratios = np.concatenate(
        (
            np.full(80, 1.0, dtype=np.float32),
            np.full(80, 1.1, dtype=np.float32),
            np.full(240, 3.0, dtype=np.float32),
        )
    ).reshape(20, 20)
    return prediction, ratios


def test_robust_log_caps_bound_one_sided_ratio_tail() -> None:
    prediction, bim = _one_sided_ratio_field()
    legacy = estimate_bim_scale(prediction, bim)
    estimate = estimate_robust_bim_scale(
        prediction,
        bim,
        q10_log_cap=0.20,
        q25_log_cap=0.05,
    )

    quantiles = dict(estimate.quantiles)
    expected = min(
        np.log(quantiles[0.45]),
        np.log(quantiles[0.25]) + 0.05,
        np.log(quantiles[0.10]) + 0.20,
    )
    assert legacy > estimate.scale
    assert estimate.scale == pytest.approx(np.exp(expected))
    assert estimate.support_count == prediction.size
    assert not estimate.fallback
    assert estimate.q10_cap_triggered or estimate.q25_cap_triggered
    assert estimate.estimator == ROBUST_LOG_CAP_SCALE_ESTIMATOR


def test_robust_estimator_has_auditable_insufficient_support_fallback() -> None:
    prediction = np.ones((8, 8), dtype=np.float32)
    estimate = estimate_robust_bim_scale(
        prediction,
        prediction * 2.0,
        q10_log_cap=0.20,
        q25_log_cap=0.05,
    )
    assert estimate.scale == 1.0
    assert estimate.support_count == 64
    assert estimate.quantiles == ()
    assert estimate.fallback
    assert not estimate.q10_cap_triggered
    assert not estimate.q25_cap_triggered


def test_robust_direct_reuses_the_previous_local_correction() -> None:
    prediction, bim = _one_sided_ratio_field()
    scaled, direct, field, support, estimate = robust_scale_and_local_features(
        prediction,
        bim,
        q10_log_cap=float("inf"),
        q25_log_cap=float("inf"),
    )
    legacy_scaled, legacy_direct, legacy_scale = previous_scale_baselines(
        prediction,
        bim,
    )
    assert estimate.scale == pytest.approx(legacy_scale)
    assert np.array_equal(scaled, legacy_scaled)
    assert np.array_equal(direct, legacy_direct)
    assert field.shape == prediction.shape
    assert support.shape == prediction.shape


def test_missing_scale_configuration_is_exactly_legacy() -> None:
    prediction, bim = _one_sided_ratio_field()
    resolved = resolve_scale_estimator_config(None)
    scaled, direct, _, _, estimate = configured_scale_and_local_features(
        prediction,
        bim,
        None,
    )
    expected_scaled, expected_direct, expected_scale = previous_scale_baselines(
        prediction,
        bim,
    )
    assert resolved["name"] == LEGACY_SCALE_ESTIMATOR
    assert estimate.scale == expected_scale
    assert np.array_equal(scaled, expected_scaled)
    assert np.array_equal(direct, expected_direct)


def test_legacy_scale_configuration_is_immutable_but_defaults_may_be_omitted() -> None:
    assert resolve_scale_estimator_config({"name": LEGACY_SCALE_ESTIMATOR}) == {
        "name": LEGACY_SCALE_ESTIMATOR,
        "ratio_min": 0.2,
        "ratio_max": 5.0,
        "min_samples": 100,
    }
    assert (
        resolve_scale_estimator_config(
            {
                "name": LEGACY_SCALE_ESTIMATOR,
                "ratio_min": 0.2,
                "ratio_max": 5.0,
                "min_samples": 100,
            }
        )["name"]
        == LEGACY_SCALE_ESTIMATOR
    )

    for field, value in (
        ("ratio_min", 0.3),
        ("ratio_max", 4.0),
        ("min_samples", 99),
        ("min_samples", 100.5),
    ):
        with pytest.raises(ValueError, match="immutable historical baseline"):
            resolve_scale_estimator_config({"name": LEGACY_SCALE_ESTIMATOR, field: value})


def test_robust_configuration_requires_both_frozen_caps() -> None:
    with pytest.raises(ValueError, match="missing required fields"):
        resolve_scale_estimator_config({"name": ROBUST_LOG_CAP_SCALE_ESTIMATOR, "q10_log_cap": 0.2})
    with pytest.raises(ValueError, match="non-negative"):
        resolve_scale_estimator_config(
            {
                "name": ROBUST_LOG_CAP_SCALE_ESTIMATOR,
                "q10_log_cap": -0.1,
                "q25_log_cap": 0.05,
            }
        )
