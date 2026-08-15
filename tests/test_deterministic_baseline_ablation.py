from __future__ import annotations

import numpy as np

from bim_priorda3.baselines import configured_scale_and_local_features
from scripts.analysis.ablate_deterministic_bim_direct import VARIANTS, variant_predictions


def _scale_parameters() -> dict[str, float | int | str]:
    return {
        "name": "log_upper_cap_v1",
        "q10_log_cap": float("inf"),
        "q25_log_cap": 0.05,
        "ratio_min": 0.2,
        "ratio_max": 5.0,
        "min_samples": 100,
    }


def test_full_variant_is_the_authoritative_public_baseline() -> None:
    base = np.ones((24, 24), dtype=np.float32)
    bim = np.linspace(0.9, 1.3, base.size, dtype=np.float32).reshape(base.shape)

    predictions, diagnostics = variant_predictions(base, bim, _scale_parameters())
    _, expected, _, _, estimate = configured_scale_and_local_features(
        base,
        bim,
        _scale_parameters(),
    )

    assert tuple(predictions) == VARIANTS
    np.testing.assert_array_equal(predictions["full"], expected)
    assert diagnostics["support_count"] == estimate.support_count
    assert diagnostics["fallback"] is False


def test_min_samples_ablation_is_inert_when_registered_support_is_sufficient() -> None:
    base = np.ones((20, 20), dtype=np.float32)
    bim = np.full_like(base, 1.2)

    predictions, _ = variant_predictions(base, bim, _scale_parameters())

    np.testing.assert_array_equal(predictions["min_samples_1"], predictions["full"])


def test_edge_ablation_is_distinct_on_a_bim_discontinuity() -> None:
    base = np.ones((32, 32), dtype=np.float32)
    bim = np.full_like(base, 1.04)
    bim[:, 16:] = 0.96

    predictions, _ = variant_predictions(base, bim, _scale_parameters())

    assert not np.array_equal(predictions["no_edge_gate"], predictions["full"])
