from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.baselines import estimate_robust_bim_scale

SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "search_stanford_scale_quantile.py"
SPEC = importlib.util.spec_from_file_location("search_stanford_scale_quantile", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_fast_absrel_sums_match_direct_evaluation() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    target = np.asarray([[1.5, 1.0], [4.0, 2.0]], dtype=np.float32)
    valid = np.asarray([[True, True], [False, True]])
    scales = np.asarray([0.5, 1.0, 1.7, 3.0], dtype=np.float64)

    actual = MODULE._absrel_sums_for_scales(base, target, valid, scales)
    selected_base = base[valid].astype(np.float64)
    selected_target = target[valid].astype(np.float64)
    expected = np.asarray(
        [
            np.sum(np.abs(selected_base * scale - selected_target) / selected_target)
            for scale in scales
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_selected_index_prefers_highest_quantile_for_exact_tie() -> None:
    assert MODULE._selected_index(np.asarray([0.2, 0.1, 0.1, 0.3])) == 2


def test_test_cli_does_not_synthesize_selection_receipt() -> None:
    args = MODULE.parse_args(["--config", "config.yaml", "--split", "test", "--output", "results"])
    assert args.selection_receipt is None

    with pytest.raises(SystemExit):
        MODULE.parse_args(
            [
                "--config",
                "config.yaml",
                "--split",
                "test",
                "--output",
                "results",
                "--quantile",
                "0.54",
            ]
        )


def test_robust_reference_scale_matches_public_estimator() -> None:
    ratios = np.linspace(0.3, 3.0, 1000, dtype=np.float32)
    parameters = {
        "name": "log_upper_cap_v1",
        "q10_log_cap": np.inf,
        "q25_log_cap": 0.05,
        "ratio_min": 0.2,
        "ratio_max": 5.0,
        "min_samples": 100,
    }
    actual = MODULE._robust_scale(ratios, parameters)
    expected = estimate_robust_bim_scale(
        np.ones_like(ratios),
        ratios,
        q10_log_cap=np.inf,
        q25_log_cap=0.05,
    ).scale
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
