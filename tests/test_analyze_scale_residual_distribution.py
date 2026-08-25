from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path("scripts/analysis/analyze_scale_residual_distribution.py")
SPEC = importlib.util.spec_from_file_location("analyze_scale_residual_distribution", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _synthetic_depth() -> np.ndarray:
    return np.repeat(np.linspace(0.21, 4.99, 500, dtype=np.float64), 20)


def test_power_fit_identifies_proportional_residual() -> None:
    gt = _synthetic_depth()
    pattern = np.tile(np.linspace(-0.08, 0.08, 20), 500)
    prediction = gt * np.exp(-pattern)
    summary = MODULE.summarize_residuals(
        gt,
        prediction,
        np.asarray(MODULE.DEFAULT_DEPTH_EDGES_M),
    )
    exponent = summary["bin_level_models_for_mean_absolute_meter_error"]["power_law"][
        "depth_exponent"
    ]
    assert 0.95 < exponent < 1.05
    stationarity = summary["cross_depth_stationarity"]
    assert stationarity["mean_absolute_log_ratio_cv"] < 0.02
    assert stationarity["mean_absolute_meter_error_cv"] > 0.4


def test_power_fit_identifies_additive_residual() -> None:
    gt = _synthetic_depth()
    pattern = np.tile(np.linspace(-0.04, 0.04, 20), 500)
    prediction = gt - pattern
    summary = MODULE.summarize_residuals(
        gt,
        prediction,
        np.asarray(MODULE.DEFAULT_DEPTH_EDGES_M),
    )
    exponent = summary["bin_level_models_for_mean_absolute_meter_error"]["power_law"][
        "depth_exponent"
    ]
    assert -0.05 < exponent < 0.05
    stationarity = summary["cross_depth_stationarity"]
    assert stationarity["mean_absolute_meter_error_cv"] < 0.02
    assert stationarity["mean_absolute_log_ratio_cv"] > 0.4
