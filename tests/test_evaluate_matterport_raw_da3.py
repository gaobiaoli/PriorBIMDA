import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "model" / "evaluate_matterport_raw_da3.py"
SPEC = importlib.util.spec_from_file_location("evaluate_matterport_raw_da3", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_processed_geometry_matches_da3_upper_bound_resize_for_matterport_shape():
    intrinsics = np.array(
        [[1072.25, 0.0, 638.382], [0.0, 1072.15, 521.447], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    height, width, processed, focal_scale = MODULE.processed_geometry(
        1024, 1280, intrinsics, 504
    )
    assert (height, width) == (406, 504)
    np.testing.assert_allclose(processed[0, 0], 422.1984375)
    np.testing.assert_allclose(processed[1, 1], 425.09072265625)
    np.testing.assert_allclose(focal_scale, 1.4121486002604168)


def test_metric_values_are_exact_for_simple_depth_arrays():
    target = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    prediction = target * 2.0
    metrics = MODULE.metric_values(prediction, target, np.ones_like(target, dtype=bool))
    assert metrics["abs_rel"] == 1.0
    assert metrics["mae_m"] == 3.75
    assert metrics["delta1"] == 0.0
    assert metrics["delta2"] == 0.0
    assert metrics["delta3"] == 0.0
    np.testing.assert_allclose(metrics["rmse_log"], np.log(2.0))
    np.testing.assert_allclose(metrics["mean_log_error"], np.log(2.0))
    np.testing.assert_allclose(metrics["silog_x100"], 0.0, atol=1e-6)


def test_aggregate_rows_reconstructs_pixel_micro_rmse():
    base = {key: 0.0 for key in MODULE.SUMMARY_METRICS}
    base["mean_log_error"] = 0.0
    rows = [
        {**base, "valid_pixels": 1, "rmse_m": 1.0, "rmse_log": 0.1},
        {**base, "valid_pixels": 3, "rmse_m": 3.0, "rmse_log": 0.3},
    ]
    result = MODULE.aggregate_rows(rows)
    np.testing.assert_allclose(result["pixel_micro"]["rmse_m"], np.sqrt(7.0))
    np.testing.assert_allclose(result["pixel_micro"]["rmse_log"], np.sqrt(0.07))
    assert result["frames"] == 2
    assert result["valid_pixels"] == 4
