import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "model"
    / "evaluate_matterport_bimnet_early_fusion.py"
)
SPEC = importlib.util.spec_from_file_location(
    "evaluate_matterport_bimnet_early_fusion",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_frozen_reference_selection_is_used_and_recomputation_is_audited():
    recomputed = {
        "gt_quality_pass": True,
        "model_support_pass": True,
        "bim_applicability_pass": True,
        "effective_pass": True,
        "filter_reasons": "",
    }
    reference = {
        "gt_quality_pass": "True",
        "model_support_pass": "False",
        "bim_applicability_pass": "False",
        "effective_pass": "False",
        "filter_reasons": "insufficient_bim_da3_ratio_support",
    }
    selected, matched = MODULE._selection(recomputed, reference)
    assert selected["model_support_pass"] is False
    assert selected["effective_pass"] is False
    assert selected["filter_reasons"] == "insufficient_bim_da3_ratio_support"
    assert matched is False


def test_dense_aggregate_uses_pixel_micro_weights_and_paired_frames():
    rows = []
    for valid_pixels, raw_abs_rel, learned_abs_rel in (
        (1, 0.4, 0.2),
        (3, 0.2, 0.1),
    ):
        row = {"status": "ok", "gt_valid_pixels": valid_pixels}
        for prefix, abs_rel in (
            ("raw", raw_abs_rel),
            ("learned", learned_abs_rel),
            ("oracle_frame_scale", 0.05),
        ):
            for metric in MODULE.METRICS:
                row[f"{prefix}_{metric}"] = abs_rel
            row[f"{prefix}_rmse_m"] = abs_rel
            row[f"{prefix}_rmse_log"] = abs_rel / 2
            row[f"{prefix}_mean_log_error"] = 0.0
        rows.append(row)

    result = MODULE.aggregate_rows(rows)
    raw = result["predictions"]["raw"]["pixel_micro"]
    learned = result["predictions"]["learned"]["pixel_micro"]
    np.testing.assert_allclose(raw["abs_rel"], 0.25)
    np.testing.assert_allclose(learned["abs_rel"], 0.125)
    np.testing.assert_allclose(
        result["learned_vs_raw"]["pixel_micro_abs_rel_relative_improvement"],
        0.5,
    )
    assert result["learned_vs_raw"]["frame_win_fraction"] == 1.0
