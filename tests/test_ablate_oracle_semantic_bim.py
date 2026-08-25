from __future__ import annotations

import importlib
import sys

import numpy as np

from bim_priorda3.data.stanford2d3ds import STANFORD_SEMANTIC_CLASSES

oracle = importlib.import_module("scripts.analysis.ablate_oracle_semantic_bim")


def _bim_categories(semantic: np.ndarray) -> np.ndarray:
    output = np.full(semantic.shape, 255, dtype=np.uint8)
    for semantic_id, category_id in oracle.SEMANTIC_TO_BIM_CATEGORY.items():
        output[semantic == semantic_id] = category_id
    return output


def _parameters() -> dict[str, object]:
    return {
        "name": "log_upper_cap_v1",
        "q10_log_cap": float("inf"),
        "q25_log_cap": 0.05,
        "ratio_min": 0.2,
        "ratio_max": 5.0,
        "min_samples": 1,
    }


def test_oracle_predictions_preserve_noncore_scaled_da3() -> None:
    shape = (40, 40)
    base = np.ones(shape, dtype=np.float32)
    bim = np.full(shape, 2.0, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    wall = STANFORD_SEMANTIC_CLASSES.index("wall")
    chair = STANFORD_SEMANTIC_CLASSES.index("chair")
    semantic = np.full(shape, chair, dtype=np.uint8)
    semantic[:, :20] = wall

    predictions, diagnostics = oracle.oracle_semantic_predictions(
        base,
        bim,
        valid,
        _bim_categories(semantic),
        semantic,
        _parameters(),
    )

    assert tuple(predictions) == oracle.VARIANTS
    for name in (
        "semantic_apply_gate",
        "semantic_source_gate",
        "semantic_classwise",
        "semantic_core_scale_classwise",
        "semantic_bim_replace_nonedge",
        "semantic_category_replace_nonedge",
        "semantic_category_replace_asymmetric",
        "semantic_category_replace_consistent",
        "semantic_category_soft_blend",
    ):
        baseline = (
            predictions["scale_core"]
            if name == "semantic_core_scale_classwise"
            else predictions["scale_all"]
        )
        np.testing.assert_array_equal(predictions[name][:, 20:], baseline[:, 20:])
    assert diagnostics["semantic_core_pixels"] == 800


def test_core_only_scale_excludes_furniture_ratios() -> None:
    shape = (20, 20)
    base = np.ones(shape, dtype=np.float32)
    bim = np.full(shape, 4.0, dtype=np.float32)
    semantic = np.full(
        shape,
        STANFORD_SEMANTIC_CLASSES.index("chair"),
        dtype=np.uint8,
    )
    semantic[:, :2] = STANFORD_SEMANTIC_CLASSES.index("wall")
    bim[:, :2] = 2.0

    predictions, diagnostics = oracle.oracle_semantic_predictions(
        base,
        bim,
        np.ones(shape, dtype=bool),
        _bim_categories(semantic),
        semantic,
        _parameters(),
    )

    assert diagnostics["core_scale"] == 2.0
    assert diagnostics["all_scale"] > diagnostics["core_scale"]
    np.testing.assert_allclose(predictions["scale_core"], 2.0)


def test_category_matched_semantics_choose_one_scale_for_the_whole_image() -> None:
    shape = (20, 20)
    base = np.ones(shape, dtype=np.float32)
    bim = np.full(shape, 4.0, dtype=np.float32)
    wall = STANFORD_SEMANTIC_CLASSES.index("wall")
    chair = STANFORD_SEMANTIC_CLASSES.index("chair")
    semantic = np.full(shape, chair, dtype=np.uint8)
    semantic[:, :10] = wall
    bim[:, :10] = 2.0

    predictions, diagnostics = oracle.oracle_semantic_predictions(
        base,
        bim,
        np.ones(shape, dtype=bool),
        _bim_categories(semantic),
        semantic,
        _parameters(),
    )

    assert diagnostics["category_match_scale"] == 2.0
    np.testing.assert_allclose(predictions["scale_category_match"], 2.0)
    np.testing.assert_array_equal(
        predictions["scale_category_match"][:, :10],
        predictions["scale_category_match"][:, 10:],
    )


def test_category_match_scale_rejects_wrong_structural_correspondence() -> None:
    shape = (20, 20)
    base = np.ones(shape, dtype=np.float32)
    bim = np.full(shape, 3.0, dtype=np.float32)
    wall = STANFORD_SEMANTIC_CLASSES.index("wall")
    semantic = np.full(shape, wall, dtype=np.uint8)
    bim_category = np.full(
        shape,
        oracle.ENVELOPE_CATEGORIES.index("floor"),
        dtype=np.uint8,
    )
    bim[:, :10] = 2.0
    bim_category[:, :10] = oracle.ENVELOPE_CATEGORIES.index("wall")

    predictions, diagnostics = oracle.oracle_semantic_predictions(
        base,
        bim,
        np.ones(shape, dtype=bool),
        bim_category,
        semantic,
        _parameters(),
    )

    assert diagnostics["category_match_scale"] == 2.0
    assert diagnostics["category_match_scale_support"] == 200
    np.testing.assert_allclose(predictions["scale_category_match"], 2.0)


def test_scale_only_study_does_not_construct_pixel_corrections(monkeypatch) -> None:
    monkeypatch.setattr(
        oracle,
        "configured_scale_and_local_features",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pixel-local baseline was constructed")
        ),
    )
    shape = (20, 20)
    wall = STANFORD_SEMANTIC_CLASSES.index("wall")
    semantic = np.full(shape, wall, dtype=np.uint8)
    predictions, diagnostics = oracle.oracle_semantic_predictions(
        np.ones(shape, dtype=np.float32),
        np.full(shape, 2.0, dtype=np.float32),
        np.ones(shape, dtype=bool),
        _bim_categories(semantic),
        semantic,
        _parameters(),
        include_pixel_corrections=False,
    )

    assert tuple(predictions) == oracle.SCALE_VARIANTS
    assert set(oracle.PIXEL_CORRECTION_VARIANTS).isdisjoint(predictions)
    assert set(oracle.SCALE_DIAGNOSTIC_FIELDS).issubset(diagnostics)
    assert set(oracle.PIXEL_DIAGNOSTIC_FIELDS).isdisjoint(diagnostics)


def test_scale_only_aggregate_does_not_require_local_reference() -> None:
    metric = oracle.MetricSums(
        count=2,
        abs_rel_sum=0.2,
        abs_error_sum=0.4,
        squared_error_sum=0.1,
        delta1_count=2,
    )
    rows = [
        {
            "room": "office_1",
            "metrics": {"all": {"scale_all": metric, "scale_core": metric}},
        }
    ]
    aggregates, _per_room, bootstrap = oracle._aggregate(
        rows,
        ("scale_all", "scale_core"),
        ("all",),
        10,
        42,
    )

    assert aggregates["all"]["scale_all"]["pixel_micro"]["count"] == 2
    assert "scale_core_vs_scale_all" in bootstrap["all"]
    assert all("local_all" not in key for key in bootstrap["all"])


def test_supported_consistent_subset_requires_semantic_core() -> None:
    wall = STANFORD_SEMANTIC_CLASSES.index("wall")
    chair = STANFORD_SEMANTIC_CLASSES.index("chair")
    semantic = np.asarray([[wall, wall, chair, chair]], dtype=np.uint8)
    gt = np.asarray([[2.0, 1.0, 2.0, 1.0]], dtype=np.float32)
    bim = np.full_like(gt, 2.0)
    subsets = oracle._evaluation_subsets(
        gt,
        np.ones_like(gt, dtype=bool),
        bim,
        np.ones_like(gt, dtype=bool),
        semantic,
        minimum_depth=0.2,
        maximum_depth=5.0,
    )
    np.testing.assert_array_equal(
        subsets["core_structure_consistent"],
        np.asarray([[True, False, False, False]]),
    )
    np.testing.assert_array_equal(
        subsets["furniture"],
        np.asarray([[False, False, True, True]]),
    )


def test_category_match_rejects_a_different_bim_component_class() -> None:
    shape = (24, 24)
    base = np.ones(shape, dtype=np.float32)
    bim = np.full(shape, 2.0, dtype=np.float32)
    semantic = np.full(
        shape,
        STANFORD_SEMANTIC_CLASSES.index("wall"),
        dtype=np.uint8,
    )
    wrong_category = np.full(
        shape,
        oracle.ENVELOPE_CATEGORIES.index("floor"),
        dtype=np.uint8,
    )
    predictions, diagnostics = oracle.oracle_semantic_predictions(
        base,
        bim,
        np.ones(shape, dtype=bool),
        wrong_category,
        semantic,
        _parameters(),
    )
    assert diagnostics["semantic_category_match_pixels"] == 0
    for name in (
        "semantic_category_replace_nonedge",
        "semantic_category_replace_asymmetric",
        "semantic_category_replace_consistent",
        "semantic_category_soft_blend",
    ):
        np.testing.assert_array_equal(predictions[name], predictions["scale_all"])


def test_cli_rejects_test_split(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["ablate_oracle_semantic_bim.py", "--split", "test"])
    try:
        oracle.parse_args()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("The oracle development script accepted the test split")


def test_json_safe_encodes_infinity_and_rejects_nan() -> None:
    assert oracle._json_safe({"cap": float("inf")}) == {"cap": "inf"}
    try:
        oracle._json_safe({"invalid": float("nan")})
    except ValueError as error:
        assert "NaN" in str(error)
    else:
        raise AssertionError("NaN was accepted in a formal result receipt")
