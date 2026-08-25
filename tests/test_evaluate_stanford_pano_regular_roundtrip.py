from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data.stanford_pano import RegularPanoLookup
from scripts.analysis import evaluate_stanford_pano_regular_roundtrip as experiment


def test_cli_is_validation_only_and_has_no_model_or_bim_escape_hatch() -> None:
    args = experiment.parse_args(["--tangent-manifest", "val.json"])

    assert not hasattr(args, "split")
    assert not hasattr(args, "checkpoint")
    assert not hasattr(args, "bim")
    assert not hasattr(args, "huber_log_delta")
    with pytest.raises(SystemExit):
        experiment.parse_args(["--tangent-manifest", "val.json", "--split", "test"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--tangent-manifest", "val.json", "--checkpoint", "x.pt"])


def test_roundtrip_uses_raw_fallback_without_changing_shape() -> None:
    lookup = RegularPanoLookup(
        erp_pixels=np.asarray(
            [[[1.5, 0.5], [2.5, 0.5]], [[1.5, 1.5], [2.5, 1.5]]],
            dtype=np.float64,
        ),
        z_per_radial_range=np.ones((2, 2), dtype=np.float64),
        pano_shape=(2, 4),
        regular_shape=(2, 2),
    )
    pano_range = np.full((2, 4), 3.0, dtype=np.float32)
    pano_valid = np.ones((2, 4), dtype=bool)
    pano_valid[:, 2:] = False
    raw = np.full((2, 2), 7.0, dtype=np.float32)

    prediction, native = experiment._roundtrip_with_raw_fallback(
        pano_range,
        pano_valid,
        lookup,
        raw,
    )

    assert prediction.shape == raw.shape
    assert native.shape == raw.shape
    assert np.all(prediction[native] == pytest.approx(3.0))
    assert np.all(prediction[~native] == 7.0)


def test_metric_row_keeps_full_regular_support_and_reports_native_coverage() -> None:
    target = np.full((2, 3), 2.0, dtype=np.float32)
    prediction = target.copy()
    support = np.ones((2, 3), dtype=bool)
    native = support.copy()
    native[0, 0] = False
    row = experiment._metric_row(
        record={
            "id": "office_1/frame_0",
            "region": "office_1",
            "camera_uuid": "a" * 32,
            "frame_number": 0,
        },
        method="candidate",
        source_set="regular_plus_tangent14",
        fusion_method="joint_huber",
        prediction=prediction,
        target=target,
        support=support,
        native_valid=native,
    )

    assert row["count"] == 6
    assert row["native_roundtrip_pixels"] == 5
    assert row["fallback_pixels"] == 1
    assert row["native_roundtrip_coverage_fraction"] == pytest.approx(5 / 6)
    assert row["abs_rel"] == pytest.approx(0.0)


def _row(room: str, method: str, abs_rel: float, count: int = 10) -> dict[str, object]:
    return {
        "sample_id": f"{room}/{method}",
        "room": room,
        "station_id": f"station_{room}",
        "frame_number": 0,
        "method": method,
        "source_set": "source",
        "fusion_method": "fusion",
        "depth_method": "raw_da3",
        "fixed_support_pixels": count,
        "native_roundtrip_pixels": count,
        "native_roundtrip_coverage_fraction": 1.0,
        "fallback_pixels": 0,
        "abs_rel": abs_rel,
        "mae": abs_rel * 2,
        "rmse": abs_rel * 3,
        "delta1": 1.0 - abs_rel,
        "delta2": 1.0,
        "delta3": 1.0,
        "count": count,
    }


def test_pixel_aggregate_and_room_bootstrap_preserve_paired_rooms() -> None:
    methods = experiment.METHOD_NAMES
    rows = []
    for room_index, room in enumerate(("room_a", "room_b")):
        for method in methods:
            value = 0.2 + 0.01 * room_index
            if method == methods[-1]:
                value -= 0.05
            rows.append(_row(room, method, value))

    aggregate = experiment._pixel_aggregate([row for row in rows if row["method"] == methods[-1]])
    assert aggregate["abs_rel"] == pytest.approx(0.155)

    first = experiment._room_cluster_bootstrap(
        rows,
        candidate=methods[-1],
        reference="raw_da3",
        repetitions=100,
    )
    second = experiment._room_cluster_bootstrap(
        rows,
        candidate=methods[-1],
        reference="raw_da3",
        repetitions=100,
    )
    assert first == second
    assert first["mean_difference"] == pytest.approx(-0.05)
    assert first["candidate_better_room_count"] == 2


def test_output_must_not_overwrite_existing_result(tmp_path: Path) -> None:
    output = tmp_path / "roundtrip"
    output.mkdir()
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment.strict_experiment._ensure_new_output(output)
