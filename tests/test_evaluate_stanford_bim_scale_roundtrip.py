from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import evaluate_stanford_bim_scale_roundtrip as experiment


def test_cli_is_validation_only_and_has_no_tangent_or_test_escape_hatch() -> None:
    args = experiment.parse_args([])

    assert not hasattr(args, "split")
    assert not hasattr(args, "checkpoint")
    assert not hasattr(args, "tangent_manifest")
    assert not hasattr(args, "bim")
    with pytest.raises(SystemExit):
        experiment.parse_args(["--split", "test"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--tangent-manifest", "anything.json"])


def test_method_matrix_keeps_same_baseline_for_every_joint_variant() -> None:
    assert experiment.METHOD_NAMES == (
        "universal_scale__per_frame",
        "universal_scale__projection_roundtrip",
        "universal_scale__joint_weighted_log",
        "universal_scale__joint_huber",
        "universal_scale__joint_synchronized_huber",
        "bim_direct__per_frame",
        "bim_direct__projection_roundtrip",
        "bim_direct__joint_weighted_log",
        "bim_direct__joint_huber",
        "bim_direct__joint_synchronized_huber",
    )
    assert experiment.PRIMARY_DEPTH_METHOD == "universal_scale"
    assert experiment.PRIMARY_FUSION_METHOD == "joint_huber"


def _row(room: str, method: str, abs_rel: float) -> dict[str, object]:
    return {
        "room": room,
        "method": method,
        "count": 10,
        "abs_rel": abs_rel,
    }


def test_room_bootstrap_is_paired_and_deterministic() -> None:
    rows = []
    for room_index, room in enumerate(("room_a", "room_b")):
        for method in experiment.METHOD_NAMES:
            value = 0.20 + 0.01 * room_index
            if method == "universal_scale__joint_huber":
                value -= 0.03
            rows.append(_row(room, method, value))
    first = experiment._room_cluster_bootstrap(
        rows,
        candidate="universal_scale__joint_huber",
        reference="universal_scale__per_frame",
        repetitions=100,
    )
    second = experiment._room_cluster_bootstrap(
        rows,
        candidate="universal_scale__joint_huber",
        reference="universal_scale__per_frame",
        repetitions=100,
    )
    assert first == second
    assert first["mean_difference"] == pytest.approx(-0.03)
    assert first["candidate_better_room_count"] == 2


def test_identity_prediction_uses_requested_scaled_method() -> None:
    pixel_count = int(np.prod(experiment.PANO_SHAPE))
    view = experiment.evaluator.ProjectedView(
        frame_id="frame",
        indices=np.asarray([3, pixel_count - 2], dtype=np.int64),
        base_weights=np.ones(2, dtype=np.float32),
        photo_weights=np.ones(2, dtype=np.float32),
        log_ranges={
            "universal_scale": np.log(np.asarray([2.0, 4.0], dtype=np.float32)),
            "bim_direct": np.log(np.asarray([3.0, 5.0], dtype=np.float32)),
        },
    )
    prediction, valid = experiment._identity_erp_prediction(view, "universal_scale")
    flat = prediction.reshape(-1)
    assert flat[3] == pytest.approx(2.0)
    assert flat[-2] == pytest.approx(4.0)
    assert int(np.count_nonzero(valid)) == 2


def test_output_must_be_new(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        experiment.artifact_utils._ensure_new_output(output)
