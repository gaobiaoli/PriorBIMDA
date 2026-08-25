from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data.stanford_pano import StanfordPanorama
from scripts.analysis import evaluate_stanford_pano_single_plus_tangent as experiment
from scripts.model import evaluate_stanford_pano as evaluator


def _view(
    frame_id: str,
    indices: list[int],
    depth: float,
) -> evaluator.ProjectedView:
    index_array = np.asarray(indices, dtype=np.int64)
    return evaluator.ProjectedView(
        frame_id=frame_id,
        indices=index_array,
        base_weights=np.ones(index_array.size, dtype=np.float32),
        photo_weights=np.ones(index_array.size, dtype=np.float32),
        log_ranges={"raw_da3": np.full(index_array.size, np.log(depth), dtype=np.float32)},
    )


def _station(tmp_path: Path) -> StanfordPanorama:
    identity = np.eye(4, dtype=np.float64)
    return StanfordPanorama(
        key="camera_key",
        sample_id="sample",
        room="office_1",
        pose_room="office_1_1",
        camera_uuid="a" * 32,
        rgb_path=tmp_path / "rgb.png",
        depth_path=tmp_path / "depth.png",
        semantic_path=tmp_path / "semantic.png",
        pose_path=tmp_path / "pose.json",
        intrinsic=np.eye(3, dtype=np.float64),
        world_to_camera=identity,
        camera_to_area=identity,
        regular_views=(),
    )


def test_cli_is_val_only_and_exposes_no_tuning_or_model_escape_hatch() -> None:
    args = experiment.parse_args(["--tangent-manifest", "val.json"])

    assert not hasattr(args, "split")
    assert not hasattr(args, "checkpoint")
    assert not hasattr(args, "huber_log_delta")
    with pytest.raises(SystemExit):
        experiment.parse_args(["--tangent-manifest", "val.json", "--split", "test"])
    with pytest.raises(SystemExit):
        experiment.parse_args(["--tangent-manifest", "val.json", "--checkpoint", "x.pt"])


def test_raw_regular_loader_never_decodes_bim_or_regular_gt(tmp_path: Path) -> None:
    sample = tmp_path / "sample.npz"
    np.savez(
        sample,
        base_depth=np.full((2, 3), 2.0, dtype=np.float32),
        base_confidence=np.full((2, 3), 0.8, dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float64),
        camera_to_area=np.eye(4, dtype=np.float64),
        # This key would fail under allow_pickle=False if decoded.  Its presence
        # proves the loader selects only the four permitted arrays.
        bim_depth=np.asarray([object()], dtype=object),
        target_depth=np.asarray([object()], dtype=object),
    )
    record = {
        "id": "frame_1",
        "sample": str(sample),
        "region": "office_1",
        "camera_uuid": "a" * 32,
    }

    frame = experiment._read_raw_regular_frame(record)

    assert set(frame.predictions) == {"raw_da3"}
    assert frame.model_arrays == {}
    assert np.all(frame.rgb == 0)
    assert frame.predictions["raw_da3"].shape == (2, 3)


def test_three_predictions_share_strict_support_and_expand_native_coverage(
    tmp_path: Path,
) -> None:
    shape = (2, 4)
    selected = _view("regular/frame_1", [0, 1, 2, 3], 2.0)
    tangents = tuple(_view(f"tangent/{index:02d}", list(range(8)), 2.0) for index in range(14))

    frozen = experiment.freeze_method_predictions(
        selected,
        tangents,
        shape,
        experiment._fusion_args(),
    )
    gt = np.full(shape, 2.0, dtype=np.float32)
    valid = np.ones(shape, dtype=bool)
    rows, _ = experiment.evaluate_frozen_methods(
        _station(tmp_path),
        selected,
        4.0,
        frozen,
        gt,
        valid,
        regular_view_count=5,
    )

    assert [row["method"] for row in rows] == [name for name, _ in experiment.METHODS]
    assert {row["fixed_support_pixels"] for row in rows} == {4}
    assert [row["native_union_pixels"] for row in rows] == [4, 8, 8]
    assert all(row["spherical_abs_rel"] == pytest.approx(0.0) for row in rows)
    assert all(row["selected_regular_view_count"] == 1 for row in rows)


def test_fixed_support_fails_if_combined_prediction_loses_selected_pixel(
    tmp_path: Path,
) -> None:
    shape = (2, 4)
    selected = _view("regular/frame_1", [0, 1, 2, 3], 2.0)
    prediction, count = experiment._prediction_from_view(selected, shape)
    broken = prediction.copy()
    broken.flat[1] = np.nan
    frozen = (
        experiment.FrozenMethod("strict_single", 0, prediction, count),
        experiment.FrozenMethod("strict_single_plus_tangent6", 6, broken, count),
        experiment.FrozenMethod("strict_single_plus_tangent14", 14, prediction, count),
    )

    with pytest.raises(RuntimeError, match="invalid predictions occur inside native coverage"):
        experiment.evaluate_frozen_methods(
            _station(tmp_path),
            selected,
            4.0,
            frozen,
            np.full(shape, 2.0, dtype=np.float32),
            np.ones(shape, dtype=bool),
            regular_view_count=5,
        )


def test_exploratory_output_cannot_alias_frozen_formal_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen v3"):
        experiment._ensure_new_output(tmp_path / "pano_val")


def test_public_receipts_reject_private_absolute_paths_and_port_split_metadata(
    tmp_path: Path,
) -> None:
    project = tmp_path / "repo"
    annotation = project / "data" / "annotation.jsonl"
    annotation.parent.mkdir(parents=True)
    annotation.write_text("", encoding="utf-8")

    portable = experiment._portable_split_provenance(
        {"annotation_file": str(annotation), "fingerprint": "abc"},
        project,
    )

    assert portable["annotation_file"] == "data/annotation.jsonl"
    experiment._assert_no_private_absolute_paths(portable)
    with pytest.raises(RuntimeError, match="private absolute"):
        experiment._assert_no_private_absolute_paths({"path": "/home/person/data"})
