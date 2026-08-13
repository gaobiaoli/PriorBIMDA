from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pipelines.run_slabim_experiments import (
    annotation_region_strides,
    regions_requiring_pose_rosbags,
)


def _write_annotation(path: Path, records: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"schema_version": 1, "id": sample_id, "split": split}) + "\n"
            for sample_id, split in records
        ),
        encoding="utf-8",
    )


def test_annotation_region_strides_uses_full_population(tmp_path: Path) -> None:
    annotation = tmp_path / "split.jsonl"
    _write_annotation(
        annotation,
        [
            ("RegionA/000000", "train"),
            ("RegionA/000002", "excluded"),
            ("RegionA/000004", "test"),
            ("RegionB/000000", "train"),
            ("RegionB/000001", "val"),
            ("RegionB/000002", "excluded"),
        ],
    )

    assert annotation_region_strides(annotation) == {"RegionA": 2, "RegionB": 1}


def test_annotation_region_strides_rejects_non_exhaustive_population(tmp_path: Path) -> None:
    annotation = tmp_path / "split.jsonl"
    _write_annotation(
        annotation,
        [
            ("RegionA/000000", "train"),
            ("RegionA/000002", "val"),
            ("RegionA/000006", "test"),
        ],
    )

    with pytest.raises(ValueError, match="constant-stride"):
        annotation_region_strides(annotation)


def test_force_pose_refresh_restores_rosbags_even_when_pose_exists(tmp_path: Path) -> None:
    pose = tmp_path / "sensor_data/5F_Region2/points/lidar_pose_local_to_bim_from_rosbag.txt"
    pose.parent.mkdir(parents=True)
    pose.write_text("existing pose\n", encoding="utf-8")

    assert (
        regions_requiring_pose_rosbags(
            tmp_path,
            ["5F_Region2"],
            force_pose_refresh=False,
        )
        == []
    )
    assert regions_requiring_pose_rosbags(
        tmp_path,
        ["5F_Region2"],
        force_pose_refresh=True,
    ) == ["5F_Region2"]
