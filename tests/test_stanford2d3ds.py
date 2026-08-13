from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bim_priorda3.data.ifc_envelope import envelope_category
from bim_priorda3.data.stanford2d3ds import (
    FURNITURE_CLASS_IDS,
    UNKNOWN_SEMANTIC_ID,
    load_stanford_depth,
    load_stanford_semantics,
    pose_matrices,
    room_key_from_pose_room,
    semantic_label_lut,
    semantic_subset_masks,
)


class FakeIFCProduct:
    def __init__(self, classes: set[str], predefined_type: str | None = None) -> None:
        self.classes = classes
        self.PredefinedType = predefined_type

    def is_a(self, name: str) -> bool:
        return name in self.classes


@pytest.mark.parametrize(
    ("classes", "predefined_type", "expected"),
    [
        ({"IfcWall", "IfcBuildingElement"}, None, "wall"),
        ({"IfcSlab"}, "FLOOR", "floor"),
        ({"IfcCovering"}, "CEILING", "ceiling"),
        ({"IfcDoor"}, None, "door"),
        ({"IfcWindow"}, None, "window"),
        ({"IfcBeam"}, None, "beam"),
        ({"IfcColumn"}, None, "column"),
        ({"IfcFurnishingElement"}, None, None),
        ({"IfcBuildingElementProxy"}, None, None),
        ({"IfcFlowTerminal"}, None, None),
        ({"IfcOpeningElement"}, None, None),
    ],
)
def test_envelope_category_is_an_allowlist(
    classes: set[str], predefined_type: str | None, expected: str | None
) -> None:
    assert envelope_category(FakeIFCProduct(classes, predefined_type)) == expected


def test_pose_matrices_use_released_world_to_camera_convention() -> None:
    theta = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    location = np.asarray([2.0, -3.0, 1.6])
    translation = -rotation @ location
    payload = {
        "camera_k_matrix": [[800.0, 0.0, 540.0], [0.0, 800.0, 540.0], [0, 0, 1]],
        "camera_rt_matrix": np.column_stack((rotation, translation)).tolist(),
        "camera_location": location.tolist(),
    }
    intrinsic, area_to_camera, camera_to_area = pose_matrices(payload)
    np.testing.assert_allclose(intrinsic[0, 0], 800.0)
    np.testing.assert_allclose(area_to_camera @ camera_to_area, np.eye(4), atol=1e-10)
    np.testing.assert_allclose(camera_to_area[:3, 3], location, atol=1e-10)


def test_pose_matrices_reject_transposed_readme_shape() -> None:
    payload = {
        "camera_k_matrix": np.eye(3).tolist(),
        "camera_rt_matrix": np.zeros((4, 3)).tolist(),
        "camera_location": [0.0, 0.0, 0.0],
    }
    with pytest.raises(ValueError, match="3x4"):
        pose_matrices(payload)


def test_room_key_removes_only_area_suffix() -> None:
    assert room_key_from_pose_room("conferenceRoom_2_1") == "conferenceRoom_2"
    with pytest.raises(ValueError, match="Area_1"):
        room_key_from_pose_room("office_1_2")


def test_depth_and_semantic_decoding(tmp_path: Path) -> None:
    depth_path = tmp_path / "depth.png"
    raw_depth = np.asarray([[512, 1024], [65535, 50]], dtype=np.uint16)
    assert cv2.imwrite(str(depth_path), raw_depth)
    depth, valid = load_stanford_depth(
        depth_path,
        (2, 2),
        min_depth=0.2,
        max_depth=5.0,
    )
    np.testing.assert_allclose(depth, [[1.0, 2.0], [0.0, 0.0]])
    np.testing.assert_array_equal(valid, [[True, True], [False, False]])

    labels_path = tmp_path / "semantic_labels.json"
    labels = ["<UNK>_0_<UNK>_0_0", "chair_1_office_1_1", "wall_1_office_1_1"]
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    semantic_path = tmp_path / "semantic.png"
    # Official label indices 1 and 2 encode as RGB #000001/#000002. cv2 takes BGR.
    bgr = np.asarray([[[1, 0, 0], [2, 0, 0]], [[13, 13, 13], [0, 0, 0]]], dtype=np.uint8)
    assert cv2.imwrite(str(semantic_path), bgr)
    classes = load_stanford_semantics(
        semantic_path,
        (2, 2),
        semantic_label_lut(labels_path),
    )
    chair_id = next(iter(FURNITURE_CLASS_IDS - {7, 9, 10}))
    assert classes[0, 0] == chair_id
    assert classes[0, 1] == 2
    assert classes[1, 0] == UNKNOWN_SEMANTIC_ID
    masks = semantic_subset_masks(classes)
    assert masks["furniture_mask"][0, 0]
    assert masks["structural_mask"][0, 1]
    assert not masks["semantic_valid"][1, 0]
