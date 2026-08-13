from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data.stanford_preparation import (
    _validate_alignment_receipt,
    _validate_reusable_sample,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alignment(ifc: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "method": "structural-mesh-symmetric-constrained-yaw-icp-v1",
        "coordinate_frames": {
            "source": "BIMSyn IFC room-local, metres, Z-up",
            "target": "Stanford Area_1 world, metres, Z-up",
            "transform": "T_area_from_bim",
        },
        "constraints": {
            "degrees_of_freedom": ["yaw_rad", "tx_m", "ty_m", "tz_m"],
            "scale": 1.0,
            "z_axis_up": True,
            "roll_rad": 0.0,
            "pitch_rad": 0.0,
            "per_frame_alignment": False,
            "uses_rgb": False,
            "uses_depth_images": False,
        },
        "sources": {
            "semantic_obj": {
                "geometry_unit": "metre",
                "obj_coordinate_conversion": {
                    "fitted": False,
                    "rotation_area_from_obj": [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0],
                        [0.0, 1.0, 0.0],
                    ],
                },
            }
        },
        "parameters": {"min_fitness": 0.55, "max_rmse_m": 0.15},
        "failures": [],
        "rooms": {
            "office_1": {
                "accepted": True,
                "status": "accepted",
                "source_ifc_sha256": _sha(ifc),
                "T_area_from_bim": np.eye(4).tolist(),
                "quality_checks": {
                    "maximum_inlier_rmse": True,
                    "minimum_fitness": True,
                    "proper_rotation": True,
                    "unit_scale": True,
                    "z_axis_up": True,
                },
                "metrics": {"fitness": 0.8, "rmse_m": 0.1},
            }
        },
    }


def test_alignment_receipt_binds_method_axis_quality_and_ifc(tmp_path: Path) -> None:
    ifc = tmp_path / "office_1.ifc"
    ifc.write_bytes(b"IFC source")
    alignment = _alignment(ifc)

    transforms = _validate_alignment_receipt(
        alignment,
        ["office_1"],
        {"office_1": ifc},
    )
    np.testing.assert_array_equal(transforms["office_1"], np.eye(4))

    alignment["rooms"]["office_1"]["source_ifc_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ValueError, match="IFC SHA256"):
        _validate_alignment_receipt(alignment, ["office_1"], {"office_1": ifc})


def test_alignment_receipt_accepts_audited_class_rerank_v2(tmp_path: Path) -> None:
    ifc = tmp_path / "office_1.ifc"
    ifc.write_bytes(b"IFC source")
    alignment = _alignment(ifc)
    alignment["schema_version"] = 2
    alignment["method"] = "structural-mesh-symmetric-constrained-yaw-icp-class-rerank-v2"
    constraints = alignment["constraints"]
    assert isinstance(constraints, dict)
    constraints.update(
        semantic_classes_affect_icp_fit=False,
        uses_semantic_face_classes_for_candidate_reranking=True,
    )
    rooms = alignment["rooms"]
    assert isinstance(rooms, dict)
    room = rooms["office_1"]
    assert isinstance(room, dict)
    room["semantic_reranking"] = {"enabled": True, "selected_candidate_index": 17}

    transforms = _validate_alignment_receipt(
        alignment,
        ["office_1"],
        {"office_1": ifc},
    )
    np.testing.assert_array_equal(transforms["office_1"], np.eye(4))


def _sample_payload(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    height, width = shape
    valid = np.ones(shape, dtype=np.uint8)
    semantic_class = np.zeros(shape, dtype=np.uint8)
    zeros = np.zeros(shape, dtype=np.uint8)
    return {
        "sample_schema_version": np.asarray(2, dtype=np.uint16),
        "base_depth": np.ones(shape, dtype=np.float16),
        "base_confidence": np.ones(shape, dtype=np.float16),
        "bim_depth": np.ones(shape, dtype=np.float16),
        "bim_valid": valid,
        "bim_normals": np.ones((3, height, width), dtype=np.float16),
        "bim_edge": zeros,
        "bim_category": zeros,
        "gt_depth": np.ones(shape, dtype=np.float32),
        "gt_valid": valid,
        "gt_weight": np.ones(shape, dtype=np.float16),
        "semantic_class": semantic_class,
        "semantic_valid": valid,
        "furniture_mask": zeros,
        "structural_mask": valid,
        "non_structural_mask": zeros,
        "intrinsic": np.eye(3, dtype=np.float32),
        "camera_to_bim": np.eye(4),
        "camera_to_area": np.eye(4),
        "area_from_bim": np.eye(4),
        "bim_scene_coordinate_frame": np.asarray("Stanford Area_1 world"),
        "global_bim_fingerprint_sha256": np.asarray("1" * 64),
        "da3_cache_artifact_sha256": np.asarray("2" * 64),
        "preparation_fingerprint_sha256": np.asarray("3" * 64),
    }


def test_reuse_validation_enforces_shapes_semantics_and_zero_invalid_bim(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample.npz"
    payload = _sample_payload((3, 4))
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as cached:
        _validate_reusable_sample(cached, path, (3, 4))

    payload["bim_valid"][0, 0] = 0
    np.savez_compressed(path, **payload)
    with (
        np.load(path, allow_pickle=False) as cached,
        pytest.raises(ValueError, match="zero depth"),
    ):
        _validate_reusable_sample(cached, path, (3, 4))
