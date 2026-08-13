from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data.stanford_registration import (
    RegistrationParameters,
    accepted_transforms,
    parse_semantic_material,
    parse_stanford_semantic_obj,
    register_yaw_translation,
    write_registration_audit,
)


def _write_obj(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_parse_semantic_material_preserves_room_type() -> None:
    material = parse_semantic_material("wall_17_open_office_12_1")
    assert material.semantic_class == "wall"
    assert material.instance_number == "17"
    assert material.room == "open_office_12"
    assert material.area_number == 1

    with pytest.raises(ValueError, match="Expected Area_1"):
        parse_semantic_material("wall_1_office_2_3")
    with pytest.raises(ValueError, match="Invalid Stanford"):
        parse_semantic_material("wall-office")


def test_parse_semantic_obj_keeps_only_structural_faces(tmp_path: Path) -> None:
    obj_path = tmp_path / "semantic.obj"
    contents = """\
# small audited Area_1 mesh
v 0 0 0
v 2 0 0
v 2 2 0
v 0 2 0
v 0 0 2
v 2 0 2
usemtl floor_1_office_1_1
f 1 2 3 4
usemtl wall_2_office_1_1
f 1 2 6 5
usemtl chair_1_office_1_1
f -4 -3 -2
usemtl <UNK>_0_<UNK>_0_0
f 1 3 5
"""
    _write_obj(obj_path, contents)

    parsed = parse_stanford_semantic_obj(obj_path)

    assert set(parsed.rooms) == {"office_1"}
    room = parsed.rooms["office_1"]
    assert room.triangles.shape == (4, 3)
    assert set(room.triangle_class_ids.tolist()) == {1, 2}
    assert room.audit["class_triangle_counts"] == {"floor": 2, "wall": 2}
    assert parsed.audit["retained_triangles"] == 4
    assert parsed.audit["faces_in_obj"] == 4
    assert parsed.audit["usemtl_statement_class_counts"]["<UNK>"] == 1
    assert parsed.audit["sha256"] == hashlib.sha256(contents.encode()).hexdigest()
    # Released OBJ uses Blender Y-up axes; registration must operate in the
    # pose/depth Area frame: (x_area, y_area, z_area)=(x_obj,-z_obj,y_obj).
    assert room.audit["bounds_min_m"] == [0.0, -2.0, 0.0]
    assert room.audit["bounds_max_m"] == [2.0, 0.0, 2.0]
    assert parsed.audit["obj_coordinate_conversion"]["fitted"] is False


def test_semantic_obj_face_without_material_fails_loudly(tmp_path: Path) -> None:
    obj_path = tmp_path / "semantic.obj"
    _write_obj(obj_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    with pytest.raises(ValueError, match="no preceding usemtl"):
        parse_stanford_semantic_obj(obj_path)


def test_registration_recovers_yaw_translation_and_unit_scale() -> None:
    rng = np.random.default_rng(42)
    # An asymmetric structural layout avoids the physically genuine 180-degree
    # ambiguity of a plain rectangular room.
    source = np.concatenate(
        (
            np.column_stack((rng.uniform(0, 5, 500), np.zeros(500), rng.uniform(0, 3, 500))),
            np.column_stack((np.zeros(350), rng.uniform(0, 3, 350), rng.uniform(0, 3, 350))),
            np.column_stack((rng.uniform(0, 2, 250), np.full(250, 3), rng.uniform(0, 1.8, 250))),
        )
    )
    expected_yaw = 0.63
    expected_translation = np.array((7.0, -2.0, 1.2))
    cosine = math.cos(expected_yaw)
    sine = math.sin(expected_yaw)
    expected_rotation = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    target = source @ expected_rotation.T + expected_translation
    target += rng.normal(0.0, 0.003, target.shape)
    parameters = RegistrationParameters(
        sample_points=len(source),
        coarse_points=300,
        yaw_starts=24,
        refine_candidates=4,
        max_iterations=15,
        correspondence_distances_m=(0.8, 0.25, 0.06),
        trim_fraction=0.9,
        huber_delta_m=0.03,
        metric_threshold_m=0.025,
        min_fitness=0.95,
        max_rmse_m=0.015,
        min_points=30,
    )

    result = register_yaw_translation(source, target, parameters)

    assert result.accepted
    assert abs((result.yaw_rad - expected_yaw + math.pi) % (2 * math.pi) - math.pi) < 0.01
    assert np.allclose(result.translation_m, expected_translation, atol=0.01)
    assert np.allclose(result.transform[:3, :3].T @ result.transform[:3, :3], np.eye(3))
    assert np.allclose(result.transform[2], (0.0, 0.0, 1.0, expected_translation[2]), atol=0.01)
    assert result.metrics["forward_fitness"] > 0.99
    assert result.metrics["reverse_fitness"] > 0.99
    assert result.metrics["symmetric_p95_m"] < 0.015


def test_class_matched_reranking_resolves_180_degree_geometry_ambiguity() -> None:
    rng = np.random.default_rng(7)
    half = np.column_stack(
        (
            rng.uniform(0.2, 3.0, 500),
            rng.uniform(-2.0, 2.0, 500),
            rng.uniform(0.0, 3.0, 500),
        )
    )
    # Geometry is exactly invariant to a 180-degree yaw.  The door class is on
    # opposite ends in source and target, so class identity selects yaw=pi.
    source = np.concatenate((half, half * np.array((-1.0, -1.0, 1.0))))
    target = source.copy()
    source_classes = np.zeros(len(source), dtype=np.int16)
    target_classes = np.zeros(len(target), dtype=np.int16)
    source_classes[source[:, 0] > 2.2] = 3  # canonical door id
    target_classes[target[:, 0] < -2.2] = 3
    parameters = RegistrationParameters(
        sample_points=len(source),
        coarse_points=300,
        yaw_starts=24,
        refine_candidates=6,
        max_iterations=12,
        correspondence_distances_m=(1.0, 0.3, 0.05),
        trim_fraction=0.95,
        huber_delta_m=0.03,
        metric_threshold_m=0.02,
        min_fitness=0.95,
        max_rmse_m=0.01,
        min_points=30,
        semantic_clip_distance_m=0.75,
        semantic_trim_fraction=0.95,
        semantic_min_points_per_class=20,
        semantic_geometric_tolerance_m=0.01,
        semantic_min_improvement_m=0.05,
    )

    geometric = register_yaw_translation(source, target, parameters)
    reranked = register_yaw_translation(
        source,
        target,
        parameters,
        source_class_ids=source_classes,
        target_class_ids=target_classes,
    )

    assert geometric.yaw_rad == pytest.approx(0.0, abs=1e-8)
    assert abs(abs(reranked.yaw_rad) - math.pi) < 1e-8
    assert reranked.accepted
    assert reranked.semantic_audit["discriminative_common_classes"] == ["door"]
    assert reranked.semantic_audit["changed_geometry_best_candidate"] is True
    assert reranked.semantic_audit["semantic_improvement_m"] > 0.5
    assert reranked.quality_checks == geometric.quality_checks

    wall_only = register_yaw_translation(
        source,
        target,
        parameters,
        source_class_ids=np.zeros(len(source), dtype=np.int16),
        target_class_ids=np.zeros(len(target), dtype=np.int16),
    )
    assert wall_only.yaw_rad == pytest.approx(geometric.yaw_rad, abs=1e-8)
    assert wall_only.semantic_audit["changed_geometry_best_candidate"] is False
    assert (
        wall_only.semantic_audit["fallback_reason"]
        == "no_common_discriminative_class_with_enough_points"
    )


def test_registration_audit_roundtrip_and_transform_validation(tmp_path: Path) -> None:
    transform = np.eye(4)
    transform[:3, 3] = (1.0, 2.0, 3.0)
    payload = {
        "rooms": {
            "office_1": {
                "accepted": True,
                "T_area_from_bim": transform.tolist(),
            },
            "office_2": {"accepted": False, "T_area_from_bim": None},
        }
    }
    output = write_registration_audit(payload, tmp_path / "alignment.json")
    loaded = json.loads(output.read_text(encoding="utf-8"))
    transforms = accepted_transforms(loaded)
    assert set(transforms) == {"office_1"}
    assert np.array_equal(transforms["office_1"], transform)
    with pytest.raises(FileExistsError):
        write_registration_audit(payload, output)

    loaded["rooms"]["office_1"]["T_area_from_bim"][0][0] = 1.1
    with pytest.raises(ValueError, match="unit-scale"):
        accepted_transforms(loaded)
