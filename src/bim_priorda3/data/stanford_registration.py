"""Auditable room-level BIMSyn-to-Stanford structural mesh registration.

This module deliberately has no image loader: its only target observation is the
released, per-face-labelled Area mesh.  A result is therefore a fixed room-level
calibration artifact, never an evaluation-frame depth alignment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from bim_priorda3.data.ifc_envelope import (
    ENVELOPE_CATEGORIES,
    IFCEnvelopeGeometry,
    load_ifc_envelope_geometry,
)
from bim_priorda3.data.stanford2d3ds import (
    STANFORD_SEMANTIC_CLASSES,
    STRUCTURAL_CLASS_IDS,
)

_STRUCTURAL_CLASSES = frozenset(STANFORD_SEMANTIC_CLASSES[index] for index in STRUCTURAL_CLASS_IDS)
_DEFAULT_DISCRIMINATIVE_CLASSES = ("door", "window", "beam", "column")
# The released OBJ uses Blender's Y-up world axes, while regular pose/depth
# use the Stanford Area frame with Z up.  This fixed proper rotation is part of
# the dataset convention, not a fitted room transform.
_OBJ_TO_AREA_ROTATION = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SemanticMaterial:
    semantic_class: str
    instance_number: str
    room: str
    area_number: int


@dataclass(frozen=True)
class StanfordStructuralMesh:
    room: str
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_class_ids: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class StanfordStructuralMeshes:
    rooms: dict[str, StanfordStructuralMesh]
    audit: dict[str, Any]


@dataclass(frozen=True)
class RegistrationParameters:
    seed: int = 20260810
    sample_points: int = 20_000
    coarse_points: int = 2_500
    yaw_starts: int = 72
    refine_candidates: int = 8
    max_iterations: int = 35
    correspondence_distances_m: tuple[float, ...] = (1.5, 0.75, 0.35, 0.18)
    trim_fraction: float = 0.85
    huber_delta_m: float = 0.12
    convergence_translation_m: float = 1e-5
    convergence_yaw_rad: float = 1e-6
    metric_threshold_m: float = 0.20
    min_fitness: float = 0.55
    max_rmse_m: float = 0.15
    min_points: int = 100
    semantic_clip_distance_m: float = 0.75
    semantic_trim_fraction: float = 0.90
    semantic_min_points_per_class: int = 24
    semantic_geometric_tolerance_m: float = 0.03
    semantic_min_improvement_m: float = 0.02
    semantic_discriminative_weight: float = 3.0
    semantic_min_yaw_separation_rad: float = math.pi / 6.0
    semantic_discriminative_classes: tuple[str, ...] = _DEFAULT_DISCRIMINATIVE_CLASSES

    def validate(self) -> None:
        if self.sample_points < self.min_points:
            raise ValueError("sample_points must be at least min_points")
        if not self.min_points >= 3:
            raise ValueError("min_points must be at least 3")
        if not 3 <= self.coarse_points <= self.sample_points:
            raise ValueError("coarse_points must be in [3, sample_points]")
        if self.yaw_starts < 4:
            raise ValueError("yaw_starts must be at least 4")
        if not 1 <= self.refine_candidates <= self.yaw_starts:
            raise ValueError("refine_candidates must be in [1, yaw_starts]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not self.correspondence_distances_m:
            raise ValueError("correspondence_distances_m cannot be empty")
        distances = np.asarray(self.correspondence_distances_m, dtype=np.float64)
        if not np.isfinite(distances).all() or np.any(distances <= 0):
            raise ValueError("correspondence distances must be finite and positive")
        if np.any(np.diff(distances) >= 0):
            raise ValueError("correspondence distances must be strictly decreasing")
        if not 0.5 <= self.trim_fraction <= 1.0:
            raise ValueError("trim_fraction must be in [0.5, 1.0]")
        if self.huber_delta_m <= 0:
            raise ValueError("huber_delta_m must be positive")
        if self.metric_threshold_m <= 0:
            raise ValueError("metric_threshold_m must be positive")
        if not 0.0 <= self.min_fitness <= 1.0:
            raise ValueError("min_fitness must be in [0, 1]")
        if self.max_rmse_m <= 0:
            raise ValueError("max_rmse_m must be positive")
        if self.semantic_clip_distance_m <= 0:
            raise ValueError("semantic_clip_distance_m must be positive")
        if not 0.5 <= self.semantic_trim_fraction <= 1.0:
            raise ValueError("semantic_trim_fraction must be in [0.5, 1.0]")
        if self.semantic_min_points_per_class < 3:
            raise ValueError("semantic_min_points_per_class must be at least 3")
        if self.semantic_geometric_tolerance_m < 0:
            raise ValueError("semantic_geometric_tolerance_m cannot be negative")
        if self.semantic_min_improvement_m < 0:
            raise ValueError("semantic_min_improvement_m cannot be negative")
        if self.semantic_discriminative_weight < 1.0:
            raise ValueError("semantic_discriminative_weight must be at least 1")
        if not 0.0 < self.semantic_min_yaw_separation_rad <= math.pi:
            raise ValueError("semantic_min_yaw_separation_rad must be in (0, pi]")
        if not self.semantic_discriminative_classes or len(
            set(self.semantic_discriminative_classes)
        ) != len(self.semantic_discriminative_classes):
            raise ValueError("semantic_discriminative_classes must be non-empty and unique")


@dataclass(frozen=True)
class RegistrationResult:
    transform: np.ndarray
    yaw_rad: float
    translation_m: np.ndarray
    accepted: bool
    status: str
    metrics: dict[str, float | int]
    quality_checks: dict[str, bool]
    candidates: list[dict[str, Any]]
    semantic_audit: dict[str, Any] = field(default_factory=dict)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_semantic_material(label: str, *, expected_area: int | None = 1) -> SemanticMaterial:
    """Parse ``class_instance_roomType_roomNum_areaNum`` from ``usemtl``.

    ``roomType`` is allowed to contain underscores.  Parsing from both ends avoids
    baking the current BIMSyn room vocabulary into the OBJ reader.
    """

    try:
        semantic_class, remainder = label.split("_", 1)
        instance_number, room_and_indices = remainder.split("_", 1)
        room_type, room_number, area_number_text = room_and_indices.rsplit("_", 2)
        area_number = int(area_number_text)
    except (ValueError, AttributeError) as error:
        raise ValueError(
            "Invalid Stanford semantic material label; expected "
            f"class_instance_roomType_roomNum_areaNum, got {label!r}"
        ) from error
    if not semantic_class or not instance_number or not room_type or not room_number.isdigit():
        raise ValueError(
            "Invalid Stanford semantic material label; expected "
            f"class_instance_roomType_roomNum_areaNum, got {label!r}"
        )
    if expected_area is not None and area_number != expected_area:
        raise ValueError(f"Expected Area_{expected_area}, but semantic material is {label!r}")
    return SemanticMaterial(
        semantic_class=semantic_class,
        instance_number=instance_number,
        room=f"{room_type}_{int(room_number)}",
        area_number=area_number,
    )


def _obj_vertex_index(token: str, vertex_count: int, *, line_number: int) -> int:
    value = token.split("/", 1)[0]
    try:
        obj_index = int(value)
    except ValueError as error:
        raise ValueError(f"Invalid OBJ face index at line {line_number}: {token!r}") from error
    if obj_index == 0:
        raise ValueError(f"OBJ index 0 is invalid at line {line_number}")
    index = obj_index - 1 if obj_index > 0 else vertex_count + obj_index
    if not 0 <= index < vertex_count:
        raise ValueError(
            f"OBJ face index {obj_index} is out of range for {vertex_count} vertices "
            f"at line {line_number}"
        )
    return index


def parse_stanford_semantic_obj(
    path: str | Path,
    *,
    expected_area: int = 1,
) -> StanfordStructuralMeshes:
    """Read structural, room-labelled faces from Stanford's ``semantic.obj``.

    The parser hashes the exact byte stream while reading it.  Only faces whose
    active ``usemtl`` class belongs to the seven structural classes are retained;
    furniture and clutter never enter registration.
    """

    obj_path = Path(path).expanduser().resolve()
    if not obj_path.is_file():
        raise FileNotFoundError(f"Stanford semantic OBJ does not exist: {obj_path}")

    vertices = array("d")
    room_faces: dict[str, array[int]] = {}
    room_classes: dict[str, array[int]] = {}
    room_class_counts: dict[str, Counter[str]] = {}
    all_material_counts: Counter[str] = Counter()
    digest = hashlib.sha256()
    active_material: SemanticMaterial | None = None
    vertex_count = 0
    face_count = 0
    retained_triangles = 0

    with obj_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8").split("#", 1)[0].strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"semantic.obj is not UTF-8 at line {line_number}") from error
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            record = fields[0]
            if record == "v":
                if len(fields) < 4:
                    raise ValueError(f"Malformed OBJ vertex at line {line_number}")
                try:
                    xyz = (float(fields[1]), float(fields[2]), float(fields[3]))
                except ValueError as error:
                    raise ValueError(f"Malformed OBJ vertex at line {line_number}") from error
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Non-finite OBJ vertex at line {line_number}")
                vertices.extend(xyz)
                vertex_count += 1
            elif record == "usemtl":
                if len(fields) != 2:
                    raise ValueError(f"Malformed usemtl statement at line {line_number}")
                active_material = parse_semantic_material(fields[1], expected_area=None)
                if active_material.area_number != expected_area and not (
                    active_material.semantic_class == "<UNK>" and active_material.area_number == 0
                ):
                    raise ValueError(
                        f"Expected Area_{expected_area}, but semantic material is "
                        f"{fields[1]!r} at line {line_number}"
                    )
                all_material_counts[active_material.semantic_class] += 1
            elif record == "f":
                face_count += 1
                if active_material is None:
                    raise ValueError(
                        f"OBJ face has no preceding usemtl label at line {line_number}"
                    )
                if len(fields) < 4:
                    raise ValueError(f"OBJ face has fewer than 3 vertices at line {line_number}")
                if active_material.semantic_class not in _STRUCTURAL_CLASSES:
                    continue
                indices = [
                    _obj_vertex_index(token, vertex_count, line_number=line_number)
                    for token in fields[1:]
                ]
                faces = room_faces.setdefault(active_material.room, array("q"))
                classes = room_classes.setdefault(active_material.room, array("B"))
                counts = room_class_counts.setdefault(active_material.room, Counter())
                class_id = STANFORD_SEMANTIC_CLASSES.index(active_material.semantic_class)
                for offset in range(1, len(indices) - 1):
                    faces.extend((indices[0], indices[offset], indices[offset + 1]))
                    classes.append(class_id)
                    counts[active_material.semantic_class] += 1
                    retained_triangles += 1

    if vertex_count == 0:
        raise RuntimeError(f"No vertices found in Stanford semantic OBJ: {obj_path}")
    if not room_faces:
        raise RuntimeError(f"No structural room faces found in Stanford semantic OBJ: {obj_path}")

    all_vertices = np.frombuffer(vertices, dtype=np.float64).reshape(-1, 3)
    rooms: dict[str, StanfordStructuralMesh] = {}
    for room in sorted(room_faces):
        global_triangles = np.frombuffer(room_faces[room], dtype=np.int64).reshape(-1, 3)
        used_vertices, inverse = np.unique(global_triangles.reshape(-1), return_inverse=True)
        compact_vertices_obj = all_vertices[used_vertices]
        compact_vertices = (compact_vertices_obj @ _OBJ_TO_AREA_ROTATION.T).astype(
            np.float32, copy=False
        )
        compact_triangles = inverse.reshape(-1, 3).astype(np.int32, copy=False)
        triangle_class_ids = np.frombuffer(room_classes[room], dtype=np.uint8).copy()
        if len(triangle_class_ids) != len(compact_triangles):
            raise RuntimeError(f"Internal semantic class/triangle mismatch for {room}")
        rooms[room] = StanfordStructuralMesh(
            room=room,
            vertices=compact_vertices,
            triangles=compact_triangles,
            triangle_class_ids=triangle_class_ids,
            audit={
                "vertices": len(compact_vertices),
                "triangles": len(compact_triangles),
                "class_triangle_counts": dict(sorted(room_class_counts[room].items())),
                "bounds_min_m": compact_vertices.min(axis=0).astype(float).tolist(),
                "bounds_max_m": compact_vertices.max(axis=0).astype(float).tolist(),
            },
        )

    return StanfordStructuralMeshes(
        rooms=rooms,
        audit={
            "path": str(obj_path),
            "sha256": digest.hexdigest(),
            "bytes": int(obj_path.stat().st_size),
            "expected_area": int(expected_area),
            "coordinate_frame": f"Stanford Area_{expected_area} world",
            "obj_coordinate_conversion": {
                "source": "released semantic.obj Blender world (Y-up)",
                "target": f"Stanford Area_{expected_area} world (Z-up)",
                "formula": "(x_area, y_area, z_area) = (x_obj, -z_obj, y_obj)",
                "rotation_area_from_obj": _OBJ_TO_AREA_ROTATION.astype(float).tolist(),
                "fitted": False,
            },
            "geometry_unit": "metre",
            "filter_policy": "stanford-structural-usemtl-v1",
            "allowed_classes": sorted(_STRUCTURAL_CLASSES),
            "vertices_in_obj": int(vertex_count),
            "faces_in_obj": int(face_count),
            "retained_triangles": int(retained_triangles),
            "rooms": len(rooms),
            "usemtl_statement_class_counts": dict(sorted(all_material_counts.items())),
        },
    )


def _sample_triangle_surface_with_face_indices(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape Nx3, got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must have shape Mx3, got {triangles.shape}")
    if count < 3:
        raise ValueError("count must be at least 3")
    if len(triangles) == 0:
        raise ValueError("cannot sample an empty triangle mesh")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ValueError("triangle indices are outside the vertex array")
    corners = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    valid = np.isfinite(areas) & (areas > 1e-10)
    if not valid.any():
        raise ValueError("triangle mesh has no finite, non-degenerate faces")
    valid_face_indices = np.flatnonzero(valid)
    corners = corners[valid_face_indices]
    probabilities = areas[valid_face_indices] / areas[valid_face_indices].sum()
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(corners), size=count, replace=True, p=probabilities)
    chosen = corners[selected]
    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    points = (
        chosen[:, 0]
        + uv[:, :1] * (chosen[:, 1] - chosen[:, 0])
        + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    )
    return points, valid_face_indices[selected]


def sample_triangle_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    points, _ = _sample_triangle_surface_with_face_indices(vertices, triangles, count, seed=seed)
    return points


def sample_labeled_triangle_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    triangle_class_ids: np.ndarray,
    count: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(triangle_class_ids)
    if labels.ndim != 1 or len(labels) != len(triangles):
        raise ValueError(
            "triangle_class_ids must have one value per triangle, got "
            f"{labels.shape} for {len(triangles)} triangles"
        )
    points, face_indices = _sample_triangle_surface_with_face_indices(
        vertices, triangles, count, seed=seed
    )
    return points, labels[face_indices].astype(np.int16, copy=False)


def _stable_room_seed(base_seed: int, room: str, stream: str) -> int:
    payload = f"{base_seed}\0{room}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _yaw_transform(yaw_rad: float, translation: np.ndarray) -> np.ndarray:
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.array(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    transform[:3, 3] = translation
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _fit_yaw_translation(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(weights.sum())
    if weight_sum <= 0 or len(source) < 3:
        raise ValueError("At least three positive-weight correspondences are required")
    source_centroid = np.sum(source * weights[:, None], axis=0) / weight_sum
    target_centroid = np.sum(target * weights[:, None], axis=0) / weight_sum
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    covariance = source_centered[:, :2].T @ (weights[:, None] * target_centered[:, :2])
    yaw = math.atan2(
        float(covariance[0, 1] - covariance[1, 0]),
        float(covariance[0, 0] + covariance[1, 1]),
    )
    rotation = _yaw_transform(yaw, np.zeros(3, dtype=np.float64))[:3, :3]
    translation = target_centroid - rotation @ source_centroid
    return _wrap_angle(yaw), translation


def _trim_mask(distances: np.ndarray, maximum: float, fraction: float) -> np.ndarray:
    finite = np.isfinite(distances) & (distances <= maximum)
    count = int(finite.sum())
    if count < 3:
        return finite
    if fraction >= 1.0:
        return finite
    threshold = float(np.quantile(distances[finite], fraction))
    return finite & (distances <= threshold)


def _huber_weights(distances: np.ndarray, delta: float) -> np.ndarray:
    result = np.ones(len(distances), dtype=np.float64)
    large = distances > delta
    result[large] = delta / np.maximum(distances[large], np.finfo(np.float64).eps)
    return result


def _symmetric_pairs(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    *,
    target_tree: cKDTree,
    maximum_distance: float,
    trim_fraction: float,
    huber_delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = _transform_points(source, transform)
    forward_distances, forward_indices = target_tree.query(transformed, workers=1)
    source_tree = cKDTree(transformed)
    reverse_distances, reverse_indices = source_tree.query(target, workers=1)
    forward_mask = _trim_mask(forward_distances, maximum_distance, trim_fraction)
    reverse_mask = _trim_mask(reverse_distances, maximum_distance, trim_fraction)
    if int(forward_mask.sum()) < 3 or int(reverse_mask.sum()) < 3:
        raise RuntimeError(
            "Insufficient symmetric correspondences within "
            f"{maximum_distance:.3f} m: forward={int(forward_mask.sum())}, "
            f"reverse={int(reverse_mask.sum())}"
        )

    source_forward = source[forward_mask]
    target_forward = target[forward_indices[forward_mask]]
    source_reverse = source[reverse_indices[reverse_mask]]
    target_reverse = target[reverse_mask]
    forward_weights = _huber_weights(forward_distances[forward_mask], huber_delta)
    reverse_weights = _huber_weights(reverse_distances[reverse_mask], huber_delta)
    # Equal directional mass makes the objective genuinely symmetric even when
    # the two meshes have different sampling densities or surface areas.
    forward_weights *= 0.5 / forward_weights.sum()
    reverse_weights *= 0.5 / reverse_weights.sum()
    return (
        np.concatenate((source_forward, source_reverse)),
        np.concatenate((target_forward, target_reverse)),
        np.concatenate((forward_weights, reverse_weights)),
    )


def _symmetric_metrics(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    *,
    threshold: float,
    trim_fraction: float,
) -> dict[str, float | int]:
    transformed = _transform_points(source, transform)
    forward, _ = cKDTree(target).query(transformed, workers=1)
    reverse, _ = cKDTree(transformed).query(target, workers=1)
    forward_inliers = forward <= threshold
    reverse_inliers = reverse <= threshold
    both = np.concatenate((forward, reverse))
    inlier_distances = np.concatenate((forward[forward_inliers], reverse[reverse_inliers]))
    trim_count = max(1, math.ceil(len(both) * trim_fraction))
    trimmed = np.partition(both, trim_count - 1)[:trim_count]
    rmse = (
        float(np.sqrt(np.mean(np.square(inlier_distances))))
        if len(inlier_distances)
        else float(np.sqrt(np.mean(np.square(both))))
    )
    return {
        "source_points": len(source),
        "target_points": len(target),
        "metric_threshold_m": float(threshold),
        "forward_fitness": float(forward_inliers.mean()),
        "reverse_fitness": float(reverse_inliers.mean()),
        "fitness": float(0.5 * (forward_inliers.mean() + reverse_inliers.mean())),
        "inlier_correspondences": len(inlier_distances),
        "rmse_m": rmse,
        "symmetric_mean_m": float(both.mean()),
        "symmetric_rmse_m": float(np.sqrt(np.mean(np.square(both)))),
        "symmetric_trimmed_rmse_m": float(np.sqrt(np.mean(np.square(trimmed)))),
        "symmetric_median_m": float(np.median(both)),
        "symmetric_p90_m": float(np.quantile(both, 0.90)),
        "symmetric_p95_m": float(np.quantile(both, 0.95)),
    }


def _initial_transform(source: np.ndarray, target: np.ndarray, yaw: float) -> np.ndarray:
    rotation = _yaw_transform(yaw, np.zeros(3))[:3, :3]
    rotated = source @ rotation.T
    # Extent centres are insensitive to non-uniform surface tessellation.  In Z,
    # the floor minima provide a stable datum even if one mesh omits its ceiling.
    source_bounds = np.quantile(rotated[:, :2], (0.02, 0.98), axis=0)
    target_bounds = np.quantile(target[:, :2], (0.02, 0.98), axis=0)
    source_xy_center = 0.5 * (source_bounds[0] + source_bounds[1])
    target_xy_center = 0.5 * (target_bounds[0] + target_bounds[1])
    translation = np.array(
        (
            target_xy_center[0] - source_xy_center[0],
            target_xy_center[1] - source_xy_center[1],
            float(np.quantile(target[:, 2], 0.01) - np.quantile(rotated[:, 2], 0.01)),
        ),
        dtype=np.float64,
    )
    return _yaw_transform(yaw, translation)


def _clipped_symmetric_rmse(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    clip_distance: float,
    target_tree: cKDTree,
) -> float:
    transformed = _transform_points(source, transform)
    forward, _ = target_tree.query(transformed, workers=1)
    reverse, _ = cKDTree(transformed).query(target, workers=1)
    distances = np.minimum(np.concatenate((forward, reverse)), clip_distance)
    return float(np.sqrt(np.mean(np.square(distances))))


def _constrained_icp(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    parameters: RegistrationParameters,
) -> tuple[np.ndarray, int]:
    transform = initial.copy()
    iterations = 0
    target_tree = cKDTree(target)
    for maximum_distance in parameters.correspondence_distances_m:
        for _ in range(parameters.max_iterations):
            matched_source, matched_target, weights = _symmetric_pairs(
                source,
                target,
                transform,
                target_tree=target_tree,
                maximum_distance=maximum_distance,
                trim_fraction=parameters.trim_fraction,
                huber_delta=parameters.huber_delta_m,
            )
            yaw, translation = _fit_yaw_translation(matched_source, matched_target, weights)
            updated = _yaw_transform(yaw, translation)
            yaw_delta = abs(
                _wrap_angle(
                    math.atan2(updated[1, 0], updated[0, 0])
                    - math.atan2(transform[1, 0], transform[0, 0])
                )
            )
            translation_delta = float(np.linalg.norm(updated[:3, 3] - transform[:3, 3]))
            transform = updated
            iterations += 1
            if (
                yaw_delta <= parameters.convergence_yaw_rad
                and translation_delta <= parameters.convergence_translation_m
            ):
                break
    return transform, iterations


def _validate_point_class_ids(
    values: np.ndarray | None,
    *,
    name: str,
    point_count: int,
    class_count: int,
) -> np.ndarray | None:
    if values is None:
        return None
    labels = np.asarray(values)
    if labels.ndim != 1 or len(labels) != point_count:
        raise ValueError(
            f"{name} must have one class id per point, got {labels.shape} for {point_count} points"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"{name} must contain integer class ids")
    labels = labels.astype(np.int16, copy=False)
    if np.any(labels < -1) or np.any(labels >= class_count):
        raise ValueError(f"{name} contains a class id outside [-1, {class_count - 1}]")
    return labels


def _trimmed_clipped_rmse(
    distances: np.ndarray,
    *,
    clip_distance: float,
    trim_fraction: float,
) -> float:
    clipped = np.minimum(np.asarray(distances, dtype=np.float64), clip_distance)
    keep = max(1, math.ceil(len(clipped) * trim_fraction))
    trimmed = np.partition(clipped, keep - 1)[:keep]
    return float(np.sqrt(np.mean(np.square(trimmed))))


def _class_matched_candidate_metrics(
    source: np.ndarray,
    target: np.ndarray,
    source_class_ids: np.ndarray,
    target_class_ids: np.ndarray,
    transform: np.ndarray,
    *,
    usable_class_ids: Sequence[int],
    class_names: Sequence[str],
    discriminative_classes: frozenset[str],
    parameters: RegistrationParameters,
) -> dict[str, Any]:
    transformed = _transform_points(source, transform)
    per_class: dict[str, dict[str, float | int | bool]] = {}
    weighted_scores = 0.0
    weight_sum = 0.0
    discriminative_scores: list[float] = []
    for class_id in usable_class_ids:
        class_name = class_names[class_id]
        source_subset = transformed[source_class_ids == class_id]
        target_subset = target[target_class_ids == class_id]
        forward, _ = cKDTree(target_subset).query(source_subset, workers=1)
        reverse, _ = cKDTree(source_subset).query(target_subset, workers=1)
        both = np.concatenate((forward, reverse))
        score = _trimmed_clipped_rmse(
            both,
            clip_distance=parameters.semantic_clip_distance_m,
            trim_fraction=parameters.semantic_trim_fraction,
        )
        is_discriminative = class_name in discriminative_classes
        weight = parameters.semantic_discriminative_weight if is_discriminative else 1.0
        weighted_scores += weight * score
        weight_sum += weight
        if is_discriminative:
            discriminative_scores.append(score)
        per_class[class_name] = {
            "source_points": len(source_subset),
            "target_points": len(target_subset),
            "discriminative": is_discriminative,
            "symmetric_score_m": score,
            "symmetric_median_m": float(np.median(both)),
            "symmetric_p90_m": float(np.quantile(both, 0.90)),
            "fitness": float(np.mean(both <= parameters.metric_threshold_m)),
        }
    return {
        "per_class": per_class,
        "weighted_common_score_m": weighted_scores / weight_sum,
        "discriminative_score_m": (
            float(np.mean(discriminative_scores)) if discriminative_scores else None
        ),
    }


def _geometry_candidate_passes(
    candidate: tuple[float, int, np.ndarray, dict[str, float | int]],
    parameters: RegistrationParameters,
) -> bool:
    metrics = candidate[3]
    return (
        float(metrics["fitness"]) >= parameters.min_fitness
        and float(metrics["rmse_m"]) <= parameters.max_rmse_m
    )


def _rerank_with_class_matches(
    refined: list[tuple[float, int, np.ndarray, dict[str, float | int]]],
    candidate_audit: list[dict[str, Any]],
    source: np.ndarray,
    target: np.ndarray,
    source_class_ids: np.ndarray | None,
    target_class_ids: np.ndarray | None,
    class_names: Sequence[str],
    parameters: RegistrationParameters,
) -> tuple[
    tuple[float, int, np.ndarray, dict[str, float | int]],
    dict[str, Any],
]:
    ranked = sorted(refined, key=lambda item: item[0])
    geometry_best = ranked[0]
    geometry_best_index = geometry_best[1]
    candidate_audit[geometry_best_index]["geometry_best"] = True
    semantic_audit: dict[str, Any] = {
        "enabled": source_class_ids is not None and target_class_ids is not None,
        "geometry_best_candidate_index": geometry_best_index,
        "geometry_best_objective_m": geometry_best[0],
        "selected_candidate_index": geometry_best_index,
        "selected_geometry_objective_m": geometry_best[0],
        "selected_geometry_objective_penalty_m": 0.0,
        "changed_geometry_best_candidate": False,
        "common_classes": [],
        "usable_common_classes": [],
        "discriminative_common_classes": [],
        "geometry_qualified_candidate_indices": [
            item[1] for item in ranked if _geometry_candidate_passes(item, parameters)
        ],
        "geometry_comparable_candidate_indices": [],
        "fallback_reason": None,
    }
    if source_class_ids is None or target_class_ids is None:
        semantic_audit["fallback_reason"] = "class_labels_not_provided"
        return geometry_best, semantic_audit

    discriminative = frozenset(parameters.semantic_discriminative_classes)
    class_counts: dict[str, dict[str, int]] = {}
    common_ids: list[int] = []
    usable_ids: list[int] = []
    discriminative_ids: list[int] = []
    for class_id, class_name in enumerate(class_names):
        source_count = int(np.sum(source_class_ids == class_id))
        target_count = int(np.sum(target_class_ids == class_id))
        class_counts[class_name] = {
            "source_points": source_count,
            "target_points": target_count,
        }
        if source_count and target_count:
            common_ids.append(class_id)
        if (
            source_count >= parameters.semantic_min_points_per_class
            and target_count >= parameters.semantic_min_points_per_class
        ):
            usable_ids.append(class_id)
            if class_name in discriminative:
                discriminative_ids.append(class_id)
    semantic_audit.update(
        {
            "class_sample_counts": class_counts,
            "common_classes": [class_names[index] for index in common_ids],
            "usable_common_classes": [class_names[index] for index in usable_ids],
            "discriminative_common_classes": [class_names[index] for index in discriminative_ids],
            "semantic_clip_distance_m": parameters.semantic_clip_distance_m,
            "semantic_trim_fraction": parameters.semantic_trim_fraction,
            "semantic_min_points_per_class": parameters.semantic_min_points_per_class,
            "semantic_geometric_tolerance_m": parameters.semantic_geometric_tolerance_m,
            "semantic_min_improvement_m": parameters.semantic_min_improvement_m,
            "semantic_discriminative_weight": parameters.semantic_discriminative_weight,
        }
    )
    if not usable_ids:
        semantic_audit["fallback_reason"] = "no_common_class_with_enough_points"
        return geometry_best, semantic_audit

    for _, index, transform, _ in refined:
        class_metrics = _class_matched_candidate_metrics(
            source,
            target,
            source_class_ids,
            target_class_ids,
            transform,
            usable_class_ids=usable_ids,
            class_names=class_names,
            discriminative_classes=discriminative,
            parameters=parameters,
        )
        candidate_audit[index]["class_matched"] = class_metrics

    if not discriminative_ids:
        semantic_audit["fallback_reason"] = "no_common_discriminative_class_with_enough_points"
        return geometry_best, semantic_audit
    if not _geometry_candidate_passes(geometry_best, parameters):
        semantic_audit["fallback_reason"] = "geometry_best_failed_quality_gate"
        return geometry_best, semantic_audit

    geometry_best_yaw = math.atan2(geometry_best[2][1, 0], geometry_best[2][0, 0])
    comparable = [
        item
        for item in ranked
        if _geometry_candidate_passes(item, parameters)
        and item[0] <= geometry_best[0] + parameters.semantic_geometric_tolerance_m
    ]
    semantic_audit["geometry_comparable_candidate_indices"] = [item[1] for item in comparable]
    distinct = [
        item
        for item in comparable
        if item[1] == geometry_best_index
        or abs(_wrap_angle(math.atan2(item[2][1, 0], item[2][0, 0]) - geometry_best_yaw))
        >= parameters.semantic_min_yaw_separation_rad
    ]
    if len(distinct) < 2:
        semantic_audit["fallback_reason"] = "no_distinct_geometry_comparable_candidate"
        return geometry_best, semantic_audit

    semantic_best = min(
        distinct,
        key=lambda item: float(
            candidate_audit[item[1]]["class_matched"]["weighted_common_score_m"]
        ),
    )
    geometry_semantic_score = float(
        candidate_audit[geometry_best_index]["class_matched"]["weighted_common_score_m"]
    )
    selected_semantic_score = float(
        candidate_audit[semantic_best[1]]["class_matched"]["weighted_common_score_m"]
    )
    improvement = geometry_semantic_score - selected_semantic_score
    geometry_discriminative_score = float(
        candidate_audit[geometry_best_index]["class_matched"]["discriminative_score_m"]
    )
    selected_discriminative_score = float(
        candidate_audit[semantic_best[1]]["class_matched"]["discriminative_score_m"]
    )
    discriminative_improvement = geometry_discriminative_score - selected_discriminative_score
    semantic_audit.update(
        {
            "geometry_best_semantic_score_m": geometry_semantic_score,
            "semantic_best_candidate_index": semantic_best[1],
            "semantic_best_score_m": selected_semantic_score,
            "semantic_improvement_m": improvement,
            "geometry_best_discriminative_score_m": geometry_discriminative_score,
            "semantic_best_discriminative_score_m": selected_discriminative_score,
            "discriminative_improvement_m": discriminative_improvement,
        }
    )
    if semantic_best[1] == geometry_best_index:
        semantic_audit["fallback_reason"] = "geometry_best_already_class_best"
        return geometry_best, semantic_audit
    if improvement < parameters.semantic_min_improvement_m:
        semantic_audit["fallback_reason"] = "semantic_improvement_below_threshold"
        return geometry_best, semantic_audit
    if discriminative_improvement < parameters.semantic_min_improvement_m:
        semantic_audit["fallback_reason"] = "discriminative_improvement_below_threshold"
        return geometry_best, semantic_audit

    semantic_audit.update(
        {
            "selected_candidate_index": semantic_best[1],
            "selected_geometry_objective_m": semantic_best[0],
            "selected_geometry_objective_penalty_m": semantic_best[0] - geometry_best[0],
            "changed_geometry_best_candidate": True,
            "fallback_reason": None,
        }
    )
    return semantic_best, semantic_audit


def register_yaw_translation(
    source_points: np.ndarray,
    target_points: np.ndarray,
    parameters: RegistrationParameters | None = None,
    *,
    source_class_ids: np.ndarray | None = None,
    target_class_ids: np.ndarray | None = None,
    class_names: Sequence[str] = ENVELOPE_CATEGORIES,
) -> RegistrationResult:
    """Robustly estimate one fixed BIM-to-Area transform with 4 constrained DoF."""

    parameters = parameters or RegistrationParameters()
    parameters.validate()
    class_names = tuple(str(name) for name in class_names)
    if not class_names or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be non-empty and unique")
    if (source_class_ids is None) != (target_class_ids is None):
        raise ValueError("source_class_ids and target_class_ids must be provided together")
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    for name, points in (("source", source), ("target", target)):
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{name}_points must have shape Nx3, got {points.shape}")
        if len(points) < parameters.min_points:
            raise ValueError(
                f"{name}_points has {len(points)} points, fewer than min_points="
                f"{parameters.min_points}"
            )
        if len(points) < parameters.coarse_points:
            raise ValueError(
                f"{name}_points has {len(points)} points, fewer than coarse_points="
                f"{parameters.coarse_points}"
            )
        if not np.isfinite(points).all():
            raise ValueError(f"{name}_points contains non-finite coordinates")
    source_labels = _validate_point_class_ids(
        source_class_ids,
        name="source_class_ids",
        point_count=len(source),
        class_count=len(class_names),
    )
    target_labels = _validate_point_class_ids(
        target_class_ids,
        name="target_class_ids",
        point_count=len(target),
        class_count=len(class_names),
    )

    coarse_source = source[
        np.linspace(0, len(source) - 1, parameters.coarse_points, dtype=np.int64)
    ]
    coarse_target = target[
        np.linspace(0, len(target) - 1, parameters.coarse_points, dtype=np.int64)
    ]
    coarse_clip = float(parameters.correspondence_distances_m[0])
    coarse_target_tree = cKDTree(coarse_target)
    coarse_candidates: list[tuple[float, int, np.ndarray]] = []
    candidate_audit: list[dict[str, Any]] = []
    for index in range(parameters.yaw_starts):
        yaw = -math.pi + (2.0 * math.pi * index / parameters.yaw_starts)
        initial = _initial_transform(coarse_source, coarse_target, yaw)
        score = _clipped_symmetric_rmse(
            coarse_source,
            coarse_target,
            initial,
            coarse_clip,
            coarse_target_tree,
        )
        coarse_candidates.append((score, index, initial))
        candidate_audit.append(
            {
                "index": index,
                "initial_yaw_rad": float(yaw),
                "coarse_clipped_symmetric_rmse_m": score,
                "refined": False,
            }
        )

    refined: list[tuple[float, int, np.ndarray, dict[str, float | int]]] = []
    errors: list[str] = []
    for _, index, initial in sorted(coarse_candidates, key=lambda item: item[0])[
        : parameters.refine_candidates
    ]:
        try:
            transform, iterations = _constrained_icp(source, target, initial, parameters)
            metrics = _symmetric_metrics(
                source,
                target,
                transform,
                threshold=parameters.metric_threshold_m,
                trim_fraction=parameters.trim_fraction,
            )
            objective = float(metrics["symmetric_trimmed_rmse_m"]) + (
                parameters.metric_threshold_m * (1.0 - float(metrics["fitness"]))
            )
            metrics["selection_objective_m"] = objective
            refined.append((objective, index, transform, metrics))
            candidate_audit[index].update(
                {
                    "refined": True,
                    "iterations": int(iterations),
                    "final_yaw_rad": float(math.atan2(transform[1, 0], transform[0, 0])),
                    "final_translation_m": transform[:3, 3].astype(float).tolist(),
                    "metrics": metrics,
                }
            )
        except (RuntimeError, ValueError) as error:
            message = f"candidate {index}: {type(error).__name__}: {error}"
            errors.append(message)
            candidate_audit[index].update({"refined": True, "error": message})
    if not refined:
        raise RuntimeError("All constrained ICP candidates failed: " + "; ".join(errors))

    selected, semantic_audit = _rerank_with_class_matches(
        refined,
        candidate_audit,
        source,
        target,
        source_labels,
        target_labels,
        class_names,
        parameters,
    )
    ranked_refined = sorted(refined, key=lambda item: item[0])
    _, best_index, best_transform, best_metrics = selected
    best_yaw = _wrap_angle(math.atan2(best_transform[1, 0], best_transform[0, 0]))
    yaw_alternatives = [
        item
        for item in ranked_refined
        if item[1] != best_index
        and abs(_wrap_angle(math.atan2(item[2][1, 0], item[2][0, 0]) - best_yaw))
        >= math.radians(10.0)
    ]
    if yaw_alternatives:
        alternative_objective, _, alternative_transform, _ = yaw_alternatives[0]
        best_metrics["alternative_yaw_separation_rad"] = abs(
            _wrap_angle(
                math.atan2(alternative_transform[1, 0], alternative_transform[0, 0]) - best_yaw
            )
        )
        best_metrics["alternative_objective_gap_m"] = alternative_objective - float(
            best_metrics["selection_objective_m"]
        )
    rotation = best_transform[:3, :3]
    singular_values = np.linalg.svd(rotation, compute_uv=False)
    quality_checks = {
        "minimum_fitness": float(best_metrics["fitness"]) >= parameters.min_fitness,
        "maximum_inlier_rmse": float(best_metrics["rmse_m"]) <= parameters.max_rmse_m,
        "unit_scale": bool(np.allclose(singular_values, 1.0, atol=1e-10, rtol=0.0)),
        "z_axis_up": bool(np.allclose(rotation[2], (0.0, 0.0, 1.0), atol=1e-10)),
        "proper_rotation": bool(np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)),
    }
    accepted = all(quality_checks.values())
    status = "accepted" if accepted else "rejected_quality"
    candidate_audit[best_index]["selected"] = True
    return RegistrationResult(
        transform=best_transform,
        yaw_rad=best_yaw,
        translation_m=best_transform[:3, 3].copy(),
        accepted=accepted,
        status=status,
        metrics=best_metrics,
        quality_checks=quality_checks,
        candidates=candidate_audit,
        semantic_audit=semantic_audit,
    )


def _sample_room_meshes(
    room: str,
    ifc: IFCEnvelopeGeometry,
    stanford: StanfordStructuralMesh,
    parameters: RegistrationParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source, source_class_ids = sample_labeled_triangle_surface(
        ifc.vertices,
        ifc.triangles,
        ifc.triangle_categories,
        parameters.sample_points,
        seed=_stable_room_seed(parameters.seed, room, "ifc"),
    )
    stanford_to_canonical = np.full(len(STANFORD_SEMANTIC_CLASSES), -1, dtype=np.int16)
    for class_id, class_name in enumerate(ENVELOPE_CATEGORIES):
        stanford_to_canonical[STANFORD_SEMANTIC_CLASSES.index(class_name)] = class_id
    target_triangle_classes = stanford_to_canonical[stanford.triangle_class_ids]
    if np.any(target_triangle_classes < 0):
        raise RuntimeError(f"Stanford room {room} contains an unmapped structural class")
    target, target_class_ids = sample_labeled_triangle_surface(
        stanford.vertices,
        stanford.triangles,
        target_triangle_classes,
        parameters.sample_points,
        seed=_stable_room_seed(parameters.seed, room, "stanford"),
    )
    return source, target, source_class_ids, target_class_ids


def _ifc_index(ifc_dir: str | Path) -> dict[str, Path]:
    root = Path(ifc_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIMSyn IFC directory does not exist: {root}")
    paths: dict[str, Path] = {}
    for path in sorted(root.glob("*.ifc")):
        if path.stem in paths:
            raise ValueError(f"Duplicate BIMSyn IFC room stem: {path.stem}")
        paths[path.stem] = path.resolve()
    if not paths:
        raise RuntimeError(f"No .ifc files found under {root}")
    return paths


def build_registration_audit(
    semantic_obj: str | Path,
    ifc_dir: str | Path,
    *,
    parameters: RegistrationParameters | None = None,
    rooms: Sequence[str] | None = None,
    expected_area: int = 1,
) -> dict[str, Any]:
    """Register all selected BIMSyn rooms and return a complete audit payload.

    Per-room errors are retained in ``rooms`` and ``failures`` rather than being
    swallowed.  Callers should treat any non-empty ``failures`` list as a failed
    dataset-preparation stage.
    """

    parameters = parameters or RegistrationParameters()
    parameters.validate()
    semantic_meshes = parse_stanford_semantic_obj(semantic_obj, expected_area=expected_area)
    ifc_paths = _ifc_index(ifc_dir)
    requested = (
        sorted(set(rooms))
        if rooms is not None
        else sorted(set(semantic_meshes.rooms) | set(ifc_paths))
    )
    if not requested:
        raise ValueError("No rooms were requested for registration")

    payload: dict[str, Any] = {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "structural-mesh-symmetric-constrained-yaw-icp-class-rerank-v2",
        "coordinate_frames": {
            "source": "BIMSyn IFC room-local, metres, Z-up",
            "target": f"Stanford Area_{expected_area} world, metres, Z-up",
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
            "uses_semantic_face_classes_for_candidate_reranking": True,
            "semantic_classes_affect_icp_fit": False,
        },
        "sources": {
            "semantic_obj": semantic_meshes.audit,
            "ifc_directory": str(Path(ifc_dir).expanduser().resolve()),
        },
        "parameters": {
            **asdict(parameters),
            "correspondence_distances_m": list(parameters.correspondence_distances_m),
            "semantic_discriminative_classes": list(parameters.semantic_discriminative_classes),
        },
        "requested_rooms": requested,
        "rooms": {},
        "failures": [],
    }
    for room in requested:
        room_entry: dict[str, Any] = {
            "accepted": False,
            "status": "pending",
            "T_area_from_bim": None,
        }
        payload["rooms"][room] = room_entry
        if room not in semantic_meshes.rooms:
            message = f"Stanford semantic.obj has no structural mesh for room {room!r}"
            room_entry.update({"status": "missing_stanford_room", "error": message})
            payload["failures"].append({"room": room, "error": message})
            continue
        if room not in ifc_paths:
            message = f"BIMSyn IFC directory has no {room}.ifc"
            room_entry.update({"status": "missing_ifc", "error": message})
            payload["failures"].append({"room": room, "error": message})
            continue

        ifc_path = ifc_paths[room]
        room_entry.update(
            {
                "source_ifc": str(ifc_path),
                "source_ifc_sha256": sha256_file(ifc_path),
                "target_structural_mesh_audit": semantic_meshes.rooms[room].audit,
            }
        )
        try:
            ifc = load_ifc_envelope_geometry(ifc_path, strict=True)
            room_entry["source_ifc_envelope_audit"] = ifc.audit
            source, target, source_class_ids, target_class_ids = _sample_room_meshes(
                room, ifc, semantic_meshes.rooms[room], parameters
            )
            result = register_yaw_translation(
                source,
                target,
                parameters,
                source_class_ids=source_class_ids,
                target_class_ids=target_class_ids,
                class_names=ENVELOPE_CATEGORIES,
            )
            room_entry.update(
                {
                    "accepted": result.accepted,
                    "status": result.status,
                    "T_area_from_bim": result.transform.astype(float).tolist(),
                    "yaw_rad": float(result.yaw_rad),
                    "translation_m": result.translation_m.astype(float).tolist(),
                    "metrics": result.metrics,
                    "quality_checks": result.quality_checks,
                    "semantic_reranking": result.semantic_audit,
                    "candidates": result.candidates,
                }
            )
            if not result.accepted:
                message = (
                    f"Room {room} registration failed quality thresholds: "
                    f"fitness={result.metrics['fitness']:.4f}, "
                    f"rmse={result.metrics['rmse_m']:.4f} m"
                )
                room_entry["error"] = message
                payload["failures"].append({"room": room, "error": message})
        except Exception as error:  # noqa: BLE001 - every per-room failure must be audited
            message = f"{type(error).__name__}: {error}"
            room_entry.update({"accepted": False, "status": "error", "error": message})
            payload["failures"].append({"room": room, "error": message})

    payload["summary"] = {
        "requested_rooms": len(requested),
        "accepted_rooms": sum(bool(entry["accepted"]) for entry in payload["rooms"].values()),
        "failed_rooms": len(payload["failures"]),
    }
    return payload


def write_registration_audit(
    payload: Mapping[str, Any],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Registration audit already exists: {output}")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return output


def accepted_transforms(payload: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate and load accepted ``T_area_from_bim`` matrices for preparation."""

    rooms = payload.get("rooms")
    if not isinstance(rooms, Mapping):
        raise TypeError("Registration audit has no rooms mapping")
    transforms: dict[str, np.ndarray] = {}
    for room, raw_entry in rooms.items():
        if not isinstance(raw_entry, Mapping):
            raise TypeError(f"Registration room entry is not a mapping: {room}")
        if not raw_entry.get("accepted", False):
            continue
        transform = np.asarray(raw_entry.get("T_area_from_bim"), dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Invalid T_area_from_bim for accepted room {room}")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
            raise ValueError(f"Non-rigid homogeneous row for accepted room {room}")
        rotation = transform[:3, :3]
        if not np.allclose(rotation[2], (0.0, 0.0, 1.0), atol=1e-10):
            raise ValueError(f"Accepted room {room} violates the Z-up yaw-only constraint")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
            raise ValueError(f"Accepted room {room} violates unit-scale rigid rotation")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
            raise ValueError(f"Accepted room {room} is not a proper yaw rotation")
        transforms[str(room)] = transform
    return transforms
