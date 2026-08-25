"""Geometry and metadata adapter for Stanford 2D-3D-S panoramas.

The released panorama depth is a radial range, whereas the regular perspective
depth maps use camera ``z`` depth.  This module keeps that distinction explicit
and uses the official 3x4 world-to-camera pose convention throughout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.data.stanford2d3ds import pose_matrices, room_key_from_pose_room

_PANO_NAME = re.compile(
    r"^(?P<key>camera_(?P<uuid>[0-9a-fA-F]{32})_(?P<room>.+)_frame_equirectangular)"
    r"_domain_*(?P<modality>rgb|depth|semantic|pose)$"
)
_REGULAR_POSE_NAME = re.compile(
    r"^(?P<key>camera_(?P<uuid>[0-9a-fA-F]{32})_(?P<room>.+)_frame_"
    r"(?P<frame>[0-9]+))_domain_*pose$"
)
_POSE_ATOL = 2e-4
_EPS = 1e-12


@dataclass(frozen=True)
class StanfordRegularView:
    """Pose metadata for one regular perspective image at a panorama station."""

    key: str
    room: str
    pose_room: str
    camera_uuid: str
    frame_number: int
    pose_path: Path
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    camera_to_area: np.ndarray


@dataclass(frozen=True)
class StanfordPanorama:
    """One Area panorama and all regular views sharing its camera centre."""

    key: str
    sample_id: str
    room: str
    pose_room: str
    camera_uuid: str
    rgb_path: Path
    depth_path: Path
    semantic_path: Path
    pose_path: Path
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    camera_to_area: np.ndarray
    regular_views: tuple[StanfordRegularView, ...]


@dataclass(frozen=True)
class RegularPanoLookup:
    """Reusable mapping from one regular raster into a co-located ERP raster."""

    erp_pixels: np.ndarray
    z_per_radial_range: np.ndarray
    pano_shape: tuple[int, int]
    regular_shape: tuple[int, int]


@dataclass(frozen=True)
class _ParsedPose:
    room: str
    pose_room: str
    camera_uuid: str
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    camera_to_area: np.ndarray
    frame_number: int | str


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse Stanford pose JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"Stanford pose must be a JSON object: {path}")
    return payload


def _parse_pose(
    path: Path,
    *,
    expected_area: int,
    expected_uuid: str,
    expected_room: str,
    panorama: bool,
) -> _ParsedPose:
    payload = _read_json_object(path)
    pose_room = str(payload.get("room", ""))
    room = room_key_from_pose_room(pose_room, expected_area=expected_area)
    if room != expected_room:
        raise ValueError(
            f"Stanford pose room {room!r} disagrees with filename room "
            f"{expected_room!r}: {path.name}"
        )

    camera_uuid = str(payload.get("camera_uuid", ""))
    point_uuid = str(payload.get("point_uuid", camera_uuid))
    if not camera_uuid:
        raise ValueError(f"Stanford pose lacks camera_uuid: {path}")
    if camera_uuid != point_uuid:
        raise ValueError(
            f"camera_uuid and point_uuid disagree in Stanford pose {path.name}: "
            f"{camera_uuid!r} != {point_uuid!r}"
        )
    if camera_uuid.lower() != expected_uuid.lower():
        raise ValueError(
            f"Stanford pose camera_uuid disagrees with filename: "
            f"{camera_uuid!r} != {expected_uuid!r} ({path.name})"
        )

    intrinsic, world_to_camera, camera_to_area = pose_matrices(payload)
    _validate_intrinsic(intrinsic)
    frame_value = payload.get("frame_num")
    if panorama:
        if frame_value != "equirectangular":
            raise ValueError(
                "Stanford panorama frame_num must be 'equirectangular', "
                f"got {frame_value!r}: {path.name}"
            )
        frame_number: int | str = "equirectangular"
    else:
        if isinstance(frame_value, bool):
            raise ValueError(f"Stanford regular frame_num must be an integer: {path.name}")
        try:
            frame_number = int(frame_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Stanford regular frame_num must be an integer: {path.name}"
            ) from error
        if str(frame_number) != str(frame_value):
            raise ValueError(f"Non-canonical Stanford regular frame_num: {frame_value!r}")

    return _ParsedPose(
        room=room,
        pose_room=pose_room,
        camera_uuid=camera_uuid.lower(),
        intrinsic=intrinsic,
        world_to_camera=world_to_camera,
        camera_to_area=camera_to_area,
        frame_number=frame_number,
    )


def _index_pano_modality(directory: Path, modality: str, pattern: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Stanford panorama modality directory is missing: {directory}")
    indexed: dict[str, Path] = {}
    for path in sorted(directory.glob(pattern)):
        match = _PANO_NAME.match(path.stem)
        if match is None or match.group("modality") != modality:
            raise ValueError(f"Unexpected Stanford panorama {modality} filename: {path.name}")
        key = str(match.group("key"))
        if key in indexed:
            raise ValueError(f"Duplicate Stanford panorama {modality} pairing key: {key}")
        indexed[key] = path.resolve()
    if not indexed:
        raise RuntimeError(f"No Stanford panorama {modality} files found in {directory}")
    return indexed


def _discover_regular_views(
    pose_directory: Path,
    *,
    expected_area: int,
) -> dict[str, list[StanfordRegularView]]:
    if not pose_directory.is_dir():
        raise FileNotFoundError(f"Stanford regular pose directory is missing: {pose_directory}")
    grouped: dict[str, list[StanfordRegularView]] = {}
    seen_keys: set[str] = set()
    for path in sorted(pose_directory.glob("*.json")):
        match = _REGULAR_POSE_NAME.match(path.stem)
        if match is None:
            raise ValueError(f"Unexpected Stanford regular pose filename: {path.name}")
        key = str(match.group("key"))
        if key in seen_keys:
            raise ValueError(f"Duplicate Stanford regular pose pairing key: {key}")
        seen_keys.add(key)
        filename_uuid = str(match.group("uuid")).lower()
        filename_room = str(match.group("room"))
        filename_frame = int(match.group("frame"))
        parsed = _parse_pose(
            path.resolve(),
            expected_area=expected_area,
            expected_uuid=filename_uuid,
            expected_room=filename_room,
            panorama=False,
        )
        if parsed.frame_number != filename_frame:
            raise ValueError(
                f"Stanford regular pose frame_num disagrees with filename: "
                f"{parsed.frame_number!r} != {filename_frame} ({path.name})"
            )
        view = StanfordRegularView(
            key=key,
            room=parsed.room,
            pose_room=parsed.pose_room,
            camera_uuid=parsed.camera_uuid,
            frame_number=filename_frame,
            pose_path=path.resolve(),
            intrinsic=parsed.intrinsic,
            world_to_camera=parsed.world_to_camera,
            camera_to_area=parsed.camera_to_area,
        )
        grouped.setdefault(parsed.camera_uuid, []).append(view)

    if not seen_keys:
        raise RuntimeError(f"No Stanford regular pose JSON files found in {pose_directory}")
    for camera_uuid, views in grouped.items():
        views.sort(key=lambda value: (value.frame_number, value.key))
        frame_numbers = [value.frame_number for value in views]
        if len(frame_numbers) != len(set(frame_numbers)):
            raise ValueError(f"Duplicate regular frame_num for camera_uuid {camera_uuid}")
    return grouped


def discover_stanford_panoramas(
    area_root: str | Path,
    *,
    expected_area: int = 1,
    center_tolerance_m: float = 2e-3,
) -> list[StanfordPanorama]:
    """Discover panoramas and pair regular pose metadata by ``camera_uuid``.

    The function reads pose JSON only.  It never decodes RGB, depth, or semantic
    images.  All four panorama modalities are required and must have identical
    filename pairing keys.  A panorama is allowed to have no regular view, as four
    such stations exist in the official Area_1 release.
    """

    if not np.isfinite(center_tolerance_m) or center_tolerance_m < 0.0:
        raise ValueError("center_tolerance_m must be a finite non-negative value")
    area_path = Path(area_root).expanduser().resolve()
    pano_root = area_path / "pano"
    specs = {
        "rgb": ("*.png", pano_root / "rgb"),
        "depth": ("*.png", pano_root / "depth"),
        "semantic": ("*.png", pano_root / "semantic"),
        "pose": ("*.json", pano_root / "pose"),
    }
    indexed = {
        modality: _index_pano_modality(directory, modality, pattern)
        for modality, (pattern, directory) in specs.items()
    }
    reference_keys = set(indexed["rgb"])
    for modality, values in indexed.items():
        missing = sorted(reference_keys - set(values))
        extra = sorted(set(values) - reference_keys)
        if missing or extra:
            raise ValueError(
                f"Stanford panorama {modality} pairing differs from RGB: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    regular_by_uuid = _discover_regular_views(
        area_path / "data" / "pose", expected_area=expected_area
    )
    panoramas: list[StanfordPanorama] = []
    seen_uuids: set[str] = set()
    for key in sorted(reference_keys):
        name_match = _PANO_NAME.match(indexed["pose"][key].stem)
        if name_match is None:  # guarded while indexing; keeps type narrowing explicit
            raise AssertionError(f"Unreachable invalid panorama pose key: {key}")
        filename_uuid = str(name_match.group("uuid")).lower()
        filename_room = str(name_match.group("room"))
        parsed = _parse_pose(
            indexed["pose"][key],
            expected_area=expected_area,
            expected_uuid=filename_uuid,
            expected_room=filename_room,
            panorama=True,
        )
        if parsed.camera_uuid in seen_uuids:
            raise ValueError(f"Duplicate Stanford panorama camera_uuid: {parsed.camera_uuid}")
        seen_uuids.add(parsed.camera_uuid)

        regular_views = tuple(regular_by_uuid.get(parsed.camera_uuid, ()))
        pano_centre = parsed.camera_to_area[:3, 3]
        for view in regular_views:
            if view.room != parsed.room:
                raise ValueError(
                    f"Regular/panorama room mismatch for camera_uuid {parsed.camera_uuid}: "
                    f"{view.room!r} != {parsed.room!r}"
                )
            centre_error = float(np.linalg.norm(view.camera_to_area[:3, 3] - pano_centre))
            if centre_error > center_tolerance_m:
                raise ValueError(
                    "Regular and panorama camera centres disagree for camera_uuid "
                    f"{parsed.camera_uuid}: {centre_error:.9g} m > "
                    f"{center_tolerance_m:.9g} m"
                )

        panoramas.append(
            StanfordPanorama(
                key=key,
                sample_id=f"{parsed.room}/{parsed.camera_uuid}/equirectangular",
                room=parsed.room,
                pose_room=parsed.pose_room,
                camera_uuid=parsed.camera_uuid,
                rgb_path=indexed["rgb"][key],
                depth_path=indexed["depth"][key],
                semantic_path=indexed["semantic"][key],
                pose_path=indexed["pose"][key],
                intrinsic=parsed.intrinsic,
                world_to_camera=parsed.world_to_camera,
                camera_to_area=parsed.camera_to_area,
                regular_views=regular_views,
            )
        )
    return panoramas


def _validate_image_shape(image_shape: tuple[int, int]) -> tuple[int, int]:
    if len(image_shape) != 2:
        raise ValueError(f"image_shape must contain (height, width), got {image_shape!r}")
    height, width = image_shape
    if isinstance(height, bool) or isinstance(width, bool):
        raise TypeError(f"image_shape values must be positive integers, got {image_shape!r}")
    if int(height) != height or int(width) != width or height <= 0 or width <= 0:
        raise ValueError(f"image_shape values must be positive integers, got {image_shape!r}")
    return int(height), int(width)


def _validate_pixels(pixels: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (..., 2), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite coordinates")
    return values


def _validate_positive_values(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    try:
        result = np.broadcast_to(result, shape)
    except ValueError as error:
        raise ValueError(
            f"{name} cannot broadcast to pixel shape {shape}: {result.shape}"
        ) from error
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only finite positive values")
    return result


def _validate_intrinsic(intrinsic: np.ndarray) -> np.ndarray:
    value = np.asarray(intrinsic, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"intrinsic must be 3x3, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("intrinsic contains non-finite values")
    if not np.allclose(value[2], [0.0, 0.0, 1.0], atol=1e-10, rtol=0.0):
        raise ValueError("intrinsic must use the standard pinhole homogeneous row [0, 0, 1]")
    if value[0, 0] <= 0.0 or value[1, 1] <= 0.0 or abs(np.linalg.det(value)) <= _EPS:
        raise ValueError("intrinsic must have positive focal lengths and be invertible")
    return value


def _validate_transform(transform: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-10, rtol=0.0):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    rotation = value[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=_POSE_ATOL, rtol=0.0):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=_POSE_ATOL, rtol=0.0):
        raise ValueError(f"{name} rotation is not proper")
    return value


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def wrap_erp_horizontal(u: np.ndarray, width: int) -> np.ndarray:
    """Wrap ERP horizontal pixel coordinates to ``[0, width)``."""

    _, validated_width = _validate_image_shape((1, width))
    values = np.asarray(u, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("ERP horizontal coordinates contain non-finite values")
    return np.mod(values, float(validated_width))


def erp_pixels_to_cv_rays(
    pixels: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert ERP pixel-centre coordinates to OpenCV camera-space unit rays.

    The OpenCV axes are ``+x`` right, ``+y`` down and ``+z`` forward.  Pixel
    coordinates use pixel centres, so the equator/front direction is located at
    ``(width / 2 - 0.5, height / 2 - 0.5)``.  Horizontal coordinates wrap.
    """

    height, width = _validate_image_shape(image_shape)
    values = _validate_pixels(pixels, "ERP pixels")
    v = values[..., 1]
    if np.any(v < -0.5) or np.any(v > height - 0.5):
        raise ValueError(
            f"ERP vertical coordinates must lie in [-0.5, {height - 0.5}], "
            f"got range [{v.min()}, {v.max()}]"
        )
    phase = np.mod(values[..., 0] + 0.5, float(width)) / float(width)
    longitude = (phase - 0.5) * (2.0 * np.pi)
    latitude = ((v + 0.5) / float(height) - 0.5) * np.pi
    cos_latitude = np.cos(latitude)
    return np.stack(
        (
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
            cos_latitude * np.cos(longitude),
        ),
        axis=-1,
    )


def cv_rays_to_erp_pixels(
    rays: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Convert OpenCV camera-space unit rays to wrapped ERP pixel coordinates."""

    height, width = _validate_image_shape(image_shape)
    values = np.asarray(rays, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"CV rays must have shape (..., 3), got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("CV rays contain non-finite values")
    norm = np.linalg.norm(values, axis=-1)
    if np.any(norm <= _EPS):
        raise ValueError("CV rays must be non-zero")
    if not np.allclose(norm, 1.0, atol=1e-7, rtol=0.0):
        raise ValueError("CV rays must be unit length")
    unit = values / norm[..., None]
    longitude = np.arctan2(unit[..., 0], unit[..., 2])
    latitude = np.arcsin(np.clip(unit[..., 1], -1.0, 1.0))
    u = float(width) * (longitude / (2.0 * np.pi) + 0.5) - 0.5
    v = float(height) * (latitude / np.pi + 0.5) - 0.5
    return np.stack((wrap_erp_horizontal(u, width), v), axis=-1)


def regular_z_depth_to_pano(
    regular_pixels: np.ndarray,
    z_depth: np.ndarray,
    intrinsic: np.ndarray,
    regular_camera_to_area: np.ndarray,
    pano_world_to_camera: np.ndarray,
    pano_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Project regular-view ``z`` depth samples to ERP pixels and radial range."""

    pixels = _validate_pixels(regular_pixels, "regular pixels")
    depth = _validate_positive_values(z_depth, pixels.shape[:-1], "z_depth")
    camera_matrix = _validate_intrinsic(intrinsic)
    regular_to_area = _validate_transform(regular_camera_to_area, "regular_camera_to_area")
    area_to_pano = _validate_transform(pano_world_to_camera, "pano_world_to_camera")
    _validate_image_shape(pano_shape)

    homogeneous_pixels = np.concatenate(
        (pixels, np.ones(pixels.shape[:-1] + (1,), dtype=np.float64)), axis=-1
    )
    regular_points = (homogeneous_pixels @ np.linalg.inv(camera_matrix).T) * depth[..., None]
    area_points = _transform_points(regular_points, regular_to_area)
    pano_points = _transform_points(area_points, area_to_pano)
    pano_range = np.linalg.norm(pano_points, axis=-1)
    if np.any(pano_range <= _EPS):
        raise ValueError("Regular depth projects to the panorama camera centre")
    pano_rays = pano_points / pano_range[..., None]
    return cv_rays_to_erp_pixels(pano_rays, pano_shape), pano_range


def pano_range_to_regular_z(
    pano_pixels: np.ndarray,
    pano_range: np.ndarray,
    pano_shape: tuple[int, int],
    pano_camera_to_area: np.ndarray,
    regular_world_to_camera: np.ndarray,
) -> np.ndarray:
    """Back-project panorama radial ranges and return signed regular-view ``z``."""

    pixels = _validate_pixels(pano_pixels, "panorama pixels")
    radial_range = _validate_positive_values(pano_range, pixels.shape[:-1], "pano_range")
    pano_to_area = _validate_transform(pano_camera_to_area, "pano_camera_to_area")
    area_to_regular = _validate_transform(regular_world_to_camera, "regular_world_to_camera")
    pano_rays = erp_pixels_to_cv_rays(pixels, pano_shape)
    pano_points = pano_rays * radial_range[..., None]
    area_points = _transform_points(pano_points, pano_to_area)
    regular_points = _transform_points(area_points, area_to_regular)
    return regular_points[..., 2]


def pano_range_to_regular_projection(
    pano_pixels: np.ndarray,
    pano_range: np.ndarray,
    pano_shape: tuple[int, int],
    pano_camera_to_area: np.ndarray,
    regular_world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Back-project panorama range into a regular camera.

    Returns regular pixel coordinates, signed ``z`` depth, and a front-facing
    mask.  Pixel coordinates are ``NaN`` where ``z`` is numerically zero; callers
    should additionally apply their regular image bounds.
    """

    pixels = _validate_pixels(pano_pixels, "panorama pixels")
    radial_range = _validate_positive_values(pano_range, pixels.shape[:-1], "pano_range")
    pano_to_area = _validate_transform(pano_camera_to_area, "pano_camera_to_area")
    area_to_regular = _validate_transform(regular_world_to_camera, "regular_world_to_camera")
    camera_matrix = _validate_intrinsic(intrinsic)
    pano_rays = erp_pixels_to_cv_rays(pixels, pano_shape)
    pano_points = pano_rays * radial_range[..., None]
    area_points = _transform_points(pano_points, pano_to_area)
    regular_points = _transform_points(area_points, area_to_regular)
    z_depth = regular_points[..., 2]
    projected = regular_points @ camera_matrix.T
    regular_pixels = np.full(pixels.shape, np.nan, dtype=np.float64)
    non_parallel = np.abs(z_depth) > _EPS
    regular_pixels[non_parallel] = projected[non_parallel, :2] / z_depth[non_parallel, None]
    return regular_pixels, z_depth, z_depth > _EPS


def build_regular_pano_lookup(
    pano_shape: tuple[int, int],
    pano_camera_to_area: np.ndarray,
    regular_world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    regular_shape: tuple[int, int],
    *,
    center_tolerance_m: float = 5e-3,
) -> RegularPanoLookup:
    """Build the depth-independent regular-to-ERP lookup for co-located cameras."""

    _validate_image_shape(pano_shape)
    if not np.isfinite(center_tolerance_m) or center_tolerance_m < 0.0:
        raise ValueError("center_tolerance_m must be finite and non-negative")
    pano_to_area = _validate_transform(pano_camera_to_area, "pano_camera_to_area")
    area_to_regular = _validate_transform(
        regular_world_to_camera,
        "regular_world_to_camera",
    )
    camera_matrix = _validate_intrinsic(intrinsic)
    regular_height, regular_width = _validate_image_shape(regular_shape)
    regular_to_area = np.linalg.inv(area_to_regular)
    center_error = float(np.linalg.norm(regular_to_area[:3, 3] - pano_to_area[:3, 3]))
    if center_error > center_tolerance_m:
        raise ValueError(
            "Panorama and regular cameras are not co-located: "
            f"centre error {center_error:.6f} m exceeds {center_tolerance_m:.6f} m"
        )

    x, y = np.meshgrid(
        np.arange(regular_width, dtype=np.float64),
        np.arange(regular_height, dtype=np.float64),
    )
    regular_pixels = np.stack((x, y), axis=-1)
    erp_pixels, radial_range_at_unit_z = regular_z_depth_to_pano(
        regular_pixels,
        np.ones(regular_shape, dtype=np.float64),
        camera_matrix,
        regular_to_area,
        np.linalg.inv(pano_to_area),
        pano_shape,
    )
    z_per_radial_range = 1.0 / radial_range_at_unit_z
    if not np.isfinite(z_per_radial_range).all() or np.any(z_per_radial_range <= 0.0):
        raise RuntimeError("Regular-to-panorama lookup produced invalid z/range factors")
    return RegularPanoLookup(
        erp_pixels=erp_pixels,
        z_per_radial_range=z_per_radial_range,
        pano_shape=tuple(map(int, pano_shape)),
        regular_shape=tuple(map(int, regular_shape)),
    )


def sample_pano_range_to_regular_z(
    pano_range: np.ndarray,
    pano_valid: np.ndarray,
    lookup: RegularPanoLookup,
    *,
    minimum_interpolation_weight: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an ERP radial-range image through a reusable regular lookup."""

    values = np.asarray(pano_range, dtype=np.float32)
    valid = np.asarray(pano_valid, dtype=bool)
    if values.ndim != 2 or values.shape != lookup.pano_shape or valid.shape != values.shape:
        raise ValueError(
            "Panorama range, validity, and lookup shapes differ: "
            f"{values.shape}, {valid.shape}, {lookup.pano_shape}"
        )
    if (
        not np.isfinite(minimum_interpolation_weight)
        or minimum_interpolation_weight <= 0.0
        or minimum_interpolation_weight > 1.0
    ):
        raise ValueError("minimum_interpolation_weight must lie in (0, 1]")
    if np.any(valid & (~np.isfinite(values) | (values <= 0.0))):
        raise ValueError("Valid panorama range samples must be finite and positive")

    # Pad one column on each side so bilinear interpolation is periodic only in
    # longitude.  Latitude remains constant-padded and is validity-normalized.
    safe_values = np.where(valid, values, 0.0).astype(np.float32, copy=False)
    padded_values = np.concatenate(
        (safe_values[:, -1:], safe_values, safe_values[:, :1]),
        axis=1,
    )
    padded_valid = np.concatenate(
        (valid[:, -1:], valid, valid[:, :1]),
        axis=1,
    ).astype(np.float32)
    map_x = (lookup.erp_pixels[..., 0] + 1.0).astype(np.float32)
    map_y = lookup.erp_pixels[..., 1].astype(np.float32)
    numerator = cv2.remap(
        padded_values,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    denominator = cv2.remap(
        padded_valid,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    output_valid = denominator >= float(minimum_interpolation_weight)
    sampled_range = np.ones(lookup.regular_shape, dtype=np.float32)
    sampled_range[output_valid] = (numerator[output_valid] / denominator[output_valid]).astype(
        np.float32
    )
    z_depth = sampled_range * lookup.z_per_radial_range
    output_valid &= np.isfinite(z_depth) & (z_depth > 0.0)
    output = np.zeros(lookup.regular_shape, dtype=np.float32)
    output[output_valid] = z_depth[output_valid].astype(np.float32)
    return output, output_valid


def pano_range_image_to_regular_z(
    pano_range: np.ndarray,
    pano_valid: np.ndarray,
    pano_camera_to_area: np.ndarray,
    regular_world_to_camera: np.ndarray,
    intrinsic: np.ndarray,
    regular_shape: tuple[int, int],
    *,
    center_tolerance_m: float = 5e-3,
    minimum_interpolation_weight: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ERP radial range back to regular ``z`` with periodic bilinear interpolation."""

    lookup = build_regular_pano_lookup(
        np.asarray(pano_range).shape,
        pano_camera_to_area,
        regular_world_to_camera,
        intrinsic,
        regular_shape,
        center_tolerance_m=center_tolerance_m,
    )
    return sample_pano_range_to_regular_z(
        pano_range,
        pano_valid,
        lookup,
        minimum_interpolation_weight=minimum_interpolation_weight,
    )
