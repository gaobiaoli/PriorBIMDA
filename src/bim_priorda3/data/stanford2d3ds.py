from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.data.geometry import scale_intrinsics

STANFORD_SEMANTIC_CLASSES = (
    "ceiling",
    "floor",
    "wall",
    "beam",
    "column",
    "window",
    "door",
    "table",
    "chair",
    "sofa",
    "bookcase",
    "board",
    "clutter",
)
STRUCTURAL_CLASS_IDS = frozenset(range(7))
FURNITURE_CLASS_IDS = frozenset(
    STANFORD_SEMANTIC_CLASSES.index(name) for name in ("table", "chair", "sofa", "bookcase")
)
UNKNOWN_SEMANTIC_ID = 255
_MODALITY_SUFFIX = re.compile(r"^(?P<key>.+)_domain_*(?P<modality>rgb|depth|semantic|pose)$")
_NEAREST = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)


@dataclass(frozen=True)
class StanfordFrame:
    key: str
    sample_id: str
    room: str
    pose_room: str
    camera_uuid: str
    frame_number: int
    rgb_path: Path
    depth_path: Path
    semantic_path: Path
    pose_path: Path
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    camera_to_area: np.ndarray


def _modality_key(path: Path, expected: str) -> str:
    match = _MODALITY_SUFFIX.match(path.stem)
    if match is None or match.group("modality") != expected:
        raise ValueError(
            f"Unexpected Stanford {expected} filename (no stable pairing key): {path.name}"
        )
    return str(match.group("key"))


def room_key_from_pose_room(value: str, *, expected_area: int = 1) -> str:
    parts = value.rsplit("_", 1)
    if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
        raise ValueError(f"Invalid Stanford pose room identifier: {value!r}")
    if int(parts[1]) != expected_area:
        raise ValueError(f"Expected Area_{expected_area}, but pose room is {value!r}")
    return parts[0]


def pose_matrices(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame K, Area-to-camera and camera-to-Area matrices.

    The released JSON data stores ``camera_rt_matrix`` as a 3x4 world-to-camera
    matrix, despite an old README typo that calls it 4x3.  We validate it against
    ``camera_location`` so a transposed or inverted convention cannot silently
    produce plausible-looking but incorrect BIM renders.
    """

    intrinsic = np.asarray(payload.get("camera_k_matrix"), dtype=np.float64)
    rt = np.asarray(payload.get("camera_rt_matrix"), dtype=np.float64)
    camera_location = np.asarray(payload.get("camera_location"), dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise ValueError(f"camera_k_matrix must be 3x3, got {intrinsic.shape}")
    if rt.shape != (3, 4):
        raise ValueError(
            "camera_rt_matrix must be the released 3x4 [R|t] world-to-camera "
            f"matrix, got {rt.shape}"
        )
    if camera_location.shape != (3,):
        raise ValueError(f"camera_location must contain three values, got {camera_location.shape}")
    if not (
        np.isfinite(intrinsic).all()
        and np.isfinite(rt).all()
        and np.isfinite(camera_location).all()
    ):
        raise ValueError("Stanford pose contains non-finite camera parameters")

    rotation = rt[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=2e-4, rtol=0.0):
        raise ValueError("camera_rt_matrix rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError("camera_rt_matrix rotation is not a proper rotation")
    location_residual = rotation @ camera_location + rt[:, 3]
    if np.linalg.norm(location_residual) > 2e-3:
        raise ValueError(
            "camera_rt_matrix is inconsistent with camera_location; expected "
            f"world-to-camera [R|t], residual={location_residual.tolist()}"
        )

    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3] = rt
    camera_to_area = np.eye(4, dtype=np.float64)
    camera_to_area[:3, :3] = rotation.T
    camera_to_area[:3, 3] = camera_location
    if not np.allclose(world_to_camera @ camera_to_area, np.eye(4), atol=2e-3):
        raise ValueError("Derived Stanford camera transform is not invertible")
    return intrinsic, world_to_camera, camera_to_area


def discover_stanford_frames(
    area_root: str | Path,
    *,
    expected_area: int = 1,
    available_bim_rooms: set[str] | None = None,
) -> list[StanfordFrame]:
    area_path = Path(area_root).expanduser().resolve()
    data_root = area_path / "data"
    modality_specs = {
        "rgb": (data_root / "rgb", "*.png"),
        "depth": (data_root / "depth", "*.png"),
        "semantic": (data_root / "semantic", "*.png"),
        "pose": (data_root / "pose", "*.json"),
    }
    indexed: dict[str, dict[str, Path]] = {}
    for modality, (directory, pattern) in modality_specs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Stanford modality directory is missing: {directory}")
        values: dict[str, Path] = {}
        for path in sorted(directory.glob(pattern)):
            key = _modality_key(path, modality)
            if key in values:
                raise ValueError(f"Duplicate Stanford {modality} pairing key: {key}")
            values[key] = path.resolve()
        if not values:
            raise RuntimeError(f"No Stanford {modality} files found in {directory}")
        indexed[modality] = values

    reference_keys = set(indexed["rgb"])
    for modality, values in indexed.items():
        missing = sorted(reference_keys - set(values))
        extra = sorted(set(values) - reference_keys)
        if missing or extra:
            raise ValueError(
                f"Stanford {modality} pairing differs from RGB: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )

    frames: list[StanfordFrame] = []
    for key in sorted(reference_keys):
        pose_path = indexed["pose"][key]
        with pose_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Stanford pose must be a JSON object: {pose_path}")
        pose_room = str(payload.get("room", ""))
        room = room_key_from_pose_room(pose_room, expected_area=expected_area)
        if available_bim_rooms is not None and room not in available_bim_rooms:
            raise ValueError(f"No BIMSyn IFC with stem {room!r} for Stanford pose {pose_path.name}")
        intrinsic, world_to_camera, camera_to_area = pose_matrices(payload)
        camera_uuid = str(payload.get("camera_uuid", payload.get("point_uuid", "")))
        if not camera_uuid:
            raise ValueError(f"Stanford pose lacks camera_uuid: {pose_path}")
        frame_number = int(payload.get("frame_num"))
        frames.append(
            StanfordFrame(
                key=key,
                sample_id=f"{room}/{key}",
                room=room,
                pose_room=pose_room,
                camera_uuid=camera_uuid,
                frame_number=frame_number,
                rgb_path=indexed["rgb"][key],
                depth_path=indexed["depth"][key],
                semantic_path=indexed["semantic"][key],
                pose_path=pose_path,
                intrinsic=intrinsic,
                world_to_camera=world_to_camera,
                camera_to_area=camera_to_area,
            )
        )
    return frames


def scaled_frame_intrinsic(
    frame: StanfordFrame,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    return scale_intrinsics(frame.intrinsic, source_shape, target_shape)


def load_stanford_depth(
    path: str | Path,
    target_shape: tuple[int, int],
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Cannot read Stanford depth image: {path}")
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(
            f"Stanford depth must be a uint16 single-channel PNG, got {raw.dtype} {raw.shape}"
        )
    height, width = target_shape
    if raw.shape != (height, width):
        raw = cv2.resize(raw, (width, height), interpolation=_NEAREST)
    depth = raw.astype(np.float32) / 512.0
    valid = (
        (raw != np.uint16(65535))
        & np.isfinite(depth)
        & (depth >= float(min_depth))
        & (depth <= float(max_depth))
    )
    depth[~valid] = 0.0
    return depth, valid


STANFORD_RGB_SUFFIX = "_domain_rgb.png"
STANFORD_DEPTH_SUFFIX = "_domain_depth.png"


def official_regular_depth_path(image_path: str | Path) -> Path:
    """Resolve the official regular-view depth PNG paired with an RGB image."""

    image = Path(image_path).expanduser().resolve()
    if not image.name.endswith(STANFORD_RGB_SUFFIX):
        raise ValueError(
            f"Stanford RGB filename must end with {STANFORD_RGB_SUFFIX!r}: {image}"
        )
    depth_name = image.name[: -len(STANFORD_RGB_SUFFIX)] + STANFORD_DEPTH_SUFFIX
    depth_path = image.parent.parent / "depth" / depth_name
    if not depth_path.is_file():
        raise FileNotFoundError(f"Official Stanford regular depth not found: {depth_path}")
    return depth_path


def load_stanford_all_valid_depth(
    path: str | Path,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load all positive official z-depth values except sentinel 65535."""

    return load_stanford_depth(
        path,
        target_shape,
        min_depth=float(np.nextafter(np.float32(0.0), np.float32(1.0))),
        max_depth=float("inf"),
    )


def semantic_label_lut(labels_path: str | Path) -> np.ndarray:
    with Path(labels_path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        labels = json.load(handle)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise TypeError("semantic_labels.json must contain a list of strings")
    class_to_id = {name: index for index, name in enumerate(STANFORD_SEMANTIC_CLASSES)}
    lut = np.full(len(labels), UNKNOWN_SEMANTIC_ID, dtype=np.uint8)
    for index, label in enumerate(labels):
        instance_class = label.split("_", 1)[0]
        if instance_class in class_to_id:
            lut[index] = class_to_id[instance_class]
    return lut


def load_stanford_semantics(
    path: str | Path,
    target_shape: tuple[int, int],
    label_lut: np.ndarray,
) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read Stanford semantic image: {path}")
    height, width = target_shape
    if bgr.shape[:2] != (height, width):
        bgr = cv2.resize(bgr, (width, height), interpolation=_NEAREST)
    rgb = bgr[..., ::-1].astype(np.uint32)
    indices = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    classes = np.full(indices.shape, UNKNOWN_SEMANTIC_ID, dtype=np.uint8)
    valid = indices < len(label_lut)
    classes[valid] = label_lut[indices[valid]]
    return classes


def semantic_subset_masks(classes: np.ndarray) -> dict[str, np.ndarray]:
    structural = np.isin(classes, np.fromiter(STRUCTURAL_CLASS_IDS, dtype=np.uint8))
    furniture = np.isin(classes, np.fromiter(FURNITURE_CLASS_IDS, dtype=np.uint8))
    known = classes != UNKNOWN_SEMANTIC_ID
    return {
        "semantic_valid": known,
        "structural_mask": structural,
        "furniture_mask": furniture,
        "non_structural_mask": known & ~structural,
    }
