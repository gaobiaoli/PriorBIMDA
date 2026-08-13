from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.config import Config, resolve_project_path
from bim_priorda3.data.geometry import depth_edges
from bim_priorda3.data.ifc_envelope import (
    GLOBAL_CORE_CATEGORIES,
    build_global_ifc_envelope_scene,
    render_ifc_envelope,
)
from bim_priorda3.data.preparation import DA3Prediction, DA3PredictionProvider
from bim_priorda3.data.splits import manifest_preparation_identity
from bim_priorda3.data.stanford2d3ds import (
    StanfordFrame,
    discover_stanford_frames,
    load_stanford_depth,
    load_stanford_semantics,
    scaled_frame_intrinsic,
    semantic_label_lut,
    semantic_subset_masks,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _store_depth(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


def _atomic_savez_compressed(path: Path, payload: dict[str, Any]) -> None:
    """Write one prepared sample without exposing a partial NPZ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_area_from_bim(value: Any, room: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{room}: T_area_from_bim must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9, rtol=0.0):
        raise ValueError(f"{room}: T_area_from_bim has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4, rtol=0.0):
        raise ValueError(f"{room}: alignment must have fixed unit scale and rigid rotation")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError(f"{room}: alignment rotation is not right-handed")
    if not np.allclose(rotation[2], [0, 0, 1], atol=2e-4, rtol=0.0):
        raise ValueError(f"{room}: alignment must preserve the shared Z-up axis")
    return transform


def _frame_fingerprint(
    frame: StanfordFrame,
    *,
    rgb_sha256: str,
    depth_sha256: str,
    semantic_sha256: str,
    semantic_labels_sha256: str,
    alignment_sha256: str,
    area_from_bim: np.ndarray,
    global_bim_fingerprint: str,
    target_shape: tuple[int, int],
    min_depth: float,
    max_depth: float,
    da3_model: str,
    da3_revision: str,
    da3_process_res: int,
    da3_cache_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": 2,
            "frame_key": frame.key,
            "rgb_sha256": rgb_sha256,
            "depth_sha256": depth_sha256,
            "semantic_sha256": semantic_sha256,
            "semantic_labels_sha256": semantic_labels_sha256,
            "pose_sha256": _sha256(frame.pose_path),
            "alignment_receipt_sha256": alignment_sha256,
            "T_area_from_bim": area_from_bim.tolist(),
            "global_bim_fingerprint_sha256": global_bim_fingerprint,
            "target_shape": list(target_shape),
            "min_depth_m": float(min_depth),
            "max_depth_m": float(max_depth),
            "da3_model": da3_model,
            "da3_revision": da3_revision,
            "da3_process_res": int(da3_process_res),
            "da3_cache_artifact_sha256": da3_cache_sha256,
            "semantic_protocol": "official-24bit-label-index-v1",
            "bim_protocol": "global-area-fixed-core-envelope-v1",
            "bim_included_categories": list(GLOBAL_CORE_CATEGORIES),
            "bim_excluded_categories": ["door", "window"],
        }
    )


def _validate_alignment_receipt(
    alignment: dict[str, Any],
    rooms: list[str],
    ifc_paths: dict[str, Path],
) -> dict[str, np.ndarray]:
    protocol = (alignment.get("schema_version"), alignment.get("method"))
    known_protocols = {
        (1, "structural-mesh-symmetric-constrained-yaw-icp-v1"),
        (2, "structural-mesh-symmetric-constrained-yaw-icp-class-rerank-v2"),
    }
    if protocol not in known_protocols:
        raise ValueError(
            "Unknown BIM alignment schema/method pair; expected one of "
            f"{sorted(known_protocols)!r}, got {protocol!r}"
        )
    expected_frames = {
        "source": "BIMSyn IFC room-local, metres, Z-up",
        "target": "Stanford Area_1 world, metres, Z-up",
        "transform": "T_area_from_bim",
    }
    if alignment.get("coordinate_frames") != expected_frames:
        raise ValueError("BIM alignment receipt coordinate_frames are not the Area_1 protocol")
    constraints = alignment.get("constraints")
    if not isinstance(constraints, dict):
        raise TypeError("BIM alignment receipt constraints must be a mapping")
    expected_constraints: dict[str, Any] = {
        "degrees_of_freedom": ["yaw_rad", "tx_m", "ty_m", "tz_m"],
        "scale": 1.0,
        "z_axis_up": True,
        "roll_rad": 0.0,
        "pitch_rad": 0.0,
        "per_frame_alignment": False,
        "uses_rgb": False,
        "uses_depth_images": False,
    }
    if protocol[0] == 2:
        expected_constraints.update(
            semantic_classes_affect_icp_fit=False,
            uses_semantic_face_classes_for_candidate_reranking=True,
        )
    if constraints != expected_constraints:
        raise ValueError("BIM alignment receipt violates the fixed unit-scale yaw protocol")
    semantic_audit = alignment.get("sources", {}).get("semantic_obj", {})
    conversion = semantic_audit.get("obj_coordinate_conversion", {})
    expected_rotation = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    if (
        semantic_audit.get("geometry_unit") != "metre"
        or conversion.get("fitted") is not False
        or conversion.get("rotation_area_from_obj") != expected_rotation
    ):
        raise ValueError(
            "BIM alignment receipt lacks the fixed Blender-Y-up to Area-Z-up conversion"
        )
    failures = alignment.get("failures")
    if failures != []:
        raise ValueError("BIM alignment receipt contains registration failures")
    parameters = alignment.get("parameters")
    if not isinstance(parameters, dict):
        raise TypeError("BIM alignment receipt parameters must be a mapping")
    min_fitness = float(parameters.get("min_fitness", float("nan")))
    max_rmse = float(parameters.get("max_rmse_m", float("nan")))
    if not np.isfinite(min_fitness) or not np.isfinite(max_rmse):
        raise ValueError("BIM alignment receipt lacks finite quality thresholds")

    alignment_rooms = alignment.get("rooms")
    if not isinstance(alignment_rooms, dict):
        raise TypeError("BIM alignment receipt rooms must be a mapping")
    transforms: dict[str, np.ndarray] = {}
    required_quality = {
        "maximum_inlier_rmse",
        "minimum_fitness",
        "proper_rotation",
        "unit_scale",
        "z_axis_up",
    }
    for room in rooms:
        entry = alignment_rooms.get(room)
        if not isinstance(entry, dict):
            raise TypeError(f"Alignment receipt has no room mapping for {room!r}")
        if entry.get("accepted") is not True or entry.get("status") != "accepted":
            raise ValueError(f"Alignment for {room!r} is not accepted")
        quality = entry.get("quality_checks")
        if (
            not isinstance(quality, dict)
            or set(quality) != required_quality
            or not all(value is True for value in quality.values())
        ):
            raise ValueError(f"{room}: alignment quality checks are incomplete or failed")
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            raise TypeError(f"{room}: alignment metrics must be a mapping")
        fitness = float(metrics.get("fitness", float("nan")))
        rmse = float(metrics.get("rmse_m", float("nan")))
        if not np.isfinite(fitness) or fitness < min_fitness:
            raise ValueError(f"{room}: alignment fitness fails the receipt threshold")
        if not np.isfinite(rmse) or rmse > max_rmse:
            raise ValueError(f"{room}: alignment RMSE fails the receipt threshold")
        if protocol[0] == 2:
            semantic_reranking = entry.get("semantic_reranking")
            if (
                not isinstance(semantic_reranking, dict)
                or semantic_reranking.get("enabled") is not True
                or type(semantic_reranking.get("selected_candidate_index")) is not int
            ):
                raise ValueError(f"{room}: v2 semantic candidate reranking audit is invalid")
        expected_ifc_sha = _sha256(ifc_paths[room])
        if entry.get("source_ifc_sha256") != expected_ifc_sha:
            raise ValueError(f"{room}: alignment IFC SHA256 differs from the current IFC file")
        transforms[room] = _validate_area_from_bim(entry.get("T_area_from_bim"), room)
    return transforms


def _reuse_record(
    sample_path: Path,
    expected_fingerprint: str,
    frame: StanfordFrame,
    source_root: Path,
    output_root: Path,
    target_shape: tuple[int, int],
) -> dict[str, Any]:
    with np.load(sample_path, allow_pickle=False) as cached:
        required = {
            "sample_schema_version",
            "base_depth",
            "base_confidence",
            "bim_depth",
            "bim_valid",
            "bim_normals",
            "bim_edge",
            "gt_depth",
            "gt_valid",
            "gt_weight",
            "intrinsic",
            "camera_to_bim",
            "camera_to_area",
            "area_from_bim",
            "bim_scene_coordinate_frame",
            "global_bim_fingerprint_sha256",
            "da3_cache_artifact_sha256",
            "semantic_class",
            "semantic_valid",
            "furniture_mask",
            "structural_mask",
            "non_structural_mask",
            "bim_category",
            "preparation_fingerprint_sha256",
        }
        missing = sorted(required - set(cached.files))
        if missing:
            raise ValueError(
                f"Existing Stanford sample has an obsolete schema ({missing}); "
                f"rerun with --overwrite: {sample_path}"
            )
        actual = str(cached["preparation_fingerprint_sha256"].item())
        if actual != expected_fingerprint:
            raise ValueError(
                "Existing Stanford sample provenance differs from the current "
                f"inputs; rerun with --overwrite: {sample_path}"
            )
        _validate_reusable_sample(cached, sample_path, target_shape)
        gt_valid = cached["gt_valid"] > 0
        gt_valid_pixels = int(gt_valid.sum())
        furniture_pixels = int(((cached["furniture_mask"] > 0) & gt_valid).sum())
    return _manifest_record(
        frame,
        sample_path,
        source_root,
        output_root,
        gt_valid_pixels=gt_valid_pixels,
        furniture_pixels=furniture_pixels,
        fingerprint=expected_fingerprint,
        da3_source="existing",
    )


def _validate_reusable_sample(
    cached: np.lib.npyio.NpzFile,
    sample_path: Path,
    target_shape: tuple[int, int],
) -> None:
    scalar_schema = cached["sample_schema_version"]
    if scalar_schema.shape != () or int(scalar_schema.item()) != 2:
        raise ValueError(f"{sample_path}: sample_schema_version must be 2")
    if str(cached["bim_scene_coordinate_frame"].item()) != "Stanford Area_1 world":
        raise ValueError(f"{sample_path}: BIM scene is not in the fixed Area_1 frame")
    for key in (
        "global_bim_fingerprint_sha256",
        "da3_cache_artifact_sha256",
        "preparation_fingerprint_sha256",
    ):
        value = cached[key]
        if value.shape != () or len(str(value.item())) != 64:
            raise ValueError(f"{sample_path}: {key} must be a scalar SHA256")
    height, width = target_shape
    image_shape = (height, width)
    for key in (
        "base_depth",
        "base_confidence",
        "bim_depth",
        "bim_valid",
        "bim_edge",
        "bim_category",
        "gt_depth",
        "gt_valid",
        "gt_weight",
        "semantic_class",
        "semantic_valid",
        "furniture_mask",
        "structural_mask",
        "non_structural_mask",
    ):
        if cached[key].shape != image_shape:
            raise ValueError(
                f"{sample_path}: {key} shape {cached[key].shape} must equal {image_shape}"
            )
    if cached["bim_normals"].shape != (3, height, width):
        raise ValueError(f"{sample_path}: bim_normals must have shape (3, {height}, {width})")
    expected_dtypes = {
        "base_depth": np.dtype(np.float16),
        "base_confidence": np.dtype(np.float16),
        "bim_depth": np.dtype(np.float16),
        "bim_normals": np.dtype(np.float16),
        "gt_depth": np.dtype(np.float32),
        "gt_weight": np.dtype(np.float16),
        "bim_valid": np.dtype(np.uint8),
        "bim_edge": np.dtype(np.uint8),
        "bim_category": np.dtype(np.uint8),
        "gt_valid": np.dtype(np.uint8),
        "semantic_class": np.dtype(np.uint8),
        "semantic_valid": np.dtype(np.uint8),
        "furniture_mask": np.dtype(np.uint8),
        "structural_mask": np.dtype(np.uint8),
        "non_structural_mask": np.dtype(np.uint8),
    }
    for key, expected_dtype in expected_dtypes.items():
        if cached[key].dtype != expected_dtype:
            raise ValueError(
                f"{sample_path}: {key} dtype {cached[key].dtype} must be {expected_dtype}"
            )
    for key, shape in {
        "intrinsic": (3, 3),
        "camera_to_bim": (4, 4),
        "camera_to_area": (4, 4),
        "area_from_bim": (4, 4),
    }.items():
        value = cached[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{sample_path}: {key} must be finite with shape {shape}")
    for key in (
        "base_depth",
        "base_confidence",
        "bim_depth",
        "bim_normals",
        "gt_depth",
        "gt_weight",
    ):
        if not np.isfinite(cached[key]).all():
            raise ValueError(f"{sample_path}: {key} contains non-finite values")
    binary_keys = (
        "bim_valid",
        "bim_edge",
        "gt_valid",
        "semantic_valid",
        "furniture_mask",
        "structural_mask",
        "non_structural_mask",
    )
    masks: dict[str, np.ndarray] = {}
    for key in binary_keys:
        value = cached[key]
        if not np.isin(value, (0, 1)).all():
            raise ValueError(f"{sample_path}: {key} must be binary")
        masks[key] = value.astype(bool)
    if np.any(cached["bim_depth"][~masks["bim_valid"]] != 0):
        raise ValueError(f"{sample_path}: invalid BIM pixels must have zero depth")
    if np.any(cached["bim_normals"][:, ~masks["bim_valid"]] != 0):
        raise ValueError(f"{sample_path}: invalid BIM pixels must have zero normals")
    if np.any(cached["gt_depth"][~masks["gt_valid"]] != 0):
        raise ValueError(f"{sample_path}: invalid GT pixels must have zero depth")
    semantic_valid = masks["semantic_valid"]
    if not np.array_equal(semantic_valid, cached["semantic_class"] != 255):
        raise ValueError(f"{sample_path}: semantic_valid disagrees with semantic_class")
    if np.any(masks["structural_mask"] & masks["non_structural_mask"]):
        raise ValueError(f"{sample_path}: structural/non-structural masks overlap")
    if np.any(masks["furniture_mask"] & ~masks["non_structural_mask"]):
        raise ValueError(f"{sample_path}: furniture_mask is not a non-structural subset")
    if np.any((masks["structural_mask"] | masks["non_structural_mask"]) & ~semantic_valid):
        raise ValueError(f"{sample_path}: semantic subset masks include unknown pixels")


def _manifest_record(
    frame: StanfordFrame,
    sample_path: Path,
    source_root: Path,
    output_root: Path,
    *,
    gt_valid_pixels: int,
    furniture_pixels: int,
    fingerprint: str,
    da3_source: str,
) -> dict[str, Any]:
    try:
        image_relative = frame.rgb_path.relative_to(source_root)
    except ValueError:
        image_relative = Path("area_1/data/rgb") / frame.rgb_path.name
    return {
        "id": frame.sample_id,
        "region": frame.room,
        "dataset": "Stanford2D3DS/Area_1+BIMSyn",
        "image": str(frame.rgb_path),
        "image_relative_to_source": str(image_relative),
        "sample": str(sample_path),
        "sample_relative_to_processed": str(sample_path.relative_to(output_root)),
        "pose": str(frame.pose_path),
        "camera_uuid": frame.camera_uuid,
        "frame_number": int(frame.frame_number),
        "frame_index": int(frame.frame_number),
        "gt_valid_pixels": int(gt_valid_pixels),
        "furniture_valid_pixels": int(furniture_pixels),
        "preparation_fingerprint_sha256": fingerprint,
        "da3_source": da3_source,
    }


def prepare_stanford_area1(
    cfg: Config,
    *,
    rooms: set[str] | None = None,
    max_frames_per_room: int | None = None,
    stride: int = 1,
    overwrite: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if stride < 1:
        raise ValueError("stride must be positive")
    if max_frames_per_room is not None and max_frames_per_room < 1:
        raise ValueError("max_frames_per_room must be positive")

    source_root = resolve_project_path(cfg, cfg.data.source_root)
    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    labels_path = resolve_project_path(cfg, cfg.data.semantic_labels)
    ifc_root = resolve_project_path(cfg, cfg.data.bimsyn_ifc_root)
    alignment_path = resolve_project_path(cfg, cfg.data.bim_alignment)
    output_root = resolve_project_path(cfg, cfg.data.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_cache = output_root / "da3_cache" / "area_1"

    ifc_paths = {path.stem: path.resolve() for path in sorted(ifc_root.glob("*.ifc"))}
    if not ifc_paths:
        raise RuntimeError(f"No BIMSyn IFC files found in {ifc_root}")
    frames = discover_stanford_frames(
        area_root,
        expected_area=1,
        available_bim_rooms=set(ifc_paths),
    )
    frames_by_room: dict[str, list[StanfordFrame]] = defaultdict(list)
    for frame in frames:
        frames_by_room[frame.room].append(frame)
    selected_rooms = sorted(rooms if rooms is not None else frames_by_room)
    prior_rooms = sorted(frames_by_room)
    unknown_rooms = sorted(set(selected_rooms) - set(frames_by_room))
    if unknown_rooms:
        raise ValueError(f"Requested rooms absent from Area_1: {unknown_rooms}")

    alignment_raw = alignment_path.read_bytes()
    alignment_sha256 = hashlib.sha256(alignment_raw).hexdigest()
    expected_alignment_sha256 = cfg.data.get("bim_alignment_sha256")
    if expected_alignment_sha256 is not None:
        expected_alignment_sha256 = str(expected_alignment_sha256).lower()
        if alignment_sha256 != expected_alignment_sha256:
            raise ValueError(
                "Pinned BIM alignment receipt SHA256 differs from the configured value: "
                f"{alignment_sha256} != {expected_alignment_sha256}"
            )
    alignment = json.loads(alignment_raw.decode("utf-8"))
    if not isinstance(alignment, dict) or not isinstance(alignment.get("rooms"), dict):
        raise TypeError(f"Invalid BIM alignment receipt: {alignment_path}")
    alignment_rooms = alignment["rooms"]
    area_from_bim_by_room = _validate_alignment_receipt(
        alignment,
        prior_rooms,
        ifc_paths,
    )
    global_scene, global_geometry = build_global_ifc_envelope_scene(
        {room: ifc_paths[room] for room in prior_rooms},
        area_from_bim_by_room,
        included_categories=GLOBAL_CORE_CATEGORIES,
    )
    global_bim_identity = {
        "schema_version": 1,
        "filter_policy": global_geometry.audit["filter_policy"],
        "coordinate_frame": global_geometry.audit["coordinate_frame"],
        "included_categories": global_geometry.audit["included_categories"],
        "excluded_envelope_categories": global_geometry.audit["excluded_envelope_categories"],
        "rooms": {
            room: {
                "source_ifc_sha256": global_geometry.audit["room_sources"][room][
                    "source_ifc_sha256"
                ],
                "T_area_from_bim": area_from_bim_by_room[room].tolist(),
            }
            for room in prior_rooms
        },
    }
    global_bim_fingerprint = _canonical_sha256(global_bim_identity)

    target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
    min_depth = float(cfg.data.min_depth)
    max_depth = float(cfg.data.max_depth)
    label_lut = semantic_label_lut(labels_path)
    labels_sha256 = _sha256(labels_path)
    provider = DA3PredictionProvider(cfg, "area_1", prediction_cache)
    records: list[dict[str, Any]] = []
    room_audits: dict[str, Any] = {}

    for room_index, room in enumerate(selected_rooms, start=1):
        alignment_room = alignment_rooms.get(room)
        if not isinstance(alignment_room, dict):
            raise TypeError(f"Alignment receipt has no room mapping for {room!r}")
        area_from_bim = area_from_bim_by_room[room]
        room_audits[room] = {
            "ifc_envelope": global_geometry.audit["room_sources"][room]["source_geometry"],
            "alignment": alignment_room,
        }
        room_frames = sorted(frames_by_room[room], key=lambda item: item.key)[::stride]
        if max_frames_per_room is not None:
            room_frames = room_frames[:max_frames_per_room]
        sample_dir = output_root / "samples" / room
        sample_dir.mkdir(parents=True, exist_ok=True)

        for frame_index, frame in enumerate(room_frames, start=1):
            sample_path = sample_dir / f"{frame.key}.npz"
            prediction: DA3Prediction = provider.get_with_provenance(
                frame.rgb_path,
                target_shape,
            )
            fingerprint = _frame_fingerprint(
                frame,
                rgb_sha256=prediction.image_sha256,
                depth_sha256=_sha256(frame.depth_path),
                semantic_sha256=_sha256(frame.semantic_path),
                semantic_labels_sha256=labels_sha256,
                alignment_sha256=alignment_sha256,
                area_from_bim=area_from_bim,
                global_bim_fingerprint=global_bim_fingerprint,
                target_shape=target_shape,
                min_depth=min_depth,
                max_depth=max_depth,
                da3_model=prediction.model_name,
                da3_revision=prediction.model_revision,
                da3_process_res=prediction.process_res,
                da3_cache_sha256=prediction.cache_sha256,
            )
            if sample_path.exists() and not overwrite:
                record = _reuse_record(
                    sample_path,
                    fingerprint,
                    frame,
                    source_root,
                    output_root,
                    target_shape,
                )
                records.append(record)
                print(
                    f"[{room_index}/{len(selected_rooms)} {room} "
                    f"{frame_index}/{len(room_frames)}] reuse {sample_path.name}",
                    flush=True,
                )
                continue

            rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
            if rgb is None:
                raise RuntimeError(f"Cannot read Stanford RGB image: {frame.rgb_path}")
            intrinsic = scaled_frame_intrinsic(frame, rgb.shape[:2], target_shape)
            camera_to_bim = np.linalg.inv(area_from_bim) @ frame.camera_to_area
            bim_depth, bim_normals, bim_category = render_ifc_envelope(
                global_scene,
                global_geometry,
                intrinsic,
                frame.camera_to_area,
                *target_shape,
                max_depth,
            )
            bim_valid = np.isfinite(bim_depth) & (bim_depth >= min_depth)
            bim_category[~bim_valid] = 255
            bim_edge = depth_edges(bim_depth, bim_valid)
            bim_depth[~bim_valid] = 0.0
            bim_normals[:, ~bim_valid] = 0.0
            gt_depth, gt_valid = load_stanford_depth(
                frame.depth_path,
                target_shape,
                min_depth=min_depth,
                max_depth=max_depth,
            )
            semantic_class = load_stanford_semantics(
                frame.semantic_path,
                target_shape,
                label_lut,
            )
            subset_masks = semantic_subset_masks(semantic_class)
            gt_weight = gt_valid.astype(np.float32)
            payload = {
                "sample_schema_version": np.asarray(2, dtype=np.uint16),
                "base_depth": _store_depth(prediction.depth),
                "base_confidence": np.nan_to_num(prediction.confidence).astype(np.float16),
                "bim_depth": _store_depth(bim_depth),
                "bim_valid": bim_valid.astype(np.uint8),
                "bim_normals": bim_normals.astype(np.float16),
                "bim_edge": bim_edge.astype(np.uint8),
                "bim_category": bim_category,
                # GT remains float32: quantizing evaluation targets to float16 can
                # move valid millimetre-scale differences across metric thresholds.
                # BIM/base stay float16 to avoid roughly 10 GiB of extra Area_1 I/O.
                "gt_depth": gt_depth.astype(np.float32),
                "gt_valid": gt_valid.astype(np.uint8),
                "gt_support": gt_valid.astype(np.uint8),
                "gt_weight": gt_weight.astype(np.float16),
                "semantic_class": semantic_class,
                "intrinsic": intrinsic.astype(np.float32),
                "camera_to_bim": camera_to_bim.astype(np.float64),
                "camera_to_area": frame.camera_to_area.astype(np.float64),
                "area_from_bim": area_from_bim.astype(np.float64),
                "bim_scene_coordinate_frame": np.asarray("Stanford Area_1 world"),
                "global_bim_fingerprint_sha256": np.asarray(global_bim_fingerprint),
                "da3_cache_artifact_sha256": np.asarray(prediction.cache_sha256),
                "preparation_fingerprint_sha256": np.asarray(fingerprint),
                **{name: mask.astype(np.uint8) for name, mask in subset_masks.items()},
            }
            _atomic_savez_compressed(sample_path, payload)
            gt_valid_pixels = int(gt_valid.sum())
            furniture_pixels = int((subset_masks["furniture_mask"] & gt_valid).sum())
            records.append(
                _manifest_record(
                    frame,
                    sample_path,
                    source_root,
                    output_root,
                    gt_valid_pixels=gt_valid_pixels,
                    furniture_pixels=furniture_pixels,
                    fingerprint=fingerprint,
                    da3_source=prediction.source,
                )
            )
            print(
                f"[{room_index}/{len(selected_rooms)} {room} "
                f"{frame_index}/{len(room_frames)}] {frame.rgb_path.name}: "
                f"GT={gt_valid_pixels}, furniture={furniture_pixels}, "
                f"DA3={prediction.source}",
                flush=True,
            )

    records.sort(key=lambda row: str(row["id"]))
    metadata = {
        "schema_version": 1,
        "dataset": "Stanford2D3DS Area_1 paired with BIMSyn IFC",
        "samples": len(records),
        "rooms": selected_rooms,
        "global_bim_rooms": prior_rooms,
        "source_root": str(source_root),
        "area_root": str(area_root),
        "semantic_labels": str(labels_path),
        "bimsyn_ifc_root": str(ifc_root),
        "alignment_receipt": str(alignment_path),
        "alignment_receipt_sha256": alignment_sha256,
        "target_shape": list(target_shape),
        "depth_protocol_m": [min_depth, max_depth],
        "ground_truth": "official regular-view z-depth uint16/512; 65535 invalid",
        "bim_prior": "Area-fixed core envelope; no doors/windows/furniture/proxy/MEP",
        "bim_scene": "single fixed global Area_1 envelope assembled from all room IFCs",
        "bim_filter_policy": "global-area-fixed-core-envelope-v1",
        "bim_included_categories": list(GLOBAL_CORE_CATEGORIES),
        "bim_excluded_categories": ["door", "window"],
        "global_bim_fingerprint_sha256": global_bim_fingerprint,
        "global_bim_audit": global_geometry.audit,
        "registration": "one geometry-only T_area_from_bim per room; no per-frame fitting",
        "da3_model": str(cfg.data.da3_model),
        "da3_revision": provider.model_revision,
        "da3_local_files_only": provider.local_files_only,
        "da3_process_res": int(cfg.data.da3_process_res),
        "stride": int(stride),
        "max_frames_per_room": max_frames_per_room,
        "room_audits": room_audits,
    }
    return records, metadata


def write_stanford_manifest(
    cfg: Config,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_root = resolve_project_path(cfg, cfg.data.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(records, key=lambda row: str(row["id"]))
    preparation_identity = manifest_preparation_identity(ordered_records)
    if preparation_identity["status"] != "verified":
        raise ValueError("Stanford manifest requires verified per-frame preparation fingerprints")
    metadata = {
        **metadata,
        "manifest_preparation_fingerprint_status": preparation_identity["status"],
        "manifest_preparation_fingerprint_sha256": preparation_identity["fingerprint_sha256"],
    }
    manifest = output_root / "manifest.jsonl"
    manifest_text = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in ordered_records
    )
    _atomic_write_text(manifest, manifest_text)
    metadata_path = output_root / "metadata.json"
    _atomic_write_text(
        metadata_path,
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
    )
    return manifest, metadata_path
