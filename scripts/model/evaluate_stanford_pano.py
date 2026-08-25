#!/usr/bin/env python3
"""Evaluate same-station regular-view fusion on Stanford Area_1 panoramas.

The released panorama depth is opened only after all predictions have been
constructed.  It is therefore an evaluation target, never an input to scale
estimation, view selection, fusion, or photometric weighting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from bim_priorda3.baselines import (
    PREVIOUS_FIXED_PARAMETERS,
    configured_scale_and_local_features,
    estimate_robust_bim_scale,
    previous_local_correction_features,
    resolve_scale_estimator_config,
)
from bim_priorda3.checkpoints import (
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_model_config,
)
from bim_priorda3.config import Config, load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.pano_tangent import (
    TangentView,
    build_pano_tangent_preset,
    tangent_z_to_erp_range,
)
from bim_priorda3.data.stanford_pano import (
    StanfordPanorama,
    discover_stanford_panoramas,
    pano_range_to_regular_projection,
)
from bim_priorda3.engine import move_batch, seed_everything
from bim_priorda3.models import BIMPriorDA3
from bim_priorda3.scale_protocol import validate_universal_scale_protocol

DEPTH_METHODS = (
    "raw_da3",
    "universal_scale",
    "bim_direct",
    "station_bim_scale",
    "station_bim_direct",
)
LEARNED_METHOD = "learned_refined"
FUSION_METHODS = (
    "single_best_view",
    "joint_weighted_log",
    "joint_huber",
    "joint_photo_huber",
    "joint_synchronized_huber",
)
STRICT_FUSION_METHODS = ("strict_single_frame", *FUSION_METHODS[1:])
METRIC_NAMES = ("abs_rel", "mae", "rmse", "delta1", "delta2", "delta3")
_PANO_MODALITIES = ("rgb", "depth", "semantic", "pose")
TANGENT_MANIFEST_SCHEMA_VERSION = 1
TANGENT_MANIFEST_PROTOCOL = "stanford-area1-pano-tangent-da3-cache-v1"
TANGENT_CACHE_SCHEMA_VERSION = 2
TANGENT_VARIANTS = {"tangent6": 6, "tangent14": 14}
TANGENT_FUSION_METHODS = (
    "single_best_view",
    "joint_weighted_log",
    "joint_huber",
    "joint_synchronized_huber",
)
COMBINED_FUSION_METHODS = TANGENT_FUSION_METHODS[1:]


@dataclass
class RegularFrame:
    sample_id: str
    room: str
    camera_uuid: str
    intrinsic: np.ndarray
    camera_to_area: np.ndarray
    rgb: np.ndarray
    base_confidence: np.ndarray
    predictions: dict[str, np.ndarray]
    model_arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class ProjectedView:
    """One regular view sampled on a sparse subset of an ERP grid."""

    frame_id: str
    indices: np.ndarray
    base_weights: np.ndarray
    photo_weights: np.ndarray
    log_ranges: Mapping[str, np.ndarray]

    def validate(self, depth_methods: Sequence[str], pixel_count: int) -> None:
        size = int(self.indices.size)
        if self.indices.ndim != 1 or self.indices.dtype.kind not in "iu":
            raise ValueError(f"{self.frame_id}: projected indices must be a 1-D integer array")
        if size and (int(self.indices.min()) < 0 or int(self.indices.max()) >= pixel_count):
            raise ValueError(f"{self.frame_id}: projected index lies outside the ERP grid")
        if size > 1 and np.any(self.indices[1:] <= self.indices[:-1]):
            raise ValueError(
                f"{self.frame_id}: projected indices must be strictly sorted and unique"
            )
        for name, values in {
            "base_weights": self.base_weights,
            "photo_weights": self.photo_weights,
            **dict(self.log_ranges),
        }.items():
            if values.shape != (size,):
                raise ValueError(
                    f"{self.frame_id}: {name} shape {values.shape} does not match {size} hits"
                )
            if not np.isfinite(values).all():
                raise ValueError(f"{self.frame_id}: {name} contains non-finite values")
        if np.any(self.base_weights <= 0) or np.any(self.photo_weights <= 0):
            raise ValueError(f"{self.frame_id}: fusion weights must be strictly positive")
        if set(self.log_ranges) != set(depth_methods):
            raise ValueError(
                f"{self.frame_id}: projected depth methods differ: "
                f"expected={sorted(depth_methods)}, actual={sorted(self.log_ranges)}"
            )


@dataclass(frozen=True)
class FusionResult:
    predictions: Mapping[str, Mapping[str, np.ndarray]]
    contributor_count: np.ndarray
    synchronization: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class TangentManifestBundle:
    """Validated immutable inputs for the optional panorama-tangent route."""

    path: Path
    sha256: str
    payload: Mapping[str, Any]
    views: tuple[TangentView, ...]
    stations: Mapping[str, Mapping[str, Any]]
    pano_only_station_ids: tuple[str, ...]
    tangent_rgb_paths: Mapping[tuple[str, int], Path]
    cache_paths: Mapping[tuple[str, int], Path]


@dataclass(frozen=True)
class TangentPrepared:
    """Prediction-only tangent products frozen before official pano GT is opened."""

    projected_views: tuple[ProjectedView, ...]
    fusions: Mapping[str, FusionResult]


class MetricTotals:
    """Accumulate exact pixel-micro depth metrics, with optional area weights."""

    def __init__(self) -> None:
        self.count = 0
        self.weight_sum = 0.0
        self.abs_rel_sum = 0.0
        self.mae_sum = 0.0
        self.squared_error_sum = 0.0
        self.delta1_sum = 0.0
        self.delta2_sum = 0.0
        self.delta3_sum = 0.0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        support: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        if prediction.shape != target.shape or support.shape != target.shape:
            raise ValueError("Prediction, target, and fixed support shapes must match")
        if weights is None:
            weights = np.ones(target.shape, dtype=np.float64)
        if weights.shape != target.shape:
            raise ValueError("Metric weight shape differs from the fixed support")
        invalid_target = support & (~np.isfinite(target) | (target <= 0))
        invalid_prediction = support & (~np.isfinite(prediction) | (prediction <= 0))
        if np.any(invalid_target):
            raise RuntimeError("Fixed panorama support contains an invalid GT range")
        if np.any(invalid_prediction):
            raise RuntimeError("Prediction is invalid on the fixed panorama support")
        pred = prediction[support].astype(np.float64, copy=False)
        gt = target[support].astype(np.float64, copy=False)
        sample_weights = weights[support].astype(np.float64, copy=False)
        if not pred.size:
            return
        if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
            raise RuntimeError("Metric weights must be finite and positive on support")
        error = pred - gt
        ratio = np.maximum(pred / gt, gt / pred)
        self.count += int(pred.size)
        self.weight_sum += float(sample_weights.sum())
        self.abs_rel_sum += float(np.sum(sample_weights * np.abs(error) / gt))
        self.mae_sum += float(np.sum(sample_weights * np.abs(error)))
        self.squared_error_sum += float(np.sum(sample_weights * np.square(error)))
        self.delta1_sum += float(np.sum(sample_weights * (ratio < 1.25)))
        self.delta2_sum += float(np.sum(sample_weights * (ratio < 1.25**2)))
        self.delta3_sum += float(np.sum(sample_weights * (ratio < 1.25**3)))

    def compute(self) -> dict[str, float | int]:
        if self.count == 0 or self.weight_sum <= 0:
            return {**{name: float("nan") for name in METRIC_NAMES}, "count": 0}
        return {
            "abs_rel": self.abs_rel_sum / self.weight_sum,
            "mae": self.mae_sum / self.weight_sum,
            "rmse": math.sqrt(self.squared_error_sum / self.weight_sum),
            "delta1": self.delta1_sum / self.weight_sum,
            "delta2": self.delta2_sum / self.weight_sum,
            "delta3": self.delta3_sum / self.weight_sum,
            "count": self.count,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_manifest_artifact(
    recorded_path: str | Path,
    *,
    expected_sha256: str,
    preset_root: Path,
    fallback_relative_path: Path,
    context: str,
) -> Path:
    """Resolve a content-addressed artifact after a workspace/data relocation."""

    recorded = Path(recorded_path).expanduser()
    candidates = [recorded if recorded.is_absolute() else preset_root / recorded]
    parts = recorded.parts
    matching_roots = [index for index, part in enumerate(parts) if part == preset_root.name]
    if matching_roots:
        candidates.append(preset_root.joinpath(*parts[matching_roots[-1] + 1 :]))
    candidates.append(preset_root / fallback_relative_path)
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(resolved)
    mismatches = []
    for candidate in unique_candidates:
        if not candidate.is_file():
            continue
        actual_sha256 = _sha256(candidate)
        if actual_sha256 == expected_sha256:
            return candidate
        mismatches.append(f"{candidate} ({actual_sha256})")
    detail = f"; SHA mismatches={mismatches}" if mismatches else ""
    raise FileNotFoundError(
        f"{context}: no content-identical relocated artifact found; "
        f"expected_sha256={expected_sha256}, candidates={[str(path) for path in unique_candidates]}"
        f"{detail}"
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a JSON object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} must be a JSON array")
    return value


def _view_geometry_payload(view: TangentView, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "name": view.name,
        "yaw_degrees": float(view.spec.yaw_degrees),
        "pitch_degrees": float(view.spec.pitch_degrees),
        "roll_degrees": float(view.spec.roll_degrees),
        "horizontal_fov_degrees": float(view.spec.horizontal_fov_degrees),
        "image_shape": [int(value) for value in view.image_shape],
        "intrinsic": view.intrinsic.tolist(),
        "T_face_from_pano": view.T_face_from_pano.tolist(),
    }


def _geometry_fingerprint(geometry: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(geometry),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npz_scalar(item: Mapping[str, np.ndarray], key: str, context: str) -> Any:
    value = np.asarray(item[key])
    if value.ndim != 0:
        raise ValueError(f"{context}: NPZ {key!r} must be scalar")
    return value.item()


def _validate_tangent_cache_metadata(
    cache_record: Mapping[str, Any],
    *,
    tangent_sha256: str,
    model: Mapping[str, Any],
    image_shape: tuple[int, int],
    context: str,
    resolved_path: Path | None = None,
) -> Path:
    required_record = {
        "path",
        "sha256",
        "image_sha256",
        "model_name",
        "model_revision",
        "process_res",
        "target_shape",
        "provenance_status",
        "depth_quantity",
    }
    if set(cache_record) != required_record:
        raise ValueError(
            f"{context}: DA3 cache record keys differ; "
            f"missing={sorted(required_record - set(cache_record))}, "
            f"extra={sorted(set(cache_record) - required_record)}"
        )
    if cache_record["depth_quantity"] != "perspective_z_depth_m":
        raise ValueError(f"{context}: tangent cache must contain perspective z-depth")
    if cache_record["provenance_status"] != "direct_inference":
        raise ValueError(f"{context}: formal tangent cache must come from direct inference")
    expected_metadata = {
        "image_sha256": tangent_sha256,
        "model_name": str(model["name"]),
        "model_revision": str(model["revision"]),
        "process_res": int(model["process_res"]),
        "target_shape": list(image_shape),
    }
    for key, expected in expected_metadata.items():
        if cache_record.get(key) != expected:
            raise ValueError(
                f"{context}: cache manifest {key} differs: "
                f"expected={expected!r}, actual={cache_record.get(key)!r}"
            )
    path = (
        resolved_path.resolve()
        if resolved_path is not None
        else Path(str(cache_record["path"])).expanduser().resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(f"{context}: tangent DA3 cache is missing: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != cache_record["sha256"]:
        raise RuntimeError(
            f"{context}: tangent DA3 cache SHA differs: "
            f"expected={cache_record['sha256']}, actual={actual_sha256}"
        )
    required_npz = {
        "schema_version",
        "depth",
        "confidence",
        "image_sha256",
        "model_name",
        "model_revision",
        "process_res",
        "target_shape",
        "provenance_status",
        "local_files_only",
    }
    with np.load(path, allow_pickle=False) as item:
        if set(item.files) != required_npz:
            raise ValueError(
                f"{context}: cache NPZ keys differ; "
                f"missing={sorted(required_npz - set(item.files))}, "
                f"extra={sorted(set(item.files) - required_npz)}"
            )
        if int(_npz_scalar(item, "schema_version", context)) != TANGENT_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{context}: unsupported tangent cache schema")
        if item["depth"].shape != image_shape or item["confidence"].shape != image_shape:
            raise ValueError(f"{context}: tangent cache array shape differs from preset")
        if item["depth"].dtype.kind != "f" or item["confidence"].dtype.kind != "f":
            raise ValueError(f"{context}: tangent depth/confidence must be floating point")
        if item["target_shape"].shape != (2,) or item["target_shape"].dtype.kind not in "iu":
            raise ValueError(f"{context}: target_shape must be a two-integer vector")
        if item["local_files_only"].shape != () or item["local_files_only"].dtype.kind != "b":
            raise ValueError(f"{context}: local_files_only must be scalar boolean")
        npz_metadata = {
            "image_sha256": str(_npz_scalar(item, "image_sha256", context)),
            "model_name": str(_npz_scalar(item, "model_name", context)),
            "model_revision": str(_npz_scalar(item, "model_revision", context)),
            "process_res": int(_npz_scalar(item, "process_res", context)),
            "target_shape": [int(value) for value in item["target_shape"].tolist()],
            "provenance_status": str(_npz_scalar(item, "provenance_status", context)),
            "local_files_only": bool(_npz_scalar(item, "local_files_only", context)),
        }
    expected_npz = {
        **expected_metadata,
        "provenance_status": str(cache_record["provenance_status"]),
        "local_files_only": bool(model["local_files_only"]),
    }
    if npz_metadata != expected_npz:
        raise ValueError(
            f"{context}: cache NPZ provenance differs: "
            f"expected={expected_npz!r}, actual={npz_metadata!r}"
        )
    return path


def _validate_tangent_manifest(
    path: Path,
    *,
    cfg: Config,
    dataset: BIMDepthDataset,
    split: str,
    confirm_test: bool,
    split_panoramas: Sequence[StanfordPanorama],
) -> TangentManifestBundle:
    """Validate every identity and immutable artifact before Route-P evaluation."""

    manifest_path = path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tangent manifest is missing: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot parse tangent manifest {manifest_path}: {error}") from error
    manifest = _mapping(payload, "tangent manifest")
    if manifest.get("schema_version") != TANGENT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported tangent manifest schema_version")
    if manifest.get("protocol") != TANGENT_MANIFEST_PROTOCOL:
        raise ValueError("Unsupported tangent manifest protocol")
    if manifest.get("split") != split:
        raise ValueError(
            f"Tangent manifest split differs: expected={split!r}, actual={manifest.get('split')!r}"
        )
    if split == "test":
        if not confirm_test or manifest.get("test_access_explicitly_authorized") is not True:
            raise ValueError(
                "Test tangent evaluation requires both runtime and cache authorization"
            )
    elif manifest.get("test_access_explicitly_authorized") is not False:
        raise ValueError("Validation tangent manifest must not claim test authorization")

    selection = _mapping(manifest.get("selection"), "tangent manifest selection")
    if selection.get("formal_protocol_eligible") is not True:
        raise ValueError("Route-P requires a formal full-split tangent manifest")
    if selection.get("max_stations") is not None:
        raise ValueError("Route-P rejects exploratory max-stations tangent manifests")
    if selection.get("room_source") != (
        "configured exhaustive split_annotation via BIMDepthDataset"
    ):
        raise ValueError("Tangent manifest room_source is not the registered annotation protocol")

    config_path = Path(str(cfg.config_path)).expanduser().resolve()
    config_record = _mapping(manifest.get("config"), "tangent manifest config")
    if config_record.get("sha256") != _sha256(config_path):
        raise ValueError("Tangent manifest config SHA differs from the active config")
    annotation_path = resolve_project_path(cfg, cfg.data.split_annotation)
    annotation_record = _mapping(
        manifest.get("split_annotation"), "tangent manifest split_annotation"
    )
    if annotation_record.get("sha256") != _sha256(annotation_path):
        raise ValueError("Tangent manifest annotation SHA differs from the active annotation")
    if manifest.get("split_provenance") != dataset.split_provenance:
        raise ValueError("Tangent manifest split provenance differs from the active dataset")

    model = _mapping(manifest.get("model"), "tangent manifest model")
    expected_model = {
        "name": str(cfg.data.da3_model),
        "revision": str(cfg.data.da3_revision),
        "process_res": int(cfg.data.da3_process_res),
        "local_files_only": bool(
            cfg.data.get("da3_local_files_only", cfg.data.get("local_files_only", False))
        ),
    }
    if expected_model["revision"] == "UNPINNED":
        raise ValueError("Route-P requires a pinned DA3 revision")
    if dict(model) != expected_model:
        raise ValueError(
            f"Tangent manifest DA3 identity differs: expected={expected_model!r}, "
            f"actual={dict(model)!r}"
        )
    expected_contract = {
        "pano_rgb_decoded": True,
        "pano_pose_metadata_decoded": True,
        "pano_depth_decoded": False,
        "pano_semantic_decoded": False,
        "regular_ground_truth_decoded": False,
        "output_depth_quantity": "perspective_z_depth_m",
    }
    if manifest.get("input_contract") != expected_contract:
        raise ValueError("Tangent manifest prediction-input contract differs")

    preset = _mapping(manifest.get("preset"), "tangent manifest preset")
    if preset.get("name") != "nested14" or int(preset.get("view_count", -1)) != 14:
        raise ValueError("Route-P v3 requires one formal nested14 manifest")
    preset_geometry = _sequence(preset.get("views"), "tangent manifest preset views")
    if len(preset_geometry) != 14:
        raise ValueError("nested14 manifest must contain exactly 14 preset views")
    first_geometry = _mapping(preset_geometry[0], "tangent preset view 0")
    first_shape = first_geometry.get("image_shape")
    if not isinstance(first_shape, list) or len(first_shape) != 2:
        raise ValueError("Tangent preset image_shape must be [height, width]")
    image_shape = (int(first_shape[0]), int(first_shape[1]))
    if image_shape[0] != image_shape[1] or image_shape[0] < 1:
        raise ValueError("Route-P requires positive square tangent views")
    views = build_pano_tangent_preset("nested14", image_shape[0])
    expected_geometry = [_view_geometry_payload(view, index) for index, view in enumerate(views)]
    if list(preset_geometry) != expected_geometry:
        raise ValueError("Tangent preset K/T/spec differs from the shared pano_tangent rebuild")
    cubemap = build_pano_tangent_preset("cubemap6", image_shape[0])
    cubemap_geometry = [_view_geometry_payload(view, index) for index, view in enumerate(cubemap)]
    if expected_geometry[:6] != cubemap_geometry:
        raise AssertionError("Shared nested14 first six views no longer equal cubemap6")
    fingerprint = _geometry_fingerprint(expected_geometry)
    if preset.get("geometry_fingerprint_sha256") != fingerprint:
        raise ValueError("Tangent preset geometry fingerprint differs")
    expected_namespace = f"nested14_r{image_shape[0]}_{fingerprint[:12]}"
    if preset.get("cache_namespace") != expected_namespace:
        raise ValueError("Tangent preset cache namespace differs")

    runtime_by_id = {station.camera_uuid: station for station in split_panoramas}
    if len(runtime_by_id) != len(split_panoramas):
        raise RuntimeError("Runtime split contains duplicate panorama camera_uuid values")
    expected_ids = [
        station.camera_uuid
        for station in sorted(split_panoramas, key=lambda item: (item.room, item.camera_uuid))
    ]
    selected_ids = selection.get("selected_station_ids")
    if selected_ids != expected_ids:
        raise ValueError("Tangent manifest station IDs/order differ from the full runtime split")
    expected_count = len(expected_ids)
    for key in (
        "selected_station_count",
        "split_pano_station_count",
        "full_split_station_count_before_exploratory_limit",
    ):
        if int(selection.get(key, -1)) != expected_count:
            raise ValueError(f"Tangent manifest selection {key} differs from runtime split")
    runtime_regular_ids = {str(record["camera_uuid"]) for record in dataset.records}
    if int(selection.get("annotation_regular_station_count", -1)) != len(runtime_regular_ids):
        raise ValueError("Tangent manifest annotation regular-station count differs")
    expected_pano_only = sorted(set(expected_ids) - runtime_regular_ids)
    if selection.get("pano_only_station_ids") != expected_pano_only:
        raise ValueError("Tangent manifest pano-only IDs differ from runtime discovery")
    if int(selection.get("shared_regular_pano_station_count", -1)) != (
        expected_count - len(expected_pano_only)
    ):
        raise ValueError("Tangent manifest shared station count differs")
    if sorted(selection.get("rooms", [])) != sorted({station.room for station in split_panoramas}):
        raise ValueError("Tangent manifest room set differs from runtime split")

    station_records = _sequence(manifest.get("stations"), "tangent manifest stations")
    if len(station_records) != expected_count:
        raise ValueError("Tangent manifest station record count differs")
    preset_root = manifest_path.parent.parent
    station_lookup: dict[str, Mapping[str, Any]] = {}
    tangent_rgb_paths: dict[tuple[str, int], Path] = {}
    cache_paths: dict[tuple[str, int], Path] = {}
    for station_index, raw_station_record in enumerate(station_records):
        station_record = _mapping(raw_station_record, f"tangent station {station_index}")
        camera_uuid = str(station_record.get("camera_uuid", ""))
        if camera_uuid != expected_ids[station_index] or camera_uuid in station_lookup:
            raise ValueError("Tangent manifest station order/identity is malformed")
        runtime_station = runtime_by_id[camera_uuid]
        if station_record.get("room") != runtime_station.room:
            raise ValueError(f"{camera_uuid}: tangent manifest room differs")
        pano_rgb_record = _mapping(station_record.get("pano_rgb"), f"{camera_uuid}: pano_rgb")
        runtime_pano_sha = _sha256(runtime_station.rgb_path)
        if pano_rgb_record.get("sha256") != runtime_pano_sha:
            raise RuntimeError(f"{camera_uuid}: panorama RGB SHA differs from tangent manifest")

        tangent_records = _sequence(
            station_record.get("tangent_views"), f"{camera_uuid}: tangent_views"
        )
        if len(tangent_records) != len(views):
            raise ValueError(f"{camera_uuid}: tangent view count differs from nested14")
        for view_index, raw_tangent_record in enumerate(tangent_records):
            context = f"{camera_uuid}/tangent{view_index:02d}"
            tangent_record = _mapping(raw_tangent_record, context)
            if tangent_record.get("view") != expected_geometry[view_index]:
                raise ValueError(f"{context}: station-level K/T differs from preset")
            if tangent_record.get("pano_rgb_sha256") != runtime_pano_sha:
                raise ValueError(f"{context}: tangent view is bound to a different panorama")
            tangent_rgb_record = _mapping(tangent_record.get("tangent_rgb"), f"{context}: RGB")
            tangent_sha = str(tangent_rgb_record.get("sha256", ""))
            tangent_path = _resolve_manifest_artifact(
                str(tangent_rgb_record.get("path", "")),
                expected_sha256=tangent_sha,
                preset_root=preset_root,
                fallback_relative_path=(
                    Path("tangent_rgb")
                    / camera_uuid
                    / Path(str(tangent_rgb_record.get("path", ""))).name
                ),
                context=f"{context}: tangent PNG",
            )
            decoded = cv2.imread(str(tangent_path), cv2.IMREAD_UNCHANGED)
            if (
                decoded is None
                or decoded.dtype != np.uint8
                or decoded.ndim != 3
                or decoded.shape != (*image_shape, 3)
            ):
                raise ValueError(f"{context}: tangent PNG decode/shape differs from preset")
            cache_record = _mapping(tangent_record.get("da3_cache"), f"{context}: cache")
            cache_path = _resolve_manifest_artifact(
                str(cache_record.get("path", "")),
                expected_sha256=str(cache_record.get("sha256", "")),
                preset_root=preset_root,
                fallback_relative_path=(
                    Path("da3_cache") / Path(str(cache_record.get("path", ""))).name
                ),
                context=f"{context}: tangent DA3 cache",
            )
            _validate_tangent_cache_metadata(
                cache_record,
                tangent_sha256=tangent_sha,
                model=model,
                image_shape=image_shape,
                context=context,
                resolved_path=cache_path,
            )
            tangent_rgb_paths[(camera_uuid, view_index)] = tangent_path
            cache_paths[(camera_uuid, view_index)] = cache_path
        station_lookup[camera_uuid] = station_record

    return TangentManifestBundle(
        path=manifest_path,
        sha256=_sha256(manifest_path),
        payload=manifest,
        views=tuple(views),
        stations=station_lookup,
        pano_only_station_ids=tuple(expected_pano_only),
        tangent_rgb_paths=tangent_rgb_paths,
        cache_paths=cache_paths,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _unit_interval(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate single-source and multi-view regular-to-panorama depth fusion "
            "on Stanford Area_1"
        )
    )
    parser.add_argument("--config", default="configs/stanford_area1.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--tangent-manifest",
        type=Path,
        help=(
            "Optional formal nested14 panorama-tangent DA3 cache manifest. "
            "Enables the independent Route-P and regular+pano raw-DA3 analyses."
        ),
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required in addition to --split test, preventing accidental test peeking",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--pano-height", type=_positive_int, default=512)
    parser.add_argument(
        "--max-stations",
        type=_positive_int,
        help=(
            "Exploratory ordered station prefix (also applies to Route-P); "
            "any use makes the output non-formal"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=_positive_int, default=10_000)
    parser.add_argument("--centrality-power", type=float, default=4.0)
    parser.add_argument("--confidence-floor", type=_unit_interval, default=0.05)
    parser.add_argument("--huber-log-delta", type=float, default=0.08)
    parser.add_argument("--consistency-log-threshold", type=float, default=0.25)
    parser.add_argument("--photo-sigma", type=float, default=0.12)
    parser.add_argument("--sync-min-overlap", type=_positive_int, default=256)
    parser.add_argument("--sync-pair-max-samples", type=_positive_int, default=4096)
    parser.add_argument("--sync-huber-log-delta", type=float, default=0.08)
    parser.add_argument("--sync-l2", type=float, default=1e-6)
    parser.add_argument("--sync-max-abs-offset", type=float, default=0.50)
    parser.add_argument("--station-scale-samples-per-view", type=_positive_int, default=50_000)
    parser.add_argument("--allow-cross-dataset-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        parser.error("--split test requires the explicit --confirm-test flag")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.pano_height < 2:
        parser.error("--pano-height must be at least two")
    for name in (
        "centrality_power",
        "huber_log_delta",
        "consistency_log_threshold",
        "photo_sigma",
        "sync_huber_log_delta",
        "sync_l2",
        "sync_max_abs_offset",
    ):
        if not np.isfinite(getattr(args, name)) or float(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    return args


def _erp_pixel_grid(height: int, width: int) -> np.ndarray:
    if height < 2 or width < 2:
        raise ValueError("ERP dimensions must both be at least two")
    x, y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    return np.stack((x, y), axis=-1)


def _latitude_area_weights(height: int, width: int) -> np.ndarray:
    latitude_low = np.pi * (np.arange(height, dtype=np.float64) / height - 0.5)
    latitude_high = np.pi * ((np.arange(height, dtype=np.float64) + 1.0) / height - 0.5)
    row_solid_angle = (np.sin(latitude_high) - np.sin(latitude_low)) * (2.0 * np.pi / width)
    return np.broadcast_to(row_solid_angle[:, None], (height, width)).copy()


def _load_pano_rgb(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Cannot read panorama RGB: {path}")
    if raw.ndim != 3 or raw.shape[2] not in (3, 4) or raw.dtype != np.uint8:
        raise ValueError(f"Panorama RGB must be uint8 RGB/RGBA, got {raw.dtype} {raw.shape}")
    bgr = raw[..., :3]
    height, width = target_shape
    if bgr.shape[:2] != target_shape:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    return bgr[..., ::-1].astype(np.float32) / 255.0


def _load_pano_gt(
    path: Path,
    target_shape: tuple[int, int],
    *,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Load official radial depth; callers invoke this only after prediction."""

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Cannot read panorama depth: {path}")
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"Panorama depth must be uint16, got {raw.dtype} {raw.shape}")
    native_valid_count = int(np.count_nonzero(raw != np.uint16(65535)))
    height, width = target_shape
    if raw.shape != target_shape:
        nearest = getattr(cv2, "INTER_NEAREST_EXACT", cv2.INTER_NEAREST)
        raw = cv2.resize(raw, (width, height), interpolation=nearest)
    depth = raw.astype(np.float32) / 512.0
    valid = (
        (raw != np.uint16(65535))
        & np.isfinite(depth)
        & (depth >= float(min_depth))
        & (depth <= float(max_depth))
    )
    depth[~valid] = 0.0
    return depth, valid, native_valid_count


def _station_bim_scale(
    records: Sequence[Mapping[str, Any]],
    cfg: Config,
    *,
    samples_per_view: int,
) -> tuple[float, dict[str, Any]]:
    """Estimate one auditable BIM scale shared by all regular views at a station."""

    parameters = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    if parameters["name"] != "log_upper_cap_v1":
        raise ValueError("Station-scale panorama evaluation requires log_upper_cap_v1")
    sampled_ratios = []
    total_support = 0
    per_view = []
    for record in sorted(records, key=lambda value: (int(value["frame_number"]), value["id"])):
        with np.load(Path(str(record["sample"]))) as item:
            base = item["base_depth"].astype(np.float32).reshape(-1)
            bim = item["bim_depth"].astype(np.float32).reshape(-1)
            bim_valid = item["bim_valid"].reshape(-1) > 0
        valid = bim_valid & np.isfinite(base) & np.isfinite(bim) & (base > 0) & (bim > 0)
        ratio = bim[valid] / base[valid]
        ratio = ratio[
            (ratio > float(parameters["ratio_min"])) & (ratio < float(parameters["ratio_max"]))
        ]
        count = int(ratio.size)
        total_support += count
        if count > samples_per_view:
            selection = np.linspace(0, count - 1, samples_per_view, dtype=np.int64)
            ratio = ratio[selection]
        sampled_ratios.append(ratio.astype(np.float32, copy=False))
        per_view.append(
            {
                "sample_id": str(record["id"]),
                "eligible_ratio_count": count,
                "selected_ratio_count": int(ratio.size),
            }
        )
    pooled = np.concatenate(sampled_ratios) if sampled_ratios else np.empty(0, dtype=np.float32)
    estimate = estimate_robust_bim_scale(
        np.ones(pooled.shape, dtype=np.float32),
        pooled,
        q10_log_cap=float(parameters["q10_log_cap"]),
        q25_log_cap=float(parameters["q25_log_cap"]),
        ratio_min=float(parameters["ratio_min"]),
        ratio_max=float(parameters["ratio_max"]),
        min_samples=int(parameters["min_samples"]),
    )
    return float(estimate.scale), {
        "estimator": estimate.estimator,
        "scale": float(estimate.scale),
        "eligible_ratio_count": total_support,
        "selected_ratio_count": int(pooled.size),
        "samples_per_view_cap": int(samples_per_view),
        "selection": "equal per-view cap; evenly spaced valid raster-order indices",
        "fallback": bool(estimate.fallback),
        "quantiles": [[float(q), float(value)] for q, value in estimate.quantiles],
        "q10_cap_triggered": bool(estimate.q10_cap_triggered),
        "q25_cap_triggered": bool(estimate.q25_cap_triggered),
        "per_view": per_view,
    }


def _read_regular_frame(
    record: Mapping[str, Any],
    cfg: Config,
    depth_methods: Sequence[str],
    *,
    station_scale: float,
) -> RegularFrame:
    sample_path = Path(str(record["sample"]))
    if not sample_path.is_file():
        raise FileNotFoundError(f"Prepared Stanford sample is missing: {sample_path}")
    with np.load(sample_path) as item:
        required = {
            "base_depth",
            "base_confidence",
            "bim_depth",
            "bim_valid",
            "bim_normals",
            "bim_edge",
            "intrinsic",
            "camera_to_area",
        }
        missing = sorted(required - set(item.files))
        if missing:
            raise RuntimeError(f"{record['id']}: prepared sample lacks {missing}")
        base = item["base_depth"].astype(np.float32)
        confidence = item["base_confidence"].astype(np.float32)
        bim_valid = item["bim_valid"].astype(np.float32)
        bim = item["bim_depth"].astype(np.float32)
        bim = np.where(bim_valid > 0, bim, 0.0).astype(np.float32)
        normals = item["bim_normals"].astype(np.float32)
        edge = item["bim_edge"].astype(np.float32)
        intrinsic = item["intrinsic"].astype(np.float64)
        camera_to_area = item["camera_to_area"].astype(np.float64)
    if base.ndim != 2 or confidence.shape != base.shape or bim.shape != base.shape:
        raise ValueError(f"{record['id']}: prepared depth/confidence shapes are invalid")
    if normals.shape != (3, *base.shape) or edge.shape != base.shape:
        raise ValueError(f"{record['id']}: prepared BIM feature shapes are invalid")
    if intrinsic.shape != (3, 3) or camera_to_area.shape != (4, 4):
        raise ValueError(f"{record['id']}: prepared camera matrices are invalid")
    if np.any(~np.isfinite(base)) or np.any(base <= 0):
        raise RuntimeError(f"{record['id']}: cached DA3 depth must be finite and positive")
    if not np.isfinite(confidence).all():
        raise RuntimeError(f"{record['id']}: cached DA3 confidence contains non-finite values")

    scaled, direct, _, _, _ = configured_scale_and_local_features(
        base,
        bim,
        cfg.model.get("scale_estimator"),
    )
    predictions = {
        "raw_da3": base,
        "universal_scale": scaled,
        "bim_direct": direct,
    }
    station_scaled = (base * float(station_scale)).astype(np.float32)
    parameters = PREVIOUS_FIXED_PARAMETERS
    station_field, _ = previous_local_correction_features(
        station_scaled,
        bim,
        consistency=float(parameters["consistency_log_threshold"]),
        sigma=float(parameters["smoothing_sigma"]),
    )
    predictions.update(
        station_bim_scale=station_scaled,
        station_bim_direct=(
            station_scaled * np.exp(float(parameters["local_correction_alpha"]) * station_field)
        ).astype(np.float32),
    )
    if set(depth_methods) not in (set(DEPTH_METHODS), {*DEPTH_METHODS, LEARNED_METHOD}):
        raise ValueError(f"Unsupported depth method set: {sorted(depth_methods)}")

    image_path = Path(str(record["image"]))
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read regular RGB image: {image_path}")
    height, width = base.shape
    if bgr.shape[:2] != base.shape:
        bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    return RegularFrame(
        sample_id=str(record["id"]),
        room=str(record["region"]),
        camera_uuid=str(record["camera_uuid"]),
        intrinsic=intrinsic,
        camera_to_area=camera_to_area,
        rgb=rgb,
        base_confidence=np.clip(confidence, 0.0, 1.0),
        predictions=predictions,
        model_arrays={
            "bim_depth": bim,
            "bim_valid": bim_valid,
            "bim_normals": normals,
            "bim_edge": edge,
        },
    )


def _regular_model_batch(frames: Sequence[RegularFrame]) -> dict[str, torch.Tensor]:
    def stack(name: str) -> torch.Tensor:
        values = [frame.model_arrays[name] for frame in frames]
        arrays = np.stack(values, axis=0)
        if name != "bim_normals":
            arrays = arrays[:, None]
        return torch.from_numpy(arrays.astype(np.float32, copy=False))

    return {
        "rgb": torch.from_numpy(
            np.stack([frame.rgb.transpose(2, 0, 1) for frame in frames]).astype(np.float32)
        ),
        "base_depth": torch.from_numpy(
            np.stack([frame.predictions["raw_da3"] for frame in frames])[:, None]
        ),
        "base_confidence": torch.from_numpy(
            np.stack([frame.base_confidence for frame in frames])[:, None]
        ),
        "scaled_depth": torch.from_numpy(
            np.stack([frame.predictions["universal_scale"] for frame in frames])[:, None]
        ),
        "bim_depth": stack("bim_depth"),
        "bim_valid": stack("bim_valid"),
        "bim_normals": stack("bim_normals"),
        "bim_edge": stack("bim_edge"),
    }


def _photo_consistency_weight(
    projected_rgb: np.ndarray,
    pano_rgb: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, float]:
    """Return exposure-normalized RGB agreement without using panorama depth."""

    if projected_rgb.shape != pano_rgb.shape or projected_rgb.shape[:2] != valid.shape:
        raise ValueError("Photometric inputs must share one HxW grid")
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Photometric sigma must be finite and positive")
    source_luma = np.mean(projected_rgb, axis=-1)
    pano_luma = np.mean(pano_rgb, axis=-1)
    calibration = valid & (source_luma > 0.02) & (pano_luma > 0.02)
    if int(np.count_nonzero(calibration)) >= 64:
        log_gain = np.median(
            np.log(pano_luma[calibration] + 1e-3) - np.log(source_luma[calibration] + 1e-3)
        )
        gain = float(np.clip(np.exp(log_gain), 0.5, 2.0))
    else:
        gain = 1.0
    corrected = np.clip(projected_rgb * gain, 0.0, 1.0)
    residual = np.mean(np.abs(corrected - pano_rgb), axis=-1)
    weight = np.exp(-residual / float(sigma))
    weight = np.clip(weight, 1e-3, 1.0).astype(np.float32)
    weight[~valid] = 1e-3
    return weight, gain


def _project_regular_view(
    frame: RegularFrame,
    station: StanfordPanorama,
    pano_pixels: np.ndarray,
    pano_rgb: np.ndarray,
    depth_methods: Sequence[str],
    *,
    centrality_power: float,
    confidence_floor: float,
    photo_sigma: float,
    center_tolerance_m: float = 5e-3,
) -> ProjectedView:
    if frame.camera_uuid != station.camera_uuid or frame.room != station.room:
        raise ValueError(f"{frame.sample_id}: regular/panorama station metadata differs")
    if pano_pixels.shape[:2] != pano_rgb.shape[:2] or pano_pixels.shape[-1] != 2:
        raise ValueError("ERP pixels and panorama RGB must share one HxW grid")
    regular_center = frame.camera_to_area[:3, 3]
    pano_center = station.camera_to_area[:3, 3]
    center_error = float(np.linalg.norm(regular_center - pano_center))
    if center_error > center_tolerance_m:
        raise RuntimeError(
            f"{frame.sample_id}: same-UUID camera centers differ by {center_error:.6f} m; "
            "center-only spherical reprojection would be invalid"
        )

    regular_pixels, unit_z, front_facing = pano_range_to_regular_projection(
        pano_pixels,
        np.ones(pano_pixels.shape[:2], dtype=np.float64),
        pano_pixels.shape[:2],
        station.camera_to_area,
        np.linalg.inv(frame.camera_to_area),
        frame.intrinsic,
    )
    ray_z = unit_z.astype(np.float32)
    source_height, source_width = frame.base_confidence.shape
    map_x = regular_pixels[..., 0]
    map_y = regular_pixels[..., 1]
    geometry_valid = (
        front_facing
        & (ray_z > 1e-4)
        & (map_x >= 0.0)
        & (map_x <= source_width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= source_height - 1.0)
    )
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    sampled_confidence = cv2.remap(
        frame.base_confidence,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    centrality = np.clip(ray_z, 0.0, 1.0) ** float(centrality_power)
    confidence = confidence_floor + (1.0 - confidence_floor) * np.clip(sampled_confidence, 0.0, 1.0)
    base_weight = centrality * confidence
    geometry_valid &= base_weight > 0

    projected_rgb = cv2.remap(
        frame.rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    photo_weight, _ = _photo_consistency_weight(
        projected_rgb,
        pano_rgb,
        geometry_valid,
        sigma=photo_sigma,
    )

    log_range_images: dict[str, np.ndarray] = {}
    for method in depth_methods:
        if method not in frame.predictions:
            raise ValueError(f"{frame.sample_id}: missing depth method {method!r}")
        sampled_z = cv2.remap(
            frame.predictions[method],
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        radial_depth = sampled_z / np.maximum(ray_z, 1e-8)
        invalid = geometry_valid & (~np.isfinite(radial_depth) | (radial_depth <= 0))
        if np.any(invalid):
            raise RuntimeError(
                f"{frame.sample_id}: {method} has {int(np.count_nonzero(invalid))} "
                "invalid values on geometrically visible panorama rays"
            )
        log_range_images[method] = np.log(np.maximum(radial_depth, 1e-8)).astype(np.float32)

    indices = np.flatnonzero(geometry_valid.reshape(-1)).astype(np.int64)
    return ProjectedView(
        frame_id=frame.sample_id,
        indices=indices,
        base_weights=base_weight.reshape(-1)[indices].astype(np.float32),
        photo_weights=photo_weight.reshape(-1)[indices].astype(np.float32),
        log_ranges={
            method: values.reshape(-1)[indices].astype(np.float32)
            for method, values in log_range_images.items()
        },
    )


def _load_tangent_projected_views(
    bundle: TangentManifestBundle,
    camera_uuid: str,
    target_shape: tuple[int, int],
    *,
    centrality_power: float,
) -> tuple[ProjectedView, ...]:
    """Project cached tangent z-depth/confidence to radial ERP contributions.

    All projection geometry comes from :mod:`bim_priorda3.data.pano_tangent`.
    Projecting an all-ones z image gives ``1 / face_z``; this lets us recover
    face centrality and projected confidence without duplicating spherical
    camera equations in the evaluator.
    """

    station_record = bundle.stations.get(camera_uuid)
    if station_record is None:
        raise KeyError(f"Tangent manifest has no station {camera_uuid}")
    tangent_records = _sequence(
        station_record.get("tangent_views"), f"{camera_uuid}: tangent_views"
    )
    if len(tangent_records) != len(bundle.views):
        raise RuntimeError(f"{camera_uuid}: validated tangent view count changed")
    pixel_count = int(np.prod(target_shape))
    outputs = []
    for view_index, view in enumerate(bundle.views):
        cache_path = bundle.cache_paths[(camera_uuid, view_index)]
        with np.load(cache_path, allow_pickle=False) as item:
            depth = item["depth"].astype(np.float32)
            confidence = item["confidence"].astype(np.float32)
        if depth.shape != view.image_shape or confidence.shape != view.image_shape:
            raise RuntimeError(f"{camera_uuid}/{view.name}: tangent cache shape changed")
        depth_valid = np.isfinite(depth) & (depth > 0.0)
        if not np.all(depth_valid):
            raise RuntimeError(
                f"{camera_uuid}/{view.name}: tangent DA3 depth must be finite and positive"
            )
        if not np.isfinite(confidence).all() or np.any((confidence < 0.0) | (confidence > 1.0)):
            raise RuntimeError(
                f"{camera_uuid}/{view.name}: tangent DA3 confidence must lie in [0,1]"
            )
        range_image, range_valid = tangent_z_to_erp_range(
            depth,
            view,
            target_shape,
            valid_mask=depth_valid,
        )
        unit_range, unit_valid = tangent_z_to_erp_range(
            np.ones(view.image_shape, dtype=np.float32),
            view,
            target_shape,
            valid_mask=depth_valid,
        )
        confidence_range, projected_confidence_valid = tangent_z_to_erp_range(
            np.maximum(confidence, np.finfo(np.float32).tiny),
            view,
            target_shape,
            # A zero confidence is a real zero-weight observation, not a
            # missing sample.  Keeping it in bilinear interpolation prevents
            # nearby positive values from being renormalized upward.
            valid_mask=depth_valid,
        )
        valid = range_valid & unit_valid & projected_confidence_valid
        centrality = np.zeros(target_shape, dtype=np.float32)
        projected_confidence = np.zeros(target_shape, dtype=np.float32)
        np.divide(1.0, unit_range, out=centrality, where=valid)
        np.divide(
            confidence_range,
            unit_range,
            out=projected_confidence,
            where=valid,
        )
        centrality = np.clip(centrality, 0.0, 1.0)
        projected_confidence = np.clip(projected_confidence, 0.0, 1.0)
        projected_confidence[projected_confidence <= 2.0 * np.finfo(np.float32).tiny] = 0.0
        base_weight = np.power(centrality, float(centrality_power)) * projected_confidence
        valid &= (
            np.isfinite(range_image)
            & (range_image > 0.0)
            & np.isfinite(base_weight)
            & (base_weight > 0.0)
        )
        indices = np.flatnonzero(valid.reshape(-1)).astype(np.int64)
        if not indices.size:
            raise RuntimeError(f"{camera_uuid}/{view.name}: tangent projection has no ERP support")
        output = ProjectedView(
            frame_id=f"tangent/{view_index:02d}_{view.name}",
            indices=indices,
            base_weights=base_weight.reshape(-1)[indices].astype(np.float32),
            photo_weights=np.ones(indices.size, dtype=np.float32),
            log_ranges={"raw_da3": np.log(range_image.reshape(-1)[indices]).astype(np.float32)},
        )
        output.validate(["raw_da3"], pixel_count)
        outputs.append(output)
    return tuple(outputs)


def _raw_only_view(view: ProjectedView) -> ProjectedView:
    if "raw_da3" not in view.log_ranges:
        raise ValueError(f"{view.frame_id}: raw_da3 is missing")
    return ProjectedView(
        frame_id=view.frame_id,
        indices=view.indices,
        base_weights=view.base_weights,
        photo_weights=view.photo_weights,
        log_ranges={"raw_da3": view.log_ranges["raw_da3"]},
    )


def _weighted_log_mean(
    views: Sequence[ProjectedView],
    depth_methods: Sequence[str],
    pixel_count: int,
    *,
    use_photo: bool,
    offsets: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    denominator = np.zeros(pixel_count, dtype=np.float64)
    numerators = {method: np.zeros(pixel_count, dtype=np.float64) for method in depth_methods}
    for view_index, view in enumerate(views):
        weight = view.base_weights.astype(np.float64)
        if use_photo:
            weight = weight * view.photo_weights
        denominator[view.indices] += weight
        for method in depth_methods:
            offset = 0.0 if offsets is None else float(offsets[method][view_index])
            numerators[method][view.indices] += weight * (view.log_ranges[method] + offset)
    means = {
        method: np.divide(
            numerator,
            denominator,
            out=np.full(pixel_count, np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        for method, numerator in numerators.items()
    }
    return means, denominator


def _robust_log_mean(
    views: Sequence[ProjectedView],
    depth_methods: Sequence[str],
    pixel_count: int,
    initial: Mapping[str, np.ndarray],
    *,
    use_photo: bool,
    huber_delta: float,
    consistency_threshold: float,
    offsets: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    numerators = {method: np.zeros(pixel_count, dtype=np.float64) for method in depth_methods}
    denominators = {method: np.zeros(pixel_count, dtype=np.float64) for method in depth_methods}
    for view_index, view in enumerate(views):
        base_weight = view.base_weights.astype(np.float64)
        if use_photo:
            base_weight = base_weight * view.photo_weights
        for method in depth_methods:
            offset = 0.0 if offsets is None else float(offsets[method][view_index])
            value = view.log_ranges[method] + offset
            residual = np.abs(value - initial[method][view.indices])
            huber_weight = np.minimum(1.0, huber_delta / np.maximum(residual, 1e-12))
            robust_weight = base_weight * huber_weight * (residual <= consistency_threshold)
            numerators[method][view.indices] += robust_weight * value
            denominators[method][view.indices] += robust_weight
    outputs = {}
    for method in depth_methods:
        robust = np.divide(
            numerators[method],
            denominators[method],
            out=np.asarray(initial[method]).copy(),
            where=denominators[method] > 0,
        )
        outputs[method] = robust
    return outputs


def _huber_pair_location(
    differences: np.ndarray,
    weights: np.ndarray,
    *,
    delta: float,
) -> tuple[float, float]:
    if differences.size == 0 or differences.shape != weights.shape:
        raise ValueError("Pairwise synchronization arrays must be non-empty and equal-shaped")
    location = float(np.median(differences))
    effective = weights.astype(np.float64, copy=False)
    for _ in range(3):
        residual = np.abs(differences - location)
        robust = np.minimum(1.0, delta / np.maximum(residual, 1e-12))
        effective = weights * robust
        denominator = float(effective.sum())
        if denominator <= 0:
            break
        location = float(np.sum(effective * differences) / denominator)
    return location, float(effective.sum())


def _sorted_unique_intersection_positions(
    left: np.ndarray,
    right: np.ndarray,
    *,
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matching positions without materializing ``np.intersect1d`` sorting.

    Projected-view indices are already sorted and unique.  Searching only the
    smaller array preserves exact intersection order while avoiding repeated
    sorting/copying of both large ERP supports for every view pair.
    """

    if (
        left.ndim != 1
        or right.ndim != 1
        or left.dtype.kind not in "iu"
        or right.dtype.kind not in "iu"
    ):
        raise ValueError("Intersection inputs must be 1-D integer arrays")
    if validate:
        if left.size > 1 and np.any(left[1:] <= left[:-1]):
            raise ValueError("Left intersection input must be sorted and unique")
        if right.size > 1 and np.any(right[1:] <= right[:-1]):
            raise ValueError("Right intersection input must be sorted and unique")
    if not left.size or not right.size:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    if left.size <= right.size:
        candidate_right = np.searchsorted(right, left)
        in_bounds = candidate_right < right.size
        left_positions = np.flatnonzero(
            in_bounds & (right[np.minimum(candidate_right, right.size - 1)] == left)
        ).astype(np.int64)
        return left_positions, candidate_right[left_positions].astype(np.int64)
    candidate_left = np.searchsorted(left, right)
    in_bounds = candidate_left < left.size
    right_positions = np.flatnonzero(
        in_bounds & (left[np.minimum(candidate_left, left.size - 1)] == right)
    ).astype(np.int64)
    return candidate_left[right_positions].astype(np.int64), right_positions


def _synchronize_view_offsets(
    views: Sequence[ProjectedView],
    depth_methods: Sequence[str],
    *,
    min_overlap: int,
    pair_max_samples: int,
    huber_delta: float,
    l2: float,
    max_abs_offset: float,
    gauge_view_count: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    """Solve prediction-only overlap graphs for per-view log-range offsets."""

    view_count = len(views)
    if view_count < 1:
        raise ValueError("Synchronization needs at least one projected view")
    registered_anchor_gauge = gauge_view_count is not None
    gauge_count = view_count if gauge_view_count is None else int(gauge_view_count)
    if gauge_count < 1 or gauge_count > view_count:
        raise ValueError("Synchronization gauge_view_count lies outside the view set")
    gauge_constraint = np.zeros(view_count, dtype=np.float64)
    if registered_anchor_gauge:
        anchor_reliability = np.asarray(
            [float(np.sum(view.base_weights, dtype=np.float64)) for view in views[:gauge_count]],
            dtype=np.float64,
        )
        if not np.isfinite(anchor_reliability).all() or np.any(anchor_reliability <= 0.0):
            raise ValueError("Synchronization anchor reliability must be finite and positive")
        gauge_constraint[:gauge_count] = anchor_reliability / anchor_reliability.sum()
    else:
        # Preserve the registered v2 behavior exactly when Route-P is absent.
        gauge_constraint[:] = 1.0
    pair_samples: list[tuple[int, int, np.ndarray, np.ndarray, int]] = []
    for left in range(view_count):
        for right in range(left + 1, view_count):
            left_positions, right_positions = _sorted_unique_intersection_positions(
                views[left].indices,
                views[right].indices,
                validate=False,
            )
            overlap = int(left_positions.size)
            if overlap < min_overlap:
                continue
            if overlap > pair_max_samples:
                selection = np.linspace(0, overlap - 1, pair_max_samples, dtype=np.int64)
                left_positions = left_positions[selection]
                right_positions = right_positions[selection]
            pair_samples.append((left, right, left_positions, right_positions, overlap))

    offsets: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    for method in depth_methods:
        edges = []
        adjacency = [set() for _ in range(view_count)]
        for left, right, left_positions, right_positions, overlap in pair_samples:
            differences = (
                views[right].log_ranges[method][right_positions]
                - views[left].log_ranges[method][left_positions]
            ).astype(np.float64)
            pair_weights = np.sqrt(
                views[left].base_weights[left_positions].astype(np.float64)
                * views[right].base_weights[right_positions].astype(np.float64)
            )
            location, effective_weight = _huber_pair_location(
                differences,
                pair_weights,
                delta=huber_delta,
            )
            if effective_weight <= 0 or not np.isfinite(location):
                continue
            edges.append((left, right, location, effective_weight, overlap))
            adjacency[left].add(right)
            adjacency[right].add(left)

        seen: set[int] = set()
        component_count = 0
        for start in range(view_count):
            if start in seen:
                continue
            component_count += 1
            stack = [start]
            seen.add(start)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)

        if edges:
            design = np.zeros((len(edges), view_count), dtype=np.float64)
            target = np.zeros(len(edges), dtype=np.float64)
            edge_weights = np.zeros(len(edges), dtype=np.float64)
            for edge_index, (left, right, location, weight, _) in enumerate(edges):
                design[edge_index, left] = 1.0
                design[edge_index, right] = -1.0
                target[edge_index] = location
                edge_weights[edge_index] = weight
            normal = design.T @ (edge_weights[:, None] * design)
            normal += float(l2) * np.eye(view_count, dtype=np.float64)
            rhs = design.T @ (edge_weights * target)
            kkt = np.zeros((view_count + 1, view_count + 1), dtype=np.float64)
            kkt[:view_count, :view_count] = normal
            kkt[:view_count, view_count] = gauge_constraint
            kkt[view_count, :view_count] = gauge_constraint
            solution = np.linalg.solve(kkt, np.concatenate((rhs, [0.0])))[:view_count]
            solution = np.clip(solution, -max_abs_offset, max_abs_offset)
            gauge_mean = (
                float(
                    np.sum(gauge_constraint[:gauge_count] * solution[:gauge_count])
                    / np.sum(gauge_constraint[:gauge_count])
                )
                if registered_anchor_gauge
                else float(solution.mean())
            )
            solution -= gauge_mean
            maximum = float(np.max(np.abs(solution)))
            if maximum > max_abs_offset:
                solution *= max_abs_offset / maximum
        else:
            solution = np.zeros(view_count, dtype=np.float64)
        offsets[method] = solution.astype(np.float32)
        diagnostics[method] = {
            "view_count": view_count,
            "pair_count": len(edges),
            "connected_component_count": component_count,
            "zero_mean_offset": float(np.mean(solution)),
            "offset_std": float(np.std(solution)),
            "max_abs_offset": float(np.max(np.abs(solution))),
            "gauge": {
                "definition": (
                    "base-weighted zero mean over a fixed ordered anchor subset"
                    if registered_anchor_gauge
                    else "unweighted zero mean over every view (registered v2 behavior)"
                ),
                "anchor_view_count": gauge_count,
                "anchor_frames": [view.frame_id for view in views[:gauge_count]],
                "normalized_anchor_weights": gauge_constraint[:gauge_count].tolist(),
                "anchor_mean_offset": float(
                    (
                        np.sum(gauge_constraint[:gauge_count] * solution[:gauge_count])
                        / np.sum(gauge_constraint[:gauge_count])
                    )
                    if registered_anchor_gauge
                    else np.mean(solution)
                ),
            },
            "offsets_by_frame": {
                view.frame_id: float(solution[index]) for index, view in enumerate(views)
            },
        }
    return offsets, diagnostics


def _aggregate_projected_views(
    views: Sequence[ProjectedView],
    depth_methods: Sequence[str],
    target_shape: tuple[int, int],
    *,
    huber_delta: float,
    consistency_threshold: float,
    sync_min_overlap: int = 256,
    sync_pair_max_samples: int = 4096,
    sync_huber_delta: float = 0.08,
    sync_l2: float = 1e-6,
    sync_max_abs_offset: float = 0.50,
    sync_gauge_view_count: int | None = None,
) -> FusionResult:
    """Fuse views without any target-depth input."""

    if not views:
        raise RuntimeError("A panorama station has no projectable regular views")
    if huber_delta <= 0 or consistency_threshold <= 0:
        raise ValueError("Robust fusion thresholds must be positive")
    height, width = target_shape
    pixel_count = height * width
    for view in views:
        view.validate(depth_methods, pixel_count)

    contributor_count = np.zeros(pixel_count, dtype=np.int32)
    best_score = np.full(pixel_count, -np.inf, dtype=np.float32)
    single = {method: np.full(pixel_count, np.nan, dtype=np.float32) for method in depth_methods}
    for view in views:
        contributor_count[view.indices] += 1
        replace = view.base_weights > best_score[view.indices]
        selected = view.indices[replace]
        best_score[selected] = view.base_weights[replace]
        for method in depth_methods:
            single[method][selected] = view.log_ranges[method][replace]

    weighted, _ = _weighted_log_mean(
        views,
        depth_methods,
        pixel_count,
        use_photo=False,
    )
    huber = _robust_log_mean(
        views,
        depth_methods,
        pixel_count,
        weighted,
        use_photo=False,
        huber_delta=huber_delta,
        consistency_threshold=consistency_threshold,
    )
    photo_initial, _ = _weighted_log_mean(
        views,
        depth_methods,
        pixel_count,
        use_photo=True,
    )
    photo_huber = _robust_log_mean(
        views,
        depth_methods,
        pixel_count,
        photo_initial,
        use_photo=True,
        huber_delta=huber_delta,
        consistency_threshold=consistency_threshold,
    )
    sync_offsets, sync_diagnostics = _synchronize_view_offsets(
        views,
        depth_methods,
        min_overlap=sync_min_overlap,
        pair_max_samples=sync_pair_max_samples,
        huber_delta=sync_huber_delta,
        l2=sync_l2,
        max_abs_offset=sync_max_abs_offset,
        gauge_view_count=sync_gauge_view_count,
    )
    synchronized_initial, _ = _weighted_log_mean(
        views,
        depth_methods,
        pixel_count,
        use_photo=False,
        offsets=sync_offsets,
    )
    synchronized_huber = _robust_log_mean(
        views,
        depth_methods,
        pixel_count,
        synchronized_initial,
        use_photo=False,
        huber_delta=huber_delta,
        consistency_threshold=consistency_threshold,
        offsets=sync_offsets,
    )
    log_outputs = {
        "single_best_view": single,
        "joint_weighted_log": weighted,
        "joint_huber": huber,
        "joint_photo_huber": photo_huber,
        "joint_synchronized_huber": synchronized_huber,
    }
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for fusion_method, values in log_outputs.items():
        predictions[fusion_method] = {
            method: np.exp(log_range).reshape(target_shape).astype(np.float32)
            for method, log_range in values.items()
        }
    return FusionResult(
        predictions=predictions,
        contributor_count=contributor_count.reshape(target_shape),
        synchronization=sync_diagnostics,
    )


def _aggregate_with_args(
    views: Sequence[ProjectedView],
    depth_methods: Sequence[str],
    target_shape: tuple[int, int],
    args: argparse.Namespace,
    *,
    sync_gauge_view_count: int | None = None,
) -> FusionResult:
    return _aggregate_projected_views(
        views,
        depth_methods,
        target_shape,
        huber_delta=float(args.huber_log_delta),
        consistency_threshold=float(args.consistency_log_threshold),
        sync_min_overlap=int(args.sync_min_overlap),
        sync_pair_max_samples=int(args.sync_pair_max_samples),
        sync_huber_delta=float(args.sync_huber_log_delta),
        sync_l2=float(args.sync_l2),
        sync_max_abs_offset=float(args.sync_max_abs_offset),
        sync_gauge_view_count=sync_gauge_view_count,
    )


def _prepare_tangent_predictions(
    bundle: TangentManifestBundle,
    station: StanfordPanorama,
    args: argparse.Namespace,
) -> TangentPrepared:
    """Freeze nested tangent6/tangent14 predictions without opening pano GT."""

    target_shape = (int(args.pano_height), 2 * int(args.pano_height))
    views = _load_tangent_projected_views(
        bundle,
        station.camera_uuid,
        target_shape,
        centrality_power=float(args.centrality_power),
    )
    fusions = {
        variant: _aggregate_with_args(
            views[:view_count],
            ["raw_da3"],
            target_shape,
            args,
            sync_gauge_view_count=min(6, view_count),
        )
        for variant, view_count in TANGENT_VARIANTS.items()
    }
    return TangentPrepared(projected_views=views, fusions=fusions)


def _selected_fusion_predictions(
    fusion: FusionResult,
    methods: Sequence[str],
) -> dict[str, Mapping[str, np.ndarray]]:
    missing = sorted(set(methods) - set(fusion.predictions))
    if missing:
        raise ValueError(f"Fusion result lacks methods {missing}")
    outputs = {}
    for method in methods:
        values = fusion.predictions[method]
        if "raw_da3" not in values:
            raise ValueError(f"Fusion result {method!r} lacks raw_da3")
        outputs[method] = {"raw_da3": values["raw_da3"]}
    return outputs


def _select_strict_single_view(views: Sequence[ProjectedView]) -> tuple[ProjectedView, float]:
    """Select exactly one regular frame without RGB, BIM, or GT-dependent scoring."""

    if not views:
        raise RuntimeError("Strict single-frame selection has no projected regular view")
    scored = [(float(np.sum(view.base_weights, dtype=np.float64)), view) for view in views]
    score, selected = min(scored, key=lambda item: (-item[0], item[1].frame_id))
    return selected, score


def _strict_comparison_predictions(
    selected_view: ProjectedView,
    joint_predictions: Mapping[str, Mapping[str, np.ndarray]],
    depth_methods: Sequence[str],
    target_shape: tuple[int, int],
) -> dict[str, dict[str, np.ndarray]]:
    pixel_count = int(np.prod(target_shape))
    selected_view.validate(depth_methods, pixel_count)
    strict_depths = {}
    for method in depth_methods:
        prediction = np.full(pixel_count, np.nan, dtype=np.float32)
        prediction[selected_view.indices] = np.exp(selected_view.log_ranges[method])
        strict_depths[method] = prediction.reshape(target_shape)
    outputs = {"strict_single_frame": strict_depths}
    for fusion_method in STRICT_FUSION_METHODS[1:]:
        if fusion_method not in joint_predictions:
            raise ValueError(f"Strict comparison lacks joint method {fusion_method!r}")
        outputs[fusion_method] = dict(joint_predictions[fusion_method])
    return outputs


def _strict_fixed_support(
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
    selected_view: ProjectedView,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    selected_coverage = np.zeros(gt_depth.size, dtype=bool)
    selected_coverage[selected_view.indices] = True
    selected_coverage = selected_coverage.reshape(gt_depth.shape)
    return _fixed_pano_support(
        gt_depth,
        gt_valid,
        selected_coverage.astype(np.int32),
        predictions,
    )


def _fixed_pano_support(
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
    contributor_count: np.ndarray,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    if gt_depth.shape != gt_valid.shape or gt_depth.shape != contributor_count.shape:
        raise ValueError("Panorama GT and coverage shapes differ")
    support = gt_valid.astype(bool) & (contributor_count > 0)
    if not np.any(support):
        raise RuntimeError("Panorama station has no fixed GT/regular-view overlap")
    expected_count = int(np.count_nonzero(support))
    for fusion_method, values in predictions.items():
        for depth_method, prediction in values.items():
            if prediction.shape != gt_depth.shape:
                raise ValueError(f"{fusion_method}/{depth_method} shape differs from panorama GT")
            invalid = support & (~np.isfinite(prediction) | (prediction <= 0))
            if np.any(invalid):
                raise RuntimeError(
                    f"{fusion_method}/{depth_method} is invalid on "
                    f"{int(np.count_nonzero(invalid))}/{expected_count} fixed-support pixels"
                )
    return support


def _metrics_for_array(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float | int]:
    totals = MetricTotals()
    totals.update(prediction, target, support, weights)
    return totals.compute()


def _relative_reduction(candidate: float, reference: float) -> float:
    return 100.0 * (reference - candidate) / reference if reference > 0 else float("nan")


def _paired_bootstrap(
    station_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_depth: str,
    candidate_fusion: str,
    reference_depth: str,
    reference_fusion: str,
    metric: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    lookup = {
        (str(row["station_id"]), str(row["depth_method"]), str(row["fusion_method"])): row
        for row in station_rows
    }
    station_ids = sorted(
        {
            str(row["station_id"])
            for row in station_rows
            if (str(row["depth_method"]), str(row["fusion_method"]))
            == (candidate_depth, candidate_fusion)
            and (
                str(row["station_id"]),
                reference_depth,
                reference_fusion,
            )
            in lookup
        }
    )
    if not station_ids:
        raise RuntimeError("Paired panorama bootstrap has no common stations")
    differences_by_room: dict[str, list[float]] = defaultdict(list)
    for station in station_ids:
        candidate_row = lookup[(station, candidate_depth, candidate_fusion)]
        reference_row = lookup[(station, reference_depth, reference_fusion)]
        if str(candidate_row["room"]) != str(reference_row["room"]):
            raise RuntimeError(f"{station}: paired rows disagree on room")
        differences_by_room[str(candidate_row["room"])].append(
            float(candidate_row[metric]) - float(reference_row[metric])
        )
    room_ids = sorted(differences_by_room)
    room_sums = np.asarray(
        [float(np.sum(differences_by_room[room])) for room in room_ids], dtype=np.float64
    )
    room_counts = np.asarray([len(differences_by_room[room]) for room in room_ids], dtype=np.int64)
    station_differences = np.concatenate(
        [np.asarray(differences_by_room[room], dtype=np.float64) for room in room_ids]
    )
    if not np.isfinite(station_differences).all() or not station_differences.size:
        raise RuntimeError("Paired panorama bootstrap received non-finite differences")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(room_ids), size=(repetitions, len(room_ids)))
    # Resample whole room clusters, but retain every station within each drawn
    # cluster.  The replicate statistic therefore estimates the same
    # equal-station mean as the formal primary metric, even when room sizes
    # differ. Drawing a room twice repeats all of its stations twice.
    distribution = room_sums[samples].sum(axis=1) / room_counts[samples].sum(axis=1)
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    room_means = room_sums / room_counts
    return {
        "metric": metric,
        "candidate": f"{candidate_depth}/{candidate_fusion}",
        "reference": f"{reference_depth}/{reference_fusion}",
        "difference_definition": "candidate - reference (negative is better)",
        "resampling_unit": "room cluster",
        "room_count": len(room_ids),
        "room_ids": room_ids,
        "station_count": len(station_ids),
        "mean_difference": float(station_differences.mean()),
        "median_difference": float(np.median(station_differences)),
        "candidate_better_room_fraction": float(np.mean(room_means < 0)),
        "confidence_interval_95": [float(lower), float(upper)],
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
    }


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_model(
    checkpoint_path: Path | None,
    cfg: Config,
    dataset: BIMDepthDataset,
    *,
    split: str,
    device: torch.device,
    allow_cross_dataset: bool,
) -> tuple[BIMPriorDA3 | None, dict[str, Any]]:
    if checkpoint_path is None:
        return None, {"status": "not_requested", "learned_method_evaluated": False}
    path = checkpoint_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or "model" not in state:
        raise TypeError("Checkpoint must contain a model state dictionary")
    checkpoint_cfg = state.get("config")
    if not isinstance(checkpoint_cfg, Mapping):
        raise TypeError("Checkpoint does not contain its training config")
    model_differences = validate_checkpoint_model_config(state, cfg.model)
    dataset_validation = validate_checkpoint_evaluation_dataset_provenance(
        state,
        dataset.split_provenance,
        split=split,
        allow_cross_dataset=allow_cross_dataset,
    )
    model = BIMPriorDA3(cfg)
    if model.e2e_da3_enabled:
        raise ValueError(
            "Panorama regular-view protocol requires cached DA3 predictions; "
            "use a non-E2E evaluation config"
        )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    return model, {
        "status": "verified",
        "learned_method_evaluated": True,
        "path": str(path),
        "sha256": _sha256(path),
        "epoch": state.get("epoch"),
        "model_config_differences": model_differences,
        "dataset_validation": dataset_validation,
    }


def _route_metric_row(
    *,
    station: StanfordPanorama,
    prediction: np.ndarray,
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
    support: np.ndarray,
    contributor_count: np.ndarray,
    fusion_method: str,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    area_weights = _latitude_area_weights(*gt_depth.shape)
    support_count = int(np.count_nonzero(support))
    gt_count = int(np.count_nonzero(gt_valid))
    if support_count <= 0 or gt_count <= 0:
        raise RuntimeError(f"{station.camera_uuid}: route support is empty")
    spherical_metrics = _metrics_for_array(
        prediction,
        gt_depth,
        support,
        area_weights,
    )
    overlap_count = int(np.count_nonzero(support & (contributor_count >= 2)))
    return {
        "station_id": station.camera_uuid,
        "room": station.room,
        **dict(extra),
        "fusion_method": fusion_method,
        "depth_method": "raw_da3",
        "fixed_support_pixels": support_count,
        "gt_valid_pixels_at_eval_resolution": gt_count,
        "coverage_fraction": support_count / gt_count,
        "solid_angle_coverage_fraction": float(
            area_weights[support].sum() / area_weights[gt_valid].sum()
        ),
        "multi_view_overlap_fraction": overlap_count / support_count,
        "mean_contributors_on_support": float(np.mean(contributor_count[support])),
        **_metrics_for_array(prediction, gt_depth, support),
        **{
            f"spherical_{name}": value
            for name, value in spherical_metrics.items()
            if name != "count"
        },
    }


def _evaluate_tangent_prepared(
    station: StanfordPanorama,
    prepared: TangentPrepared,
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
    *,
    pano_only: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    """Evaluate tangent6/14 on native and nested common supports."""

    selected_predictions = {
        variant: _selected_fusion_predictions(fusion, TANGENT_FUSION_METHODS)
        for variant, fusion in prepared.fusions.items()
    }
    native_supports = {
        variant: _fixed_pano_support(
            gt_depth,
            gt_valid,
            fusion.contributor_count,
            selected_predictions[variant],
        )
        for variant, fusion in prepared.fusions.items()
    }
    common_coverage = np.logical_and.reduce(
        [fusion.contributor_count > 0 for fusion in prepared.fusions.values()]
    )
    common_supports = {
        variant: _fixed_pano_support(
            gt_depth,
            gt_valid,
            common_coverage.astype(np.int32),
            selected_predictions[variant],
        )
        for variant in TANGENT_VARIANTS
    }
    rows: list[dict[str, Any]] = []
    arrays = {}
    support_by_scope = {
        "native": native_supports,
        "common_tangent6_tangent14": common_supports,
    }
    for support_scope, supports in support_by_scope.items():
        for variant, view_count in TANGENT_VARIANTS.items():
            fusion = prepared.fusions[variant]
            support = supports[variant]
            for fusion_method in TANGENT_FUSION_METHODS:
                prediction = fusion.predictions[fusion_method]["raw_da3"]
                rows.append(
                    _route_metric_row(
                        station=station,
                        prediction=prediction,
                        gt_depth=gt_depth,
                        gt_valid=gt_valid,
                        support=support,
                        contributor_count=fusion.contributor_count,
                        fusion_method=fusion_method,
                        extra={
                            "pano_only": bool(pano_only),
                            "route_variant": variant,
                            "view_count": view_count,
                            "support_scope": support_scope,
                        },
                    )
                )
                arrays[(support_scope, variant, fusion_method)] = (
                    prediction,
                    gt_depth,
                    support,
                )
    area_weights = _latitude_area_weights(*gt_depth.shape)
    info = {
        "station_id": station.camera_uuid,
        "room": station.room,
        "pano_only": bool(pano_only),
        "view_count_ablation": dict(TANGENT_VARIANTS),
        "native_coverage": {
            variant: {
                "pixels": int(np.count_nonzero(support)),
                "pixel_fraction": float(np.mean(support[gt_valid])),
                "solid_angle_fraction": float(
                    area_weights[support].sum() / area_weights[gt_valid].sum()
                ),
            }
            for variant, support in native_supports.items()
        },
        "common_support": {
            "pixels": int(np.count_nonzero(common_supports["tangent6"])),
            "pixel_fraction": float(np.mean(common_supports["tangent6"][gt_valid])),
            "solid_angle_fraction": float(
                area_weights[common_supports["tangent6"]].sum() / area_weights[gt_valid].sum()
            ),
        },
        "overlap_log_scale_synchronization": {
            variant: fusion.synchronization for variant, fusion in prepared.fusions.items()
        },
    }
    return rows, arrays, info


def _prepare_regular_pano_joint(
    regular_views: Sequence[ProjectedView],
    tangent: TangentPrepared,
    target_shape: tuple[int, int],
    args: argparse.Namespace,
) -> dict[str, FusionResult]:
    """Freeze raw-only regular/tangent fusion matrices before GT is available."""

    regular_raw = tuple(_raw_only_view(view) for view in regular_views)
    regular_raw_fusion = _aggregate_with_args(
        regular_raw,
        ["raw_da3"],
        target_shape,
        args,
        sync_gauge_view_count=len(regular_raw),
    )
    outputs = {"regular_only": regular_raw_fusion}
    for variant, view_count in TANGENT_VARIANTS.items():
        outputs[f"regular_plus_{variant}"] = _aggregate_with_args(
            (*regular_raw, *tangent.projected_views[:view_count]),
            ["raw_da3"],
            target_shape,
            args,
            sync_gauge_view_count=len(regular_raw),
        )
    return outputs


def _evaluate_regular_pano_joint(
    station: StanfordPanorama,
    fusions: Mapping[str, FusionResult],
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    """Compare combined raw views on regular-fixed and native-union supports."""

    expected_sources = (
        "regular_only",
        "regular_plus_tangent6",
        "regular_plus_tangent14",
    )
    if tuple(fusions) != expected_sources:
        raise ValueError(f"Combined route source order differs: {tuple(fusions)}")
    prediction_matrix = {
        source: _selected_fusion_predictions(fusions[source], COMBINED_FUSION_METHODS)
        for source in expected_sources
    }
    flattened = {
        f"{source}/{fusion_method}": values
        for source, source_predictions in prediction_matrix.items()
        for fusion_method, values in source_predictions.items()
    }
    common_regular_support = _fixed_pano_support(
        gt_depth,
        gt_valid,
        fusions["regular_only"].contributor_count,
        flattened,
    )
    native_supports = {
        source: _fixed_pano_support(
            gt_depth,
            gt_valid,
            fusions[source].contributor_count,
            prediction_matrix[source],
        )
        for source in expected_sources
    }
    rows: list[dict[str, Any]] = []
    arrays = {}
    for support_scope in ("common_regular", "native_union"):
        for source in expected_sources:
            support = (
                common_regular_support
                if support_scope == "common_regular"
                else native_supports[source]
            )
            fusion = fusions[source]
            tangent_view_count = {
                "regular_only": 0,
                "regular_plus_tangent6": 6,
                "regular_plus_tangent14": 14,
            }[source]
            for fusion_method in COMBINED_FUSION_METHODS:
                prediction = fusion.predictions[fusion_method]["raw_da3"]
                rows.append(
                    _route_metric_row(
                        station=station,
                        prediction=prediction,
                        gt_depth=gt_depth,
                        gt_valid=gt_valid,
                        support=support,
                        contributor_count=fusion.contributor_count,
                        fusion_method=fusion_method,
                        extra={
                            "source_set": source,
                            "tangent_view_count": tangent_view_count,
                            "support_scope": support_scope,
                        },
                    )
                )
                arrays[(support_scope, source, fusion_method)] = (
                    prediction,
                    gt_depth,
                    support,
                )
    area_weights = _latitude_area_weights(*gt_depth.shape)
    info = {
        "station_id": station.camera_uuid,
        "room": station.room,
        "fixed_regular_support_pixels": int(np.count_nonzero(common_regular_support)),
        "native_union_coverage": {
            source: {
                "pixels": int(np.count_nonzero(support)),
                "pixel_fraction": float(np.mean(support[gt_valid])),
                "solid_angle_fraction": float(
                    area_weights[support].sum() / area_weights[gt_valid].sum()
                ),
            }
            for source, support in native_supports.items()
        },
        "overlap_log_scale_synchronization": {
            source: fusions[source].synchronization for source in expected_sources
        },
    }
    return rows, arrays, info


def _evaluate_station(
    station: StanfordPanorama,
    records: Sequence[Mapping[str, Any]],
    cfg: Config,
    model: BIMPriorDA3 | None,
    device: torch.device,
    args: argparse.Namespace,
    tangent: TangentPrepared | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    list[dict[str, Any]],
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, Any],
    dict[str, Any],
]:
    target_shape = (int(args.pano_height), 2 * int(args.pano_height))
    pano_pixels = _erp_pixel_grid(*target_shape)
    pano_rgb = _load_pano_rgb(station.rgb_path, target_shape)
    depth_methods = list(DEPTH_METHODS)
    if model is not None:
        depth_methods.append(LEARNED_METHOD)
    projected_views: list[ProjectedView] = []
    station_scale, station_scale_receipt = _station_bim_scale(
        records,
        cfg,
        samples_per_view=int(args.station_scale_samples_per_view),
    )

    ordered_records = sorted(records, key=lambda value: (int(value["frame_number"]), value["id"]))
    for start in range(0, len(ordered_records), int(args.batch_size)):
        batch_records = ordered_records[start : start + int(args.batch_size)]
        frames = [
            _read_regular_frame(
                record,
                cfg,
                depth_methods,
                station_scale=station_scale,
            )
            for record in batch_records
        ]
        if model is not None:
            batch = move_batch(_regular_model_batch(frames), device)
            with torch.no_grad():
                learned = model(batch)["depth"].detach().float().cpu().numpy()[:, 0]
            if learned.shape != (len(frames), *frames[0].base_confidence.shape):
                raise RuntimeError("Learned checkpoint returned an unexpected depth shape")
            for frame, depth in zip(frames, learned):
                frame.predictions[LEARNED_METHOD] = depth.astype(np.float32)
        for frame in frames:
            projected_views.append(
                _project_regular_view(
                    frame,
                    station,
                    pano_pixels,
                    pano_rgb,
                    depth_methods,
                    centrality_power=float(args.centrality_power),
                    confidence_floor=float(args.confidence_floor),
                    photo_sigma=float(args.photo_sigma),
                )
            )

    fusion = _aggregate_projected_views(
        projected_views,
        depth_methods,
        target_shape,
        huber_delta=float(args.huber_log_delta),
        consistency_threshold=float(args.consistency_log_threshold),
        sync_min_overlap=int(args.sync_min_overlap),
        sync_pair_max_samples=int(args.sync_pair_max_samples),
        sync_huber_delta=float(args.sync_huber_log_delta),
        sync_l2=float(args.sync_l2),
        sync_max_abs_offset=float(args.sync_max_abs_offset),
    )
    combined_fusions = (
        _prepare_regular_pano_joint(
            projected_views,
            tangent,
            target_shape,
            args,
        )
        if tangent is not None
        else None
    )
    strict_view, strict_selector_score = _select_strict_single_view(projected_views)
    strict_predictions = _strict_comparison_predictions(
        strict_view,
        fusion.predictions,
        depth_methods,
        target_shape,
    )

    # Intentional protocol boundary: official pano depth is opened only here,
    # after single-view selection and all joint estimates are immutable.
    gt_depth, gt_valid, native_gt_valid = _load_pano_gt(
        station.depth_path,
        target_shape,
        min_depth=float(cfg.data.min_depth),
        max_depth=float(cfg.data.max_depth),
    )
    support = _fixed_pano_support(
        gt_depth,
        gt_valid,
        fusion.contributor_count,
        fusion.predictions,
    )
    strict_support = _strict_fixed_support(
        gt_depth,
        gt_valid,
        strict_view,
        strict_predictions,
    )
    area_weights = _latitude_area_weights(*target_shape)
    support_count = int(np.count_nonzero(support))
    gt_count = int(np.count_nonzero(gt_valid))
    solid_angle_coverage = float(area_weights[support].sum() / area_weights[gt_valid].sum())
    overlap_count = int(np.count_nonzero(support & (fusion.contributor_count >= 2)))
    rows: list[dict[str, Any]] = []
    metric_arrays = {}
    for fusion_method in FUSION_METHODS:
        for depth_method in depth_methods:
            prediction = fusion.predictions[fusion_method][depth_method]
            pixel_metrics = _metrics_for_array(prediction, gt_depth, support)
            spherical_metrics = _metrics_for_array(
                prediction,
                gt_depth,
                support,
                area_weights,
            )
            row = {
                "station_id": station.camera_uuid,
                "room": station.room,
                "regular_view_count": len(records),
                "fusion_method": fusion_method,
                "depth_method": depth_method,
                "fixed_support_pixels": support_count,
                "gt_valid_pixels_at_eval_resolution": gt_count,
                "regular_coverage_fraction": support_count / gt_count,
                "solid_angle_coverage_fraction": solid_angle_coverage,
                "multi_view_overlap_fraction": overlap_count / support_count,
                "mean_contributors_on_support": float(np.mean(fusion.contributor_count[support])),
                **pixel_metrics,
                **{
                    f"spherical_{name}": value
                    for name, value in spherical_metrics.items()
                    if name != "count"
                },
            }
            rows.append(row)
            metric_arrays[(fusion_method, depth_method)] = (prediction, gt_depth, support)
    strict_support_count = int(np.count_nonzero(strict_support))
    strict_solid_angle_coverage = float(
        area_weights[strict_support].sum() / area_weights[gt_valid].sum()
    )
    strict_rows: list[dict[str, Any]] = []
    strict_metric_arrays = {}
    for fusion_method in STRICT_FUSION_METHODS:
        for depth_method in depth_methods:
            prediction = strict_predictions[fusion_method][depth_method]
            pixel_metrics = _metrics_for_array(prediction, gt_depth, strict_support)
            spherical_metrics = _metrics_for_array(
                prediction,
                gt_depth,
                strict_support,
                area_weights,
            )
            strict_rows.append(
                {
                    "station_id": station.camera_uuid,
                    "room": station.room,
                    "regular_view_count": len(records),
                    "selected_frame_id": strict_view.frame_id,
                    "selector_score_sum_base_weights": strict_selector_score,
                    "fusion_method": fusion_method,
                    "depth_method": depth_method,
                    "fixed_support_pixels": strict_support_count,
                    "gt_valid_pixels_at_eval_resolution": gt_count,
                    "strict_single_coverage_fraction": strict_support_count / gt_count,
                    "strict_single_solid_angle_coverage_fraction": (strict_solid_angle_coverage),
                    **pixel_metrics,
                    **{
                        f"spherical_{name}": value
                        for name, value in spherical_metrics.items()
                        if name != "count"
                    },
                }
            )
            strict_metric_arrays[(fusion_method, depth_method)] = (
                prediction,
                gt_depth,
                strict_support,
            )
    station_info = {
        "station_id": station.camera_uuid,
        "room": station.room,
        "regular_view_count": len(records),
        "projected_view_count": len(projected_views),
        "native_pano_valid_pixels_before_range_filter": native_gt_valid,
        "gt_valid_pixels_at_eval_resolution": gt_count,
        "fixed_support_pixels": support_count,
        "regular_coverage_fraction": support_count / gt_count,
        "solid_angle_coverage_fraction": solid_angle_coverage,
        "multi_view_overlap_fraction": overlap_count / support_count,
        "mean_contributors_on_support": float(np.mean(fusion.contributor_count[support])),
        "station_bim_scale": station_scale_receipt,
        "overlap_log_scale_synchronization": fusion.synchronization,
        "strict_single_frame": {
            "selector": "max sum(base_weights), stable frame_id tie-break",
            "selected_frame_id": strict_view.frame_id,
            "selector_score_sum_base_weights": strict_selector_score,
            "fixed_support_pixels": strict_support_count,
            "coverage_fraction": strict_support_count / gt_count,
            "solid_angle_coverage_fraction": strict_solid_angle_coverage,
        },
    }
    route_outputs: dict[str, Any] = {
        "tangent_rows": [],
        "tangent_arrays": {},
        "tangent_info": None,
        "combined_rows": [],
        "combined_arrays": {},
        "combined_info": None,
    }
    if tangent is not None:
        tangent_rows, tangent_arrays, tangent_info = _evaluate_tangent_prepared(
            station,
            tangent,
            gt_depth,
            gt_valid,
            pano_only=False,
        )
        if combined_fusions is None:
            raise AssertionError("Combined fusions were not frozen before GT")
        combined_rows, combined_arrays, combined_info = _evaluate_regular_pano_joint(
            station,
            combined_fusions,
            gt_depth,
            gt_valid,
        )
        route_outputs = {
            "tangent_rows": tangent_rows,
            "tangent_arrays": tangent_arrays,
            "tangent_info": tangent_info,
            "combined_rows": combined_rows,
            "combined_arrays": combined_arrays,
            "combined_info": combined_info,
        }
    return rows, metric_arrays, strict_rows, strict_metric_arrays, station_info, route_outputs


def _evaluate_tangent_only_station(
    station: StanfordPanorama,
    tangent: TangentPrepared,
    cfg: Config,
    args: argparse.Namespace,
    *,
    pano_only: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    """Evaluate a pano-only station after its Route-P predictions are frozen."""

    target_shape = (int(args.pano_height), 2 * int(args.pano_height))
    gt_depth, gt_valid, native_gt_valid = _load_pano_gt(
        station.depth_path,
        target_shape,
        min_depth=float(cfg.data.min_depth),
        max_depth=float(cfg.data.max_depth),
    )
    rows, arrays, info = _evaluate_tangent_prepared(
        station,
        tangent,
        gt_depth,
        gt_valid,
        pano_only=pano_only,
    )
    return rows, arrays, {**info, "native_pano_valid_pixels_before_range_filter": native_gt_valid}


def _macro_metrics(
    rows: Sequence[Mapping[str, Any]],
    depth_method: str,
    fusion_method: str,
) -> dict[str, float | int]:
    selected = [
        row
        for row in rows
        if row["depth_method"] == depth_method and row["fusion_method"] == fusion_method
    ]
    if not selected:
        raise RuntimeError(f"No station rows for {depth_method}/{fusion_method}")
    return {
        **{name: float(np.mean([float(row[name]) for row in selected])) for name in METRIC_NAMES},
        **{
            f"spherical_{name}": float(
                np.mean([float(row[f"spherical_{name}"]) for row in selected])
            )
            for name in METRIC_NAMES
        },
        "station_count": len(selected),
        "pixel_count": sum(int(row["count"]) for row in selected),
    }


def _per_room_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["room"]), str(row["fusion_method"]), str(row["depth_method"]))].append(row)
    outputs = []
    for (room, fusion_method, depth_method), station_values in sorted(grouped.items()):
        station_ids = [str(row["station_id"]) for row in station_values]
        if len(station_ids) != len(set(station_ids)):
            raise RuntimeError(
                f"Duplicate station row in room aggregate: {room}/{fusion_method}/{depth_method}"
            )
        outputs.append(
            {
                "room": room,
                "fusion_method": fusion_method,
                "depth_method": depth_method,
                "station_count": len(station_values),
                "fixed_support_pixels": sum(
                    int(row["fixed_support_pixels"]) for row in station_values
                ),
                **{
                    f"spherical_{name}": float(
                        np.mean([float(row[f"spherical_{name}"]) for row in station_values])
                    )
                    for name in METRIC_NAMES
                },
            }
        )
    return outputs


def _room_macro_metrics(
    room_rows: Sequence[Mapping[str, Any]],
    depth_method: str,
    fusion_method: str,
) -> dict[str, float | int]:
    selected = [
        row
        for row in room_rows
        if row["depth_method"] == depth_method and row["fusion_method"] == fusion_method
    ]
    if not selected:
        raise RuntimeError(f"No room rows for {depth_method}/{fusion_method}")
    return {
        **{
            f"spherical_{name}": float(
                np.mean([float(row[f"spherical_{name}"]) for row in selected])
            )
            for name in METRIC_NAMES
        },
        "room_count": len(selected),
        "station_count": sum(int(row["station_count"]) for row in selected),
        "pixel_count": sum(int(row["fixed_support_pixels"]) for row in selected),
    }


def _grouped_route_summary(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
) -> list[dict[str, Any]]:
    """Equal-station and equal-room summaries for an auxiliary route matrix."""

    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    outputs = []
    for group, station_rows in sorted(grouped.items()):
        station_ids = [str(row["station_id"]) for row in station_rows]
        if len(station_ids) != len(set(station_ids)):
            raise RuntimeError(f"Duplicate auxiliary route station row for {group}")
        rooms: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in station_rows:
            rooms[str(row["room"])].append(row)
        room_metrics = {
            room: {
                name: float(np.mean([float(row[f"spherical_{name}"]) for row in values]))
                for name in METRIC_NAMES
            }
            for room, values in rooms.items()
        }
        outputs.append(
            {
                **dict(zip(group_fields, group)),
                "station_count": len(station_rows),
                "room_count": len(rooms),
                "pixel_count": sum(int(row["fixed_support_pixels"]) for row in station_rows),
                "station_macro": {
                    **{
                        name: float(np.mean([float(row[name]) for row in station_rows]))
                        for name in METRIC_NAMES
                    },
                    **{
                        f"spherical_{name}": float(
                            np.mean([float(row[f"spherical_{name}"]) for row in station_rows])
                        )
                        for name in METRIC_NAMES
                    },
                },
                "room_macro": {
                    f"spherical_{name}": float(
                        np.mean([values[name] for values in room_metrics.values()])
                    )
                    for name in METRIC_NAMES
                },
            }
        )
    return outputs


def _route_filter(
    rows: Sequence[Mapping[str, Any]],
    filters: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in filters.items())
    ]


def _route_paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    candidate_rows = _route_filter(rows, candidate)
    reference_rows = _route_filter(rows, reference)
    candidate_lookup = {str(row["station_id"]): row for row in candidate_rows}
    reference_lookup = {str(row["station_id"]): row for row in reference_rows}
    station_ids = sorted(set(candidate_lookup) & set(reference_lookup))
    if (
        not station_ids
        or len(station_ids) != len(candidate_rows)
        or len(station_ids) != len(reference_rows)
    ):
        raise RuntimeError(f"{kind}: auxiliary contrast does not have one paired row per station")
    differences_by_room: dict[str, list[float]] = defaultdict(list)
    for station_id in station_ids:
        candidate_row = candidate_lookup[station_id]
        reference_row = reference_lookup[station_id]
        if candidate_row["room"] != reference_row["room"]:
            raise RuntimeError(f"{kind}/{station_id}: paired rooms differ")
        if candidate_row["fixed_support_pixels"] != reference_row["fixed_support_pixels"]:
            raise RuntimeError(f"{kind}/{station_id}: paired fixed-support sizes differ")
        differences_by_room[str(candidate_row["room"])].append(
            float(candidate_row["spherical_abs_rel"]) - float(reference_row["spherical_abs_rel"])
        )
    room_ids = sorted(differences_by_room)
    room_sums = np.asarray(
        [float(np.sum(differences_by_room[room])) for room in room_ids], dtype=np.float64
    )
    room_counts = np.asarray([len(differences_by_room[room]) for room in room_ids], dtype=np.int64)
    station_differences = np.concatenate(
        [np.asarray(differences_by_room[room], dtype=np.float64) for room in room_ids]
    )
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(room_ids), size=(repetitions, len(room_ids)))
    distribution = room_sums[samples].sum(axis=1) / room_counts[samples].sum(axis=1)
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    room_means = room_sums / room_counts
    candidate_primary = float(
        np.mean([float(candidate_lookup[station]["spherical_abs_rel"]) for station in station_ids])
    )
    reference_primary = float(
        np.mean([float(reference_lookup[station]["spherical_abs_rel"]) for station in station_ids])
    )
    return {
        "kind": kind,
        "candidate": dict(candidate),
        "reference": dict(reference),
        "primary_metric": "equal-station macro exact-solid-angle AbsRel",
        "primary_abs_rel_candidate": candidate_primary,
        "primary_abs_rel_reference": reference_primary,
        "primary_abs_rel_absolute_reduction": reference_primary - candidate_primary,
        "primary_abs_rel_relative_reduction_percent": _relative_reduction(
            candidate_primary, reference_primary
        ),
        "room_cluster_paired_bootstrap_primary_abs_rel": {
            "difference_definition": "candidate - reference (negative is better)",
            "resampling_unit": "room cluster",
            "room_count": len(room_ids),
            "room_ids": room_ids,
            "station_count": len(station_ids),
            "mean_difference": float(station_differences.mean()),
            "median_difference": float(np.median(station_differences)),
            "candidate_better_room_fraction": float(np.mean(room_means < 0)),
            "confidence_interval_95": [float(lower), float(upper)],
            "bootstrap_repetitions": int(repetitions),
            "seed": int(seed),
        },
    }


def _contrast(
    metric_table: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    station_rows: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    candidate_depth: str,
    candidate_fusion: str,
    reference_depth: str,
    reference_fusion: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    candidate = metric_table[candidate_fusion][candidate_depth]
    reference = metric_table[reference_fusion][reference_depth]
    candidate_primary = float(candidate["station_macro"]["spherical_abs_rel"])
    reference_primary = float(reference["station_macro"]["spherical_abs_rel"])
    return {
        "kind": kind,
        "candidate": f"{candidate_depth}/{candidate_fusion}",
        "reference": f"{reference_depth}/{reference_fusion}",
        "primary_metric": "equal-station macro of exact ERP-pixel-solid-angle AbsRel",
        "primary_abs_rel_candidate": candidate_primary,
        "primary_abs_rel_reference": reference_primary,
        "primary_abs_rel_absolute_reduction": reference_primary - candidate_primary,
        "primary_abs_rel_relative_reduction_percent": _relative_reduction(
            candidate_primary, reference_primary
        ),
        "room_cluster_paired_bootstrap_primary_abs_rel": _paired_bootstrap(
            station_rows,
            candidate_depth=candidate_depth,
            candidate_fusion=candidate_fusion,
            reference_depth=reference_depth,
            reference_fusion=reference_fusion,
            metric="spherical_abs_rel",
            seed=seed,
            repetitions=repetitions,
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(int(args.seed))
    cfg = load_config(args.config)
    scale_protocol = validate_universal_scale_protocol(cfg)
    dataset = BIMDepthDataset(cfg, args.split, augment=False, require_ground_truth=False)
    device = _device_from_arg(args.device)
    model, checkpoint_provenance = _load_model(
        args.checkpoint,
        cfg,
        dataset,
        split=args.split,
        device=device,
        allow_cross_dataset=bool(args.allow_cross_dataset_checkpoint),
    )
    depth_methods = list(DEPTH_METHODS) + ([LEARNED_METHOD] if model is not None else [])

    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    all_stations = discover_stanford_panoramas(area_root)
    all_paired_panos = [station for station in all_stations if station.regular_views]
    all_pano_only_ids = sorted(
        station.camera_uuid for station in all_stations if not station.regular_views
    )
    station_by_id = {station.camera_uuid: station for station in all_stations}
    records_by_station: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset.records:
        records_by_station[str(record["camera_uuid"])].append(record)
    selected_rooms = {str(record["region"]) for record in dataset.records}
    split_panos = [station for station in all_stations if station.room in selected_rooms]
    split_pano_ids = {station.camera_uuid for station in split_panos}
    missing_pano = sorted(set(records_by_station) - split_pano_ids)
    pano_only = sorted(split_pano_ids - set(records_by_station))
    available_ids = set(records_by_station) & split_pano_ids
    if not available_ids:
        raise RuntimeError(f"No {args.split} regular camera UUID has a paired panorama")
    shared_available_count = len(available_ids)
    tangent_bundle = (
        _validate_tangent_manifest(
            args.tangent_manifest,
            cfg=cfg,
            dataset=dataset,
            split=str(args.split),
            confirm_test=bool(args.confirm_test),
            split_panoramas=split_panos,
        )
        if args.tangent_manifest is not None
        else None
    )
    if tangent_bundle is None:
        available = sorted(available_ids)
        if args.max_stations is not None:
            available = available[: int(args.max_stations)]
        selected_stations = [station_by_id[station_id] for station_id in available]
        route_p_stations: list[StanfordPanorama] = []
    else:
        route_p_stations = [station_by_id[camera_uuid] for camera_uuid in tangent_bundle.stations]
        if args.max_stations is not None:
            route_p_stations = route_p_stations[: int(args.max_stations)]
        selected_stations = [
            station for station in route_p_stations if station.camera_uuid in available_ids
        ]
        if not selected_stations:
            raise RuntimeError(
                "The requested tangent evaluation prefix contains no shared regular/pano station"
            )

    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else resolve_project_path(cfg, f"results/stanford_area1/pano_{args.split}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    station_infos = []
    micro = {
        fusion: {method: MetricTotals() for method in depth_methods} for fusion in FUSION_METHODS
    }
    spherical_micro = {
        fusion: {method: MetricTotals() for method in depth_methods} for fusion in FUSION_METHODS
    }
    strict_all_rows: list[dict[str, Any]] = []
    strict_micro = {
        fusion: {method: MetricTotals() for method in depth_methods}
        for fusion in STRICT_FUSION_METHODS
    }
    strict_spherical_micro = {
        fusion: {method: MetricTotals() for method in depth_methods}
        for fusion in STRICT_FUSION_METHODS
    }
    tangent_all_rows: list[dict[str, Any]] = []
    tangent_station_infos: list[dict[str, Any]] = []
    combined_all_rows: list[dict[str, Any]] = []
    combined_station_infos: list[dict[str, Any]] = []
    area_weights = _latitude_area_weights(int(args.pano_height), 2 * int(args.pano_height))

    def consume_regular_station(
        station: StanfordPanorama,
        tangent: TangentPrepared | None,
    ) -> None:
        (
            rows,
            arrays,
            strict_rows,
            strict_arrays,
            station_info,
            route_outputs,
        ) = _evaluate_station(
            station,
            records_by_station[station.camera_uuid],
            cfg,
            model,
            device,
            args,
            tangent,
        )
        all_rows.extend(rows)
        strict_all_rows.extend(strict_rows)
        station_infos.append(station_info)
        for (fusion_method, depth_method), (prediction, target, support) in arrays.items():
            micro[fusion_method][depth_method].update(prediction, target, support)
            spherical_micro[fusion_method][depth_method].update(
                prediction, target, support, area_weights
            )
        for (fusion_method, depth_method), (prediction, target, support) in strict_arrays.items():
            strict_micro[fusion_method][depth_method].update(prediction, target, support)
            strict_spherical_micro[fusion_method][depth_method].update(
                prediction, target, support, area_weights
            )
        if tangent is not None:
            tangent_all_rows.extend(route_outputs["tangent_rows"])
            tangent_station_infos.append(route_outputs["tangent_info"])
            combined_all_rows.extend(route_outputs["combined_rows"])
            combined_station_infos.append(route_outputs["combined_info"])

    if tangent_bundle is None:
        for index, station in enumerate(selected_stations, start=1):
            print(
                f"[{index}/{len(selected_stations)}] {station.room}/{station.camera_uuid} "
                f"({len(records_by_station[station.camera_uuid])} regular views)",
                flush=True,
            )
            consume_regular_station(station, None)
    else:
        selected_regular_ids = {station.camera_uuid for station in selected_stations}
        for index, station in enumerate(route_p_stations, start=1):
            route_kind = "shared" if station.camera_uuid in records_by_station else "pano-only"
            print(
                f"[Route-P {index}/{len(route_p_stations)}] "
                f"{station.room}/{station.camera_uuid} ({route_kind})",
                flush=True,
            )
            tangent = _prepare_tangent_predictions(tangent_bundle, station, args)
            if station.camera_uuid in selected_regular_ids:
                consume_regular_station(station, tangent)
            else:
                tangent_rows, _, tangent_info = _evaluate_tangent_only_station(
                    station,
                    tangent,
                    cfg,
                    args,
                    pano_only=station.camera_uuid in tangent_bundle.pano_only_station_ids,
                )
                tangent_all_rows.extend(tangent_rows)
                tangent_station_infos.append(tangent_info)

    expected_rows = len(selected_stations) * len(FUSION_METHODS) * len(depth_methods)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} station rows, got {len(all_rows)}")
    expected_strict_rows = len(selected_stations) * len(STRICT_FUSION_METHODS) * len(depth_methods)
    if len(strict_all_rows) != expected_strict_rows:
        raise RuntimeError(
            f"Expected {expected_strict_rows} strict-single rows, got {len(strict_all_rows)}"
        )
    if tangent_bundle is not None:
        expected_tangent_rows = (
            len(route_p_stations) * 2 * len(TANGENT_VARIANTS) * len(TANGENT_FUSION_METHODS)
        )
        if len(tangent_all_rows) != expected_tangent_rows:
            raise RuntimeError(
                f"Expected {expected_tangent_rows} Route-P rows, got {len(tangent_all_rows)}"
            )
        expected_combined_rows = len(selected_stations) * 2 * 3 * len(COMBINED_FUSION_METHODS)
        if len(combined_all_rows) != expected_combined_rows:
            raise RuntimeError(
                f"Expected {expected_combined_rows} combined rows, got {len(combined_all_rows)}"
            )
    per_room_rows = _per_room_rows(all_rows)
    strict_per_room_rows = _per_room_rows(strict_all_rows)
    metric_table: dict[str, dict[str, dict[str, Any]]] = {}
    for fusion_method in FUSION_METHODS:
        metric_table[fusion_method] = {}
        for depth_method in depth_methods:
            metric_table[fusion_method][depth_method] = {
                **micro[fusion_method][depth_method].compute(),
                "spherical_area_weighted": spherical_micro[fusion_method][depth_method].compute(),
                "station_macro": _macro_metrics(all_rows, depth_method, fusion_method),
                "room_macro": _room_macro_metrics(per_room_rows, depth_method, fusion_method),
            }
    strict_metric_table: dict[str, dict[str, dict[str, Any]]] = {}
    for fusion_method in STRICT_FUSION_METHODS:
        strict_metric_table[fusion_method] = {}
        for depth_method in depth_methods:
            strict_metric_table[fusion_method][depth_method] = {
                **strict_micro[fusion_method][depth_method].compute(),
                "spherical_area_weighted": strict_spherical_micro[fusion_method][
                    depth_method
                ].compute(),
                "station_macro": _macro_metrics(strict_all_rows, depth_method, fusion_method),
                "room_macro": _room_macro_metrics(
                    strict_per_room_rows, depth_method, fusion_method
                ),
            }

    contrasts = []
    for depth_method in depth_methods:
        for joint_method in FUSION_METHODS[1:]:
            contrasts.append(
                _contrast(
                    metric_table,
                    all_rows,
                    kind="pano_joint_over_per_ray_single_source_mosaic",
                    candidate_depth=depth_method,
                    candidate_fusion=joint_method,
                    reference_depth=depth_method,
                    reference_fusion="single_best_view",
                    seed=int(args.seed),
                    repetitions=int(args.bootstrap_repetitions),
                )
            )
    for fusion_method in FUSION_METHODS:
        for enhanced_method in depth_methods[1:]:
            contrasts.append(
                _contrast(
                    metric_table,
                    all_rows,
                    kind="bim_enhancement_over_raw_da3",
                    candidate_depth=enhanced_method,
                    candidate_fusion=fusion_method,
                    reference_depth="raw_da3",
                    reference_fusion=fusion_method,
                    seed=int(args.seed),
                    repetitions=int(args.bootstrap_repetitions),
                )
            )
        contrasts.append(
            _contrast(
                metric_table,
                all_rows,
                kind="local_bim_correction_over_scale_only",
                candidate_depth="bim_direct",
                candidate_fusion=fusion_method,
                reference_depth="universal_scale",
                reference_fusion=fusion_method,
                seed=int(args.seed),
                repetitions=int(args.bootstrap_repetitions),
            )
        )
        contrasts.append(
            _contrast(
                metric_table,
                all_rows,
                kind="station_local_bim_correction_over_station_scale_only",
                candidate_depth="station_bim_direct",
                candidate_fusion=fusion_method,
                reference_depth="station_bim_scale",
                reference_fusion=fusion_method,
                seed=int(args.seed),
                repetitions=int(args.bootstrap_repetitions),
            )
        )
        contrasts.append(
            _contrast(
                metric_table,
                all_rows,
                kind="station_shared_bim_scale_over_per_view_scale",
                candidate_depth="station_bim_scale",
                candidate_fusion=fusion_method,
                reference_depth="universal_scale",
                reference_fusion=fusion_method,
                seed=int(args.seed),
                repetitions=int(args.bootstrap_repetitions),
            )
        )

    strict_contrasts = []
    for depth_method in depth_methods:
        for joint_method in STRICT_FUSION_METHODS[1:]:
            strict_contrasts.append(
                _contrast(
                    strict_metric_table,
                    strict_all_rows,
                    kind="pano_joint_over_strict_single_frame",
                    candidate_depth=depth_method,
                    candidate_fusion=joint_method,
                    reference_depth=depth_method,
                    reference_fusion="strict_single_frame",
                    seed=int(args.seed),
                    repetitions=int(args.bootstrap_repetitions),
                )
            )

    tangent_metric_table: list[dict[str, Any]] = []
    tangent_pano_only_metric_table: list[dict[str, Any]] = []
    tangent_contrasts: list[dict[str, Any]] = []
    combined_metric_table: list[dict[str, Any]] = []
    combined_contrasts: list[dict[str, Any]] = []
    if tangent_bundle is not None:
        tangent_group_fields = (
            "support_scope",
            "route_variant",
            "fusion_method",
            "depth_method",
        )
        tangent_metric_table = _grouped_route_summary(
            tangent_all_rows,
            tangent_group_fields,
        )
        tangent_pano_only_metric_table = _grouped_route_summary(
            [row for row in tangent_all_rows if bool(row["pano_only"])],
            tangent_group_fields,
        )
        for support_scope in ("native", "common_tangent6_tangent14"):
            for variant in TANGENT_VARIANTS:
                for joint_method in TANGENT_FUSION_METHODS[1:]:
                    tangent_contrasts.append(
                        _route_paired_contrast(
                            tangent_all_rows,
                            kind="tangent_joint_over_per_ray_single_source_mosaic",
                            candidate={
                                "support_scope": support_scope,
                                "route_variant": variant,
                                "fusion_method": joint_method,
                            },
                            reference={
                                "support_scope": support_scope,
                                "route_variant": variant,
                                "fusion_method": "single_best_view",
                            },
                            seed=int(args.seed),
                            repetitions=int(args.bootstrap_repetitions),
                        )
                    )
        for fusion_method in TANGENT_FUSION_METHODS:
            tangent_contrasts.append(
                _route_paired_contrast(
                    tangent_all_rows,
                    kind="tangent14_over_tangent6_view_count_ablation",
                    candidate={
                        "support_scope": "common_tangent6_tangent14",
                        "route_variant": "tangent14",
                        "fusion_method": fusion_method,
                    },
                    reference={
                        "support_scope": "common_tangent6_tangent14",
                        "route_variant": "tangent6",
                        "fusion_method": fusion_method,
                    },
                    seed=int(args.seed),
                    repetitions=int(args.bootstrap_repetitions),
                )
            )

        combined_metric_table = _grouped_route_summary(
            combined_all_rows,
            ("support_scope", "source_set", "fusion_method", "depth_method"),
        )
        for fusion_method in COMBINED_FUSION_METHODS:
            for source in ("regular_plus_tangent6", "regular_plus_tangent14"):
                combined_contrasts.append(
                    _route_paired_contrast(
                        combined_all_rows,
                        kind="regular_plus_pano_tangent_over_regular_only_raw",
                        candidate={
                            "support_scope": "common_regular",
                            "source_set": source,
                            "fusion_method": fusion_method,
                        },
                        reference={
                            "support_scope": "common_regular",
                            "source_set": "regular_only",
                            "fusion_method": fusion_method,
                        },
                        seed=int(args.seed),
                        repetitions=int(args.bootstrap_repetitions),
                    )
                )
            combined_contrasts.append(
                _route_paired_contrast(
                    combined_all_rows,
                    kind="regular_plus_tangent14_over_regular_plus_tangent6",
                    candidate={
                        "support_scope": "common_regular",
                        "source_set": "regular_plus_tangent14",
                        "fusion_method": fusion_method,
                    },
                    reference={
                        "support_scope": "common_regular",
                        "source_set": "regular_plus_tangent6",
                        "fusion_method": fusion_method,
                    },
                    seed=int(args.seed),
                    repetitions=int(args.bootstrap_repetitions),
                )
            )

    per_station_path = output_dir / "per_station.csv"
    fieldnames = list(all_rows[0])
    with per_station_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    strict_per_station_path = output_dir / "strict_single_per_station.csv"
    strict_fieldnames = list(strict_all_rows[0])
    with strict_per_station_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=strict_fieldnames)
        writer.writeheader()
        writer.writerows(strict_all_rows)
    per_room_path = output_dir / "per_room.csv"
    room_fieldnames = list(per_room_rows[0])
    with per_room_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=room_fieldnames)
        writer.writeheader()
        writer.writerows(per_room_rows)
    tangent_per_station_path: Path | None = None
    combined_per_station_path: Path | None = None
    if tangent_bundle is not None:
        tangent_per_station_path = output_dir / "tangent_per_station.csv"
        with tangent_per_station_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(tangent_all_rows[0]))
            writer.writeheader()
            writer.writerows(tangent_all_rows)
        combined_per_station_path = output_dir / "regular_pano_joint_per_station.csv"
        with combined_per_station_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined_all_rows[0]))
            writer.writeheader()
            writer.writerows(combined_all_rows)

    selected_pano_files = {
        f"{station.camera_uuid}/{modality}": {
            "path": str(getattr(station, f"{modality}_path")),
            "sha256": _sha256(getattr(station, f"{modality}_path")),
        }
        for station in selected_stations
        for modality in _PANO_MODALITIES
    }
    evaluated_route_p_pano_only_ids = [
        str(info["station_id"]) for info in tangent_station_infos if bool(info["pano_only"])
    ]
    route_p_pano_depth_files = (
        {
            station.camera_uuid: {
                "path": str(station.depth_path),
                "sha256": _sha256(station.depth_path),
            }
            for station in route_p_stations
        }
        if tangent_bundle is not None
        else {}
    )
    route_p_pano_only_files = (
        {
            camera_uuid: {
                "room": tangent_bundle.stations[camera_uuid]["room"],
                "pano_rgb": {
                    "manifest_recorded_path": tangent_bundle.stations[camera_uuid]["pano_rgb"][
                        "path"
                    ],
                    "resolved_path": str(station_by_id[camera_uuid].rgb_path),
                    "sha256": tangent_bundle.stations[camera_uuid]["pano_rgb"]["sha256"],
                },
                "pano_depth_evaluation_target": route_p_pano_depth_files[camera_uuid],
                "tangent_views": [
                    {
                        "index": record["view"]["index"],
                        "name": record["view"]["name"],
                        "tangent_rgb": {
                            "manifest_recorded_path": record["tangent_rgb"]["path"],
                            "resolved_path": str(
                                tangent_bundle.tangent_rgb_paths[
                                    (camera_uuid, int(record["view"]["index"]))
                                ]
                            ),
                            "sha256": record["tangent_rgb"]["sha256"],
                        },
                        "da3_cache": {
                            **record["da3_cache"],
                            "manifest_recorded_path": record["da3_cache"]["path"],
                            "path": str(
                                tangent_bundle.cache_paths[
                                    (camera_uuid, int(record["view"]["index"]))
                                ]
                            ),
                        },
                    }
                    for record in tangent_bundle.stations[camera_uuid]["tangent_views"]
                ],
            }
            for camera_uuid in tangent_bundle.pano_only_station_ids
            if camera_uuid in route_p_pano_depth_files
        }
        if tangent_bundle is not None
        else {}
    )
    project_root = Path(__file__).resolve().parents[2]
    geometry_code_identity = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in {
            "stanford_pano": project_root / "src/bim_priorda3/data/stanford_pano.py",
            "pano_tangent": project_root / "src/bim_priorda3/data/pano_tangent.py",
        }.items()
    }
    manifest_path = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
    annotation_path = resolve_project_path(cfg, cfg.data.split_annotation)
    provenance = {
        "schema_version": 3,
        "protocol": "stanford-area1-regular-and-pano-tangent-depth-v3",
        "split": args.split,
        "test_access_explicitly_authorized": bool(args.confirm_test),
        "formal_protocol_eligible": args.max_stations is None,
        "seed": int(args.seed),
        "config": {
            "path": str(Path(args.config).resolve()),
            "sha256": _sha256(Path(args.config).resolve()),
        },
        "evaluator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "shared_geometry_code_identity": geometry_code_identity,
        "checkpoint": checkpoint_provenance,
        "universal_scale_protocol": scale_protocol,
        "dataset_split_provenance": dataset.split_provenance,
        "prepared_manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        "split_annotation": {"path": str(annotation_path), "sha256": _sha256(annotation_path)},
        "selected_pano_files": selected_pano_files,
        "prediction_input_contract": {
            "regular_rgb": True,
            "cached_da3_depth_and_confidence": True,
            "regular_bim_render": True,
            "pano_rgb_for_photometric_weight_only": True,
            "regular_ground_truth_arrays_loaded": False,
            "pano_depth_used_only_after_all_predictions_are_frozen": True,
            "pano_depth_runtime_role": "evaluation target only",
        },
        "route_p_tangent_inputs": (
            {
                "status": "verified_formal_manifest",
                "manifest": {
                    "path": str(tangent_bundle.path),
                    "sha256": tangent_bundle.sha256,
                    "schema_version": TANGENT_MANIFEST_SCHEMA_VERSION,
                    "protocol": TANGENT_MANIFEST_PROTOCOL,
                },
                "cache_schema_version": TANGENT_CACHE_SCHEMA_VERSION,
                "evaluation_scope": {
                    "formal_full_manifest": args.max_stations is None,
                    "ordered_prefix_limit": args.max_stations,
                    "evaluated_station_count": len(route_p_stations),
                    "evaluated_station_ids": [station.camera_uuid for station in route_p_stations],
                },
                "identity_checks": [
                    "formal full-split status",
                    "runtime split and explicit test authorization",
                    "config and annotation SHA256",
                    "exact dataset split provenance",
                    "pinned DA3 name/revision/process_res/load policy",
                    "station UUID/room/panorama RGB SHA256",
                    "every tangent PNG and DA3 cache SHA256",
                    "every schema-v2 cache metadata field",
                    "nested14 preset K and T_face_from_pano rebuilt via shared pano_tangent",
                ],
                "prediction_quantity": "cached tangent perspective z-depth converted to ERP radial range",
                "weight": "face_centrality**centrality_power * cached_DA3_confidence",
                "gt_or_bim_used_for_scale_weight_or_selection": False,
                "photometric_method": (
                    "not applicable: all tangent RGB views are deterministic resamplings "
                    "of the same panorama RGB"
                ),
                "view_count_ablation": {
                    "tangent6": "first six nested14 views; exactly shared cubemap6 geometry",
                    "tangent14": "all nested14 views",
                },
                "tangent_only_supports": {
                    "native": "valid pano GT intersected with each variant's own coverage",
                    "common_tangent6_tangent14": (
                        "valid pano GT intersected with both variants; view-count quality claim"
                    ),
                },
                "combined_supports": {
                    "common_regular": (
                        "regular-only coverage intersected with valid pano GT; fixed across "
                        "regular-only and both combined source sets"
                    ),
                    "native_union": "valid pano GT intersected with each source set's own coverage",
                },
                "synchronization_gauge": {
                    "tangent6_and_tangent14": (
                        "base-weighted zero mean over the identical first-six cubemap views"
                    ),
                    "regular_plus_tangent": (
                        "base-weighted zero mean over the ordered regular contributors, "
                        "matching the regular-only anchor set"
                    ),
                    "purpose": (
                        "prevent view-count changes from introducing an arbitrary global "
                        "log-scale shift"
                    ),
                },
                "claim_boundary": (
                    "Route-P uses raw DA3 only and is reported independently from Route-R "
                    "BIM/direct/learned comparisons"
                ),
                "official_pano_depth_evaluation_targets": route_p_pano_depth_files,
                "pano_only_file_identities": route_p_pano_only_files,
            }
            if tangent_bundle is not None
            else {"status": "not_requested"}
        ),
        "geometry": {
            "projection": "pinhole z-depth to same-center equirectangular radial range",
            "pano_shape": [int(args.pano_height), 2 * int(args.pano_height)],
            "single_best_view_semantics": (
                "per-ray single-source mosaic: independently choose the maximum "
                "centrality/confidence contributor at each ERP pixel"
            ),
            "centrality_power": float(args.centrality_power),
            "confidence_floor": float(args.confidence_floor),
        },
        "strict_single_frame_protocol": {
            "selector": "one projected regular frame with maximum sum(base_weights)",
            "tie_break": "lexicographically smallest stable frame_id",
            "selector_inputs": "projection centrality and cached DA3 confidence only",
            "gt_used_by_selector": False,
            "bim_used_by_selector": False,
            "pano_rgb_used_by_selector": False,
            "support": "selected frame coverage intersected with valid pano GT",
            "joint_comparators": list(STRICT_FUSION_METHODS[1:]),
            "identical_support_for_all_depth_and_strict_comparison_methods": True,
        },
        "fusion": {
            "methods": list(FUSION_METHODS),
            "domain": "log radial range",
            "huber_log_delta": float(args.huber_log_delta),
            "consistency_log_threshold": float(args.consistency_log_threshold),
            "photometric_sigma": float(args.photo_sigma),
            "photometric_exposure_alignment": "per-view median log-luminance gain, clipped [0.5,2]",
            "overlap_scale_synchronization": {
                "input": "prediction overlap plus centrality/confidence only; no GT",
                "pair_location": "median initialization plus three Huber IRLS iterations",
                "min_overlap": int(args.sync_min_overlap),
                "pair_max_deterministic_samples": int(args.sync_pair_max_samples),
                "huber_log_delta": float(args.sync_huber_log_delta),
                "graph_constraint": (
                    "weighted least squares, registered-anchor zero-mean KKT constraint, L2-to-zero"
                ),
                "l2": float(args.sync_l2),
                "max_abs_log_offset": float(args.sync_max_abs_offset),
            },
            "station_bim_scale": {
                "scope": "one universal BIM/DA3 scale shared by all regular views",
                "samples_per_view_cap": int(args.station_scale_samples_per_view),
                "sampling": "deterministic evenly spaced valid raster-order indices",
                "local_variant": "same registered deterministic local correction per view",
            },
        },
        "evaluation_support": {
            "depth_range_m": [float(cfg.data.min_depth), float(cfg.data.max_depth)],
            "definition": "pano GT valid and covered by at least one regular view",
            "identical_for_every_depth_and_fusion_method": True,
            "pixel_micro_role": "secondary diagnostic",
            "exact_erp_pixel_solid_angle_metrics_reported": True,
            "formal_primary": "equal-station macro of exact solid-angle-weighted metrics",
            "uncertainty_resampling_unit": "room cluster",
        },
        "station_selection": {
            "all_discovered_pano_stations": len(all_stations),
            "all_paired_pano_stations": len(all_paired_panos),
            "all_pano_only_station_ids": all_pano_only_ids,
            "split_total_pano_stations": len(split_panos),
            "split_regular_stations": len(records_by_station),
            "selected_paired_stations": len(selected_stations),
            "regular_stations_missing_pano": missing_pano,
            "split_pano_only_station_ids": pano_only,
            "shared_station_fraction_of_split_panos": shared_available_count / len(split_panos),
            "max_stations_ordered_prefix": args.max_stations,
            "selected_station_ids": [station.camera_uuid for station in selected_stations],
        },
        "per_station_csv": {"path": str(per_station_path), "sha256": _sha256(per_station_path)},
        "strict_single_per_station_csv": {
            "path": str(strict_per_station_path),
            "sha256": _sha256(strict_per_station_path),
        },
        "per_room_csv": {"path": str(per_room_path), "sha256": _sha256(per_room_path)},
        "tangent_per_station_csv": (
            {
                "path": str(tangent_per_station_path),
                "sha256": _sha256(tangent_per_station_path),
            }
            if tangent_per_station_path is not None
            else None
        ),
        "regular_pano_joint_per_station_csv": (
            {
                "path": str(combined_per_station_path),
                "sha256": _sha256(combined_per_station_path),
            }
            if combined_per_station_path is not None
            else None
        ),
    }
    provenance_path = output_dir / "provenance.json"
    _write_json(provenance_path, provenance)
    summary = {
        "schema_version": 3,
        "protocol": provenance["protocol"],
        "split": args.split,
        "formal_protocol_eligible": provenance["formal_protocol_eligible"],
        "depth_methods": depth_methods,
        "fusion_methods": list(FUSION_METHODS),
        "fusion_method_semantics": {
            "single_best_view": "per-ray single-source mosaic, not a strict single frame",
            "strict_single_frame_location": "strict_single_frame_evaluation",
        },
        "strict_comparison_methods": list(STRICT_FUSION_METHODS),
        "primary_metric_protocol": {
            "metric": "AbsRel",
            "pixel_weighting": (
                "exact ERP pixel solid angle: [sin(latitude_high)-sin(latitude_low)] * 2*pi/width"
            ),
            "station_aggregation": "equal-station macro",
            "room_macro_supporting_analysis": (
                "mean stations within each room, then equal mean across rooms"
            ),
            "bootstrap_resampling_unit": "room cluster",
            "fixed_support_shared_by_all_methods": True,
        },
        "area1_pano_inventory": {
            "total": len(all_stations),
            "paired_with_regular": len(all_paired_panos),
            "pano_only": len(all_pano_only_ids),
            "pano_only_station_ids": all_pano_only_ids,
        },
        "station_count": len(selected_stations),
        "split_pano_station_count": len(split_panos),
        "shared_station_count_before_optional_prefix": shared_available_count,
        "pano_only_station_ids": pano_only,
        "shared_station_fraction_of_split_panos": shared_available_count / len(split_panos),
        "fixed_support_pixels": sum(int(info["fixed_support_pixels"]) for info in station_infos),
        "mean_regular_coverage_fraction": float(
            np.mean([float(info["regular_coverage_fraction"]) for info in station_infos])
        ),
        "mean_multi_view_overlap_fraction": float(
            np.mean([float(info["multi_view_overlap_fraction"]) for info in station_infos])
        ),
        "mean_solid_angle_coverage_fraction": float(
            np.mean([float(info["solid_angle_coverage_fraction"]) for info in station_infos])
        ),
        "metrics": metric_table,
        "primary_metrics": {
            fusion_method: {
                depth_method: {
                    name: metric_table[fusion_method][depth_method]["station_macro"][
                        f"spherical_{name}"
                    ]
                    for name in METRIC_NAMES
                }
                for depth_method in depth_methods
            }
            for fusion_method in FUSION_METHODS
        },
        "contrasts": contrasts,
        "strict_single_frame_evaluation": {
            "selector": "max sum(base_weights), stable frame_id tie-break",
            "support": "selected-frame coverage intersected with valid pano GT",
            "mean_pixel_coverage_fraction": float(
                np.mean(
                    [
                        float(info["strict_single_frame"]["coverage_fraction"])
                        for info in station_infos
                    ]
                )
            ),
            "mean_solid_angle_coverage_fraction": float(
                np.mean(
                    [
                        float(info["strict_single_frame"]["solid_angle_coverage_fraction"])
                        for info in station_infos
                    ]
                )
            ),
            "metrics": strict_metric_table,
            "primary_metrics": {
                fusion_method: {
                    depth_method: {
                        name: strict_metric_table[fusion_method][depth_method]["station_macro"][
                            f"spherical_{name}"
                        ]
                        for name in METRIC_NAMES
                    }
                    for depth_method in depth_methods
                }
                for fusion_method in STRICT_FUSION_METHODS
            },
            "contrasts": strict_contrasts,
        },
        "route_p_tangent_only": (
            {
                "status": "evaluated",
                "claim": (
                    "DA3 on deterministic pano tangents, without BIM, learned refinement, "
                    "GT-derived scale, or GT-derived view weighting"
                ),
                "station_count": len(tangent_station_infos),
                "shared_station_count": len(tangent_station_infos)
                - len(evaluated_route_p_pano_only_ids),
                "pano_only_station_count": len(evaluated_route_p_pano_only_ids),
                "pano_only_station_ids": evaluated_route_p_pano_only_ids,
                "variants": dict(TANGENT_VARIANTS),
                "fusion_methods": list(TANGENT_FUSION_METHODS),
                "photo_method_status": "not_applicable_same_source_pano_rgb",
                "formal_primary": (
                    "equal-station macro of exact ERP solid-angle metrics; "
                    "room-cluster paired bootstrap"
                ),
                "metrics": tangent_metric_table,
                "pano_only_descriptive_metrics": tangent_pano_only_metric_table,
                "contrasts": tangent_contrasts,
                "stations": tangent_station_infos,
            }
            if tangent_bundle is not None
            else {"status": "not_requested"}
        ),
        "route_p_regular_plus_pano_raw": (
            {
                "status": "evaluated",
                "claim": (
                    "raw regular DA3 plus raw pano-tangent DA3 versus raw regular DA3; "
                    "no BIM/direct/learned method is included"
                ),
                "station_count": len(combined_station_infos),
                "source_sets": [
                    "regular_only",
                    "regular_plus_tangent6",
                    "regular_plus_tangent14",
                ],
                "fusion_methods": list(COMBINED_FUSION_METHODS),
                "quality_support": "common_regular",
                "coverage_support": "native_union",
                "metrics": combined_metric_table,
                "contrasts": combined_contrasts,
                "stations": combined_station_infos,
            }
            if tangent_bundle is not None
            else {"status": "not_requested"}
        ),
        "stations": station_infos,
        "artifacts": {
            "per_station_csv": str(per_station_path),
            "per_station_csv_sha256": _sha256(per_station_path),
            "strict_single_per_station_csv": str(strict_per_station_path),
            "strict_single_per_station_csv_sha256": _sha256(strict_per_station_path),
            "per_room_csv": str(per_room_path),
            "per_room_csv_sha256": _sha256(per_room_path),
            "tangent_per_station_csv": (
                str(tangent_per_station_path) if tangent_per_station_path is not None else None
            ),
            "tangent_per_station_csv_sha256": (
                _sha256(tangent_per_station_path) if tangent_per_station_path is not None else None
            ),
            "regular_pano_joint_per_station_csv": (
                str(combined_per_station_path) if combined_per_station_path is not None else None
            ),
            "regular_pano_joint_per_station_csv_sha256": (
                _sha256(combined_per_station_path)
                if combined_per_station_path is not None
                else None
            ),
            "provenance": str(provenance_path),
            "provenance_sha256": _sha256(provenance_path),
        },
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
