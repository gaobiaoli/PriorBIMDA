#!/usr/bin/env python3
"""Render Stanford Area_1 panorama tangents and cache pinned DA3 predictions.

This is a prediction-input preparation command. It reads panorama RGB and pose
metadata only. Panorama depth, panorama semantics, and regular-view ground truth
are never decoded or used. Test split access requires an explicit confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.config import Config, load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.pano_tangent import (
    PANO_TANGENT_PRESETS,
    TangentView,
    build_pano_tangent_preset,
    erp_rgb_to_tangent,
)
from bim_priorda3.data.preparation import DA3PredictionProvider, sha256_file
from bim_priorda3.data.stanford_pano import StanfordPanorama, discover_stanford_panoramas

MANIFEST_SCHEMA_VERSION = 1
PROTOCOL = "stanford-area1-pano-tangent-da3-cache-v1"


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache pinned DA3 predictions for Stanford Area_1 panorama tangents"
    )
    parser.add_argument("--config", default="configs/stanford_area1.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--confirm-test",
        action="store_true",
        help="Required with --split test to prevent accidental test-set prediction access",
    )
    parser.add_argument("--preset", choices=PANO_TANGENT_PRESETS, default="nested14")
    parser.add_argument("--face-resolution", type=_positive_int, default=504)
    parser.add_argument("--max-stations", type=_positive_int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--log-every", type=_positive_int, default=1)
    args = parser.parse_args(argv)
    if args.split == "test" and not args.confirm_test:
        parser.error("--split test requires the explicit --confirm-test flag")
    if args.split != "test" and args.confirm_test:
        parser.error("--confirm-test is only valid with --split test")
    return args


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _view_geometry(view: TangentView, index: int) -> dict[str, Any]:
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


def _preset_identity(
    preset: str, face_resolution: int
) -> tuple[tuple[TangentView, ...], dict[str, Any]]:
    views = build_pano_tangent_preset(preset, face_resolution)
    geometry = [_view_geometry(view, index) for index, view in enumerate(views)]
    fingerprint = _sha256_bytes(
        json.dumps(geometry, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    identity = {
        "name": preset,
        "view_count": len(views),
        "geometry_fingerprint_sha256": fingerprint,
        "cache_namespace": f"{preset}_r{face_resolution}_{fingerprint[:12]}",
        "views": geometry,
    }
    return views, identity


def _read_pano_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read panorama RGB: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Panorama RGB must be uint8 BGR/BGRA, got {image.dtype} {image.shape}")
    return np.ascontiguousarray(image[..., :3][..., ::-1])


def _decode_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot decode existing tangent PNG: {path}")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Existing tangent PNG must be uint8 three-channel: {path}")
    return np.ascontiguousarray(image[..., ::-1])


def _publish_bytes_once(path: Path, payload: bytes) -> str:
    """Atomically publish bytes without replacing a concurrent/existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"Refusing to replace a different immutable artifact: {path}")
        return "reused"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != payload:
                raise FileExistsError(
                    f"Concurrent writer published a different artifact: {path}"
                ) from None
        return "generated"
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_tangent_png(path: Path, rgb: np.ndarray) -> tuple[str, str]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Rendered tangent RGB must be uint8 with three channels")
    if path.exists():
        existing = _decode_rgb(path)
        if existing.shape != rgb.shape or not np.array_equal(existing, rgb):
            raise FileExistsError(f"Existing tangent PNG pixels differ from current render: {path}")
        return sha256_file(path), "reused"
    encoded_ok, encoded = cv2.imencode(".png", rgb[..., ::-1])
    if not encoded_ok:
        raise RuntimeError(f"OpenCV could not encode tangent PNG: {path}")
    status = _publish_bytes_once(path, encoded.tobytes())
    if not np.array_equal(_decode_rgb(path), rgb):
        raise RuntimeError(f"Atomic tangent PNG verification failed: {path}")
    return sha256_file(path), status


def _select_split_stations(
    cfg: Config,
    split: str,
) -> tuple[list[StanfordPanorama], dict[str, Any], dict[str, Any]]:
    """Resolve a room-disjoint annotation split without opening any GT array."""

    if not cfg.data.get("split_annotation"):
        raise ValueError("Panorama caching requires data.split_annotation")
    dataset = BIMDepthDataset(cfg, split, augment=False, require_ground_truth=False)
    if dataset.split_provenance.get("mode") != "annotations":
        raise ValueError("Panorama caching requires annotation-based split provenance")
    selected_rooms = {str(record["region"]) for record in dataset.records}
    regular_station_rooms: dict[str, str] = {}
    for record in dataset.records:
        camera_uuid = str(record.get("camera_uuid", ""))
        room = str(record["region"])
        if not camera_uuid:
            raise ValueError(f"Annotation-selected record lacks camera_uuid: {record.get('id')}")
        previous = regular_station_rooms.setdefault(camera_uuid, room)
        if previous != room:
            raise ValueError(f"camera_uuid {camera_uuid} appears in two rooms: {previous}/{room}")

    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    panoramas = discover_stanford_panoramas(area_root)
    selected = sorted(
        (station for station in panoramas if station.room in selected_rooms),
        key=lambda station: (station.room, station.camera_uuid),
    )
    if not selected:
        raise RuntimeError(f"No panorama station belongs to annotation split {split!r}")
    selected_by_uuid = {station.camera_uuid: station for station in selected}
    for camera_uuid, room in regular_station_rooms.items():
        station = selected_by_uuid.get(camera_uuid)
        if station is not None and station.room != room:
            raise ValueError(
                f"Regular/panorama room mismatch for {camera_uuid}: {room}/{station.room}"
            )
    shared = sorted(set(selected_by_uuid) & set(regular_station_rooms))
    pano_only = sorted(set(selected_by_uuid) - set(regular_station_rooms))
    selection = {
        "room_source": "configured exhaustive split_annotation via BIMDepthDataset",
        "rooms": sorted(selected_rooms),
        "annotation_regular_station_count": len(regular_station_rooms),
        "split_pano_station_count": len(selected),
        "shared_regular_pano_station_count": len(shared),
        "pano_only_station_ids": pano_only,
    }
    return selected, dict(dataset.split_provenance), selection


def _manifest_path(
    preset_root: Path,
    split: str,
    max_stations: int | None,
) -> Path:
    suffix = "full" if max_stations is None else f"exploratory_max{max_stations}"
    return preset_root / "manifests" / f"{split}_{suffix}.json"


def _verify_prediction(
    prediction: Any,
    provider: DA3PredictionProvider,
    tangent_sha256: str,
    target_shape: tuple[int, int],
) -> None:
    if prediction.image_sha256 != tangent_sha256:
        raise RuntimeError("DA3 cache is not bound to the exact tangent PNG")
    if prediction.model_name != provider.model_name:
        raise RuntimeError("DA3 cache model_name differs from the active provider")
    if prediction.model_revision != provider.model_revision:
        raise RuntimeError("DA3 cache model_revision differs from the active provider")
    if int(prediction.process_res) != provider.process_res:
        raise RuntimeError("DA3 cache process_res differs from the active provider")
    if tuple(prediction.target_shape) != target_shape:
        raise RuntimeError("DA3 cache target_shape differs from the tangent shape")
    cache_path = Path(prediction.cache_path).resolve()
    if not cache_path.is_file() or sha256_file(cache_path) != prediction.cache_sha256:
        raise RuntimeError(f"DA3 cache SHA verification failed: {cache_path}")


def _preflight_existing_manifest(
    path: Path, static_payload: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot parse existing immutable manifest {path}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"Existing manifest must be a JSON object: {path}")
    for key, expected in static_payload.items():
        if payload.get(key) != expected:
            raise FileExistsError(
                f"Existing manifest has incompatible {key!r}; use a different --output-root: {path}"
            )
    stations = payload.get("stations")
    if not isinstance(stations, list):
        raise TypeError(f"Existing manifest stations must be a list: {path}")
    camera_uuids = [str(item.get("camera_uuid", "")) for item in stations if isinstance(item, dict)]
    if len(camera_uuids) != len(stations) or any(not value for value in camera_uuids):
        raise ValueError(f"Existing manifest contains malformed station entries: {path}")
    if len(camera_uuids) != len(set(camera_uuids)):
        raise ValueError(f"Existing manifest contains duplicate camera_uuid entries: {path}")
    return payload


def cache_stanford_pano_da3(args: argparse.Namespace) -> dict[str, Any]:
    if args.split == "test" and not args.confirm_test:
        raise ValueError("test split caching requires explicit confirm_test=True")
    if args.split not in {"val", "test"}:
        raise ValueError("split must be 'val' or 'test'")
    if args.max_stations is not None and int(args.max_stations) < 1:
        raise ValueError("max_stations must be positive")
    if int(args.log_every) < 1:
        raise ValueError("log_every must be positive")

    config_path = Path(args.config).expanduser().resolve()
    cfg = load_config(config_path)
    views, preset = _preset_identity(str(args.preset), int(args.face_resolution))
    base_output = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root is not None
        else resolve_project_path(cfg, cfg.data.processed_root) / "pano_da3"
    )
    preset_root = base_output / str(preset["cache_namespace"])
    manifest_path = _manifest_path(preset_root, str(args.split), args.max_stations)
    provider = DA3PredictionProvider(
        cfg,
        f"stanford_pano_{preset['cache_namespace']}",
        preset_root / "da3_cache",
    )
    if provider.model_revision == "UNPINNED":
        raise ValueError("Panorama DA3 caching requires a pinned data.da3_revision")

    stations, split_provenance, selection = _select_split_stations(cfg, str(args.split))
    full_station_count = len(stations)
    if args.max_stations is not None:
        stations = stations[: int(args.max_stations)]
    selected_station_ids = [station.camera_uuid for station in stations]
    selection = {
        **selection,
        "selected_station_count": len(stations),
        "selected_station_ids": selected_station_ids,
        "max_stations": args.max_stations,
        "formal_protocol_eligible": args.max_stations is None,
        "full_split_station_count_before_exploratory_limit": full_station_count,
    }
    annotation_path = resolve_project_path(cfg, cfg.data.split_annotation)
    static_payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "split": str(args.split),
        "test_access_explicitly_authorized": bool(args.confirm_test),
        "preset": preset,
        "selection": selection,
        "split_provenance": split_provenance,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "split_annotation": {
            "path": str(annotation_path),
            "sha256": sha256_file(annotation_path),
        },
        "model": {
            "name": provider.model_name,
            "revision": provider.model_revision,
            "process_res": provider.process_res,
            "local_files_only": provider.local_files_only,
        },
        "code": {
            "cache_script_sha256": sha256_file(Path(__file__).resolve()),
            "pano_tangent_module_sha256": sha256_file(
                Path(__file__).resolve().parents[2]
                / "src"
                / "bim_priorda3"
                / "data"
                / "pano_tangent.py"
            ),
        },
        "input_contract": {
            "pano_rgb_decoded": True,
            "pano_pose_metadata_decoded": True,
            "pano_depth_decoded": False,
            "pano_semantic_decoded": False,
            "regular_ground_truth_decoded": False,
            "output_depth_quantity": "perspective_z_depth_m",
        },
    }
    existing_manifest = _preflight_existing_manifest(manifest_path, static_payload)
    existing_station_lookup = (
        {str(item["camera_uuid"]): item for item in existing_manifest.get("stations", [])}
        if existing_manifest is not None
        else {}
    )

    station_records: list[dict[str, Any]] = []
    generated_pngs = 0
    reused_pngs = 0
    start = time.perf_counter()
    target_shape = (int(args.face_resolution), int(args.face_resolution))
    for station_index, station in enumerate(stations, start=1):
        pano_sha256 = sha256_file(station.rgb_path)
        previous_station = existing_station_lookup.get(station.camera_uuid)
        if (
            previous_station is not None
            and previous_station.get("pano_rgb", {}).get("sha256") != pano_sha256
        ):
            raise FileExistsError(
                f"Panorama RGB changed since immutable manifest publication: {station.rgb_path}"
            )
        pano_rgb = _read_pano_rgb(station.rgb_path)
        tangent_records = []
        for view_index, view in enumerate(views):
            tangent_rgb = erp_rgb_to_tangent(pano_rgb, view)
            stem = (
                f"pano_{station.camera_uuid}__{preset['cache_namespace']}__"
                f"{view_index:02d}_{view.name}"
            )
            tangent_path = (
                preset_root / "tangent_rgb" / station.camera_uuid / f"{stem}.png"
            ).resolve()
            tangent_sha256, png_status = _ensure_tangent_png(tangent_path, tangent_rgb)
            generated_pngs += int(png_status == "generated")
            reused_pngs += int(png_status == "reused")
            prediction = provider.get_with_provenance(tangent_path, target_shape)
            _verify_prediction(prediction, provider, tangent_sha256, target_shape)
            tangent_records.append(
                {
                    "view": _view_geometry(view, view_index),
                    "pano_rgb_sha256": pano_sha256,
                    "tangent_rgb": {
                        "path": str(tangent_path),
                        "sha256": tangent_sha256,
                    },
                    "da3_cache": {
                        "path": str(Path(prediction.cache_path).resolve()),
                        "sha256": prediction.cache_sha256,
                        "image_sha256": prediction.image_sha256,
                        "model_name": prediction.model_name,
                        "model_revision": prediction.model_revision,
                        "process_res": int(prediction.process_res),
                        "target_shape": [int(value) for value in prediction.target_shape],
                        "provenance_status": prediction.provenance_status,
                        "depth_quantity": "perspective_z_depth_m",
                    },
                }
            )
        station_record = {
            "camera_uuid": station.camera_uuid,
            "room": station.room,
            "pano_rgb": {"path": str(station.rgb_path.resolve()), "sha256": pano_sha256},
            "tangent_views": tangent_records,
        }
        if previous_station is not None and previous_station != station_record:
            raise FileExistsError(
                f"Recomputed station differs from immutable manifest: {station.camera_uuid}"
            )
        station_records.append(station_record)
        if (
            station_index == 1
            or station_index % int(args.log_every) == 0
            or station_index == len(stations)
        ):
            elapsed = time.perf_counter() - start
            remaining = elapsed / station_index * (len(stations) - station_index)
            print(
                f"[{station_index}/{len(stations)}] {station.room}/{station.camera_uuid}; "
                f"views={len(views)} elapsed={elapsed:.1f}s ETA={remaining / 60.0:.1f}min",
                flush=True,
            )

    manifest = {**static_payload, "stations": station_records}
    manifest_status = _publish_bytes_once(manifest_path, _canonical_json_bytes(manifest))
    manifest_sha256 = sha256_file(manifest_path)
    elapsed = time.perf_counter() - start
    print(
        f"Manifest {manifest_status}: {manifest_path} ({manifest_sha256}); "
        f"PNG generated/reused={generated_pngs}/{reused_pngs}; elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "manifest_status": manifest_status,
        "generated_pngs": generated_pngs,
        "reused_pngs": reused_pngs,
        "elapsed_s": elapsed,
    }


def main(argv: list[str] | None = None) -> None:
    cache_stanford_pano_da3(parse_args(argv))


if __name__ == "__main__":
    main()
