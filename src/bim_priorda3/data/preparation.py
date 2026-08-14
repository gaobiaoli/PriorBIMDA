from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from plyfile import PlyData

from bim_priorda3.config import Config, resolve_project_path, resolve_slabim_root
from bim_priorda3.data.geometry import (
    approximate_depth_normals,
    depth_edges,
    fuse_front_depth_cluster,
    project_depth,
    read_poses,
    read_timestamps,
    scale_intrinsics,
    slam_global_to_lidar_local,
    synchronize,
    transform_points,
)

MESH_NAMES = ("walls_tri.ply", "walls.ply", "floors.ply", "doors.ply", "columns.ply")
DA3_CACHE_SCHEMA_VERSION = 2
DA3_CACHE_REQUIRED_KEYS = {
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_savez_compressed(path: Path, **payload: Any) -> None:
    """Write an NPZ without exposing a partially written cache to other workers."""

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


@dataclass(frozen=True)
class DA3Prediction:
    depth: np.ndarray
    confidence: np.ndarray
    source: str
    cache_path: Path
    cache_sha256: str
    image_sha256: str
    model_name: str
    model_revision: str
    process_res: int
    target_shape: tuple[int, int]
    provenance_status: str


def _load_polygon_ply(path: Path) -> o3d.t.geometry.TriangleMesh:
    """Load Rhino PLYs and explicitly triangulate polygon faces."""
    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    vertices = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float32)
    polygon_faces = ply["face"]["vertex_indices"]
    lengths = np.fromiter((len(face) for face in polygon_faces), dtype=np.int32)
    triangles = []
    for size in np.unique(lengths):
        if size < 3:
            continue
        faces = np.stack(polygon_faces[lengths == size]).astype(np.int32)
        for offset in range(1, int(size) - 1):
            triangles.append(faces[:, (0, offset, offset + 1)])
    if not triangles:
        raise RuntimeError(f"No polygon faces could be triangulated: {path}")
    indices = np.concatenate(triangles)
    return o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
        o3d.core.Tensor(indices, dtype=o3d.core.Dtype.Int32),
    )


def build_bim_scene(mesh_dir: Path) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    added = 0
    wall_added = False
    for name in MESH_NAMES:
        if name.startswith("walls") and wall_added:
            continue
        path = mesh_dir / name
        if not path.exists():
            continue
        if name == "walls.ply":
            tensor_mesh = _load_polygon_ply(path)
        else:
            mesh = o3d.io.read_triangle_mesh(str(path))
            if not len(mesh.triangles):
                continue
            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(tensor_mesh)
        added += 1
        wall_added |= name.startswith("walls")
    if added == 0:
        raise RuntimeError(f"No usable BIM triangle meshes found in {mesh_dir}")
    return scene


def render_bim(
    scene: o3d.t.geometry.RaycastingScene,
    intrinsic: np.ndarray,
    camera_to_bim: np.ndarray,
    height: int,
    width: int,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float32) + 0.5,
        np.arange(height, dtype=np.float32) + 0.5,
    )
    pixels = np.stack((u, v, np.ones_like(u)), axis=-1)
    directions_camera = pixels @ np.linalg.inv(intrinsic).T
    directions_bim = directions_camera @ camera_to_bim[:3, :3].T
    origins = np.broadcast_to(camera_to_bim[:3, 3], directions_bim.shape)
    rays = np.concatenate((origins, directions_bim), axis=-1).astype(np.float32)
    result = scene.cast_rays(o3d.core.Tensor(rays))
    depth = result["t_hit"].numpy().astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth)
    depth[~valid] = np.nan

    if "primitive_normals" in result:
        normals_bim = result["primitive_normals"].numpy().astype(np.float32)
        normals_camera = normals_bim @ camera_to_bim[:3, :3]
        normals_camera[~valid] = 0.0
        normals = normals_camera.transpose(2, 0, 1)
    else:
        normals = approximate_depth_normals(depth, intrinsic)
    return depth, normals.astype(np.float32)


class DA3PredictionProvider:
    """Load provenance-bound DA3 output, or lazily infer a new atomic cache."""

    def __init__(self, cfg: Config, region_name: str, output_cache: Path) -> None:
        cache_value = cfg.data.da3_cache_roots.get(region_name)
        self.read_cache = Path(cache_value).expanduser().resolve() if cache_value else None
        self.write_cache = output_cache
        self.write_cache.mkdir(parents=True, exist_ok=True)
        self.model_name = str(cfg.data.da3_model)
        revision_value = cfg.data.get("da3_revision")
        self.model_revision = (
            str(revision_value).strip() if revision_value is not None else "UNPINNED"
        )
        if not self.model_revision:
            self.model_revision = "UNPINNED"
        self.local_files_only = bool(
            cfg.data.get(
                "da3_local_files_only",
                cfg.data.get("local_files_only", False),
            )
        )
        if bool(cfg.data.get("da3_require_pinned_revision", False)) and (
            self.model_revision == "UNPINNED"
        ):
            raise ValueError(
                "data.da3_require_pinned_revision=true requires a non-empty data.da3_revision"
            )
        self.process_res = int(cfg.data.da3_process_res)
        if self.process_res < 1:
            raise ValueError("data.da3_process_res must be positive")
        self.model = None

    @staticmethod
    def _confidence_fallback(depth: np.ndarray) -> np.ndarray:
        log_depth = np.log(np.maximum(depth, 1e-4))
        laplacian = np.abs(cv2.Laplacian(log_depth, cv2.CV_32F))
        scale = np.quantile(laplacian, 0.9) + 1e-6
        return np.exp(-laplacian / scale).astype(np.float32)

    @staticmethod
    def _scalar_string(item: np.lib.npyio.NpzFile, key: str, path: Path) -> str:
        value = item[key]
        if value.shape != () or value.dtype.kind not in {"U", "S"}:
            raise ValueError(f"{path}: {key} must be a scalar string")
        result = str(value.item())
        if not result:
            raise ValueError(f"{path}: {key} must not be empty")
        return result

    @staticmethod
    def _scalar_integer(item: np.lib.npyio.NpzFile, key: str, path: Path) -> int:
        value = item[key]
        if value.shape != () or value.dtype.kind not in {"i", "u"}:
            raise ValueError(f"{path}: {key} must be a scalar integer")
        return int(value.item())

    def _validate_cache(
        self,
        path: Path,
        image_path: Path,
        shape: tuple[int, int],
        *,
        image_sha256: str | None = None,
    ) -> DA3Prediction:
        expected_image_sha = image_sha256 or sha256_file(image_path)
        try:
            with np.load(path, allow_pickle=False) as item:
                missing = sorted(DA3_CACHE_REQUIRED_KEYS - set(item.files))
                if missing:
                    raise ValueError(
                        f"{path}: legacy/invalid DA3 cache lacks {missing}; run "
                        "scripts/data/cache_stanford_da3.py --finalize-legacy with an "
                        "explicit generation attestation"
                    )
                schema_version = self._scalar_integer(item, "schema_version", path)
                if schema_version != DA3_CACHE_SCHEMA_VERSION:
                    raise ValueError(
                        f"{path}: unsupported DA3 cache schema {schema_version}; "
                        f"expected {DA3_CACHE_SCHEMA_VERSION}"
                    )
                cache_shape_array = item["target_shape"]
                if cache_shape_array.shape != (2,) or cache_shape_array.dtype.kind not in {
                    "i",
                    "u",
                }:
                    raise ValueError(f"{path}: target_shape must be a two-integer vector")
                cache_shape = tuple(int(value) for value in cache_shape_array)
                if cache_shape != tuple(shape):
                    raise ValueError(
                        f"{path}: target_shape={cache_shape} differs from requested {shape}"
                    )
                depth = item["depth"].astype(np.float32)
                confidence = item["confidence"].astype(np.float32)
                if depth.shape != shape or confidence.shape != shape:
                    raise ValueError(
                        f"{path}: depth/confidence shapes {depth.shape}/{confidence.shape} "
                        f"must both equal {shape}"
                    )
                if not np.isfinite(depth).all() or np.any(depth <= 0):
                    raise ValueError(f"{path}: depth must be finite and strictly positive")
                if not np.isfinite(confidence).all() or np.any(confidence < 0):
                    raise ValueError(f"{path}: confidence must be finite and non-negative")
                cache_image_sha = self._scalar_string(item, "image_sha256", path)
                cache_model = self._scalar_string(item, "model_name", path)
                cache_revision = self._scalar_string(item, "model_revision", path)
                status = self._scalar_string(item, "provenance_status", path)
                process_res = self._scalar_integer(item, "process_res", path)
                local_files_only = item["local_files_only"]
                if local_files_only.shape != () or local_files_only.dtype.kind != "b":
                    raise ValueError(f"{path}: local_files_only must be a scalar boolean")
                if cache_image_sha != expected_image_sha:
                    raise ValueError(f"{path}: image_sha256 does not match {image_path}")
                if cache_model != self.model_name:
                    raise ValueError(
                        f"{path}: model_name={cache_model!r} differs from "
                        f"configured {self.model_name!r}"
                    )
                if cache_revision != self.model_revision:
                    raise ValueError(
                        f"{path}: model_revision={cache_revision!r} differs from "
                        f"configured {self.model_revision!r}"
                    )
                if process_res != self.process_res:
                    raise ValueError(
                        f"{path}: process_res={process_res} differs from configured "
                        f"{self.process_res}"
                    )
                if bool(local_files_only.item()) != self.local_files_only:
                    raise ValueError(
                        f"{path}: local_files_only differs from the configured load policy"
                    )
                if status not in {"direct_inference", "legacy_user_attested"}:
                    raise ValueError(f"{path}: unsupported provenance_status={status!r}")
                if status == "legacy_user_attested":
                    migration_keys = {
                        "legacy_original_sha256",
                        "legacy_generation_attestation",
                    }
                    missing_migration = sorted(migration_keys - set(item.files))
                    if missing_migration:
                        raise ValueError(f"{path}: attested legacy cache lacks {missing_migration}")
                    self._scalar_string(item, "legacy_original_sha256", path)
                    self._scalar_string(item, "legacy_generation_attestation", path)
        except (OSError, ValueError, KeyError) as error:
            if isinstance(error, ValueError) and str(error).startswith(str(path)):
                raise
            raise ValueError(f"Cannot validate DA3 cache {path}: {error}") from error

        return DA3Prediction(
            depth=depth,
            confidence=confidence,
            source=f"cache:{path}",
            cache_path=path.resolve(),
            cache_sha256=sha256_file(path),
            image_sha256=expected_image_sha,
            model_name=cache_model,
            model_revision=cache_revision,
            process_res=process_res,
            target_shape=cache_shape,
            provenance_status=status,
        )

    def _cache_payload(
        self,
        depth: np.ndarray,
        confidence: np.ndarray,
        image_sha256: str,
        shape: tuple[int, int],
        *,
        provenance_status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": np.asarray(DA3_CACHE_SCHEMA_VERSION, dtype=np.uint16),
            "depth": depth.astype(np.float16),
            "confidence": confidence.astype(np.float16),
            "image_sha256": np.asarray(image_sha256),
            "model_name": np.asarray(self.model_name),
            "model_revision": np.asarray(self.model_revision),
            "process_res": np.asarray(self.process_res, dtype=np.int32),
            "target_shape": np.asarray(shape, dtype=np.int32),
            "provenance_status": np.asarray(provenance_status),
            "local_files_only": np.asarray(self.local_files_only, dtype=np.bool_),
        }
        if extra:
            payload.update(extra)
        return payload

    def inspect_legacy_cache(
        self,
        image_path: Path,
        shape: tuple[int, int],
    ) -> dict[str, Any]:
        """Preflight one old two-array cache without changing it or inferring."""

        path = self.write_cache / f"{image_path.stem}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Legacy cache is absent: {path}")
        with np.load(path, allow_pickle=False) as item:
            if DA3_CACHE_REQUIRED_KEYS.issubset(item.files):
                prediction = self._validate_cache(path, image_path, shape)
                return {
                    "status": "already_finalized",
                    "path": path,
                    "original_sha256": prediction.cache_sha256,
                    "image_sha256": prediction.image_sha256,
                }
            if set(item.files) != {"depth", "confidence"}:
                raise ValueError(
                    f"{path}: legacy migration accepts exactly depth/confidence, "
                    f"found {sorted(item.files)}"
                )
            depth = item["depth"].astype(np.float32)
            confidence = item["confidence"].astype(np.float32)
        if depth.shape != shape or confidence.shape != shape:
            raise ValueError(
                f"{path}: legacy shapes {depth.shape}/{confidence.shape} differ from {shape}"
            )
        if not np.isfinite(depth).all() or np.any(depth <= 0):
            raise ValueError(f"{path}: legacy depth must be finite and strictly positive")
        if not np.isfinite(confidence).all() or np.any(confidence < 0):
            raise ValueError(f"{path}: legacy confidence must be finite and non-negative")
        return {
            "status": "legacy_pending",
            "path": path,
            "original_sha256": sha256_file(path),
            "image_sha256": sha256_file(image_path),
        }

    def finalize_legacy_cache(
        self,
        image_path: Path,
        shape: tuple[int, int],
        *,
        generation_attestation: str,
        inspection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Attach explicit metadata to a legacy result without rerunning DA3."""

        attestation = generation_attestation.strip()
        if not attestation:
            raise ValueError("Legacy migration requires a non-empty generation attestation")
        if self.model_revision == "UNPINNED":
            raise ValueError("Legacy migration requires data.da3_revision to be pinned")
        inspected = inspection or self.inspect_legacy_cache(image_path, shape)
        if inspected["status"] == "already_finalized":
            return inspected
        path = Path(inspected["path"])
        original_sha256 = str(inspected["original_sha256"])
        if sha256_file(path) != original_sha256:
            raise RuntimeError(f"Legacy cache changed after preflight: {path}")
        with np.load(path, allow_pickle=False) as item:
            if set(item.files) != {"depth", "confidence"}:
                raise RuntimeError(f"Legacy cache schema changed after preflight: {path}")
            depth = item["depth"].astype(np.float32)
            confidence = item["confidence"].astype(np.float32)
        payload = self._cache_payload(
            depth,
            confidence,
            str(inspected["image_sha256"]),
            shape,
            provenance_status="legacy_user_attested",
            extra={
                "legacy_original_sha256": np.asarray(original_sha256),
                "legacy_generation_attestation": np.asarray(attestation),
            },
        )
        _atomic_savez_compressed(path, **payload)
        prediction = self._validate_cache(path, image_path, shape)
        return {
            "status": "migrated_legacy",
            "path": path,
            "original_sha256": original_sha256,
            "final_sha256": prediction.cache_sha256,
            "image_sha256": prediction.image_sha256,
        }

    def get_with_provenance(
        self,
        image_path: Path,
        shape: tuple[int, int],
    ) -> DA3Prediction:
        height, width = shape
        if height < 1 or width < 1:
            raise ValueError("DA3 target shape must be positive")
        image_path = image_path.expanduser().resolve()
        image_sha256 = sha256_file(image_path)
        candidates = []
        if self.read_cache:
            candidates.append(self.read_cache / f"{image_path.stem}.npz")
            candidates.append(self.read_cache / f"{image_path.stem}.npy")
        candidates.append(self.write_cache / f"{image_path.stem}.npz")
        candidates.append(self.write_cache / f"{image_path.stem}.npy")
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix != ".npz":
                raise ValueError(
                    f"{path}: unversioned NPY DA3 cache is not provenance-safe; "
                    "regenerate it as schema-v2 NPZ"
                )
            return self._validate_cache(
                path,
                image_path,
                shape,
                image_sha256=image_sha256,
            )

        if self.model is None:
            import torch
            from depth_anything_3.api import DepthAnything3

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            revision = None if self.model_revision == "UNPINNED" else self.model_revision
            self.model = (
                DepthAnything3.from_pretrained(
                    self.model_name,
                    revision=revision,
                    local_files_only=self.local_files_only,
                )
                .to(device)
                .eval()
            )
            loaded_revision = getattr(self.model, "_commit_hash", None)
            if revision is not None and loaded_revision not in {None, revision}:
                raise RuntimeError(
                    f"Loaded DA3 revision {loaded_revision!r}, expected {revision!r}"
                )
        result = self.model.inference(
            [str(image_path)], process_res=self.process_res, export_dir=None
        )
        depth = cv2.resize(
            result.depth[0].astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
        )
        raw_conf = getattr(result, "conf", None)
        if raw_conf is None:
            raw_conf = getattr(result, "depth_conf", None)
        confidence = (
            cv2.resize(raw_conf[0].astype(np.float32), (width, height))
            if raw_conf is not None
            else self._confidence_fallback(depth)
        )
        cache_path = self.write_cache / f"{image_path.stem}.npz"
        _atomic_savez_compressed(
            cache_path,
            **self._cache_payload(
                depth,
                confidence,
                image_sha256,
                shape,
                provenance_status="direct_inference",
            ),
        )
        prediction = self._validate_cache(
            cache_path,
            image_path,
            shape,
            image_sha256=image_sha256,
        )
        return DA3Prediction(
            **{
                **prediction.__dict__,
                "source": f"inference:{self.model_name}@{self.model_revision}",
            }
        )

    def get(self, image_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
        """Compatibility wrapper used by the original SLABIM preparation path."""

        prediction = self.get_with_provenance(image_path, shape)
        return prediction.depth, prediction.confidence, prediction.source


@lru_cache(maxsize=128)
def _load_pcd_points(path: Path) -> np.ndarray:
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float64)


def create_fused_gt(
    lidar_paths: list[Path],
    lidar_to_slam: np.ndarray,
    lidar_to_bim: np.ndarray,
    center_index: int,
    camera_to_lidar: np.ndarray,
    intrinsic: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    radius = int(cfg.data.fusion_radius)
    start = max(0, center_index - radius)
    stop = min(len(lidar_paths), center_index + radius + 1)
    bim_to_center_lidar = np.linalg.inv(lidar_to_bim[center_index])
    lidar_to_camera = np.linalg.inv(camera_to_lidar)
    scan_depths: list[np.ndarray] = []
    names: list[str] = []
    for index in range(start, stop):
        global_points = _load_pcd_points(lidar_paths[index])
        local_points = slam_global_to_lidar_local(global_points, lidar_to_slam[index])
        points_bim = transform_points(local_points, lidar_to_bim[index])
        points_center_lidar = transform_points(points_bim, bim_to_center_lidar)
        points_camera = transform_points(points_center_lidar, lidar_to_camera)
        scan_depths.append(
            project_depth(
                points_camera,
                intrinsic,
                int(cfg.data.target_height),
                int(cfg.data.target_width),
                float(cfg.data.min_depth),
                float(cfg.data.max_depth),
                int(cfg.data.splat_radius),
            )
        )
        names.append(lidar_paths[index].name)
    depth, support, weight = fuse_front_depth_cluster(
        scan_depths,
        float(cfg.data.occlusion_abs_m),
        float(cfg.data.occlusion_rel),
        int(cfg.data.min_gt_support),
    )
    return depth, support, weight, names


def _store_depth(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float16)


def prepare_region(
    cfg: Config,
    region_name: str,
    max_frames: int | None = None,
    stride: int = 1,
    overwrite: bool = False,
    refresh_gt_only: bool = False,
    inference_only: bool = False,
) -> list[dict[str, Any]]:
    if refresh_gt_only and inference_only:
        raise ValueError("refresh_gt_only and inference_only are mutually exclusive")
    slabim = resolve_slabim_root(cfg)
    region = slabim / "sensor_data" / region_name
    floor = region_name.split("_", 1)[0]
    output_root = resolve_project_path(cfg, cfg.data.processed_root)
    sample_dir = output_root / "samples" / region_name
    prediction_cache = output_root / "da3_cache" / region_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((region / "images/data").glob("*.png"))
    lidar_paths = sorted((region / "points/data").glob("*.pcd"))
    image_times = read_timestamps(region / "images/timestamps.txt")
    lidar_times = read_timestamps(region / "points/timestamps.txt")
    if len(image_paths) != len(image_times):
        raise ValueError(f"{region_name}: image file and timestamp counts differ")
    if not inference_only and len(lidar_paths) != len(lidar_times):
        raise ValueError(f"{region_name}: file and timestamp counts differ")
    matches = synchronize(image_times, lidar_times, float(cfg.data.max_time_diff))[::stride]
    if max_frames is not None:
        matches = matches[:max_frames]

    intrinsic_full = np.loadtxt(slabim / "calibration_files/cam_intrinsics.txt")
    camera_to_lidar = np.loadtxt(slabim / "calibration_files/cam_to_lidar.txt")
    lidar_to_bim = read_poses(region / "points" / cfg.data.pose_bim_file, lidar_times)
    lidar_to_slam = (
        None
        if inference_only
        else read_poses(region / "points" / cfg.data.pose_slam_file, lidar_times)
    )
    scene = None if refresh_gt_only else build_bim_scene(slabim / "BIM" / floor / "mesh")
    provider = (
        None if refresh_gt_only else DA3PredictionProvider(cfg, region_name, prediction_cache)
    )
    records: list[dict[str, Any]] = []

    for order, (image_index, lidar_index, time_difference) in enumerate(matches, 1):
        image_path = image_paths[image_index]
        sample_path = sample_dir / f"{image_path.stem}.npz"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        source_shape = image.shape[:2]
        target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
        intrinsic = scale_intrinsics(intrinsic_full, source_shape, target_shape)
        record = {
            "id": f"{region_name}/{image_path.stem}",
            "region": region_name,
            "image": str(image_path),
            "sample": str(sample_path),
            "image_timestamp": float(image_times[image_index]),
            "center_lidar": (
                lidar_paths[lidar_index].name
                if lidar_index < len(lidar_paths)
                else f"index:{lidar_index}"
            ),
            "lidar_index": int(lidar_index),
            "time_difference_s": time_difference,
        }
        if sample_path.exists() and not overwrite and not refresh_gt_only:
            with np.load(sample_path) as cached:
                if {"gt_depth", "gt_valid", "gt_weight"}.issubset(cached.files):
                    record["gt_valid_pixels"] = int((cached["gt_valid"] > 0).sum())
                else:
                    record["inference_only"] = True
            record["da3_source"] = "existing"
            records.append(record)
            print(f"[{region_name} {order}/{len(matches)}] reuse {sample_path.name}", flush=True)
            continue

        gt_depth = gt_support = gt_weight = None
        fused_names: list[str] = []
        if not inference_only:
            assert lidar_to_slam is not None
            gt_depth, gt_support, gt_weight, fused_names = create_fused_gt(
                lidar_paths,
                lidar_to_slam,
                lidar_to_bim,
                lidar_index,
                camera_to_lidar,
                intrinsic,
                cfg,
            )
        if refresh_gt_only:
            assert gt_depth is not None
            assert gt_support is not None
            assert gt_weight is not None
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"--refresh-gt-only requires an existing sample: {sample_path}"
                )
            with np.load(sample_path) as old:
                payload = {
                    key: old[key]
                    for key in old.files
                    if key not in {"gt_depth", "gt_valid", "gt_support", "gt_weight"}
                }
            payload.update(
                gt_depth=_store_depth(gt_depth),
                gt_valid=np.isfinite(gt_depth).astype(np.uint8),
                gt_support=gt_support,
                gt_weight=gt_weight.astype(np.float16),
            )
            np.savez_compressed(sample_path, **payload)
            source = "existing"
        else:
            assert provider is not None and scene is not None
            base_depth, base_confidence, source = provider.get(image_path, target_shape)
            camera_to_bim = lidar_to_bim[lidar_index] @ camera_to_lidar
            bim_depth, bim_normals = render_bim(
                scene,
                intrinsic,
                camera_to_bim,
                *target_shape,
                float(cfg.data.max_depth),
            )
            bim_valid = (
                np.isfinite(bim_depth)
                & (bim_depth >= float(cfg.data.min_depth))
                & (bim_depth <= float(cfg.data.max_depth))
            )
            bim_edge = depth_edges(bim_depth, bim_valid)
            # A single support contract is shared with the Stanford adapter and
            # every CPU/tensor scale estimator: invalid BIM is represented by
            # exact zero depth and zero normals, never by a positive hidden hit.
            bim_depth = np.where(bim_valid, bim_depth, 0.0).astype(np.float32)
            bim_normals = np.where(bim_valid[None], bim_normals, 0.0).astype(np.float32)
            payload = {
                "base_depth": _store_depth(base_depth),
                "base_confidence": np.nan_to_num(base_confidence).astype(np.float16),
                "bim_depth": _store_depth(bim_depth),
                "bim_valid": bim_valid.astype(np.uint8),
                "bim_normals": bim_normals.astype(np.float16),
                "bim_edge": bim_edge.astype(np.uint8),
                "intrinsic": intrinsic.astype(np.float32),
            }
            if gt_depth is not None:
                assert gt_support is not None
                assert gt_weight is not None
                payload.update(
                    gt_depth=_store_depth(gt_depth),
                    gt_valid=np.isfinite(gt_depth).astype(np.uint8),
                    gt_support=gt_support,
                    gt_weight=gt_weight.astype(np.float16),
                )
            np.savez_compressed(sample_path, **payload)
        record["da3_source"] = source
        if gt_depth is not None:
            record["fused_lidars"] = fused_names
            record["gt_valid_pixels"] = int(np.isfinite(gt_depth).sum())
        else:
            record["inference_only"] = True
        records.append(record)
        detail = f"GT={record['gt_valid_pixels']}" if gt_depth is not None else "GT=not prepared"
        print(
            f"[{region_name} {order}/{len(matches)}] {image_path.name}: {detail}, source={source}",
            flush=True,
        )
    return records


def write_manifest(
    cfg: Config,
    records: list[dict[str, Any]],
    replace_regions: set[str] | None = None,
) -> Path:
    output_root = resolve_project_path(cfg, cfg.data.processed_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    old = json.loads(line)
                    if replace_regions and old["region"] in replace_regions:
                        continue
                    merged[old["id"]] = old
    for record in records:
        merged[record["id"]] = record
    records = sorted(merged.values(), key=lambda row: row["id"])
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    has_ground_truth = any("gt_valid_pixels" in row for row in records)
    metadata = {
        "samples": len(records),
        "regions": sorted({row["region"] for row in records}),
        "coordinate_chain": (
            (
                "SLAM-global PCD -> inverse(local-to-SLAM) -> local LiDAR -> "
                "local-to-BIM -> center LiDAR -> camera"
            )
            if has_ground_truth
            else "camera -> camera-to-LiDAR -> local-to-BIM"
        ),
        "ground_truth": (
            "per-scan z-buffer; closest consistent depth cluster across neighboring scans"
            if has_ground_truth
            else None
        ),
        "fusion_radius": int(cfg.data.fusion_radius),
        "maximum_fused_scans": 2 * int(cfg.data.fusion_radius) + 1,
        "pcd_used_at_inference": False,
        "inference_only": not has_ground_truth,
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return manifest
