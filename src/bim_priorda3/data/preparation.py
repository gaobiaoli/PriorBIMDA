from __future__ import annotations

from functools import lru_cache
import json
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
    """Load cached DA3 output, or lazily infer when a cache is unavailable."""

    def __init__(self, cfg: Config, region_name: str, output_cache: Path) -> None:
        cache_value = cfg.data.da3_cache_roots.get(region_name)
        self.read_cache = Path(cache_value).expanduser().resolve() if cache_value else None
        self.write_cache = output_cache
        self.write_cache.mkdir(parents=True, exist_ok=True)
        self.model_name = cfg.data.da3_model
        self.process_res = int(cfg.data.da3_process_res)
        self.model = None

    @staticmethod
    def _confidence_fallback(depth: np.ndarray) -> np.ndarray:
        log_depth = np.log(np.maximum(depth, 1e-4))
        laplacian = np.abs(cv2.Laplacian(log_depth, cv2.CV_32F))
        scale = np.quantile(laplacian, 0.9) + 1e-6
        return np.exp(-laplacian / scale).astype(np.float32)

    def get(
        self, image_path: Path, shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        height, width = shape
        candidates = []
        if self.read_cache:
            candidates.append(self.read_cache / f"{image_path.stem}.npz")
            candidates.append(self.read_cache / f"{image_path.stem}.npy")
        candidates.append(self.write_cache / f"{image_path.stem}.npz")
        candidates.append(self.write_cache / f"{image_path.stem}.npy")
        for path in candidates:
            if not path.exists():
                continue
            if path.suffix == ".npz":
                item = np.load(path)
                depth = item["depth"].astype(np.float32)
                confidence = (
                    item["confidence"].astype(np.float32)
                    if "confidence" in item
                    else self._confidence_fallback(depth)
                )
            else:
                depth = np.load(path).astype(np.float32)
                confidence = self._confidence_fallback(depth)
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)
            confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_AREA)
            return depth, confidence, f"cache:{path}"

        if self.model is None:
            import torch
            from depth_anything_3.api import DepthAnything3

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = DepthAnything3.from_pretrained(self.model_name).to(device).eval()
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
        np.savez_compressed(
            self.write_cache / f"{image_path.stem}.npz",
            depth=depth.astype(np.float16),
            confidence=confidence.astype(np.float16),
        )
        return depth, confidence, f"inference:{self.model_name}"


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
) -> list[dict[str, Any]]:
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
    if len(image_paths) != len(image_times) or len(lidar_paths) != len(lidar_times):
        raise ValueError(f"{region_name}: file and timestamp counts differ")
    matches = synchronize(image_times, lidar_times, float(cfg.data.max_time_diff))[::stride]
    if max_frames is not None:
        matches = matches[:max_frames]

    intrinsic_full = np.loadtxt(slabim / "calibration_files/cam_intrinsics.txt")
    camera_to_lidar = np.loadtxt(slabim / "calibration_files/cam_to_lidar.txt")
    lidar_to_bim = read_poses(
        region / "points" / cfg.data.pose_bim_file, lidar_times
    )
    lidar_to_slam = read_poses(
        region / "points" / cfg.data.pose_slam_file, lidar_times
    )
    scene = None if refresh_gt_only else build_bim_scene(slabim / "BIM" / floor / "mesh")
    provider = (
        None
        if refresh_gt_only
        else DA3PredictionProvider(cfg, region_name, prediction_cache)
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
            "center_lidar": lidar_paths[lidar_index].name,
            "lidar_index": int(lidar_index),
            "time_difference_s": time_difference,
        }
        if sample_path.exists() and not overwrite and not refresh_gt_only:
            records.append(record)
            print(f"[{region_name} {order}/{len(matches)}] reuse {sample_path.name}", flush=True)
            continue

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
            bim_valid = np.isfinite(bim_depth) & (bim_depth >= float(cfg.data.min_depth))
            bim_edge = depth_edges(bim_depth, bim_valid)
            np.savez_compressed(
                sample_path,
                base_depth=_store_depth(base_depth),
                base_confidence=np.nan_to_num(base_confidence).astype(np.float16),
                bim_depth=_store_depth(bim_depth),
                bim_valid=bim_valid.astype(np.uint8),
                bim_normals=bim_normals.astype(np.float16),
                bim_edge=bim_edge.astype(np.uint8),
                gt_depth=_store_depth(gt_depth),
                gt_valid=np.isfinite(gt_depth).astype(np.uint8),
                gt_support=gt_support,
                gt_weight=gt_weight.astype(np.float16),
                intrinsic=intrinsic.astype(np.float32),
            )
        record["da3_source"] = source
        record["fused_lidars"] = fused_names
        record["gt_valid_pixels"] = int(np.isfinite(gt_depth).sum())
        records.append(record)
        print(
            f"[{region_name} {order}/{len(matches)}] {image_path.name}: "
            f"GT={record['gt_valid_pixels']}, source={source}",
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
    metadata = {
        "samples": len(records),
        "regions": sorted({row["region"] for row in records}),
        "coordinate_chain": (
            "SLAM-global PCD -> inverse(local-to-SLAM) -> local LiDAR -> "
            "local-to-BIM -> center LiDAR -> camera"
        ),
        "ground_truth": (
            "per-scan z-buffer; closest consistent depth cluster across neighboring scans"
        ),
        "fusion_radius": int(cfg.data.fusion_radius),
        "maximum_fused_scans": 2 * int(cfg.data.fusion_radius) + 1,
        "pcd_used_at_inference": False,
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    return manifest
