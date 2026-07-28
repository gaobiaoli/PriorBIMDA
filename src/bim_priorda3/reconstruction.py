from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def depth_to_world_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_to_world: np.ndarray,
    pixel_stride: int = 4,
    min_depth: float = 0.2,
    max_depth: float = 5.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Unproject camera-z depth and transform the points into a common world frame."""
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {depth.shape}")
    height, width = depth.shape
    y, x = np.mgrid[0:height:pixel_stride, 0:width:pixel_stride]
    z = depth[::pixel_stride, ::pixel_stride]
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if valid_mask is not None:
        if valid_mask.shape != depth.shape:
            raise ValueError("valid_mask and depth shapes differ")
        valid &= valid_mask[::pixel_stride, ::pixel_stride].astype(bool)
    x = x[valid].astype(np.float64)
    y = y[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    camera = np.column_stack(
        (
            (x - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (y - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        )
    )
    return (
        camera @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    ).astype(np.float32)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(voxel_size)
    return np.asarray(cloud.points, dtype=np.float32)


def _distance_statistics(distances: np.ndarray) -> dict[str, float]:
    return {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "rmse_m": float(np.sqrt(np.mean(distances**2))),
        "p90_m": float(np.quantile(distances, 0.90)),
        "p95_m": float(np.quantile(distances, 0.95)),
    }


def reconstruction_metrics(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    thresholds: tuple[float, ...] | list[float] = (0.05, 0.10, 0.20),
) -> dict:
    """Symmetric nearest-neighbor surface metrics without post-hoc alignment."""
    if not len(reconstruction) or not len(reference):
        raise ValueError("Reconstruction and reference point clouds must be non-empty")
    reference_tree = cKDTree(reference)
    reconstruction_tree = cKDTree(reconstruction)
    accuracy_distance = reference_tree.query(reconstruction, workers=-1)[0]
    completeness_distance = reconstruction_tree.query(reference, workers=-1)[0]
    result = {
        "reconstruction_points": int(len(reconstruction)),
        "reference_points": int(len(reference)),
        "accuracy_pred_to_gt": _distance_statistics(accuracy_distance),
        "completeness_gt_to_pred": _distance_statistics(completeness_distance),
        "chamfer_l1_m": float(
            (accuracy_distance.mean() + completeness_distance.mean()) / 2.0
        ),
        "threshold_metrics": {},
    }
    for threshold in thresholds:
        precision = float(np.mean(accuracy_distance < threshold))
        recall = float(np.mean(completeness_distance < threshold))
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        result["threshold_metrics"][f"{threshold:g}m"] = {
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        }
    return result


def save_point_cloud(path: str, points: np.ndarray, color: tuple[float, float, float]) -> None:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.paint_uniform_color(color)
    if not o3d.io.write_point_cloud(path, cloud, compressed=True):
        raise OSError(f"Failed to write point cloud: {path}")

