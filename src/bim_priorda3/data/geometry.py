from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_timestamps(path: str | Path) -> np.ndarray:
    values = np.atleast_1d(np.loadtxt(path, dtype=np.float64))
    if values.ndim != 1:
        raise ValueError(f"Timestamp file must contain one column: {path}")
    return values


def read_poses(path: str | Path, timestamps: np.ndarray) -> np.ndarray:
    """Read timestamp, translation, quaternion(xyzw) as local-to-world transforms."""
    values = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if values.shape != (len(timestamps), 8):
        raise ValueError(f"Expected {len(timestamps)}x8 poses, got {values.shape}: {path}")
    if not np.allclose(values[:, 0], timestamps, atol=1e-4, rtol=0.0):
        raise ValueError(f"Pose timestamps are not aligned line by line: {path}")
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(values), axis=0)
    transforms[:, :3, :3] = Rotation.from_quat(values[:, 4:8]).as_matrix()
    transforms[:, :3, 3] = values[:, 1:4]
    return transforms


def synchronize(
    image_timestamps: np.ndarray,
    lidar_timestamps: np.ndarray,
    max_time_diff: float,
) -> list[tuple[int, int, float]]:
    right = np.searchsorted(lidar_timestamps, image_timestamps)
    matches: list[tuple[int, int, float]] = []
    for image_index, candidate in enumerate(right):
        choices = [
            index
            for index in (candidate - 1, candidate)
            if 0 <= index < len(lidar_timestamps)
        ]
        if not choices:
            continue
        lidar_index = min(
            choices, key=lambda index: abs(lidar_timestamps[index] - image_timestamps[image_index])
        )
        difference = float(lidar_timestamps[lidar_index] - image_timestamps[image_index])
        if abs(difference) <= max_time_diff:
            matches.append((image_index, lidar_index, difference))
    return matches


def scale_intrinsics(
    intrinsic: np.ndarray,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    scaled = intrinsic.astype(np.float64).copy()
    scaled[0] *= target_width / source_width
    scaled[1] *= target_height / source_height
    return scaled


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def slam_global_to_lidar_local(points: np.ndarray, lidar_to_slam: np.ndarray) -> np.ndarray:
    """Official PCDs are SLAM-global; recover points in the instantaneous LiDAR frame."""
    return (points - lidar_to_slam[:3, 3]) @ lidar_to_slam[:3, :3]


def project_depth(
    points_camera: np.ndarray,
    intrinsic: np.ndarray,
    height: int,
    width: int,
    min_depth: float,
    max_depth: float,
    splat_radius: int = 0,
) -> np.ndarray:
    """Z-buffer camera-space points, optionally splatting each point over nearby pixels."""
    if not len(points_camera):
        return np.full((height, width), np.nan, dtype=np.float32)
    z = points_camera[:, 2]
    valid = (
        np.isfinite(points_camera).all(axis=1)
        & (z >= min_depth)
        & (z <= max_depth)
    )
    points_camera = points_camera[valid]
    z = z[valid]
    if not len(z):
        return np.full((height, width), np.nan, dtype=np.float32)
    pixels = points_camera @ intrinsic.T
    u = np.rint(pixels[:, 0] / z).astype(np.int64)
    v = np.rint(pixels[:, 1] / z).astype(np.int64)
    flat = np.full(height * width, np.inf, dtype=np.float32)
    z = z.astype(np.float32)
    for dy in range(-splat_radius, splat_radius + 1):
        for dx in range(-splat_radius, splat_radius + 1):
            sx, sy = u + dx, v + dy
            inside = (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            np.minimum.at(flat, sy[inside] * width + sx[inside], z[inside])
    flat[~np.isfinite(flat)] = np.nan
    return flat.reshape(height, width)


def fuse_front_depth_cluster(
    scan_depths: list[np.ndarray],
    occlusion_abs_m: float,
    occlusion_rel: float,
    min_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse only the closest mutually consistent cluster to reject occluded surfaces."""
    if not scan_depths:
        raise ValueError("At least one projected scan is required")
    stack = np.stack(scan_depths).astype(np.float32)
    finite = np.isfinite(stack)
    nearest = np.min(np.where(finite, stack, np.inf), axis=0)
    nearest[~np.isfinite(nearest)] = np.nan
    tolerance = np.maximum(occlusion_abs_m, occlusion_rel * nearest)
    front = finite & (np.abs(stack - nearest[None]) <= tolerance[None])
    support = front.sum(axis=0).astype(np.uint8)

    front_values = np.where(front, stack, np.nan)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth = np.nanmedian(front_values, axis=0).astype(np.float32)
        mad = np.nanmedian(np.abs(front_values - depth[None]), axis=0).astype(np.float32)
    invalid = support < min_support
    depth[invalid] = np.nan
    mad[invalid] = np.nan

    support_score = np.clip(support.astype(np.float32) / 3.0, 0.0, 1.0)
    dispersion_score = np.exp(-np.nan_to_num(mad, nan=1.0) / max(occlusion_abs_m, 1e-4))
    weight = support_score * dispersion_score
    weight[invalid] = 0.0
    return depth, support, weight.astype(np.float32)


def depth_edges(depth: np.ndarray, valid: np.ndarray, threshold_m: float = 0.08) -> np.ndarray:
    safe = np.where(valid, depth, 0.0).astype(np.float32)
    gx = cv2.Sobel(safe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(safe, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.hypot(gx, gy) > threshold_m) | (~valid)
    return cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)


def approximate_depth_normals(depth: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    """Compute camera-space normals from a depth image; invalid normals are zero."""
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    z = np.nan_to_num(depth, nan=0.0)
    x = (u - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (v - intrinsic[1, 2]) * z / intrinsic[1, 1]
    xyz = np.stack((x, y, z), axis=-1).astype(np.float32)
    tangent_x = np.roll(xyz, -1, axis=1) - np.roll(xyz, 1, axis=1)
    tangent_y = np.roll(xyz, -1, axis=0) - np.roll(xyz, 1, axis=0)
    normals = np.cross(tangent_x, tangent_y)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    valid = np.isfinite(depth) & (depth > 0) & (norm[..., 0] > 1e-6)
    normals = normals / np.maximum(norm, 1e-6)
    normals[~valid] = 0.0
    return normals.transpose(2, 0, 1).astype(np.float32)
