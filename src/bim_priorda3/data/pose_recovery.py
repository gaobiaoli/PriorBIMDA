from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


@dataclass
class PoseRecoveryResult:
    region: str
    frames: int
    median_fitness: float
    median_rmse_m: float
    p95_rmse_m: float
    local_to_slam: str
    local_to_slam_smoothed: str
    local_to_bim: str
    diagnostics: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allow_nonstandard_slabim_magic() -> None:
    """Accept SLABIM's nonstandard '#ROSBAG  V2.0' two-space header."""
    import rosbags.rosbag1.reader as rosbag1_reader

    if getattr(rosbag1_reader, "_bim_priorda3_magic_patch", False):
        return
    original = rosbag1_reader.re.match

    def compatible(pattern: str, string: str, *args: Any, **kwargs: Any) -> Any:
        if pattern.startswith("#ROSBAG V"):
            pattern = r"#ROSBAG +V(\d+).(\d+)\n"
        return original(pattern, string, *args, **kwargs)

    rosbag1_reader.re.match = compatible
    rosbag1_reader._bim_priorda3_magic_patch = True


def nearest_index(timestamps: np.ndarray, value: float) -> int:
    right = int(np.searchsorted(timestamps, value))
    choices = [index for index in (right - 1, right) if 0 <= index < len(timestamps)]
    if not choices:
        raise IndexError("Cannot match a timestamp against an empty sequence")
    return min(choices, key=lambda index: abs(timestamps[index] - value))


def _cloud(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.remove_non_finite_points()
    return cloud.voxel_down_sample(voxel_size)


def save_pose_table(path: Path, timestamps: np.ndarray, transforms: np.ndarray) -> None:
    quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    np.savetxt(
        path,
        np.column_stack((timestamps, transforms[:, :3, 3], quaternions)),
        fmt="%.18e",
    )


def smooth_poses(transforms: np.ndarray, requested_window: int = 5) -> np.ndarray:
    if len(transforms) < 3:
        return transforms.copy()
    window = min(requested_window, len(transforms) if len(transforms) % 2 else len(transforms) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    polynomial = min(3, window - 1)
    result = transforms.copy()
    result[:, :3, 3] = savgol_filter(transforms[:, :3, 3], window, polynomial, axis=0)
    quaternions = Rotation.from_matrix(transforms[:, :3, :3]).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    quaternions = savgol_filter(quaternions, window, polynomial, axis=0)
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    result[:, :3, :3] = Rotation.from_quat(quaternions).as_matrix()
    return result


def _constant_map_to_bim(path: Path, timestamps: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.loadtxt(path, dtype=np.float64))
    if values.shape != (len(timestamps), 8):
        raise ValueError(f"Expected {len(timestamps)}x8 map-to-BIM rows, got {values.shape}")
    if not np.allclose(values[:, 0], timestamps, atol=1e-4, rtol=0.0):
        raise ValueError("pose_frame_to_bim timestamps do not match LiDAR timestamps")
    translation_spread = np.max(np.linalg.norm(values[:, 1:4] - values[0, 1:4], axis=1))
    rotations = Rotation.from_quat(values[:, 4:8]).as_matrix()
    relative = rotations @ rotations[0].T
    rotation_spread = np.max(Rotation.from_matrix(relative).magnitude())
    if translation_spread > 1e-6 or rotation_spread > 1e-6:
        raise ValueError(
            "pose_frame_to_bim is not a constant SLAM-map-to-BIM transform; "
            "automatic composition would be ambiguous"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotations[0]
    transform[:3, 3] = values[0, 1:4]
    return transform


def recover_lidar_poses(
    region_root: Path,
    voxel_size: float = 0.2,
    threshold: float = 0.35,
    iterations: int = 25,
    max_time_difference: float = 0.01,
    smoothing_window: int = 5,
    overwrite: bool = False,
    delete_rosbags: bool = False,
) -> PoseRecoveryResult:
    """Register raw local Livox scans to official SLAM-global exported PCDs."""
    region_root = region_root.resolve()
    points_root = region_root / "points"
    raw_output = points_root / "lidar_pose_local_to_slam.txt"
    smooth_output = points_root / "lidar_pose_local_to_slam_smoothed.txt"
    bim_output = points_root / "lidar_pose_local_to_bim_from_rosbag.txt"
    diagnostics_output = raw_output.with_suffix(".diagnostics.npz")
    required_outputs = (raw_output, smooth_output, bim_output, diagnostics_output)
    if not overwrite and all(path.exists() for path in required_outputs):
        diagnostics = np.load(diagnostics_output)
        if delete_rosbags:
            for bag_path in (region_root / "rosbag").glob("*.bag"):
                bag_path.unlink()
            bag_root = region_root / "rosbag"
            if bag_root.exists() and not any(bag_root.iterdir()):
                bag_root.rmdir()
        return PoseRecoveryResult(
            region=region_root.name,
            frames=len(diagnostics["fitness"]),
            median_fitness=float(np.nanmedian(diagnostics["fitness"])),
            median_rmse_m=float(np.nanmedian(diagnostics["rmse"])),
            p95_rmse_m=float(np.nanquantile(diagnostics["rmse"], 0.95)),
            local_to_slam=str(raw_output),
            local_to_slam_smoothed=str(smooth_output),
            local_to_bim=str(bim_output),
            diagnostics=str(diagnostics_output),
        )

    try:
        from rosbags.highlevel import AnyReader
    except ImportError as exc:
        raise RuntimeError(
            "rosbags is required for pose recovery; install with `pip install -e '.[slabim]'`"
        ) from exc

    timestamps = np.atleast_1d(np.loadtxt(points_root / "timestamps.txt", dtype=np.float64))
    pcd_paths = sorted((points_root / "data").glob("*.pcd"))
    if len(pcd_paths) != len(timestamps):
        raise ValueError(
            f"{region_root.name}: {len(pcd_paths)} PCD files != {len(timestamps)} timestamps"
        )
    bag_paths = sorted((region_root / "rosbag").glob("*.bag"))
    if not bag_paths:
        raise FileNotFoundError(f"No .bag files found under {region_root / 'rosbag'}")
    allow_nonstandard_slabim_magic()

    transforms = np.repeat(np.eye(4, dtype=np.float64)[None], len(timestamps), axis=0)
    fitness = np.full(len(timestamps), np.nan, dtype=np.float64)
    rmse = np.full(len(timestamps), np.nan, dtype=np.float64)
    recovered = np.zeros(len(timestamps), dtype=bool)
    estimate = np.eye(4, dtype=np.float64)
    for bag_path in bag_paths:
        with AnyReader([bag_path]) as reader:
            connections = [
                connection
                for connection in reader.connections
                if connection.topic == "/livox/lidar"
            ]
            if not connections:
                raise RuntimeError(f"{bag_path} has no /livox/lidar topic")
            for connection, _, raw in reader.messages(connections=connections):
                message = reader.deserialize(raw, connection.msgtype)
                stamp = float(message.timebase) / 1e9
                index = nearest_index(timestamps, stamp)
                if recovered[index] or abs(timestamps[index] - stamp) > max_time_difference:
                    continue
                source_points = np.asarray(
                    [(point.x, point.y, point.z) for point in message.points],
                    dtype=np.float64,
                )
                target_points = np.asarray(
                    o3d.io.read_point_cloud(str(pcd_paths[index])).points,
                    dtype=np.float64,
                )
                source = _cloud(source_points, voxel_size)
                target = _cloud(target_points, voxel_size)
                if not source.has_points() or not target.has_points():
                    continue
                schedules = (
                    ((1.0, 60), (0.5, 60), (0.2, 80), (0.1, 80))
                    if not recovered.any()
                    else ((threshold, iterations),)
                )
                registration = None
                for correspondence, maximum_iterations in schedules:
                    registration = o3d.pipelines.registration.registration_icp(
                        source,
                        target,
                        correspondence,
                        estimate,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=maximum_iterations
                        ),
                    )
                    estimate = registration.transformation
                assert registration is not None
                transforms[index] = estimate
                fitness[index] = registration.fitness
                rmse[index] = registration.inlier_rmse
                recovered[index] = True
                if int(recovered.sum()) % 100 == 0:
                    print(
                        f"{region_root.name}: recovered {recovered.sum()}/{len(timestamps)}, "
                        f"fitness={registration.fitness:.3f}, "
                        f"RMSE={registration.inlier_rmse:.3f} m",
                        flush=True,
                    )
    if not recovered.all():
        missing = np.flatnonzero(~recovered)
        raise RuntimeError(
            f"{region_root.name}: missing {len(missing)} poses; first indices={missing[:20].tolist()}"
        )

    smoothed = smooth_poses(transforms, smoothing_window)
    map_to_bim = _constant_map_to_bim(points_root / "pose_frame_to_bim.txt", timestamps)
    local_to_bim = map_to_bim[None] @ smoothed
    save_pose_table(raw_output, timestamps, transforms)
    save_pose_table(smooth_output, timestamps, smoothed)
    save_pose_table(bim_output, timestamps, local_to_bim)
    np.savez_compressed(
        diagnostics_output,
        timestamps=timestamps,
        transforms=transforms,
        smoothed=smoothed,
        fitness=fitness,
        rmse=rmse,
        recovered=recovered,
    )
    if delete_rosbags:
        for bag_path in bag_paths:
            bag_path.unlink()
        bag_root = region_root / "rosbag"
        if bag_root.exists() and not any(bag_root.iterdir()):
            bag_root.rmdir()
    return PoseRecoveryResult(
        region=region_root.name,
        frames=len(timestamps),
        median_fitness=float(np.median(fitness)),
        median_rmse_m=float(np.median(rmse)),
        p95_rmse_m=float(np.quantile(rmse, 0.95)),
        local_to_slam=str(raw_output),
        local_to_slam_smoothed=str(smooth_output),
        local_to_bim=str(bim_output),
        diagnostics=str(diagnostics_output),
    )
