import numpy as np

from bim_priorda3.reconstruction import depth_to_world_points, reconstruction_metrics


def test_depth_to_world_points_applies_intrinsics_and_pose() -> None:
    depth = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    intrinsic = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    pose[:3, 3] = (1.0, 2.0, 3.0)
    points = depth_to_world_points(
        depth,
        intrinsic,
        pose,
        pixel_stride=1,
        min_depth=0.1,
        max_depth=5.0,
    )
    expected = np.array([[1.0, 2.0, 5.0], [2.0, 2.0, 5.0], [1.0, 3.0, 5.0], [2.0, 3.0, 5.0]])
    assert np.allclose(points, expected)


def test_reconstruction_metrics_are_exact_for_identical_clouds() -> None:
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    metrics = reconstruction_metrics(points, points, thresholds=(0.01,))
    assert metrics["chamfer_l1_m"] == 0.0
    assert metrics["threshold_metrics"]["0.01m"]["fscore"] == 1.0
