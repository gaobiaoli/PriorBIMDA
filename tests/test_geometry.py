import numpy as np

from bim_priorda3.data.geometry import fuse_front_depth_cluster


def test_front_cluster_rejects_occluded_surface() -> None:
    front_a = np.full((4, 4), 2.00, np.float32)
    front_b = np.full((4, 4), 2.03, np.float32)
    occluded = np.full((4, 4), 3.00, np.float32)
    depth, support, weight = fuse_front_depth_cluster(
        [front_a, front_b, occluded],
        occlusion_abs_m=0.06,
        occlusion_rel=0.01,
        min_support=1,
    )
    assert np.allclose(depth, 2.015)
    assert np.all(support == 2)
    assert np.all(weight > 0)

