from pathlib import Path

import cv2
import numpy as np

from bim_priorda3.config import load_config
from bim_priorda3.data import preparation


def test_inference_preparation_does_not_require_pcd_or_gt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    slabim = tmp_path / "SLABIM"
    region_name = "9F_Region1"
    region = slabim / "sensor_data" / region_name
    image_dir = region / "images" / "data"
    points_dir = region / "points"
    calibration = slabim / "calibration_files"
    image_dir.mkdir(parents=True)
    points_dir.mkdir(parents=True)
    calibration.mkdir(parents=True)
    cv2.imwrite(
        str(image_dir / "000000.png"),
        np.full((8, 8, 3), 127, dtype=np.uint8),
    )
    (region / "images" / "timestamps.txt").write_text("1.0\n", encoding="utf-8")
    (points_dir / "timestamps.txt").write_text("1.0\n", encoding="utf-8")
    (points_dir / "lidar_pose_local_to_bim_from_rosbag.txt").write_text(
        "1.0 0 0 0 0 0 0 1\n",
        encoding="utf-8",
    )
    np.savetxt(calibration / "cam_intrinsics.txt", np.eye(3))
    np.savetxt(calibration / "cam_to_lidar.txt", np.eye(4))

    class FakeProvider:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def get(self, _path: Path, shape: tuple[int, int]):
            depth = np.full(shape, 2.0, dtype=np.float32)
            return depth, np.ones(shape, dtype=np.float32), "fake-da3"

    monkeypatch.setattr(preparation, "DA3PredictionProvider", FakeProvider)
    monkeypatch.setattr(preparation, "build_bim_scene", lambda _path: object())
    monkeypatch.setattr(
        preparation,
        "render_bim",
        lambda _scene, _intrinsic, _pose, height, width, _maximum: (
            np.full((height, width), 2.2, dtype=np.float32),
            np.zeros((3, height, width), dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        preparation,
        "create_fused_gt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("GT creation must not run during inference preparation")
        ),
    )

    cfg = load_config("configs/slabim.yaml")
    cfg.project_root = str(tmp_path)
    cfg.data.slabim_root = str(slabim)
    cfg.data.processed_root = "processed"
    cfg.data.target_height = 8
    cfg.data.target_width = 8
    records = preparation.prepare_region(
        cfg,
        region_name,
        inference_only=True,
    )

    assert len(records) == 1
    assert records[0]["inference_only"] is True
    assert not (points_dir / cfg.data.pose_slam_file).exists()
    with np.load(records[0]["sample"]) as sample:
        assert "base_depth" in sample
        assert "bim_depth" in sample
        assert "gt_depth" not in sample
