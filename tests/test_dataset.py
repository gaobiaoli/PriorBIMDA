import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.dataset import _enforce_bim_depth_mask_contract


def test_bim_depth_mask_contract_rejects_conflicts_and_zeros_roundoff() -> None:
    valid = np.asarray([[0, 0, 1]], dtype=np.float32)
    tolerated = np.asarray([[1e-7, -1e-7, 2.0]], dtype=np.float32)
    _enforce_bim_depth_mask_contract(
        tolerated,
        valid,
        sample_id="room/frame",
    )
    assert np.array_equal(tolerated, np.asarray([[0.0, 0.0, 2.0]]))

    contradictory = np.asarray([[0.0, 1e-4, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match=r"atol=1e-06.*violations=1"):
        _enforce_bim_depth_mask_contract(
            contradictory,
            valid,
            sample_id="room/frame",
        )


def test_inference_dataset_does_not_require_ground_truth(tmp_path: Path) -> None:
    height = width = 16
    image_path = tmp_path / "frame.png"
    sample_path = tmp_path / "frame.npz"
    processed = tmp_path / "processed"
    processed.mkdir()
    cv2.imwrite(
        str(image_path),
        np.full((height, width, 3), 127, dtype=np.uint8),
    )
    base = np.full((height, width), 2.0, dtype=np.float32)
    scaled = np.full((height, width), 2.2, dtype=np.float32)
    np.savez_compressed(
        sample_path,
        base_depth=base,
        base_confidence=np.ones_like(base),
        bim_depth=scaled,
        bim_valid=np.ones_like(base, dtype=np.uint8),
        bim_normals=np.zeros((3, height, width), dtype=np.float32),
        bim_edge=np.zeros_like(base, dtype=np.uint8),
        scaled_depth=scaled,
        anchor_depth=scaled,
        intrinsic=np.asarray(
            [[600.0, 0.0, 8.0], [0.0, 300.0, 8.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )
    manifest = {
        "id": "new_region/frame",
        "region": "new_region",
        "sample": str(sample_path),
        "image": str(image_path),
    }
    (processed / "manifest.jsonl").write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )

    cfg = load_config("configs/slabim.yaml")
    cfg.project_root = str(tmp_path)
    cfg.data.processed_root = "processed"
    cfg.data.regions = ["new_region"]
    cfg.data.split_annotation = None
    cfg.data.split_annotation_sha256 = None
    cfg.data.split_fingerprint_sha256 = None
    dataset = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=False,
    )
    item = dataset[0]
    assert item["sample_id"] == "new_region/frame"
    assert "gt_depth" not in item
    assert "trust_target" not in item
    assert "anchor_depth" not in item
    assert np.isclose(float(item["scaled_depth"].mean()), 2.2)
    assert tuple(item["intrinsic"].shape) == (3, 3)
    assert float(item["da3_metric_scale"]) == pytest.approx(1.5)
    assert not bool(item["da3_metric_scale_applied"])

    cfg.data.apply_da3_metric_focal_scaling = True
    cfg.data.recompute_cached_baselines = True
    metric_item = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=False,
    )[0]
    assert bool(metric_item["da3_metric_scale_applied"])
    assert float(metric_item["base_depth"].mean()) == pytest.approx(3.0)
    # Recomputed BIM scale uses the corrected metric base rather than the
    # stale cached scaled_depth array.
    assert float(metric_item["scaled_depth"].mean()) == pytest.approx(2.2)

    supervised = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=True,
    )
    with pytest.raises(RuntimeError, match="lacks training/evaluation fields"):
        supervised[0]


def test_supervised_dataset_can_recompute_float32_baselines(
    tmp_path: Path,
) -> None:
    height = width = 16
    image_path = tmp_path / "frame.png"
    sample_path = tmp_path / "frame.npz"
    processed = tmp_path / "processed"
    processed.mkdir()
    cv2.imwrite(
        str(image_path),
        np.full((height, width, 3), 127, dtype=np.uint8),
    )
    base = np.full((height, width), 2.0, dtype=np.float32)
    bim = np.full((height, width), 3.0, dtype=np.float32)
    np.savez_compressed(
        sample_path,
        base_depth=base,
        base_confidence=np.ones_like(base),
        bim_depth=bim,
        bim_valid=np.ones_like(base, dtype=np.uint8),
        bim_normals=np.zeros((3, height, width), dtype=np.float32),
        bim_edge=np.zeros_like(base, dtype=np.uint8),
        gt_depth=bim,
        gt_valid=np.ones_like(base, dtype=np.uint8),
        gt_weight=np.ones_like(base),
        scaled_depth=np.full_like(base, 9.0),
        anchor_depth=np.full_like(base, 9.0),
    )
    (processed / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "new_region/frame",
                "region": "new_region",
                "sample": str(sample_path),
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = load_config("configs/slabim_pretrain.yaml")
    cfg.project_root = str(tmp_path)
    cfg.data.processed_root = "processed"
    cfg.data.regions = ["new_region"]
    cfg.data.split_annotation = None
    cfg.data.split_annotation_sha256 = None
    cfg.data.split_fingerprint_sha256 = None
    dataset = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=True,
    )
    item = dataset[0]
    assert np.isclose(float(item["scaled_depth"].mean()), 3.0)
    assert np.isclose(float(item["anchor_depth"].mean()), 3.0)


def test_dataset_can_reload_all_valid_official_stanford_ground_truth(
    tmp_path: Path,
) -> None:
    height, width = 2, 4
    processed = tmp_path / "processed"
    rgb_dir = tmp_path / "area_1" / "data" / "rgb"
    depth_dir = tmp_path / "area_1" / "data" / "depth"
    processed.mkdir()
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)
    stem = "camera_x_office_1_frame_0"
    image_path = rgb_dir / f"{stem}_domain_rgb.png"
    depth_path = depth_dir / f"{stem}_domain_depth.png"
    sample_path = tmp_path / "frame.npz"
    assert cv2.imwrite(
        str(image_path),
        np.full((height, width, 3), 127, dtype=np.uint8),
    )
    raw_depth = np.asarray(
        [[0, 1024, 3072, 65535], [512, 2560, 5120, 1536]],
        dtype=np.uint16,
    )
    assert cv2.imwrite(str(depth_path), raw_depth)
    base = np.full((height, width), 2.0, dtype=np.float32)
    np.savez_compressed(
        sample_path,
        base_depth=base,
        base_confidence=np.ones_like(base),
        bim_depth=np.full_like(base, 3.0),
        bim_valid=np.ones_like(base, dtype=np.uint8),
        bim_normals=np.zeros((3, height, width), dtype=np.float32),
        bim_edge=np.zeros_like(base, dtype=np.uint8),
        gt_depth=np.full_like(base, 1.0),
        gt_valid=np.ones_like(base, dtype=np.uint8),
        gt_weight=np.ones_like(base),
    )
    (processed / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "office_1/frame_0",
                "region": "office_1",
                "sample": str(sample_path),
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config("configs/slabim_pretrain.yaml")
    cfg.project_root = str(tmp_path)
    cfg.data.processed_root = "processed"
    cfg.data.regions = ["office_1"]
    cfg.data.target_height = height
    cfg.data.target_width = width
    cfg.data.split_annotation = None
    cfg.data.split_annotation_sha256 = None
    cfg.data.split_fingerprint_sha256 = None
    cfg.data.ground_truth_support = "official_all_valid"

    dataset = BIMDepthDataset(cfg, split=None, augment=False, require_ground_truth=True)
    item = dataset[0]

    expected_valid = (raw_depth != 0) & (raw_depth != 65535)
    expected_depth = raw_depth.astype(np.float32) / 512.0
    expected_depth[~expected_valid] = 0.0
    np.testing.assert_array_equal(item["gt_valid"].numpy()[0] > 0, expected_valid)
    np.testing.assert_allclose(item["gt_depth"].numpy()[0], expected_depth)
    np.testing.assert_array_equal(item["gt_weight"].numpy()[0] > 0, expected_valid)
    assert float(item["gt_depth"].max()) == pytest.approx(10.0)
    assert dataset.split_provenance["ground_truth_support"]["mode"] == "official_all_valid"


def test_dataset_uses_configured_robust_scale_for_scaled_anchor_and_trust(
    tmp_path: Path,
) -> None:
    height = width = 20
    processed = tmp_path / "processed"
    processed.mkdir()
    image_path = tmp_path / "frame.png"
    sample_path = tmp_path / "frame.npz"
    cv2.imwrite(
        str(image_path),
        np.full((height, width, 3), 127, dtype=np.uint8),
    )
    base = np.ones((height, width), dtype=np.float32)
    bim = np.concatenate(
        (
            np.full(80, 1.0, dtype=np.float32),
            np.full(80, 1.1, dtype=np.float32),
            np.full(240, 3.0, dtype=np.float32),
        )
    ).reshape(height, width)
    np.savez_compressed(
        sample_path,
        base_depth=base,
        base_confidence=np.ones_like(base),
        bim_depth=bim,
        bim_valid=np.ones_like(base, dtype=np.uint8),
        bim_normals=np.zeros((3, height, width), dtype=np.float32),
        bim_edge=np.zeros_like(base, dtype=np.uint8),
        gt_depth=np.full_like(base, 1.05),
        gt_valid=np.ones_like(base, dtype=np.uint8),
        gt_weight=np.ones_like(base),
        scaled_depth=np.full_like(base, 9.0),
        anchor_depth=np.full_like(base, 9.0),
    )
    (processed / "manifest.jsonl").write_text(
        json.dumps(
            {
                "id": "new_region/frame",
                "region": "new_region",
                "sample": str(sample_path),
                "image": str(image_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = load_config("configs/slabim.yaml")
    cfg.project_root = str(tmp_path)
    cfg.data.processed_root = "processed"
    cfg.data.regions = ["new_region"]
    cfg.data.split_annotation = None
    cfg.data.split_annotation_sha256 = None
    cfg.data.split_fingerprint_sha256 = None
    cfg.data.recompute_cached_baselines = False
    cfg.model.scale_estimator = {
        "name": "log_upper_cap_v1",
        "q10_log_cap": 0.20,
        "q25_log_cap": 0.05,
    }

    item = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=True,
    )[0]
    assert float(item["scaled_depth"].mean()) < 2.0
    assert float(item["anchor_depth"].mean()) < 2.0
    assert torch.all(torch.isfinite(item["trust_target"]))


def test_dataset_uses_exhaustive_annotation_without_copying_records(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    records = [
        {
            "id": f"RegionA/{index:06d}",
            "region": "RegionA",
            "sample": str(tmp_path / f"{index}.npz"),
            "image": str(tmp_path / f"{index}.png"),
            "fused_lidars": [f"{index}.pcd"],
        }
        for index in range(4)
    ]
    for index in range(4):
        (tmp_path / f"{index}.npz").touch()
        (tmp_path / f"{index}.png").touch()
    (processed / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    annotation_path = tmp_path / "split.jsonl"
    annotations = [
        {"schema_version": 1, "id": "RegionA/000000", "split": "train"},
        {"schema_version": 1, "id": "RegionA/000001", "split": "val"},
        {"schema_version": 1, "id": "RegionA/000002", "split": "test"},
        {
            "schema_version": 1,
            "id": "RegionA/000003",
            "split": "excluded",
            "reason": "bad source",
        },
    ]
    annotation_path.write_text(
        "".join(json.dumps(annotation) + "\n" for annotation in annotations),
        encoding="utf-8",
    )

    cfg = load_config("configs/slabim.yaml")
    cfg.project_root = str(tmp_path)
    cfg.data.processed_root = "processed"
    cfg.data.regions = ["RegionA"]
    cfg.data.train_regions = []
    cfg.data.val_regions = []
    cfg.data.test_regions = []
    cfg.data.record_stride_by_region = {}
    cfg.data.split_annotation = str(annotation_path)
    cfg.data.split_annotation_sha256 = None
    cfg.data.split_fingerprint_sha256 = None

    train = BIMDepthDataset(cfg, "train", augment=False)
    val = BIMDepthDataset(cfg, "val", augment=False)
    test = BIMDepthDataset(cfg, "test", augment=False)
    all_active = BIMDepthDataset(
        cfg,
        split=None,
        augment=False,
        require_ground_truth=False,
    )

    assert [record["id"] for record in train.records] == ["RegionA/000000"]
    assert [record["id"] for record in val.records] == ["RegionA/000001"]
    assert [record["id"] for record in test.records] == ["RegionA/000002"]
    assert [record["id"] for record in all_active.records] == [
        "RegionA/000000",
        "RegionA/000001",
        "RegionA/000002",
    ]
    assert train.records[0]["sample"] == str((tmp_path / "0.npz").resolve())
    assert train.split_provenance["mode"] == "annotations"
    assert (
        train.split_provenance["fingerprint_sha256"]
        == val.split_provenance["fingerprint_sha256"]
        == test.split_provenance["fingerprint_sha256"]
    )
