from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bim_priorda3.config import Config
from scripts.data.audit_dataset import build_audit_report


def _write_sample(path: Path, value: float) -> None:
    shape = (2, 2)
    base = np.full(shape, value, dtype=np.float32)
    bim = np.full(shape, value * 1.1, dtype=np.float32)
    np.savez_compressed(
        path,
        base_depth=base,
        scaled_depth=base,
        bim_depth=bim,
        bim_valid=np.ones(shape, dtype=np.uint8),
        gt_depth=base,
        gt_valid=np.ones(shape, dtype=np.uint8),
    )


def _record(tmp_path: Path, sample_id: str, index: int) -> dict[str, object]:
    sample_path = tmp_path / f"sample_{index}.npz"
    image_path = tmp_path / f"image_{index}.png"
    _write_sample(sample_path, 1.0 + index * 0.1)
    image_path.touch()
    region = sample_id.split("/", maxsplit=1)[0]
    return {
        "id": sample_id,
        "region": region,
        "sample": str(sample_path),
        "image": str(image_path),
        "fused_lidars": [f"lidar_{index}.pcd"],
    }


def _config(
    tmp_path: Path,
    *,
    regions: list[str],
    train_regions: list[str],
    val_regions: list[str],
    test_regions: list[str],
    annotation: Path | None = None,
    strides: dict[str, int] | None = None,
) -> Config:
    data = Config(
        {
            "slabim_root": str(tmp_path / "source"),
            "processed_root": "processed",
            "regions": regions,
            "train_regions": train_regions,
            "val_regions": val_regions,
            "test_regions": test_regions,
            "record_stride_by_region": strides or {},
            "target_height": 2,
            "target_width": 2,
        }
    )
    if annotation is not None:
        data.split_annotation = str(annotation)
    cfg = Config(
        {
            "project_root": str(tmp_path),
            "data": data,
            "model": Config({}),
            "loss": Config({"trust_margin": 0.005, "trust_temperature": 0.03}),
        }
    )
    return cfg


def _write_manifest(tmp_path: Path, records: list[dict[str, object]]) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_annotation_audit_excludes_ignore_and_embargo_from_active_population(
    tmp_path: Path,
) -> None:
    records = [
        _record(tmp_path, "RegionA/000000", 0),
        _record(tmp_path, "RegionA/000001", 1),
        _record(tmp_path, "RegionA/000002", 2),
        _record(tmp_path, "RegionA/000003", 3),
        _record(tmp_path, "RegionA/000004", 4),
    ]
    _write_manifest(tmp_path, records)
    annotation = tmp_path / "split.jsonl"
    annotation.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"schema_version": 1, "id": "RegionA/000000", "split": "train"},
                {"schema_version": 1, "id": "RegionA/000001", "split": "val"},
                {"schema_version": 1, "id": "RegionA/000002", "split": "test"},
                {
                    "schema_version": 1,
                    "id": "RegionA/000003",
                    "split": "excluded",
                    "reason": "source_data_error",
                },
                {
                    "schema_version": 1,
                    "id": "RegionA/000004",
                    "split": "excluded",
                    "reason": "fused_lidar_leakage_guard",
                },
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "ignore.txt").write_text("RegionA/000003\n", encoding="utf-8")
    cfg = _config(
        tmp_path,
        regions=["RegionA"],
        train_regions=[],
        val_regions=[],
        test_regions=[],
        annotation=annotation,
    )

    report = build_audit_report(cfg)

    assert report["split_mode"] == "annotations"
    assert report["manifest_samples"] == 5
    assert report["samples"] == 3
    assert report["prepared_train_samples"] == 1
    assert report["prepared_val_samples"] == 1
    assert report["prepared_test_samples"] == 1
    population = report["split_population"]
    assert population["annotation_split_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
        "excluded": 2,
    }
    assert population["dataset_matches_annotation_resolver"] is True
    assert population["sample_id_disjoint"] is True
    assert report["ignore_population"]["dataset_input_exclusion_verified"] is True
    assert report["ignore_population"]["annotation_source_data_error_count"] == 1
    assert report["ignore_population"]["active_population_match_count"] == 0
    assert report["regions"]["RegionA"]["samples"] == 3


def test_legacy_region_audit_uses_dataset_stride_and_remains_compatible(
    tmp_path: Path,
) -> None:
    records = [
        _record(tmp_path, "Train/000000", 0),
        _record(tmp_path, "Train/000001", 1),
        _record(tmp_path, "Val/000000", 2),
        _record(tmp_path, "Test/000000", 3),
    ]
    _write_manifest(tmp_path, records)
    (tmp_path / "ignore.txt").write_text("Train/000001\n", encoding="utf-8")
    cfg = _config(
        tmp_path,
        regions=["Train", "Val", "Test"],
        train_regions=["Train"],
        val_regions=["Val"],
        test_regions=["Test"],
        strides={"Train": 2},
    )

    report = build_audit_report(cfg)

    assert report["split_mode"] == "regions"
    assert report["manifest_samples"] == 4
    assert report["samples"] == 3
    assert report["prepared_train_samples"] == 1
    assert report["prepared_val_samples"] == 1
    assert report["prepared_test_samples"] == 1
    population = report["split_population"]
    assert population["record_stride_by_region"] == {"Train": 2}
    assert population["region_disjoint"] is True
    assert population["sample_id_disjoint"] is True
    ignore = report["ignore_population"]
    assert ignore["applicable"] is True
    assert ignore["active_population_match_count"] == 0
    assert ignore["dataset_input_exclusion_verified"] is False
    assert ignore["policy"] == ("legacy_region_split_reports_but_does_not_apply_ignore_list")
