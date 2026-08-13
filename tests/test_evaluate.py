from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.model.evaluate import (
    apply_ignore_filter,
    read_ignore_sample_ids,
    validate_quality_filter_scope,
)


def _write_manifest(path: Path, sample_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"id": sample_id, "region": sample_id.split("/", 1)[0]}) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )


def test_apply_ignore_filter_records_exact_split_receipt(tmp_path: Path) -> None:
    ignore_path = tmp_path / "ignore.txt"
    ignore_path.write_text(
        "# low-quality frames\nRegionA/000002\n\nRegionB/000003  # motion blur\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest_path,
        ["RegionA/000001", "RegionA/000002", "RegionB/000003"],
    )
    dataset = SimpleNamespace(
        split="test",
        records=[
            {"id": "RegionA/000001", "region": "RegionA"},
            {"id": "RegionA/000002", "region": "RegionA"},
        ],
    )

    receipt = apply_ignore_filter(dataset, ignore_path, manifest_path)

    assert dataset.records == [{"id": "RegionA/000001", "region": "RegionA"}]
    assert receipt["declared_sample_count"] == 2
    assert receipt["samples_before"] == 2
    assert receipt["samples_ignored"] == 1
    assert receipt["samples_after"] == 1
    assert receipt["ignored_sample_ids"] == ["RegionA/000002"]
    assert receipt["declared_outside_selected_split_ids"] == ["RegionB/000003"]
    assert receipt["ignore_file_sha256"] == hashlib.sha256(ignore_path.read_bytes()).hexdigest()


def test_ignore_file_rejects_duplicate_ids(tmp_path: Path) -> None:
    ignore_path = tmp_path / "ignore.txt"
    ignore_path.write_text(
        "RegionA/000001\nRegionA/000001\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample ID"):
        read_ignore_sample_ids(ignore_path)


def test_ignore_filter_rejects_ids_missing_from_manifest(tmp_path: Path) -> None:
    ignore_path = tmp_path / "ignore.txt"
    ignore_path.write_text("RegionA/999999\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, ["RegionA/000001"])
    dataset = SimpleNamespace(
        split="test",
        records=[{"id": "RegionA/000001", "region": "RegionA"}],
    )

    with pytest.raises(ValueError, match="absent from"):
        apply_ignore_filter(dataset, ignore_path, manifest_path)


def test_ignore_filter_rejects_non_test_split() -> None:
    with pytest.raises(ValueError, match="restricted to --split test"):
        validate_quality_filter_scope("val", Path("ignore.txt"))

    validate_quality_filter_scope("test", Path("ignore.txt"))
    validate_quality_filter_scope("val", None)


def test_ignore_filter_does_not_mutate_dataset_when_every_sample_is_removed(
    tmp_path: Path,
) -> None:
    ignore_path = tmp_path / "ignore.txt"
    ignore_path.write_text("\ufeffRegionA/000001\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(manifest_path, ["RegionA/000001"])
    original_records = [{"id": "RegionA/000001", "region": "RegionA"}]
    dataset = SimpleNamespace(
        split="test",
        records=list(original_records),
    )

    with pytest.raises(RuntimeError, match="removed every sample"):
        apply_ignore_filter(dataset, ignore_path, manifest_path)

    assert dataset.records == original_records
