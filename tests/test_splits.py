from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bim_priorda3.data.splits import (
    ANNOTATION_SPLITS,
    LEGACY_PREPARATION_FINGERPRINT_SHA256,
    manifest_preparation_identity,
    resolve_annotation_splits,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_annotations(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _record(
    sample_id: str,
    region: str,
    fused_lidars: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "region": region,
        "sample": f"/prepared/{sample_id}.npz",
        "fused_lidars": fused_lidars or [],
    }


def test_resolver_filters_in_manifest_order_and_records_provenance(
    tmp_path: Path,
) -> None:
    records = [
        _record("A/000000", "A", ["a0.pcd"]),
        _record("A/000001", "A", ["a1.pcd"]),
        _record("B/000000", "B", ["b0.pcd"]),
        _record("B/000001", "B", ["shared-only-with-excluded.pcd"]),
    ]
    annotation_path = tmp_path / "splits.jsonl"
    rows = [
        {"schema_version": 1, "id": "B/000001", "split": "excluded", "reason": "bad GT"},
        {"schema_version": 1, "id": "A/000001", "split": "val"},
        {"schema_version": 1, "id": "B/000000", "split": "test"},
        {"schema_version": 1, "id": "A/000000", "split": "train"},
    ]
    _write_annotations(annotation_path, rows)

    resolution = resolve_annotation_splits(records, annotation_path)

    assert resolution.records_for("train") == [records[0]]
    assert resolution.records_for("val") == [records[1]]
    assert resolution.records_for("test") == [records[2]]
    assert resolution.records_for("excluded") == [records[3]]
    assert resolution.records_for("train")[0] is records[0]
    assert resolution.excluded_reasons == {"B/000001": "bad GT"}

    provenance = resolution.provenance
    assert (
        provenance["annotation_raw_sha256"]
        == hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    )
    assert provenance["manifest_ordered_ids_sha256"] == _canonical_sha256(
        [record["id"] for record in records]
    )
    canonical_assignments = [
        {"id": "A/000000", "split": "train"},
        {"id": "A/000001", "split": "val"},
        {"id": "B/000000", "split": "test"},
        {"id": "B/000001", "split": "excluded", "reason": "bad GT"},
    ]
    assert provenance["canonical_assignment_sha256"] == _canonical_sha256(canonical_assignments)
    assert provenance["split_counts"] == {
        "train": 1,
        "val": 1,
        "test": 1,
        "excluded": 1,
    }
    assert provenance["split_region_counts"] == {
        "train": {"A": 1, "B": 0},
        "val": {"A": 1, "B": 0},
        "test": {"A": 0, "B": 1},
        "excluded": {"A": 0, "B": 1},
    }
    assert provenance["ordered_ids_sha256"] == {
        "train": _canonical_sha256(["A/000000"]),
        "val": _canonical_sha256(["A/000001"]),
        "test": _canonical_sha256(["B/000000"]),
        "excluded": _canonical_sha256(["B/000001"]),
    }
    assert provenance["fused_lidar_validation"]["disjoint"] is True
    assert provenance["manifest_preparation_fingerprint_status"] == "legacy_missing"
    assert (
        provenance["manifest_preparation_fingerprint_sha256"]
        == LEGACY_PREPARATION_FINGERPRINT_SHA256
    )


def test_canonical_fingerprint_ignores_annotation_order_and_formatting(
    tmp_path: Path,
) -> None:
    records = [_record("A/0", "A"), _record("A/1", "A")]
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    _write_annotations(
        first_path,
        [
            {"schema_version": 1, "id": "A/0", "split": "train"},
            {"schema_version": 1, "id": "A/1", "split": "test"},
        ],
    )
    second_path.write_text(
        '{"split": "test", "id": "A/1", "schema_version": 1}\n'
        '{"id":"A/0","schema_version":1,"split":"train"}\n',
        encoding="utf-8",
    )

    first = resolve_annotation_splits(records, first_path).provenance
    second = resolve_annotation_splits(records, second_path).provenance

    assert first["annotation_raw_sha256"] != second["annotation_raw_sha256"]
    assert first["canonical_assignment_sha256"] == second["canonical_assignment_sha256"]
    assert first["fingerprint_sha256"] == second["fingerprint_sha256"]


def test_resolver_rejects_duplicate_annotation_ids(tmp_path: Path) -> None:
    path = tmp_path / "splits.jsonl"
    _write_annotations(
        path,
        [
            {"schema_version": 1, "id": "A/0", "split": "train"},
            {"schema_version": 1, "id": "A/0", "split": "val"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate sample ID.*first declared"):
        resolve_annotation_splits([_record("A/0", "A")], path)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"schema_version": 2, "id": "A/0", "split": "train"}, "schema_version"),
        ({"schema_version": 1, "id": "A/0", "split": "dev"}, "split must be"),
        (
            {"schema_version": 1, "id": "A/0", "split": "excluded"},
            "requires a non-empty reason",
        ),
        (
            {
                "schema_version": 1,
                "id": "A/0",
                "split": "train",
                "typo": True,
            },
            "unknown annotation keys",
        ),
    ],
)
def test_resolver_rejects_invalid_annotation_schema(
    tmp_path: Path,
    row: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "splits.jsonl"
    _write_annotations(path, [row])

    with pytest.raises(ValueError, match=message):
        resolve_annotation_splits([_record("A/0", "A")], path)


def test_resolver_rejects_unknown_and_missing_manifest_ids(tmp_path: Path) -> None:
    records = [_record("A/0", "A"), _record("A/1", "A")]
    unknown_path = tmp_path / "unknown.jsonl"
    _write_annotations(
        unknown_path,
        [
            {"schema_version": 1, "id": "A/0", "split": "train"},
            {"schema_version": 1, "id": "A/2", "split": "test"},
        ],
    )
    with pytest.raises(ValueError, match="IDs absent from the manifest.*A/2"):
        resolve_annotation_splits(records, unknown_path)

    missing_path = tmp_path / "missing.jsonl"
    _write_annotations(
        missing_path,
        [{"schema_version": 1, "id": "A/0", "split": "train"}],
    )
    with pytest.raises(ValueError, match="missing 1 manifest IDs.*A/1"):
        resolve_annotation_splits(records, missing_path)


def test_resolver_rejects_duplicate_manifest_ids(tmp_path: Path) -> None:
    path = tmp_path / "splits.jsonl"
    _write_annotations(
        path,
        [{"schema_version": 1, "id": "A/0", "split": "train"}],
    )
    records = [_record("A/0", "A"), _record("A/0", "A")]

    with pytest.raises(ValueError, match="duplicates sample ID"):
        resolve_annotation_splits(records, path)


def test_fused_lidar_overlap_across_active_splits_fails_fast(
    tmp_path: Path,
) -> None:
    records = [
        _record("A/0", "A", ["shared.pcd"]),
        _record("A/1", "A", ["shared.pcd"]),
        _record("A/2", "A", ["shared.pcd"]),
    ]
    overlapping_path = tmp_path / "overlapping.jsonl"
    _write_annotations(
        overlapping_path,
        [
            {"schema_version": 1, "id": "A/0", "split": "train"},
            {"schema_version": 1, "id": "A/1", "split": "test"},
            {
                "schema_version": 1,
                "id": "A/2",
                "split": "excluded",
                "reason": "boundary guard",
            },
        ],
    )

    with pytest.raises(
        ValueError,
        match="fused_lidars overlap across train/val/test.*shared",
    ):
        resolve_annotation_splits(records, overlapping_path)

    excluded_path = tmp_path / "excluded.jsonl"
    _write_annotations(
        excluded_path,
        [
            {"schema_version": 1, "id": "A/0", "split": "train"},
            {
                "schema_version": 1,
                "id": "A/1",
                "split": "excluded",
                "reason": "boundary guard",
            },
            {
                "schema_version": 1,
                "id": "A/2",
                "split": "excluded",
                "reason": "boundary guard",
            },
        ],
    )
    resolution = resolve_annotation_splits(records, excluded_path)
    assert resolution.provenance["fused_lidar_validation"]["disjoint"] is True


def test_records_for_rejects_unknown_split(tmp_path: Path) -> None:
    path = tmp_path / "splits.jsonl"
    _write_annotations(
        path,
        [{"schema_version": 1, "id": "A/0", "split": "train"}],
    )
    resolution = resolve_annotation_splits([_record("A/0", "A")], path)

    with pytest.raises(ValueError, match="Unknown annotated split"):
        resolution.records_for("development")
    assert set(resolution.records_by_split) == set(ANNOTATION_SPLITS)


def test_split_fingerprint_binds_verified_manifest_preparation(tmp_path: Path) -> None:
    rows = [
        {"schema_version": 1, "id": "A/0", "split": "train"},
        {"schema_version": 1, "id": "A/1", "split": "test"},
    ]
    annotation = tmp_path / "split.jsonl"
    _write_annotations(annotation, rows)
    first_records = [
        {**_record("A/0", "A"), "preparation_fingerprint_sha256": "1" * 64},
        {**_record("A/1", "A"), "preparation_fingerprint_sha256": "2" * 64},
    ]
    second_records = [dict(record) for record in first_records]
    second_records[1]["preparation_fingerprint_sha256"] = "3" * 64

    first = resolve_annotation_splits(first_records, annotation).provenance
    second = resolve_annotation_splits(second_records, annotation).provenance

    assert first["manifest_preparation_fingerprint_status"] == "verified"
    assert (
        first["manifest_preparation_fingerprint_sha256"]
        != second["manifest_preparation_fingerprint_sha256"]
    )
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


def test_manifest_preparation_rejects_partial_or_malformed_identity() -> None:
    with pytest.raises(ValueError, match="mixes records"):
        manifest_preparation_identity(
            [
                _record("A/0", "A"),
                {
                    **_record("A/1", "A"),
                    "preparation_fingerprint_sha256": "1" * 64,
                },
            ]
        )
    with pytest.raises(ValueError, match="lowercase hex SHA256"):
        manifest_preparation_identity(
            [
                {
                    **_record("A/0", "A"),
                    "preparation_fingerprint_sha256": "not-a-hash",
                }
            ]
        )
