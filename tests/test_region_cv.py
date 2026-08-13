from __future__ import annotations

import json
from pathlib import Path

import pytest

from bim_priorda3.region_cv import (
    build_region_fold_plans,
    dataset_fingerprint_from_manifest,
    parse_region_cv_protocol,
    region_macro_summary,
)


def _protocol() -> dict[str, object]:
    return {
        "regions": ["A", "B", "C", "D"],
        "validation_map": {
            "A": "B",
            "B": "C",
            "C": "D",
            "D": "A",
        },
        "seeds": [42, 7],
    }


def test_protocol_parsing_and_fold_generation_are_deterministic() -> None:
    protocol = parse_region_cv_protocol(_protocol())
    assert protocol.regions == ("A", "B", "C", "D")
    assert protocol.validation_map == {
        "A": "B",
        "B": "C",
        "C": "D",
        "D": "A",
    }
    assert protocol.seeds == (42, 7)

    plans = build_region_fold_plans(protocol)
    assert len(plans) == 4
    assert plans[0].to_dict() == {
        "fold_index": 0,
        "fold_id": "fold_00_A",
        "train_regions": ["C", "D"],
        "val_regions": ["B"],
        "test_regions": ["A"],
        "seeds": [42, 7],
    }
    assert plans[3].train_regions == ("B", "C")
    assert plans[3].val_regions == ("A",)
    assert plans[3].test_regions == ("D",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("regions", ["A", "A", "C"], "duplicates"),
        (
            "validation_map",
            {"A": "A", "B": "C", "C": "B", "D": "D"},
            "cannot use the test region",
        ),
        (
            "validation_map",
            {"A": "B", "B": "A", "C": "A", "D": "C"},
            "every region exactly once",
        ),
        ("seeds", [42, 42], "duplicates"),
        ("seeds", [True], "integers"),
    ],
)
def test_protocol_rejects_invalid_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _protocol()
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        parse_region_cv_protocol(raw)


def _write_manifest(path: Path, *, change_metadata: bool = False) -> None:
    records = [
        {"region": "B", "id": "B/0", "sample": "/old/0.npz"},
        {"region": "A", "id": "A/0", "sample": "/old/1.npz"},
        {"region": "B", "id": "B/1", "sample": "/old/2.npz"},
        {"region": "A", "id": "A/1", "sample": "/old/3.npz"},
        {"region": "B", "id": "B/2", "sample": "/old/4.npz"},
        {"region": "A", "id": "A/2", "sample": "/old/5.npz"},
    ]
    if change_metadata:
        for index, record in enumerate(records):
            record["sample"] = f"/relocated/{index}.npz"
            record["gt_valid_pixels"] = index
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_manifest_fingerprint_is_read_only_id_based_and_uniformly_strided(
    tmp_path: Path,
) -> None:
    first_manifest = tmp_path / "first.jsonl"
    second_manifest = tmp_path / "second.jsonl"
    _write_manifest(first_manifest)
    _write_manifest(second_manifest, change_metadata=True)
    original = first_manifest.read_bytes()

    first = dataset_fingerprint_from_manifest(
        first_manifest,
        regions=["B", "A"],
        stride=2,
    )
    reordered = dataset_fingerprint_from_manifest(
        second_manifest,
        regions=["A", "B"],
        stride=2,
    )

    assert first.sampled_ids_by_region == (
        ("B", ("B/0", "B/2")),
        ("A", ("A/0", "A/2")),
    )
    assert first.region_counts == {"B": 2, "A": 2}
    assert first.stride_by_region == (("B", 2), ("A", 2))
    assert first.sample_count == 4
    assert first.sha256 == reordered.sha256
    assert first_manifest.read_bytes() == original
    assert (
        dataset_fingerprint_from_manifest(
            first_manifest,
            regions=["A", "B"],
            stride=1,
        ).sha256
        != first.sha256
    )


def test_manifest_fingerprint_supports_per_region_stride(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest)
    fingerprint = dataset_fingerprint_from_manifest(
        manifest,
        regions=["A", "B"],
        stride=1,
        stride_by_region={"B": 2},
    )
    assert fingerprint.sampled_ids_by_region == (
        ("A", ("A/0", "A/1", "A/2")),
        ("B", ("B/0", "B/2")),
    )
    assert fingerprint.stride_by_region == (("A", 1), ("B", 2))
    assert fingerprint.region_counts == {"A": 3, "B": 2}


def test_manifest_fingerprint_rejects_duplicate_ids_and_missing_regions(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"region":"A","id":"A/0"}\n{"region":"A","id":"A/0"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate record ID"):
        dataset_fingerprint_from_manifest(manifest)

    _write_manifest(manifest)
    with pytest.raises(ValueError, match="requested regions are missing"):
        dataset_fingerprint_from_manifest(manifest, regions=["A", "C"])
    with pytest.raises(ValueError, match="positive integer"):
        dataset_fingerprint_from_manifest(manifest, stride=0)
    with pytest.raises(ValueError, match="unselected regions"):
        dataset_fingerprint_from_manifest(
            manifest,
            regions=["A"],
            stride_by_region={"B": 2},
        )


def test_region_macro_summary_weights_regions_equally() -> None:
    summary = region_macro_summary(
        {
            "large": {"abs_rel": 0.1, "rmse": 1.0, "count": 1_000_000},
            "small": {"abs_rel": 0.3, "rmse": 3.0, "count": 10},
        }
    )
    assert summary["regions"] == ["large", "small"]
    assert summary["region_count"] == 2
    assert "count" not in summary["metrics"]
    assert summary["metrics"]["abs_rel"] == pytest.approx(
        {
            "mean": 0.2,
            "std": 2**0.5 / 10,
            "min": 0.1,
            "max": 0.3,
        }
    )
    assert summary["metrics"]["rmse"]["mean"] == pytest.approx(2.0)


def test_region_macro_summary_requires_complete_finite_metrics() -> None:
    with pytest.raises(ValueError, match="missing metric"):
        region_macro_summary(
            {"A": {"abs_rel": 0.1}, "B": {"rmse": 1.0}},
            metric_names=["abs_rel"],
        )
    with pytest.raises(ValueError, match="finite"):
        region_macro_summary(
            {"A": {"abs_rel": float("nan")}},
            metric_names=["abs_rel"],
        )
