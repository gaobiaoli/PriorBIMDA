from __future__ import annotations

import pytest

from bim_priorda3.checkpoints import (
    make_training_dataset_provenance,
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_training_dataset_provenance,
)


def _annotation_provenance(
    *,
    fingerprint: str = "fingerprint",
    annotation_file: str = "/machine-a/splits.jsonl",
    preparation_fingerprint: str | None = None,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "mode": "annotations",
        "annotation_file": annotation_file,
        "fingerprint_sha256": fingerprint,
        "annotation_raw_sha256": "annotation-raw",
        "canonical_assignment_sha256": "canonical-assignment",
        "manifest_ordered_ids_sha256": "manifest-ids",
        "selected_regions": ["RegionB", "RegionA"],
    }
    if preparation_fingerprint is not None:
        provenance.update(
            manifest_preparation_fingerprint_status="verified",
            manifest_preparation_fingerprint_sha256=preparation_fingerprint,
        )
    return provenance


def _region_provenance(regions: list[str]) -> dict[str, object]:
    return {
        "mode": "regions",
        "selected_regions": regions,
        "record_stride_by_region": {"RegionB": 2},
    }


def _with_runtime_subset(provenance: dict[str, object]) -> dict[str, object]:
    return {
        **provenance,
        "runtime_subset": {
            "schema_version": 1,
            "selection": "ordered_prefix",
            "requested_max_samples": 2,
            "sample_count": 2,
            "preparation_fingerprint_status": "verified",
            "ordered_sample_ids_sha256": "1" * 64,
            "ordered_sample_preparation_fingerprints_sha256": "2" * 64,
            "fingerprint_sha256": "3" * 64,
        },
    }


def _checkpoint(dataset: dict[str, object] | None) -> dict[str, object]:
    provenance = {} if dataset is None else {"dataset": dataset}
    return {
        "provenance": provenance,
        "config": {
            "data": {
                "train_regions": ["RegionA"],
                "val_regions": ["RegionB"],
                "test_regions": ["RegionC"],
                "record_stride_by_region": {"RegionB": 2},
            }
        },
    }


def test_training_dataset_provenance_ignores_machine_specific_path() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(annotation_file="/machine-a/splits.jsonl"),
        _annotation_provenance(annotation_file="/machine-a/splits.jsonl"),
    )
    runtime_dataset = make_training_dataset_provenance(
        _annotation_provenance(annotation_file="/machine-b/splits.jsonl"),
        _annotation_provenance(annotation_file="/machine-b/splits.jsonl"),
    )

    receipt = validate_checkpoint_training_dataset_provenance(
        _checkpoint(checkpoint_dataset),
        runtime_dataset,
    )

    assert receipt["status"] == "verified"
    assert receipt["verified"] is True


def test_training_dataset_provenance_rejects_changed_annotation() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(),
        _annotation_provenance(),
    )
    runtime_dataset = make_training_dataset_provenance(
        _annotation_provenance(fingerprint="changed"),
        _annotation_provenance(fingerprint="changed"),
    )

    with pytest.raises(ValueError, match="does not match the runtime training population"):
        validate_checkpoint_training_dataset_provenance(
            _checkpoint(checkpoint_dataset),
            runtime_dataset,
        )


def test_smoke_subset_checkpoint_cannot_masquerade_as_full_population() -> None:
    smoke_split = _with_runtime_subset(_annotation_provenance())
    smoke_dataset = make_training_dataset_provenance(smoke_split, smoke_split)
    full_dataset = make_training_dataset_provenance(
        _annotation_provenance(),
        _annotation_provenance(),
    )
    checkpoint = _checkpoint(smoke_dataset)

    with pytest.raises(ValueError, match="does not match the runtime training population"):
        validate_checkpoint_training_dataset_provenance(checkpoint, full_dataset)

    with pytest.raises(ValueError, match="does not match the checkpoint"):
        validate_checkpoint_evaluation_dataset_provenance(
            checkpoint,
            _annotation_provenance(),
            split="test",
        )


def test_checkpoint_identity_rejects_changed_prepared_artifacts() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(preparation_fingerprint="prepared-a"),
        _annotation_provenance(preparation_fingerprint="prepared-a"),
    )
    runtime_dataset = make_training_dataset_provenance(
        _annotation_provenance(preparation_fingerprint="prepared-b"),
        _annotation_provenance(preparation_fingerprint="prepared-b"),
    )

    with pytest.raises(ValueError, match="does not match the runtime training population"):
        validate_checkpoint_training_dataset_provenance(
            _checkpoint(checkpoint_dataset),
            runtime_dataset,
        )


def test_training_cross_dataset_initialization_requires_explicit_opt_in() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(fingerprint="source"),
        _annotation_provenance(fingerprint="source"),
    )
    runtime_dataset = make_training_dataset_provenance(
        _annotation_provenance(fingerprint="target"),
        _annotation_provenance(fingerprint="target"),
    )

    receipt = validate_checkpoint_training_dataset_provenance(
        _checkpoint(checkpoint_dataset),
        runtime_dataset,
        allow_cross_dataset=True,
    )

    assert receipt["status"] == "accepted_cross_dataset"
    assert receipt["accepted"] is True
    assert receipt["verified"] is False
    assert receipt["dataset_match"] is False
    assert receipt["cross_dataset"] is True
    assert receipt["explicit_cross_dataset_opt_in"] is True
    assert receipt["source"]["dataset_provenance"] == checkpoint_dataset
    assert receipt["target"]["dataset_provenance"] == runtime_dataset


def test_missing_legacy_checkpoint_dataset_provenance_is_recorded() -> None:
    runtime_dataset = make_training_dataset_provenance(
        _annotation_provenance(),
        _annotation_provenance(),
    )

    receipt = validate_checkpoint_training_dataset_provenance(
        _checkpoint(None),
        runtime_dataset,
    )

    assert receipt["status"] == "legacy_checkpoint_missing"
    assert receipt["verified"] is False


def test_annotation_evaluation_must_match_training_fingerprint() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(),
        _annotation_provenance(),
    )
    checkpoint = _checkpoint(checkpoint_dataset)

    receipt = validate_checkpoint_evaluation_dataset_provenance(
        checkpoint,
        _annotation_provenance(annotation_file="/evaluation/splits.jsonl"),
        split="test",
    )
    assert receipt["verified"] is True

    with pytest.raises(ValueError, match="does not match the checkpoint"):
        validate_checkpoint_evaluation_dataset_provenance(
            checkpoint,
            _annotation_provenance(fingerprint="changed"),
            split="test",
        )


def test_cross_dataset_evaluation_receipt_records_source_and_target() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(fingerprint="source"),
        _annotation_provenance(fingerprint="source"),
    )
    checkpoint = _checkpoint(checkpoint_dataset)
    target = _annotation_provenance(fingerprint="target")

    receipt = validate_checkpoint_evaluation_dataset_provenance(
        checkpoint,
        target,
        split="test",
        allow_cross_dataset=True,
    )

    assert receipt["status"] == "accepted_cross_dataset"
    assert receipt["accepted"] is True
    assert receipt["verified"] is False
    assert receipt["dataset_match"] is False
    assert receipt["cross_dataset"] is True
    assert receipt["source"]["dataset_provenance"] == checkpoint_dataset
    assert receipt["target"]["split_provenance"] == target


def test_cross_dataset_evaluation_can_change_split_provenance_mode() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _annotation_provenance(fingerprint="source"),
        _annotation_provenance(fingerprint="source"),
    )

    receipt = validate_checkpoint_evaluation_dataset_provenance(
        _checkpoint(checkpoint_dataset),
        _region_provenance(["TargetRegion"]),
        split="test",
        allow_cross_dataset=True,
    )

    assert receipt["status"] == "accepted_cross_dataset"
    assert receipt["source"]["expected_evaluation_split_identity"]["mode"] == ("annotations")
    assert receipt["target"]["split_identity"]["mode"] == "regions"


def test_region_evaluation_uses_checkpoint_configured_split() -> None:
    checkpoint_dataset = make_training_dataset_provenance(
        _region_provenance(["RegionA"]),
        _region_provenance(["RegionB"]),
    )
    checkpoint = _checkpoint(checkpoint_dataset)

    receipt = validate_checkpoint_evaluation_dataset_provenance(
        checkpoint,
        {
            "mode": "regions",
            "selected_regions": ["RegionC"],
            "record_stride_by_region": {"RegionB": 2},
        },
        split="test",
    )
    assert receipt["verified"] is True

    with pytest.raises(ValueError, match="does not match the checkpoint"):
        validate_checkpoint_evaluation_dataset_provenance(
            checkpoint,
            {
                "mode": "regions",
                "selected_regions": ["OtherRegion"],
                "record_stride_by_region": {"RegionB": 2},
            },
            split="test",
        )
