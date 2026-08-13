from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANNOTATION_SCHEMA_VERSION = 1
ACTIVE_SPLITS = ("train", "val", "test")
ANNOTATION_SPLITS = (*ACTIVE_SPLITS, "excluded")
_ANNOTATION_KEYS = {"schema_version", "id", "split", "reason"}

Record = Mapping[str, Any]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


LEGACY_PREPARATION_FINGERPRINT_SHA256 = _canonical_sha256(
    {"schema_version": 1, "status": "legacy_manifest_without_preparation_fingerprint"}
)


def _preview(values: Sequence[str], limit: int = 10) -> str:
    displayed = ", ".join(repr(value) for value in values[:limit])
    remaining = len(values) - limit
    return displayed if remaining <= 0 else f"{displayed}, ... (+{remaining})"


def manifest_preparation_identity(records: Sequence[Record]) -> dict[str, str]:
    """Bind a manifest to ordered per-frame preparation artifacts when available.

    Original SLABIM manifests predate per-frame preparation fingerprints.  Their
    stable, explicit ``legacy_missing`` identity preserves compatibility without
    pretending those inputs were cryptographically verified.
    """

    present = ["preparation_fingerprint_sha256" in record for record in records]
    if not any(present):
        return {
            "status": "legacy_missing",
            "fingerprint_sha256": LEGACY_PREPARATION_FINGERPRINT_SHA256,
        }
    if not all(present):
        raise ValueError("manifest mixes records with and without preparation_fingerprint_sha256")
    entries: list[dict[str, str]] = []
    for record in records:
        sample_id = record.get("id")
        fingerprint = record.get("preparation_fingerprint_sha256")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("manifest preparation identity requires non-empty record IDs")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(
                f"{sample_id!r}: preparation_fingerprint_sha256 must be lowercase hex SHA256"
            )
        entries.append({"id": sample_id, "preparation_fingerprint_sha256": fingerprint})
    return {
        "status": "verified",
        "fingerprint_sha256": _canonical_sha256(
            {"schema_version": 1, "ordered_frame_preparation": entries}
        ),
    }


def _validate_manifest_records(
    records: Sequence[Record],
) -> tuple[list[Record], list[str], dict[str, str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("manifest records must be a sequence of mappings")
    if not records:
        raise ValueError("manifest contains no records")

    validated: list[Record] = []
    ordered_ids: list[str] = []
    first_index_by_id: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"manifest record {index} must be a mapping")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id or sample_id.strip() != sample_id:
            raise ValueError(f"manifest record {index} has an invalid non-empty string 'id'")
        region = record.get("region")
        if not isinstance(region, str) or not region or region.strip() != region:
            raise ValueError(f"manifest record {index} ({sample_id!r}) has an invalid region")
        if sample_id in first_index_by_id:
            raise ValueError(
                f"manifest record {index} duplicates sample ID {sample_id!r}; "
                f"first seen at record {first_index_by_id[sample_id]}"
            )
        first_index_by_id[sample_id] = index
        validated.append(record)
        ordered_ids.append(sample_id)
    return validated, ordered_ids, manifest_preparation_identity(validated)


def _read_annotation_file(
    path: Path,
) -> tuple[bytes, dict[str, dict[str, str | int]]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Split annotation file does not exist: {path}")

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: split annotation file is not valid UTF-8") from error

    annotations: dict[str, dict[str, str | int]] = {}
    first_line_by_id: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, Mapping):
            raise TypeError(f"{path}:{line_number}: annotation must be a JSON object")

        keys = set(value)
        missing_keys = {"schema_version", "id", "split"} - keys
        if missing_keys:
            raise ValueError(
                f"{path}:{line_number}: missing annotation keys {sorted(missing_keys)}"
            )
        unknown_keys = keys - _ANNOTATION_KEYS
        if unknown_keys:
            raise ValueError(
                f"{path}:{line_number}: unknown annotation keys {sorted(unknown_keys)}"
            )

        schema_version = value["schema_version"]
        if type(schema_version) is not int or schema_version != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(
                f"{path}:{line_number}: schema_version must be {ANNOTATION_SCHEMA_VERSION}"
            )
        sample_id = value["id"]
        if not isinstance(sample_id, str) or not sample_id or sample_id.strip() != sample_id:
            raise ValueError(
                f"{path}:{line_number}: id must be a non-empty string without "
                "surrounding whitespace"
            )
        split = value["split"]
        if split not in ANNOTATION_SPLITS:
            raise ValueError(
                f"{path}:{line_number}: split must be one of {list(ANNOTATION_SPLITS)}"
            )

        reason = value.get("reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError(
                f"{path}:{line_number}: reason must be a non-empty string when present"
            )
        if split == "excluded" and reason is None:
            raise ValueError(
                f"{path}:{line_number}: excluded sample {sample_id!r} requires a non-empty reason"
            )
        if sample_id in first_line_by_id:
            raise ValueError(
                f"{path}:{line_number}: duplicate sample ID {sample_id!r}; "
                f"first declared on line {first_line_by_id[sample_id]}"
            )

        normalized: dict[str, str | int] = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "id": sample_id,
            "split": str(split),
        }
        if reason is not None:
            normalized["reason"] = reason.strip()
        annotations[sample_id] = normalized
        first_line_by_id[sample_id] = line_number

    if not annotations:
        raise ValueError(f"{path}: split annotation file contains no records")
    return raw, annotations


def validate_fused_lidar_disjoint(
    records_by_split: Mapping[str, Sequence[Record]],
) -> dict[str, int]:
    """Require fused-LiDAR source scans to belong to at most one active split."""

    owners: dict[tuple[str, str], dict[str, set[str]]] = {}
    token_counts: dict[str, int] = {}
    for split in ACTIVE_SPLITS:
        split_tokens: set[tuple[str, str]] = set()
        for record in records_by_split.get(split, ()):
            sample_id = record.get("id")
            region = record.get("region")
            if not isinstance(sample_id, str) or not isinstance(region, str):
                raise TypeError(f"{split} record must contain string 'id' and 'region' values")
            fused_lidars = record.get("fused_lidars", ())
            if isinstance(fused_lidars, (str, bytes)) or not isinstance(
                fused_lidars,
                Sequence,
            ):
                raise TypeError(f"{sample_id!r}: fused_lidars must be a sequence of strings")
            for lidar_name in fused_lidars:
                if not isinstance(lidar_name, str) or not lidar_name:
                    raise ValueError(f"{sample_id!r}: fused_lidars contains an invalid name")
                token = (region, lidar_name)
                split_tokens.add(token)
                owners.setdefault(token, {}).setdefault(split, set()).add(sample_id)
        token_counts[split] = len(split_tokens)

    overlaps = [
        (token, split_owners) for token, split_owners in owners.items() if len(split_owners) > 1
    ]
    if overlaps:
        overlaps.sort(key=lambda item: item[0])
        descriptions = []
        for (region, lidar_name), split_owners in overlaps[:5]:
            owner_text = ", ".join(
                f"{split}={sorted(sample_ids)!r}"
                for split, sample_ids in sorted(split_owners.items())
            )
            descriptions.append(f"({region!r}, {lidar_name!r}): {owner_text}")
        remaining = len(overlaps) - len(descriptions)
        suffix = "" if remaining <= 0 else f"; ... (+{remaining} scans)"
        raise ValueError(
            "fused_lidars overlap across train/val/test splits: " + "; ".join(descriptions) + suffix
        )
    return token_counts


@dataclass(frozen=True)
class AnnotationSplitResolution:
    records_by_split: dict[str, tuple[Record, ...]]
    assignments: dict[str, str]
    excluded_reasons: dict[str, str]
    provenance: dict[str, Any]

    def records_for(self, split: str) -> list[Record]:
        if split not in ANNOTATION_SPLITS:
            raise ValueError(
                f"Unknown annotated split {split!r}; expected one of {list(ANNOTATION_SPLITS)}"
            )
        return list(self.records_by_split[split])


def resolve_annotation_splits(
    manifest_records: Sequence[Record],
    annotation_path: str | Path,
) -> AnnotationSplitResolution:
    """Resolve an exhaustive manifest-ID annotation into immutable split views."""

    records, manifest_ordered_ids, preparation_identity = _validate_manifest_records(
        manifest_records
    )
    path = Path(annotation_path).expanduser().resolve()
    raw_annotation, annotations = _read_annotation_file(path)

    manifest_id_set = set(manifest_ordered_ids)
    annotation_id_set = set(annotations)
    unknown_ids = sorted(annotation_id_set - manifest_id_set)
    if unknown_ids:
        raise ValueError(
            f"{path}: annotation contains {len(unknown_ids)} IDs absent from "
            f"the manifest: {_preview(unknown_ids)}"
        )
    missing_ids = sorted(manifest_id_set - annotation_id_set)
    if missing_ids:
        raise ValueError(
            f"{path}: annotation is missing {len(missing_ids)} manifest IDs: "
            f"{_preview(missing_ids)}"
        )

    grouped: dict[str, list[Record]] = {split: [] for split in ANNOTATION_SPLITS}
    assignments: dict[str, str] = {}
    excluded_reasons: dict[str, str] = {}
    for record in records:
        sample_id = str(record["id"])
        annotation = annotations[sample_id]
        split = str(annotation["split"])
        grouped[split].append(record)
        assignments[sample_id] = split
        if split == "excluded":
            excluded_reasons[sample_id] = str(annotation["reason"])

    frozen_groups = {split: tuple(grouped[split]) for split in ANNOTATION_SPLITS}
    fused_lidar_counts = validate_fused_lidar_disjoint(frozen_groups)

    canonical_assignments = [
        {key: annotation[key] for key in ("id", "split", "reason") if key in annotation}
        for _, annotation in sorted(annotations.items())
    ]
    manifest_ids_sha256 = _canonical_sha256(manifest_ordered_ids)
    canonical_assignment_sha256 = _canonical_sha256(canonical_assignments)
    ordered_ids = {
        split: [str(record["id"]) for record in frozen_groups[split]] for split in ANNOTATION_SPLITS
    }
    ordered_ids_sha256 = {
        split: _canonical_sha256(sample_ids) for split, sample_ids in ordered_ids.items()
    }
    regions = sorted({str(record["region"]) for record in records})
    split_region_counts = {
        split: {
            region: sum(1 for record in frozen_groups[split] if str(record["region"]) == region)
            for region in regions
        }
        for split in ANNOTATION_SPLITS
    }
    fingerprint_payload = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "manifest_ordered_ids": manifest_ordered_ids,
        "assignments": canonical_assignments,
    }
    if preparation_identity["status"] == "verified":
        fingerprint_payload["manifest_preparation_fingerprint_sha256"] = preparation_identity[
            "fingerprint_sha256"
        ]
    provenance = {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_file": str(path),
        "annotation_raw_sha256": hashlib.sha256(raw_annotation).hexdigest(),
        "canonical_assignment_sha256": canonical_assignment_sha256,
        "manifest_ordered_ids_sha256": manifest_ids_sha256,
        "manifest_preparation_fingerprint_status": preparation_identity["status"],
        "manifest_preparation_fingerprint_sha256": preparation_identity["fingerprint_sha256"],
        "fingerprint_sha256": _canonical_sha256(fingerprint_payload),
        "manifest_record_count": len(records),
        "annotation_record_count": len(annotations),
        "split_counts": {split: len(frozen_groups[split]) for split in ANNOTATION_SPLITS},
        "split_region_counts": split_region_counts,
        "ordered_ids_sha256": ordered_ids_sha256,
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "fused_lidar_validation": {
            "disjoint": True,
            "unique_source_counts": fused_lidar_counts,
        },
    }
    return AnnotationSplitResolution(
        records_by_split=frozen_groups,
        assignments=assignments,
        excluded_reasons=excluded_reasons,
        provenance=provenance,
    )
