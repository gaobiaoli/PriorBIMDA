#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}
PROTOCOL_TAG = "clean-global-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    ids = [record.get("id") for record in records]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        raise ValueError(f"{path}: every manifest record must have a non-empty string ID")
    duplicates = sorted(sample_id for sample_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{path}: duplicate manifest IDs: {duplicates[:10]}")
    return records


def read_ignore_ids(path: Path) -> tuple[str, ...]:
    ids: list[str] = []
    first_line: dict[str, int] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        sample_id = raw_line.partition("#")[0].strip()
        if not sample_id:
            continue
        parts = sample_id.split("/")
        if len(parts) != 2 or not all(parts) or any(character.isspace() for character in sample_id):
            raise ValueError(f"{path}:{line_number}: expected an exact '<region>/<frame>' ID")
        if sample_id in first_line:
            raise ValueError(
                f"{path}:{line_number}: duplicate ID {sample_id!r}; "
                f"first declared on line {first_line[sample_id]}"
            )
        first_line[sample_id] = line_number
        ids.append(sample_id)
    if not ids:
        raise ValueError(f"{path}: ignore file contains no IDs")
    return tuple(ids)


def _fused_tokens(
    records: list[dict[str, Any]],
    labels: list[str | None],
) -> dict[str, set[str]]:
    tokens = {split: set() for split in SPLITS}
    for record, split in zip(records, labels):
        if split is None:
            continue
        region = str(record["region"])
        tokens[split].update(f"{region}/{name}" for name in record.get("fused_lidars", []))
    return tokens


def _overlap_count(tokens: dict[str, set[str]]) -> int:
    return sum(
        len(tokens[first] & tokens[second]) for first, second in itertools.combinations(SPLITS, 2)
    )


def apportion_region_quotas(
    counts_by_region: dict[str, int],
    ratios: dict[str, float],
    *,
    seed: int,
) -> dict[str, dict[str, int]]:
    """Largest-remainder apportionment with deterministic SHA256 tie breaks."""
    total = sum(counts_by_region.values())
    global_floor = {split: math.floor(total * ratios[split]) for split in SPLITS}
    global_remaining = total - sum(global_floor.values())
    global_order = sorted(
        SPLITS,
        key=lambda split: (
            -(total * ratios[split] - global_floor[split]),
            hashlib.sha256(f"{PROTOCOL_TAG}|seed={seed}|global-quota={split}".encode()).digest(),
        ),
    )
    global_targets = dict(global_floor)
    for split in global_order[:global_remaining]:
        global_targets[split] += 1

    quotas = {
        region: {split: math.floor(count * ratios[split]) for split in SPLITS}
        for region, count in counts_by_region.items()
    }
    for split in SPLITS:
        remaining = global_targets[split] - sum(
            region_quotas[split] for region_quotas in quotas.values()
        )
        candidates = sorted(
            counts_by_region,
            key=lambda region: (
                -(counts_by_region[region] * ratios[split] - quotas[region][split]),
                hashlib.sha256(
                    (f"{PROTOCOL_TAG}|seed={seed}|quota={split}|region={region}").encode()
                ).digest(),
            ),
        )
        for region in candidates[:remaining]:
            quotas[region][split] += 1

    invalid = {
        region: (sum(quotas[region].values()), count)
        for region, count in counts_by_region.items()
        if sum(quotas[region].values()) != count
    }
    if invalid:
        raise RuntimeError(f"Region apportionment did not preserve counts: {invalid}")
    return quotas


def balanced_segment_orders(
    regions: list[str],
    *,
    seed: int,
) -> dict[str, tuple[str, str, str]]:
    if len(regions) != len(tuple(itertools.permutations(SPLITS))):
        raise ValueError(
            "Balanced temporal ordering requires exactly six regions so every "
            "train/val/test permutation is used once"
        )
    ordered_regions = sorted(
        regions,
        key=lambda region: hashlib.sha256(
            f"{PROTOCOL_TAG}|seed={seed}|region={region}".encode()
        ).digest(),
    )
    ordered_permutations = sorted(
        itertools.permutations(SPLITS),
        key=lambda order: hashlib.sha256(
            (f"{PROTOCOL_TAG}|seed={seed}|order={'/'.join(order)}").encode()
        ).digest(),
    )
    return dict(zip(ordered_regions, ordered_permutations))


def _segment_fused_tokens(records: list[dict[str, Any]]) -> set[str]:
    region = str(records[0]["region"]) if records else ""
    return {
        f"{region}/{lidar_name}"
        for record in records
        for lidar_name in record.get("fused_lidars", [])
    }


def assign_priority_embargo(
    records: list[dict[str, Any]],
    order: tuple[str, str, str],
    quotas: dict[str, int],
    *,
    max_guard_per_region: int,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Cut contiguous segments and remove lower-priority boundary samples."""
    segments: dict[str, list[dict[str, Any]]] = {}
    position = 0
    for split in order:
        count = quotas[split]
        segments[split] = list(records[position : position + count])
        position += count
    if position != len(records):
        raise AssertionError("Temporal segment quotas do not cover the region")

    embargo: list[tuple[str, str]] = []
    for left_split, right_split in zip(order, order[1:]):
        while _segment_fused_tokens(segments[left_split]) & _segment_fused_tokens(
            segments[right_split]
        ):
            if len(embargo) >= max_guard_per_region:
                raise RuntimeError(
                    f"Exceeded {max_guard_per_region} embargo samples for {records[0]['region']}"
                )
            if SPLIT_PRIORITY[left_split] < SPLIT_PRIORITY[right_split]:
                removed = segments[left_split].pop()
                original_split = left_split
            else:
                removed = segments[right_split].pop(0)
                original_split = right_split
            embargo.append((str(removed["id"]), original_split))

    assignments = {
        str(record["id"]): split for split, segment in segments.items() for record in segment
    }
    labels = [assignments.get(str(record["id"])) for record in records]
    if _overlap_count(_fused_tokens(records, labels)) != 0:
        raise AssertionError("Priority embargo failed to remove fused-LiDAR leakage")
    return assignments, embargo


def build_annotations(
    records: list[dict[str, Any]],
    ignore_ids: set[str],
    *,
    ratios: dict[str, float],
    seed: int,
    stride_by_region: dict[str, int],
    max_guard_per_region: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = {str(record["id"]) for record in records}
    unknown_ignore = sorted(ignore_ids - manifest_ids)
    if unknown_ignore:
        raise ValueError(f"Ignore IDs absent from manifest: {unknown_ignore[:10]}")

    statuses: dict[str, tuple[str, str | None]] = {}
    candidates_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    region_indices: Counter[str] = Counter()
    for record in records:
        sample_id = str(record["id"])
        region = str(record["region"])
        index = region_indices[region]
        region_indices[region] += 1
        stride = int(stride_by_region.get(region, 1))
        if stride < 1:
            raise ValueError(f"Stride for {region} must be positive")
        if sample_id in ignore_ids:
            statuses[sample_id] = ("excluded", "source_data_error")
        elif index % stride:
            statuses[sample_id] = ("excluded", "sampling_stride")
        else:
            candidates_by_region[region].append(record)

    for region_records in candidates_by_region.values():
        region_records.sort(
            key=lambda record: (
                float(record.get("image_timestamp", 0.0)),
                str(record["id"]),
            )
        )
    quotas_by_region = apportion_region_quotas(
        {region: len(region_records) for region, region_records in candidates_by_region.items()},
        ratios,
        seed=seed,
    )
    orders_by_region = balanced_segment_orders(
        sorted(candidates_by_region),
        seed=seed,
    )
    region_orders: dict[str, list[str]] = {}
    embargo_by_region: dict[str, list[dict[str, str]]] = {}
    for region in sorted(candidates_by_region):
        region_records = candidates_by_region[region]
        order = orders_by_region[region]
        assignments, embargo = assign_priority_embargo(
            region_records,
            order,
            quotas_by_region[region],
            max_guard_per_region=max_guard_per_region,
        )
        region_orders[region] = list(order)
        embargo_by_region[region] = [
            {"id": sample_id, "original_split": original_split}
            for sample_id, original_split in embargo
        ]
        embargo_ids = {sample_id for sample_id, _ in embargo}
        for record in region_records:
            sample_id = str(record["id"])
            statuses[sample_id] = (
                ("excluded", "fused_lidar_leakage_guard")
                if sample_id in embargo_ids
                else (assignments[sample_id], None)
            )

    annotations: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["id"])
        split, reason = statuses[sample_id]
        annotation = {
            "schema_version": SCHEMA_VERSION,
            "id": sample_id,
            "split": split,
        }
        if reason is not None:
            annotation["reason"] = reason
        annotations.append(annotation)

    assigned_records = {
        str(record["id"]): record for record in records if statuses[str(record["id"])][0] in SPLITS
    }
    split_tokens = {split: set() for split in SPLITS}
    for sample_id, record in assigned_records.items():
        split = statuses[sample_id][0]
        split_tokens[split].update(
            f"{record['region']}/{name}" for name in record.get("fused_lidars", [])
        )
    overlap_by_pair = {
        f"{first}__{second}": len(split_tokens[first] & split_tokens[second])
        for first, second in itertools.combinations(SPLITS, 2)
    }
    if any(overlap_by_pair.values()):
        raise AssertionError(f"Cross-split fused LiDAR leakage: {overlap_by_pair}")

    counts = Counter(annotation["split"] for annotation in annotations)
    excluded_reasons = Counter(
        str(annotation["reason"]) for annotation in annotations if annotation["split"] == "excluded"
    )
    region_split_counts: dict[str, dict[str, int]] = {}
    region_by_id = {str(record["id"]): str(record["region"]) for record in records}
    for region in sorted(candidates_by_region):
        region_split_counts[region] = dict(
            sorted(
                Counter(
                    annotation["split"]
                    for annotation in annotations
                    if region_by_id[str(annotation["id"])] == region
                ).items()
            )
        )
    assigned_total = sum(counts[split] for split in SPLITS)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "per_region_contiguous_segments_global_ratio",
        "seed": seed,
        "requested_ratios": ratios,
        "realized_ratios": {split: counts[split] / assigned_total for split in SPLITS},
        "stride_by_region": dict(sorted(stride_by_region.items())),
        "region_quotas_before_embargo": quotas_by_region,
        "region_segment_orders": region_orders,
        "region_embargo_samples": embargo_by_region,
        "counts": dict(sorted(counts.items())),
        "excluded_reasons": dict(sorted(excluded_reasons.items())),
        "region_split_counts": region_split_counts,
        "leakage_guard": {
            "key": "(region, fused_lidar_filename)",
            "overlap_by_pair": overlap_by_pair,
            "verified_disjoint": True,
        },
        "assignment_semantic_sha256": canonical_sha256(
            [
                {
                    "id": annotation["id"],
                    "split": annotation["split"],
                    "reason": annotation.get("reason"),
                }
                for annotation in annotations
            ]
        ),
    }
    return annotations, metadata


def parse_stride(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        region, separator, raw_stride = value.partition("=")
        if not separator or not region or not raw_stride:
            raise ValueError(f"Invalid --stride value {value!r}; expected REGION=INTEGER")
        parsed[region] = int(raw_stride)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one global clean train/val/test annotation without copying data."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/slabim_504_r50/manifest.jsonl"),
    )
    parser.add_argument("--ignore-file", type=Path, default=Path("ignore.txt"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/annotations/slabim_clean_global_v1.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--stride",
        action="append",
        default=[],
        help="Post-manifest sampling stride, repeatable as REGION=INTEGER.",
    )
    parser.add_argument("--max-guard-per-region", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": float(args.test_ratio),
    }
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Train/val/test ratios must be positive and sum to 1")

    manifest_path = args.manifest.resolve()
    ignore_path = args.ignore_file.resolve()
    output_path = args.output.resolve()
    records = read_manifest(manifest_path)
    ignore_ids = read_ignore_ids(ignore_path)
    annotations, metadata = build_annotations(
        records,
        set(ignore_ids),
        ratios=ratios,
        seed=int(args.seed),
        stride_by_region=parse_stride(args.stride),
        max_guard_per_region=int(args.max_guard_per_region),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            handle.write(json.dumps(annotation, ensure_ascii=False) + "\n")
    metadata.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "ignore_file": str(ignore_path),
            "ignore_file_sha256": file_sha256(ignore_path),
            "ignore_id_count": len(ignore_ids),
            "annotation": str(output_path),
            "annotation_sha256": file_sha256(output_path),
        }
    )
    metadata_path = output_path.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
