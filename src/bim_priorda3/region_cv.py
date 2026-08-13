from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

_FINGERPRINT_SCHEMA = "bim-priorda3-region-cv-dataset-v2"
_NON_METRIC_FIELDS = {
    "count",
    "frames",
    "region_count",
    "sample_count",
    "valid_pixels",
}


def _parse_regions(value: object, *, field: str = "regions") -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence of region names")
    regions = tuple(value)
    if not regions:
        raise ValueError(f"{field} must not be empty")
    if any(not isinstance(region, str) or not region.strip() for region in regions):
        raise ValueError(f"{field} must contain non-empty strings")
    duplicate_regions = sorted(region for region, count in Counter(regions).items() if count > 1)
    if duplicate_regions:
        raise ValueError(f"{field} contains duplicates: {duplicate_regions}")
    return regions


def _parse_seeds(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("seeds must be a sequence of integers")
    seeds = tuple(value)
    if not seeds:
        raise ValueError("seeds must not be empty")
    if any(isinstance(seed, bool) or not isinstance(seed, Integral) for seed in seeds):
        raise ValueError("seeds must contain integers, not booleans")
    normalized = tuple(int(seed) for seed in seeds)
    if any(seed < 0 or seed > 2**32 - 1 for seed in normalized):
        raise ValueError("seeds must be in the range [0, 2**32 - 1]")
    duplicate_seeds = sorted(seed for seed, count in Counter(normalized).items() if count > 1)
    if duplicate_seeds:
        raise ValueError(f"seeds contains duplicates: {duplicate_seeds}")
    return normalized


@dataclass(frozen=True)
class RegionCVProtocol:
    """Validated region-level cross-validation protocol."""

    regions: tuple[str, ...]
    validation_pairs: tuple[tuple[str, str], ...]
    seeds: tuple[int, ...]

    @property
    def validation_map(self) -> dict[str, str]:
        return dict(self.validation_pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "regions": list(self.regions),
            "validation_map": self.validation_map,
            "seeds": list(self.seeds),
        }


def parse_region_cv_protocol(protocol: Mapping[str, object]) -> RegionCVProtocol:
    """Parse and validate a balanced leave-one-region-out protocol.

    ``validation_map`` assigns one validation region to every test region. It
    must be a derangement of ``regions`` so every region is used exactly once
    for validation and never validates its own test fold.
    """

    if not isinstance(protocol, Mapping):
        raise TypeError("protocol must be a mapping")
    missing = {"regions", "validation_map", "seeds"} - set(protocol)
    if missing:
        raise ValueError(f"protocol is missing required fields: {sorted(missing)}")

    regions = _parse_regions(protocol["regions"])
    if len(regions) < 3:
        raise ValueError("region cross-validation requires at least three regions")
    region_set = set(regions)

    raw_validation_map = protocol["validation_map"]
    if not isinstance(raw_validation_map, Mapping):
        raise TypeError("validation_map must be a mapping")
    unknown_keys = set(raw_validation_map) - region_set
    missing_keys = region_set - set(raw_validation_map)
    if unknown_keys or missing_keys:
        raise ValueError(
            "validation_map keys must exactly match regions; "
            f"missing={sorted(missing_keys)}, unknown={sorted(unknown_keys)}"
        )

    validation_pairs: list[tuple[str, str]] = []
    for test_region in regions:
        validation_region = raw_validation_map[test_region]
        if not isinstance(validation_region, str) or not validation_region.strip():
            raise ValueError("validation_map values must be non-empty region names")
        if validation_region not in region_set:
            raise ValueError(
                f"validation_map[{test_region!r}] references unknown region {validation_region!r}"
            )
        if validation_region == test_region:
            raise ValueError(f"validation_map[{test_region!r}] cannot use the test region itself")
        validation_pairs.append((test_region, validation_region))

    validation_counts = Counter(validation for _, validation in validation_pairs)
    if set(validation_counts) != region_set or any(
        count != 1 for count in validation_counts.values()
    ):
        raise ValueError("validation_map values must use every region exactly once")

    return RegionCVProtocol(
        regions=regions,
        validation_pairs=tuple(validation_pairs),
        seeds=_parse_seeds(protocol["seeds"]),
    )


@dataclass(frozen=True)
class RegionFoldPlan:
    """One outer test fold with its fixed validation region and training pool."""

    fold_index: int
    fold_id: str
    train_regions: tuple[str, ...]
    val_regions: tuple[str, ...]
    test_regions: tuple[str, ...]
    seeds: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "fold_id": self.fold_id,
            "train_regions": list(self.train_regions),
            "val_regions": list(self.val_regions),
            "test_regions": list(self.test_regions),
            "seeds": list(self.seeds),
        }


def build_region_fold_plans(
    protocol: RegionCVProtocol | Mapping[str, object],
) -> tuple[RegionFoldPlan, ...]:
    """Generate deterministic train/validation/test plans in protocol order."""

    parsed = (
        protocol if isinstance(protocol, RegionCVProtocol) else parse_region_cv_protocol(protocol)
    )
    validation_map = parsed.validation_map
    plans = []
    for fold_index, test_region in enumerate(parsed.regions):
        validation_region = validation_map[test_region]
        train_regions = tuple(
            region for region in parsed.regions if region not in {test_region, validation_region}
        )
        plans.append(
            RegionFoldPlan(
                fold_index=fold_index,
                fold_id=f"fold_{fold_index:02d}_{test_region}",
                train_regions=train_regions,
                val_regions=(validation_region,),
                test_regions=(test_region,),
                seeds=parsed.seeds,
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class DatasetFingerprint:
    """Content identity for a read-only, uniformly subsampled manifest."""

    sha256: str
    stride: int
    stride_by_region: tuple[tuple[str, int], ...]
    sample_count: int
    sampled_ids_by_region: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def region_counts(self) -> dict[str, int]:
        return {region: len(record_ids) for region, record_ids in self.sampled_ids_by_region}

    def to_dict(self, *, include_record_ids: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": _FINGERPRINT_SCHEMA,
            "sha256": self.sha256,
            "stride": self.stride,
            "stride_by_region": dict(self.stride_by_region),
            "sample_count": self.sample_count,
            "region_counts": self.region_counts,
        }
        if include_record_ids:
            result["record_ids"] = {
                region: list(record_ids) for region, record_ids in self.sampled_ids_by_region
            }
        return result


def dataset_fingerprint_from_manifest(
    manifest_path: str | Path,
    *,
    regions: Sequence[str] | None = None,
    stride: int = 1,
    stride_by_region: Mapping[str, int] | None = None,
) -> DatasetFingerprint:
    """Fingerprint region-wise sampled records using record IDs only.

    The manifest is never rewritten. Non-ID fields such as absolute sample
    paths, timestamps, and cached statistics do not affect the digest. The
    default ``stride`` may be overridden for individual regions, matching
    ``data.record_stride_by_region`` in the training dataset.
    """

    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("stride must be a positive integer")
    normalized_stride = int(stride)
    path = Path(manifest_path)

    ids_by_region: dict[str, list[str]] = {}
    first_seen_regions: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, Mapping):
                raise TypeError(f"{path}:{line_number}: record must be an object")
            region = record.get("region")
            record_id = record.get("id")
            if not isinstance(region, str) or not region.strip():
                raise ValueError(f"{path}:{line_number}: record has an invalid region")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"{path}:{line_number}: record has an invalid id")
            pair = (region, record_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"{path}:{line_number}: duplicate record ID {record_id!r} in region {region!r}"
                )
            seen_pairs.add(pair)
            if region not in ids_by_region:
                ids_by_region[region] = []
                first_seen_regions.append(region)
            ids_by_region[region].append(record_id)

    selected_regions = _parse_regions(regions) if regions is not None else tuple(first_seen_regions)
    if not selected_regions:
        raise ValueError(f"{path}: manifest contains no records")
    missing_regions = [region for region in selected_regions if region not in ids_by_region]
    if missing_regions:
        raise ValueError(f"{path}: requested regions are missing: {sorted(missing_regions)}")

    raw_overrides = stride_by_region or {}
    if not isinstance(raw_overrides, Mapping):
        raise TypeError("stride_by_region must be a mapping")
    unknown_overrides = set(raw_overrides) - set(selected_regions)
    if unknown_overrides:
        raise ValueError(
            f"stride_by_region references unselected regions: {sorted(unknown_overrides)}"
        )
    normalized_overrides = {}
    for region, region_stride in raw_overrides.items():
        if (
            isinstance(region_stride, bool)
            or not isinstance(region_stride, Integral)
            or region_stride < 1
        ):
            raise ValueError(f"stride_by_region[{region!r}] must be a positive integer")
        normalized_overrides[str(region)] = int(region_stride)
    effective_strides = tuple(
        (
            region,
            normalized_overrides.get(region, normalized_stride),
        )
        for region in selected_regions
    )
    effective_stride_map = dict(effective_strides)
    sampled = tuple(
        (
            region,
            tuple(ids_by_region[region][:: effective_stride_map[region]]),
        )
        for region in selected_regions
    )
    # Region ordering is a protocol concern, not dataset content. Sort only the
    # digest payload so the same selected dataset has the same fingerprint.
    canonical_regions = [
        {"region": region, "record_ids": list(record_ids)} for region, record_ids in sorted(sampled)
    ]
    canonical_payload = {
        "schema": _FINGERPRINT_SCHEMA,
        "sampling": {
            "default_stride": normalized_stride,
            "stride_by_region": dict(sorted(normalized_overrides.items())),
        },
        "regions": canonical_regions,
    }
    encoded = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return DatasetFingerprint(
        sha256=hashlib.sha256(encoded).hexdigest(),
        stride=normalized_stride,
        stride_by_region=effective_strides,
        sample_count=sum(len(record_ids) for _, record_ids in sampled),
        sampled_ids_by_region=sampled,
    )


def region_macro_summary(
    per_region: Mapping[str, Mapping[str, Real]],
    metric_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Summarize metrics with every region receiving equal weight."""

    if not isinstance(per_region, Mapping) or not per_region:
        raise ValueError("per_region must be a non-empty mapping")
    regions = tuple(sorted(per_region))
    if any(not isinstance(region, str) or not region for region in regions):
        raise ValueError("per_region keys must be non-empty region names")
    if any(not isinstance(per_region[region], Mapping) for region in regions):
        raise TypeError("each per_region value must be a metric mapping")

    if metric_names is None:
        common_names = set(per_region[regions[0]])
        for region in regions[1:]:
            common_names &= set(per_region[region])
        metrics = tuple(
            sorted(
                name
                for name in common_names
                if isinstance(name, str)
                and name not in _NON_METRIC_FIELDS
                and not name.endswith("_count")
            )
        )
    else:
        metrics = _parse_regions(metric_names, field="metric_names")
    if not metrics:
        raise ValueError("no common metrics are available for region-macro summary")

    summaries: dict[str, dict[str, float]] = {}
    for metric in metrics:
        values = []
        for region in regions:
            if metric not in per_region[region]:
                raise ValueError(f"region {region!r} is missing metric {metric!r}")
            value = per_region[region][metric]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"region {region!r} metric {metric!r} must be numeric")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"region {region!r} metric {metric!r} must be finite")
            values.append(normalized)
        summaries[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "aggregation": "region_macro",
        "region_count": len(regions),
        "regions": list(regions),
        "metrics": summaries,
    }
