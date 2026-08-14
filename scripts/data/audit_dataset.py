#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bim_priorda3.baselines import configured_scale_and_local_features
from bim_priorda3.config import (
    Config,
    load_config,
    resolve_project_path,
    resolve_slabim_root,
)
from bim_priorda3.data import (
    AnnotationSplitResolution,
    BIMDepthDataset,
    relocate_record,
    resolve_annotation_splits,
)

ACTIVE_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit prepared data and split integrity")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--ignore-file",
        type=Path,
        help=(
            "Quality-exclusion list to verify against the manifest and annotation. "
            "By default data.ignore_file is used, then a project-root ignore.txt "
            "when it belongs to this dataset."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number}: manifest row must be an object")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no records: {path}")
    return records


def _read_ignore_ids(path: Path) -> tuple[str, ...]:
    sample_ids: list[str] = []
    first_line_by_id: dict[str, int] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        sample_id = raw_line.partition("#")[0].strip()
        if not sample_id:
            continue
        parts = sample_id.split("/")
        if len(parts) != 2 or not all(parts) or any(character.isspace() for character in sample_id):
            raise ValueError(
                f"{path}:{line_number}: expected an exact '<region>/<frame>' sample ID"
            )
        if sample_id in first_line_by_id:
            raise ValueError(
                f"{path}:{line_number}: duplicate sample ID {sample_id!r}; "
                f"first declared on line {first_line_by_id[sample_id]}"
            )
        first_line_by_id[sample_id] = line_number
        sample_ids.append(sample_id)
    if not sample_ids:
        raise ValueError(f"Ignore file contains no sample IDs: {path}")
    return tuple(sample_ids)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_region_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["region"]) for record in records).items()))


def _configured_ignore_path(
    cfg: Config,
    explicit_path: Path | None,
) -> tuple[Path | None, str, bool]:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve(), "command_line", True
    config_value = cfg.data.get("ignore_file")
    if config_value:
        return resolve_project_path(cfg, config_value), "config", True
    discovered = Path(cfg.project_root) / "ignore.txt"
    if discovered.is_file():
        return discovered.resolve(), "project_default", False
    return None, "not_found", False


def _audit_ignore_population(
    cfg: Config,
    manifest_records: list[dict[str, Any]],
    active_records: list[dict[str, Any]],
    resolution: AnnotationSplitResolution | None,
    explicit_path: Path | None,
) -> dict[str, Any]:
    path, source, strict = _configured_ignore_path(cfg, explicit_path)
    if path is None:
        return {
            "present": False,
            "applicable": False,
            "source": source,
            "dataset_input_exclusion_verified": False,
        }
    if not path.is_file():
        raise FileNotFoundError(f"Ignore file does not exist: {path}")

    declared_ids = _read_ignore_ids(path)
    declared_set = set(declared_ids)
    manifest_ids = {str(record["id"]) for record in manifest_records}
    matched_ids = declared_set & manifest_ids
    unknown_ids = sorted(declared_set - manifest_ids)

    # A repository may contain an ignore list for SLABIM while auditing another
    # annotated dataset (for example Stanford Area_1).  Only an automatically
    # discovered list with no matching IDs is treated as non-applicable.
    if not matched_ids and not strict:
        return {
            "present": True,
            "applicable": False,
            "source": source,
            "ignore_file": str(path),
            "ignore_file_sha256": _file_sha256(path),
            "declared_sample_count": len(declared_ids),
            "manifest_match_count": 0,
            "foreign_sample_count": len(unknown_ids),
            "dataset_input_exclusion_verified": False,
        }
    if unknown_ids:
        preview = ", ".join(unknown_ids[:10])
        suffix = "" if len(unknown_ids) <= 10 else f", ... (+{len(unknown_ids) - 10})"
        raise ValueError(
            f"{path}: {len(unknown_ids)} ignore IDs are absent from the manifest: {preview}{suffix}"
        )

    active_ids = {str(record["id"]) for record in active_records}
    report: dict[str, Any] = {
        "present": True,
        "applicable": True,
        "source": source,
        "ignore_file": str(path),
        "ignore_file_sha256": _file_sha256(path),
        "declared_sample_count": len(declared_ids),
        "manifest_match_count": len(matched_ids),
        "active_population_match_count": len(declared_set & active_ids),
    }
    if resolution is None:
        report.update(
            {
                "dataset_input_exclusion_verified": False,
                "policy": "legacy_region_split_reports_but_does_not_apply_ignore_list",
            }
        )
        return report

    source_error_ids = {
        sample_id
        for sample_id, reason in resolution.excluded_reasons.items()
        if reason == "source_data_error"
    }
    not_excluded = sorted(
        sample_id for sample_id in declared_ids if resolution.assignments[sample_id] != "excluded"
    )
    wrong_reason = sorted(declared_set - source_error_ids)
    undeclared_source_errors = sorted(source_error_ids - declared_set)
    if not_excluded or wrong_reason or undeclared_source_errors:
        raise ValueError(
            "Ignore list and exhaustive annotation disagree: "
            f"not_excluded={not_excluded[:10]}, "
            f"wrong_exclusion_reason={wrong_reason[:10]}, "
            f"source_data_errors_missing_from_ignore={undeclared_source_errors[:10]}"
        )
    report.update(
        {
            "annotation_source_data_error_count": len(source_error_ids),
            "annotation_excluded_count": len(resolution.excluded_reasons),
            "dataset_input_exclusion_verified": True,
            "policy": "annotation_excluded_with_reason_source_data_error",
        }
    )
    return report


def _resolve_dataset_splits(
    cfg: Config,
    records: list[dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    AnnotationSplitResolution | None,
    dict[str, Any],
]:
    annotation_value = cfg.data.get("split_annotation")
    resolution: AnnotationSplitResolution | None = None
    if annotation_value:
        annotation_path = resolve_project_path(cfg, annotation_value)
        resolution = resolve_annotation_splits(records, annotation_path)

    train_regions = {str(region) for region in cfg.data.train_regions}
    val_regions = {str(region) for region in cfg.data.val_regions}
    test_regions = {str(region) for region in cfg.data.test_regions}
    if resolution is None:
        region_overlap = (
            (train_regions & val_regions)
            | (train_regions & test_regions)
            | (val_regions & test_regions)
        )
        if region_overlap:
            raise RuntimeError(f"Train/validation/test region leakage: {sorted(region_overlap)}")

    datasets = {split: BIMDepthDataset(cfg, split, augment=False) for split in ACTIVE_SPLITS}
    records_by_split = {split: list(dataset.records) for split, dataset in datasets.items()}
    ids_by_split = {
        split: [str(record["id"]) for record in split_records]
        for split, split_records in records_by_split.items()
    }
    owners: dict[str, list[str]] = defaultdict(list)
    for split, sample_ids in ids_by_split.items():
        for sample_id in sample_ids:
            owners[sample_id].append(split)
    overlaps = sorted(sample_id for sample_id, splits in owners.items() if len(splits) > 1)
    if overlaps:
        raise RuntimeError("Train/validation/test sample leakage: " + ", ".join(overlaps[:10]))

    if resolution is not None:
        selected_regions = {str(region) for region in cfg.data.regions}
        for split in ACTIVE_SPLITS:
            expected_ids = [
                str(record["id"])
                for record in resolution.records_for(split)
                if str(record["region"]) in selected_regions
            ]
            if ids_by_split[split] != expected_ids:
                raise RuntimeError(
                    f"BIMDepthDataset {split!r} records disagree with resolve_annotation_splits"
                )
        population = {
            "mode": "annotations",
            "manifest_samples": len(records),
            "annotation_samples": resolution.provenance["annotation_record_count"],
            "annotation_split_counts": resolution.provenance["split_counts"],
            "annotation_split_region_counts": resolution.provenance["split_region_counts"],
            "selected_regions": sorted(selected_regions),
            "dataset_split_counts": {
                split: len(records_by_split[split]) for split in ACTIVE_SPLITS
            },
            "dataset_split_region_counts": {
                split: _record_region_counts(records_by_split[split]) for split in ACTIVE_SPLITS
            },
            "active_samples": sum(len(records_by_split[split]) for split in ACTIVE_SPLITS),
            "excluded_samples": len(resolution.records_for("excluded")),
            "sample_id_disjoint": True,
            "fused_lidar_disjoint": resolution.provenance["fused_lidar_validation"]["disjoint"],
            "dataset_matches_annotation_resolver": True,
            "annotation_provenance": resolution.provenance,
        }
        return records_by_split, resolution, population

    population = {
        "mode": "regions",
        "manifest_samples": len(records),
        "configured_split_regions": {
            "train": sorted(train_regions),
            "val": sorted(val_regions),
            "test": sorted(test_regions),
        },
        "record_stride_by_region": dict(
            sorted(
                (str(region), int(stride))
                for region, stride in cfg.data.get(
                    "record_stride_by_region",
                    {},
                ).items()
            )
        ),
        "dataset_split_counts": {split: len(records_by_split[split]) for split in ACTIVE_SPLITS},
        "dataset_split_region_counts": {
            split: _record_region_counts(records_by_split[split]) for split in ACTIVE_SPLITS
        },
        "active_samples": sum(len(records_by_split[split]) for split in ACTIVE_SPLITS),
        "sample_id_disjoint": True,
        "region_disjoint": True,
    }
    return records_by_split, None, population


def _depth_statistics(
    records: list[dict[str, Any]],
    scale_estimator: dict[str, Any],
) -> dict[str, dict[str, float]]:
    region_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "samples": 0,
            "gt_pixels": 0,
            "bim_pixels": 0,
            "overlap_pixels": 0,
            "bim_wins": 0,
            "scaled_da3_wins": 0,
            "scaled_abs_error_sum": 0.0,
            "bim_abs_error_sum": 0.0,
        }
    )
    for record in records:
        with np.load(record["sample"]) as sample:
            base = sample["base_depth"].astype(np.float32)
            bim = sample["bim_depth"].astype(np.float32)
            scaled = configured_scale_and_local_features(
                base,
                bim,
                scale_estimator,
            )[0]
            gt = sample["gt_depth"].astype(np.float32)
            gt_valid = sample["gt_valid"] > 0
            bim_valid = sample["bim_valid"] > 0
        overlap_mask = gt_valid & bim_valid & (scaled > 0) & (bim > 0) & (gt > 0)
        scaled_error = np.abs(
            np.log(np.maximum(scaled[overlap_mask], 1e-4)) - np.log(gt[overlap_mask])
        )
        bim_error = np.abs(np.log(np.maximum(bim[overlap_mask], 1e-4)) - np.log(gt[overlap_mask]))
        stats = region_stats[str(record["region"])]
        stats["samples"] += 1
        stats["gt_pixels"] += int(gt_valid.sum())
        stats["bim_pixels"] += int(bim_valid.sum())
        stats["overlap_pixels"] += int(overlap_mask.sum())
        stats["bim_wins"] += int((bim_error < scaled_error).sum())
        stats["scaled_da3_wins"] += int((scaled_error <= bim_error).sum())
        stats["scaled_abs_error_sum"] += float(scaled_error.sum())
        stats["bim_abs_error_sum"] += float(bim_error.sum())

    for stats in region_stats.values():
        samples = max(stats["samples"], 1)
        overlap_pixels = max(stats["overlap_pixels"], 1)
        stats["mean_gt_pixels"] = stats["gt_pixels"] / samples
        stats["mean_bim_pixels"] = stats["bim_pixels"] / samples
        stats["bim_win_fraction"] = stats["bim_wins"] / overlap_pixels
        stats["mean_scaled_log_error"] = stats["scaled_abs_error_sum"] / overlap_pixels
        stats["mean_bim_log_error"] = stats["bim_abs_error_sum"] / overlap_pixels
    return dict(region_stats)


def build_audit_report(
    cfg: Config,
    *,
    ignore_file: Path | None = None,
) -> dict[str, Any]:
    root = resolve_project_path(cfg, cfg.data.processed_root)
    manifest_path = root / "manifest.jsonl"
    manifest_records = _read_manifest(manifest_path)
    source_value = cfg.data.get("source_root")
    source_root = resolve_project_path(cfg, source_value) if source_value else None
    slabim_root = resolve_slabim_root(cfg) if cfg.data.get("slabim_root") else source_root
    if slabim_root is None:
        raise ValueError("data.slabim_root or data.source_root must be configured")
    records = [
        relocate_record(record, root, slabim_root, source_root) for record in manifest_records
    ]

    records_by_split, resolution, split_population = _resolve_dataset_splits(
        cfg,
        records,
    )
    active_records = [record for split in ACTIVE_SPLITS for record in records_by_split[split]]
    ignore_population = _audit_ignore_population(
        cfg,
        records,
        active_records,
        resolution,
        ignore_file,
    )
    region_stats = _depth_statistics(
        active_records,
        dict(cfg.model.get("scale_estimator", {})),
    )

    configured_regions = {
        split: sorted(str(region) for region in cfg.data.get(f"{split}_regions", []))
        for split in ACTIVE_SPLITS
    }
    return {
        "schema_version": 2,
        "manifest": str(manifest_path.resolve()),
        "manifest_samples": len(records),
        "samples": len(active_records),
        "split_mode": split_population["mode"],
        # Retain the legacy top-level fields for report consumers.
        "train_regions": configured_regions["train"],
        "val_regions": configured_regions["val"],
        "test_regions": configured_regions["test"],
        "region_leakage": False,
        "prepared_train_samples": len(records_by_split["train"]),
        "prepared_val_samples": len(records_by_split["val"]),
        "prepared_test_samples": len(records_by_split["test"]),
        "split_population": split_population,
        "ignore_population": ignore_population,
        "regions": region_stats,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    report = build_audit_report(cfg, ignore_file=args.ignore_file)
    root = resolve_project_path(cfg, cfg.data.processed_root)
    output = args.output or root / "audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
