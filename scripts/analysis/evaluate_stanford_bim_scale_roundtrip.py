#!/usr/bin/env python3
"""Validation-only BIM-scale regular -> ERP -> regular paired evaluation.

Every method is evaluated on every original regular frame and the exact same
prepared 0.2--5.0 m GT pixels.  The confirmatory comparison changes only one
factor: per-frame universal BIM scale either remains in its original regular
view or is projected to ERP, fused with the other same-station scaled regular
views, and reprojected back.  No panorama RGB/depth, tangent view, learned
checkpoint, test split, or GT-dependent fusion parameter is used.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.stanford_pano import (
    StanfordPanorama,
    build_regular_pano_lookup,
    discover_stanford_panoramas,
)
from bim_priorda3.engine import seed_everything

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import evaluate_stanford_pano_regular_roundtrip as roundtrip
from scripts.analysis import evaluate_stanford_pano_single_plus_tangent as artifact_utils
from scripts.model import evaluate_stanford_pano as evaluator

PROTOCOL = "stanford-area1-val-bim-scale-regular-erp-roundtrip-v1"
SCHEMA_VERSION = 1
SPLIT = "val"
PANO_SHAPE = (512, 1024)
REGULAR_SHAPE = (504, 504)
SEED = 42
BOOTSTRAP_REPETITIONS = 10_000
EXPECTED_FRAME_COUNT = 1_673
EXPECTED_STATION_COUNT = 30
EXPECTED_ROOM_COUNT = 7
DEPTH_METHODS = ("universal_scale", "bim_direct")
FUSION_METHODS = (
    "joint_weighted_log",
    "joint_huber",
    "joint_synchronized_huber",
)
PRIMARY_DEPTH_METHOD = "universal_scale"
PRIMARY_FUSION_METHOD = "joint_huber"


def _method_name(depth_method: str, operation: str) -> str:
    return f"{depth_method}__{operation}"


METHOD_NAMES = tuple(
    name
    for depth_method in DEPTH_METHODS
    for name in (
        _method_name(depth_method, "per_frame"),
        _method_name(depth_method, "projection_roundtrip"),
        *(_method_name(depth_method, fusion_method) for fusion_method in FUSION_METHODS),
    )
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-frame BIM-scale regular depth with the same scaled predictions "
            "after same-station ERP fusion and regular-view reprojection. Validation only."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stanford_area1.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stanford_area1/pano_val_bim_scale_regular_roundtrip"),
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        help="Ordered-prefix smoke test; omit for the exhaustive validation experiment.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_stations is not None and args.max_stations <= 0:
        parser.error("--max-stations must be positive")
    return args


def _identity_erp_prediction(
    view: evaluator.ProjectedView,
    depth_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    pixel_count = int(np.prod(PANO_SHAPE))
    if depth_method not in DEPTH_METHODS:
        raise ValueError(f"Unsupported identity depth method: {depth_method}")
    view.validate(DEPTH_METHODS, pixel_count)
    flat = np.zeros(pixel_count, dtype=np.float32)
    valid = np.zeros(pixel_count, dtype=bool)
    flat[view.indices] = np.exp(view.log_ranges[depth_method]).astype(np.float32)
    valid[view.indices] = True
    return flat.reshape(PANO_SHAPE), valid.reshape(PANO_SHAPE)


def _room_cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: str,
    reference: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = SEED,
) -> dict[str, Any]:
    by_room: defaultdict[str, defaultdict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        by_room[str(row["room"])][str(row["method"])].append(row)
    room_ids = sorted(by_room)
    for room in room_ids:
        if set(by_room[room]) != set(METHOD_NAMES):
            raise RuntimeError(f"{room}: incomplete paired method matrix")

    def sufficient(room: str, method: str) -> tuple[float, int]:
        selected = by_room[room][method]
        count = sum(int(row["count"]) for row in selected)
        numerator = sum(float(row["abs_rel"]) * int(row["count"]) for row in selected)
        return numerator, count

    candidate_stats = [sufficient(room, candidate) for room in room_ids]
    reference_stats = [sufficient(room, reference) for room in room_ids]
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        draw = rng.integers(0, len(room_ids), size=len(room_ids))
        candidate_num = sum(candidate_stats[item][0] for item in draw)
        candidate_count = sum(candidate_stats[item][1] for item in draw)
        reference_num = sum(reference_stats[item][0] for item in draw)
        reference_count = sum(reference_stats[item][1] for item in draw)
        differences[index] = candidate_num / candidate_count - reference_num / reference_count

    def aggregate_abs_rel(method: str) -> float:
        selected = [row for row in rows if row["method"] == method]
        count = sum(int(row["count"]) for row in selected)
        return sum(float(row["abs_rel"]) * int(row["count"]) for row in selected) / count

    candidate_metric = aggregate_abs_rel(candidate)
    reference_metric = aggregate_abs_rel(reference)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    room_differences = []
    for room, candidate_stat, reference_stat in zip(
        room_ids,
        candidate_stats,
        reference_stats,
    ):
        difference = candidate_stat[0] / candidate_stat[1] - reference_stat[0] / reference_stat[1]
        room_differences.append((room, difference))
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": "original regular-frame fixed-support pixel-micro AbsRel",
        "difference_definition": "candidate - reference (negative is better)",
        "candidate_abs_rel": candidate_metric,
        "reference_abs_rel": reference_metric,
        "mean_difference": candidate_metric - reference_metric,
        "relative_reduction_percent": 100.0
        * (reference_metric - candidate_metric)
        / reference_metric,
        "confidence_interval_95": [float(lower), float(upper)],
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "resampling_unit": "room; retain all stations and frames in each sampled room",
        "room_count": len(room_ids),
        "candidate_better_room_count": sum(value < 0.0 for _, value in room_differences),
        "room_differences": [
            {"room": room, "candidate_minus_reference_abs_rel": difference}
            for room, difference in room_differences
        ],
    }


def _evaluate_station(
    station: StanfordPanorama,
    records: Sequence[Mapping[str, Any]],
    cfg: Any,
    fusion_args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_records = sorted(
        records, key=lambda value: (int(value["frame_number"]), str(value["id"]))
    )
    frames = [
        evaluator._read_regular_frame(
            record,
            cfg,
            evaluator.DEPTH_METHODS,
            station_scale=1.0,
        )
        for record in ordered_records
    ]
    frame_by_id = {frame.sample_id: frame for frame in frames}
    if len(frame_by_id) != len(frames):
        raise RuntimeError(f"{station.camera_uuid}: duplicate regular frame ID")
    pano_pixels = evaluator._erp_pixel_grid(*PANO_SHAPE)
    pano_rgb_placeholder = np.zeros((*PANO_SHAPE, 3), dtype=np.float32)
    projected_views = [
        evaluator._project_regular_view(
            frame,
            station,
            pano_pixels,
            pano_rgb_placeholder,
            DEPTH_METHODS,
            centrality_power=float(fusion_args.centrality_power),
            confidence_floor=float(fusion_args.confidence_floor),
            photo_sigma=float(fusion_args.photo_sigma),
        )
        for frame in frames
    ]
    view_by_id = {view.frame_id: view for view in projected_views}
    fusion = evaluator._aggregate_with_args(
        projected_views,
        DEPTH_METHODS,
        PANO_SHAPE,
        fusion_args,
    )

    rows: list[dict[str, Any]] = []
    station_coverages: defaultdict[str, list[float]] = defaultdict(list)
    for record in ordered_records:
        sample_id = str(record["id"])
        frame = frame_by_id[sample_id]
        view = view_by_id[sample_id]
        lookup = build_regular_pano_lookup(
            PANO_SHAPE,
            station.camera_to_area,
            np.linalg.inv(frame.camera_to_area),
            frame.intrinsic,
            REGULAR_SHAPE,
        )
        predictions: dict[str, tuple[np.ndarray, np.ndarray, str, str]] = {}
        for depth_method in DEPTH_METHODS:
            baseline = frame.predictions[depth_method]
            predictions[_method_name(depth_method, "per_frame")] = (
                baseline,
                np.ones(REGULAR_SHAPE, dtype=bool),
                "single_regular",
                "identity_no_roundtrip",
            )
            identity_range, identity_valid = _identity_erp_prediction(view, depth_method)
            predictions[_method_name(depth_method, "projection_roundtrip")] = (
                *roundtrip._roundtrip_with_raw_fallback(
                    identity_range,
                    identity_valid,
                    lookup,
                    baseline,
                ),
                "single_regular",
                "projection_roundtrip_control",
            )
            for fusion_method in FUSION_METHODS:
                pano_prediction = fusion.predictions[fusion_method][depth_method]
                pano_valid = (
                    (fusion.contributor_count > 0)
                    & np.isfinite(pano_prediction)
                    & (pano_prediction > 0.0)
                )
                predictions[_method_name(depth_method, fusion_method)] = (
                    *roundtrip._roundtrip_with_raw_fallback(
                        pano_prediction,
                        pano_valid,
                        lookup,
                        baseline,
                    ),
                    "all_regular",
                    fusion_method,
                )
        if tuple(predictions) != METHOD_NAMES:
            raise AssertionError("BIM-scale round-trip method order differs from protocol")

        # All station-level predictions are frozen before prepared regular GT is opened.
        gt_depth, gt_valid = roundtrip._read_regular_gt(record, REGULAR_SHAPE)
        for method, (prediction, native_valid, source_set, fusion_method) in predictions.items():
            row = roundtrip._metric_row(
                record=record,
                method=method,
                source_set=source_set,
                fusion_method=fusion_method,
                prediction=prediction,
                target=gt_depth,
                support=gt_valid,
                native_valid=native_valid,
            )
            rows.append(row)
            station_coverages[method].append(float(row["native_roundtrip_coverage_fraction"]))
    if len(rows) != len(ordered_records) * len(METHOD_NAMES):
        raise RuntimeError(f"{station.camera_uuid}: incomplete frame/method matrix")
    return rows, {
        "station_id": station.camera_uuid,
        "room": station.room,
        "regular_frame_count": len(ordered_records),
        "regular_frame_ids": [str(record["id"]) for record in ordered_records],
        "regular_projected_view_count": len(projected_views),
        "prediction_frozen_before_regular_gt_open": True,
        "mean_native_roundtrip_coverage": {
            method: float(np.mean(values)) for method, values in sorted(station_coverages.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(SEED)
    cfg = load_config(args.config)
    if float(cfg.data.min_depth) != 0.2 or float(cfg.data.max_depth) != 5.0:
        raise ValueError("Protocol requires the frozen 0.2--5.0 m regular range")
    dataset = BIMDepthDataset(cfg, SPLIT, augment=False, require_ground_truth=False)
    if len(dataset.records) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAME_COUNT} val frames, got {len(dataset.records)}"
        )
    records_by_station: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in dataset.records:
        records_by_station[str(record["camera_uuid"])].append(record)

    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    station_by_id = {
        station.camera_uuid: station for station in discover_stanford_panoramas(area_root)
    }
    station_ids = sorted(records_by_station)
    if len(station_ids) != EXPECTED_STATION_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_STATION_COUNT} paired val stations, got {len(station_ids)}"
        )
    if args.max_stations is not None:
        station_ids = station_ids[: int(args.max_stations)]
    stations = [station_by_id[camera_uuid] for camera_uuid in station_ids]
    output_dir = args.output.expanduser().resolve()
    artifact_utils._ensure_new_output(output_dir)
    fusion_args = roundtrip._fusion_args()

    frame_rows: list[dict[str, Any]] = []
    station_receipts = []
    for index, station in enumerate(stations, start=1):
        print(
            f"[{index}/{len(stations)}] {station.room}/{station.camera_uuid} "
            f"({len(records_by_station[station.camera_uuid])} regular frames)",
            flush=True,
        )
        rows, receipt = _evaluate_station(
            station,
            records_by_station[station.camera_uuid],
            cfg,
            fusion_args,
        )
        frame_rows.extend(rows)
        station_receipts.append(receipt)

    expected_frames = sum(len(records_by_station[station_id]) for station_id in station_ids)
    if len(frame_rows) != expected_frames * len(METHOD_NAMES):
        raise RuntimeError("BIM-scale round-trip matrix is incomplete")
    for sample_id in sorted({str(row["sample_id"]) for row in frame_rows}):
        selected = [row for row in frame_rows if row["sample_id"] == sample_id]
        if {str(row["method"]) for row in selected} != set(METHOD_NAMES):
            raise RuntimeError(f"{sample_id}: incomplete method set")
        if len({int(row["count"]) for row in selected}) != 1:
            raise RuntimeError(f"{sample_id}: support differs across methods")

    station_rows = roundtrip._group_rows(frame_rows, "station_id")
    room_rows = roundtrip._group_rows(frame_rows, "room")
    metrics = {
        method: roundtrip._method_summary(frame_rows, station_rows, room_rows, method)
        for method in METHOD_NAMES
    }
    contrasts = []
    for depth_method in DEPTH_METHODS:
        baseline = _method_name(depth_method, "per_frame")
        identity = _method_name(depth_method, "projection_roundtrip")
        contrasts.append(
            _room_cluster_bootstrap(frame_rows, candidate=identity, reference=baseline)
        )
        for fusion_method in FUSION_METHODS:
            contrasts.append(
                _room_cluster_bootstrap(
                    frame_rows,
                    candidate=_method_name(depth_method, fusion_method),
                    reference=baseline,
                )
            )
    primary_reference = _method_name(PRIMARY_DEPTH_METHOD, "per_frame")
    primary_candidate = _method_name(PRIMARY_DEPTH_METHOD, PRIMARY_FUSION_METHOD)
    primary_contrast = _room_cluster_bootstrap(
        frame_rows,
        candidate=primary_candidate,
        reference=primary_reference,
    )

    per_frame_path = output_dir / "per_frame.csv"
    per_station_path = output_dir / "per_station.csv"
    per_room_path = output_dir / "per_room.csv"
    artifact_utils._write_csv(per_frame_path, frame_rows)
    artifact_utils._write_csv(per_station_path, station_rows)
    artifact_utils._write_csv(per_room_path, room_rows)
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(str(cfg.config_path)).resolve()
    annotation_path = resolve_project_path(cfg, cfg.data.split_annotation)
    prepared_manifest = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
    status = "exploratory_validation_only" if args.max_stations is None else "nonformal_smoke"
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "publication_status": "candidate_for_future_preregistration",
        "split": SPLIT,
        "test_access": {
            "authorized": False,
            "test_raw_files_opened": False,
            "test_csv_opened": False,
        },
        "method_scope": {
            "training_free_only": True,
            "checkpoint": False,
            "learned": False,
            "BIM_scale": True,
            "BIM_direct_secondary": True,
            "pano_RGB": False,
            "pano_GT": False,
            "pano_tangent": False,
            "regular_GT_opened_after_station_prediction_freeze_only": True,
        },
        "script": {
            "path": artifact_utils._repo_relative(Path(__file__), project_root),
            "sha256": artifact_utils._sha256(Path(__file__)),
        },
        "shared_code": {
            name: {
                "path": artifact_utils._repo_relative(path, project_root),
                "sha256": artifact_utils._sha256(path),
            }
            for name, path in {
                "pano_evaluator": Path(evaluator.__file__).resolve(),
                "roundtrip_evaluator": Path(roundtrip.__file__).resolve(),
                "stanford_pano_geometry": project_root / "src/bim_priorda3/data/stanford_pano.py",
                "baselines": project_root / "src/bim_priorda3/baselines.py",
            }.items()
        },
        "config": {
            "path": artifact_utils._repo_relative(config_path, project_root),
            "sha256": artifact_utils._sha256(config_path),
        },
        "dataset_split_provenance": artifact_utils._portable_split_provenance(
            dataset.split_provenance,
            project_root,
        ),
        "split_annotation": {
            "path": artifact_utils._repo_relative(annotation_path, project_root),
            "sha256": artifact_utils._sha256(annotation_path),
        },
        "prepared_manifest": {
            "path": artifact_utils._repo_relative(prepared_manifest, project_root),
            "sha256": artifact_utils._sha256(prepared_manifest),
        },
        "parameters": {
            "pano_shape": list(PANO_SHAPE),
            "regular_shape": list(REGULAR_SHAPE),
            "depth_range_m": [float(cfg.data.min_depth), float(cfg.data.max_depth)],
            "depth_methods": list(DEPTH_METHODS),
            "fusion_methods": list(FUSION_METHODS),
            "primary_depth_method": PRIMARY_DEPTH_METHOD,
            "primary_fusion_method": PRIMARY_FUSION_METHOD,
            "primary_fusion_selection": "frozen joint_huber from prior validation protocol",
            "scale_estimator": dict(cfg.model.scale_estimator),
            "fusion_parameters": artifact_utils.FUSION_PARAMETERS,
            "missing_roundtrip_policy": "per-frame same-depth-method fallback; coverage reported",
            "bootstrap": {
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": SEED,
                "cluster": "room",
            },
        },
        "population": {
            "full_validation": args.max_stations is None,
            "station_count": len(stations),
            "frame_count": expected_frames,
            "room_count": len({station.room for station in stations}),
            "ordered_station_ids": station_ids,
            "station_receipts": station_receipts,
        },
        "artifacts": {
            "per_frame_csv": {
                "path": artifact_utils._repo_relative(per_frame_path, project_root),
                "sha256": artifact_utils._sha256(per_frame_path),
            },
            "per_station_csv": {
                "path": artifact_utils._repo_relative(per_station_path, project_root),
                "sha256": artifact_utils._sha256(per_station_path),
            },
            "per_room_csv": {
                "path": artifact_utils._repo_relative(per_room_path, project_root),
                "sha256": artifact_utils._sha256(per_room_path),
            },
        },
    }
    artifact_utils._assert_no_private_absolute_paths(provenance)
    provenance_path = output_dir / "provenance.json"
    artifact_utils._write_json(provenance_path, provenance)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "publication_status": "candidate_for_future_preregistration",
        "split": SPLIT,
        "test_data_or_csv_accessed": False,
        "question": (
            "On identical original regular frames and GT pixels, does same-station ERP fusion "
            "improve predictions after every frame receives the same universal BIM-scale method?"
        ),
        "population": {
            "frame_count": expected_frames,
            "station_count": len(stations),
            "room_count": len({station.room for station in stations}),
        },
        "fixed_support": (
            "each original regular frame's complete prepared 0.2--5.0 m gt_valid; "
            "identical for per-frame, identity-roundtrip, and joint methods"
        ),
        "primary_contrast": primary_contrast,
        "methods": metrics,
        "all_predeclared_contrasts": contrasts,
        "method_scope": provenance["method_scope"],
        "artifacts": {
            **provenance["artifacts"],
            "provenance": {
                "path": artifact_utils._repo_relative(provenance_path, project_root),
                "sha256": artifact_utils._sha256(provenance_path),
            },
        },
    }
    artifact_utils._assert_no_private_absolute_paths(summary)
    summary_path = output_dir / "summary.json"
    artifact_utils._write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
