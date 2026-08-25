#!/usr/bin/env python3
"""Validation-only regular -> ERP fusion -> regular round-trip evaluation.

This protocol evaluates the question that a regular-view benchmark actually
cares about: after all same-station regular predictions (and optionally pano
tangent predictions) are fused on the sphere, does reprojecting that joint
estimate improve the *original regular frames on their original GT pixels*?

It is deliberately validation-only and training-free.  Panorama GT is never
opened; prepared regular GT is opened only after every station-level spherical
prediction has been frozen.  Missing round-trip samples fall back to the
original frame prediction so all methods share the complete regular GT support;
native round-trip coverage is reported separately.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.stanford_pano import (
    StanfordPanorama,
    build_regular_pano_lookup,
    discover_stanford_panoramas,
    sample_pano_range_to_regular_z,
)
from bim_priorda3.engine import seed_everything

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.analysis import evaluate_stanford_pano_single_plus_tangent as strict_experiment
from scripts.model import evaluate_stanford_pano as evaluator

PROTOCOL = "stanford-area1-val-regular-erp-roundtrip-v1"
SCHEMA_VERSION = 1
SPLIT = "val"
DEPTH_METHOD = "raw_da3"
PANO_HEIGHT = 512
PANO_WIDTH = 1024
SEED = 42
BOOTSTRAP_REPETITIONS = 10_000
EXPECTED_FRAME_COUNT = 1_673
EXPECTED_PAIRED_STATIONS = 30
EXPECTED_ROOM_COUNT = 7
SOURCE_SETS = (
    "regular_only",
    "regular_plus_tangent6",
    "regular_plus_tangent14",
)
FUSION_METHODS = evaluator.COMBINED_FUSION_METHODS


@dataclass(frozen=True)
class MethodSpec:
    name: str
    source_set: str
    fusion_method: str


JOINT_METHODS = tuple(
    MethodSpec(
        name=f"{source_set}__{fusion_method}",
        source_set=source_set,
        fusion_method=fusion_method,
    )
    for source_set in SOURCE_SETS
    for fusion_method in FUSION_METHODS
)
METHOD_NAMES = ("raw_da3", "single_projection_roundtrip", *(item.name for item in JOINT_METHODS))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate regular predictions after same-centre ERP fusion and round-trip "
            "reprojection. This exploratory protocol is validation-only and exposes no "
            "checkpoint, BIM, test, or fusion-tuning option."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stanford_area1.yaml"))
    parser.add_argument(
        "--tangent-manifest",
        type=Path,
        required=True,
        help="Full Area_1 validation nested14 tangent-cache manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stanford_area1/pano_val_regular_roundtrip"),
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        help="Ordered-prefix smoke test. Omit for the exhaustive formal validation matrix.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_stations is not None and args.max_stations <= 0:
        parser.error("--max-stations must be positive")
    return args


def _fusion_args() -> argparse.Namespace:
    return argparse.Namespace(
        pano_height=PANO_HEIGHT,
        **strict_experiment.FUSION_PARAMETERS,
    )


def _identity_erp_prediction(
    view: evaluator.ProjectedView,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    view.validate([DEPTH_METHOD], int(np.prod(target_shape)))
    flat = np.zeros(int(np.prod(target_shape)), dtype=np.float32)
    valid = np.zeros(flat.shape, dtype=bool)
    flat[view.indices] = np.exp(view.log_ranges[DEPTH_METHOD]).astype(np.float32)
    valid[view.indices] = True
    return flat.reshape(target_shape), valid.reshape(target_shape)


def _roundtrip_with_raw_fallback(
    pano_range: np.ndarray,
    pano_valid: np.ndarray,
    lookup: Any,
    raw_depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    roundtrip, native_valid = sample_pano_range_to_regular_z(
        pano_range,
        pano_valid,
        lookup,
    )
    if roundtrip.shape != raw_depth.shape or native_valid.shape != raw_depth.shape:
        raise ValueError("Round-trip and raw regular depth shapes differ")
    if np.any(native_valid & (~np.isfinite(roundtrip) | (roundtrip <= 0.0))):
        raise RuntimeError("Round-trip prediction is invalid inside native coverage")
    output = raw_depth.astype(np.float32, copy=True)
    output[native_valid] = roundtrip[native_valid]
    if not np.isfinite(output).all() or np.any(output <= 0.0):
        raise RuntimeError("Round-trip system prediction must be finite and positive")
    return output, native_valid


def _read_regular_gt(
    record: Mapping[str, Any], expected_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(Path(str(record["sample"])), allow_pickle=False) as item:
        if "gt_depth" not in item.files or "gt_valid" not in item.files:
            raise RuntimeError(f"{record['id']}: prepared sample lacks regular GT")
        depth = item["gt_depth"].astype(np.float32)
        valid = item["gt_valid"].astype(bool)
    if depth.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError(f"{record['id']}: regular GT shape differs from prediction")
    invalid_support = valid & (~np.isfinite(depth) | (depth <= 0.0))
    if np.any(invalid_support):
        raise RuntimeError(f"{record['id']}: regular GT is invalid inside gt_valid")
    if not np.any(valid):
        raise RuntimeError(f"{record['id']}: regular GT support is empty")
    return depth, valid


def _metric_row(
    *,
    record: Mapping[str, Any],
    method: str,
    source_set: str,
    fusion_method: str,
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    native_valid: np.ndarray,
) -> dict[str, Any]:
    if prediction.shape != target.shape or support.shape != target.shape:
        raise ValueError(f"{record['id']}/{method}: regular evaluation shapes differ")
    if np.any(support & (~np.isfinite(prediction) | (prediction <= 0.0))):
        raise RuntimeError(f"{record['id']}/{method}: invalid prediction on fixed support")
    metrics = evaluator._metrics_for_array(prediction, target, support)
    fixed_count = int(metrics["count"])
    native_count = int(np.count_nonzero(support & native_valid))
    return {
        "sample_id": str(record["id"]),
        "room": str(record["region"]),
        "station_id": str(record["camera_uuid"]),
        "frame_number": int(record["frame_number"]),
        "method": method,
        "source_set": source_set,
        "fusion_method": fusion_method,
        "depth_method": DEPTH_METHOD,
        "fixed_support_pixels": fixed_count,
        "native_roundtrip_pixels": native_count,
        "native_roundtrip_coverage_fraction": native_count / fixed_count,
        "fallback_pixels": fixed_count - native_count,
        **metrics,
    }


def _pixel_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise RuntimeError("Cannot aggregate an empty metric row set")
    count = sum(int(row["count"]) for row in rows)
    if count <= 0:
        raise RuntimeError("Aggregate support is empty")
    weighted = lambda key: sum(float(row[key]) * int(row["count"]) for row in rows) / count
    return {
        "abs_rel": weighted("abs_rel"),
        "mae": weighted("mae"),
        "rmse": math.sqrt(sum(float(row["rmse"]) ** 2 * int(row["count"]) for row in rows) / count),
        "delta1": weighted("delta1"),
        "delta2": weighted("delta2"),
        "delta3": weighted("delta3"),
        "count": count,
    }


def _macro_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    if not rows:
        raise RuntimeError("Cannot macro-average an empty metric row set")
    return {
        **{
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in evaluator.METRIC_NAMES
        },
        "count": len(rows),
    }


def _group_rows(
    frame_rows: Sequence[Mapping[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        grouped[(str(row[group_key]), str(row["method"]))].append(row)
    output = []
    for (group, method), rows in sorted(grouped.items()):
        pixel = _pixel_aggregate(rows)
        output.append(
            {
                group_key: group,
                "room": str(rows[0]["room"]),
                "method": method,
                "source_set": str(rows[0]["source_set"]),
                "fusion_method": str(rows[0]["fusion_method"]),
                "frame_count": len(rows),
                "fixed_support_pixels": int(pixel["count"]),
                "mean_native_roundtrip_coverage_fraction": float(
                    np.mean([float(row["native_roundtrip_coverage_fraction"]) for row in rows])
                ),
                **{name: pixel[name] for name in evaluator.METRIC_NAMES},
            }
        )
    return output


def _method_summary(
    frame_rows: Sequence[Mapping[str, Any]],
    station_rows: Sequence[Mapping[str, Any]],
    room_rows: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    frames = [row for row in frame_rows if row["method"] == method]
    stations = [row for row in station_rows if row["method"] == method]
    rooms = [row for row in room_rows if row["method"] == method]
    return {
        "pixel_micro": _pixel_aggregate(frames),
        "frame_macro": _macro_aggregate(frames),
        "station_macro": _macro_aggregate(stations),
        "room_macro": _macro_aggregate(rooms),
        "native_roundtrip_coverage": {
            "mean_frame_fraction": float(
                np.mean([float(row["native_roundtrip_coverage_fraction"]) for row in frames])
            ),
            "minimum_frame_fraction": float(
                np.min([float(row["native_roundtrip_coverage_fraction"]) for row in frames])
            ),
            "total_fallback_pixels": sum(int(row["fallback_pixels"]) for row in frames),
        },
    }


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
            raise RuntimeError(f"{room}: incomplete method matrix for paired bootstrap")

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
    candidate_metric = float(
        _pixel_aggregate([r for r in rows if r["method"] == candidate])["abs_rel"]
    )
    reference_metric = float(
        _pixel_aggregate([r for r in rows if r["method"] == reference])["abs_rel"]
    )
    lower, upper = np.quantile(differences, [0.025, 0.975])
    room_differences = []
    for room, cand, ref in zip(room_ids, candidate_stats, reference_stats):
        room_differences.append((room, cand[0] / cand[1] - ref[0] / ref[1]))
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": "regular-frame fixed-support pixel-micro AbsRel",
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
        "resampling_unit": "room; all stations and regular frames retained within sampled room",
        "room_count": len(room_ids),
        "candidate_better_room_count": sum(value < 0.0 for _, value in room_differences),
        "room_differences": [
            {"room": room, "candidate_minus_reference_abs_rel": value}
            for room, value in room_differences
        ],
    }


def _evaluate_station(
    station: StanfordPanorama,
    records: Sequence[Mapping[str, Any]],
    tangent_bundle: evaluator.TangentManifestBundle,
    fusion_args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_records = sorted(
        records,
        key=lambda value: (int(value["frame_number"]), str(value["id"])),
    )
    frames = [strict_experiment._read_raw_regular_frame(record) for record in ordered_records]
    frame_by_id = {frame.sample_id: frame for frame in frames}
    if len(frame_by_id) != len(frames):
        raise RuntimeError(f"{station.camera_uuid}: duplicate regular frame ID")
    target_shape = (PANO_HEIGHT, PANO_WIDTH)
    pano_pixels = evaluator._erp_pixel_grid(*target_shape)
    # The selected fusion methods do not consume photo_weights.  A zero image
    # avoids opening pano RGB while retaining the shared projection implementation.
    pano_rgb_placeholder = np.zeros((*target_shape, 3), dtype=np.float32)
    projected_views = [
        evaluator._project_regular_view(
            frame,
            station,
            pano_pixels,
            pano_rgb_placeholder,
            [DEPTH_METHOD],
            centrality_power=float(fusion_args.centrality_power),
            confidence_floor=float(fusion_args.confidence_floor),
            photo_sigma=float(fusion_args.photo_sigma),
        )
        for frame in frames
    ]
    view_by_id = {view.frame_id: view for view in projected_views}
    tangent = evaluator._prepare_tangent_predictions(tangent_bundle, station, fusion_args)
    fusions = evaluator._prepare_regular_pano_joint(
        projected_views,
        tangent,
        target_shape,
        fusion_args,
    )
    # Prediction freeze boundary: no regular GT array has been opened above.
    rows = []
    station_native_coverages: defaultdict[str, list[float]] = defaultdict(list)
    for record in ordered_records:
        sample_id = str(record["id"])
        frame = frame_by_id[sample_id]
        view = view_by_id[sample_id]
        raw_depth = frame.predictions[DEPTH_METHOD]
        lookup = build_regular_pano_lookup(
            target_shape,
            station.camera_to_area,
            np.linalg.inv(frame.camera_to_area),
            frame.intrinsic,
            raw_depth.shape,
        )
        identity_range, identity_valid_erp = _identity_erp_prediction(view, target_shape)
        predictions: dict[str, tuple[np.ndarray, np.ndarray, str, str]] = {
            "raw_da3": (
                raw_depth,
                np.ones(raw_depth.shape, dtype=bool),
                "single_regular",
                "identity_no_roundtrip",
            ),
            "single_projection_roundtrip": (
                *_roundtrip_with_raw_fallback(
                    identity_range,
                    identity_valid_erp,
                    lookup,
                    raw_depth,
                ),
                "single_regular",
                "projection_roundtrip_control",
            ),
        }
        for spec in JOINT_METHODS:
            fusion = fusions[spec.source_set]
            pano_prediction = fusion.predictions[spec.fusion_method][DEPTH_METHOD]
            pano_valid = (
                (fusion.contributor_count > 0)
                & np.isfinite(pano_prediction)
                & (pano_prediction > 0.0)
            )
            system, native_valid = _roundtrip_with_raw_fallback(
                pano_prediction,
                pano_valid,
                lookup,
                raw_depth,
            )
            predictions[spec.name] = (
                system,
                native_valid,
                spec.source_set,
                spec.fusion_method,
            )
        if tuple(predictions) != METHOD_NAMES:
            raise AssertionError("Round-trip method order differs from protocol")

        # Official prepared regular GT is opened only after the complete station
        # prediction matrix has been frozen.
        gt_depth, gt_valid = _read_regular_gt(record, raw_depth.shape)
        for method, (prediction, native_valid, source_set, fusion_method) in predictions.items():
            row = _metric_row(
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
            station_native_coverages[method].append(
                float(row["native_roundtrip_coverage_fraction"])
            )
    expected_rows = len(ordered_records) * len(METHOD_NAMES)
    if len(rows) != expected_rows:
        raise RuntimeError(f"{station.camera_uuid}: expected {expected_rows} rows, got {len(rows)}")
    return rows, {
        "station_id": station.camera_uuid,
        "room": station.room,
        "regular_frame_count": len(ordered_records),
        "regular_frame_ids": [str(record["id"]) for record in ordered_records],
        "regular_projected_view_count": len(projected_views),
        "tangent_view_count": len(tangent.projected_views),
        "prediction_frozen_before_regular_gt_open": True,
        "mean_native_roundtrip_coverage": {
            method: float(np.mean(values))
            for method, values in sorted(station_native_coverages.items())
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(SEED)
    cfg = load_config(args.config)
    if float(cfg.data.min_depth) != 0.2 or float(cfg.data.max_depth) != 5.0:
        raise ValueError("Protocol requires the frozen 0.2–5.0 m regular depth range")
    dataset = BIMDepthDataset(cfg, SPLIT, augment=False, require_ground_truth=False)
    records_by_station: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in dataset.records:
        records_by_station[str(record["camera_uuid"])].append(record)
    if len(dataset.records) != EXPECTED_FRAME_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_FRAME_COUNT} regular validation frames, got {len(dataset.records)}"
        )
    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    all_stations = discover_stanford_panoramas(area_root)
    station_by_id = {station.camera_uuid: station for station in all_stations}
    selected_rooms = {str(record["region"]) for record in dataset.records}
    split_panos = [station for station in all_stations if station.room in selected_rooms]
    tangent_bundle = evaluator._validate_tangent_manifest(
        args.tangent_manifest,
        cfg=cfg,
        dataset=dataset,
        split=SPLIT,
        confirm_test=False,
        split_panoramas=split_panos,
    )
    station_ids = [
        camera_uuid for camera_uuid in tangent_bundle.stations if camera_uuid in records_by_station
    ]
    if len(station_ids) != EXPECTED_PAIRED_STATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_STATIONS} paired val stations, got {len(station_ids)}"
        )
    if args.max_stations is not None:
        station_ids = station_ids[: int(args.max_stations)]
    stations = [station_by_id[camera_uuid] for camera_uuid in station_ids]
    output_dir = args.output.expanduser().resolve()
    strict_experiment._ensure_new_output(output_dir)
    fusion_args = _fusion_args()
    frame_rows = []
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
            tangent_bundle,
            fusion_args,
        )
        frame_rows.extend(rows)
        station_receipts.append(receipt)

    expected_frames = sum(len(records_by_station[station.camera_uuid]) for station in stations)
    if len(frame_rows) != expected_frames * len(METHOD_NAMES):
        raise RuntimeError("Round-trip frame/method matrix is incomplete")
    for sample_id in sorted({str(row["sample_id"]) for row in frame_rows}):
        methods = {str(row["method"]) for row in frame_rows if row["sample_id"] == sample_id}
        if methods != set(METHOD_NAMES):
            raise RuntimeError(f"{sample_id}: incomplete round-trip methods")
        counts = {int(row["count"]) for row in frame_rows if row["sample_id"] == sample_id}
        if len(counts) != 1:
            raise RuntimeError(f"{sample_id}: fixed support differs across methods")

    station_rows = _group_rows(frame_rows, "station_id")
    room_rows = _group_rows(frame_rows, "room")
    metrics = {
        method: _method_summary(frame_rows, station_rows, room_rows, method)
        for method in METHOD_NAMES
    }
    contrasts = []
    for fusion_method in FUSION_METHODS:
        regular = f"regular_only__{fusion_method}"
        tangent6 = f"regular_plus_tangent6__{fusion_method}"
        tangent14 = f"regular_plus_tangent14__{fusion_method}"
        contrasts.extend(
            [
                _room_cluster_bootstrap(frame_rows, candidate=regular, reference="raw_da3"),
                _room_cluster_bootstrap(frame_rows, candidate=tangent6, reference=regular),
                _room_cluster_bootstrap(frame_rows, candidate=tangent14, reference=regular),
                _room_cluster_bootstrap(frame_rows, candidate=tangent14, reference=tangent6),
            ]
        )
    eligible_pano_methods = [
        spec.name for spec in JOINT_METHODS if spec.source_set == "regular_plus_tangent14"
    ]
    selected_method = min(
        eligible_pano_methods,
        key=lambda method: (float(metrics[method]["pixel_micro"]["abs_rel"]), method),
    )
    selected_fusion = next(
        spec.fusion_method for spec in JOINT_METHODS if spec.name == selected_method
    )
    selected_regular_reference = f"regular_only__{selected_fusion}"
    primary_contrast = _room_cluster_bootstrap(
        frame_rows,
        candidate=selected_method,
        reference=selected_regular_reference,
    )

    per_frame_path = output_dir / "per_frame.csv"
    per_station_path = output_dir / "per_station.csv"
    per_room_path = output_dir / "per_room.csv"
    strict_experiment._write_csv(per_frame_path, frame_rows)
    strict_experiment._write_csv(per_station_path, station_rows)
    strict_experiment._write_csv(per_room_path, room_rows)
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
            "BIM": False,
            "pano_GT": False,
            "regular_GT_opened_after_station_prediction_freeze_only": True,
        },
        "script": {
            "path": strict_experiment._repo_relative(Path(__file__), project_root),
            "sha256": strict_experiment._sha256(Path(__file__)),
        },
        "shared_code": {
            name: {
                "path": strict_experiment._repo_relative(path, project_root),
                "sha256": strict_experiment._sha256(path),
            }
            for name, path in {
                "pano_evaluator": Path(evaluator.__file__).resolve(),
                "stanford_pano_geometry": project_root / "src/bim_priorda3/data/stanford_pano.py",
                "pano_tangent_geometry": project_root / "src/bim_priorda3/data/pano_tangent.py",
            }.items()
        },
        "config": {
            "path": strict_experiment._repo_relative(config_path, project_root),
            "sha256": strict_experiment._sha256(config_path),
        },
        "dataset_split_provenance": strict_experiment._portable_split_provenance(
            dataset.split_provenance,
            project_root,
        ),
        "split_annotation": {
            "path": strict_experiment._repo_relative(annotation_path, project_root),
            "sha256": strict_experiment._sha256(annotation_path),
        },
        "prepared_manifest": {
            "path": strict_experiment._repo_relative(prepared_manifest, project_root),
            "sha256": strict_experiment._sha256(prepared_manifest),
        },
        "tangent_manifest": {
            "path": strict_experiment._repo_relative(tangent_bundle.path, project_root),
            "sha256": tangent_bundle.sha256,
        },
        "parameters": {
            "pano_shape": [PANO_HEIGHT, PANO_WIDTH],
            "regular_shape": [504, 504],
            "depth_range_m": [float(cfg.data.min_depth), float(cfg.data.max_depth)],
            "fusion_parameters": strict_experiment.FUSION_PARAMETERS,
            "fusion_methods": list(FUSION_METHODS),
            "source_sets": list(SOURCE_SETS),
            "missing_roundtrip_policy": "raw regular fallback; native coverage reported",
            "selection_objective": (
                "lowest validation fixed-support regular-frame pixel-micro AbsRel among "
                "regular_plus_tangent14 fusion variants"
            ),
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
                "path": strict_experiment._repo_relative(per_frame_path, project_root),
                "sha256": strict_experiment._sha256(per_frame_path),
            },
            "per_station_csv": {
                "path": strict_experiment._repo_relative(per_station_path, project_root),
                "sha256": strict_experiment._sha256(per_station_path),
            },
            "per_room_csv": {
                "path": strict_experiment._repo_relative(per_room_path, project_root),
                "sha256": strict_experiment._sha256(per_room_path),
            },
        },
    }
    strict_experiment._assert_no_private_absolute_paths(provenance)
    provenance_path = output_dir / "provenance.json"
    strict_experiment._write_json(provenance_path, provenance)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "publication_status": "candidate_for_future_preregistration",
        "split": SPLIT,
        "test_data_or_csv_accessed": False,
        "claim": (
            "All same-station regular DA3 predictions are fused on ERP, optionally with "
            "pano tangents, then reprojected and evaluated on every original regular GT pixel."
        ),
        "population": {
            "frame_count": expected_frames,
            "station_count": len(stations),
            "room_count": len({station.room for station in stations}),
        },
        "fixed_support": (
            "each original regular frame's complete prepared 0.2–5.0 m gt_valid mask; "
            "same pixels for every method; missing round-trip samples use raw fallback"
        ),
        "methods": metrics,
        "all_predeclared_contrasts": contrasts,
        "validation_selected_pano_candidate": {
            "method": selected_method,
            "fusion_method": selected_fusion,
            "same_fusion_regular_reference": selected_regular_reference,
            "selection_objective": "minimum pixel-micro AbsRel",
            "primary_contrast": primary_contrast,
        },
        "identity_roundtrip_control": _room_cluster_bootstrap(
            frame_rows,
            candidate="single_projection_roundtrip",
            reference="raw_da3",
        ),
        "method_scope": provenance["method_scope"],
        "artifacts": {
            **provenance["artifacts"],
            "provenance": {
                "path": strict_experiment._repo_relative(provenance_path, project_root),
                "sha256": strict_experiment._sha256(provenance_path),
            },
        },
    }
    strict_experiment._assert_no_private_absolute_paths(summary)
    summary_path = output_dir / "summary.json"
    strict_experiment._write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
