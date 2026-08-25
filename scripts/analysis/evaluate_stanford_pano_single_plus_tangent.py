#!/usr/bin/env python3
"""Exploratory val-only strict-single plus panorama-tangent evaluation.

This independent protocol answers one narrow question without changing the
frozen v3 evaluator or its formal artifacts: how much does training-free pano
fusion change the result when the regular-image input is restricted to exactly
one whole frame?

The selector, spherical projection, tangent-cache validation, tangent-to-ERP
projection, robust fusion, exact solid-angle weights, grouped summaries, and
room-cluster bootstrap are all delegated to the existing evaluator.  This file
contains orchestration and result serialization only; it intentionally has no
test-split option, checkpoint option, BIM branch, learned branch, GT scale, or
configurable fusion hyperparameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from bim_priorda3.data.stanford_pano import StanfordPanorama, discover_stanford_panoramas
from bim_priorda3.engine import seed_everything

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model import evaluate_stanford_pano as evaluator

PROTOCOL = "stanford-area1-val-strict-single-plus-pano-tangent-v4-candidate-v1"
SCHEMA_VERSION = 1
SPLIT = "val"
FUSION_METHOD = "joint_huber"
DEPTH_METHOD = "raw_da3"
PANO_HEIGHT = 512
PANO_WIDTH = 1024
SEED = 42
BOOTSTRAP_REPETITIONS = 10_000
EXPECTED_PAIRED_STATIONS = 30
EXPECTED_PANO_ONLY_STATIONS = 2
METHODS = (
    ("strict_single", 0),
    ("strict_single_plus_tangent6", 6),
    ("strict_single_plus_tangent14", 14),
)

# Frozen v3 joint-Huber parameters.  They are constants rather than CLI flags,
# preventing validation-driven tuning in this exploratory comparison.
FUSION_PARAMETERS = {
    "centrality_power": 4.0,
    "confidence_floor": 0.05,
    "huber_log_delta": 0.08,
    "consistency_log_threshold": 0.25,
    "photo_sigma": 0.12,
    "sync_min_overlap": 256,
    "sync_pair_max_samples": 4096,
    "sync_huber_log_delta": 0.08,
    "sync_l2": 1e-6,
    "sync_max_abs_offset": 0.50,
}


@dataclass(frozen=True)
class FrozenMethod:
    """One prediction frozen before the official pano target is opened."""

    name: str
    tangent_view_count: int
    prediction: np.ndarray
    contributor_count: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the validation-only strict-single + tangent6/tangent14 raw-DA3 "
            "comparison. Fusion parameters are frozen and no test/checkpoint/BIM "
            "option exists."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stanford_area1.yaml"))
    parser.add_argument(
        "--tangent-manifest",
        type=Path,
        required=True,
        help="Formal full-validation nested14 tangent-cache manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/stanford_area1/pano_val_single_plus_tangent"),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path, project_root: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Public result path lies outside the repository: {path}") from error


def _area_relative(path: Path, area_root: Path) -> str:
    try:
        return path.expanduser().resolve().relative_to(area_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Dataset file lies outside configured Area root: {path}") from error


def _portable_split_provenance(
    payload: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    portable = json.loads(json.dumps(payload))
    annotation = portable.get("annotation_file")
    if not isinstance(annotation, str):
        raise TypeError("Dataset split provenance lacks annotation_file")
    portable["annotation_file"] = _repo_relative(Path(annotation), project_root)
    return portable


def _assert_no_private_absolute_paths(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if "/home/" in encoded or "/Users/" in encoded:
        raise RuntimeError("Public result JSON contains a private absolute home path")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty result table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fusion_args() -> argparse.Namespace:
    return argparse.Namespace(pano_height=PANO_HEIGHT, **FUSION_PARAMETERS)


def _read_raw_regular_frame(
    record: Mapping[str, Any],
) -> evaluator.RegularFrame:
    """Load only cached DA3/camera arrays; BIM and regular GT stay unopened."""

    sample_path = Path(str(record["sample"]))
    if not sample_path.is_file():
        raise FileNotFoundError(f"Prepared Stanford sample is missing: {sample_path}")
    with np.load(sample_path, allow_pickle=False) as item:
        required = {"base_depth", "base_confidence", "intrinsic", "camera_to_area"}
        missing = sorted(required - set(item.files))
        if missing:
            raise RuntimeError(f"{record['id']}: prepared sample lacks {missing}")
        depth = item["base_depth"].astype(np.float32)
        confidence = item["base_confidence"].astype(np.float32)
        intrinsic = item["intrinsic"].astype(np.float64)
        camera_to_area = item["camera_to_area"].astype(np.float64)
    if depth.ndim != 2 or confidence.shape != depth.shape:
        raise ValueError(f"{record['id']}: cached DA3 depth/confidence shapes differ")
    if intrinsic.shape != (3, 3) or camera_to_area.shape != (4, 4):
        raise ValueError(f"{record['id']}: cached camera matrix shape is invalid")
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise RuntimeError(f"{record['id']}: cached DA3 depth must be finite and positive")
    if not np.isfinite(confidence).all():
        raise RuntimeError(f"{record['id']}: cached DA3 confidence is non-finite")
    # RGB is required by the shared projection API only to construct a
    # photometric weight.  Joint Huber and the strict selector never consume
    # that weight, so a deterministic zero placeholder avoids opening regular
    # RGB while preserving the exact shared geometry and base weights.
    rgb_placeholder = np.zeros((*depth.shape, 3), dtype=np.float32)
    return evaluator.RegularFrame(
        sample_id=str(record["id"]),
        room=str(record["region"]),
        camera_uuid=str(record["camera_uuid"]),
        intrinsic=intrinsic,
        camera_to_area=camera_to_area,
        rgb=rgb_placeholder,
        base_confidence=np.clip(confidence, 0.0, 1.0),
        predictions={DEPTH_METHOD: depth},
        model_arrays={},
    )


def _prediction_from_view(
    view: evaluator.ProjectedView,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    view.validate([DEPTH_METHOD], int(np.prod(target_shape)))
    flat = np.full(int(np.prod(target_shape)), np.nan, dtype=np.float32)
    flat[view.indices] = np.exp(view.log_ranges[DEPTH_METHOD]).astype(np.float32)
    coverage = np.zeros(flat.shape, dtype=np.int32)
    coverage[view.indices] = 1
    return flat.reshape(target_shape), coverage.reshape(target_shape)


def freeze_method_predictions(
    selected_view: evaluator.ProjectedView,
    tangent_views: Sequence[evaluator.ProjectedView],
    target_shape: tuple[int, int],
    fusion_args: argparse.Namespace,
) -> tuple[FrozenMethod, ...]:
    """Construct all three GT-free predictions via the shared fusion routine."""

    if len(tangent_views) != 14:
        raise ValueError(f"Expected 14 validated tangent views, got {len(tangent_views)}")
    selected_raw = evaluator._raw_only_view(selected_view)
    strict_prediction, strict_count = _prediction_from_view(selected_raw, target_shape)
    outputs = [
        FrozenMethod(
            name="strict_single",
            tangent_view_count=0,
            prediction=strict_prediction,
            contributor_count=strict_count,
        )
    ]
    for name, view_count in METHODS[1:]:
        fusion = evaluator._aggregate_with_args(
            (selected_raw, *tangent_views[:view_count]),
            [DEPTH_METHOD],
            target_shape,
            fusion_args,
            # The synchronized branch is computed by the shared routine but is
            # not evaluated here.  Anchor it to the sole regular input anyway.
            sync_gauge_view_count=1,
        )
        prediction = fusion.predictions[FUSION_METHOD][DEPTH_METHOD]
        outputs.append(
            FrozenMethod(
                name=name,
                tangent_view_count=view_count,
                prediction=prediction,
                contributor_count=fusion.contributor_count,
            )
        )
    if tuple(item.name for item in outputs) != tuple(item[0] for item in METHODS):
        raise AssertionError("Frozen method order differs from the exploratory protocol")
    return tuple(outputs)


def _coverage_mask(
    method: FrozenMethod,
    gt_valid: np.ndarray,
    *,
    station_id: str,
) -> np.ndarray:
    if (
        method.prediction.shape != gt_valid.shape
        or method.contributor_count.shape != gt_valid.shape
    ):
        raise ValueError(f"{station_id}/{method.name}: ERP array shapes differ")
    covered = method.contributor_count > 0
    invalid = covered & (~np.isfinite(method.prediction) | (method.prediction <= 0))
    if np.any(invalid):
        raise RuntimeError(
            f"{station_id}/{method.name}: {int(np.count_nonzero(invalid))} invalid "
            "predictions occur inside native coverage"
        )
    return gt_valid & covered


def evaluate_frozen_methods(
    station: StanfordPanorama,
    selected_view: evaluator.ProjectedView,
    selector_score: float,
    frozen: Sequence[FrozenMethod],
    gt_depth: np.ndarray,
    gt_valid: np.ndarray,
    *,
    regular_view_count: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Evaluate all methods on one strict common support and native coverage."""

    if tuple(item.name for item in frozen) != tuple(item[0] for item in METHODS):
        raise ValueError("Frozen method set differs from the exploratory protocol")
    native_supports = {
        method.name: _coverage_mask(method, gt_valid, station_id=station.camera_uuid)
        for method in frozen
    }
    strict_support = native_supports["strict_single"]
    if not np.any(strict_support):
        raise RuntimeError(f"{station.camera_uuid}: selected strict frame has no valid GT support")
    # Every combined method contains the selected regular view.  Therefore all
    # of them must be valid on its complete GT-valid coverage; fail instead of
    # silently shrinking the common support.
    for method in frozen:
        missing = strict_support & ~native_supports[method.name]
        if np.any(missing):
            raise RuntimeError(
                f"{station.camera_uuid}/{method.name}: missing "
                f"{int(np.count_nonzero(missing))} selected-frame support pixels"
            )
    common_support = strict_support
    area_weights = evaluator._latitude_area_weights(*gt_depth.shape)
    gt_count = int(np.count_nonzero(gt_valid))
    fixed_count = int(np.count_nonzero(common_support))
    if gt_count <= 0 or fixed_count <= 0:
        raise RuntimeError(f"{station.camera_uuid}: empty GT or fixed support")
    rows = []
    arrays = {}
    for method in frozen:
        native_support = native_supports[method.name]
        pixel_metrics = evaluator._metrics_for_array(
            method.prediction,
            gt_depth,
            common_support,
        )
        spherical_metrics = evaluator._metrics_for_array(
            method.prediction,
            gt_depth,
            common_support,
            area_weights,
        )
        rows.append(
            {
                "station_id": station.camera_uuid,
                "room": station.room,
                "regular_view_count_before_selection": int(regular_view_count),
                "selected_frame_id": selected_view.frame_id,
                "selector_score_sum_base_weights": float(selector_score),
                "method": method.name,
                "fusion_method": (
                    "identity_single_frame" if method.name == "strict_single" else FUSION_METHOD
                ),
                "depth_method": DEPTH_METHOD,
                "selected_regular_view_count": 1,
                "tangent_view_count": method.tangent_view_count,
                "fixed_support_pixels": fixed_count,
                "gt_valid_pixels_at_eval_resolution": gt_count,
                "common_strict_coverage_fraction": fixed_count / gt_count,
                "common_strict_solid_angle_coverage_fraction": float(
                    area_weights[common_support].sum() / area_weights[gt_valid].sum()
                ),
                "native_union_pixels": int(np.count_nonzero(native_support)),
                "native_union_coverage_fraction": float(np.mean(native_support[gt_valid])),
                "native_union_solid_angle_coverage_fraction": float(
                    area_weights[native_support].sum() / area_weights[gt_valid].sum()
                ),
                "mean_contributors_on_native_union": float(
                    np.mean(method.contributor_count[native_support])
                ),
                **pixel_metrics,
                **{
                    f"spherical_{name}": value
                    for name, value in spherical_metrics.items()
                    if name != "count"
                },
            }
        )
        arrays[method.name] = (method.prediction, gt_depth, common_support)
    counts = {int(row["fixed_support_pixels"]) for row in rows}
    if len(counts) != 1:
        raise RuntimeError(f"{station.camera_uuid}: fixed support counts differ across methods")
    return rows, arrays


def per_room_rows(station_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in station_rows:
        grouped[(str(row["room"]), str(row["method"]))].append(row)
    outputs = []
    for (room, method), rows in sorted(grouped.items()):
        station_ids = [str(row["station_id"]) for row in rows]
        if len(station_ids) != len(set(station_ids)):
            raise RuntimeError(f"Duplicate station in room aggregate: {room}/{method}")
        outputs.append(
            {
                "room": room,
                "method": method,
                "fusion_method": str(rows[0]["fusion_method"]),
                "depth_method": DEPTH_METHOD,
                "station_count": len(rows),
                "fixed_support_pixels": sum(int(row["fixed_support_pixels"]) for row in rows),
                "mean_native_union_solid_angle_coverage_fraction": float(
                    np.mean(
                        [float(row["native_union_solid_angle_coverage_fraction"]) for row in rows]
                    )
                ),
                **{
                    f"spherical_{name}": float(
                        np.mean([float(row[f"spherical_{name}"]) for row in rows])
                    )
                    for name in evaluator.METRIC_NAMES
                },
            }
        )
    return outputs


def _metric_summary(
    station_rows: Sequence[Mapping[str, Any]],
    room_rows: Sequence[Mapping[str, Any]],
    method: str,
    pixel_micro: evaluator.MetricTotals,
    spherical_micro: evaluator.MetricTotals,
) -> dict[str, Any]:
    selected = [row for row in station_rows if row["method"] == method]
    selected_rooms = [row for row in room_rows if row["method"] == method]
    if len(selected) != EXPECTED_PAIRED_STATIONS or not selected_rooms:
        raise RuntimeError(f"{method}: incomplete station/room summary")
    return {
        "station_macro": {
            **{
                name: float(np.mean([float(row[name]) for row in selected]))
                for name in evaluator.METRIC_NAMES
            },
            **{
                f"spherical_{name}": float(
                    np.mean([float(row[f"spherical_{name}"]) for row in selected])
                )
                for name in evaluator.METRIC_NAMES
            },
            "station_count": len(selected),
        },
        "room_macro": {
            **{
                f"spherical_{name}": float(
                    np.mean([float(row[f"spherical_{name}"]) for row in selected_rooms])
                )
                for name in evaluator.METRIC_NAMES
            },
            "room_count": len(selected_rooms),
        },
        "pixel_micro": pixel_micro.compute(),
        "spherical_pixel_micro": spherical_micro.compute(),
        "native_union_coverage": {
            "mean_pixel_fraction": float(
                np.mean([float(row["native_union_coverage_fraction"]) for row in selected])
            ),
            "mean_solid_angle_fraction": float(
                np.mean(
                    [float(row["native_union_solid_angle_coverage_fraction"]) for row in selected]
                )
            ),
        },
        "common_strict_support": {
            "fixed_pixels": sum(int(row["fixed_support_pixels"]) for row in selected),
            "mean_pixel_fraction": float(
                np.mean([float(row["common_strict_coverage_fraction"]) for row in selected])
            ),
            "mean_solid_angle_fraction": float(
                np.mean(
                    [float(row["common_strict_solid_angle_coverage_fraction"]) for row in selected]
                )
            ),
        },
    }


def _paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    candidate: str,
    reference: str,
) -> dict[str, Any]:
    return evaluator._route_paired_contrast(
        rows,
        kind="strict_single_plus_pano_tangent_over_strict_single",
        candidate={"method": candidate},
        reference={"method": reference},
        seed=SEED,
        repetitions=BOOTSTRAP_REPETITIONS,
    )


def _ensure_new_output(output_dir: Path) -> None:
    resolved = output_dir.expanduser().resolve()
    if resolved.name in {"pano_val", "pano_test"}:
        raise ValueError("This exploratory protocol cannot target a frozen v3 result directory")
    if resolved.exists() and any(resolved.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full 30-station validation-only experiment."""

    seed_everything(SEED)
    cfg = load_config(args.config)
    if float(cfg.data.min_depth) != 0.2 or float(cfg.data.max_depth) != 5.0:
        raise ValueError("Protocol requires the frozen 0.2–5.0 m evaluation range")
    dataset = BIMDepthDataset(cfg, SPLIT, augment=False, require_ground_truth=False)
    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    all_stations = discover_stanford_panoramas(area_root)
    station_by_id = {station.camera_uuid: station for station in all_stations}
    records_by_station: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in dataset.records:
        records_by_station[str(record["camera_uuid"])].append(record)
    selected_rooms = {str(record["region"]) for record in dataset.records}
    split_panos = [station for station in all_stations if station.room in selected_rooms]
    split_pano_ids = {station.camera_uuid for station in split_panos}
    missing_pano_ids = sorted(set(records_by_station) - split_pano_ids)
    pano_only_ids = sorted(split_pano_ids - set(records_by_station))
    if missing_pano_ids:
        raise RuntimeError(f"Validation annotation stations missing panorama: {missing_pano_ids}")
    if len(records_by_station) != EXPECTED_PAIRED_STATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_PAIRED_STATIONS} paired validation stations, "
            f"found {len(records_by_station)}"
        )
    if len(pano_only_ids) != EXPECTED_PANO_ONLY_STATIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_PANO_ONLY_STATIONS} val pano-only stations, "
            f"found {len(pano_only_ids)}"
        )
    tangent_bundle = evaluator._validate_tangent_manifest(
        args.tangent_manifest,
        cfg=cfg,
        dataset=dataset,
        split=SPLIT,
        confirm_test=False,
        split_panoramas=split_panos,
    )
    selected_stations = [
        station_by_id[camera_uuid]
        for camera_uuid in tangent_bundle.stations
        if camera_uuid in records_by_station
    ]
    if len(selected_stations) != EXPECTED_PAIRED_STATIONS:
        raise RuntimeError("Validated tangent manifest does not cover all paired val stations")

    output_dir = args.output.expanduser().resolve()
    _ensure_new_output(output_dir)
    target_shape = (PANO_HEIGHT, PANO_WIDTH)
    pano_pixels = evaluator._erp_pixel_grid(*target_shape)
    fusion_args = _fusion_args()
    all_rows: list[dict[str, Any]] = []
    station_receipts = []
    micro = {method: evaluator.MetricTotals() for method, _ in METHODS}
    spherical_micro = {method: evaluator.MetricTotals() for method, _ in METHODS}
    area_weights = evaluator._latitude_area_weights(*target_shape)

    for index, station in enumerate(selected_stations, start=1):
        records = sorted(
            records_by_station[station.camera_uuid],
            key=lambda value: (int(value["frame_number"]), str(value["id"])),
        )
        print(
            f"[{index}/{len(selected_stations)}] {station.room}/{station.camera_uuid} "
            f"({len(records)} regular candidates)",
            flush=True,
        )
        pano_rgb = evaluator._load_pano_rgb(station.rgb_path, target_shape)
        projected_views = [
            evaluator._project_regular_view(
                _read_raw_regular_frame(record),
                station,
                pano_pixels,
                pano_rgb,
                [DEPTH_METHOD],
                centrality_power=FUSION_PARAMETERS["centrality_power"],
                confidence_floor=FUSION_PARAMETERS["confidence_floor"],
                photo_sigma=FUSION_PARAMETERS["photo_sigma"],
            )
            for record in records
        ]
        selected_view, selector_score = evaluator._select_strict_single_view(projected_views)
        tangent_views = evaluator._load_tangent_projected_views(
            tangent_bundle,
            station.camera_uuid,
            target_shape,
            centrality_power=FUSION_PARAMETERS["centrality_power"],
        )
        frozen = freeze_method_predictions(
            selected_view,
            tangent_views,
            target_shape,
            fusion_args,
        )

        # Protocol boundary: official validation pano depth is opened only
        # after selection and all three predictions are immutable.
        gt_depth, gt_valid, native_valid_count = evaluator._load_pano_gt(
            station.depth_path,
            target_shape,
            min_depth=float(cfg.data.min_depth),
            max_depth=float(cfg.data.max_depth),
        )
        rows, arrays = evaluate_frozen_methods(
            station,
            selected_view,
            selector_score,
            frozen,
            gt_depth,
            gt_valid,
            regular_view_count=len(records),
        )
        all_rows.extend(rows)
        for method, (prediction, target, support) in arrays.items():
            micro[method].update(prediction, target, support)
            spherical_micro[method].update(prediction, target, support, area_weights)
        fixed_counts = {int(row["fixed_support_pixels"]) for row in rows}
        if len(fixed_counts) != 1:
            raise RuntimeError(f"{station.camera_uuid}: method fixed counts differ")
        station_receipts.append(
            {
                "station_id": station.camera_uuid,
                "room": station.room,
                "regular_candidate_count": len(records),
                "selected_frame_id": selected_view.frame_id,
                "selector_score_sum_base_weights": float(selector_score),
                "native_pano_valid_pixels_before_range_filter": native_valid_count,
                "gt_valid_pixels_at_eval_resolution": int(np.count_nonzero(gt_valid)),
                "fixed_support_pixels": next(iter(fixed_counts)),
            }
        )

    expected_rows = EXPECTED_PAIRED_STATIONS * len(METHODS)
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} station rows, got {len(all_rows)}")
    station_method_counts = defaultdict(set)
    for row in all_rows:
        station_method_counts[str(row["station_id"])].add(str(row["method"]))
    expected_methods = {name for name, _ in METHODS}
    if any(methods != expected_methods for methods in station_method_counts.values()):
        raise RuntimeError("At least one station lacks a complete three-method matrix")

    room_rows = per_room_rows(all_rows)
    metrics = {
        method: _metric_summary(
            all_rows,
            room_rows,
            method,
            micro[method],
            spherical_micro[method],
        )
        for method, _ in METHODS
    }
    contrasts = [
        _paired_contrast(all_rows, "strict_single_plus_tangent6", "strict_single"),
        _paired_contrast(all_rows, "strict_single_plus_tangent14", "strict_single"),
        _paired_contrast(
            all_rows,
            "strict_single_plus_tangent14",
            "strict_single_plus_tangent6",
        ),
    ]

    per_station_path = output_dir / "per_station.csv"
    per_room_path = output_dir / "per_room.csv"
    _write_csv(per_station_path, all_rows)
    _write_csv(per_room_path, room_rows)
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(str(cfg.config_path)).resolve()
    annotation_path = resolve_project_path(cfg, cfg.data.split_annotation)
    prepared_manifest = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
    depth_targets = {
        station.camera_uuid: {
            "area_root_relative_path": _area_relative(station.depth_path, area_root),
            "sha256": _sha256(station.depth_path),
        }
        for station in selected_stations
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "split": SPLIT,
        "status": "exploratory_validation_only",
        "publication_status": "candidate_for_future_preregistration",
        "formal_v3_artifacts_mutated": False,
        "test_access": {
            "authorized": False,
            "test_raw_files_opened": False,
            "test_csv_opened": False,
        },
        "seed": SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "script": {
            "path": _repo_relative(Path(__file__), project_root),
            "sha256": _sha256(Path(__file__)),
        },
        "upstream_evaluator": {
            "path": _repo_relative(Path(evaluator.__file__), project_root),
            "sha256": _sha256(Path(evaluator.__file__).resolve()),
            "reused_functions": [
                "_validate_tangent_manifest",
                "_erp_pixel_grid",
                "_project_regular_view",
                "_select_strict_single_view",
                "_load_tangent_projected_views",
                "_aggregate_with_args",
                "_load_pano_gt",
                "_latitude_area_weights",
                "_metrics_for_array",
                "_route_paired_contrast",
            ],
        },
        "shared_geometry": {
            name: {"path": _repo_relative(path, project_root), "sha256": _sha256(path)}
            for name, path in {
                "stanford_pano": project_root / "src/bim_priorda3/data/stanford_pano.py",
                "pano_tangent": project_root / "src/bim_priorda3/data/pano_tangent.py",
            }.items()
        },
        "config": {
            "path": _repo_relative(config_path, project_root),
            "sha256": _sha256(config_path),
        },
        "dataset_split_provenance": _portable_split_provenance(
            dataset.split_provenance, project_root
        ),
        "split_annotation": {
            "path": _repo_relative(annotation_path, project_root),
            "sha256": _sha256(annotation_path),
        },
        "prepared_manifest": {
            "path": _repo_relative(prepared_manifest, project_root),
            "sha256": _sha256(prepared_manifest),
        },
        "tangent_manifest": {
            "path": _repo_relative(tangent_bundle.path, project_root),
            "sha256": tangent_bundle.sha256,
            "protocol": evaluator.TANGENT_MANIFEST_PROTOCOL,
            "validation": "full val manifest verified by upstream evaluator",
            "view_count_contract": {
                "tangent6": "first six nested14 views, exactly cubemap6",
                "tangent14": "all nested14 views",
            },
        },
        "prediction_input_contract": {
            "cached_regular_raw_da3_depth": True,
            "cached_regular_da3_confidence": True,
            "cached_tangent_raw_da3_depth_and_confidence": True,
            "pano_rgb": "projection API only; photo weights are not consumed by joint Huber",
            "regular_rgb_opened": False,
            "regular_ground_truth_opened": False,
            "BIM_arrays_opened": False,
            "BIM_method_evaluated": False,
            "checkpoint_opened": False,
            "learned_method_evaluated": False,
            "GT_scale_or_view_weighting": False,
            "pano_depth_opened_after_prediction_freeze_only": True,
        },
        "method_protocol": {
            "selector": "existing max sum(base_weights), stable frame_id tie-break",
            "selector_inputs": "projection centrality and cached DA3 confidence only",
            "methods": [name for name, _ in METHODS],
            "regular_view_count_after_selection": 1,
            "fusion_method": FUSION_METHOD,
            "fusion_domain": "log radial range",
            "fusion_parameters": FUSION_PARAMETERS,
            "quality_support": (
                "selected-frame valid coverage intersected with 0.2–5.0 m pano GT; "
                "all three predictions required valid, fail-fast on any selected-support gap"
            ),
            "coverage_support": "each method native union intersected with valid pano GT",
            "primary_metric": "equal-station macro exact-ERP-solid-angle AbsRel",
            "room_macro": "station mean within room, then equal-room mean",
            "uncertainty": "10k room-cluster paired bootstrap, seed 42",
        },
        "station_selection": {
            "source": "exhaustive val annotation",
            "paired_station_count": len(selected_stations),
            "paired_station_ids": [station.camera_uuid for station in selected_stations],
            "pano_only_excluded_count": len(pano_only_ids),
            "pano_only_excluded_ids": pano_only_ids,
            "receipts": station_receipts,
        },
        "validation_pano_depth_targets": depth_targets,
        "per_station_csv": {
            "path": _repo_relative(per_station_path, project_root),
            "sha256": _sha256(per_station_path),
        },
        "per_room_csv": {
            "path": _repo_relative(per_room_path, project_root),
            "sha256": _sha256(per_room_path),
        },
    }
    _assert_no_private_absolute_paths(provenance)
    provenance_path = output_dir / "provenance.json"
    _write_json(provenance_path, provenance)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "split": SPLIT,
        "status": "exploratory_validation_only",
        "publication_status": "candidate_for_future_preregistration",
        "formal_v3_result": False,
        "formal_v3_artifacts_mutated": False,
        "test_data_or_csv_accessed": False,
        "claim": (
            "Raw DA3 from one GT-free selected whole regular frame versus that same frame "
            "jointly fused with 6 or 14 deterministic pano tangents."
        ),
        "method_scope": {
            "training_free_only": True,
            "depth_method": DEPTH_METHOD,
            "checkpoint": False,
            "learned": False,
            "BIM": False,
            "GT_scale": False,
            "fusion_method": FUSION_METHOD,
            "fusion_parameters_frozen": True,
        },
        "evaluation_protocol": {
            "depth_range_m": [0.2, 5.0],
            "pano_shape": [PANO_HEIGHT, PANO_WIDTH],
            "quality_support": "common strict selected-frame support",
            "coverage_support": "per-method native union",
            "pixel_weighting": "exact ERP pixel solid angle",
            "station_aggregation": "equal-station macro",
            "room_macro": "equal room after within-room station mean",
            "bootstrap": {
                "resampling_unit": "room cluster",
                "repetitions": BOOTSTRAP_REPETITIONS,
                "seed": SEED,
            },
        },
        "station_count": len(selected_stations),
        "room_count": len({station.room for station in selected_stations}),
        "pano_only_excluded_count": len(pano_only_ids),
        "pano_only_excluded_ids": pano_only_ids,
        "methods": [name for name, _ in METHODS],
        "metrics": metrics,
        "primary_spherical_abs_rel": {
            method: metrics[method]["station_macro"]["spherical_abs_rel"] for method, _ in METHODS
        },
        "native_union_mean_solid_angle_coverage": {
            method: metrics[method]["native_union_coverage"]["mean_solid_angle_fraction"]
            for method, _ in METHODS
        },
        "contrasts": contrasts,
        "fixed_support_audit": {
            "same_count_for_all_methods_per_station": True,
            "station_fixed_pixel_counts": {
                receipt["station_id"]: receipt["fixed_support_pixels"]
                for receipt in station_receipts
            },
        },
        "artifacts": {
            "per_station_csv": _repo_relative(per_station_path, project_root),
            "per_station_csv_sha256": _sha256(per_station_path),
            "per_room_csv": _repo_relative(per_room_path, project_root),
            "per_room_csv_sha256": _sha256(per_room_path),
            "provenance": _repo_relative(provenance_path, project_root),
            "provenance_sha256": _sha256(provenance_path),
        },
    }
    _assert_no_private_absolute_paths(summary)
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    print(f"Wrote {summary_path}", flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": summary["status"],
                "station_count": summary["station_count"],
                "primary_spherical_abs_rel": summary["primary_spherical_abs_rel"],
                "native_union_mean_solid_angle_coverage": summary[
                    "native_union_mean_solid_angle_coverage"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
