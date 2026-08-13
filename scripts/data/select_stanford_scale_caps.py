#!/usr/bin/env python3
"""Select robust BIM-scale caps using only annotated Stanford train rooms.

The protocol in this file is intentionally not configurable from the command
line.  In particular, there are no room, sample-count, or stride overrides:
the exhaustive split annotation is the sole population authority.  Ground
truth and semantic-derived masks are used only to score candidates on train;
the runtime scale estimator itself receives only DA3 and BIM depth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.baselines import (
    bim_scale_and_local_features,
    robust_scale_and_local_features,
)
from bim_priorda3.config import Config, load_config, resolve_project_path
from bim_priorda3.data.splits import ACTIVE_SPLITS, resolve_annotation_splits

C10_GRID = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, math.inf)
C25_GRID = (0.025, 0.05, 0.075, 0.10, 0.15, math.inf)
SUBSETS = ("all", "furniture", "non_structural")
RATIO_MIN = 0.2
RATIO_MAX = 5.0
MIN_SCALE_SAMPLES = 100
TIE_TOLERANCE = 1e-12


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_unchanged(path: Path, expected_sha256: str, *, label: str) -> None:
    actual = _file_sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} changed during registered scale selection: "
            f"start={expected_sha256}, current={actual}"
        )


def _json_cap(value: float) -> float | str:
    return "inf" if math.isinf(value) else float(value)


def _cap_label(value: float) -> str:
    return "inf" if math.isinf(value) else format(value, ".12g")


@dataclass(frozen=True)
class Candidate:
    q10_log_cap: float
    q25_log_cap: float

    @property
    def candidate_id(self) -> str:
        return f"q10={_cap_label(self.q10_log_cap)},q25={_cap_label(self.q25_log_cap)}"

    def as_json(self) -> dict[str, float | str]:
        return {
            "candidate_id": self.candidate_id,
            "q10_log_cap": _json_cap(self.q10_log_cap),
            "q25_log_cap": _json_cap(self.q25_log_cap),
        }


CANDIDATES = tuple(Candidate(c10, c25) for c10 in C10_GRID for c25 in C25_GRID)


@dataclass
class MetricSums:
    count: int = 0
    abs_rel_sum: float = 0.0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    delta1_count: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        if prediction.shape != target.shape or target.shape != mask.shape:
            raise ValueError(
                "Metric inputs must have equal shapes; "
                f"prediction={prediction.shape}, target={target.shape}, mask={mask.shape}"
            )
        selected_prediction = prediction[mask].astype(np.float64, copy=False)
        selected_target = target[mask].astype(np.float64, copy=False)
        if selected_target.size == 0:
            return
        if not (
            np.all(np.isfinite(selected_prediction))
            and np.all(selected_prediction > 0)
            and np.all(np.isfinite(selected_target))
            and np.all(selected_target > 0)
        ):
            raise ValueError("Metric support contains a non-finite or non-positive depth")
        difference = selected_prediction - selected_target
        ratio = np.maximum(
            selected_prediction / selected_target,
            selected_target / selected_prediction,
        )
        self.count += int(selected_target.size)
        self.abs_rel_sum += float(np.sum(np.abs(difference) / selected_target))
        self.abs_error_sum += float(np.sum(np.abs(difference)))
        self.squared_error_sum += float(np.sum(np.square(difference)))
        self.delta1_count += int(np.count_nonzero(ratio < 1.25))

    def merge(self, other: MetricSums) -> None:
        self.count += other.count
        self.abs_rel_sum += other.abs_rel_sum
        self.abs_error_sum += other.abs_error_sum
        self.squared_error_sum += other.squared_error_sum
        self.delta1_count += other.delta1_count

    def compute(self) -> dict[str, float | int | None]:
        if not self.count:
            return {
                "count": 0,
                "abs_rel": None,
                "mae_m": None,
                "rmse_m": None,
                "delta1": None,
            }
        return {
            "count": self.count,
            "abs_rel": self.abs_rel_sum / self.count,
            "mae_m": self.abs_error_sum / self.count,
            "rmse_m": math.sqrt(self.squared_error_sum / self.count),
            "delta1": self.delta1_count / self.count,
        }


@dataclass
class ScaleSummary:
    frames: int = 0
    scale_sum: float = 0.0
    scale_min: float = math.inf
    scale_max: float = -math.inf
    fallback_frames: int = 0
    q10_cap_triggered_frames: int = 0
    q25_cap_triggered_frames: int = 0

    def update(
        self,
        scale: float,
        *,
        fallback: bool,
        q10_triggered: bool,
        q25_triggered: bool,
    ) -> None:
        self.frames += 1
        self.scale_sum += scale
        self.scale_min = min(self.scale_min, scale)
        self.scale_max = max(self.scale_max, scale)
        self.fallback_frames += int(fallback)
        self.q10_cap_triggered_frames += int(q10_triggered)
        self.q25_cap_triggered_frames += int(q25_triggered)

    def compute(self) -> dict[str, float | int]:
        if not self.frames:
            raise RuntimeError("Cannot summarize zero scale estimates")
        return {
            "frames": self.frames,
            "mean": self.scale_sum / self.frames,
            "min": self.scale_min,
            "max": self.scale_max,
            "fallback_frames": self.fallback_frames,
            "q10_cap_triggered_frames": self.q10_cap_triggered_frames,
            "q25_cap_triggered_frames": self.q25_cap_triggered_frames,
        }


@dataclass
class CandidateStatistics:
    rooms: dict[str, dict[str, MetricSums]] = field(default_factory=dict)
    scales: ScaleSummary = field(default_factory=ScaleSummary)

    def room(self, room: str) -> dict[str, MetricSums]:
        return self.rooms.setdefault(room, {subset: MetricSums() for subset in SUBSETS})


@dataclass(frozen=True)
class PreparedSample:
    base_depth: np.ndarray
    bim_depth: np.ndarray
    gt_depth: np.ndarray
    masks: dict[str, np.ndarray]


@dataclass(frozen=True)
class DirectAuditFrame:
    sample_id: str
    room: str
    metrics: dict[str, dict[str, MetricSums]]
    fallback: bool
    q10_cap_triggered: bool
    q25_cap_triggered: bool


def _plain_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_config(item)
            for key, item in value.items()
            if key not in {"config_path", "project_root"}
        }
    if isinstance(value, list):
        return [_plain_config(item) for item in value]
    return value


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: manifest record must be an object")
        records.append(value)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _validate_locked_config(
    cfg: Config,
    resolution_provenance: Mapping[str, Any],
) -> None:
    if not cfg.data.get("split_annotation"):
        raise ValueError("Scale selection requires data.split_annotation")
    forbidden_region_fields = {
        name: list(cfg.data.get(name, []))
        for name in ("train_regions", "val_regions", "test_regions")
        if cfg.data.get(name, [])
    }
    if forbidden_region_fields:
        raise ValueError(
            "Scale selection accepts only annotation train IDs; region split "
            f"overrides must be empty: {sorted(forbidden_region_fields)}"
        )
    if cfg.data.get("record_stride_by_region", {}):
        raise ValueError(
            "Scale selection accepts the exhaustive annotation population; "
            "record_stride_by_region must be empty"
        )
    expected_annotation_sha = cfg.data.get("split_annotation_sha256")
    expected_split_sha = cfg.data.get("split_fingerprint_sha256")
    if not expected_annotation_sha or not expected_split_sha:
        raise ValueError(
            "Scale selection requires pinned data.split_annotation_sha256 and "
            "data.split_fingerprint_sha256"
        )
    actual_annotation_sha = str(resolution_provenance["annotation_raw_sha256"])
    actual_split_sha = str(resolution_provenance["fingerprint_sha256"])
    if str(expected_annotation_sha) != actual_annotation_sha:
        raise ValueError(
            "split_annotation_sha256 mismatch: "
            f"configured={expected_annotation_sha}, actual={actual_annotation_sha}"
        )
    if str(expected_split_sha) != actual_split_sha:
        raise ValueError(
            "split_fingerprint_sha256 mismatch: "
            f"configured={expected_split_sha}, actual={actual_split_sha}"
        )
    if float(cfg.data.get("min_depth", -1)) != 0.2 or float(cfg.data.get("max_depth", -1)) != 5.0:
        raise ValueError("Registered Stanford cap selection requires depth range 0.2-5.0 m")


def _assert_room_disjoint(
    records: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, str],
) -> dict[str, str]:
    owners: dict[str, set[str]] = {}
    for record in records:
        sample_id = str(record["id"])
        split = assignments[sample_id]
        if split not in ACTIVE_SPLITS:
            continue
        owners.setdefault(str(record["region"]), set()).add(split)
    leaked = {room: sorted(splits) for room, splits in owners.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Rooms overlap active splits: {leaked}")
    return {room: next(iter(splits)) for room, splits in sorted(owners.items())}


def _sample_path(record: Mapping[str, Any], processed_root: Path) -> Path:
    configured = Path(str(record["sample"])).expanduser()
    if configured.is_file():
        return configured.resolve()
    relative = record.get("sample_relative_to_processed")
    if relative:
        relocated = processed_root / str(relative)
        if relocated.is_file():
            return relocated.resolve()
    raise FileNotFoundError(f"{record['id']}: prepared sample does not exist: {configured}")


def _load_prepared_sample(record: Mapping[str, Any], processed_root: Path) -> PreparedSample:
    path = _sample_path(record, processed_root)
    required = {
        "base_depth",
        "bim_depth",
        "gt_depth",
        "gt_valid",
        "semantic_class",
        "furniture_mask",
        "non_structural_mask",
        "preparation_fingerprint_sha256",
    }
    with np.load(path, allow_pickle=False) as item:
        missing = sorted(required - set(item.files))
        if missing:
            raise ValueError(f"{record['id']}: prepared sample lacks fields {missing}")
        base = item["base_depth"].astype(np.float32)
        bim = item["bim_depth"].astype(np.float32)
        gt = item["gt_depth"].astype(np.float32)
        gt_valid = item["gt_valid"] > 0
        furniture = item["furniture_mask"] > 0
        non_structural = item["non_structural_mask"] > 0
        semantic = item["semantic_class"]
        embedded_preparation_fingerprint = str(item["preparation_fingerprint_sha256"].item())
    manifest_preparation_fingerprint = str(record.get("preparation_fingerprint_sha256", ""))
    if embedded_preparation_fingerprint != manifest_preparation_fingerprint:
        raise ValueError(
            f"{record['id']}: embedded preparation fingerprint "
            f"{embedded_preparation_fingerprint!r} differs from manifest "
            f"{manifest_preparation_fingerprint!r}"
        )
    shapes = {
        "base_depth": base.shape,
        "bim_depth": bim.shape,
        "gt_depth": gt.shape,
        "gt_valid": gt_valid.shape,
        "semantic_class": semantic.shape,
        "furniture_mask": furniture.shape,
        "non_structural_mask": non_structural.shape,
    }
    if len(set(shapes.values())) != 1 or base.ndim != 2:
        raise ValueError(f"{record['id']}: prepared arrays must be equal 2-D shapes: {shapes}")
    if np.any(furniture & ~non_structural):
        raise ValueError(f"{record['id']}: furniture_mask is not non-structural")
    if np.any(gt_valid & (~np.isfinite(gt) | (gt < 0.2) | (gt > 5.0))):
        raise ValueError(f"{record['id']}: gt_valid violates the fixed 0.2-5.0 m support")
    if np.any(gt_valid & (~np.isfinite(base) | (base <= 0))):
        raise ValueError(
            f"{record['id']}: base_depth is invalid on fixed GT support; "
            "candidate comparison cannot silently change support"
        )
    return PreparedSample(
        base_depth=base,
        bim_depth=bim,
        gt_depth=gt,
        masks={
            "all": gt_valid,
            "furniture": gt_valid & furniture,
            "non_structural": gt_valid & non_structural,
        },
    )


def _frame_candidate_scales(
    base_depth: np.ndarray,
    bim_depth: np.ndarray,
) -> dict[Candidate, tuple[float, bool, bool, bool]]:
    """Compute the fixed grid from BIM/base ratios, without GT or semantics."""

    valid = np.isfinite(base_depth) & np.isfinite(bim_depth) & (base_depth > 0) & (bim_depth > 0)
    # Preserve the authoritative estimator's float32 division before promoting
    # log-quantile reduction to float64; the post-selection audit checks exact
    # agreement with ``estimate_robust_bim_scale`` through the public wrapper.
    ratios = bim_depth[valid] / base_depth[valid]
    ratios = ratios[(ratios > RATIO_MIN) & (ratios < RATIO_MAX)]
    if ratios.size < MIN_SCALE_SAMPLES:
        return {candidate: (1.0, True, False, False) for candidate in CANDIDATES}
    q10, q25, q45 = (
        float(value)
        for value in np.quantile(
            np.log(ratios.astype(np.float64, copy=False)),
            (0.10, 0.25, 0.45),
        )
    )
    scales: dict[Candidate, tuple[float, bool, bool, bool]] = {}
    tolerance = 1e-12
    for candidate in CANDIDATES:
        q10_bound = q10 + candidate.q10_log_cap
        q25_bound = q25 + candidate.q25_log_cap
        robust_log_scale = min(q45, q25_bound, q10_bound)
        scales[candidate] = (
            math.exp(robust_log_scale),
            False,
            q10_bound < q45 - tolerance and q10_bound <= q25_bound,
            q25_bound < q45 - tolerance and q25_bound <= q10_bound,
        )
    return scales


def _update_prediction_metrics(
    destination: dict[str, MetricSums],
    prediction: np.ndarray,
    sample: PreparedSample,
) -> None:
    for subset in SUBSETS:
        destination[subset].update(prediction, sample.gt_depth, sample.masks[subset])


def _aggregate_rooms(
    room_metrics: Mapping[str, Mapping[str, MetricSums]],
    rooms: Iterable[str],
    subset: str,
) -> MetricSums:
    aggregate = MetricSums()
    for room in rooms:
        aggregate.merge(room_metrics[room][subset])
    return aggregate


def _room_macro(
    room_metrics: Mapping[str, Mapping[str, MetricSums]],
    rooms: Iterable[str],
    subset: str,
) -> dict[str, float | int | None]:
    computed = [room_metrics[room][subset].compute() for room in rooms]
    supported = [metric for metric in computed if int(metric["count"]) > 0]
    if not supported:
        return {
            "supported_rooms": 0,
            "abs_rel": None,
            "mae_m": None,
            "rmse_m": None,
            "delta1": None,
        }
    return {
        "supported_rooms": len(supported),
        "abs_rel": float(np.mean([float(metric["abs_rel"]) for metric in supported])),
        "mae_m": float(np.mean([float(metric["mae_m"]) for metric in supported])),
        "rmse_m": float(np.mean([float(metric["rmse_m"]) for metric in supported])),
        "delta1": float(np.mean([float(metric["delta1"]) for metric in supported])),
    }


def _candidate_objective(
    statistics: CandidateStatistics,
    rooms: Sequence[str],
) -> tuple[float, float]:
    primary = _room_macro(statistics.rooms, rooms, "all")["abs_rel"]
    furniture = _room_macro(statistics.rooms, rooms, "furniture")["abs_rel"]
    if primary is None:
        raise RuntimeError("Candidate has no all-pixel support in the selected train rooms")
    return float(primary), math.inf if furniture is None else float(furniture)


def _select_candidate(
    statistics: Mapping[Candidate, CandidateStatistics],
    rooms: Sequence[str],
) -> tuple[Candidate, dict[str, Any]]:
    scores = {
        candidate: _candidate_objective(candidate_statistics, rooms)
        for candidate, candidate_statistics in statistics.items()
    }
    minimum_primary = min(score[0] for score in scores.values())
    primary_ties = [
        candidate
        for candidate, score in scores.items()
        if score[0] <= minimum_primary + TIE_TOLERANCE
    ]
    minimum_secondary = min(scores[candidate][1] for candidate in primary_ties)
    if math.isinf(minimum_secondary):
        secondary_ties = primary_ties
    else:
        secondary_ties = [
            candidate
            for candidate in primary_ties
            if scores[candidate][1] <= minimum_secondary + TIE_TOLERANCE
        ]
    # The least restrictive cap is closest to the historical q=.45 estimator.
    # Prefer it when metric differences are only numerical ties.
    selected = max(
        secondary_ties,
        key=lambda candidate: (candidate.q10_log_cap, candidate.q25_log_cap),
    )
    primary, secondary = scores[selected]
    return selected, {
        "room_macro_abs_rel": primary,
        "furniture_room_macro_abs_rel": None if math.isinf(secondary) else secondary,
        "primary_tie_count": len(primary_ties),
        "secondary_tie_count": len(secondary_ties),
    }


def _serialize_candidate(
    candidate: Candidate,
    statistics: CandidateStatistics,
    train_rooms: Sequence[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    per_room: dict[str, Any] = {}
    for subset in SUBSETS:
        metrics[subset] = {
            "pixel_micro": _aggregate_rooms(statistics.rooms, train_rooms, subset).compute(),
            "room_macro": _room_macro(statistics.rooms, train_rooms, subset),
        }
    for room in train_rooms:
        per_room[room] = {subset: statistics.rooms[room][subset].compute() for subset in SUBSETS}
    objective = _candidate_objective(statistics, train_rooms)
    return {
        **candidate.as_json(),
        "selection_objective": {
            "room_macro_abs_rel": objective[0],
            "furniture_room_macro_abs_rel": (None if math.isinf(objective[1]) else objective[1]),
        },
        "scale_summary": statistics.scales.compute(),
        "metrics": metrics,
        "per_room": per_room,
    }


DIRECT_AUDIT_METHODS = (
    "raw_da3",
    "legacy_q45_scale_only",
    "legacy_q45_scale_local",
    "selected_robust_scale_only",
    "selected_robust_scale_local",
)


def _direct_audit_frame(
    record: Mapping[str, Any],
    processed_root: Path,
    selected: Candidate,
) -> DirectAuditFrame:
    sample_id = str(record["id"])
    room = str(record["region"])
    sample = _load_prepared_sample(record, processed_root)
    legacy_scaled, legacy_local, _, _, _ = bim_scale_and_local_features(
        sample.base_depth,
        sample.bim_depth,
    )
    robust_scaled, robust_local, _, _, estimate = robust_scale_and_local_features(
        sample.base_depth,
        sample.bim_depth,
        q10_log_cap=selected.q10_log_cap,
        q25_log_cap=selected.q25_log_cap,
        ratio_min=RATIO_MIN,
        ratio_max=RATIO_MAX,
        min_samples=MIN_SCALE_SAMPLES,
    )
    expected_scale = _frame_candidate_scales(
        sample.base_depth,
        sample.bim_depth,
    )[selected][0]
    if not math.isclose(estimate.scale, expected_scale, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(
            f"{sample_id}: grid scale {expected_scale} differs from runtime "
            f"estimator scale {estimate.scale}"
        )
    predictions = {
        "raw_da3": sample.base_depth,
        "legacy_q45_scale_only": legacy_scaled,
        "legacy_q45_scale_local": legacy_local,
        "selected_robust_scale_only": robust_scaled,
        "selected_robust_scale_local": robust_local,
    }
    metrics = {
        method: {subset: MetricSums() for subset in SUBSETS} for method in DIRECT_AUDIT_METHODS
    }
    for method, prediction in predictions.items():
        _update_prediction_metrics(metrics[method], prediction, sample)
    return DirectAuditFrame(
        sample_id=sample_id,
        room=room,
        metrics=metrics,
        fallback=estimate.fallback,
        q10_cap_triggered=estimate.q10_cap_triggered,
        q25_cap_triggered=estimate.q25_cap_triggered,
    )


@contextmanager
def _opencv_thread_limit(workers: int) -> Iterable[None]:
    previous_threads = int(cv2.getNumThreads())
    changed = workers > 1 and previous_threads != 1
    if changed:
        cv2.setNumThreads(1)
    try:
        yield
    finally:
        if changed:
            cv2.setNumThreads(previous_threads)


def _direct_audit(
    train_records: Sequence[Mapping[str, Any]],
    processed_root: Path,
    selected: Candidate,
    *,
    log_every: int,
    workers: int,
) -> tuple[dict[str, Any], list[str]]:
    if workers < 1:
        raise ValueError("direct audit workers must be positive")
    room_metrics: dict[str, dict[str, dict[str, MetricSums]]] = {}
    accessed_ids: list[str] = []
    fallback_frames = 0
    q10_triggered_frames = 0
    q25_triggered_frames = 0

    def reduce_results(results: Iterable[DirectAuditFrame]) -> None:
        nonlocal fallback_frames, q10_triggered_frames, q25_triggered_frames
        for index, frame in enumerate(results, start=1):
            # ``ThreadPoolExecutor.map`` yields in input order.  Reducing here,
            # rather than inside workers, keeps floating-point accumulation and
            # receipt hashes identical to workers=1.
            accessed_ids.append(frame.sample_id)
            fallback_frames += int(frame.fallback)
            q10_triggered_frames += int(frame.q10_cap_triggered)
            q25_triggered_frames += int(frame.q25_cap_triggered)
            destinations = room_metrics.setdefault(
                frame.room,
                {
                    method: {subset: MetricSums() for subset in SUBSETS}
                    for method in DIRECT_AUDIT_METHODS
                },
            )
            for method in DIRECT_AUDIT_METHODS:
                for subset in SUBSETS:
                    destinations[method][subset].merge(frame.metrics[method][subset])
            if log_every and (index % log_every == 0 or index == len(train_records)):
                print(
                    f"[direct audit {index}/{len(train_records)}] {frame.sample_id}",
                    flush=True,
                )

    def process(record: Mapping[str, Any]) -> DirectAuditFrame:
        return _direct_audit_frame(record, processed_root, selected)

    with _opencv_thread_limit(workers):
        if workers == 1:
            reduce_results(map(process, train_records))
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="stanford-direct-audit",
            ) as executor:
                reduce_results(executor.map(process, train_records))

    rooms = sorted(room_metrics)
    serialized_methods: dict[str, Any] = {}
    for method in DIRECT_AUDIT_METHODS:
        method_rooms = {room: room_metrics[room][method] for room in rooms}
        serialized_methods[method] = {
            subset: {
                "pixel_micro": _aggregate_rooms(method_rooms, rooms, subset).compute(),
                "room_macro": _room_macro(method_rooms, rooms, subset),
            }
            for subset in SUBSETS
        }
    return (
        {
            "purpose": (
                "post-selection train-only audit; local correction is evaluated only "
                "for the selected robust candidate and the registered legacy reference"
            ),
            "methods": serialized_methods,
            "selected_runtime_estimator": {
                "fallback_frames": fallback_frames,
                "q10_cap_triggered_frames": q10_triggered_frames,
                "q25_cap_triggered_frames": q25_triggered_frames,
            },
            "deterministic_reduction": "annotation input order",
        },
        accessed_ids,
    )


def _selector_protocol() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "stanford-area1-train-only-robust-log-cap-selection-v1",
        "population_authority": (
            "all and only IDs assigned split=train by the exhaustive pinned annotation; "
            "no room, stride, or sample-count override"
        ),
        "depth_support_m": [0.2, 5.0],
        "runtime_estimator_inputs": ["base_depth", "bim_depth"],
        "selection_statistics_inputs": [
            "gt_depth",
            "gt_valid",
            "semantic-derived furniture_mask",
            "semantic-derived non_structural_mask",
        ],
        "estimator": {
            "name": "log_upper_cap_v1",
            "formula": "exp(min(Q45(log(BIM/base)), Q25+c25, Q10+c10))",
            "ratio_filter": [RATIO_MIN, RATIO_MAX],
            "minimum_ratio_samples": MIN_SCALE_SAMPLES,
            "insufficient_support_fallback_scale": 1.0,
        },
        "candidate_grid": {
            "q10_log_cap": [_json_cap(value) for value in C10_GRID],
            "q25_log_cap": [_json_cap(value) for value in C25_GRID],
            "cartesian_candidate_count": len(CANDIDATES),
        },
        "selection": {
            "method": "leave-one-train-room-out plus final refit on all train rooms",
            "primary_objective": (
                "minimize equal-room mean of per-room pixel-pooled scale-only AbsRel "
                "on development train rooms"
            ),
            "secondary_tie_break": (
                "within absolute primary tolerance 1e-12, minimize equal-supported-room "
                "furniture AbsRel within absolute tolerance 1e-12"
            ),
            "final_tie_break": (
                "prefer largest c10, then largest c25 (least restrictive, closest to q45)"
            ),
            "tie_tolerance": TIE_TOLERANCE,
            "final_parameters": "apply the same rule using all train rooms",
        },
        "post_selection_audit": (
            "evaluate scale+registered local correction only for the selected robust "
            "candidate and legacy q45, still on train only"
        ),
        "validation_and_test_policy": "no validation/test prepared sample may be opened",
    }


def _code_identity(project_root: Path) -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        project_root / "src/bim_priorda3/baselines.py",
        project_root / "src/bim_priorda3/data/splits.py",
    )
    hashes = {str(path.relative_to(project_root)): _file_sha256(path) for path in paths}
    return {
        "files_sha256": hashes,
        "composite_sha256": _canonical_sha256(hashes),
    }


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Immutable selection receipt already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"Immutable selection receipt already exists: {path}")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def select_scale_caps(
    config_path: str | Path,
    output_path: str | Path,
    *,
    log_every: int = 100,
    workers: int = 8,
) -> dict[str, Any]:
    """Run the registered train-only selector and atomically publish its receipt."""

    if log_every < 0:
        raise ValueError("log_every must be non-negative")
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Immutable selection receipt already exists: {output}")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    config = Path(config_path).expanduser().resolve()
    config_raw_sha256 = _file_sha256(config)
    cfg = load_config(config)
    code_project_root = Path(__file__).resolve().parents[2]
    code_identity = _code_identity(code_project_root)
    protocol = _selector_protocol()
    effective_config = _plain_config(cfg)
    processed_root = resolve_project_path(cfg, cfg.data.processed_root)
    manifest = processed_root / "manifest.jsonl"
    annotation = resolve_project_path(cfg, cfg.data.split_annotation)
    manifest_raw_sha256 = _file_sha256(manifest)
    annotation_raw_sha256 = _file_sha256(annotation)
    records = _read_manifest(manifest)
    resolution = resolve_annotation_splits(records, annotation)
    _require_unchanged(config, config_raw_sha256, label="config")
    _require_unchanged(manifest, manifest_raw_sha256, label="manifest")
    _require_unchanged(annotation, annotation_raw_sha256, label="annotation")
    _validate_locked_config(cfg, resolution.provenance)
    room_owners = _assert_room_disjoint(records, resolution.assignments)
    train_records = resolution.records_for("train")
    train_ids = [str(record["id"]) for record in train_records]
    if not train_ids:
        raise ValueError("Pinned annotation has no train IDs")
    train_rooms = sorted({str(record["region"]) for record in train_records})
    if len(train_rooms) < 2:
        raise ValueError("Leave-one-train-room-out selection requires at least two train rooms")
    if any(room_owners.get(room) != "train" for room in train_rooms):
        raise RuntimeError("Internal error: selected train record belongs to a non-train room")

    statistics = {candidate: CandidateStatistics() for candidate in CANDIDATES}
    selection_accessed_ids: list[str] = []
    for index, record in enumerate(train_records, start=1):
        sample_id = str(record["id"])
        if resolution.assignments[sample_id] != "train":
            raise RuntimeError(f"Refusing to open non-train sample {sample_id}")
        room = str(record["region"])
        sample = _load_prepared_sample(record, processed_root)
        selection_accessed_ids.append(sample_id)
        candidate_scales = _frame_candidate_scales(sample.base_depth, sample.bim_depth)
        scale_groups: dict[float, list[Candidate]] = {}
        for candidate, (scale, fallback, q10_triggered, q25_triggered) in candidate_scales.items():
            statistics[candidate].scales.update(
                scale,
                fallback=fallback,
                q10_triggered=q10_triggered,
                q25_triggered=q25_triggered,
            )
            scale_groups.setdefault(scale, []).append(candidate)
        for scale, candidates in scale_groups.items():
            prediction = sample.base_depth * scale
            frame_metrics = {subset: MetricSums() for subset in SUBSETS}
            _update_prediction_metrics(frame_metrics, prediction, sample)
            for candidate in candidates:
                destination = statistics[candidate].room(room)
                for subset in SUBSETS:
                    destination[subset].merge(frame_metrics[subset])
        if log_every and (index % log_every == 0 or index == len(train_records)):
            print(f"[scale selection {index}/{len(train_records)}] {sample_id}", flush=True)

    if selection_accessed_ids != train_ids:
        raise RuntimeError("Selection pass did not access exactly the ordered annotation train IDs")
    for candidate_statistics in statistics.values():
        missing_rooms = sorted(set(train_rooms) - set(candidate_statistics.rooms))
        if missing_rooms:
            raise RuntimeError(f"Candidate statistics lack train rooms: {missing_rooms}")

    loro_folds: list[dict[str, Any]] = []
    for held_out_room in train_rooms:
        development_rooms = [room for room in train_rooms if room != held_out_room]
        fold_candidate, development_score = _select_candidate(
            statistics,
            development_rooms,
        )
        held_out_metrics = {
            subset: statistics[fold_candidate].rooms[held_out_room][subset].compute()
            for subset in SUBSETS
        }
        loro_folds.append(
            {
                "held_out_train_room": held_out_room,
                "development_room_count": len(development_rooms),
                "selected_candidate": fold_candidate.as_json(),
                "development_objective": development_score,
                "held_out_metrics": held_out_metrics,
            }
        )

    selected, final_score = _select_candidate(statistics, train_rooms)
    direct_audit, direct_accessed_ids = _direct_audit(
        train_records,
        processed_root,
        selected,
        log_every=log_every,
        workers=workers,
    )
    if direct_accessed_ids != train_ids:
        raise RuntimeError("Direct audit did not access exactly the ordered annotation train IDs")

    _require_unchanged(config, config_raw_sha256, label="config")
    _require_unchanged(manifest, manifest_raw_sha256, label="manifest")
    _require_unchanged(annotation, annotation_raw_sha256, label="annotation")
    current_code_identity = _code_identity(code_project_root)
    if current_code_identity != code_identity:
        raise RuntimeError(
            "Selector/runtime code changed during registered scale selection: "
            f"start={code_identity['composite_sha256']}, "
            f"current={current_code_identity['composite_sha256']}"
        )
    canonical_scale_estimator = {
        "name": "log_upper_cap_v1",
        "q10_log_cap": _json_cap(selected.q10_log_cap),
        "q25_log_cap": _json_cap(selected.q25_log_cap),
        "ratio_min": RATIO_MIN,
        "ratio_max": RATIO_MAX,
        "min_samples": MIN_SCALE_SAMPLES,
    }
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": time.monotonic() - started,
        "protocol": protocol,
        "protocol_sha256": _canonical_sha256(protocol),
        "execution": {
            "direct_audit_workers": workers,
            "opencv_internal_threads_during_parallel_audit": 1 if workers > 1 else None,
            "reduction_order": "annotation input order",
            "affects_protocol_or_metrics": False,
        },
        "provenance": {
            "config": str(config),
            "config_raw_sha256": config_raw_sha256,
            "effective_config_sha256": _canonical_sha256(effective_config),
            "manifest": str(manifest.resolve()),
            "manifest_raw_sha256": manifest_raw_sha256,
            "manifest_preparation_fingerprint_status": resolution.provenance[
                "manifest_preparation_fingerprint_status"
            ],
            "manifest_preparation_fingerprint_sha256": resolution.provenance[
                "manifest_preparation_fingerprint_sha256"
            ],
            "annotation": str(annotation.resolve()),
            "annotation_raw_sha256": resolution.provenance["annotation_raw_sha256"],
            "split_fingerprint_sha256": resolution.provenance["fingerprint_sha256"],
            "code": code_identity,
        },
        "split_isolation": {
            "annotation_split_counts": resolution.provenance["split_counts"],
            "room_disjoint": True,
            "room_owners": room_owners,
            "train_sample_count": len(train_ids),
            "train_room_count": len(train_rooms),
            "train_rooms": train_rooms,
            "ordered_train_ids_sha256": _canonical_sha256(train_ids),
            "annotation_ordered_train_ids_sha256": resolution.provenance["ordered_ids_sha256"][
                "train"
            ],
            "selection_accessed_ids_sha256": _canonical_sha256(selection_accessed_ids),
            "direct_audit_accessed_ids_sha256": _canonical_sha256(direct_accessed_ids),
            "validation_samples_opened": 0,
            "test_samples_opened": 0,
        },
        "candidate_results": [
            _serialize_candidate(candidate, statistics[candidate], train_rooms)
            for candidate in CANDIDATES
        ],
        "leave_one_train_room_out": {
            "fold_count": len(loro_folds),
            "folds": loro_folds,
        },
        "final_selection": {
            "canonical_scale_estimator": canonical_scale_estimator,
            "all_train_objective": final_score,
            "selection_scope": "train only",
        },
        "selected_candidate_direct_audit": direct_audit,
        "run_started_utc": started_at.isoformat(),
    }
    _atomic_write_new_json(output, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select robust log-scale caps from all and only pinned Stanford annotation train IDs"
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Progress interval only; does not alter the selected population or protocol",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Ordered direct-audit worker count; affects runtime only, not population, "
            "protocol, reduction order, or metrics"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    receipt = select_scale_caps(
        args.config,
        args.output,
        log_every=args.log_every,
        workers=args.workers,
    )
    output = args.output.expanduser().resolve()
    print(
        json.dumps(
            {
                "output": str(output),
                "receipt_sha256": _file_sha256(output),
                "protocol_sha256": receipt["protocol_sha256"],
                "train_samples": receipt["split_isolation"]["train_sample_count"],
                "train_rooms": receipt["split_isolation"]["train_room_count"],
                "direct_audit_workers": receipt["execution"]["direct_audit_workers"],
                "scale_estimator": receipt["final_selection"]["canonical_scale_estimator"],
                "paste_under": "model.scale_estimator",
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
