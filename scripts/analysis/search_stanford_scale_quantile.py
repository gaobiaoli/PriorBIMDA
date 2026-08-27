#!/usr/bin/env python3
"""Select a fixed BIM/DA3 scale quantile on train, then audit it on test.

The runtime estimator uses only registered BIM depth and frozen DA3 depth. GT
is used to select one dataset-level quantile on the pinned train split and to
score frozen predictions. Test mode requires the immutable train receipt and
does not expose a command-line quantile override.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.baselines import (
    PREVIOUS_FIXED_PARAMETERS,
    previous_local_correction_features,
    resolve_scale_estimator_config,
)
from bim_priorda3.config import load_config
from bim_priorda3.data.dataset import BIMDepthDataset
from bim_priorda3.data.stanford2d3ds import (
    load_stanford_all_valid_depth,
    official_regular_depth_path,
)
from bim_priorda3.engine import semantic_config_sha256

QUANTILES = tuple(round(value / 100.0, 2) for value in range(5, 96))
RATIO_MIN = 0.2
RATIO_MAX = 5.0
MIN_SCALE_SAMPLES = 100
METHODS = (
    "raw_da3",
    "q45_scale",
    "current_robust_scale",
    "selected_quantile_scale",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_sha256() -> str:
    return _sha256(Path(__file__).resolve())


@dataclass
class MetricSums:
    count: int = 0
    abs_rel_sum: float = 0.0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    delta1_count: int = 0
    delta2_count: int = 0
    delta3_count: int = 0

    def update(self, prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> None:
        predicted = prediction[valid].astype(np.float64, copy=False)
        expected = target[valid].astype(np.float64, copy=False)
        if not expected.size:
            return
        difference = predicted - expected
        ratio = np.maximum(predicted / expected, expected / predicted)
        self.count += int(expected.size)
        self.abs_rel_sum += float(np.sum(np.abs(difference) / expected))
        self.absolute_error_sum += float(np.sum(np.abs(difference)))
        self.squared_error_sum += float(np.sum(np.square(difference)))
        self.delta1_count += int(np.count_nonzero(ratio < 1.25))
        self.delta2_count += int(np.count_nonzero(ratio < 1.25**2))
        self.delta3_count += int(np.count_nonzero(ratio < 1.25**3))

    def merge(self, other: MetricSums) -> None:
        self.count += other.count
        self.abs_rel_sum += other.abs_rel_sum
        self.absolute_error_sum += other.absolute_error_sum
        self.squared_error_sum += other.squared_error_sum
        self.delta1_count += other.delta1_count
        self.delta2_count += other.delta2_count
        self.delta3_count += other.delta3_count

    def compute(self) -> dict[str, float | int]:
        if not self.count:
            raise RuntimeError("Cannot compute metrics without valid pixels")
        return {
            "abs_rel": self.abs_rel_sum / self.count,
            "mae": self.absolute_error_sum / self.count,
            "rmse": math.sqrt(self.squared_error_sum / self.count),
            "delta1": self.delta1_count / self.count,
            "delta2": self.delta2_count / self.count,
            "delta3": self.delta3_count / self.count,
            "count": self.count,
        }


@dataclass(frozen=True)
class LoadedFrame:
    sample_id: str
    room: str
    base_depth: np.ndarray
    bim_depth: np.ndarray
    target_depth: np.ndarray
    target_valid: np.ndarray
    ratios: np.ndarray


@dataclass(frozen=True)
class SelectionFrame:
    sample_id: str
    room: str
    count: int
    candidate_abs_rel_sums: np.ndarray
    robust_abs_rel_sum: float
    candidate_bim_direct_abs_rel_sums: np.ndarray | None
    robust_bim_direct_abs_rel_sum: float | None
    candidate_scales: np.ndarray
    robust_scale: float
    fallback: bool


@dataclass(frozen=True)
class EvaluationFrame:
    sample_id: str
    room: str
    scales: dict[str, float]
    fallback: bool
    metrics: dict[str, MetricSums]


def _load_frame(
    record: Mapping[str, Any],
    target_shape: tuple[int, int],
    *,
    apply_da3_metric_focal_scaling: bool,
) -> LoadedFrame:
    sample_path = Path(str(record["sample"]))
    with np.load(sample_path, allow_pickle=False) as item:
        base = item["base_depth"].astype(np.float32)
        bim = item["bim_depth"].astype(np.float32)
        bim_valid = item["bim_valid"] > 0
        if apply_da3_metric_focal_scaling:
            if "intrinsic" not in item.files:
                raise RuntimeError(
                    f"{record['id']}: focal-corrected quantile search requires intrinsics"
                )
            intrinsic = item["intrinsic"].astype(np.float32)
            if intrinsic.shape != (3, 3):
                raise ValueError(f"{record['id']}: intrinsic must be 3x3")
            focal_scale = float((intrinsic[0, 0] + intrinsic[1, 1]) / 2.0 / 300.0)
            if not np.isfinite(focal_scale) or focal_scale <= 0:
                raise ValueError(f"{record['id']}: invalid DA3 focal scale {focal_scale}")
            base *= focal_scale
    if base.shape != bim.shape or base.shape != bim_valid.shape or base.ndim != 2:
        raise ValueError(f"{record['id']}: prepared depth arrays have incompatible shapes")
    invalid_bim = ~bim_valid
    if np.any(np.abs(bim[invalid_bim]) > 1e-6) or np.any(~np.isfinite(bim[invalid_bim])):
        raise ValueError(f"{record['id']}: invalid BIM pixels do not contain canonical zeros")
    bim[invalid_bim] = 0.0
    target, target_valid = load_stanford_all_valid_depth(
        official_regular_depth_path(record["image"]),
        target_shape,
    )
    metric_valid = (
        target_valid & np.isfinite(target) & (target > 0) & np.isfinite(base) & (base > 0)
    )
    ratio_valid = bim_valid & np.isfinite(base) & (base > 0) & np.isfinite(bim) & (bim > 0)
    ratios = bim[ratio_valid] / base[ratio_valid]
    ratios = ratios[(ratios > RATIO_MIN) & (ratios < RATIO_MAX)]
    return LoadedFrame(
        sample_id=str(record["id"]),
        room=str(record["region"]),
        base_depth=base,
        bim_depth=bim,
        target_depth=target,
        target_valid=metric_valid,
        ratios=ratios,
    )


def _candidate_scales(ratios: np.ndarray) -> tuple[np.ndarray, bool]:
    if ratios.size < MIN_SCALE_SAMPLES:
        return np.ones(len(QUANTILES), dtype=np.float64), True
    return np.quantile(
        ratios.astype(np.float64, copy=False),
        QUANTILES,
    ), False


def _robust_scale(ratios: np.ndarray, parameters: Mapping[str, Any]) -> float:
    if ratios.size < int(parameters["min_samples"]):
        return 1.0
    log_values = np.log(ratios.astype(np.float64, copy=False))
    q10, q25, q45 = np.quantile(log_values, (0.10, 0.25, 0.45))
    return float(
        np.exp(
            min(
                q45,
                q25 + float(parameters["q25_log_cap"]),
                q10 + float(parameters["q10_log_cap"]),
            )
        )
    )


def _absrel_sums_for_scales(
    base_depth: np.ndarray,
    target_depth: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Compute exact AbsRel sums for many scalar scales in O(N log N + Q log N)."""

    base = base_depth[valid].astype(np.float64, copy=False)
    target = target_depth[valid].astype(np.float64, copy=False)
    thresholds = np.sort(target / base)
    weights = 1.0 / thresholds
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(weights)))
    indices = np.searchsorted(thresholds, scales, side="left")
    lower_weights = prefix[indices]
    count = int(thresholds.size)
    sums = scales * (2.0 * lower_weights - prefix[-1]) + count - 2.0 * indices
    return np.maximum(sums, 0.0)


def _bim_direct_absrel_sums_for_scales(
    base_depth: np.ndarray,
    bim_depth: np.ndarray,
    target_depth: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    """Evaluate the complete deterministic local BIM correction for each scale."""

    parameters = PREVIOUS_FIXED_PARAMETERS
    expected = target_depth[valid].astype(np.float64, copy=False)
    sums = np.empty(len(scales), dtype=np.float64)
    for index, scale in enumerate(scales):
        scaled = (base_depth * float(scale)).astype(np.float32)
        field, _ = previous_local_correction_features(
            scaled,
            bim_depth,
            float(parameters["consistency_log_threshold"]),
            float(parameters["smoothing_sigma"]),
        )
        prediction = scaled * np.exp(float(parameters["local_correction_alpha"]) * field)
        predicted = prediction[valid].astype(np.float64, copy=False)
        sums[index] = float(np.sum(np.abs(predicted - expected) / expected))
    return sums


def _selection_frame(
    record: Mapping[str, Any],
    target_shape: tuple[int, int],
    robust_parameters: Mapping[str, Any],
    *,
    apply_da3_metric_focal_scaling: bool,
    include_local_direct: bool,
) -> SelectionFrame:
    frame = _load_frame(
        record,
        target_shape,
        apply_da3_metric_focal_scaling=apply_da3_metric_focal_scaling,
    )
    scales, fallback = _candidate_scales(frame.ratios)
    robust = _robust_scale(frame.ratios, robust_parameters)
    all_sums = _absrel_sums_for_scales(
        frame.base_depth,
        frame.target_depth,
        frame.target_valid,
        np.concatenate((scales, np.asarray([robust], dtype=np.float64))),
    )
    direct_sums: np.ndarray | None = None
    robust_direct_sum: float | None = None
    if include_local_direct:
        all_direct_sums = _bim_direct_absrel_sums_for_scales(
            frame.base_depth,
            frame.bim_depth,
            frame.target_depth,
            frame.target_valid,
            np.concatenate((scales, np.asarray([robust], dtype=np.float64))),
        )
        direct_sums = all_direct_sums[:-1]
        robust_direct_sum = float(all_direct_sums[-1])
    return SelectionFrame(
        sample_id=frame.sample_id,
        room=frame.room,
        count=int(np.count_nonzero(frame.target_valid)),
        candidate_abs_rel_sums=all_sums[:-1],
        robust_abs_rel_sum=float(all_sums[-1]),
        candidate_bim_direct_abs_rel_sums=direct_sums,
        robust_bim_direct_abs_rel_sum=robust_direct_sum,
        candidate_scales=scales,
        robust_scale=robust,
        fallback=fallback,
    )


def _room_macro(room_sums: Mapping[str, np.ndarray], room_counts: Mapping[str, int]) -> np.ndarray:
    return np.mean(
        np.stack([room_sums[room] / room_counts[room] for room in sorted(room_sums)]),
        axis=0,
    )


def _selected_index(scores: np.ndarray) -> int:
    minimum = float(np.min(scores))
    tied = np.flatnonzero(np.isclose(scores, minimum, rtol=0.0, atol=1e-12))
    return int(tied[-1])


def _run_selection(
    cfg: Any,
    config_path: Path,
    output: Path,
    *,
    split: str,
    include_local_direct: bool,
    workers: int,
    log_every: int,
) -> None:
    if split not in {"train", "val"}:
        raise ValueError("Quantile selection split must be train or val")
    dataset = BIMDepthDataset(cfg, split, augment=False, require_ground_truth=False)
    if str(cfg.data.get("ground_truth_support", "")) != "official_all_valid":
        raise ValueError("Quantile search requires data.ground_truth_support=official_all_valid")
    robust_parameters = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    if robust_parameters["name"] != "log_upper_cap_v1":
        raise ValueError("Current comparator must use log_upper_cap_v1")
    target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
    apply_da3_metric_focal_scaling = bool(
        cfg.data.get("apply_da3_metric_focal_scaling", False)
    )
    room_sums: dict[str, np.ndarray] = {}
    room_robust_sums: dict[str, float] = defaultdict(float)
    room_counts: dict[str, int] = defaultdict(int)
    global_sums = np.zeros(len(QUANTILES), dtype=np.float64)
    global_robust_sum = 0.0
    global_direct_sums = np.zeros(len(QUANTILES), dtype=np.float64)
    global_robust_direct_sum = 0.0
    room_direct_sums: dict[str, np.ndarray] = {}
    room_robust_direct_sums: dict[str, float] = defaultdict(float)
    global_count = 0
    scale_sum = np.zeros(len(QUANTILES), dtype=np.float64)
    scale_min = np.full(len(QUANTILES), np.inf, dtype=np.float64)
    scale_max = np.full(len(QUANTILES), -np.inf, dtype=np.float64)
    robust_scale_sum = 0.0
    fallback_frames = 0
    accessed_ids: list[str] = []

    def process(record: Mapping[str, Any]) -> SelectionFrame:
        return _selection_frame(
            record,
            target_shape,
            robust_parameters,
            apply_da3_metric_focal_scaling=apply_da3_metric_focal_scaling,
            include_local_direct=include_local_direct,
        )

    executor: ThreadPoolExecutor | None = None
    results: Iterable[SelectionFrame]
    if workers == 1:
        results = map(process, dataset.records)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quantile-search")
        results = executor.map(process, dataset.records)
    try:
        for index, frame in enumerate(results, start=1):
            destination = room_sums.setdefault(
                frame.room,
                np.zeros(len(QUANTILES), dtype=np.float64),
            )
            destination += frame.candidate_abs_rel_sums
            room_robust_sums[frame.room] += frame.robust_abs_rel_sum
            room_counts[frame.room] += frame.count
            global_sums += frame.candidate_abs_rel_sums
            global_robust_sum += frame.robust_abs_rel_sum
            if include_local_direct:
                assert frame.candidate_bim_direct_abs_rel_sums is not None
                assert frame.robust_bim_direct_abs_rel_sum is not None
                direct_destination = room_direct_sums.setdefault(
                    frame.room,
                    np.zeros(len(QUANTILES), dtype=np.float64),
                )
                direct_destination += frame.candidate_bim_direct_abs_rel_sums
                room_robust_direct_sums[
                    frame.room
                ] += frame.robust_bim_direct_abs_rel_sum
                global_direct_sums += frame.candidate_bim_direct_abs_rel_sums
                global_robust_direct_sum += frame.robust_bim_direct_abs_rel_sum
            global_count += frame.count
            scale_sum += frame.candidate_scales
            scale_min = np.minimum(scale_min, frame.candidate_scales)
            scale_max = np.maximum(scale_max, frame.candidate_scales)
            robust_scale_sum += frame.robust_scale
            fallback_frames += int(frame.fallback)
            accessed_ids.append(frame.sample_id)
            if log_every and (index % log_every == 0 or index == len(dataset.records)):
                print(
                    f"[{split} quantile search {index}/{len(dataset.records)}] "
                    f"{frame.sample_id}"
                )
    finally:
        if executor is not None:
            executor.shutdown()

    pixel_scores = global_sums / global_count
    room_scores = _room_macro(room_sums, room_counts)
    pixel_selected = _selected_index(pixel_scores)
    room_selected = _selected_index(room_scores)
    direct_pixel_scores = global_direct_sums / global_count
    direct_room_scores = (
        _room_macro(room_direct_sums, room_counts) if include_local_direct else None
    )
    direct_pixel_selected = (
        _selected_index(direct_pixel_scores) if include_local_direct else None
    )
    direct_room_selected = (
        _selected_index(direct_room_scores)
        if direct_room_scores is not None
        else None
    )
    rooms = sorted(room_sums)
    loo_winners: list[float] = []
    for held_out in rooms:
        development_rooms = [room for room in rooms if room != held_out]
        development_scores = np.mean(
            np.stack([room_sums[room] / room_counts[room] for room in development_rooms]),
            axis=0,
        )
        loo_winners.append(QUANTILES[_selected_index(development_scores)])
    loo_counts = Counter(loo_winners)
    candidates = []
    for index, quantile in enumerate(QUANTILES):
        candidate = {
            "quantile": quantile,
            "train_pixel_micro_abs_rel": float(pixel_scores[index]),
            "train_room_macro_abs_rel": float(room_scores[index]),
            "mean_scale": float(scale_sum[index] / len(dataset.records)),
            "min_scale": float(scale_min[index]),
            "max_scale": float(scale_max[index]),
            "leave_one_room_out_win_count": int(loo_counts.get(quantile, 0)),
        }
        if include_local_direct:
            assert direct_room_scores is not None
            candidate.update(
                train_pixel_micro_bim_direct_abs_rel=float(
                    direct_pixel_scores[index]
                ),
                train_room_macro_bim_direct_abs_rel=float(
                    direct_room_scores[index]
                ),
            )
        candidates.append(candidate)
    q45_index = QUANTILES.index(0.45)
    selected_quantile = QUANTILES[pixel_selected]
    selected_direct_quantile = (
        QUANTILES[direct_pixel_selected]
        if direct_pixel_selected is not None
        else None
    )
    summary = {
        "schema_version": 1,
        "protocol": "stanford-area1-development-fixed-scale-quantile-search-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _source_sha256(),
        "config": {
            "path": str(config_path),
            "raw_sha256": _sha256(config_path),
            "semantic_sha256": semantic_config_sha256(cfg),
        },
        "dataset": {
            "split": split,
            "sample_count": len(dataset.records),
            "rooms": rooms,
            "room_count": len(rooms),
            "valid_pixel_count": global_count,
            "split_provenance": dataset.split_provenance,
            "accessed_sample_ids_sha256": _canonical_sha256(accessed_ids),
        },
        "runtime_estimator": {
            "formula": "Q_q(BIM/DA3) per frame",
            "inputs": [
                "registered hit-only BIM depth",
                (
                    "focal-corrected frozen DA3 metric depth"
                    if apply_da3_metric_focal_scaling
                    else "frozen cached DA3 canonical-focal depth"
                ),
            ],
            "da3_metric_focal_scaling_applied": apply_da3_metric_focal_scaling,
            "ratio_filter": [RATIO_MIN, RATIO_MAX],
            "minimum_scale_samples": MIN_SCALE_SAMPLES,
            "fallback_scale": 1.0,
            "ground_truth_used_at_runtime": False,
        },
        "selection": {
            "candidate_grid": {
                "minimum": QUANTILES[0],
                "maximum": QUANTILES[-1],
                "step": 0.01,
                "count": len(QUANTILES),
            },
            "primary_objective": (
                f"minimum official-all-valid {split} pixel-micro AbsRel"
            ),
            "selected_quantile": selected_quantile,
            "selected_train_pixel_micro_abs_rel": float(pixel_scores[pixel_selected]),
            "selected_train_room_macro_abs_rel": float(room_scores[pixel_selected]),
            "room_macro_optimal_quantile_development_only_not_tested": QUANTILES[
                room_selected
            ],
            "room_macro_optimal_abs_rel": float(room_scores[room_selected]),
            "leave_one_development_room_out_winner_counts": {
                format(quantile, ".2f"): count for quantile, count in sorted(loo_counts.items())
            },
            "test_policy": "freeze selected_quantile; do not use test to revise it",
            **(
                {
                    "selected_bim_direct_quantile": selected_direct_quantile,
                    "selected_development_pixel_micro_bim_direct_abs_rel": float(
                        direct_pixel_scores[direct_pixel_selected]
                    ),
                    "room_macro_optimal_bim_direct_quantile": QUANTILES[
                        direct_room_selected
                    ],
                    "room_macro_optimal_bim_direct_abs_rel": float(
                        direct_room_scores[direct_room_selected]
                    ),
                }
                if include_local_direct
                and direct_pixel_selected is not None
                and direct_room_selected is not None
                and direct_room_scores is not None
                else {}
            ),
        },
        "references": {
            "q45": {
                "train_pixel_micro_abs_rel": float(pixel_scores[q45_index]),
                "train_room_macro_abs_rel": float(room_scores[q45_index]),
            },
            "current_log_upper_cap_v1": {
                "parameters": robust_parameters,
                "train_pixel_micro_abs_rel": global_robust_sum / global_count,
                "train_room_macro_abs_rel": float(
                    np.mean([room_robust_sums[room] / room_counts[room] for room in rooms])
                ),
                "mean_scale": robust_scale_sum / len(dataset.records),
                **(
                    {
                        "development_pixel_micro_bim_direct_abs_rel": (
                            global_robust_direct_sum / global_count
                        ),
                        "development_room_macro_bim_direct_abs_rel": float(
                            np.mean(
                                [
                                    room_robust_direct_sums[room] / room_counts[room]
                                    for room in rooms
                                ]
                            )
                        ),
                    }
                    if include_local_direct
                    else {}
                ),
            },
        },
        "local_bim_direct_search": {
            "enabled": include_local_direct,
            "consistency_log_threshold": PREVIOUS_FIXED_PARAMETERS[
                "consistency_log_threshold"
            ],
            "smoothing_sigma": PREVIOUS_FIXED_PARAMETERS["smoothing_sigma"],
            "local_correction_alpha": PREVIOUS_FIXED_PARAMETERS[
                "local_correction_alpha"
            ],
        },
        "fallback_frames": fallback_frames,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"selected q={selected_quantile:.2f}; receipt={output}")


def _evaluate_frame(
    record: Mapping[str, Any],
    target_shape: tuple[int, int],
    selected_quantile: float,
    robust_parameters: Mapping[str, Any],
    *,
    apply_da3_metric_focal_scaling: bool,
) -> EvaluationFrame:
    frame = _load_frame(
        record,
        target_shape,
        apply_da3_metric_focal_scaling=apply_da3_metric_focal_scaling,
    )
    fallback = frame.ratios.size < MIN_SCALE_SAMPLES
    if fallback:
        selected_scale = q45_scale = 1.0
    else:
        q45_scale, selected_scale = np.quantile(
            frame.ratios.astype(np.float64, copy=False),
            (0.45, selected_quantile),
        )
    robust_scale = _robust_scale(frame.ratios, robust_parameters)
    scales = {
        "raw_da3": 1.0,
        "q45_scale": float(q45_scale),
        "current_robust_scale": robust_scale,
        "selected_quantile_scale": float(selected_scale),
    }
    metrics: dict[str, MetricSums] = {}
    for method, scale in scales.items():
        prediction = (frame.base_depth * scale).astype(np.float32)
        sums = MetricSums()
        sums.update(prediction, frame.target_depth, frame.target_valid)
        metrics[method] = sums
    return EvaluationFrame(
        sample_id=frame.sample_id,
        room=frame.room,
        scales=scales,
        fallback=fallback,
        metrics=metrics,
    )


def _metric_mean(values: Sequence[dict[str, float | int]]) -> dict[str, float]:
    keys = ("abs_rel", "mae", "rmse", "delta1", "delta2", "delta3")
    return {key: float(np.mean([float(value[key]) for value in values])) for key in keys}


def _paired_room_bootstrap(
    candidate: Mapping[str, dict[str, float | int]],
    reference: Mapping[str, dict[str, float | int]],
    *,
    repetitions: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    rooms = sorted(candidate)
    differences = np.asarray(
        [float(candidate[room]["abs_rel"]) - float(reference[room]["abs_rel"]) for room in rooms],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rooms), size=(repetitions, len(rooms)))
    samples = np.mean(differences[indices], axis=1)
    return {
        "difference": "selected_quantile_scale - current_robust_scale; negative is better",
        "rooms": rooms,
        "mean_room_difference": float(np.mean(differences)),
        "candidate_better_room_fraction": float(np.mean(differences < 0)),
        "confidence_interval_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "repetitions": repetitions,
        "seed": seed,
    }


def _run_test(
    cfg: Any,
    config_path: Path,
    selection_path: Path,
    output_dir: Path,
    *,
    workers: int,
    log_every: int,
) -> None:
    selection = json.loads(selection_path.read_text())
    if selection.get("protocol") not in {
        "stanford-area1-train-only-fixed-scale-quantile-search-v1",
        "stanford-area1-development-fixed-scale-quantile-search-v2",
    }:
        raise ValueError("Selection receipt uses an unsupported protocol")
    selected_quantile = float(selection["selection"]["selected_quantile"])
    if selected_quantile not in QUANTILES:
        raise ValueError("Selection receipt quantile is outside the immutable candidate grid")
    if selection["config"]["semantic_sha256"] != semantic_config_sha256(cfg):
        raise ValueError("Selection and evaluation configurations are semantically different")
    dataset = BIMDepthDataset(cfg, "test", augment=False, require_ground_truth=False)
    train_provenance = selection["dataset"]["split_provenance"]
    for key in ("annotation_raw_sha256", "fingerprint_sha256"):
        if train_provenance.get(key) != dataset.split_provenance.get(key):
            raise ValueError(f"Train receipt and test split provenance differ for {key}")
    robust_parameters = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
    apply_da3_metric_focal_scaling = bool(
        cfg.data.get("apply_da3_metric_focal_scaling", False)
    )
    micro = {method: MetricSums() for method in METHODS}
    rooms: dict[str, dict[str, MetricSums]] = defaultdict(
        lambda: {method: MetricSums() for method in METHODS}
    )
    frame_metrics: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    accessed_ids: list[str] = []
    fallback_frames = 0

    def process(record: Mapping[str, Any]) -> EvaluationFrame:
        return _evaluate_frame(
            record,
            target_shape,
            selected_quantile,
            robust_parameters,
            apply_da3_metric_focal_scaling=apply_da3_metric_focal_scaling,
        )

    executor: ThreadPoolExecutor | None = None
    results: Iterable[EvaluationFrame]
    if workers == 1:
        results = map(process, dataset.records)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="quantile-test")
        results = executor.map(process, dataset.records)
    try:
        for index, frame in enumerate(results, start=1):
            row: dict[str, Any] = {
                "sample_id": frame.sample_id,
                "room": frame.room,
                "fallback": frame.fallback,
            }
            for method in METHODS:
                micro[method].merge(frame.metrics[method])
                rooms[frame.room][method].merge(frame.metrics[method])
                computed = frame.metrics[method].compute()
                frame_metrics[method].append(computed)
                row[f"{method}_scale"] = frame.scales[method]
                for metric in ("abs_rel", "mae", "rmse", "delta1"):
                    row[f"{method}_{metric}"] = computed[metric]
            rows.append(row)
            accessed_ids.append(frame.sample_id)
            fallback_frames += int(frame.fallback)
            if log_every and (index % log_every == 0 or index == len(dataset.records)):
                print(f"[frozen q test {index}/{len(dataset.records)}] {frame.sample_id}")
    finally:
        if executor is not None:
            executor.shutdown()

    room_metrics = {
        room: {method: values[method].compute() for method in METHODS}
        for room, values in sorted(rooms.items())
    }
    aggregates: dict[str, Any] = {}
    for method in METHODS:
        aggregates[method] = {
            "pixel_micro": micro[method].compute(),
            "frame_macro": _metric_mean(frame_metrics[method]),
            "room_macro": _metric_mean(
                [room_metrics[room][method] for room in sorted(room_metrics)]
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = output_dir / "per_frame.csv"
    with per_frame_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1,
        "protocol": "stanford-area1-frozen-train-selected-scale-quantile-test-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": _source_sha256(),
        "config": {
            "path": str(config_path),
            "raw_sha256": _sha256(config_path),
            "semantic_sha256": semantic_config_sha256(cfg),
        },
        "selection_receipt": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
            "selected_quantile": selected_quantile,
            "train_objective": selection["selection"]["primary_objective"],
            "selected_train_abs_rel": selection["selection"]["selected_train_pixel_micro_abs_rel"],
        },
        "dataset": {
            "split": "test",
            "sample_count": len(dataset.records),
            "rooms": sorted(room_metrics),
            "valid_pixel_count": micro["raw_da3"].count,
            "split_provenance": dataset.split_provenance,
            "accessed_sample_ids_sha256": _canonical_sha256(accessed_ids),
        },
        "runtime_estimator": selection["runtime_estimator"],
        "fallback_frames": fallback_frames,
        "aggregates": aggregates,
        "per_room": room_metrics,
        "paired_room_bootstrap_vs_current_robust": _paired_room_bootstrap(
            {room: room_metrics[room]["selected_quantile_scale"] for room in room_metrics},
            {room: room_metrics[room]["current_robust_scale"] for room in room_metrics},
        ),
        "per_frame_csv": str(per_frame_path),
        "per_frame_csv_sha256": _sha256(per_frame_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"test summary={summary_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-receipt", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument(
        "--include-local-direct",
        action="store_true",
        help="Also scan the full consistency-gated Gaussian BIM-direct output",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.log_every < 0:
        raise ValueError("--log-every must be nonnegative")
    if args.include_local_direct:
        # Python workers already parallelize frames; nested OpenCV worker pools
        # otherwise oversubscribe the host during the 91-candidate scan.
        cv2.setNumThreads(1)
    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)
    if args.split in {"train", "val"}:
        if args.selection_receipt is not None:
            raise ValueError("Train selection does not accept --selection-receipt")
        _run_selection(
            cfg,
            config_path,
            args.output.expanduser().resolve(),
            split=args.split,
            include_local_direct=args.include_local_direct,
            workers=args.workers,
            log_every=args.log_every,
        )
        return
    if args.selection_receipt is None:
        raise ValueError("Test evaluation requires the frozen train --selection-receipt")
    _run_test(
        cfg,
        config_path,
        args.selection_receipt.expanduser().resolve(),
        args.output.expanduser().resolve(),
        workers=args.workers,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
