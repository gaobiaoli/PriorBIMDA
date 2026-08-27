#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch

from bim_priorda3.baselines import (
    LEGACY_SCALE_ESTIMATOR,
    PREVIOUS_FIXED_PARAMETERS,
    ROBUST_LOG_CAP_SCALE_ESTIMATOR,
    ScaleEstimate,
    previous_scale_baselines,
    resolve_scale_estimator_config,
    robust_scale_and_local_features,
)
from bim_priorda3.checkpoints import (
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_model_config,
)
from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import (
    BIMDepthDataset,
    load_stanford_all_valid_depth,
    official_regular_depth_path,
)
from bim_priorda3.engine import build_loader, move_batch, seed_everything
from bim_priorda3.models import BIMPriorDA3
from bim_priorda3.scale_protocol import validate_universal_scale_protocol

METRIC_NAMES = ("abs_rel", "rmse", "mae", "delta1", "delta2", "delta3")
_BASELINE_MAX_WORKERS = 8
_BaselineResult = TypeVar("_BaselineResult")

_ROBUST_SELECTION_PROTOCOL_V1: dict[str, Any] = {
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
        "ratio_filter": [0.2, 5.0],
        "minimum_ratio_samples": 100,
        "insufficient_support_fallback_scale": 1.0,
    },
    "candidate_grid": {
        "q10_log_cap": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, "inf"],
        "q25_log_cap": [0.025, 0.05, 0.075, 0.1, 0.15, "inf"],
        "cartesian_candidate_count": 48,
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
        "tie_tolerance": 1e-12,
        "final_parameters": "apply the same rule using all train rooms",
    },
    "post_selection_audit": (
        "evaluate scale+registered local correction only for the selected robust "
        "candidate and legacy q45, still on train only"
    ),
    "validation_and_test_policy": "no validation/test prepared sample may be opened",
}

_ROBUST_SELECTION_CORE_CODE_FILES = {
    "src/bim_priorda3/baselines.py",
    "src/bim_priorda3/data/splits.py",
}
_ROBUST_SELECTION_SELECTOR_PATHS = {
    # The first path is retained solely to verify immutable pre-reorganization
    # receipts. New receipts always record the responsibility-based data path.
    "scripts/select_stanford_scale_caps.py",
    "scripts/data/select_stanford_scale_caps.py",
}


class MetricSums:
    def __init__(self) -> None:
        self.count = 0
        self.abs_rel_sum = 0.0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.delta1_sum = 0
        self.delta2_sum = 0
        self.delta3_sum = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        *,
        prediction_name: str = "prediction",
        context: str = "",
    ) -> None:
        if prediction.shape != target.shape or valid.shape != target.shape:
            raise ValueError(
                f"{context}: metric tensors must have identical shapes; "
                f"{prediction_name}={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}, support={tuple(valid.shape)}"
            )
        mask = valid.bool()
        invalid_target = mask & (~torch.isfinite(target) | (target <= 0))
        invalid_target_count = int(invalid_target.sum().item())
        if invalid_target_count:
            raise RuntimeError(
                f"{context}: fixed metric support contains {invalid_target_count} "
                "non-finite or non-positive target values"
            )
        invalid_prediction = mask & (~torch.isfinite(prediction) | (prediction <= 0))
        invalid_prediction_count = int(invalid_prediction.sum().item())
        if invalid_prediction_count:
            raise RuntimeError(
                f"{context}: {prediction_name} has {invalid_prediction_count} "
                "non-finite or non-positive values on the fixed metric support"
            )
        pred = prediction[mask].double()
        gt = target[mask].double()
        if not pred.numel():
            return
        difference = pred - gt
        ratio = torch.maximum(pred / gt, gt / pred)
        self.count += int(pred.numel())
        self.abs_rel_sum += float((difference.abs() / gt).sum())
        self.squared_error_sum += float(difference.square().sum())
        self.absolute_error_sum += float(difference.abs().sum())
        self.delta1_sum += int((ratio < 1.25).sum())
        self.delta2_sum += int((ratio < 1.25**2).sum())
        self.delta3_sum += int((ratio < 1.25**3).sum())

    def compute(self) -> dict[str, float | int]:
        if not self.count:
            return {**{name: float("nan") for name in METRIC_NAMES}, "count": 0}
        return {
            "abs_rel": self.abs_rel_sum / self.count,
            "rmse": (self.squared_error_sum / self.count) ** 0.5,
            "mae": self.absolute_error_sum / self.count,
            "delta1": self.delta1_sum / self.count,
            "delta2": self.delta2_sum / self.count,
            "delta3": self.delta3_sum / self.count,
            "count": self.count,
        }


def _macro(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    valid_rows = [row for row in rows if int(row["count"]) > 0]
    if not valid_rows:
        return {
            **{name: float("nan") for name in METRIC_NAMES},
            "groups": 0,
            "pixel_count": 0,
        }
    return {
        **{name: float(np.mean([float(row[name]) for row in valid_rows])) for name in METRIC_NAMES},
        "groups": len(valid_rows),
        "pixel_count": sum(int(row["count"]) for row in valid_rows),
    }


def _assert_comparable_counts(
    metrics_by_method: dict[str, dict[str, float | int]],
    expected_count: int,
    *,
    context: str,
) -> None:
    actual = {method: int(metrics["count"]) for method, metrics in metrics_by_method.items()}
    mismatched = {method: count for method, count in actual.items() if count != expected_count}
    if mismatched:
        raise RuntimeError(
            f"{context}: comparable methods do not share the fixed support; "
            f"expected={expected_count}, actual={actual}"
        )


def _support_size(metrics: dict[str, float | int]) -> int:
    if "count" in metrics:
        return int(metrics["count"])
    return int(metrics.get("groups", 0))


def _beats_on_absrel_and_mae(
    candidate: dict[str, float | int],
    reference: dict[str, float | int],
) -> bool | None:
    candidate_support = _support_size(candidate)
    reference_support = _support_size(reference)
    if candidate_support != reference_support:
        raise RuntimeError(
            "Candidate/reference supports differ: "
            f"candidate={candidate_support}, reference={reference_support}"
        )
    if candidate_support == 0:
        return None
    values = [float(candidate[metric]) for metric in ("abs_rel", "mae")] + [
        float(reference[metric]) for metric in ("abs_rel", "mae")
    ]
    if not np.isfinite(values).all():
        raise RuntimeError("Non-finite metric encountered on non-empty fixed support")
    return all(float(candidate[metric]) < float(reference[metric]) for metric in ("abs_rel", "mae"))


def _coverage_fraction(hit_pixels: int, gt_pixels: int) -> float:
    return float(hit_pixels / gt_pixels) if gt_pixels > 0 else float("nan")


def _official_regular_depth_path(record: Mapping[str, Any]) -> Path:
    """Resolve an official Stanford regular-view depth PNG from its RGB record."""

    return official_regular_depth_path(str(record["image"]))


def _load_all_valid_regular_depth(
    record: Mapping[str, Any],
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Load every positive, non-sentinel official z-depth value."""

    return load_stanford_all_valid_depth(
        _official_regular_depth_path(record),
        target_shape,
    )


def _previous_baseline_tensors(
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if base_depth.shape != bim_depth.shape or base_depth.ndim != 4:
        raise ValueError(
            "Fixed BIM baseline expects equal [B, 1, H, W] depth tensors; "
            f"base={tuple(base_depth.shape)}, bim={tuple(bim_depth.shape)}"
        )
    if base_depth.shape[0] != 1 or base_depth.shape[1] != 1:
        raise ValueError(
            "Fixed BIM baseline evaluator requires batch size and channels equal to one"
        )
    base_np = base_depth[0, 0].detach().cpu().numpy()
    bim_np = bim_depth[0, 0].detach().cpu().numpy()
    scaled_np, direct_np, scale = previous_scale_baselines(base_np, bim_np)
    tensor_options = {"device": base_depth.device, "dtype": base_depth.dtype}
    scaled = torch.from_numpy(scaled_np)[None, None].to(**tensor_options)
    direct = torch.from_numpy(direct_np)[None, None].to(**tensor_options)
    return scaled, direct, float(scale)


def _legacy_scale_estimate(
    base_depth: np.ndarray,
    bim_depth: np.ndarray,
) -> ScaleEstimate:
    """Return diagnostics for the immutable historical q=.45 estimator."""

    valid = np.isfinite(base_depth) & np.isfinite(bim_depth) & (base_depth > 0) & (bim_depth > 0)
    ratios = bim_depth[valid] / base_depth[valid]
    ratios = ratios[(ratios > 0.2) & (ratios < 5.0)]
    support_count = int(ratios.size)
    fallback = support_count < 100
    quantiles = (
        tuple(
            (quantile, float(value))
            for quantile, value in zip(
                (0.10, 0.25, 0.45),
                np.quantile(ratios.astype(np.float64, copy=False), (0.10, 0.25, 0.45)),
            )
        )
        if not fallback
        else ()
    )
    return ScaleEstimate(
        scale=(dict(quantiles)[0.45] if quantiles else 1.0),
        support_count=support_count,
        quantiles=quantiles,
        fallback=fallback,
        q10_cap_triggered=False,
        q25_cap_triggered=False,
        estimator=LEGACY_SCALE_ESTIMATOR,
    )


def _ordered_baseline_map(
    function: Callable[[int], _BaselineResult],
    sample_count: int,
    *,
    max_workers: int | None = None,
) -> list[_BaselineResult]:
    """Run CPU baselines concurrently and retain the exact input order.

    OpenCV process-global thread settings are deliberately left untouched.
    """

    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    worker_limit = _BASELINE_MAX_WORKERS if max_workers is None else int(max_workers)
    if worker_limit < 1:
        raise ValueError("max_workers must be positive")
    if sample_count == 0:
        return []
    worker_count = min(sample_count, worker_limit)
    if worker_count == 1:
        return [function(index) for index in range(sample_count)]
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="stanford-baseline",
    ) as executor:
        return list(executor.map(function, range(sample_count)))


def _previous_baseline_batch_tensors(
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    *,
    max_workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[ScaleEstimate]]:
    """Apply the authoritative NumPy/OpenCV baseline independently per frame."""
    if base_depth.shape != bim_depth.shape or base_depth.ndim != 4:
        raise ValueError(
            "Fixed BIM baseline expects equal [B, 1, H, W] depth tensors; "
            f"base={tuple(base_depth.shape)}, bim={tuple(bim_depth.shape)}"
        )
    if base_depth.shape[0] < 1 or base_depth.shape[1] != 1:
        raise ValueError("Fixed BIM baseline requires a non-empty batch and one channel")

    base_cpu = base_depth.detach().cpu().contiguous()
    bim_cpu = bim_depth.detach().cpu().contiguous()

    def compute_sample(sample_index: int) -> tuple[torch.Tensor, torch.Tensor, ScaleEstimate]:
        base_np = base_cpu[sample_index, 0].numpy()
        bim_np = bim_cpu[sample_index, 0].numpy()
        scaled_np, direct_np, scale = previous_scale_baselines(base_np, bim_np)
        estimate = _legacy_scale_estimate(
            base_np.astype(np.float32, copy=False),
            bim_np.astype(np.float32, copy=False),
        )
        if not np.isclose(scale, estimate.scale, rtol=1e-6, atol=1e-7):
            raise RuntimeError("Historical q=.45 scale and its diagnostic receipt disagree")
        return torch.from_numpy(scaled_np), torch.from_numpy(direct_np), estimate

    results = _ordered_baseline_map(
        compute_sample,
        int(base_depth.shape[0]),
        max_workers=max_workers,
    )
    scaled_samples, direct_samples, estimates = zip(*results, strict=True)
    options = {"device": base_depth.device, "dtype": base_depth.dtype}
    return (
        torch.stack(scaled_samples).unsqueeze(1).to(**options),
        torch.stack(direct_samples).unsqueeze(1).to(**options),
        list(estimates),
    )


def _robust_baseline_batch_tensors(
    base_depth: torch.Tensor,
    bim_depth: torch.Tensor,
    parameters: Mapping[str, Any],
    *,
    max_workers: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[ScaleEstimate]]:
    """Apply one frozen robust estimator independently to each frame."""

    resolved = resolve_scale_estimator_config(parameters)
    if resolved["name"] != ROBUST_LOG_CAP_SCALE_ESTIMATOR:
        raise ValueError("Robust Stanford comparator must use log_upper_cap_v1")
    if base_depth.shape != bim_depth.shape or base_depth.ndim != 4:
        raise ValueError(
            "Robust BIM baseline expects equal [B, 1, H, W] tensors; "
            f"base={tuple(base_depth.shape)}, bim={tuple(bim_depth.shape)}"
        )
    if base_depth.shape[0] < 1 or base_depth.shape[1] != 1:
        raise ValueError("Robust BIM baseline requires a non-empty batch and one channel")
    base_cpu = base_depth.detach().float().cpu().contiguous()
    bim_cpu = bim_depth.detach().float().cpu().contiguous()

    def compute_sample(sample_index: int) -> tuple[torch.Tensor, torch.Tensor, ScaleEstimate]:
        scaled, direct, _, _, estimate = robust_scale_and_local_features(
            base_cpu[sample_index, 0].numpy(),
            bim_cpu[sample_index, 0].numpy(),
            q10_log_cap=float(resolved["q10_log_cap"]),
            q25_log_cap=float(resolved["q25_log_cap"]),
            ratio_min=float(resolved["ratio_min"]),
            ratio_max=float(resolved["ratio_max"]),
            min_samples=int(resolved["min_samples"]),
        )
        return torch.from_numpy(scaled), torch.from_numpy(direct), estimate

    results = _ordered_baseline_map(
        compute_sample,
        int(base_depth.shape[0]),
        max_workers=max_workers,
    )
    scaled_samples, direct_samples, estimates = zip(*results, strict=True)
    options = {"device": base_depth.device, "dtype": base_depth.dtype}
    return (
        torch.stack(scaled_samples).unsqueeze(1).to(**options),
        torch.stack(direct_samples).unsqueeze(1).to(**options),
        list(estimates),
    )


def _scale_estimate_columns(
    prefix: str,
    estimate: ScaleEstimate,
) -> dict[str, Any]:
    quantiles = dict(estimate.quantiles)
    return {
        f"{prefix}_estimator": estimate.estimator,
        f"{prefix}_scale": float(estimate.scale),
        f"{prefix}_support_count": int(estimate.support_count),
        f"{prefix}_fallback": bool(estimate.fallback),
        f"{prefix}_q10": float(quantiles.get(0.10, float("nan"))),
        f"{prefix}_q25": float(quantiles.get(0.25, float("nan"))),
        f"{prefix}_q45": float(quantiles.get(0.45, float("nan"))),
        f"{prefix}_q10_cap_triggered": bool(estimate.q10_cap_triggered),
        f"{prefix}_q25_cap_triggered": bool(estimate.q25_cap_triggered),
        f"{prefix}_cap_triggered": bool(estimate.q10_cap_triggered or estimate.q25_cap_triggered),
    }


def _resolve_robust_comparator(
    cfg: Any,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve a comparator without changing the checkpoint's model inputs."""

    model_scale = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    evaluation = getattr(cfg, "evaluation", {})
    evaluation_scale = (
        evaluation.get("robust_scale_estimator") if isinstance(evaluation, Mapping) else None
    )
    if evaluation_scale is not None:
        resolved = resolve_scale_estimator_config(evaluation_scale)
        if resolved["name"] != ROBUST_LOG_CAP_SCALE_ESTIMATOR:
            raise ValueError("evaluation.robust_scale_estimator must use log_upper_cap_v1")
        if model_scale["name"] == ROBUST_LOG_CAP_SCALE_ESTIMATOR and resolved != model_scale:
            raise ValueError(
                "A robust model and evaluation.robust_scale_estimator must use "
                "identical frozen parameters"
            )
        return resolved, "evaluation.robust_scale_estimator"
    if model_scale["name"] == ROBUST_LOG_CAP_SCALE_ESTIMATOR:
        return model_scale, "model.scale_estimator"
    return None, "unavailable"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_lowercase_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase hexadecimal SHA256")
    return value


def _validate_registered_robust_protocol(
    payload: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> str:
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise TypeError("Robust scale selection receipt protocol must be a mapping")
    protocol_payload = dict(protocol)
    recorded_sha256 = _require_lowercase_sha256(
        payload.get("protocol_sha256"),
        label="Robust scale selection receipt protocol_sha256",
    )
    recomputed_sha256 = _canonical_sha256(protocol_payload)
    if recorded_sha256 != recomputed_sha256:
        raise ValueError(
            "Robust scale selection receipt protocol canonical SHA256 mismatch: "
            f"recorded={recorded_sha256}, recomputed={recomputed_sha256}"
        )

    if protocol_payload.get("schema_version") != 1:
        raise ValueError("Unknown robust scale selection protocol schema_version")
    if protocol_payload.get("name") != _ROBUST_SELECTION_PROTOCOL_V1["name"]:
        raise ValueError("Unknown robust scale selection protocol name")
    if protocol_payload.get("candidate_grid") != _ROBUST_SELECTION_PROTOCOL_V1["candidate_grid"]:
        raise ValueError("Robust scale selection protocol is not the fixed 48-candidate grid")
    estimator = protocol_payload.get("estimator")
    if (
        not isinstance(estimator, Mapping)
        or estimator.get("formula") != (_ROBUST_SELECTION_PROTOCOL_V1["estimator"]["formula"])
    ):
        raise ValueError("Robust scale selection protocol estimator formula differs")
    if protocol_payload.get("selection") != _ROBUST_SELECTION_PROTOCOL_V1["selection"]:
        raise ValueError("Robust scale selection protocol objective or tie-break differs")
    if (
        protocol_payload.get("validation_and_test_policy")
        != (_ROBUST_SELECTION_PROTOCOL_V1["validation_and_test_policy"])
    ):
        raise ValueError("Robust scale selection protocol validation/test policy differs")
    if protocol_payload != _ROBUST_SELECTION_PROTOCOL_V1:
        raise ValueError(
            "Robust scale selection receipt does not use the complete registered v1 protocol"
        )

    configured_sha256 = evaluation.get("robust_scale_selection_protocol_sha256")
    if configured_sha256 is not None:
        configured_sha256 = _require_lowercase_sha256(
            configured_sha256,
            label="evaluation.robust_scale_selection_protocol_sha256",
        )
        if configured_sha256 != recomputed_sha256:
            raise ValueError(
                "Configured robust scale selection protocol SHA256 mismatch: "
                f"configured={configured_sha256}, receipt={recomputed_sha256}"
            )
    return recomputed_sha256


def _validate_fixed_candidate_results(payload: Mapping[str, Any]) -> None:
    candidate_results = payload.get("candidate_results")
    if not isinstance(candidate_results, list):
        raise TypeError("Robust scale selection candidate_results must be a list")
    q10_values = _ROBUST_SELECTION_PROTOCOL_V1["candidate_grid"]["q10_log_cap"]
    q25_values = _ROBUST_SELECTION_PROTOCOL_V1["candidate_grid"]["q25_log_cap"]
    expected_candidates = [(q10, q25) for q10 in q10_values for q25 in q25_values]
    if len(candidate_results) != len(expected_candidates):
        raise ValueError(
            "Robust scale selection receipt must contain all 48 fixed-grid "
            f"candidate results; found={len(candidate_results)}"
        )
    for index, (result, (expected_q10, expected_q25)) in enumerate(
        zip(candidate_results, expected_candidates, strict=True)
    ):
        if not isinstance(result, Mapping):
            raise TypeError(f"Robust scale candidate result {index} must be a mapping")
        if result.get("q10_log_cap") != expected_q10 or result.get("q25_log_cap") != expected_q25:
            raise ValueError(
                "Robust scale selection candidate_results do not exactly cover the "
                f"registered fixed grid in canonical order at index {index}"
            )


def _validate_selector_code_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("Robust scale selection receipt provenance must be a mapping")
    code = provenance.get("code")
    if not isinstance(code, Mapping):
        raise TypeError("Robust scale selection receipt code identity must be a mapping")
    files = code.get("files_sha256")
    if not isinstance(files, Mapping):
        raise TypeError("Robust scale selection receipt files_sha256 must be a mapping")
    normalized_files: dict[str, str] = {}
    for path, value in files.items():
        if not isinstance(path, str) or not path:
            raise ValueError("Selector code identity paths must be non-empty strings")
        normalized_files[path] = _require_lowercase_sha256(
            value,
            label=f"Selector code identity SHA256 for {path}",
        )
    missing = sorted(_ROBUST_SELECTION_CORE_CODE_FILES - set(normalized_files))
    if missing:
        raise ValueError(
            f"Robust scale selection receipt code identity lacks required files: {missing}"
        )
    selector_paths = sorted(_ROBUST_SELECTION_SELECTOR_PATHS & set(normalized_files))
    if len(selector_paths) != 1:
        raise ValueError(
            "Robust scale selection receipt must identify exactly one registered selector path"
        )
    composite = _require_lowercase_sha256(
        code.get("composite_sha256"),
        label="Robust scale selection receipt code composite_sha256",
    )
    recomputed_composite = _canonical_sha256(normalized_files)
    if composite != recomputed_composite:
        raise ValueError(
            "Robust scale selection receipt code composite SHA256 mismatch: "
            f"recorded={composite}, recomputed={recomputed_composite}"
        )
    return {
        "files_sha256": normalized_files,
        "composite_sha256": composite,
        "selector_path": selector_paths[0],
    }


def _validate_train_only_receipt_binding(
    payload: Mapping[str, Any],
    split_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if split_provenance.get("mode") != "annotations":
        raise ValueError("Formal robust receipt verification requires annotation split provenance")
    ordered_ids_sha256 = split_provenance.get("ordered_ids_sha256")
    if not isinstance(ordered_ids_sha256, Mapping):
        raise TypeError("Current split provenance ordered_ids_sha256 must be a mapping")
    current_train_ids_sha256 = _require_lowercase_sha256(
        ordered_ids_sha256.get("train"),
        label="Current split provenance ordered train IDs SHA256",
    )
    split_counts = split_provenance.get("split_counts")
    if not isinstance(split_counts, Mapping):
        raise TypeError("Current split provenance split_counts must be a mapping")
    current_train_sample_count = split_counts.get("train")
    if type(current_train_sample_count) is not int or current_train_sample_count < 1:
        raise ValueError("Current annotation train sample count must be a positive integer")
    split_region_counts = split_provenance.get("split_region_counts")
    if not isinstance(split_region_counts, Mapping):
        raise TypeError("Current split provenance split_region_counts must be a mapping")
    current_train_region_counts = split_region_counts.get("train")
    if not isinstance(current_train_region_counts, Mapping):
        raise TypeError("Current annotation train region counts must be a mapping")
    invalid_region_counts = {
        str(room): count
        for room, count in current_train_region_counts.items()
        if type(count) is not int or count < 0
    }
    if invalid_region_counts:
        raise ValueError(
            "Current annotation train region counts contain invalid values: "
            f"{invalid_region_counts}"
        )
    current_train_rooms = sorted(
        str(room) for room, count in current_train_region_counts.items() if count > 0
    )
    if sum(int(count) for count in current_train_region_counts.values()) != (
        current_train_sample_count
    ):
        raise ValueError("Current annotation train region counts do not sum to split_counts.train")

    split_isolation = payload.get("split_isolation")
    if not isinstance(split_isolation, Mapping):
        raise TypeError("Robust scale receipt split_isolation must be a mapping")
    if split_isolation.get("room_disjoint") is not True:
        raise ValueError("Robust scale selection receipt does not attest room disjointness")
    receipt_split_counts = split_isolation.get("annotation_split_counts")
    if receipt_split_counts != dict(split_counts):
        raise ValueError(
            "Robust scale selection receipt annotation split counts differ from "
            "the current annotation"
        )
    receipt_train_count = split_isolation.get("train_sample_count")
    if receipt_train_count != current_train_sample_count:
        raise ValueError(
            "Robust scale selection receipt train sample count differs from the "
            f"current annotation: receipt={receipt_train_count!r}, "
            f"current={current_train_sample_count}"
        )
    receipt_train_room_count = split_isolation.get("train_room_count")
    if receipt_train_room_count != len(current_train_rooms):
        raise ValueError(
            "Robust scale selection receipt train room count differs from the "
            f"current annotation: receipt={receipt_train_room_count!r}, "
            f"current={len(current_train_rooms)}"
        )
    if split_isolation.get("train_rooms") != current_train_rooms:
        raise ValueError(
            "Robust scale selection receipt train room IDs differ from the current annotation"
        )

    train_hash_fields = (
        "ordered_train_ids_sha256",
        "annotation_ordered_train_ids_sha256",
        "selection_accessed_ids_sha256",
        "direct_audit_accessed_ids_sha256",
    )
    receipt_hashes = {
        key: _require_lowercase_sha256(
            split_isolation.get(key),
            label=f"Robust scale selection receipt {key}",
        )
        for key in train_hash_fields
    }
    mismatched_hashes = {
        key: value for key, value in receipt_hashes.items() if value != current_train_ids_sha256
    }
    if mismatched_hashes:
        raise ValueError(
            "Robust scale selection receipt train ID access hashes do not all match "
            "the current annotation ordered train hash: "
            f"current={current_train_ids_sha256}, receipt={mismatched_hashes}"
        )
    for split in ("validation", "test"):
        opened = split_isolation.get(f"{split}_samples_opened")
        if opened != 0:
            raise ValueError(f"Robust scale selection receipt opened {split} samples: {opened!r}")
    return {
        "ordered_train_ids_sha256": current_train_ids_sha256,
        "train_sample_count": current_train_sample_count,
        "train_room_count": len(current_train_rooms),
        "train_rooms": current_train_rooms,
    }


def _checkpoint_robust_receipt_binding(
    checkpoint_config: Mapping[str, Any],
    *,
    receipt_sha256: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    checkpoint_model = checkpoint_config.get("model", {})
    if not isinstance(checkpoint_model, Mapping):
        raise TypeError("Checkpoint training config model must be a mapping")
    checkpoint_scale = resolve_scale_estimator_config(checkpoint_model.get("scale_estimator"))
    if checkpoint_scale["name"] != ROBUST_LOG_CAP_SCALE_ESTIMATOR:
        return {
            "status": "legacy_source_independent_target_train_comparator",
            "checkpoint_scale_estimator": checkpoint_scale,
            "receipt_match_required": False,
        }

    checkpoint_evaluation = checkpoint_config.get("evaluation")
    if not isinstance(checkpoint_evaluation, Mapping):
        raise TypeError(
            "A robust target checkpoint requires evaluation receipt provenance in "
            "its training config"
        )
    checkpoint_receipt_sha256 = _require_lowercase_sha256(
        checkpoint_evaluation.get("robust_scale_selection_receipt_sha256"),
        label=("Checkpoint training config evaluation.robust_scale_selection_receipt_sha256"),
    )
    if checkpoint_receipt_sha256 != receipt_sha256:
        raise ValueError(
            "Runtime robust selection receipt SHA256 differs from the robust target "
            "checkpoint training config: "
            f"runtime={receipt_sha256}, checkpoint={checkpoint_receipt_sha256}"
        )
    checkpoint_protocol_sha256 = checkpoint_evaluation.get("robust_scale_selection_protocol_sha256")
    if checkpoint_protocol_sha256 is not None:
        checkpoint_protocol_sha256 = _require_lowercase_sha256(
            checkpoint_protocol_sha256,
            label=("Checkpoint training config evaluation.robust_scale_selection_protocol_sha256"),
        )
        if checkpoint_protocol_sha256 != protocol_sha256:
            raise ValueError(
                "Runtime robust selection protocol SHA256 differs from the robust "
                "target checkpoint training config"
            )
    return {
        "status": "robust_target_checkpoint_receipt_match",
        "checkpoint_scale_estimator": checkpoint_scale,
        "receipt_match_required": True,
        "checkpoint_receipt_sha256": checkpoint_receipt_sha256,
        "checkpoint_protocol_sha256": checkpoint_protocol_sha256,
    }


def _robust_selection_receipt_provenance(
    cfg: Any,
    comparator: Mapping[str, Any] | None,
    split_provenance: Mapping[str, Any],
    *,
    allow_unverified: bool,
    checkpoint_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify an optional immutable train-only cap-selection receipt."""

    evaluation = getattr(cfg, "evaluation", {})
    if not isinstance(evaluation, Mapping):
        if comparator is None:
            return {"status": "not_applicable"}
        raise TypeError("Formal robust evaluation config must be a mapping")
    raw_path = evaluation.get("robust_scale_selection_receipt")
    expected_sha256 = evaluation.get("robust_scale_selection_receipt_sha256")
    configured_protocol_sha256 = evaluation.get("robust_scale_selection_protocol_sha256")
    if raw_path is None and expected_sha256 is None:
        if configured_protocol_sha256 is not None:
            raise ValueError(
                "evaluation.robust_scale_selection_protocol_sha256 cannot be "
                "configured without a receipt path and receipt SHA256"
            )
        if comparator is None:
            return {"status": "not_applicable"}
        if not allow_unverified:
            raise ValueError(
                "A robust comparator in the formal Stanford protocol requires "
                "evaluation.robust_scale_selection_receipt and its SHA256. "
                "Use --allow-unverified-robust-comparator only for exploratory runs."
            )
        return {
            "status": "unverified_explicit_exploratory_opt_out",
            "formal_protocol_eligible": False,
        }
    if raw_path is None or expected_sha256 is None:
        raise ValueError(
            "evaluation robust scale selection receipt path and SHA256 must be configured together"
        )
    if comparator is None:
        raise ValueError("A robust scale selection receipt requires an available robust comparator")
    expected_sha256 = _require_lowercase_sha256(
        expected_sha256,
        label="evaluation.robust_scale_selection_receipt_sha256",
    )
    receipt_path = resolve_project_path(cfg, raw_path)
    actual_sha256 = _sha256(receipt_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "Robust scale selection receipt SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Robust scale selection receipt must be a JSON object")
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError("Robust scale selection receipt is not a complete schema-v1 receipt")
    protocol_sha256 = _validate_registered_robust_protocol(payload, evaluation)
    _validate_fixed_candidate_results(payload)
    code_identity = _validate_selector_code_identity(payload)
    try:
        selected_raw = payload["final_selection"]["canonical_scale_estimator"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Robust scale selection receipt lacks final canonical estimator") from exc
    selected = resolve_scale_estimator_config(selected_raw)
    if selected != dict(comparator):
        raise ValueError("Robust comparator parameters differ from the pinned selection receipt")
    if payload["final_selection"].get("selection_scope") != "train only":
        raise ValueError("Robust scale selection receipt was not selected on train only")
    train_binding = _validate_train_only_receipt_binding(payload, split_provenance)
    split_isolation = payload["split_isolation"]
    leave_one_out = payload.get("leave_one_train_room_out")
    if (
        not isinstance(leave_one_out, Mapping)
        or leave_one_out.get("fold_count") != (train_binding["train_room_count"])
    ):
        raise ValueError(
            "Robust scale selection leave-one-train-room-out fold count differs "
            "from the current annotation train room count"
        )

    receipt_provenance = payload.get("provenance")
    if not isinstance(receipt_provenance, Mapping):
        raise TypeError("Robust scale selection receipt provenance must be a mapping")
    annotation = resolve_project_path(cfg, cfg.data.split_annotation)
    manifest = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
    current_identities = {
        "annotation_raw_sha256": _sha256(annotation),
        "split_fingerprint_sha256": str(split_provenance.get("fingerprint_sha256", "")),
        "manifest_raw_sha256": _sha256(manifest),
        "manifest_preparation_fingerprint_sha256": str(
            split_provenance.get(
                "manifest_preparation_fingerprint_sha256",
                "",
            )
        ),
    }
    configured_annotation_sha256 = str(cfg.data.get("split_annotation_sha256", ""))
    configured_split_sha256 = str(cfg.data.get("split_fingerprint_sha256", ""))
    if configured_annotation_sha256 != current_identities["annotation_raw_sha256"]:
        raise ValueError("Configured Stanford annotation SHA256 differs from the current file")
    if configured_split_sha256 != current_identities["split_fingerprint_sha256"]:
        raise ValueError("Configured Stanford split fingerprint differs from the resolved dataset")
    for key, current in current_identities.items():
        receipt_value = str(receipt_provenance.get(key, ""))
        if receipt_value != current:
            raise ValueError(
                f"Robust scale selection receipt {key} differs from the current "
                f"Stanford dataset: receipt={receipt_value}, current={current}"
            )
    checkpoint_binding = _checkpoint_robust_receipt_binding(
        checkpoint_config,
        receipt_sha256=actual_sha256,
        protocol_sha256=protocol_sha256,
    )
    return {
        "status": "verified",
        "formal_protocol_eligible": True,
        "path": str(receipt_path),
        "sha256": actual_sha256,
        "protocol_sha256": protocol_sha256,
        "selection_scope": payload["final_selection"].get("selection_scope"),
        "train_sample_count": train_binding["train_sample_count"],
        "train_room_count": train_binding["train_room_count"],
        "ordered_train_ids_sha256": train_binding["ordered_train_ids_sha256"],
        "validation_samples_opened": split_isolation.get("validation_samples_opened"),
        "test_samples_opened": split_isolation.get("test_samples_opened"),
        "dataset_identities": current_identities,
        "selector_code_identity": code_identity,
        "checkpoint_binding": checkpoint_binding,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_paired_rooms(
    room_metrics: dict[str, dict[str, dict[str, float | int]]],
    *,
    candidate: str,
    reference: str,
    metric: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    rooms = sorted(
        room
        for room, values in room_metrics.items()
        if candidate in values
        and reference in values
        and int(values[candidate]["count"]) > 0
        and int(values[reference]["count"]) > 0
    )
    empty_result = {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "difference_definition": "candidate - reference (negative is better)",
        "rooms": 0,
        "room_ids": [],
        "mean_difference": float("nan"),
        "median_difference": float("nan"),
        "candidate_better_room_fraction": float("nan"),
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
        "confidence_interval_95": [float("nan"), float("nan")],
    }
    if not rooms:
        return empty_result
    for room in rooms:
        candidate_count = int(room_metrics[room][candidate]["count"])
        reference_count = int(room_metrics[room][reference]["count"])
        if candidate_count != reference_count:
            raise RuntimeError(
                f"{room}: paired bootstrap support differs for {candidate} "
                f"({candidate_count}) and {reference} ({reference_count})"
            )
    differences = np.asarray(
        [
            float(room_metrics[room][candidate][metric])
            - float(room_metrics[room][reference][metric])
            for room in rooms
        ],
        dtype=np.float64,
    )
    if not np.isfinite(differences).all():
        raise RuntimeError("Paired bootstrap received non-finite room differences")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rooms), size=(repetitions, len(rooms)))
    bootstrap = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "difference_definition": "candidate - reference (negative is better)",
        "rooms": len(rooms),
        "room_ids": rooms,
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "candidate_better_room_fraction": float((differences < 0).mean()),
        "bootstrap_repetitions": int(repetitions),
        "seed": int(seed),
        "confidence_interval_95": [float(lower), float(upper)],
    }


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PriorBIMDA on Stanford Area_1 semantic depth subsets"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-cross-dataset-checkpoint",
        "--cross-dataset",
        dest="allow_cross_dataset_checkpoint",
        action="store_true",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--inference-seed",
        type=_nonnegative_int_arg,
        help="Seed model inference; defaults to experiment.seed (or 42 if absent)",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--batch-size", type=_positive_int_arg, default=1)
    parser.add_argument(
        "--depth-support",
        choices=("configured", "all-valid"),
        default="configured",
        help=(
            "Metric GT support: configured uses data.min_depth--max_depth from the "
            "prepared samples; all-valid reloads official regular-view z-depth and "
            "keeps every positive value except the uint16 65535 invalid sentinel"
        ),
    )
    parser.add_argument(
        "--allow-unverified-robust-comparator",
        action="store_true",
        help=(
            "Exploratory-only opt-out when robust caps have no pinned train-only "
            "selection receipt; recorded as ineligible for the formal protocol"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("--bootstrap-repetitions must be positive")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    cfg = load_config(args.config)
    universal_scale_protocol = validate_universal_scale_protocol(cfg)
    inference_seed = (
        int(args.inference_seed)
        if args.inference_seed is not None
        else int(getattr(cfg.experiment, "seed", 42))
    )
    if inference_seed < 0:
        raise ValueError("experiment.seed used for inference must be nonnegative")
    seed_everything(inference_seed)
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset.records = dataset.records[: args.max_samples]
    records_by_id = {str(record["id"]): record for record in dataset.records}
    loader = build_loader(
        dataset,
        int(args.batch_size),
        int(cfg.train.num_workers),
        shuffle=False,
    )

    checkpoint_path = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_cfg = state.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise TypeError("Checkpoint does not contain a training config")
    dataset_validation = validate_checkpoint_evaluation_dataset_provenance(
        state,
        dataset.split_provenance,
        split=args.split,
        allow_cross_dataset=args.allow_cross_dataset_checkpoint,
    )
    model_overrides = validate_checkpoint_model_config(state, cfg.model)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    model_output_max_depth = float(
        getattr(
            model,
            "output_max_depth",
            float(getattr(model, "max_depth", cfg.data.max_depth)) * 2.0,
        )
    )

    e2e_enabled = bool(model.e2e_da3_enabled)
    training_ground_truth_support = str(
        cfg.data.get("ground_truth_support", "prepared")
        if hasattr(cfg.data, "get")
        else getattr(cfg.data, "ground_truth_support", "prepared")
    )
    additive_residual_enabled = bool(
        getattr(model, "additive_residual_enabled", False)
    )
    residual_stage_ablation_enabled = bool(
        getattr(model, "attention_scale_enabled", False)
        and not getattr(model, "use_frame_residual", True)
    )
    iterative_scale_count = int(
        cfg.model.get("attention_scale", {}).get("iterative_updates", 0)
    )
    model_scale_estimator = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    model_robust_enabled = model_scale_estimator["name"] == ROBUST_LOG_CAP_SCALE_ESTIMATOR
    robust_comparator, robust_comparator_source = _resolve_robust_comparator(cfg)
    robust_selection_receipt = _robust_selection_receipt_provenance(
        cfg,
        robust_comparator,
        dataset.split_provenance,
        allow_unverified=args.allow_unverified_robust_comparator,
        checkpoint_config=checkpoint_cfg,
    )
    robust_comparison_enabled = robust_comparator is not None
    methods = [
        "raw_da3",
        "raw_da3_canonical",
        "legacy_global_scale_q45",
        "legacy_bim_direct_q45",
    ]
    if robust_comparison_enabled:
        methods.extend(["robust_global_scale", "robust_bim_direct"])
    if e2e_enabled:
        methods.extend(
            [
                "live_da3",
                "live_da3_canonical",
                "live_legacy_global_scale_q45",
                "live_legacy_bim_direct_q45",
            ]
        )
        if robust_comparison_enabled:
            methods.extend(["live_robust_global_scale", "live_robust_bim_direct"])
    methods.append("coarse")
    methods.extend(
        f"iterative_scale_round_{index + 1}"
        for index in range(iterative_scale_count)
    )
    if residual_stage_ablation_enabled:
        methods.append("scale_plus_low")
    if additive_residual_enabled:
        methods.append("proportional_refined")
    methods.append("refined")
    subset_names = (
        "all",
        "furniture",
        "non_structural",
        "bim_foreground_conflict",
        "bim_consistent",
        "bim_no_hit",
    )
    micro = {subset: {method: MetricSums() for method in methods} for subset in subset_names}
    frame_values: dict[str, dict[str, list[dict[str, float | int]]]] = {
        subset: {method: [] for method in methods} for subset in subset_names
    }
    room_sums: dict[str, dict[str, dict[str, MetricSums]]] = defaultdict(
        lambda: {subset: {method: MetricSums() for method in methods} for subset in subset_names}
    )
    support_counts = {subset: 0 for subset in subset_names}
    room_support_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {subset: 0 for subset in subset_names}
    )
    bim_envelope_micro = MetricSums()
    bim_envelope_frame_values: list[dict[str, float | int]] = []
    bim_envelope_room_sums: dict[str, MetricSums] = defaultdict(MetricSums)
    bim_envelope_gt_pixels = 0
    bim_envelope_hit_pixels = 0
    bim_envelope_frame_coverages: list[float] = []
    bim_envelope_room_support: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rows: list[dict[str, Any]] = []
    gt_support_statistics = {
        "valid_pixels": 0,
        "below_configured_min_pixels": 0,
        "within_configured_range_pixels": 0,
        "above_configured_max_pixels": 0,
    }
    evaluated_gt_min = float("inf")
    evaluated_gt_max = float("-inf")

    evaluated_samples = 0
    with torch.no_grad():
        for batch in loader:
            sample_ids = [str(value) for value in batch["sample_id"]]
            rooms = [str(value) for value in batch["region"]]
            batch_sample_count = len(sample_ids)
            if len(rooms) != batch_sample_count:
                raise RuntimeError(
                    "Collated sample_id and region metadata have different batch sizes"
                )
            missing = [
                key
                for key in (
                    "furniture_mask",
                    "non_structural_mask",
                    "structural_mask",
                    "semantic_valid",
                )
                if key not in batch
            ]
            if missing:
                raise RuntimeError(
                    f"{','.join(sample_ids)}: Stanford semantic fields are missing: {missing}"
                )
            batch = move_batch(batch, device)
            if e2e_enabled:
                # The model deliberately skips this CPU/OpenCV comparator in
                # ordinary inference unless the evaluator explicitly requests it.
                batch["request_live_bim_direct"] = True
            output = model(batch)
            if e2e_enabled and output.get("uses_live_da3") is not True:
                raise RuntimeError("E2E checkpoint did not run its live DA3 branch")
            if e2e_enabled and not torch.equal(output["coarse_depth"], output["scaled_depth"]):
                raise RuntimeError("E2E coarse_depth must be the exact live_scale alias")

            legacy_scaled, legacy_direct, legacy_estimates = _previous_baseline_batch_tensors(
                batch["base_depth"],
                batch["bim_depth"],
            )
            if "da3_metric_scale" not in batch:
                raise RuntimeError(
                    "Prepared samples lack da3_metric_scale; regenerate samples with "
                    "processing-resolution intrinsics before reporting raw DA3 metric depth"
                )
            da3_metric_scale = batch["da3_metric_scale"].reshape(-1, 1, 1, 1)
            metric_applied = batch.get("da3_metric_scale_applied")
            if metric_applied is None:
                metric_applied = torch.zeros(
                    batch_sample_count,
                    dtype=torch.bool,
                    device=device,
                )
            metric_applied = metric_applied.bool().reshape(-1)
            if bool(metric_applied.all()):
                raw_da3_metric = batch["base_depth"]
                raw_da3_canonical = batch["base_depth"] / da3_metric_scale
            elif bool((~metric_applied).all()):
                raw_da3_metric = batch["base_depth"] * da3_metric_scale
                raw_da3_canonical = batch["base_depth"]
            else:
                raise RuntimeError(
                    "A batch cannot mix canonical and focal-corrected DA3 model inputs"
                )
            robust_scaled: torch.Tensor | None = None
            robust_direct: torch.Tensor | None = None
            robust_estimates: list[ScaleEstimate] | None = None
            if robust_comparator is not None:
                robust_scaled, robust_direct, robust_estimates = _robust_baseline_batch_tensors(
                    batch["base_depth"],
                    batch["bim_depth"],
                    robust_comparator,
                )
            predictions: dict[str, torch.Tensor] = {
                "raw_da3": raw_da3_metric,
                "raw_da3_canonical": raw_da3_canonical,
                "legacy_global_scale_q45": legacy_scaled,
                "legacy_bim_direct_q45": legacy_direct,
                "coarse": output["coarse_depth"],
                "refined": output["depth"],
            }
            if iterative_scale_count:
                iteration_log_scales = output.get("attention_iteration_log_scales")
                expected_shape = (batch_sample_count, iterative_scale_count, 1, 1)
                if iteration_log_scales is None or tuple(iteration_log_scales.shape) != (
                    expected_shape
                ):
                    raise RuntimeError(
                        "Iterative scale checkpoint must expose log scales with shape "
                        f"{expected_shape}; got "
                        f"{None if iteration_log_scales is None else tuple(iteration_log_scales.shape)}"
                    )
                for iteration_index in range(iterative_scale_count):
                    predictions[f"iterative_scale_round_{iteration_index + 1}"] = (
                        output["base_depth"]
                        * iteration_log_scales[:, iteration_index].exp().unsqueeze(1)
                    ).clamp(
                        min=1e-3,
                        max=model_output_max_depth,
                    )
            if residual_stage_ablation_enabled:
                predictions["scale_plus_low"] = (
                    output["refinement_anchor_depth"]
                    * torch.exp(output["low_log_residual"])
                ).clamp(
                    min=1e-3,
                    max=model_output_max_depth,
                )
            if additive_residual_enabled:
                # Isolate the additive head under the exact same model-output
                # safety bound as the final prediction. The public 0.2--5 m
                # protocol restricts GT support; it does not clip predictions
                # to that interval.
                predictions["proportional_refined"] = output[
                    "proportional_depth"
                ].clamp(
                    min=1e-3,
                    max=model_output_max_depth,
                )
            if robust_scaled is not None and robust_direct is not None:
                predictions.update(
                    robust_global_scale=robust_scaled,
                    robust_bim_direct=robust_direct,
                )

            live_legacy_estimates: list[ScaleEstimate] | None = None
            live_robust_estimates: list[ScaleEstimate] | None = None
            if e2e_enabled:
                live_base = output["base_depth"]
                (
                    live_legacy_scaled,
                    live_legacy_direct,
                    live_legacy_estimates,
                ) = _previous_baseline_batch_tensors(
                    live_base,
                    batch["bim_depth"],
                )
                predictions.update(
                    live_da3=live_base * da3_metric_scale,
                    live_da3_canonical=live_base,
                    live_legacy_global_scale_q45=live_legacy_scaled,
                    live_legacy_bim_direct_q45=live_legacy_direct,
                )
                if robust_comparator is not None:
                    (
                        live_robust_scaled,
                        live_robust_direct,
                        live_robust_estimates,
                    ) = _robust_baseline_batch_tensors(
                        live_base,
                        batch["bim_depth"],
                        robust_comparator,
                    )
                    predictions.update(
                        live_robust_global_scale=live_robust_scaled,
                        live_robust_bim_direct=live_robust_direct,
                    )

                configured_live_key = (
                    "live_robust_bim_direct"
                    if model_robust_enabled
                    else "live_legacy_bim_direct_q45"
                )
                generic_live = output.get("live_bim_direct")
                configured_live = output.get(configured_live_key)
                if configured_live is None and not model_robust_enabled:
                    configured_live = generic_live
                if configured_live is None:
                    raise RuntimeError(
                        "E2E checkpoint did not return its explicitly requested "
                        f"{configured_live_key} comparator"
                    )
                expected_live = (
                    predictions["live_robust_bim_direct"]
                    if model_robust_enabled
                    else predictions["live_legacy_bim_direct_q45"]
                )
                if not torch.allclose(
                    configured_live,
                    expected_live,
                    rtol=1e-6,
                    atol=1e-6,
                ):
                    raise RuntimeError(
                        f"Model output {configured_live_key} disagrees with the "
                        "authoritative configured BIM-direct comparator"
                    )
                if model_robust_enabled:
                    # The robust model output, not a generic alias, is the main
                    # live comparator used in its evaluation table.
                    predictions["live_robust_bim_direct"] = output["live_robust_bim_direct"]

            metric_gt = batch["gt_depth"]
            metric_gt_valid = batch["gt_valid"] > 0
            if (
                args.depth_support == "all-valid"
                and training_ground_truth_support != "official_all_valid"
            ):
                all_valid_depths: list[np.ndarray] = []
                all_valid_masks: list[np.ndarray] = []
                target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
                for sample_id in sample_ids:
                    depth, valid = _load_all_valid_regular_depth(
                        records_by_id[sample_id],
                        target_shape,
                    )
                    all_valid_depths.append(depth[None])
                    all_valid_masks.append(valid[None])
                metric_gt = torch.from_numpy(np.stack(all_valid_depths)).to(device=device)
                metric_gt_valid = torch.from_numpy(np.stack(all_valid_masks)).to(device=device)

            batched_tensors = {
                "gt_depth": metric_gt,
                "gt_valid": metric_gt_valid,
                "bim_depth": batch["bim_depth"],
                "bim_valid": batch["bim_valid"],
                "furniture_mask": batch["furniture_mask"],
                "non_structural_mask": batch["non_structural_mask"],
                **predictions,
            }
            inconsistent_batch_sizes = {
                name: int(tensor.shape[0])
                for name, tensor in batched_tensors.items()
                if tensor.ndim < 1 or tensor.shape[0] != batch_sample_count
            }
            if inconsistent_batch_sizes:
                raise RuntimeError(
                    "Model/data tensors do not match the collated metadata batch size "
                    f"{batch_sample_count}: {inconsistent_batch_sizes}"
                )

            for batch_index, (sample_id, room) in enumerate(zip(sample_ids, rooms)):
                evaluated_samples += 1
                gt = metric_gt[batch_index : batch_index + 1]
                gt_valid = metric_gt_valid[batch_index : batch_index + 1]
                valid_gt_values = gt[gt_valid]
                valid_gt_count = int(valid_gt_values.numel())
                gt_support_statistics["valid_pixels"] += valid_gt_count
                gt_support_statistics["below_configured_min_pixels"] += int(
                    (valid_gt_values < float(cfg.data.min_depth)).sum().item()
                )
                gt_support_statistics["within_configured_range_pixels"] += int(
                    (
                        (valid_gt_values >= float(cfg.data.min_depth))
                        & (valid_gt_values <= float(cfg.data.max_depth))
                    ).sum().item()
                )
                gt_support_statistics["above_configured_max_pixels"] += int(
                    (valid_gt_values > float(cfg.data.max_depth)).sum().item()
                )
                if valid_gt_count:
                    evaluated_gt_min = min(
                        evaluated_gt_min,
                        float(valid_gt_values.min().item()),
                    )
                    evaluated_gt_max = max(
                        evaluated_gt_max,
                        float(valid_gt_values.max().item()),
                    )
                bim_depth = batch["bim_depth"][batch_index : batch_index + 1]
                bim_valid = (batch["bim_valid"][batch_index : batch_index + 1] > 0) & (
                    bim_depth > 0
                )
                tolerance = torch.maximum(
                    torch.full_like(bim_depth, 0.10),
                    0.05 * bim_depth,
                )
                subsets = {
                    "all": gt_valid,
                    "furniture": gt_valid
                    & (batch["furniture_mask"][batch_index : batch_index + 1] > 0),
                    "non_structural": gt_valid
                    & (batch["non_structural_mask"][batch_index : batch_index + 1] > 0),
                    "bim_foreground_conflict": (
                        gt_valid & bim_valid & (gt < bim_depth - tolerance)
                    ),
                    "bim_consistent": (
                        gt_valid & bim_valid & ((gt - bim_depth).abs() <= tolerance)
                    ),
                    "bim_no_hit": gt_valid & ~bim_valid,
                }
                sample_predictions = {
                    method: prediction[batch_index : batch_index + 1]
                    for method, prediction in predictions.items()
                }
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "room": room,
                    "camera_uuid": str(records_by_id[sample_id].get("camera_uuid", "")),
                    **_scale_estimate_columns(
                        "legacy",
                        legacy_estimates[batch_index],
                    ),
                }
                if robust_estimates is not None:
                    row.update(
                        _scale_estimate_columns(
                            "robust",
                            robust_estimates[batch_index],
                        )
                    )
                if live_legacy_estimates is not None:
                    row.update(
                        _scale_estimate_columns(
                            "live_legacy",
                            live_legacy_estimates[batch_index],
                        )
                    )
                if live_robust_estimates is not None:
                    row.update(
                        _scale_estimate_columns(
                            "live_robust",
                            live_robust_estimates[batch_index],
                        )
                    )
                for subset, mask in subsets.items():
                    expected_count = int(mask.sum().item())
                    row[f"{subset}_gt_pixels"] = expected_count
                    support_counts[subset] += expected_count
                    room_support_counts[room][subset] += expected_count
                    frame_metrics: dict[str, dict[str, float | int]] = {}
                    for method, prediction in sample_predictions.items():
                        frame_sum = MetricSums()
                        metric_context = f"{sample_id}/{subset}/{method}"
                        frame_sum.update(
                            prediction,
                            gt,
                            mask,
                            prediction_name=method,
                            context=metric_context,
                        )
                        metrics = frame_sum.compute()
                        frame_metrics[method] = metrics
                        micro[subset][method].update(
                            prediction,
                            gt,
                            mask,
                            prediction_name=method,
                            context=metric_context,
                        )
                        room_sums[room][subset][method].update(
                            prediction,
                            gt,
                            mask,
                            prediction_name=method,
                            context=metric_context,
                        )
                        if int(metrics["count"]) > 0:
                            frame_values[subset][method].append(metrics)
                        for metric in ("abs_rel", "mae", "rmse", "count"):
                            row[f"{subset}_{method}_{metric}"] = metrics[metric]
                    _assert_comparable_counts(
                        frame_metrics,
                        expected_count,
                        context=f"{sample_id}/{subset}",
                    )

                envelope_support = gt_valid & bim_valid
                envelope_gt_count = int(gt_valid.sum().item())
                envelope_hit_count = int(envelope_support.sum().item())
                envelope_context = f"{sample_id}/bim_envelope_hit"
                envelope_frame_sum = MetricSums()
                envelope_frame_sum.update(
                    bim_depth,
                    gt,
                    envelope_support,
                    prediction_name="bim_envelope",
                    context=envelope_context,
                )
                envelope_metrics = envelope_frame_sum.compute()
                bim_envelope_micro.update(
                    bim_depth,
                    gt,
                    envelope_support,
                    prediction_name="bim_envelope",
                    context=envelope_context,
                )
                bim_envelope_room_sums[room].update(
                    bim_depth,
                    gt,
                    envelope_support,
                    prediction_name="bim_envelope",
                    context=envelope_context,
                )
                if envelope_hit_count:
                    bim_envelope_frame_values.append(envelope_metrics)
                bim_envelope_gt_pixels += envelope_gt_count
                bim_envelope_hit_pixels += envelope_hit_count
                bim_envelope_room_support[room][0] += envelope_hit_count
                bim_envelope_room_support[room][1] += envelope_gt_count
                if envelope_gt_count:
                    bim_envelope_frame_coverages.append(
                        _coverage_fraction(envelope_hit_count, envelope_gt_count)
                    )
                row["bim_envelope_gt_pixels"] = envelope_gt_count
                row["bim_envelope_hit_pixels"] = envelope_hit_count
                row["bim_envelope_coverage"] = _coverage_fraction(
                    envelope_hit_count,
                    envelope_gt_count,
                )
                for metric in ("abs_rel", "mae", "rmse", "count"):
                    row[f"bim_envelope_hit_{metric}"] = envelope_metrics[metric]
                rows.append(row)
                if (
                    evaluated_samples == 1
                    or evaluated_samples % args.log_every == 0
                    or evaluated_samples == len(dataset)
                ):
                    print(
                        f"[{evaluated_samples}/{len(dataset)}] {sample_id} all AbsRel "
                        f"{row['all_raw_da3_abs_rel']:.4f} -> "
                        f"{row['all_refined_abs_rel']:.4f}",
                        flush=True,
                    )

    room_metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for room in sorted(room_sums):
        room_metrics[room] = {
            subset: {method: room_sums[room][subset][method].compute() for method in methods}
            for subset in subset_names
        }
        for subset in subset_names:
            _assert_comparable_counts(
                room_metrics[room][subset],
                room_support_counts[room][subset],
                context=f"room={room}/{subset}",
            )
    aggregate: dict[str, dict[str, Any]] = {}
    for subset in subset_names:
        aggregate[subset] = {}
        for method in methods:
            aggregate[subset][method] = {
                "pixel_micro": micro[subset][method].compute(),
                "frame_macro": _macro(frame_values[subset][method]),
                "room_macro": _macro(
                    [room_metrics[room][subset][method] for room in sorted(room_metrics)]
                ),
            }
        _assert_comparable_counts(
            {method: aggregate[subset][method]["pixel_micro"] for method in methods},
            support_counts[subset],
            context=f"aggregate/{subset}/pixel_micro",
        )

    bim_envelope_room_metrics = {
        room: bim_envelope_room_sums[room].compute() for room in sorted(bim_envelope_room_sums)
    }
    bim_envelope_room_coverages = [
        _coverage_fraction(hit_pixels, gt_pixels)
        for hit_pixels, gt_pixels in bim_envelope_room_support.values()
        if gt_pixels > 0
    ]
    standalone_bim_envelope = {
        "support_definition": "gt_valid & bim_valid; excluded from fixed-support method comparisons",
        "coverage": {
            "pixel_micro": {
                "fraction": _coverage_fraction(
                    bim_envelope_hit_pixels,
                    bim_envelope_gt_pixels,
                ),
                "hit_pixels": bim_envelope_hit_pixels,
                "gt_pixels": bim_envelope_gt_pixels,
            },
            "frame_macro": {
                "fraction": (
                    float(np.mean(bim_envelope_frame_coverages))
                    if bim_envelope_frame_coverages
                    else float("nan")
                ),
                "groups": len(bim_envelope_frame_coverages),
            },
            "room_macro": {
                "fraction": (
                    float(np.mean(bim_envelope_room_coverages))
                    if bim_envelope_room_coverages
                    else float("nan")
                ),
                "groups": len(bim_envelope_room_coverages),
            },
        },
        "metrics_on_hit_support": {
            "pixel_micro": bim_envelope_micro.compute(),
            "frame_macro": _macro(bim_envelope_frame_values),
            "room_macro": _macro(list(bim_envelope_room_metrics.values())),
        },
        "per_room": {
            room: {
                "coverage": {
                    "fraction": _coverage_fraction(
                        bim_envelope_room_support[room][0],
                        bim_envelope_room_support[room][1],
                    ),
                    "hit_pixels": bim_envelope_room_support[room][0],
                    "gt_pixels": bim_envelope_room_support[room][1],
                },
                "metrics_on_hit_support": bim_envelope_room_metrics[room],
            }
            for room in sorted(bim_envelope_room_metrics)
        },
    }

    if robust_comparison_enabled:
        primary_bim_direct = "live_robust_bim_direct" if e2e_enabled else "robust_bim_direct"
    else:
        primary_bim_direct = "legacy_bim_direct_q45"
    comparison_references = [primary_bim_direct]
    if e2e_enabled and robust_comparison_enabled:
        comparison_references.append("robust_bim_direct")
    comparison_references = list(dict.fromkeys(comparison_references))
    bootstrap_by_reference = {
        reference: {
            subset: {
                metric: _bootstrap_paired_rooms(
                    {room: room_metrics[room][subset] for room in room_metrics},
                    candidate="refined",
                    reference=reference,
                    metric=metric,
                    seed=args.bootstrap_seed,
                    repetitions=args.bootstrap_repetitions,
                )
                for metric in ("abs_rel", "mae")
            }
            for subset in ("all", "furniture", "bim_foreground_conflict")
        }
        for reference in comparison_references
    }
    learned_beats_by_reference = {
        reference: {
            subset: {
                aggregation: _beats_on_absrel_and_mae(
                    aggregate[subset]["refined"][aggregation],
                    aggregate[subset][reference][aggregation],
                )
                for aggregation in ("pixel_micro", "frame_macro", "room_macro")
            }
            for subset in ("all", "furniture", "bim_foreground_conflict")
        }
        for reference in comparison_references
    }
    bootstrap = bootstrap_by_reference[primary_bim_direct]
    learned_beats_direct = learned_beats_by_reference[primary_bim_direct]

    output_dir = args.output or (
        resolve_project_path(cfg, cfg.experiment.output_dir) / f"stanford_{args.split}_evaluation"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = output_dir / "per_frame.csv"
    if not rows:
        raise RuntimeError("No Stanford samples were evaluated")
    with per_frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 4,
        "protocol": "stanford-area1-fixed-envelope-depth-v4-da3-focal-metric",
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_training_config": checkpoint_cfg.get("config_path"),
        "checkpoint_provenance": state.get("provenance", {}),
        "evaluation_config": str(Path(cfg.config_path).resolve()),
        "split": args.split,
        "sample_count": len(dataset),
        "rooms": sorted(room_metrics),
        "runtime": {
            "batch_size": int(args.batch_size),
            "num_workers": int(cfg.train.num_workers),
            "inference_seed": inference_seed,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "unverified_robust_comparator_opt_out": bool(args.allow_unverified_robust_comparator),
        },
        "dataset": {
            "split_provenance": dataset.split_provenance,
            "checkpoint_validation": dataset_validation,
            "cross_dataset_checkpoint_opt_in": args.allow_cross_dataset_checkpoint,
        },
        "model_config_overrides": model_overrides,
        "universal_scale_protocol": universal_scale_protocol,
        "scale_estimators": {
            "model_input": model_scale_estimator,
            "robust_comparator": {
                "status": ("available" if robust_comparison_enabled else "unavailable"),
                "source": robust_comparator_source,
                "parameters": robust_comparator,
                "selection_receipt": robust_selection_receipt,
            },
            "primary_bim_direct_reference": primary_bim_direct,
        },
        "depth_protocol_m": (
            [float(cfg.data.min_depth), float(cfg.data.max_depth)]
            if args.depth_support == "configured"
            else [0.0, None]
        ),
        "training_depth_protocol_m": (
            [0.0, None]
            if training_ground_truth_support == "official_all_valid"
            else [float(cfg.data.min_depth), float(cfg.data.max_depth)]
        ),
        "training_ground_truth_support": training_ground_truth_support,
        "model_output_max_depth_m": model_output_max_depth,
        "ground_truth_support": {
            "mode": args.depth_support,
            "definition": (
                "prepared official regular-view z-depth within the configured range"
                if args.depth_support == "configured"
                else (
                    "all positive official regular-view z-depth values; uint16 0 and "
                    "65535 are invalid; no metric depth cutoff"
                )
            ),
            "source_encoding": "official regular-view z-depth uint16/512 metres",
            "observed_valid_depth_range_m": [
                (evaluated_gt_min if np.isfinite(evaluated_gt_min) else None),
                (evaluated_gt_max if np.isfinite(evaluated_gt_max) else None),
            ],
            "pixel_counts": gt_support_statistics,
            "status": (
                "primary configured benchmark"
                if args.depth_support == "configured"
                else (
                    "primary all-valid benchmark"
                    if training_ground_truth_support == "official_all_valid"
                    else "exploratory out-of-training-support diagnostic"
                )
            ),
        },
        "subset_definitions": {
            "all": "all valid official z-depth pixels",
            "furniture": "table/chair/sofa/bookcase semantic pixels",
            "non_structural": "all known non-envelope semantic pixels",
            "bim_foreground_conflict": ("GT is closer than BIM by max(0.10m, 5% of BIM depth)"),
            "bim_consistent": "absolute GT-BIM difference <= max(0.10m, 5%)",
            "bim_no_hit": "valid GT with no fixed-envelope BIM ray hit",
        },
        "method_definitions": {
            "raw_da3": (
                "standalone frozen DA3METRIC output converted to metric depth as "
                "cached canonical depth * mean(fx,fy)/300; no BIM or GT scaling"
            ),
            "raw_da3_canonical": (
                "cached standalone DA3METRIC canonical-focal output retained only "
                "to audit historical results that mislabeled it as raw metric depth"
            ),
            "legacy_global_scale_q45": ("historical q=.45 BIM scale applied to cached DA3"),
            "legacy_bim_direct_q45": ("historical q=.45 scale plus registered local correction"),
            **(
                {
                    "robust_global_scale": (
                        "frozen train-selected robust log-cap BIM scale applied to cached DA3"
                    ),
                    "robust_bim_direct": (
                        "robust global scale plus the unchanged registered local correction"
                    ),
                }
                if robust_comparison_enabled
                else {}
            ),
            **(
                {
                    "live_da3": (
                        "live standalone DA3METRIC canonical output converted using "
                        "the same per-frame mean(fx,fy)/300 factor"
                    ),
                    "live_da3_canonical": (
                        "unscaled live standalone DA3METRIC canonical-focal output"
                    ),
                }
                if e2e_enabled
                else {}
            ),
            "bim_envelope": (
                "raw fixed-envelope render; standalone hit-support metrics and coverage only"
            ),
            "coarse": (
                "model metric-scale input before learned local refinement; exact alias "
                "of live_scale for E2E checkpoints"
            ),
            **{
                f"iterative_scale_round_{index + 1}": (
                    f"shared-weight residual-conditioned BIM/DA3 scale estimate after "
                    f"iteration {index + 1}, before local refinement"
                )
                for index in range(iterative_scale_count)
            },
            **(
                {
                    "scale_plus_low": (
                        "attention-scale output after the routed low-frequency log "
                        "residual but before the detail log residual; frame residual "
                        "is disabled for this staged ablation"
                    )
                }
                if residual_stage_ablation_enabled
                else {}
            ),
            **(
                {
                    "proportional_refined": (
                        "scale output after multiplicative low/detail log residual but "
                        "before the pixel-level additive metric residual, with the same "
                        "model-output safety bound as the final prediction"
                    )
                }
                if additive_residual_enabled
                else {}
            ),
            "refined": "learned BIM/RGB refinement",
            **(
                {
                    "live_da3": "checkpoint's live partially trainable DA3 output",
                    "live_legacy_global_scale_q45": (
                        "historical q=.45 BIM scale applied to live DA3"
                    ),
                    "live_legacy_bim_direct_q45": (
                        "historical q=.45 scale and local correction applied to live DA3"
                    ),
                    **(
                        {
                            "live_robust_global_scale": (
                                "frozen robust comparator applied to live DA3"
                            ),
                            "live_robust_bim_direct": (
                                "frozen robust scale and unchanged local correction "
                                "applied to live DA3"
                            ),
                        }
                        if robust_comparison_enabled
                        else {}
                    ),
                }
                if e2e_enabled
                else {}
            ),
        },
        "comparison_support_policy": {
            "comparable_methods": methods,
            "rule": (
                "Every comparable method uses the exact declared GT subset; any "
                "non-finite or non-positive prediction on support is a fatal error"
            ),
            "bim_envelope_excluded": True,
        },
        "method_relationships": {
            "e2e_coarse_scale_estimator": (model_scale_estimator["name"] if e2e_enabled else None),
        },
        "schema_aliases_not_in_comparison_table": {
            "global_scale": "legacy_global_scale_q45",
            "bim_direct": "legacy_bim_direct_q45",
            **(
                {
                    "live_scale": (
                        "live_robust_global_scale"
                        if model_robust_enabled
                        else "live_legacy_global_scale_q45"
                    ),
                    "live_bim_direct": (
                        "live_robust_bim_direct"
                        if model_robust_enabled
                        else "live_legacy_bim_direct_q45"
                    ),
                }
                if e2e_enabled
                else {}
            ),
        },
        "previous_scale_parameters": PREVIOUS_FIXED_PARAMETERS,
        "aggregates": aggregate,
        "per_room": room_metrics,
        "standalone_bim_envelope": standalone_bim_envelope,
        "paired_room_bootstrap": bootstrap,
        "paired_room_bootstrap_by_reference": bootstrap_by_reference,
        "learned_beats_primary_bim_direct_absrel_and_mae": learned_beats_direct,
        "learned_beats_bim_direct_by_reference": learned_beats_by_reference,
        "learned_beats_bim_direct_absrel_and_mae": learned_beats_direct,
        "per_frame_csv": str(per_frame_path),
        "per_frame_csv_sha256": _sha256(per_frame_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
