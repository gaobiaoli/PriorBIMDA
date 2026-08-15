#!/usr/bin/env python3
"""Ablate every active factor in the deterministic universal BIM baseline.

The study is deliberately validation-only.  Predictions use cached DA3 and BIM
arrays; GT and semantic masks are read only after every variant prediction has
been constructed.  This prevents the diagnostic from becoming another scale
selection procedure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bim_priorda3.baselines import (
    PREVIOUS_FIXED_PARAMETERS,
    configured_scale_and_local_features,
    estimate_robust_bim_scale,
    resolve_scale_estimator_config,
)
from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.scale_protocol import validate_universal_scale_protocol

VARIANTS = (
    "full",
    "scale_only",
    "no_q25_cap",
    "wide_ratio_bounds",
    "min_samples_1",
    "no_consistency_gate",
    "no_edge_gate",
    "no_gaussian_propagation",
    "no_support_cutoff",
    "alpha_1_0",
)

VARIANT_FACTORS = {
    "full": "registered universal BIM-direct",
    "scale_only": "remove the complete local-correction stage",
    "no_q25_cap": "remove Q25+0.05 upper cap (log-Q45 remains)",
    "wide_ratio_bounds": "replace 0.2<ratio<5.0 with 1e-6<ratio<1e6",
    "min_samples_1": "replace the 100-ratio fallback threshold with one ratio",
    "no_consistency_gate": "remove |log(BIM/scaled DA3)|<=0.10 support gate",
    "no_edge_gate": "remove Sobel BIM-depth gradient rejection (<0.25)",
    "no_gaussian_propagation": "remove sigma=64 normalized spatial propagation",
    "no_support_cutoff": "remove smoothed-support denominator cutoff (<0.05)",
    "alpha_1_0": "replace local correction multiplier 1.25 with 1.0",
}


@dataclass
class Sums:
    count: int = 0
    abs_rel: float = 0.0
    abs_error: float = 0.0
    squared_error: float = 0.0

    def add(self, other: Sums) -> None:
        self.count += other.count
        self.abs_rel += other.abs_rel
        self.abs_error += other.abs_error
        self.squared_error += other.squared_error

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "abs_rel": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
            }
        return {
            "count": self.count,
            "abs_rel": self.abs_rel / self.count,
            "mae": self.abs_error / self.count,
            "rmse": math.sqrt(self.squared_error / self.count),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only ablation of deterministic BIM-direct factors"
    )
    parser.add_argument(
        "--slabim-config",
        type=Path,
        default=Path("configs/slabim.yaml"),
    )
    parser.add_argument(
        "--stanford-config",
        type=Path,
        default=Path("configs/stanford_area1.yaml"),
    )
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/deterministic_baseline_ablation/summary.json"),
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.bootstrap_repetitions < 1:
        parser.error("--bootstrap-repetitions must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _custom_scale(
    base: np.ndarray,
    bim: np.ndarray,
    parameters: dict[str, Any],
    *,
    q25_log_cap: float | None = None,
    ratio_min: float | None = None,
    ratio_max: float | None = None,
    min_samples: int | None = None,
) -> tuple[np.ndarray, Any]:
    estimate = estimate_robust_bim_scale(
        base,
        bim,
        q10_log_cap=float(parameters["q10_log_cap"]),
        q25_log_cap=(
            float(parameters["q25_log_cap"]) if q25_log_cap is None else float(q25_log_cap)
        ),
        ratio_min=float(parameters["ratio_min"] if ratio_min is None else ratio_min),
        ratio_max=float(parameters["ratio_max"] if ratio_max is None else ratio_max),
        min_samples=int(parameters["min_samples"] if min_samples is None else min_samples),
    )
    return (base * estimate.scale).astype(np.float32), estimate


def _custom_local(
    scaled: np.ndarray,
    bim: np.ndarray,
    *,
    consistency_gate: bool = True,
    edge_gate: bool = True,
    gaussian_propagation: bool = True,
    support_cutoff: bool = True,
    alpha: float = 1.25,
) -> np.ndarray:
    consistency = float(PREVIOUS_FIXED_PARAMETERS["consistency_log_threshold"])
    residual = np.log(np.maximum(bim, 1e-6)) - np.log(np.maximum(scaled, 1e-6))
    valid = np.isfinite(bim) & (bim > 0) & np.isfinite(residual)
    if consistency_gate:
        valid &= np.abs(residual) <= consistency
    if edge_gate:
        safe_bim = np.nan_to_num(bim, nan=0.0)
        gradient_x = cv2.Sobel(safe_bim, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(safe_bim, cv2.CV_32F, 0, 1, ksize=3)
        valid &= np.hypot(gradient_x, gradient_y) < 0.25
    if gaussian_propagation:
        sigma = float(PREVIOUS_FIXED_PARAMETERS["smoothing_sigma"])
        numerator = cv2.GaussianBlur(
            np.where(valid, residual, 0.0).astype(np.float32),
            (0, 0),
            sigma,
        )
        denominator = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigma)
        field = numerator / np.maximum(denominator, 1e-4)
        if support_cutoff:
            field[denominator < 0.05] = 0.0
    else:
        field = np.where(valid, residual, 0.0)
    field = np.clip(field, -consistency, consistency).astype(np.float32)
    return (scaled * np.exp(float(alpha) * field)).astype(np.float32)


def variant_predictions(
    base: np.ndarray,
    bim: np.ndarray,
    scale_parameters: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct every prediction without consulting GT or semantic masks."""

    scaled, full, _, _, estimate = configured_scale_and_local_features(
        base,
        bim,
        scale_parameters,
    )
    no_cap_scaled, _ = _custom_scale(
        base,
        bim,
        scale_parameters,
        q25_log_cap=float("inf"),
    )
    wide_scaled, _ = _custom_scale(
        base,
        bim,
        scale_parameters,
        ratio_min=1e-6,
        ratio_max=1e6,
    )
    min_one_scaled, _ = _custom_scale(
        base,
        bim,
        scale_parameters,
        min_samples=1,
    )
    predictions = {
        "full": full,
        "scale_only": scaled,
        "no_q25_cap": _custom_local(no_cap_scaled, bim),
        "wide_ratio_bounds": _custom_local(wide_scaled, bim),
        "min_samples_1": _custom_local(min_one_scaled, bim),
        "no_consistency_gate": _custom_local(scaled, bim, consistency_gate=False),
        "no_edge_gate": _custom_local(scaled, bim, edge_gate=False),
        "no_gaussian_propagation": _custom_local(
            scaled,
            bim,
            gaussian_propagation=False,
        ),
        "no_support_cutoff": _custom_local(scaled, bim, support_cutoff=False),
        "alpha_1_0": _custom_local(scaled, bim, alpha=1.0),
    }
    return predictions, {
        "support_count": estimate.support_count,
        "fallback": estimate.fallback,
        "q10_cap_triggered": estimate.q10_cap_triggered,
        "q25_cap_triggered": estimate.q25_cap_triggered,
    }


def _sums(prediction: np.ndarray, gt: np.ndarray, support: np.ndarray) -> Sums:
    count = int(support.sum())
    if count == 0:
        return Sums()
    selected = prediction[support]
    if not np.isfinite(selected).all() or np.any(selected <= 0):
        raise RuntimeError("A deterministic baseline produced an invalid supported prediction")
    target = gt[support]
    error = np.abs(selected.astype(np.float64) - target.astype(np.float64))
    return Sums(
        count=count,
        abs_rel=float(np.sum(error / target, dtype=np.float64)),
        abs_error=float(np.sum(error, dtype=np.float64)),
        squared_error=float(np.sum(error * error, dtype=np.float64)),
    )


def _evaluate_record(
    record: dict[str, Any],
    scale_parameters: dict[str, Any],
    depth_min: float,
    depth_max: float,
) -> dict[str, Any]:
    with np.load(record["sample"]) as sample:
        base = sample["base_depth"].astype(np.float32)
        bim = sample["bim_depth"].astype(np.float32)
        bim_valid = sample["bim_valid"] > 0
        bim = np.where(bim_valid, bim, 0.0).astype(np.float32)
        predictions, scale_diagnostics = variant_predictions(base, bim, scale_parameters)

        # GT-dependent data is deliberately accessed after all predictions exist.
        gt = sample["gt_depth"].astype(np.float32)
        gt_valid = sample["gt_valid"] > 0
        fixed = gt_valid & np.isfinite(gt) & (gt >= depth_min) & (gt <= depth_max)
        subsets = {"all": fixed}
        if "furniture_mask" in sample:
            subsets["furniture"] = fixed & (sample["furniture_mask"] > 0)
            tolerance = np.maximum(0.10, 0.05 * bim)
            subsets["bim_foreground_conflict"] = fixed & bim_valid & (gt < bim - tolerance)
    return {
        "sample_id": str(record["id"]),
        "group": str(record["region"]),
        "metrics": {
            subset: {
                variant: _sums(prediction, gt, support)
                for variant, prediction in predictions.items()
            }
            for subset, support in subsets.items()
        },
        "scale_diagnostics": scale_diagnostics,
    }


def _paired_group_bootstrap(
    group_metrics: dict[str, dict[str, dict[str, dict[str, float | int]]]],
    subset: str,
    variant: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    groups = sorted(
        group
        for group, subsets in group_metrics.items()
        if subset in subsets
        and int(subsets[subset][variant]["count"]) > 0
        and int(subsets[subset]["full"]["count"]) > 0
    )
    differences = np.asarray(
        [
            float(group_metrics[group][subset][variant]["abs_rel"])
            - float(group_metrics[group][subset]["full"]["abs_rel"])
            for group in groups
        ],
        dtype=np.float64,
    )
    if not groups:
        return {"groups": 0, "mean_difference": None, "confidence_interval_95": [None, None]}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    bootstrap = differences[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "groups": len(groups),
        "group_ids": groups,
        "difference": "variant - full; positive means the removed/changed factor was helpful",
        "mean_difference": float(differences.mean()),
        "confidence_interval_95": [float(lower), float(upper)],
        "variant_better_group_fraction": float((differences < 0).mean()),
        "bootstrap_repetitions": repetitions,
        "seed": seed,
    }


def evaluate_dataset(
    name: str,
    config_path: Path,
    split: str,
    workers: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    protocol = validate_universal_scale_protocol(cfg)
    scale_parameters = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    dataset = BIMDepthDataset(cfg, split, augment=False, require_ground_truth=True)
    records = list(dataset.records)
    evaluate = lambda record: _evaluate_record(
        record,
        scale_parameters,
        float(cfg.data.min_depth),
        float(cfg.data.max_depth),
    )
    if workers == 1:
        rows = [evaluate(record) for record in records]
    else:
        previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(evaluate, records))
        finally:
            cv2.setNumThreads(previous_threads)

    pixel: dict[str, dict[str, Sums]] = defaultdict(lambda: defaultdict(Sums))
    frame: dict[str, dict[str, list[dict[str, float | int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_sums: dict[str, dict[str, dict[str, Sums]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Sums))
    )
    diagnostics = defaultdict(int)
    support_counts: list[int] = []
    for row in rows:
        diagnostics["fallback_frames"] += int(row["scale_diagnostics"]["fallback"])
        diagnostics["q10_cap_triggered_frames"] += int(
            row["scale_diagnostics"]["q10_cap_triggered"]
        )
        diagnostics["q25_cap_triggered_frames"] += int(
            row["scale_diagnostics"]["q25_cap_triggered"]
        )
        support_counts.append(int(row["scale_diagnostics"]["support_count"]))
        for subset, variants in row["metrics"].items():
            for variant, sums in variants.items():
                pixel[subset][variant].add(sums)
                group_sums[row["group"]][subset][variant].add(sums)
                if sums.count:
                    frame[subset][variant].append(sums.metrics())

    group_metrics = {
        group: {
            subset: {variant: sums.metrics() for variant, sums in variants.items()}
            for subset, variants in subsets.items()
        }
        for group, subsets in group_sums.items()
    }
    aggregates: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for subset, variants in pixel.items():
        aggregates[subset] = {}
        bootstrap[subset] = {}
        for variant, sums in variants.items():
            frame_rows = frame[subset][variant]
            eligible_groups = [
                group_metrics[group][subset][variant]
                for group in sorted(group_metrics)
                if subset in group_metrics[group]
                and int(group_metrics[group][subset][variant]["count"]) > 0
            ]
            aggregates[subset][variant] = {
                "factor_change": VARIANT_FACTORS[variant],
                "pixel_micro": sums.metrics(),
                "frame_macro": {
                    "frames": len(frame_rows),
                    **{
                        metric: float(np.mean([float(row[metric]) for row in frame_rows]))
                        for metric in ("abs_rel", "mae", "rmse")
                    },
                },
                "group_macro": {
                    "groups": len(eligible_groups),
                    **{
                        metric: float(np.mean([float(row[metric]) for row in eligible_groups]))
                        for metric in ("abs_rel", "mae", "rmse")
                    },
                },
            }
            if variant != "full":
                bootstrap[subset][variant] = _paired_group_bootstrap(
                    group_metrics,
                    subset,
                    variant,
                    bootstrap_repetitions,
                    bootstrap_seed,
                )

    return {
        "dataset": name,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "split": split,
        "samples": len(records),
        "groups": len(group_metrics),
        "split_provenance": dataset.split_provenance,
        "universal_scale_protocol": protocol,
        "scale_diagnostics": {
            **dict(diagnostics),
            "minimum_support_count": min(support_counts),
            "median_support_count": float(np.median(support_counts)),
        },
        "aggregates": aggregates,
        "per_group": group_metrics,
        "paired_group_bootstrap_abs_rel": bootstrap,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    datasets = {
        "slabim": evaluate_dataset(
            "SLABIM",
            args.slabim_config,
            args.split,
            args.workers,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        ),
        "stanford_area1": evaluate_dataset(
            "2D-3D-S Area 1",
            args.stanford_config,
            args.split,
            args.workers,
            args.bootstrap_repetitions,
            args.bootstrap_seed,
        ),
    }
    output = {
        "schema_version": 1,
        "protocol": "post_hoc_validation_leave_one_factor_out_v1",
        "selection_prohibited": True,
        "claim_scope": (
            "Diagnostic validation ablation after the registered blind tests; "
            "must not replace or retune the frozen universal protocol."
        ),
        "variant_order": list(VARIANTS),
        "variant_factors": VARIANT_FACTORS,
        "non_learning_inputs": {
            "used": ["cached DA3 depth", "BIM depth", "BIM-valid encoded as zero depth"],
            "not_used": [
                "RGB",
                "GT",
                "semantic/furniture masks",
                "stored BIM normals",
                "stored BIM edge channel",
            ],
            "edge_note": (
                "The active edge factor is a Sobel gradient gate recomputed from BIM depth, "
                "not the stored bim_edge tensor."
            ),
        },
        "datasets": datasets,
        "generator": "scripts/analysis/ablate_deterministic_bim_direct.py",
        "generator_sha256": _sha256(Path(__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(output), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "datasets": list(datasets)}, indent=2))


if __name__ == "__main__":
    main()
