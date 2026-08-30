#!/usr/bin/env python3
"""Reaggregate Hxp zero-shot predictions with a frozen three-rule frame filter."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "abs_rel",
    "sq_rel",
    "rmse_m",
    "mae_m",
    "rmse_log",
    "mean_log_error",
    "mean_abs_log_error",
    "log10_error",
    "silog_x100",
    "delta1",
    "delta2",
    "delta3",
)
LINEAR_MICRO_METRICS = (
    "abs_rel",
    "sq_rel",
    "mae_m",
    "mean_log_error",
    "mean_abs_log_error",
    "log10_error",
    "delta1",
    "delta2",
    "delta3",
)
METHODS = {
    "full_regression_scale": {
        "label": "Full regression scale",
        "directory": "hxp_full_regression_scale_zero_shot",
    },
    "fixed_attention_huber_scale": {
        "label": "Fixed-attention Huber scale",
        "directory": "hxp_fixed_attention_huber_zero_shot",
    },
    "iterative_attention_huber_scale": {
        "label": "Iterative-attention Huber scale",
        "directory": "hxp_iterative_attention_huber_zero_shot",
    },
    "bim_early_fusion_dense": {
        "label": "BIM early-fusion dense",
        "directory": "hxp_bim_early_fusion_dense_zero_shot",
    },
    "iterative_scale_refiner_sota": {
        "label": "Area_1 iterative scale+refiner SOTA (final)",
        "directory": "hxp_iterative_scale_refiner_sota_zero_shot",
        "prediction_prefix": "final",
    },
    "fixed_attention_huber_reduced_refiner": {
        "label": "Fixed-attention Huber + reduced refiner (final)",
        "directory": "hxp_fixed_attention_huber_reduced_refiner_continuation_zero_shot",
        "prediction_prefix": "final",
    },
    "iterative_attention_huber_reduced_refiner": {
        "label": "Iterative-attention Huber + reduced refiner (final)",
        "directory": "hxp_iterative_attention_huber_reduced_refiner_continuation_zero_shot",
        "prediction_prefix": "final",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/matterport3d"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/matterport3d/hxp_three_rule_zero_shot_comparison.json"),
    )
    parser.add_argument("--gt-min-valid-fraction", type=float, default=0.10)
    parser.add_argument("--bim-min-hit-fraction", type=float, default=0.20)
    parser.add_argument("--aabb-margin-m", type=float, default=0.0)
    return parser.parse_args()


def _number(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if text.casefold() in {"true", "false"}:
        return text.casefold() == "true"
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return value


def read_latest_rows(path: Path) -> list[dict[str, Any]]:
    latest = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = {key: _number(value) for key, value in raw.items()}
            latest[str(row["frame_id"])] = row
    return list(latest.values())


def select_frames(
    rows: Iterable[Mapping[str, Any]],
    *,
    aabb_min: list[float],
    aabb_max: list[float],
    gt_min_valid_fraction: float,
    bim_min_hit_fraction: float,
    aabb_margin_m: float,
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        if not float(row["gt_valid_fraction"]) > gt_min_valid_fraction:
            continue
        if not float(row["bim_hit_fraction"]) > bim_min_hit_fraction:
            continue
        position = [float(row[f"camera_{axis}"]) for axis in "xyz"]
        inside = all(
            aabb_min[index] - aabb_margin_m
            <= position[index]
            <= aabb_max[index] + aabb_margin_m
            for index in range(3)
        )
        if inside:
            selected.append(dict(row))
    return selected


def aggregate_metrics(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, float | int]:
    weights = np.asarray([int(row["gt_valid_pixels"]) for row in rows], dtype=np.float64)
    total = float(weights.sum())
    output: dict[str, float | int] = {
        metric: float(
            np.average(
                np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows]),
                weights=weights,
            )
        )
        for metric in LINEAR_MICRO_METRICS
    }
    for metric in ("rmse_m", "rmse_log"):
        output[metric] = float(
            math.sqrt(
                np.average(
                    np.asarray([float(row[f"{prefix}_{metric}"]) ** 2 for row in rows]),
                    weights=weights,
                )
            )
        )
    output["silog_x100"] = float(
        100.0
        * math.sqrt(
            max(
                0.0,
                float(output["rmse_log"]) ** 2 - float(output["mean_log_error"]) ** 2,
            )
        )
    )
    output["valid_pixels"] = int(total)
    return output


def frame_macro_metrics(rows: list[Mapping[str, Any]], prefix: str) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[f"{prefix}_{metric}"]) for row in rows]))
        for metric in METRICS
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_set_sha256(frame_ids: set[str]) -> str:
    payload = "\n".join(sorted(frame_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 <= args.gt_min_valid_fraction <= 1:
        raise ValueError("--gt-min-valid-fraction must be in [0, 1]")
    if not 0 <= args.bim_min_hit_fraction <= 1:
        raise ValueError("--bim-min-hit-fraction must be in [0, 1]")
    if args.aabb_margin_m < 0:
        raise ValueError("--aabb-margin-m must be non-negative")

    results_root = args.results_root.expanduser().resolve()
    method_results = {}
    selected_sets: dict[str, set[str]] = {}
    aabb_receipt = None
    for key, definition in METHODS.items():
        directory = results_root / definition["directory"]
        summary_path = directory / "summary.json"
        csv_path = directory / "per_frame.csv"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        aabb_min = [float(value) for value in summary["bim"]["aabb_min"]]
        aabb_max = [float(value) for value in summary["bim"]["aabb_max"]]
        current_aabb = {"min": aabb_min, "max": aabb_max}
        if aabb_receipt is None:
            aabb_receipt = current_aabb
        elif current_aabb != aabb_receipt:
            raise RuntimeError(f"{key} uses a different BIM AABB")

        rows = read_latest_rows(csv_path)
        selected = select_frames(
            rows,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            gt_min_valid_fraction=args.gt_min_valid_fraction,
            bim_min_hit_fraction=args.bim_min_hit_fraction,
            aabb_margin_m=args.aabb_margin_m,
        )
        frame_ids = {str(row["frame_id"]) for row in selected}
        selected_sets[key] = frame_ids
        raw = aggregate_metrics(selected, "raw")
        prediction_prefix = str(definition.get("prediction_prefix", "learned"))
        prediction = aggregate_metrics(selected, prediction_prefix)
        oracle = aggregate_metrics(selected, "oracle_frame_scale")
        raw_abs_rel = float(raw["abs_rel"])
        prediction_abs_rel = float(prediction["abs_rel"])
        method_results[key] = {
            "label": definition["label"],
            "source": {
                "summary": str(summary_path),
                "per_frame_csv": str(csv_path),
                "per_frame_csv_sha256": _sha256(csv_path),
            },
            "frames": len(selected),
            "valid_pixels": int(raw["valid_pixels"]),
            "raw_da3": {
                "pixel_micro": raw,
                "frame_macro": frame_macro_metrics(selected, "raw"),
            },
            "prediction": {
                "pixel_micro": prediction,
                "frame_macro": frame_macro_metrics(selected, prediction_prefix),
            },
            "oracle_frame_scale": {
                "pixel_micro": oracle,
                "frame_macro": frame_macro_metrics(selected, "oracle_frame_scale"),
            },
            "prediction_vs_raw": {
                "pixel_micro_abs_rel_difference": prediction_abs_rel - raw_abs_rel,
                "pixel_micro_abs_rel_relative_improvement": (
                    raw_abs_rel - prediction_abs_rel
                )
                / raw_abs_rel,
                "frame_win_fraction": float(
                    np.mean(
                        [
                            float(row[f"{prediction_prefix}_abs_rel"])
                            < float(row["raw_abs_rel"])
                            for row in selected
                        ]
                    )
                ),
            },
        }

    reference_key = next(iter(selected_sets))
    reference_set = selected_sets[reference_key]
    mismatches = {
        key: len(frame_ids.symmetric_difference(reference_set))
        for key, frame_ids in selected_sets.items()
    }
    if any(mismatches.values()):
        raise RuntimeError(f"Methods selected different frame sets: {mismatches}")

    raw_reference = method_results[reference_key]["raw_da3"]["pixel_micro"]
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scene": "HxpKQynjfin",
        "bimnet_scene": "train/hxp",
        "protocol": {
            "name": "three-rule valid-frame zero-shot benchmark",
            "rules": [
                f"GT positive-depth fraction > {args.gt_min_valid_fraction}",
                f"BIM ray-hit fraction > {args.bim_min_hit_fraction}",
                "camera center inside BIM axis-aligned bounding box",
            ],
            "aabb_margin_m": args.aabb_margin_m,
            "excluded_rules": [
                "BIM/GT depth agreement",
                "BIM/DA3 ratio support",
                "prediction error or model-dependent selection",
            ],
            "depth_support": "all finite positive Matterport GT depth",
            "aggregation": "pixel-micro; raw absolute metric predictions; no alignment",
        },
        "bim_aabb": aabb_receipt,
        "selection": {
            "frames": len(reference_set),
            "frame_ids_sha256": _frame_set_sha256(reference_set),
            "identical_across_methods": True,
            "symmetric_difference_counts": mismatches,
        },
        "shared_raw_da3_reference": raw_reference,
        "methods": method_results,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, output_path)
    print(f"frames={len(reference_set)} frame_ids_sha256={output['selection']['frame_ids_sha256']}")
    print(f"raw_da3_abs_rel={float(raw_reference['abs_rel']):.6f}")
    for result in method_results.values():
        print(
            f"{result['label']}: abs_rel="
            f"{float(result['prediction']['pixel_micro']['abs_rel']):.6f} "
            f"relative_improvement="
            f"{float(result['prediction_vs_raw']['pixel_micro_abs_rel_relative_improvement']):+.2%}"
        )
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
