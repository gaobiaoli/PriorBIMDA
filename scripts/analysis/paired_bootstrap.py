#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired frame-level bootstrap comparison from evaluate.py output."
    )
    parser.add_argument("per_frame_csv", type=Path)
    parser.add_argument("--candidate", default="refined_abs_rel")
    parser.add_argument("--baseline", default="previous_scale_local_abs_rel")
    parser.add_argument(
        "--aggregation",
        choices=("pixel-pooled", "frame-macro"),
        default="pixel-pooled",
        help="Match evaluate.py's pooled primary metric or average frames equally.",
    )
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--block-size",
        type=int,
        default=1,
        help=(
            "Moving-block length for temporally correlated frames. "
            "Use 1 for the ordinary iid frame bootstrap."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def bootstrap_mean_differences(
    difference: np.ndarray,
    samples: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    if not 1 <= block_size <= len(difference):
        raise ValueError("block_size must be between 1 and the number of frames")
    rng = np.random.default_rng(seed)
    blocks_per_sample = int(np.ceil(len(difference) / block_size))
    starts = rng.integers(
        0,
        len(difference) - block_size + 1,
        size=(samples, blocks_per_sample),
    )
    offsets = np.arange(block_size)
    indices = (starts[..., None] + offsets).reshape(samples, -1)
    indices = indices[:, : len(difference)]
    return difference[indices].mean(axis=1)


def aggregate_metric(
    values: np.ndarray,
    weights: np.ndarray,
    root_mean_square: bool,
) -> float:
    if root_mean_square:
        return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))
    return float(np.sum(weights * values) / np.sum(weights))


def bootstrap_paired_differences(
    candidate: np.ndarray,
    baseline: np.ndarray,
    weights: np.ndarray,
    samples: int,
    block_size: int,
    seed: int,
    root_mean_square: bool = False,
) -> np.ndarray:
    if not 1 <= block_size <= len(candidate):
        raise ValueError("block_size must be between 1 and the number of frames")
    rng = np.random.default_rng(seed)
    blocks_per_sample = int(np.ceil(len(candidate) / block_size))
    starts = rng.integers(
        0,
        len(candidate) - block_size + 1,
        size=(samples, blocks_per_sample),
    )
    offsets = np.arange(block_size)
    indices = (starts[..., None] + offsets).reshape(samples, -1)
    indices = indices[:, : len(candidate)]
    sampled_weights = weights[indices]
    denominator = sampled_weights.sum(axis=1)
    if root_mean_square:
        candidate_aggregate = np.sqrt(
            (sampled_weights * candidate[indices] ** 2).sum(axis=1) / denominator
        )
        baseline_aggregate = np.sqrt(
            (sampled_weights * baseline[indices] ** 2).sum(axis=1) / denominator
        )
    else:
        candidate_aggregate = (sampled_weights * candidate[indices]).sum(axis=1) / denominator
        baseline_aggregate = (sampled_weights * baseline[indices]).sum(axis=1) / denominator
    return candidate_aggregate - baseline_aggregate


def main() -> None:
    args = parse_args()
    with args.per_frame_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidate = np.asarray([float(row[args.candidate]) for row in rows])
    baseline = np.asarray([float(row[args.baseline]) for row in rows])
    difference = candidate - baseline
    root_mean_square = args.candidate.endswith("_rmse")
    if args.aggregation == "pixel-pooled":
        weights = np.asarray([float(row["valid_pixels"]) for row in rows])
        candidate_aggregate = aggregate_metric(
            candidate,
            weights,
            root_mean_square,
        )
        baseline_aggregate = aggregate_metric(
            baseline,
            weights,
            root_mean_square,
        )
        bootstrap = bootstrap_paired_differences(
            candidate,
            baseline,
            weights,
            args.samples,
            args.block_size,
            args.seed,
            root_mean_square,
        )
    else:
        candidate_aggregate = float(candidate.mean())
        baseline_aggregate = float(baseline.mean())
        bootstrap = bootstrap_mean_differences(
            difference,
            args.samples,
            args.block_size,
            args.seed,
        )
    result = {
        "frames": len(rows),
        "candidate_column": args.candidate,
        "baseline_column": args.baseline,
        "candidate_frame_macro_mean": float(candidate.mean()),
        "baseline_frame_macro_mean": float(baseline.mean()),
        "mean_paired_difference": float(difference.mean()),
        "aggregation": args.aggregation,
        "candidate_aggregate": candidate_aggregate,
        "baseline_aggregate": baseline_aggregate,
        "aggregate_difference": candidate_aggregate - baseline_aggregate,
        "paired_difference_ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "frame_win_rate": float(np.mean(candidate < baseline)),
        "bootstrap_method": ("iid_frame" if args.block_size == 1 else "moving_block"),
        "block_size": args.block_size,
        "bootstrap_samples": args.samples,
        "seed": args.seed,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
