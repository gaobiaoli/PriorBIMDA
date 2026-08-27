#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _bootstrap_paired_rooms(
    room_metrics: dict[str, dict[str, dict[str, float | int]]],
    *,
    candidate: str,
    reference: str,
    metric: str,
    seed: int,
    repetitions: int,
) -> dict[str, Any]:
    rooms = sorted(room_metrics)
    differences = np.asarray(
        [
            float(room_metrics[room][candidate][metric])
            - float(room_metrics[room][reference][metric])
            for room in rooms
            if int(room_metrics[room][candidate]["count"]) > 0
            and int(room_metrics[room][reference]["count"]) > 0
        ],
        dtype=np.float64,
    )
    if repetitions < 1 or not len(differences) or not np.isfinite(differences).all():
        raise ValueError("paired room bootstrap requires finite supported rooms")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    bootstrap = differences[sampled].mean(axis=1)
    return {
        "difference_definition": "candidate - reference (negative is better)",
        "rooms": len(differences),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "candidate_better_room_fraction": float((differences < 0).mean()),
        "bootstrap_repetitions": repetitions,
        "seed": seed,
        "confidence_interval_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "aggregates" not in payload:
        raise ValueError(f"{path}: not a Stanford evaluation summary")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two Stanford evaluation summaries with paired room bootstrap."
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate-method", default="refined")
    parser.add_argument("--reference-method", default="refined")
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = _load(args.candidate)
    reference = _load(args.reference)
    for key in ("split", "sample_count", "rooms", "ground_truth_support"):
        if candidate.get(key) != reference.get(key):
            raise ValueError(f"candidate/reference {key} differs")
    subsets = sorted(set(candidate["aggregates"]) & set(reference["aggregates"]))
    comparisons: dict[str, Any] = {}
    for subset in subsets:
        candidate_micro = candidate["aggregates"][subset][args.candidate_method][
            "pixel_micro"
        ]
        reference_micro = reference["aggregates"][subset][args.reference_method][
            "pixel_micro"
        ]
        if int(candidate_micro["count"]) != int(reference_micro["count"]):
            raise ValueError(f"{subset}: candidate/reference pixel support differs")
        room_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
        for room in candidate["per_room"]:
            candidate_room = candidate["per_room"][room][subset][args.candidate_method]
            reference_room = reference["per_room"][room][subset][args.reference_method]
            room_metrics[room] = {
                "candidate": candidate_room,
                "reference": reference_room,
            }
        candidate_value = float(candidate_micro["abs_rel"])
        reference_value = float(reference_micro["abs_rel"])
        comparisons[subset] = {
            "candidate_abs_rel": candidate_value,
            "reference_abs_rel": reference_value,
            "difference": candidate_value - reference_value,
            "relative_change_fraction": (
                candidate_value / reference_value - 1.0
                if reference_value > 0
                else float("nan")
            ),
            "pixel_count": int(candidate_micro["count"]),
            "paired_room_bootstrap": _bootstrap_paired_rooms(
                room_metrics,
                candidate="candidate",
                reference="reference",
                metric="abs_rel",
                seed=args.seed,
                repetitions=args.repetitions,
            ),
        }

    result = {
        "schema_version": 1,
        "split": candidate["split"],
        "candidate_summary": str(args.candidate),
        "candidate_summary_sha256": _sha256(args.candidate),
        "candidate_method": args.candidate_method,
        "reference_summary": str(args.reference),
        "reference_summary_sha256": _sha256(args.reference),
        "reference_method": args.reference_method,
        "sample_count": candidate["sample_count"],
        "rooms": candidate["rooms"],
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": args.seed,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
