#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


DIAGNOSTICS = (
    "learned_frame_trust",
    "learned_mean_pixel_trust",
    "learned_mean_variance",
    "learned_mean_abs_log_residual",
    "learned_mean_support",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a frame-level learned/previous-method fallback rule on validation "
            "frames, then apply the frozen rule to test frames."
        )
    )
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_metrics(
    rows: list[dict[str, str]], use_learned: list[bool]
) -> dict[str, float | int]:
    total = sum(int(row["valid_pixels"]) for row in rows)
    result: dict[str, float | int] = {
        "count": total,
        "learned_frames": sum(use_learned),
    }
    for metric in ("abs_rel", "mae", "delta1", "delta2", "delta3"):
        numerator = 0.0
        for row, learned in zip(rows, use_learned):
            prefix = "refined_" if learned else "previous_scale_local_"
            numerator += int(row["valid_pixels"]) * float(row[prefix + metric])
        result[metric] = numerator / total
    squared_error = 0.0
    for row, learned in zip(rows, use_learned):
        prefix = "refined_" if learned else "previous_scale_local_"
        squared_error += (
            int(row["valid_pixels"]) * float(row[prefix + "rmse"]) ** 2
        )
    result["rmse"] = math.sqrt(squared_error / total)
    return result


def select_rule(rows: list[dict[str, str]]) -> dict[str, float | str]:
    best: dict[str, float | str] | None = None
    for diagnostic in DIAGNOSTICS:
        candidates = sorted({float(row[diagnostic]) for row in rows})
        for direction in ("le", "ge"):
            for threshold in candidates:
                decisions = [
                    float(row[diagnostic]) <= threshold
                    if direction == "le"
                    else float(row[diagnostic]) >= threshold
                    for row in rows
                ]
                score = pooled_metrics(rows, decisions)["abs_rel"]
                candidate: dict[str, float | str] = {
                    "diagnostic": diagnostic,
                    "direction": direction,
                    "threshold": threshold,
                    "validation_abs_rel": float(score),
                }
                if best is None or score < float(best["validation_abs_rel"]):
                    best = candidate
    assert best is not None
    return best


def decisions_from_rule(
    rows: list[dict[str, str]], rule: dict[str, float | str]
) -> list[bool]:
    diagnostic = str(rule["diagnostic"])
    threshold = float(rule["threshold"])
    if rule["direction"] == "le":
        return [float(row[diagnostic]) <= threshold for row in rows]
    return [float(row[diagnostic]) >= threshold for row in rows]


def main() -> None:
    args = parse_args()
    validation = read_rows(args.validation)
    test = read_rows(args.test)
    rule = select_rule(validation)
    result = {
        "selection_protocol": (
            "Rule and threshold selected only on validation AbsRel; test labels were "
            "not used for selection."
        ),
        "rule": rule,
        "validation": {
            "previous_scale_local": pooled_metrics(
                validation, [False] * len(validation)
            ),
            "learned": pooled_metrics(validation, [True] * len(validation)),
            "gated": pooled_metrics(
                validation, decisions_from_rule(validation, rule)
            ),
        },
        "test": {
            "previous_scale_local": pooled_metrics(test, [False] * len(test)),
            "learned": pooled_metrics(test, [True] * len(test)),
            "gated": pooled_metrics(test, decisions_from_rule(test, rule)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
