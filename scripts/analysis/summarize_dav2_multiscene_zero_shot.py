#!/usr/bin/env python3
"""Summarize the four DAv2 scale+r_low models on three BIMNet scenes."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCENES = {
    "HxpKQynjfin": {
        "audit_key": "train/hxp",
        "joint": "hxp_dav2_early_fusion_joint_scale_low18_low36_6epoch_continuation_zero_shot",
        "r36": "hxp_dav2_early_fusion_scale_low36_only_6epoch_continuation_zero_shot",
        "r72": "hxp_dav2_early_fusion_scale_low72_only_6epoch_continuation_zero_shot",
    },
    "759xd9YjKW5": {
        "audit_key": "train/759",
        "joint": "759_dav2_early_fusion_joint_scale_low18_low36_zero_shot",
        "r36": "759_dav2_early_fusion_scale_low36_only_zero_shot",
        "r72": "759_dav2_early_fusion_scale_low72_only_zero_shot",
    },
    "1pXnuDYAj8r": {
        "audit_key": "train/1px",
        "joint": "1px_dav2_early_fusion_joint_scale_low18_low36_zero_shot",
        "r36": "1px_dav2_early_fusion_scale_low36_only_zero_shot",
        "r72": "1px_dav2_early_fusion_scale_low72_only_zero_shot",
    },
}
MODELS = {
    "scale+r18": ("joint", "scale_low"),
    "scale+r18+r36": ("joint", "final"),
    "scale+r36": ("r36", "final"),
    "scale+r72": ("r72", "final"),
}
LINEAR_METRICS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results/matterport3d"))
    parser.add_argument(
        "--selection-audit",
        type=Path,
        default=Path("data/provenance/matterport_bimnet_three_rule_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/matterport3d/dav2_scale_low_three_scene_zero_shot.json"),
    )
    return parser.parse_args()


def combined_metrics(metrics: list[dict[str, Any]]) -> dict[str, float | int]:
    weights = [int(item["valid_pixels"]) for item in metrics]
    total = sum(weights)
    output = {
        name: sum(float(item[name]) * weight for item, weight in zip(metrics, weights, strict=True))
        / total
        for name in LINEAR_METRICS
    }
    for name in ("rmse_m", "rmse_log"):
        output[name] = math.sqrt(
            sum(float(item[name]) ** 2 * weight for item, weight in zip(metrics, weights, strict=True))
            / total
        )
    output["silog_x100"] = 100 * math.sqrt(
        max(0.0, output["rmse_log"] ** 2 - output["mean_log_error"] ** 2)
    )
    output["valid_pixels"] = total
    return output


def main() -> None:
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    audit = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for scene_id, definition in SCENES.items():
        expected = audit["scenes"][definition["audit_key"]]
        for checkpoint_key in ("joint", "r36", "r72"):
            path = root / definition[checkpoint_key] / "summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            selection = summary["selection"]
            if selection["frame_ids_sha256"] != expected["selected_frame_ids_sha256"]:
                raise RuntimeError(f"Frame-set mismatch for {scene_id}/{checkpoint_key}")
            if int(selection["frames"]) != int(expected["selected_frames"]):
                raise RuntimeError(f"Frame-count mismatch for {scene_id}/{checkpoint_key}")
            summaries[(scene_id, checkpoint_key)] = summary

    scene_results: dict[str, Any] = {}
    for scene_id in SCENES:
        model_results = {}
        for model_name, (checkpoint_key, prediction_key) in MODELS.items():
            subset = summaries[(scene_id, checkpoint_key)]["subsets"]["three_rule_valid"]
            prediction = subset["predictions"][prediction_key]["pixel_micro"]
            scale = subset["predictions"]["scale"]["pixel_micro"]
            raw = subset["predictions"]["raw"]["pixel_micro"]
            model_results[model_name] = {
                "raw": raw,
                "scale": scale,
                "prediction": prediction,
                "relative_abs_rel_improvement_vs_raw": (
                    float(raw["abs_rel"]) - float(prediction["abs_rel"])
                )
                / float(raw["abs_rel"]),
            }
        audit_scene = audit["scenes"][SCENES[scene_id]["audit_key"]]
        scene_results[scene_id] = {
            "source_frames": audit_scene["source_frames"],
            "selected_frames": audit_scene["selected_frames"],
            "selected_fraction": audit_scene["selected_fraction"],
            "models": model_results,
        }

    groups = {
        "759xd9YjKW5+1pXnuDYAj8r": ("759xd9YjKW5", "1pXnuDYAj8r"),
        "all_three_scenes": tuple(SCENES),
    }
    group_results = {}
    for group_name, scene_ids in groups.items():
        models = {}
        for model_name, (checkpoint_key, prediction_key) in MODELS.items():
            prediction_parts = []
            scale_parts = []
            raw_parts = []
            for scene_id in scene_ids:
                predictions = summaries[(scene_id, checkpoint_key)]["subsets"][
                    "three_rule_valid"
                ]["predictions"]
                prediction_parts.append(predictions[prediction_key]["pixel_micro"])
                scale_parts.append(predictions["scale"]["pixel_micro"])
                raw_parts.append(predictions["raw"]["pixel_micro"])
            prediction = combined_metrics(prediction_parts)
            scale = combined_metrics(scale_parts)
            raw = combined_metrics(raw_parts)
            models[model_name] = {
                "raw": raw,
                "scale": scale,
                "prediction": prediction,
                "relative_abs_rel_improvement_vs_raw": (
                    float(raw["abs_rel"]) - float(prediction["abs_rel"])
                )
                / float(raw["abs_rel"]),
            }
        group_results[group_name] = {"scenes": list(scene_ids), "models": models}

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_audit": str(args.selection_audit.expanduser().resolve()),
        "aggregation": "pixel-micro; no test-time scale or affine alignment",
        "scenes": scene_results,
        "groups": group_results,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    for scene_id, scene in scene_results.items():
        print(f"\n{scene_id} ({scene['selected_frames']} frames)")
        for model_name, values in scene["models"].items():
            print(
                f"  {model_name:18s} scale={values['scale']['abs_rel']:.5f} "
                f"final={values['prediction']['abs_rel']:.5f}"
            )
    for group_name, group in group_results.items():
        print(f"\n{group_name}")
        for model_name, values in group["models"].items():
            print(
                f"  {model_name:18s} scale={values['scale']['abs_rel']:.5f} "
                f"final={values['prediction']['abs_rel']:.5f}"
            )


if __name__ == "__main__":
    main()
