#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bim_priorda3.baselines import (
    configured_scale_and_local_features,
    resolve_scale_estimator_config,
)
from bim_priorda3.checkpoints import (
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_model_config,
)
from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import build_loader, move_batch
from bim_priorda3.metrics import depth_metrics
from bim_priorda3.models import BIMPriorDA3
from bim_priorda3.scale_protocol import validate_universal_scale_protocol


class MetricAccumulator:
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
    ) -> None:
        mask = valid.bool() & torch.isfinite(prediction) & torch.isfinite(target)
        mask &= (prediction > 0) & (target > 0)
        pred, gt = prediction[mask].double(), target[mask].double()
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

    def compute(self) -> dict[str, float]:
        if not self.count:
            return {
                "abs_rel": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "delta1": float("nan"),
                "delta2": float("nan"),
                "delta3": float("nan"),
                "count": 0,
            }
        return {
            "abs_rel": self.abs_rel_sum / self.count,
            "rmse": (self.squared_error_sum / self.count) ** 0.5,
            "mae": self.absolute_error_sum / self.count,
            "delta1": self.delta1_sum / self.count,
            "delta2": self.delta2_sum / self.count,
            "delta3": self.delta3_sum / self.count,
            "count": self.count,
        }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_sample_ids_sha256(sample_ids: list[str] | tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        digest.update(sample_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _region_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(record["region"]) for record in records).items()))


def read_ignore_sample_ids(path: Path) -> tuple[str, ...]:
    """Read exact manifest sample IDs from a line-oriented ignore file."""
    if not path.is_file():
        raise FileNotFoundError(f"Ignore file does not exist: {path}")

    sample_ids: list[str] = []
    first_line_by_id: dict[str, int] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        sample_id = raw_line.partition("#")[0].strip()
        if not sample_id:
            continue
        parts = sample_id.split("/")
        if len(parts) != 2 or not all(parts) or any(character.isspace() for character in sample_id):
            raise ValueError(
                f"{path}:{line_number}: expected an exact '<region>/<frame>' sample ID"
            )
        if sample_id in first_line_by_id:
            raise ValueError(
                f"{path}:{line_number}: duplicate sample ID {sample_id!r}; "
                f"first declared on line {first_line_by_id[sample_id]}"
            )
        first_line_by_id[sample_id] = line_number
        sample_ids.append(sample_id)

    if not sample_ids:
        raise ValueError(f"Ignore file contains no sample IDs: {path}")
    return tuple(sample_ids)


def read_manifest_sample_ids(path: Path) -> set[str]:
    """Read and validate the ID column needed for ignore-list auditing."""
    manifest_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: missing non-empty string 'id'")
            if sample_id in manifest_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample ID {sample_id!r}")
            manifest_ids.add(sample_id)
    if not manifest_ids:
        raise ValueError(f"Manifest contains no samples: {path}")
    return manifest_ids


def apply_ignore_filter(
    dataset: BIMDepthDataset,
    ignore_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Remove declared low-quality samples and return an auditable receipt."""
    ignore_path = ignore_path.resolve()
    manifest_path = manifest_path.resolve()
    declared_ids = read_ignore_sample_ids(ignore_path)
    declared_set = set(declared_ids)
    manifest_ids = read_manifest_sample_ids(manifest_path)
    unknown_ids = sorted(declared_set - manifest_ids)
    if unknown_ids:
        preview = ", ".join(unknown_ids[:10])
        suffix = "" if len(unknown_ids) <= 10 else f", ... (+{len(unknown_ids) - 10})"
        raise ValueError(
            f"{ignore_path}: {len(unknown_ids)} sample IDs are absent from "
            f"{manifest_path}: {preview}{suffix}"
        )

    original_records = list(dataset.records)
    original_ids = [str(record["id"]) for record in original_records]
    ignored_ids = [str(record["id"]) for record in original_records if record["id"] in declared_set]
    remaining_records = [record for record in original_records if record["id"] not in declared_set]
    if not remaining_records:
        raise RuntimeError(
            f"Ignore file removed every sample from the selected {dataset.split!r} split"
        )
    dataset.records = remaining_records

    ignored_set = set(ignored_ids)
    outside_split_ids = [sample_id for sample_id in declared_ids if sample_id not in ignored_set]
    evaluated_ids = [str(record["id"]) for record in remaining_records]
    return {
        "schema_version": 1,
        "enabled": True,
        "scope": "evaluation_only:test",
        "match_key": "manifest.id",
        "match_policy": ("exact_case_sensitive_global_manifest_post_stride_intersection"),
        "ignore_file": str(ignore_path),
        "ignore_file_sha256": _file_sha256(ignore_path),
        "manifest": str(manifest_path),
        "declared_sample_count": len(declared_ids),
        "declared_sample_ids": list(declared_ids),
        "manifest_match_count": len(declared_ids),
        "manifest_unmatched_sample_ids": [],
        "samples_before": len(original_records),
        "samples_ignored": len(ignored_ids),
        "samples_after": len(remaining_records),
        "ignored_sample_ids": ignored_ids,
        "applied_sample_ids": ignored_ids,
        "declared_outside_selected_split_count": len(outside_split_ids),
        "declared_outside_selected_split_ids": outside_split_ids,
        "outside_effective_split_ids": outside_split_ids,
        "region_counts_before": _region_counts(original_records),
        "region_counts_after": _region_counts(remaining_records),
        "original_ordered_ids_sha256": _ordered_sample_ids_sha256(original_ids),
        "evaluated_ordered_ids_sha256": _ordered_sample_ids_sha256(evaluated_ids),
    }


def validate_quality_filter_scope(split: str, ignore_file: Path | None) -> None:
    if ignore_file is not None and split != "test":
        raise ValueError("--ignore-file is restricted to --split test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BIM-PriorDA3 on held-out regions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--ignore-file",
        type=Path,
        help=(
            "Line-oriented list of exact '<region>/<frame>' manifest IDs to exclude. "
            "The file and applied IDs are recorded in summary.json."
        ),
    )
    parser.add_argument(
        "--allow-inference-calibration",
        action="store_true",
        help=(
            "Allow only routing depth/temperature to differ from the checkpoint "
            "training config, and record those overrides in the summary."
        ),
    )
    parser.add_argument(
        "--allow-cross-dataset-checkpoint",
        "--cross-dataset",
        dest="allow_cross_dataset_checkpoint",
        action="store_true",
        help=(
            "Explicitly allow a checkpoint trained on a different dataset. "
            "Only dataset-provenance matching is relaxed; model config and "
            "state_dict loading remain strict."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_quality_filter_scope(args.split, args.ignore_file)
    cfg = load_config(args.config)
    universal_scale_protocol = validate_universal_scale_protocol(cfg)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    original_ordered_ids = [str(record["id"]) for record in dataset.records]
    original_suffix_ids = original_ordered_ids[-82:]
    quality_filter: dict[str, Any] = {
        "schema_version": 1,
        "enabled": False,
        "samples_before": len(dataset),
        "samples_ignored": 0,
        "samples_after": len(dataset),
        "region_counts_before": _region_counts(dataset.records),
        "region_counts_after": _region_counts(dataset.records),
        "original_ordered_ids_sha256": _ordered_sample_ids_sha256(original_ordered_ids),
        "evaluated_ordered_ids_sha256": _ordered_sample_ids_sha256(original_ordered_ids),
    }
    if args.ignore_file is not None:
        ignore_path = resolve_project_path(cfg, args.ignore_file)
        manifest_path = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
        quality_filter = apply_ignore_filter(dataset, ignore_path, manifest_path)
    loader = build_loader(dataset, 1, int(cfg.train.num_workers), shuffle=False)
    model = BIMPriorDA3(cfg).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_cfg = state.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise TypeError("Checkpoint does not contain a training config")
    dataset_provenance_validation = validate_checkpoint_evaluation_dataset_provenance(
        state,
        dataset.split_provenance,
        split=args.split,
        allow_cross_dataset=args.allow_cross_dataset_checkpoint,
    )
    model_overrides = validate_checkpoint_model_config(
        state,
        cfg.model,
        allow_inference_calibration=args.allow_inference_calibration,
    )
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    scale_estimator = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))

    e2e_enabled = model.e2e_da3_enabled
    methods = (
        (
            "base",
            "universal_global_scale",
            "universal_bim_direct",
            "live_da3",
            "live_universal_scale",
            "live_universal_bim_direct",
            "coarse",
            "refined",
        )
        if e2e_enabled
        else (
            "base",
            "universal_global_scale",
            "universal_bim_direct",
            "coarse",
            "refined",
        )
    )
    rows = []
    overall_accumulators = {method: MetricAccumulator() for method in methods}
    suffix_accumulators = {method: MetricAccumulator() for method in methods}
    bins = [float(value) for value in cfg.evaluation.distance_bins]
    range_accumulators = {
        f"{lower:g}-{upper:g}m": {method: MetricAccumulator() for method in methods}
        for lower, upper in zip(bins[:-1], bins[1:])
    }
    suffix_id_set = set(original_suffix_ids)
    with torch.no_grad():
        for batch in loader:
            identifiers = batch["sample_id"]
            regions = batch["region"]
            image_timestamps = batch["image_timestamp"]
            frame_indices = batch["frame_index"]
            batch = move_batch(batch, device)
            if e2e_enabled:
                batch["request_live_bim_direct"] = True
            output = model(batch)
            if e2e_enabled and output.get("uses_live_da3") is not True:
                raise RuntimeError(
                    "E2E evaluation expected a live DA3 prediction, but the "
                    "model did not report uses_live_da3=true"
                )
            valid = batch["gt_valid"] > 0
            row = {
                "sample_id": identifiers[0],
                "region": regions[0],
                "image_timestamp": float(image_timestamps[0]),
                "frame_index": int(frame_indices[0]),
                "valid_pixels": int(valid.sum()),
            }
            scaled_np, local_np, _, _, scale_receipt = configured_scale_and_local_features(
                batch["base_depth"][0, 0].cpu().numpy(),
                batch["bim_depth"][0, 0].cpu().numpy(),
                scale_estimator,
            )
            scaled = torch.from_numpy(scaled_np)[None, None].to(device)
            previous_local = torch.from_numpy(local_np)[None, None].to(device)
            row["universal_bim_scale"] = float(scale_receipt.scale)
            row["universal_bim_scale_support"] = int(scale_receipt.support_count)
            row["universal_bim_scale_fallback"] = bool(scale_receipt.fallback)
            row["learned_frame_trust"] = float(
                torch.sigmoid(output["frame_trust_logits"]).mean().cpu()
            )
            row["learned_mean_pixel_trust"] = float(output["trust_probability"].mean().cpu())
            row["learned_mean_variance"] = float(torch.exp(output["log_variance"]).mean().cpu())
            row["learned_mean_abs_log_residual"] = float(output["log_residual"].abs().mean().cpu())
            row["learned_mean_support"] = float(output["support"].mean().cpu())
            if e2e_enabled:
                row["live_da3_scale"] = float(output["da3_scale"].mean().cpu())
                row["live_da3_scale_support"] = int(output["da3_scale_support"].sum().cpu())
            for name in (
                "frame_log_residual",
                "low_log_residual",
                "detail_log_residual",
            ):
                if name in output:
                    row[f"learned_mean_abs_{name}"] = float(output[name].abs().mean().cpu())
            predictions = [
                ("base", batch["base_depth"]),
                ("universal_global_scale", scaled),
                ("universal_bim_direct", previous_local),
            ]
            if e2e_enabled:
                predictions.extend(
                    [
                        ("live_da3", output["base_depth"]),
                        ("live_universal_scale", output["scaled_depth"]),
                        (
                            "live_universal_bim_direct",
                            output["live_robust_bim_direct"],
                        ),
                    ]
                )
            predictions.extend(
                [
                    ("coarse", output["coarse_depth"]),
                    ("refined", output["depth"]),
                ]
            )
            for method, prediction in predictions:
                metric = depth_metrics(prediction, batch["gt_depth"], valid)
                row.update({f"{method}_{key}": value for key, value in metric.items()})
                overall_accumulators[method].update(prediction, batch["gt_depth"], valid)
                if identifiers[0] in suffix_id_set:
                    suffix_accumulators[method].update(prediction, batch["gt_depth"], valid)
                for lower, upper in zip(bins[:-1], bins[1:]):
                    name = f"{lower:g}-{upper:g}m"
                    range_mask = valid & (batch["gt_depth"] >= lower) & (batch["gt_depth"] < upper)
                    range_accumulators[name][method].update(
                        prediction, batch["gt_depth"], range_mask
                    )
            rows.append(row)
            print(
                f"{identifiers[0]} AbsRel {row['base_abs_rel']:.4f}->{row['refined_abs_rel']:.4f}",
                flush=True,
            )

    output_dir = args.output or (
        resolve_project_path(cfg, cfg.experiment.output_dir) / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame_path = output_dir / "per_frame.csv"
    with per_frame_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    evaluated_ordered_ids = [str(record["id"]) for record in dataset.records]
    evaluated_id_set = set(evaluated_ordered_ids)
    evaluated_suffix_ids = [
        sample_id for sample_id in original_suffix_ids if sample_id in evaluated_id_set
    ]
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation_config": str(Path(cfg.config_path).resolve()),
        "evaluation_script": str(Path(__file__).resolve()),
        "evaluation_script_sha256": _file_sha256(Path(__file__).resolve()),
        "per_frame_csv_sha256": _file_sha256(per_frame_path),
        "checkpoint_training_config": checkpoint_cfg.get("config_path"),
        "checkpoint_provenance": state.get("provenance", {}),
        "inference_model_overrides": model_overrides,
        "split": args.split,
        "evaluated_sample_count": len(dataset),
        "quality_filter": quality_filter,
        "dataset": {
            "split_provenance": dataset.split_provenance,
            "checkpoint_validation": dataset_provenance_validation,
            "cross_dataset_checkpoint_opt_in": (args.allow_cross_dataset_checkpoint),
            "transfer_provenance": {
                "source": dataset_provenance_validation["source"],
                "target": dataset_provenance_validation["target"],
            },
        },
        "regions": sorted({str(record["region"]) for record in dataset.records}),
        "universal_scale_protocol": universal_scale_protocol,
        "scale_estimator": scale_estimator,
        "stage_definitions": {
            "base": {
                "source": "dataset cached frozen DA3",
                "learned_in_this_run": False,
                "uses_bim": False,
            },
            "universal_global_scale": {
                "source": "universal robust BIM scale applied to cached frozen DA3",
                "learned_in_this_run": False,
                "uses_bim": True,
            },
            "universal_bim_direct": {
                "source": (
                    "universal robust BIM scale and fixed local correction "
                    "applied to cached frozen DA3"
                ),
                "learned_in_this_run": False,
                "uses_bim": True,
                "primary_non_learning_bim_baseline": True,
            },
            "coarse": {
                "source": (
                    "live DA3 plus detached BIM scale" if e2e_enabled else "dataset scaled depth"
                ),
                "learned_in_this_run": e2e_enabled,
                "uses_bim": True,
            },
            "refined": {
                "source": "BIM/RGB learned refiner",
                "learned_in_this_run": True,
                "uses_bim": True,
            },
            **(
                {
                    "live_da3": {
                        "source": "checkpoint DA3 decoder live inference",
                        "learned_in_this_run": True,
                        "uses_bim": False,
                    },
                    "live_universal_scale": {
                        "source": "live DA3 plus detached universal robust BIM scale",
                        "learned_in_this_run": True,
                        "uses_bim": True,
                    },
                    "live_universal_bim_direct": {
                        "source": "live DA3 plus the same universal BIM-direct correction",
                        "learned_in_this_run": False,
                        "uses_bim": True,
                    },
                }
                if e2e_enabled
                else {}
            ),
        },
        "metric_aliases": ({"coarse": "live_universal_scale"} if e2e_enabled else {}),
        "overall": {
            method: accumulator.compute() for method, accumulator in overall_accumulators.items()
        },
        "chronological_suffix_82": {
            method: accumulator.compute() for method, accumulator in suffix_accumulators.items()
        },
        "chronological_suffix_82_population": {
            "definition": (
                "Original post-stride population's final 82 samples; ignored "
                "samples are removed without backfilling from earlier frames."
            ),
            "samples_before_filter": len(original_suffix_ids),
            "samples_ignored": len(original_suffix_ids) - len(evaluated_suffix_ids),
            "samples_evaluated": len(evaluated_suffix_ids),
            "original_ordered_ids_sha256": _ordered_sample_ids_sha256(original_suffix_ids),
            "evaluated_ordered_ids_sha256": _ordered_sample_ids_sha256(evaluated_suffix_ids),
        },
        "distance_ranges": {
            name: {
                method: accumulator.compute() for method, accumulator in method_accumulators.items()
            }
            for name, method_accumulators in range_accumulators.items()
        },
    }
    base_abs_rel = summary["overall"]["base"]["abs_rel"]
    refined_abs_rel = summary["overall"]["refined"]["abs_rel"]
    summary["abs_rel_relative_improvement"] = (
        (base_abs_rel - refined_abs_rel) / base_abs_rel if base_abs_rel > 0 else np.nan
    )
    direct_abs_rel = summary["overall"]["universal_bim_direct"]["abs_rel"]
    summary["abs_rel_improvement_over_direct_bim"] = (
        (direct_abs_rel - refined_abs_rel) / direct_abs_rel if direct_abs_rel > 0 else np.nan
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
