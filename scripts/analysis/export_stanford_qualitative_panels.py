#!/usr/bin/env python3
"""Export auditable, title-free Stanford Area_1 qualitative figure panels.

The exporter is intentionally validation-only.  It either accepts explicitly
named validation samples or deterministically selects three illustrative
options from the frozen-model validation per-frame CSV.  It never opens a test
result file and it validates that the selection CSV population is exactly the
runtime validation population before using it.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from bim_priorda3.baselines import configured_scale_and_local_features
from bim_priorda3.checkpoints import (
    dataset_split_identity,
    validate_checkpoint_evaluation_dataset_provenance,
    validate_checkpoint_model_config,
)
from bim_priorda3.config import Config, load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import resolved_config_sha256, seed_everything, semantic_config_sha256
from bim_priorda3.models import BIMPriorDA3

PROTOCOL_SPLIT = "val"
PROTOCOL_MIN_DEPTH_M = 0.2
PROTOCOL_MAX_DEPTH_M = 5.0
PROTOCOL_HEIGHT = 504
PROTOCOL_WIDTH = 504
PROTOCOL_SEED = 42
PRESET_NAME = "three-options"
SUBSET_NAMES = (
    "all",
    "furniture",
    "non_structural",
    "bim_foreground_conflict",
    "bim_consistent",
    "bim_no_hit",
)
METHOD_NAMES = (
    "raw_da3",
    "robust_global_scale",
    "robust_bim_direct",
    "refined",
)
DIAGNOSTIC_NAMES = (
    "reliability",
    "routing_gate",
    "total_log_residual",
    "frame_log_residual",
    "low_log_residual",
    "detail_log_residual",
)

DISPLAY_SCALES: dict[str, dict[str, Any]] = {
    "depth": {
        "cmap": "turbo",
        "vmin": PROTOCOL_MIN_DEPTH_M,
        "vmax": PROTOCOL_MAX_DEPTH_M,
        "label": "Depth (m)",
    },
    "absrel": {
        "cmap": "magma",
        "vmin": 0.0,
        "vmax": 0.5,
        "label": "Per-pixel absolute relative error",
    },
    "delta_absrel": {
        "cmap": "coolwarm",
        "vmin": -0.25,
        "vcenter": 0.0,
        "vmax": 0.25,
        "label": "Refined - BIM-direct absolute relative error",
    },
    "unit_interval": {
        "cmap": "viridis",
        "vmin": 0.0,
        "vmax": 1.0,
        "label": "Unit interval",
    },
    "log_residual": {
        "cmap": "coolwarm",
        "vmin": -0.45,
        "vcenter": 0.0,
        "vmax": 0.45,
        "label": "Log-depth residual",
    },
}

SELECTION_RULES: dict[str, dict[str, Any]] = {
    "option_a_typical": {
        "population": "validation rows with all_gt_pixels > 100000",
        "score": "absolute distance from the eligible-population median of "
        "(all_robust_bim_direct_abs_rel - all_refined_abs_rel)",
        "choice": "minimum score; sample_id ascending breaks exact ties",
    },
    "option_b_furniture_conflict_success": {
        "population": "validation rows with furniture_gt_pixels > 30000",
        "score": "(furniture_gt_pixels / all_gt_pixels) * "
        "(furniture_robust_bim_direct_abs_rel - furniture_refined_abs_rel)",
        "choice": "maximum score; sample_id ascending breaks exact ties",
    },
    "option_c_failure": {
        "population": "validation rows with all_gt_pixels > 100000",
        "score": "all_robust_bim_direct_abs_rel - all_refined_abs_rel",
        "choice": "minimum (most negative) score; sample_id ascending breaks exact ties",
    },
}

_DOMAIN_SUFFIX = re.compile(r"_domain_*(?:rgb|depth|semantic|pose)$", re.IGNORECASE)


@dataclass(frozen=True)
class Selection:
    role: str
    sample_id: str
    rule: dict[str, Any]
    csv_row: dict[str, str] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export independent, title-free 504x504 qualitative panels from the "
            "frozen Stanford Area_1 model. The protocol is validation-only."
        )
    )
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument(
        "--sample-id",
        action="append",
        help=(
            "Explicit validation annotation ID or unambiguous long basename. "
            "Repeat the option to export more than one sample."
        ),
    )
    choice.add_argument(
        "--preset",
        choices=(PRESET_NAME,),
        help="Select three deterministic illustrative options from validation CSV only.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stanford_area1.yaml"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/stanford_area1/accepted.pt"),
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=Path("results/stanford_area1/frozen_val_per_frame.csv"),
        help="Authoritative frozen-model validation per-frame CSV used only by the preset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/paper_evaluation/qualitative"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(list(argv) if argv is not None else None)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _required_float(row: Mapping[str, str], key: str) -> float:
    raw = row.get(key)
    try:
        value = float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Selection CSV has non-numeric {key!r}: {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Selection CSV has non-finite {key!r}: {raw!r}")
    return value


def _optional_float(row: Mapping[str, str], key: str) -> float:
    raw = row.get(key)
    try:
        return float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Selection CSV has non-numeric {key!r}: {raw!r}") from exc


def read_selection_csv(path: Path) -> list[dict[str, str]]:
    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise RuntimeError(f"Selection CSV contains no rows: {path}")
    required = {
        "sample_id",
        "all_gt_pixels",
        "all_robust_bim_direct_abs_rel",
        "all_refined_abs_rel",
        "furniture_gt_pixels",
        "furniture_robust_bim_direct_abs_rel",
        "furniture_refined_abs_rel",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Selection CSV lacks required columns: {missing}")
    identifiers = [str(row["sample_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Selection CSV contains duplicate sample_id rows")
    return rows


def validate_selection_csv_population(
    rows: Sequence[Mapping[str, str]],
    validation_records: Sequence[Mapping[str, Any]],
) -> None:
    """Prove that a preset CSV contains the whole validation population and no test rows."""
    csv_ids = {str(row["sample_id"]) for row in rows}
    val_ids = {str(record["id"]) for record in validation_records}
    if csv_ids != val_ids:
        missing = sorted(val_ids - csv_ids)[:5]
        extra = sorted(csv_ids - val_ids)[:5]
        raise ValueError(
            "Preset selection CSV is not exactly the runtime validation population; "
            f"missing_val_ids={missing}, non_val_ids={extra}, "
            f"csv_count={len(csv_ids)}, val_count={len(val_ids)}"
        )


def select_three_options(rows: Sequence[dict[str, str]]) -> list[Selection]:
    enriched: list[tuple[dict[str, str], int, int, float, float]] = []
    for row in rows:
        valid_pixels = int(_required_float(row, "all_gt_pixels"))
        furniture_pixels = int(_required_float(row, "furniture_gt_pixels"))
        all_gain = _required_float(row, "all_robust_bim_direct_abs_rel") - _required_float(
            row, "all_refined_abs_rel"
        )
        furniture_gain = _optional_float(
            row, "furniture_robust_bim_direct_abs_rel"
        ) - _optional_float(row, "furniture_refined_abs_rel")
        enriched.append((row, valid_pixels, furniture_pixels, all_gain, furniture_gain))

    valid_candidates = [entry for entry in enriched if entry[1] > 100_000]
    if not valid_candidates:
        raise RuntimeError("No validation row satisfies all_gt_pixels > 100000")
    median_gain = statistics.median(entry[3] for entry in valid_candidates)
    typical = min(
        valid_candidates,
        key=lambda entry: (abs(entry[3] - median_gain), str(entry[0]["sample_id"])),
    )

    furniture_candidates = [
        entry
        for entry in enriched
        if entry[2] > 30_000 and entry[1] > 0 and math.isfinite(entry[4])
    ]
    if not furniture_candidates:
        raise RuntimeError("No validation row satisfies furniture_gt_pixels > 30000")

    def furniture_score(entry: tuple[dict[str, str], int, int, float, float]) -> float:
        return (entry[2] / entry[1]) * entry[4]

    furniture = min(
        furniture_candidates,
        key=lambda entry: (-furniture_score(entry), str(entry[0]["sample_id"])),
    )
    failure = min(valid_candidates, key=lambda entry: (entry[3], str(entry[0]["sample_id"])))

    chosen = (typical, furniture, failure)
    roles = (
        "option_a_typical",
        "option_b_furniture_conflict_success",
        "option_c_failure",
    )
    selections = []
    for role, entry in zip(roles, chosen):
        row, valid_pixels, furniture_pixels, all_gain, furniture_gain = entry
        rule = {
            **SELECTION_RULES[role],
            "all_gt_pixels": valid_pixels,
            "furniture_gt_pixels": furniture_pixels,
            "all_absrel_gain": all_gain,
            "furniture_absrel_gain": furniture_gain,
        }
        if role == "option_a_typical":
            rule["eligible_population_median_all_absrel_gain"] = median_gain
            rule["distance_to_median"] = abs(all_gain - median_gain)
        if role == "option_b_furniture_conflict_success":
            rule["furniture_fraction_times_gain"] = furniture_score(entry)
        selections.append(
            Selection(
                role=role,
                sample_id=str(row["sample_id"]),
                rule=rule,
                csv_row=dict(row),
            )
        )
    if len({selection.sample_id for selection in selections}) != len(selections):
        raise RuntimeError("The deterministic preset selected the same sample for multiple roles")
    return selections


def _query_basename(query: str) -> str:
    value = Path(query.strip()).name
    for suffix in (".png", ".json", ".npz"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return _DOMAIN_SUFFIX.sub("", value)


def resolve_validation_sample(records: Sequence[Mapping[str, Any]], query: str) -> tuple[int, str]:
    stripped = query.strip().strip("/")
    if not stripped:
        raise ValueError("--sample-id cannot be empty")
    direct = [index for index, record in enumerate(records) if str(record["id"]) == stripped]
    if len(direct) == 1:
        return direct[0], str(records[direct[0]]["id"])
    basename = _query_basename(stripped)
    matches = [
        index
        for index, record in enumerate(records)
        if str(record["id"]).rsplit("/", 1)[-1] == basename
    ]
    if not matches:
        raise KeyError(
            f"Sample {query!r} is absent from the validation annotations; "
            "the exporter never searches or substitutes a test frame"
        )
    if len(matches) > 1:
        identifiers = [str(records[index]["id"]) for index in matches]
        raise ValueError(f"Ambiguous validation sample basename {basename!r}: {identifiers}")
    return matches[0], str(records[matches[0]]["id"])


def require_locked_protocol(cfg: Config) -> None:
    depth_range = (float(cfg.data.min_depth), float(cfg.data.max_depth))
    if depth_range != (PROTOCOL_MIN_DEPTH_M, PROTOCOL_MAX_DEPTH_M):
        raise ValueError(
            "Qualitative exporter requires the locked Stanford metric range "
            f"{(PROTOCOL_MIN_DEPTH_M, PROTOCOL_MAX_DEPTH_M)}, got {depth_range}"
        )
    dimensions = (int(cfg.data.target_height), int(cfg.data.target_width))
    if dimensions != (PROTOCOL_HEIGHT, PROTOCOL_WIDTH):
        raise ValueError(
            f"Qualitative panels must be 504x504, got configured dimensions {dimensions}"
        )
    if not cfg.data.get("split_annotation"):
        raise ValueError("Validation-only export requires data.split_annotation")


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is unavailable")
    return device


def load_validated_frozen_model(
    *,
    cfg: Config,
    checkpoint_path: Path,
    split_provenance: Mapping[str, Any],
    device: torch.device,
) -> tuple[BIMPriorDA3, dict[str, Any]]:
    """Strictly validate model and validation-dataset provenance before inference."""
    checkpoint_path = checkpoint_path.expanduser().resolve()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    dataset_validation = validate_checkpoint_evaluation_dataset_provenance(
        state,
        split_provenance,
        split=PROTOCOL_SPLIT,
        allow_cross_dataset=False,
    )
    if dataset_validation.get("verified") is not True:
        raise RuntimeError(
            "Checkpoint validation-dataset provenance is not strictly verified; "
            f"status={dataset_validation.get('status')!r}"
        )
    model_differences = validate_checkpoint_model_config(
        state,
        cfg.model,
        allow_inference_calibration=False,
    )
    if model_differences:
        raise RuntimeError(
            f"Strict model provenance unexpectedly has differences: {model_differences}"
        )
    model = BIMPriorDA3(cfg)
    if bool(model.e2e_da3_enabled):
        raise ValueError("This exporter accepts only the frozen-DA3 Stanford model")
    model_state = state.get("model")
    if not isinstance(model_state, Mapping):
        raise TypeError(f"Checkpoint lacks a model state mapping: {checkpoint_path}")
    model.load_state_dict(model_state, strict=True)
    model.to(device).eval()
    checkpoint_config = state.get("config")
    checkpoint_provenance = state.get("provenance")
    receipt = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "epoch": state.get("epoch"),
        "best_metric": state.get("best_metric"),
        "training_config_path": (
            checkpoint_config.get("config_path") if isinstance(checkpoint_config, Mapping) else None
        ),
        "checkpoint_semantic_config_sha256": (
            semantic_config_sha256(dict(checkpoint_config))
            if isinstance(checkpoint_config, Mapping)
            else None
        ),
        "recorded_semantic_config_sha256": (
            checkpoint_provenance.get("semantic_config_sha256")
            if isinstance(checkpoint_provenance, Mapping)
            else None
        ),
        "strict_model_config_validation": {
            "verified": True,
            "differences": model_differences,
            "inference_calibration_override": False,
        },
        "strict_validation_dataset_provenance": dataset_validation,
    }
    del state
    gc.collect()
    return model, receipt


def _batched_item(item: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }


def _output_hw(output: Mapping[str, Any], key: str) -> np.ndarray:
    value = output.get(key)
    if not torch.is_tensor(value) or value.ndim != 4 or tuple(value.shape[:2]) != (1, 1):
        shape = tuple(value.shape) if torch.is_tensor(value) else None
        raise ValueError(f"Model output {key!r} must have shape [1, 1, H, W], got {shape}")
    return value[0, 0].detach().float().cpu().numpy()


def infer_frozen_sample(
    model: BIMPriorDA3,
    item: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    batch = _batched_item(item, device)
    with torch.inference_mode():
        output = model(batch)
    if bool(output.get("uses_live_da3", False)):
        raise RuntimeError("Frozen export unexpectedly executed a live DA3 branch")
    key_mapping = {
        "refined": "depth",
        "reliability": "bim_reliability",
        "routing_gate": "residual_routing_gate",
        "total_log_residual": "log_residual",
        "frame_log_residual": "frame_log_residual",
        "low_log_residual": "low_log_residual",
        "detail_log_residual": "detail_log_residual",
    }
    arrays = {name: _output_hw(output, key) for name, key in key_mapping.items()}
    del batch, output
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def item_hw(item: Mapping[str, Any], key: str) -> np.ndarray:
    value = item.get(key)
    if not torch.is_tensor(value) or value.ndim != 3 or value.shape[0] != 1:
        shape = tuple(value.shape) if torch.is_tensor(value) else None
        raise ValueError(f"Dataset item {key!r} must have shape [1, H, W], got {shape}")
    return value[0].detach().float().cpu().numpy()


def build_supports(
    item: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    required = (
        "gt_depth",
        "gt_valid",
        "bim_depth",
        "bim_valid",
        "furniture_mask",
        "non_structural_mask",
        "semantic_valid",
    )
    missing = [key for key in required if key not in item]
    if missing:
        raise RuntimeError(f"Prepared Stanford sample lacks required fields: {missing}")
    gt = item_hw(item, "gt_depth")
    gt_valid = item_hw(item, "gt_valid") > 0
    fixed = gt_valid & np.isfinite(gt) & (gt >= PROTOCOL_MIN_DEPTH_M) & (gt <= PROTOCOL_MAX_DEPTH_M)
    if not fixed.any():
        raise RuntimeError("Selected validation frame has no official z-depth in 0.2-5.0 m")
    bim = item_hw(item, "bim_depth")
    bim_valid = (item_hw(item, "bim_valid") > 0) & np.isfinite(bim) & (bim > 0)
    furniture = item_hw(item, "furniture_mask") > 0
    non_structural = item_hw(item, "non_structural_mask") > 0
    semantic_valid = item_hw(item, "semantic_valid") > 0
    if np.any(furniture & ~non_structural):
        raise RuntimeError("furniture_mask is not a subset of non_structural_mask")
    if np.any((furniture | non_structural) & ~semantic_valid):
        raise RuntimeError("Semantic subset masks include unknown semantic pixels")
    tolerance = np.maximum(0.10, 0.05 * bim)
    subsets = {
        "all": fixed,
        "furniture": fixed & furniture,
        "non_structural": fixed & non_structural,
        "bim_foreground_conflict": fixed & bim_valid & (gt < bim - tolerance),
        "bim_consistent": fixed & bim_valid & (np.abs(gt - bim) <= tolerance),
        "bim_no_hit": fixed & ~bim_valid,
    }
    return fixed, bim_valid, subsets


def fixed_support_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    *,
    name: str,
) -> dict[str, float | int]:
    if prediction.shape != target.shape or support.shape != target.shape:
        raise ValueError(f"{name}: prediction/target/support shapes differ")
    invalid = support & (~np.isfinite(prediction) | (prediction <= 0))
    if invalid.any():
        raise RuntimeError(
            f"{name} has {int(invalid.sum())} invalid predictions on fixed metric support"
        )
    pred = prediction[support].astype(np.float64, copy=False)
    gt = target[support].astype(np.float64, copy=False)
    if not pred.size:
        return {
            "abs_rel": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "delta1": float("nan"),
            "delta2": float("nan"),
            "delta3": float("nan"),
            "count": 0,
        }
    difference = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "abs_rel": float(np.mean(np.abs(difference) / gt)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "mae": float(np.mean(np.abs(difference))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
        "count": int(pred.size),
    }


def evaluate_predictions(
    predictions: Mapping[str, np.ndarray],
    gt: np.ndarray,
    subsets: Mapping[str, np.ndarray],
) -> dict[str, dict[str, dict[str, float | int]]]:
    result = {
        method: {
            subset: fixed_support_metrics(prediction, gt, support, name=f"{method}/{subset}")
            for subset, support in subsets.items()
        }
        for method, prediction in predictions.items()
    }
    for subset, support in subsets.items():
        counts = {method: int(result[method][subset]["count"]) for method in result}
        if any(count != int(support.sum()) for count in counts.values()):
            raise RuntimeError(f"Methods do not share fixed support for {subset}: {counts}")
    return result


def absrel_map(prediction: np.ndarray, gt: np.ndarray, support: np.ndarray) -> np.ndarray:
    result = np.full(gt.shape, np.nan, dtype=np.float32)
    result[support] = np.abs(prediction[support] - gt[support]) / gt[support]
    return result


def _colorize_scalar(
    array: np.ndarray,
    *,
    scale: str,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    spec = DISPLAY_SCALES[scale]
    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Scalar panel must be HxW, got {values.shape}")
    if "vcenter" in spec:
        norm = TwoSlopeNorm(
            vmin=float(spec["vmin"]),
            vcenter=float(spec["vcenter"]),
            vmax=float(spec["vmax"]),
        )
    else:
        norm = Normalize(vmin=float(spec["vmin"]), vmax=float(spec["vmax"]), clip=True)
    safe = np.nan_to_num(
        values, nan=float(spec["vmin"]), posinf=float(spec["vmax"]), neginf=float(spec["vmin"])
    )
    rgb = matplotlib.colormaps[str(spec["cmap"])](norm(safe), bytes=True)[..., :3].copy()
    panel_valid = np.isfinite(values)
    if valid is not None:
        if valid.shape != values.shape:
            raise ValueError(f"Panel mask shape differs: {valid.shape} != {values.shape}")
        panel_valid &= valid
    rgb[~panel_valid] = 0
    return rgb


def _binary_panel(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    return np.repeat((value.astype(np.uint8) * 255)[..., None], 3, axis=2)


def _coverage_panel(coverage: np.ndarray) -> np.ndarray:
    value = np.asarray(coverage)
    rgb = np.zeros((*value.shape, 3), dtype=np.uint8)
    rgb[value == 0] = (242, 142, 43)
    rgb[value == 1] = (89, 161, 79)
    return rgb


def save_title_free_png(path: Path, rgb: np.ndarray) -> None:
    value = np.asarray(rgb)
    expected = (PROTOCOL_HEIGHT, PROTOCOL_WIDTH, 3)
    if value.shape != expected:
        raise ValueError(f"Title-free panel must be exactly {expected}, got {value.shape}")
    if value.dtype != np.uint8:
        raise TypeError(f"Title-free panel must be uint8 RGB, got {value.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, value)


def build_export_arrays(
    *,
    item: Mapping[str, Any],
    model_arrays: Mapping[str, np.ndarray],
    scale_parameters: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fixed, bim_valid, subsets = build_supports(item)
    gt = item_hw(item, "gt_depth")
    raw = item_hw(item, "base_depth")
    bim = item_hw(item, "bim_depth")
    global_scale, bim_direct, correction, correction_support, estimate = (
        configured_scale_and_local_features(raw, bim, scale_parameters)
    )
    predictions = {
        "raw_da3": raw,
        "robust_global_scale": global_scale,
        "robust_bim_direct": bim_direct,
        "refined": np.asarray(model_arrays["refined"], dtype=np.float32),
    }
    metrics = evaluate_predictions(predictions, gt, subsets)
    raw_error = absrel_map(raw, gt, fixed)
    direct_error = absrel_map(bim_direct, gt, fixed)
    refined_error = absrel_map(predictions["refined"], gt, fixed)
    coverage = np.full(gt.shape, -1, dtype=np.int8)
    coverage[fixed & ~bim_valid] = 0
    coverage[fixed & bim_valid] = 1
    rgb = item["rgb"].detach().float().cpu().numpy().transpose(1, 2, 0)
    arrays: dict[str, np.ndarray] = {
        "rgb": np.clip(rgb, 0.0, 1.0).astype(np.float32),
        "gt": gt.astype(np.float32),
        "raw_da3": raw.astype(np.float32),
        "bim_depth": bim.astype(np.float32),
        "robust_global_scale": global_scale.astype(np.float32),
        "robust_bim_direct": bim_direct.astype(np.float32),
        "refined": predictions["refined"].astype(np.float32),
        "raw_absrel": raw_error,
        "direct_absrel": direct_error,
        "refined_absrel": refined_error,
        "refined_minus_direct": (refined_error - direct_error).astype(np.float32),
        "furniture_mask": subsets["furniture"].astype(np.uint8),
        "conflict_mask": subsets["bim_foreground_conflict"].astype(np.uint8),
        "bim_coverage": coverage,
        "fixed_support": fixed.astype(np.uint8),
        "bim_valid": bim_valid.astype(np.uint8),
        "local_correction_log_field": correction.astype(np.float32),
        "local_correction_support": correction_support.astype(np.float32),
    }
    for name in DIAGNOSTIC_NAMES:
        arrays[name] = np.asarray(model_arrays[name], dtype=np.float32)
    return arrays, {
        "methods": metrics,
        "subset_pixels": {name: int(mask.sum()) for name, mask in subsets.items()},
        "scale_estimate": {
            "estimator": estimate.estimator,
            "scale": estimate.scale,
            "support_count": estimate.support_count,
            "quantiles": list(estimate.quantiles),
            "fallback": estimate.fallback,
            "q10_cap_triggered": estimate.q10_cap_triggered,
            "q25_cap_triggered": estimate.q25_cap_triggered,
        },
    }


def panel_images(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    fixed = np.asarray(arrays["fixed_support"], dtype=bool)
    bim_valid = np.asarray(arrays["bim_valid"], dtype=bool)
    rgb = np.rint(np.clip(arrays["rgb"], 0.0, 1.0) * 255.0).astype(np.uint8)
    panels = {
        "rgb": rgb,
        "gt": _colorize_scalar(arrays["gt"], scale="depth", valid=fixed),
        "raw_da3": _colorize_scalar(arrays["raw_da3"], scale="depth"),
        "bim_depth": _colorize_scalar(arrays["bim_depth"], scale="depth", valid=bim_valid),
        "robust_global_scale": _colorize_scalar(arrays["robust_global_scale"], scale="depth"),
        "robust_bim_direct": _colorize_scalar(arrays["robust_bim_direct"], scale="depth"),
        "refined": _colorize_scalar(arrays["refined"], scale="depth"),
        "raw_absrel": _colorize_scalar(arrays["raw_absrel"], scale="absrel"),
        "direct_absrel": _colorize_scalar(arrays["direct_absrel"], scale="absrel"),
        "refined_absrel": _colorize_scalar(arrays["refined_absrel"], scale="absrel"),
        "refined_minus_direct": _colorize_scalar(
            arrays["refined_minus_direct"], scale="delta_absrel"
        ),
        "furniture_mask": _binary_panel(arrays["furniture_mask"]),
        "conflict_mask": _binary_panel(arrays["conflict_mask"]),
        "bim_coverage": _coverage_panel(arrays["bim_coverage"]),
        "reliability": _colorize_scalar(arrays["reliability"], scale="unit_interval"),
        "routing_gate": _colorize_scalar(arrays["routing_gate"], scale="unit_interval"),
        "total_log_residual": _colorize_scalar(arrays["total_log_residual"], scale="log_residual"),
        "frame_log_residual": _colorize_scalar(arrays["frame_log_residual"], scale="log_residual"),
        "low_log_residual": _colorize_scalar(arrays["low_log_residual"], scale="log_residual"),
        "detail_log_residual": _colorize_scalar(
            arrays["detail_log_residual"], scale="log_residual"
        ),
    }
    return panels


def write_contact_sheet(
    panel_paths: Mapping[str, Path],
    output: Path,
    *,
    ordered_names: Sequence[str] | None = None,
    columns: int = 5,
) -> None:
    names = list(ordered_names or panel_paths)
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.15 * columns, 3.35 * rows))
    axes_array = np.asarray(axes, dtype=object).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.axis("off")
    for axis, name in zip(axes_array.flat, names):
        axis.imshow(plt.imread(panel_paths[name]))
        axis.set_title(name.replace("_", " "), fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=140, facecolor="white")
    plt.close(fig)


def write_shared_colorbars(output_dir: Path) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, str]] = {}
    for name, spec in DISPLAY_SCALES.items():
        if "vcenter" in spec:
            norm = TwoSlopeNorm(
                vmin=float(spec["vmin"]),
                vcenter=float(spec["vcenter"]),
                vmax=float(spec["vmax"]),
            )
        else:
            norm = Normalize(float(spec["vmin"]), float(spec["vmax"]))
        fig, axis = plt.subplots(figsize=(5.4, 0.75))
        colorbar = fig.colorbar(
            matplotlib.cm.ScalarMappable(norm=norm, cmap=str(spec["cmap"])),
            cax=axis,
            orientation="horizontal",
        )
        colorbar.set_label(str(spec["label"]), fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        files: dict[str, str] = {}
        for suffix in ("png", "svg", "pdf"):
            path = output_dir / f"{name}_colorbar.{suffix}"
            fig.savefig(path, dpi=220, bbox_inches="tight", transparent=True)
            files[suffix] = str(path)
        plt.close(fig)
        artifacts[name] = files
    return artifacts


def _validate_preset_row_against_export(
    selection: Selection,
    metrics: Mapping[str, Any],
) -> None:
    if selection.csv_row is None:
        return
    row = selection.csv_row
    checks = {
        "all_gt_pixels": int(metrics["subset_pixels"]["all"]),
        "furniture_gt_pixels": int(metrics["subset_pixels"]["furniture"]),
        "all_robust_bim_direct_abs_rel": float(
            metrics["methods"]["robust_bim_direct"]["all"]["abs_rel"]
        ),
        "all_refined_abs_rel": float(metrics["methods"]["refined"]["all"]["abs_rel"]),
        "furniture_robust_bim_direct_abs_rel": float(
            metrics["methods"]["robust_bim_direct"]["furniture"]["abs_rel"]
        ),
        "furniture_refined_abs_rel": float(metrics["methods"]["refined"]["furniture"]["abs_rel"]),
    }
    mismatches = {}
    for key, actual in checks.items():
        expected = _required_float(row, key)
        tolerance = 0.0 if key.endswith("pixels") else 2e-6
        if abs(actual - expected) > tolerance:
            mismatches[key] = {"csv": expected, "recomputed": actual}
    if mismatches:
        raise RuntimeError(
            f"Preset CSV does not reproduce with the selected config/checkpoint: {mismatches}"
        )


def export_sample_assets(
    *,
    selection: Selection,
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    output_dir: Path,
    common_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    sample_dir = output_dir / selection.role
    sample_dir.mkdir(parents=True, exist_ok=True)
    panels = panel_images(arrays)
    panel_paths: dict[str, Path] = {}
    panel_receipts: dict[str, Any] = {}
    for name, image in panels.items():
        path = sample_dir / f"{name}.png"
        save_title_free_png(path, image)
        panel_paths[name] = path
        panel_receipts[name] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "pixel_dimensions": [PROTOCOL_WIDTH, PROTOCOL_HEIGHT],
            "title_or_axes_embedded": False,
        }

    arrays_path = sample_dir / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    metrics_path = sample_dir / "metrics.json"
    write_json(
        metrics_path,
        {
            "schema_version": 1,
            "protocol": "stanford-area1-frozen-validation-fixed-support-v1",
            "sample_id": selection.sample_id,
            "annotation_split": PROTOCOL_SPLIT,
            "depth_range_m": [PROTOCOL_MIN_DEPTH_M, PROTOCOL_MAX_DEPTH_M],
            **metrics,
        },
    )
    preview_path = sample_dir / "preview_contact_sheet.png"
    write_contact_sheet(panel_paths, preview_path)

    prepared_sample = Path(str(record["sample"])).expanduser().resolve()
    source_image = Path(str(record["image"])).expanduser().resolve()
    manifest_path = sample_dir / "manifest.json"
    manifest = {
        "schema_version": 1,
        "protocol": "stanford-area1-qualitative-material-export-v1",
        "purpose": (
            "Validation-only illustrative material for figure-layout selection; "
            "it is not held-out test evidence and must not be reported as such."
        ),
        "selection": {
            "role": selection.role,
            "sample_id": selection.sample_id,
            "annotation_split": PROTOCOL_SPLIT,
            "rule": selection.rule,
        },
        "sample": {
            "id": str(record["id"]),
            "region": str(record["region"]),
            "camera_uuid": str(record.get("camera_uuid", "")),
            "frame_number": record.get("frame_number"),
            "preparation_fingerprint_sha256": record.get("preparation_fingerprint_sha256"),
            "prepared_sample": {
                "path": str(prepared_sample),
                "sha256": file_sha256(prepared_sample),
            },
            "source_rgb": {"path": str(source_image), "sha256": file_sha256(source_image)},
        },
        "display_contract": {
            "all_panels_are_title_free_504x504_png": True,
            "invalid_scalar_pixels_are_black": True,
            "bim_coverage_colors": {
                "outside_fixed_support": [0, 0, 0],
                "no_hit": [242, 142, 43],
                "hit": [89, 161, 79],
            },
            "refined_minus_direct_semantics": (
                "refined per-pixel AbsRel minus universal BIM-direct per-pixel AbsRel; "
                "negative values mean learned refinement is better"
            ),
            "scales": DISPLAY_SCALES,
        },
        "provenance": dict(common_provenance),
        "artifacts": {
            "panels": panel_receipts,
            "arrays": {"path": str(arrays_path), "sha256": file_sha256(arrays_path)},
            "metrics": {"path": str(metrics_path), "sha256": file_sha256(metrics_path)},
            "preview_contact_sheet": {
                "path": str(preview_path),
                "sha256": file_sha256(preview_path),
                "selection_aid_only": True,
            },
        },
    }
    write_json(manifest_path, manifest)
    return {
        "role": selection.role,
        "sample_id": selection.sample_id,
        "directory": str(sample_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "panels": {name: str(path) for name, path in panel_paths.items()},
        "metrics": metrics,
    }


def write_batch_preview(exports: Sequence[Mapping[str, Any]], output: Path) -> None:
    names = (
        "rgb",
        "gt",
        "bim_depth",
        "robust_bim_direct",
        "refined",
        "refined_minus_direct",
    )
    fig, axes = plt.subplots(len(exports), len(names), figsize=(18, 3.25 * len(exports)))
    axes_array = np.asarray(axes, dtype=object).reshape(len(exports), len(names))
    for row_index, export in enumerate(exports):
        for column_index, name in enumerate(names):
            axis = axes_array[row_index, column_index]
            axis.imshow(plt.imread(export["panels"][name]))
            axis.axis("off")
            if row_index == 0:
                axis.set_title(name.replace("_", " "), fontsize=9)
            if column_index == 0:
                axis.set_ylabel(str(export["role"]).replace("_", " "), fontsize=9)
    fig.tight_layout(pad=0.45)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


def _preparation_provenance(cfg: Config, dataset: BIMDepthDataset) -> dict[str, Any]:
    manifest = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
    annotation = resolve_project_path(cfg, cfg.data.split_annotation)
    config_path = Path(cfg.config_path).resolve()
    return {
        "runtime_config": {
            "path": str(config_path),
            "raw_sha256": file_sha256(config_path),
            "resolved_config_sha256": resolved_config_sha256(cfg),
            "semantic_config_sha256": semantic_config_sha256(cfg),
        },
        "prepared_manifest": {
            "path": str(manifest),
            "raw_sha256": file_sha256(manifest),
            "preparation_fingerprint_status": dataset.split_provenance.get(
                "manifest_preparation_fingerprint_status"
            ),
            "preparation_fingerprint_sha256": dataset.split_provenance.get(
                "manifest_preparation_fingerprint_sha256"
            ),
        },
        "split_annotation": {
            "path": str(annotation),
            "raw_sha256": file_sha256(annotation),
            "configured_raw_sha256": cfg.data.get("split_annotation_sha256"),
            "configured_fingerprint_sha256": cfg.data.get("split_fingerprint_sha256"),
        },
        "validation_population_identity": dataset_split_identity(dataset.split_provenance),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(PROTOCOL_SEED)
    cfg = load_config(args.config)
    require_locked_protocol(cfg)
    dataset = BIMDepthDataset(cfg, PROTOCOL_SPLIT, augment=False)

    selection_source: dict[str, Any]
    if args.preset == PRESET_NAME:
        selection_csv = args.selection_csv.expanduser().resolve()
        rows = read_selection_csv(selection_csv)
        validate_selection_csv_population(rows, dataset.records)
        selections = select_three_options(rows)
        selection_source = {
            "mode": "deterministic_validation_preset",
            "preset": PRESET_NAME,
            "validation_csv": {"path": str(selection_csv), "sha256": file_sha256(selection_csv)},
            "rules": SELECTION_RULES,
            "rules_sha256": canonical_sha256(SELECTION_RULES),
            "csv_population_exactly_matches_runtime_validation": True,
            "test_results_read": False,
        }
    else:
        assert args.sample_id is not None
        selections = [
            Selection(
                role=f"explicit_{index:02d}",
                sample_id=query,
                rule={
                    "population": "runtime validation annotations only",
                    "choice": "explicit user-supplied sample ID; exact or unambiguous basename",
                },
            )
            for index, query in enumerate(args.sample_id, start=1)
        ]
        selection_source = {
            "mode": "explicit_validation_ids",
            "queries": list(args.sample_id),
            "test_results_read": False,
        }

    resolved: list[tuple[Selection, int, str]] = []
    seen: set[str] = set()
    for selection in selections:
        index, sample_id = resolve_validation_sample(dataset.records, selection.sample_id)
        if sample_id in seen:
            raise ValueError(f"Duplicate resolved validation sample: {sample_id}")
        seen.add(sample_id)
        resolved.append(
            (
                Selection(selection.role, sample_id, selection.rule, selection.csv_row),
                index,
                sample_id,
            )
        )

    device = resolve_device(args.device)
    model, checkpoint_receipt = load_validated_frozen_model(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        split_provenance=dataset.split_provenance,
        device=device,
    )
    preparation_receipt = _preparation_provenance(cfg, dataset)
    common_provenance = {
        **preparation_receipt,
        "checkpoint": checkpoint_receipt,
        "selection_source": selection_source,
        "determinism": {
            "seed": PROTOCOL_SEED,
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "augmentation": False,
            "checkpoint_map_location": "cpu",
            "model_load_count": 1,
        },
        "exporter": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    colorbars = write_shared_colorbars(output_dir / "shared_colorbars")
    exports = []
    for selection, index, _sample_id in resolved:
        item = dataset[index]
        record = dataset.records[index]
        model_arrays = infer_frozen_sample(model, item, device)
        arrays, metrics = build_export_arrays(
            item=item,
            model_arrays=model_arrays,
            scale_parameters=cfg.model.get("scale_estimator"),
        )
        _validate_preset_row_against_export(selection, metrics)
        exports.append(
            export_sample_assets(
                selection=selection,
                record=record,
                arrays=arrays,
                metrics=metrics,
                output_dir=output_dir,
                common_provenance=common_provenance,
            )
        )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    batch_preview = output_dir / "three_options_preview_contact_sheet.png"
    write_batch_preview(exports, batch_preview)
    root_manifest = {
        "schema_version": 1,
        "protocol": "stanford-area1-qualitative-material-export-v1",
        "annotation_split": PROTOCOL_SPLIT,
        "illustrative_validation_only": True,
        "held_out_test_evidence": False,
        "test_results_read": False,
        "selection_source": selection_source,
        "exports": [
            {key: value for key, value in export.items() if key not in {"panels", "metrics"}}
            for export in exports
        ],
        "shared_colorbars": colorbars,
        "batch_preview": {
            "path": str(batch_preview),
            "sha256": file_sha256(batch_preview),
            "selection_aid_only": True,
        },
        "provenance": common_provenance,
    }
    root_manifest_path = output_dir / "manifest.json"
    write_json(root_manifest_path, root_manifest)
    print(json.dumps(_json_safe(root_manifest), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
