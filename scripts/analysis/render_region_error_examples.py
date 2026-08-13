#!/usr/bin/env python3
"""Render real fold-specific predictions for region-error case studies."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from bim_priorda3.baselines import estimate_bim_scale
from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.metrics import depth_metrics
from bim_priorda3.models import BIMPriorDA3

EXAMPLES = (
    {
        "sample_id": "5F_Region2/000128",
        "fold": "fold_04_5F_Region2",
        "case": "Low-error region: direct already strong",
    },
    {
        "sample_id": "3F_Region2/000270",
        "fold": "fold_00_3F_Region2",
        "case": "Hard region: learned correction helps",
    },
    {
        "sample_id": "5F_Region3/000068",
        "fold": "fold_05_5F_Region3",
        "case": "Scale-anchor outlier",
    },
    {
        "sample_id": "3F_Region3/000138",
        "fold": "fold_01_3F_Region3",
        "case": "Dark glass: learned over-correction",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-root",
        type=Path,
        default=Path("outputs/slabim_region_cv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/analysis/region_error_analysis"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_batch_dimension(item: dict[str, Any]) -> dict[str, Any]:
    batch = {}
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, (int, float)):
            batch[key] = torch.tensor([value])
        else:
            batch[key] = [value]
    return batch


def find_sample(dataset: BIMDepthDataset, sample_id: str) -> dict[str, Any]:
    for index, record in enumerate(dataset.records):
        if record["id"] == sample_id:
            return dataset[index]
    raise KeyError(f"{sample_id} is not part of this fold's test split")


def masked_depth(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = depth.astype(np.float32).copy()
    mask = np.isfinite(result) & (result > 0)
    if valid is not None:
        mask &= valid
    result[~mask] = np.nan
    return result


def log_error(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    error = np.full_like(target, np.nan, dtype=np.float32)
    mask = valid & np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
    error[mask] = np.abs(
        np.log(np.maximum(prediction[mask], 1e-6)) - np.log(np.maximum(target[mask], 1e-6))
    )
    return error


def infer_example(
    cv_root: Path,
    example: dict[str, str],
    device: torch.device,
) -> dict[str, Any]:
    fold = example["fold"]
    config_path = cv_root / "configs" / f"{fold}__seed_42__final.yaml"
    checkpoint_path = cv_root / "folds" / fold / "seed_42" / "best.pt"
    cfg = load_config(config_path)
    dataset = BIMDepthDataset(cfg, "test", augment=False)
    item = find_sample(dataset, example["sample_id"])
    batch = add_batch_dimension(item)
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    model = BIMPriorDA3(cfg).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(tensor_batch)

    def array(key: str, source: dict[str, Any] = tensor_batch) -> np.ndarray:
        return source[key][0, 0].detach().float().cpu().numpy()

    rgb = tensor_batch["rgb"][0].detach().float().cpu().numpy().transpose(1, 2, 0)
    base = array("base_depth")
    scaled = array("scaled_depth")
    direct = array("anchor_depth")
    bim = array("bim_depth")
    bim_valid = array("bim_valid") > 0
    target = array("gt_depth")
    valid = array("gt_valid") > 0
    learned = array("depth", output)
    reliability = array("bim_reliability", output)
    residual = array("log_residual", output)
    support = array("support", output)
    direct_error = log_error(direct, target, valid)
    learned_error = log_error(learned, target, valid)
    delta_error = learned_error - direct_error
    gt_for_scale = np.where(valid, target, 0.0)
    actual_scale = estimate_bim_scale(base, bim)
    oracle_scale = estimate_bim_scale(base, gt_for_scale)

    valid_tensor = tensor_batch["gt_valid"] > 0
    direct_metrics = depth_metrics(
        tensor_batch["anchor_depth"],
        tensor_batch["gt_depth"],
        valid_tensor,
    )
    learned_metrics = depth_metrics(
        output["depth"],
        tensor_batch["gt_depth"],
        valid_tensor,
    )
    return {
        **example,
        "rgb": rgb,
        "base": base,
        "scaled": scaled,
        "direct": direct,
        "bim": bim,
        "bim_valid": bim_valid,
        "target": target,
        "valid": valid,
        "learned": learned,
        "reliability": reliability,
        "residual": residual,
        "support": support,
        "direct_error": direct_error,
        "learned_error": learned_error,
        "delta_error": delta_error,
        "actual_scale": actual_scale,
        "oracle_scale": oracle_scale,
        "direct_metrics": direct_metrics,
        "learned_metrics": learned_metrics,
    }


def render_main_grid(results: list[dict[str, Any]], output: Path) -> None:
    columns = (
        ("rgb", "RGB input"),
        ("base", "Raw DA3"),
        ("bim", "BIM depth"),
        ("scaled", "Global scale"),
        ("direct", "Direct BIM"),
        ("learned", "Learned"),
        ("target", "LiDAR GT"),
        ("direct_error", "Direct |log error|"),
        ("learned_error", "Learned |log error|"),
        ("delta_error", "Learned - direct error"),
    )
    depth_norm = Normalize(0.2, 5.0)
    error_norm = Normalize(0.0, 0.5)
    delta_norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
    fig, axes = plt.subplots(
        len(results),
        len(columns),
        figsize=(25, 10.5),
        constrained_layout=True,
    )
    for row_index, result in enumerate(results):
        for column_index, (key, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            if key == "rgb":
                axis.imshow(np.clip(result[key], 0.0, 1.0))
            elif key == "bim":
                axis.imshow(
                    masked_depth(result[key], result["bim_valid"]),
                    cmap="turbo",
                    norm=depth_norm,
                )
            elif key in {"base", "scaled", "direct", "learned"}:
                axis.imshow(
                    masked_depth(result[key]),
                    cmap="turbo",
                    norm=depth_norm,
                )
            elif key == "target":
                axis.imshow(
                    masked_depth(result[key], result["valid"]),
                    cmap="turbo",
                    norm=depth_norm,
                )
            elif key in {"direct_error", "learned_error"}:
                axis.imshow(result[key], cmap="magma", norm=error_norm)
            else:
                axis.imshow(result[key], cmap="coolwarm", norm=delta_norm)
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

        direct = result["direct_metrics"]
        learned = result["learned_metrics"]
        label = (
            f"{result['sample_id']} — {result['case']}\n"
            f"scale BIM/oracle={result['actual_scale']:.3f}/"
            f"{result['oracle_scale']:.3f}; "
            f"AbsRel {direct['abs_rel']:.3f} -> {learned['abs_rel']:.3f}"
        )
        axes[row_index, 0].text(
            0.02,
            0.98,
            label,
            transform=axes[row_index, 0].transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=7.5,
            bbox={"facecolor": "black", "alpha": 0.62, "pad": 2.5},
        )
    fig.suptitle(
        "Why visually similar regions have different depth errors "
        "(fold-specific seed-42 predictions)",
        fontsize=14,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def render_failure_diagnostics(
    results: list[dict[str, Any]],
    output: Path,
) -> None:
    columns = (
        ("rgb", "RGB"),
        ("bim_valid", "BIM valid"),
        ("support", "Direct support"),
        ("reliability", "Learned BIM reliability"),
        ("residual", "Learned log residual"),
        ("delta_error", "Learned - direct error"),
    )
    fig, axes = plt.subplots(
        len(results),
        len(columns),
        figsize=(15, 10),
        constrained_layout=True,
    )
    for row_index, result in enumerate(results):
        for column_index, (key, title) in enumerate(columns):
            axis = axes[row_index, column_index]
            if key == "rgb":
                axis.imshow(np.clip(result[key], 0.0, 1.0))
            elif key in {"bim_valid", "support", "reliability"}:
                axis.imshow(result[key], cmap="viridis", vmin=0.0, vmax=1.0)
            elif key == "residual":
                axis.imshow(result[key], cmap="coolwarm", vmin=-0.2, vmax=0.2)
            else:
                axis.imshow(result[key], cmap="coolwarm", vmin=-0.3, vmax=0.3)
            if row_index == 0:
                axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row_index, 0].set_ylabel(result["sample_id"], fontsize=9)
    fig.suptitle(
        "BIM support and learned correction diagnostics "
        "(blue error delta = improvement, red = degradation)",
        fontsize=13,
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def write_metrics(results: list[dict[str, Any]], output: Path) -> None:
    fields = (
        "sample_id",
        "case",
        "actual_scale",
        "oracle_scale",
        "direct_abs_rel",
        "learned_abs_rel",
        "direct_rmse",
        "learned_rmse",
        "direct_mae",
        "learned_mae",
        "direct_delta1",
        "learned_delta1",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            direct = result["direct_metrics"]
            learned = result["learned_metrics"]
            writer.writerow(
                {
                    "sample_id": result["sample_id"],
                    "case": result["case"],
                    "actual_scale": result["actual_scale"],
                    "oracle_scale": result["oracle_scale"],
                    "direct_abs_rel": direct["abs_rel"],
                    "learned_abs_rel": learned["abs_rel"],
                    "direct_rmse": direct["rmse"],
                    "learned_rmse": learned["rmse"],
                    "direct_mae": direct["mae"],
                    "learned_mae": learned["mae"],
                    "direct_delta1": direct["delta1"],
                    "learned_delta1": learned["delta1"],
                }
            )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested_device = args.device
    device = torch.device(
        requested_device
        if requested_device.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    print(f"render device: {device}", flush=True)
    results = []
    for example in EXAMPLES:
        print(f"running {example['sample_id']}", flush=True)
        results.append(infer_example(args.cv_root.resolve(), example, device))
    render_main_grid(results, output / "region_prediction_examples.png")
    render_failure_diagnostics(
        results,
        output / "region_failure_diagnostics.png",
    )
    write_metrics(results, output / "example_metrics.csv")
    print(f"wrote figures to {output}", flush=True)


if __name__ == "__main__":
    main()
