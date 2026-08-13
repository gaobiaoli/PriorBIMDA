#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize, TwoSlopeNorm

from bim_priorda3.baselines import bim_scale_and_local_features
from bim_priorda3.checkpoints import validate_checkpoint_model_config
from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.metrics import depth_metrics
from bim_priorda3.models import BIMPriorDA3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize fixed BIM baselines, a frozen-DA3 learned refiner, and "
            "a partially fine-tuned E2E model on one annotated sample."
        )
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--frozen-config", required=True)
    parser.add_argument("--frozen-checkpoint", required=True, type=Path)
    parser.add_argument("--e2e-config", required=True)
    parser.add_argument("--e2e-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/visualizations/prediction_sample"),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def normalize_sample_id(sample_id: str) -> str:
    """Accept either annotation IDs or filename-friendly sample IDs."""
    if "/" in sample_id:
        return sample_id
    region, separator, frame = sample_id.rpartition("_")
    if separator and region and frame.isdigit():
        return f"{region}/{frame}"
    return sample_id


def add_batch_dimension(item: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.unsqueeze(0).to(device) if torch.is_tensor(value) else value
        for key, value in item.items()
    }


def find_item(dataset: BIMDepthDataset, sample_id: str) -> dict[str, Any]:
    for index, record in enumerate(dataset.records):
        if str(record["id"]) == sample_id:
            return dataset[index]
    raise KeyError(f"Sample {sample_id!r} is not in the active annotation population")


def read_annotation_split(cfg: dict[str, Any], sample_id: str) -> str:
    annotation = cfg.data.get("split_annotation")
    if not annotation:
        return "region-defined"
    path = resolve_project_path(cfg, annotation)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("id")) == sample_id:
                return str(record["split"])
    raise KeyError(f"Sample {sample_id!r} is absent from {path}")


def infer_model(
    config_path: str | Path,
    checkpoint_path: Path,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray | float | bool]:
    cfg = load_config(config_path)
    model = BIMPriorDA3(cfg).to(device)
    checkpoint_path = checkpoint_path.expanduser().resolve()
    # Keep optimizer/scheduler state in a full training checkpoint off the GPU.
    # load_state_dict copies only the model tensors to the model's device.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_model_config(state, cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(batch)

    arrays: dict[str, np.ndarray | float | bool] = {
        "depth": output["depth"][0, 0].detach().float().cpu().numpy(),
        "base_depth": output["base_depth"][0, 0].detach().float().cpu().numpy(),
        "scaled_depth": output["scaled_depth"][0, 0].detach().float().cpu().numpy(),
        "reliability": output["bim_reliability"][0, 0].detach().float().cpu().numpy(),
        "log_residual": output["log_residual"][0, 0].detach().float().cpu().numpy(),
        "routing_gate": output["residual_routing_gate"][0, 0].detach().float().cpu().numpy(),
        "uses_live_da3": bool(output.get("uses_live_da3", False)),
    }
    if "da3_scale" in output:
        arrays["da3_scale"] = float(output["da3_scale"].mean().detach().cpu())
    del state, output, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays


def metric_dict(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    result = depth_metrics(
        torch.from_numpy(prediction)[None, None],
        torch.from_numpy(target)[None, None],
        torch.from_numpy(valid)[None, None],
    )
    return {key: int(value) if key == "count" else float(value) for key, value in result.items()}


def masked_depth(depth: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    result = depth.astype(np.float32).copy()
    mask = np.isfinite(result) & (result > 0)
    if valid is not None:
        mask &= valid
    result[~mask] = np.nan
    return result


def relative_error(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    result = np.full(target.shape, np.nan, dtype=np.float32)
    mask = valid & np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
    result[mask] = np.abs(prediction[mask] - target[mask]) / target[mask]
    return result


def render_comparison(
    result: dict[str, Any],
    output: Path,
) -> None:
    depth_norm = Normalize(vmin=result["depth_min"], vmax=result["depth_max"])
    error_norm = Normalize(vmin=0.0, vmax=0.5)
    delta_norm = TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
    depth_cmap = plt.get_cmap("turbo").copy()
    error_cmap = plt.get_cmap("magma").copy()
    delta_cmap = plt.get_cmap("coolwarm").copy()
    for cmap in (depth_cmap, error_cmap, delta_cmap):
        cmap.set_bad("black")

    panels = (
        ("rgb", "RGB", "rgb"),
        ("gt", "Fused LiDAR GT", "depth_gt"),
        ("base", "A. Frozen DA3", "depth"),
        ("bim", "BIM depth", "depth_bim"),
        ("global_scale", "B. BIM global scale", "depth"),
        ("direct", "C. BIM direct", "depth"),
        ("frozen", "D. Frozen learned", "depth"),
        ("e2e", "E. Partial E2E", "depth"),
        ("direct_error", "C relative error", "error"),
        ("frozen_error", "D relative error", "error"),
        ("e2e_error", "E relative error", "error"),
        ("delta_error", "E - C error (blue=better)", "delta"),
    )
    fig, axes = plt.subplots(3, 4, figsize=(20, 14), constrained_layout=True)
    last_depth = last_error = last_delta = None
    for axis, (key, title, kind) in zip(axes.flat, panels):
        value = result[key]
        if kind == "rgb":
            axis.imshow(np.clip(value, 0.0, 1.0))
        elif kind == "depth_gt":
            last_depth = axis.imshow(
                masked_depth(value, result["gt_valid"]),
                cmap=depth_cmap,
                norm=depth_norm,
            )
        elif kind == "depth_bim":
            last_depth = axis.imshow(
                masked_depth(value, result["bim_valid"]),
                cmap=depth_cmap,
                norm=depth_norm,
            )
        elif kind == "depth":
            last_depth = axis.imshow(
                masked_depth(value),
                cmap=depth_cmap,
                norm=depth_norm,
            )
            metrics = result["metrics"][key]
            title += f"\nAbsRel={metrics['abs_rel']:.4f}, RMSE={metrics['rmse']:.3f}"
        elif kind == "error":
            last_error = axis.imshow(value, cmap=error_cmap, norm=error_norm)
        else:
            last_delta = axis.imshow(value, cmap=delta_cmap, norm=delta_norm)
        axis.set_title(title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])

    if last_depth is not None:
        fig.colorbar(
            last_depth,
            ax=axes[:2, :].ravel().tolist(),
            shrink=0.62,
            label="Depth (m)",
        )
    if last_error is not None:
        fig.colorbar(
            last_error,
            ax=axes[2, :3].ravel().tolist(),
            shrink=0.72,
            label="Absolute relative error",
        )
    if last_delta is not None:
        fig.colorbar(
            last_delta,
            ax=[axes[2, 3]],
            shrink=0.72,
            label="E2E error - BIM direct error",
        )
    evaluation_note = "held-out evaluation" if result["split"] == "test" else "qualitative only"
    fig.suptitle(
        f"{result['sample_id']} ({result['split']} split, {evaluation_note})\n"
        f"fixed BIM scale={result['fixed_scale']:.4f}, "
        f"live E2E scale={result['live_scale']:.4f}",
        fontsize=15,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_diagnostics(result: dict[str, Any], output: Path) -> None:
    depth_norm = Normalize(vmin=result["depth_min"], vmax=result["depth_max"])
    depth_cmap = plt.get_cmap("turbo").copy()
    unit_cmap = plt.get_cmap("viridis").copy()
    depth_cmap.set_bad("black")
    unit_cmap.set_bad("black")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = (
        ("live_da3", "Live DA3", "depth"),
        ("live_scaled", "Live DA3 + BIM scale", "depth"),
        ("e2e", "Final E2E", "depth"),
        ("bim_valid", "BIM valid", "unit"),
        ("reliability", "Learned BIM reliability", "unit"),
        ("log_residual", "Learned log residual", "residual"),
    )
    for axis, (key, title, kind) in zip(axes.flat, panels):
        value = result[key]
        if kind == "depth":
            image = axis.imshow(masked_depth(value), cmap=depth_cmap, norm=depth_norm)
            fig.colorbar(image, ax=axis, shrink=0.72)
        elif kind == "unit":
            if key == "reliability":
                value = np.where(result["bim_valid"], value, np.nan)
            image = axis.imshow(value, cmap=unit_cmap, vmin=0.0, vmax=1.0)
            fig.colorbar(image, ax=axis, shrink=0.72)
        else:
            finite = np.abs(value[np.isfinite(value)])
            residual_limit = max(0.05, float(finite.max(initial=0.0)))
            image = axis.imshow(
                value,
                cmap="coolwarm",
                vmin=-residual_limit,
                vmax=residual_limit,
            )
            fig.colorbar(image, ax=axis, shrink=0.72)
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(f"E2E diagnostics — {result['sample_id']}", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.sample_id = normalize_sample_id(args.sample_id)
    e2e_cfg = load_config(args.e2e_config)
    split = read_annotation_split(e2e_cfg, args.sample_id)
    if split == "excluded":
        raise ValueError(f"Refusing to visualize excluded sample {args.sample_id!r}")
    dataset = BIMDepthDataset(
        e2e_cfg,
        split=None,
        augment=False,
        require_ground_truth=True,
    )
    item = find_item(dataset, args.sample_id)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    batch = add_batch_dimension(item, device)

    frozen_output = infer_model(
        args.frozen_config,
        args.frozen_checkpoint,
        batch,
        device,
    )
    e2e_output = infer_model(
        args.e2e_config,
        args.e2e_checkpoint,
        batch,
        device,
    )
    if not bool(e2e_output["uses_live_da3"]):
        raise RuntimeError("The configured E2E model did not run live DA3")

    def item_array(key: str) -> np.ndarray:
        return item[key][0].detach().float().cpu().numpy()

    rgb = item["rgb"].detach().float().cpu().numpy().transpose(1, 2, 0)
    base = item_array("base_depth")
    bim = item_array("bim_depth")
    gt = item_array("gt_depth")
    depth_min = float(e2e_cfg.data.min_depth)
    depth_max = float(e2e_cfg.data.max_depth)
    gt_valid = (
        (item_array("gt_valid") > 0) & np.isfinite(gt) & (gt >= depth_min) & (gt <= depth_max)
    )
    bim_valid = item_array("bim_valid") > 0
    global_scale, direct, _, _, fixed_scale = bim_scale_and_local_features(base, bim)
    frozen = np.asarray(frozen_output["depth"])
    e2e = np.asarray(e2e_output["depth"])
    live_da3 = np.asarray(e2e_output["base_depth"])
    live_scaled = np.asarray(e2e_output["scaled_depth"])
    predictions = {
        "base": base,
        "global_scale": global_scale,
        "direct": direct,
        "frozen": frozen,
        "live_da3": live_da3,
        "live_scaled": live_scaled,
        "e2e": e2e,
    }
    metrics = {
        name: metric_dict(prediction, gt, gt_valid) for name, prediction in predictions.items()
    }
    direct_error = relative_error(direct, gt, gt_valid)
    frozen_error = relative_error(frozen, gt, gt_valid)
    e2e_error = relative_error(e2e, gt, gt_valid)
    delta_error = e2e_error - direct_error
    result = {
        "sample_id": args.sample_id,
        "split": split,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "rgb": rgb,
        "gt": gt,
        "gt_valid": gt_valid,
        "bim": bim,
        "bim_valid": bim_valid,
        **predictions,
        "direct_error": direct_error,
        "frozen_error": frozen_error,
        "e2e_error": e2e_error,
        "delta_error": delta_error,
        "reliability": np.asarray(e2e_output["reliability"]),
        "log_residual": np.asarray(e2e_output["log_residual"]),
        "routing_gate": np.asarray(e2e_output["routing_gate"]),
        "fixed_scale": fixed_scale,
        "live_scale": float(e2e_output["da3_scale"]),
        "metrics": metrics,
    }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "model_comparison.png"
    diagnostics_path = output_dir / "e2e_diagnostics.png"
    render_comparison(result, comparison_path)
    render_diagnostics(result, diagnostics_path)
    np.savez_compressed(
        output_dir / "predictions.npz",
        **{name: value.astype(np.float32) for name, value in predictions.items()},
        gt=gt.astype(np.float32),
        gt_valid=gt_valid.astype(np.uint8),
        bim=bim.astype(np.float32),
        bim_valid=bim_valid.astype(np.uint8),
        reliability=np.asarray(e2e_output["reliability"]).astype(np.float32),
        log_residual=np.asarray(e2e_output["log_residual"]).astype(np.float32),
    )
    receipt = {
        "sample_id": args.sample_id,
        "annotation_split": split,
        "protocol_depth_m": [
            depth_min,
            depth_max,
        ],
        "qualitative_only": split != "test",
        "fixed_bim_scale": fixed_scale,
        "live_e2e_scale": float(e2e_output["da3_scale"]),
        "valid_gt_pixels": int(gt_valid.sum()),
        "metrics": metrics,
        "frozen_config": str(Path(args.frozen_config).expanduser().resolve()),
        "frozen_checkpoint": str(args.frozen_checkpoint.expanduser().resolve()),
        "e2e_config": str(Path(args.e2e_config).expanduser().resolve()),
        "e2e_checkpoint": str(args.e2e_checkpoint.expanduser().resolve()),
        "comparison_figure": str(comparison_path),
        "diagnostics_figure": str(diagnostics_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
