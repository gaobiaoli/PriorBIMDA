#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from bim_priorda3.baselines import PREVIOUS_FIXED_PARAMETERS, previous_scale_baselines
from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import build_loader, move_batch
from bim_priorda3.metrics import depth_metrics
from bim_priorda3.models import BIMPriorDA3


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BIM-PriorDA3 on held-out regions")
    parser.add_argument("--config", default="configs/slabim_single_frame.yaml")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    loader = build_loader(dataset, 1, int(cfg.train.num_workers), shuffle=False)
    model = BIMPriorDA3(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    methods = ("base", "global_scale", "previous_scale_local", "coarse", "refined")
    rows = []
    overall_accumulators = {method: MetricAccumulator() for method in methods}
    suffix_accumulators = {method: MetricAccumulator() for method in methods}
    bins = [float(value) for value in cfg.evaluation.distance_bins]
    range_accumulators = {
        f"{lower:g}-{upper:g}m": {
            method: MetricAccumulator() for method in methods
        }
        for lower, upper in zip(bins[:-1], bins[1:])
    }
    suffix_start = max(0, len(dataset) - 82)
    with torch.no_grad():
        for sample_index, batch in enumerate(loader):
            identifiers = batch["sample_id"]
            batch = move_batch(batch, device)
            output = model(batch)
            valid = batch["gt_valid"] > 0
            row = {"sample_id": identifiers[0], "valid_pixels": int(valid.sum())}
            scaled_np, local_np, scale = previous_scale_baselines(
                batch["base_depth"][0, 0].cpu().numpy(),
                batch["bim_depth"][0, 0].cpu().numpy(),
            )
            scaled = torch.from_numpy(scaled_np)[None, None].to(device)
            previous_local = torch.from_numpy(local_np)[None, None].to(device)
            row["previous_bim_scale"] = scale
            row["learned_frame_trust"] = float(
                torch.sigmoid(output["frame_trust_logits"]).mean().cpu()
            )
            row["learned_mean_pixel_trust"] = float(
                output["trust_probability"].mean().cpu()
            )
            row["learned_mean_variance"] = float(
                torch.exp(output["log_variance"]).mean().cpu()
            )
            row["learned_mean_abs_log_residual"] = float(
                output["log_residual"].abs().mean().cpu()
            )
            row["learned_mean_support"] = float(output["support"].mean().cpu())
            for method, prediction in (
                ("base", batch["base_depth"]),
                ("global_scale", scaled),
                ("previous_scale_local", previous_local),
                ("coarse", output["coarse_depth"]),
                ("refined", output["depth"]),
            ):
                metric = depth_metrics(prediction, batch["gt_depth"], valid)
                row.update({f"{method}_{key}": value for key, value in metric.items()})
                overall_accumulators[method].update(
                    prediction, batch["gt_depth"], valid
                )
                if sample_index >= suffix_start:
                    suffix_accumulators[method].update(
                        prediction, batch["gt_depth"], valid
                    )
                for lower, upper in zip(bins[:-1], bins[1:]):
                    name = f"{lower:g}-{upper:g}m"
                    range_mask = (
                        valid
                        & (batch["gt_depth"] >= lower)
                        & (batch["gt_depth"] < upper)
                    )
                    range_accumulators[name][method].update(
                        prediction, batch["gt_depth"], range_mask
                    )
            rows.append(row)
            print(
                f"{identifiers[0]} AbsRel "
                f"{row['base_abs_rel']:.4f}->{row['refined_abs_rel']:.4f}",
                flush=True,
            )

    output_dir = args.output or (
        resolve_project_path(cfg, cfg.experiment.output_dir) / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "regions": list(
            cfg.data.test_regions if args.split == "test" else cfg.data.val_regions
        ),
        "previous_scale_parameters": PREVIOUS_FIXED_PARAMETERS,
        "overall": {
            method: accumulator.compute()
            for method, accumulator in overall_accumulators.items()
        },
        "chronological_suffix_82": {
            method: accumulator.compute()
            for method, accumulator in suffix_accumulators.items()
        },
        "distance_ranges": {
            name: {
                method: accumulator.compute()
                for method, accumulator in method_accumulators.items()
            }
            for name, method_accumulators in range_accumulators.items()
        },
    }
    base_abs_rel = summary["overall"]["base"]["abs_rel"]
    refined_abs_rel = summary["overall"]["refined"]["abs_rel"]
    summary["abs_rel_relative_improvement"] = (
        (base_abs_rel - refined_abs_rel) / base_abs_rel if base_abs_rel > 0 else np.nan
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
