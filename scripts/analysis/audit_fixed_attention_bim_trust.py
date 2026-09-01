#!/usr/bin/env python3
"""Audit whether fixed-attention scale tokens correspond to accurate BIM depth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional

from bim_priorda3.config import load_config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import build_loader, move_batch, seed_everything
from bim_priorda3.models import BIMPriorDA3


class AbsRelSums:
    def __init__(self) -> None:
        self.error_sum = 0.0
        self.count = 0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        support: torch.Tensor,
    ) -> None:
        values = ((prediction - target).abs() / target.clamp_min(1e-6))[support]
        self.error_sum += float(values.double().sum().item())
        self.count += int(values.numel())

    def compute(self, reference_count: int | None = None) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "abs_rel": self.error_sum / self.count if self.count else float("nan"),
            "count": self.count,
        }
        if reference_count is not None:
            result["support_fraction"] = (
                self.count / reference_count if reference_count else float("nan")
            )
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--top-fractions",
        type=float,
        nargs="+",
        default=(0.10, 0.25, 0.50),
    )
    return parser.parse_args()


def top_token_mask(
    distribution: torch.Tensor,
    valid: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select the top fraction independently in every frame."""

    selected = torch.zeros_like(valid, dtype=torch.bool)
    for batch_index in range(distribution.shape[0]):
        flat_distribution = distribution[batch_index].flatten()
        flat_valid = valid[batch_index].flatten().bool()
        valid_indices = torch.nonzero(flat_valid, as_tuple=False).flatten()
        if not valid_indices.numel():
            continue
        count = max(1, math.ceil(float(fraction) * int(valid_indices.numel())))
        ranked = torch.topk(flat_distribution[valid_indices], count).indices
        selected[batch_index].view(-1)[valid_indices[ranked]] = True
    return selected


def weighted_frame_absrel(
    prediction: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    weights: torch.Tensor,
) -> list[float]:
    relative_error = (prediction - target).abs() / target.clamp_min(1e-6)
    values: list[float] = []
    for index in range(prediction.shape[0]):
        weight = weights[index] * support[index].float()
        denominator = weight.double().sum()
        if float(denominator.item()) > 0:
            numerator = (relative_error[index].double() * weight.double()).sum()
            values.append(float((numerator / denominator).item()))
    return values


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if any(not 0.0 < value <= 1.0 for value in args.top_fractions):
        raise ValueError("--top-fractions values must lie in (0, 1]")

    cfg = load_config(args.config)
    seed_everything(int(getattr(cfg.experiment, "seed", 42)))
    dataset = BIMDepthDataset(cfg, args.split, augment=False)
    loader = build_loader(
        dataset,
        args.batch_size,
        args.num_workers,
        shuffle=False,
    )

    checkpoint = args.checkpoint.expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = BIMPriorDA3(cfg)
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    head = model.attention_scale
    if head is None or bool(head.iterative_refresh_attention):
        raise ValueError("This audit requires a fixed-attention scale head")

    policies = ("learned_attention", "huber_effective_attention")
    fraction_keys = [f"top_{round(value * 100):02d}_percent" for value in args.top_fractions]
    all_hit = AbsRelSums()
    all_eligible = AbsRelSums()
    selected = {
        policy: {
            key: {"bim_hit": AbsRelSums(), "scale_eligible": AbsRelSums()}
            for key in fraction_keys
        }
        for policy in policies
    }
    weighted_frame_values = {policy: [] for policy in policies}

    processed = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            scale_output = model._estimate_attention_scale(batch, batch["base_depth"])
            attention = scale_output["attention_token_distribution"].float()
            head_mixture = scale_output["head_mixture"].float()
            token_valid = scale_output["attention_token_valid"].squeeze(1).bool()
            learned_distribution = (
                attention * head_mixture[:, :, None, None]
            ).sum(dim=1)

            base = batch["base_depth"]
            bim = batch["bim_depth"]
            gt = batch["gt_depth"]
            gt_valid = (batch["gt_valid"] > 0) & torch.isfinite(gt) & (gt > 0)
            bim_hit = (batch["bim_valid"] > 0) & torch.isfinite(bim) & (bim > 0)
            ratio = bim / base.clamp_min(1e-6)
            scale_eligible = (
                bim_hit
                & torch.isfinite(base)
                & torch.isfinite(ratio)
                & (base > 0)
                & (ratio > float(cfg.model.attention_scale.ratio_min))
                & (ratio < float(cfg.model.attention_scale.ratio_max))
            )
            hit_support = gt_valid & bim_hit
            eligible_support = gt_valid & scale_eligible
            all_hit.update(bim, gt, hit_support)
            all_eligible.update(bim, gt, eligible_support)

            valid_float = scale_eligible.float()
            token_fraction = functional.adaptive_avg_pool2d(
                valid_float,
                learned_distribution.shape[-2:],
            )
            token_numerator = functional.adaptive_avg_pool2d(
                torch.where(scale_eligible, ratio.clamp_min(1e-6).log(), 0.0),
                learned_distribution.shape[-2:],
            )
            token_log_ratio = (token_numerator / token_fraction.clamp_min(1e-6)).squeeze(1)
            head_center = scale_output["head_log_scale"].float()
            residual = token_log_ratio[:, None] - head_center[:, :, None, None]
            robust_weight = torch.rsqrt(1.0 + (residual / float(head.huber_delta)).square())
            effective_per_head = attention * robust_weight
            effective_per_head = effective_per_head / effective_per_head.flatten(2).sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-8).unsqueeze(-1)
            effective_distribution = (
                effective_per_head * head_mixture[:, :, None, None]
            ).sum(dim=1)

            distributions = {
                "learned_attention": learned_distribution,
                "huber_effective_attention": effective_distribution,
            }
            output_size = gt.shape[-2:]
            for policy, distribution in distributions.items():
                pixel_weights = functional.interpolate(
                    distribution[:, None],
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
                weighted_frame_values[policy].extend(
                    weighted_frame_absrel(bim, gt, eligible_support, pixel_weights)
                )
                for fraction, key in zip(args.top_fractions, fraction_keys):
                    token_selection = top_token_mask(distribution, token_valid, fraction)
                    pixel_selection = functional.interpolate(
                        token_selection[:, None].float(),
                        size=output_size,
                        mode="nearest",
                    ).bool()
                    selected[policy][key]["bim_hit"].update(
                        bim,
                        gt,
                        hit_support & pixel_selection,
                    )
                    selected[policy][key]["scale_eligible"].update(
                        bim,
                        gt,
                        eligible_support & pixel_selection,
                    )

            processed += int(gt.shape[0])
            if processed == len(dataset) or processed % args.log_every < int(gt.shape[0]):
                print(f"[{processed}/{len(dataset)}]", flush=True)

    result: dict[str, Any] = {
        "schema_version": 1,
        "definition": {
            "attention": (
                "Head-mixture-weighted fixed neural attention, ranked independently "
                "within each frame over valid 63x63 ratio tokens."
            ),
            "huber_effective_attention": (
                "Fixed neural attention multiplied by the final pseudo-Huber weight "
                "and renormalized per head before applying the learned head mixture."
            ),
            "top_region_projection": (
                "Top-k token membership is nearest-neighbor projected to 504x504; "
                "metrics are pixel-micro over the selected support."
            ),
            "ground_truth_support": str(cfg.data.ground_truth_support),
            "scale_ratio_bounds": [
                float(cfg.model.attention_scale.ratio_min),
                float(cfg.model.attention_scale.ratio_max),
            ],
        },
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(checkpoint),
        "split": args.split,
        "sample_count": len(dataset),
        "all_bim_hit": all_hit.compute(),
        "all_scale_eligible": all_eligible.compute(all_hit.count),
        "policies": {},
    }
    for policy in policies:
        frame_values = weighted_frame_values[policy]
        result["policies"][policy] = {
            "attention_weighted_frame_macro_abs_rel_on_scale_eligible": (
                sum(frame_values) / len(frame_values) if frame_values else float("nan")
            ),
            "selected_regions": {
                key: {
                    support: accumulator.compute(
                        all_hit.count if support == "bim_hit" else all_eligible.count
                    )
                    for support, accumulator in supports.items()
                }
                for key, supports in selected[policy].items()
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
