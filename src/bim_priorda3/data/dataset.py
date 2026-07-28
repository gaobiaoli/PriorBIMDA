from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from bim_priorda3.baselines import strong_anchor_features
from bim_priorda3.config import Config, resolve_project_path, resolve_slabim_root


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relocate_record(
    record: dict[str, Any],
    processed_root: Path,
    slabim_root: Path,
) -> dict[str, Any]:
    """Resolve manifest paths after copying the project to another machine."""
    relocated = dict(record)
    sample = Path(record["sample"]).expanduser()
    if not sample.exists():
        sample = (
            processed_root
            / "samples"
            / str(record["region"])
            / sample.name
        )
    image = Path(record["image"]).expanduser()
    if not image.exists():
        image = (
            slabim_root
            / "sensor_data"
            / str(record["region"])
            / "images"
            / "data"
            / image.name
        )
    relocated["sample"] = str(sample.resolve())
    relocated["image"] = str(image.resolve())
    return relocated


def _shift(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.roll(array, shift=(dy, dx), axis=(-2, -1))
    if dy > 0:
        shifted[..., :dy, :] = 0
    elif dy < 0:
        shifted[..., dy:, :] = 0
    if dx > 0:
        shifted[..., :, :dx] = 0
    elif dx < 0:
        shifted[..., :, dx:] = 0
    return shifted


class BIMDepthDataset(Dataset):
    """Prepared single-frame DA3/BIM samples with sparse fused-LiDAR supervision."""

    def __init__(self, cfg: Config, split: str, augment: bool | None = None) -> None:
        self.cfg = cfg
        self.split = split
        self.augment = split == "train" if augment is None else augment
        root = resolve_project_path(cfg, cfg.data.processed_root)
        records = _read_manifest(root / "manifest.jsonl")
        slabim_root = resolve_slabim_root(cfg)
        records = [
            relocate_record(record, root, slabim_root) for record in records
        ]
        if split == "train":
            regions = set(cfg.data.train_regions)
        elif split == "val":
            regions = set(cfg.data.val_regions)
        elif split == "test":
            regions = set(cfg.data.test_regions)
        else:
            raise ValueError(f"Unknown dataset split: {split}")
        self.records = [record for record in records if record["region"] in regions]
        if not self.records:
            raise RuntimeError(f"No '{split}' records for regions {sorted(regions)} in {root}")
        self.height = int(cfg.data.target_height)
        self.width = int(cfg.data.target_width)
        self.margin = float(cfg.loss.trust_margin)
        self.temperature = float(cfg.loss.trust_temperature)

    def __len__(self) -> int:
        return len(self.records)

    def _augment_rgb(self, rgb: np.ndarray) -> np.ndarray:
        amount = float(self.cfg.train.augment.color_jitter)
        gain = random.uniform(1.0 - amount, 1.0 + amount)
        bias = random.uniform(-amount, amount)
        return np.clip(rgb * gain + bias, 0.0, 1.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        item = np.load(record["sample"])
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {record['image']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        rgb = image.transpose(2, 0, 1).astype(np.float32) / 255.0

        arrays = {
            "base_depth": item["base_depth"].astype(np.float32)[None],
            "base_confidence": item["base_confidence"].astype(np.float32)[None],
            "bim_depth": item["bim_depth"].astype(np.float32)[None],
            "bim_valid": item["bim_valid"].astype(np.float32)[None],
            "bim_normals": item["bim_normals"].astype(np.float32),
            "bim_edge": item["bim_edge"].astype(np.float32)[None],
            "gt_depth": item["gt_depth"].astype(np.float32)[None],
            "gt_valid": item["gt_valid"].astype(np.float32)[None],
            "gt_weight": item["gt_weight"].astype(np.float32)[None],
        }
        candidate_frame_trust = None
        if {
            "candidate_depth",
            "candidate_log_variance",
            "candidate_trust",
            "candidate_frame_trust",
        }.issubset(item.files):
            arrays.update(
                {
                    "candidate_depth": item["candidate_depth"].astype(np.float32)[None],
                    "candidate_log_variance": item[
                        "candidate_log_variance"
                    ].astype(np.float32)[None],
                    "candidate_trust": item["candidate_trust"].astype(np.float32)[None],
                }
            )
            candidate_frame_trust = float(item["candidate_frame_trust"])
        if {
            "scaled_depth",
            "anchor_depth",
            "anchor_field",
            "anchor_support",
        }.issubset(item.files):
            arrays.update(
                {
                    "scaled_depth": item["scaled_depth"].astype(np.float32)[None],
                    "anchor_depth": item["anchor_depth"].astype(np.float32)[None],
                    "anchor_field": item["anchor_field"].astype(np.float32)[None],
                    "anchor_support": item["anchor_support"].astype(np.float32)[None],
                }
            )
        else:
            scaled, anchor, field, support, _ = strong_anchor_features(
                arrays["base_depth"][0],
                arrays["bim_depth"][0],
            )
            arrays.update(
                {
                    "scaled_depth": scaled[None],
                    "anchor_depth": anchor[None],
                    "anchor_field": field[None],
                    "anchor_support": support[None],
                }
            )

        if self.augment:
            rgb = self._augment_rgb(rgb)
            aug = self.cfg.train.augment
            bim_changed = False
            if random.random() < float(aug.bim_shift_probability):
                limit = int(aug.bim_shift_pixels)
                dx, dy = random.randint(-limit, limit), random.randint(-limit, limit)
                for key in ("bim_depth", "bim_valid", "bim_normals", "bim_edge"):
                    arrays[key] = _shift(arrays[key], dx, dy)
                bim_changed = True
            if random.random() < float(aug.bim_dropout_probability):
                area = int(float(aug.bim_dropout_fraction) * self.height * self.width)
                side = max(1, int(np.sqrt(area)))
                y = random.randint(0, max(0, self.height - side))
                x = random.randint(0, max(0, self.width - side))
                arrays["bim_valid"][..., y : y + side, x : x + side] = 0
                arrays["bim_depth"][..., y : y + side, x : x + side] = 0
                arrays["bim_normals"][..., y : y + side, x : x + side] = 0
                arrays["bim_edge"][..., y : y + side, x : x + side] = 1
                bim_changed = True
            if bim_changed:
                scaled, anchor, field, support, _ = strong_anchor_features(
                    arrays["base_depth"][0],
                    arrays["bim_depth"][0],
                )
                arrays.update(
                    {
                        "scaled_depth": scaled[None],
                        "anchor_depth": anchor[None],
                        "anchor_field": field[None],
                        "anchor_support": support[None],
                    }
                )
            if random.random() < float(aug.horizontal_flip_probability):
                rgb = rgb[..., ::-1].copy()
                for key in arrays:
                    arrays[key] = arrays[key][..., ::-1].copy()
                arrays["bim_normals"][0] *= -1
            crop_height = int(aug.get("crop_height", self.height))
            crop_width = int(aug.get("crop_width", self.width))
            if crop_height < self.height or crop_width < self.width:
                y = random.randint(0, self.height - crop_height)
                x = random.randint(0, self.width - crop_width)
                rgb = rgb[..., y : y + crop_height, x : x + crop_width]
                arrays = {
                    key: value[..., y : y + crop_height, x : x + crop_width]
                    for key, value in arrays.items()
                }

        base = arrays["base_depth"]
        scaled = arrays["scaled_depth"]
        bim = arrays["bim_depth"]
        gt = arrays["gt_depth"]
        trust_mask = (
            (arrays["gt_valid"] > 0)
            & (arrays["bim_valid"] > 0)
            & (base > 0)
            & (bim > 0)
            & (gt > 0)
        )
        # BIM reliability must be judged after DA3 metric scale recovery.  Comparing
        # against unscaled DA3 mostly teaches the global scale mismatch.
        base_error = np.abs(
            np.log(np.maximum(scaled, 1e-4)) - np.log(np.maximum(gt, 1e-4))
        )
        bim_error = np.abs(np.log(np.maximum(bim, 1e-4)) - np.log(np.maximum(gt, 1e-4)))
        advantage = base_error - bim_error - self.margin
        trust_logit = np.clip(advantage / self.temperature, -30.0, 30.0)
        trust_target = 1.0 / (1.0 + np.exp(-trust_logit))
        trust_target[~trust_mask] = 0.0

        output: dict[str, Any] = {
            "rgb": torch.from_numpy(rgb.copy()),
            **{key: torch.from_numpy(value.copy()) for key, value in arrays.items()},
            "trust_target": torch.from_numpy(trust_target.astype(np.float32)),
            "trust_mask": torch.from_numpy(trust_mask.astype(np.float32)),
            "sample_id": record["id"],
            "region": record["region"],
        }
        if candidate_frame_trust is not None:
            output["candidate_frame_trust"] = torch.tensor(
                candidate_frame_trust, dtype=torch.float32
            )
        return output
