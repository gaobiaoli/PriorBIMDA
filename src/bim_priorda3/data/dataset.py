from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from bim_priorda3.baselines import (
    BIM_INVALID_DEPTH_ATOL,
    LEGACY_SCALE_ESTIMATOR,
    configured_scale_and_local_features,
    resolve_scale_estimator_config,
)
from bim_priorda3.config import Config, resolve_project_path, resolve_slabim_root

from .da3_features import load_cached_features, load_feature_cache_manifest, sha256_file
from .splits import (
    ACTIVE_SPLITS,
    manifest_preparation_identity,
    resolve_annotation_splits,
)
from .stanford2d3ds import load_stanford_all_valid_depth, official_regular_depth_path

PREPARED_GROUND_TRUTH_SUPPORT = "prepared"
OFFICIAL_ALL_VALID_GROUND_TRUTH_SUPPORT = "official_all_valid"
GROUND_TRUTH_SUPPORT_MODES = {
    PREPARED_GROUND_TRUTH_SUPPORT,
    OFFICIAL_ALL_VALID_GROUND_TRUTH_SUPPORT,
}


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relocate_record(
    record: dict[str, Any],
    processed_root: Path,
    slabim_root: Path,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Resolve manifest paths after copying the project to another machine."""
    relocated = dict(record)
    sample = Path(record["sample"]).expanduser()
    if not sample.exists():
        relative_sample = record.get("sample_relative_to_processed")
        sample = (
            processed_root / str(relative_sample)
            if relative_sample
            else processed_root / "samples" / str(record["region"]) / sample.name
        )
    image = Path(record["image"]).expanduser()
    if not image.exists():
        relative_image = record.get("image_relative_to_source")
        if relative_image:
            image = (source_root or slabim_root) / str(relative_image)
        else:
            image = (
                slabim_root / "sensor_data" / str(record["region"]) / "images" / "data" / image.name
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


def _enforce_bim_depth_mask_contract(
    bim_depth: np.ndarray,
    bim_valid: np.ndarray,
    *,
    sample_id: str,
) -> None:
    """Reject contradictory BIM support and canonicalize tolerated roundoff."""

    if bim_depth.shape != bim_valid.shape:
        raise ValueError(
            f"{sample_id}: bim_depth and bim_valid shapes differ: "
            f"{bim_depth.shape} != {bim_valid.shape}"
        )
    if not np.all(np.isfinite(bim_valid)):
        raise ValueError(f"{sample_id}: bim_valid contains non-finite values")
    invalid = bim_valid <= 0
    invalid_depth = bim_depth[invalid]
    violations = (~np.isfinite(invalid_depth)) | (np.abs(invalid_depth) > BIM_INVALID_DEPTH_ATOL)
    if np.any(violations):
        finite_magnitudes = np.abs(invalid_depth[np.isfinite(invalid_depth)])
        maximum = float(finite_magnitudes.max()) if finite_magnitudes.size else float("nan")
        raise ValueError(
            f"{sample_id}: bim_depth must be zero within "
            f"atol={BIM_INVALID_DEPTH_ATOL:g} wherever bim_valid <= 0; "
            f"violations={int(np.count_nonzero(violations))}, "
            f"max_abs_finite={maximum:g}"
        )
    # The CPU reference estimator historically infers support from depth > 0.
    # Exact zeroing makes it identical to the explicit tensor mask while
    # accepting harmless serialization roundoff inside the stated tolerance.
    bim_depth[invalid] = 0.0


class BIMDepthDataset(Dataset):
    """Prepared single-frame DA3/BIM samples with sparse fused-LiDAR supervision."""

    def __init__(
        self,
        cfg: Config,
        split: str | None,
        augment: bool | None = None,
        *,
        require_ground_truth: bool = True,
        regions: list[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.split = split or "inference"
        self.require_ground_truth = require_ground_truth
        self.augment = split == "train" if augment is None else augment
        self.ground_truth_support = str(
            cfg.data.get("ground_truth_support", PREPARED_GROUND_TRUTH_SUPPORT)
        )
        if self.ground_truth_support not in GROUND_TRUTH_SUPPORT_MODES:
            raise ValueError(
                "data.ground_truth_support must be one of "
                f"{sorted(GROUND_TRUTH_SUPPORT_MODES)}; got {self.ground_truth_support!r}"
            )
        root = resolve_project_path(cfg, cfg.data.processed_root)
        source_manifest = root / "manifest.jsonl"
        records = _read_manifest(source_manifest)
        all_record_ids = [str(record["id"]) for record in records]
        source_value = cfg.data.get("source_root")
        source_root = resolve_project_path(cfg, source_value) if source_value else None
        slabim_root = resolve_slabim_root(cfg) if cfg.data.get("slabim_root") else source_root
        if slabim_root is None:
            raise ValueError("data.slabim_root or data.source_root must be configured")
        records = [relocate_record(record, root, slabim_root, source_root) for record in records]
        preparation_identity = manifest_preparation_identity(records)
        annotation_value = cfg.data.get("split_annotation")
        if annotation_value:
            configured_region_splits = {
                key: list(cfg.data.get(key, []))
                for key in ("train_regions", "val_regions", "test_regions")
                if cfg.data.get(key, [])
            }
            if configured_region_splits:
                raise ValueError(
                    "split_annotation is mutually exclusive with non-empty "
                    f"region split fields: {sorted(configured_region_splits)}"
                )
            if cfg.data.get("record_stride_by_region", {}):
                raise ValueError(
                    "split_annotation already defines the effective population; "
                    "record_stride_by_region must be empty"
                )
            annotation_path = resolve_project_path(cfg, annotation_value)
            resolution = resolve_annotation_splits(records, annotation_path)
            expected_annotation_sha = cfg.data.get("split_annotation_sha256")
            actual_annotation_sha = resolution.provenance["annotation_raw_sha256"]
            if expected_annotation_sha and str(expected_annotation_sha) != actual_annotation_sha:
                raise ValueError(
                    "split_annotation_sha256 mismatch: "
                    f"configured={expected_annotation_sha}, "
                    f"actual={actual_annotation_sha}"
                )
            expected_fingerprint = cfg.data.get("split_fingerprint_sha256")
            actual_fingerprint = resolution.provenance["fingerprint_sha256"]
            if expected_fingerprint and str(expected_fingerprint) != actual_fingerprint:
                raise ValueError(
                    "split_fingerprint_sha256 mismatch: "
                    f"configured={expected_fingerprint}, "
                    f"actual={actual_fingerprint}"
                )
            selected_regions = set(regions if regions is not None else cfg.data.regions)
            if split in ACTIVE_SPLITS:
                annotated_records = resolution.records_for(str(split))
            elif split is None:
                annotated_records = [
                    record
                    for record in records
                    if resolution.assignments[str(record["id"])] in ACTIVE_SPLITS
                ]
            else:
                raise ValueError(f"Unknown dataset split: {split}")
            self.records = [
                record for record in annotated_records if record["region"] in selected_regions
            ]
            self.split_provenance = {
                **resolution.provenance,
                "mode": "annotations",
                "selected_regions": sorted(selected_regions),
            }
        else:
            if regions is not None:
                selected_regions = set(regions)
            elif split == "train":
                selected_regions = set(cfg.data.train_regions)
            elif split == "val":
                selected_regions = set(cfg.data.val_regions)
            elif split == "test":
                selected_regions = set(cfg.data.test_regions)
            elif split is None:
                selected_regions = set(cfg.data.regions)
            else:
                raise ValueError(f"Unknown dataset split: {split}")
            self.records = [record for record in records if record["region"] in selected_regions]
            stride_by_region = {
                str(region): int(stride)
                for region, stride in cfg.data.get(
                    "record_stride_by_region",
                    {},
                ).items()
            }
            if stride_by_region:
                region_indices: dict[str, int] = {}
                sampled_records = []
                for record in self.records:
                    region = str(record["region"])
                    index = region_indices.get(region, 0)
                    region_indices[region] = index + 1
                    stride = stride_by_region.get(region, 1)
                    if stride < 1:
                        raise ValueError(f"record_stride_by_region[{region!r}] must be positive")
                    if index % stride == 0:
                        sampled_records.append(record)
                self.records = sampled_records
            self.split_provenance = {
                "mode": "regions",
                "selected_regions": sorted(selected_regions),
                "record_stride_by_region": dict(sorted(stride_by_region.items())),
                "manifest_preparation_fingerprint_status": preparation_identity["status"],
                "manifest_preparation_fingerprint_sha256": preparation_identity[
                    "fingerprint_sha256"
                ],
            }
        if not self.records:
            raise RuntimeError(
                f"No '{self.split}' records for regions {sorted(selected_regions)} in {root}"
            )
        self.height = int(cfg.data.target_height)
        self.width = int(cfg.data.target_width)
        self.margin = float(cfg.loss.trust_margin)
        self.temperature = float(cfg.loss.trust_temperature)
        if self.ground_truth_support == OFFICIAL_ALL_VALID_GROUND_TRUTH_SUPPORT:
            self.split_provenance["ground_truth_support"] = {
                "mode": OFFICIAL_ALL_VALID_GROUND_TRUTH_SUPPORT,
                "encoding": "official regular-view z-depth uint16/512 metres",
                "validity": "raw != 0 and raw != 65535; no metric depth cutoff",
                "resize": "OpenCV nearest-exact",
            }
        self.scale_estimator = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
        feature_config = Config(cfg.model.get("da3_feature_fusion", {}))
        shared_feature_fusion = bool(feature_config.get("enabled", False))
        self.da3_feature_fusion_enabled = bool(
            feature_config.get("scale_enabled", shared_feature_fusion)
        ) or bool(feature_config.get("refiner_enabled", shared_feature_fusion))
        self.da3_feature_cache_root: Path | None = None
        self.da3_feature_layers: tuple[int, int] = (11, 23)
        self.da3_feature_channels = 0
        self.da3_feature_grid_shape: tuple[int, int] = (0, 0)
        if self.da3_feature_fusion_enabled:
            cache_value = cfg.data.get("da3_feature_cache_root")
            if not cache_value:
                raise ValueError(
                    "data.da3_feature_cache_root is required when "
                    "model.da3_feature_fusion.enabled is true"
                )
            self.da3_feature_cache_root = resolve_project_path(cfg, cache_value)
            configured_layers = tuple(
                int(value) for value in feature_config.get("layers", (11, 23))
            )
            if len(configured_layers) != 2:
                raise ValueError("model.da3_feature_fusion.layers must contain two layers")
            self.da3_feature_layers = configured_layers
            self.da3_feature_channels = int(feature_config.get("channels", 1024))
            feature_manifest = load_feature_cache_manifest(
                self.da3_feature_cache_root,
                source_manifest=source_manifest,
                expected_record_ids=all_record_ids,
                model_name=str(cfg.data.da3_model),
                model_revision=str(cfg.data.da3_revision),
                process_res=int(cfg.data.da3_process_res),
                layers=self.da3_feature_layers,
                channels=self.da3_feature_channels,
            )
            self.da3_feature_grid_shape = tuple(
                int(value) for value in feature_manifest["grid_shape"]
            )
            self.split_provenance["da3_feature_cache"] = {
                "manifest_sha256": sha256_file(
                    self.da3_feature_cache_root / "manifest.json"
                ),
                "source_manifest_sha256": feature_manifest["source_manifest_sha256"],
                "model_name": feature_manifest["model_name"],
                "model_revision": feature_manifest["model_revision"],
                "process_res": feature_manifest["process_res"],
                "layers": feature_manifest["layers"],
                "channels": feature_manifest["channels"],
                "grid_shape": feature_manifest["grid_shape"],
                "dtype": feature_manifest.get("dtype", "float16"),
            }
            if self.augment:
                aug = cfg.train.augment
                crop_shape = (
                    int(aug.get("crop_height", self.height)),
                    int(aug.get("crop_width", self.width)),
                )
                if crop_shape != (self.height, self.width):
                    raise ValueError(
                        "Cached DA3 feature fusion currently requires full-image training "
                        "crops so feature/image geometry stays exact"
                    )

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
        }
        for key in (
            "semantic_valid",
            "structural_mask",
            "furniture_mask",
            "non_structural_mask",
        ):
            if key in item.files:
                arrays[key] = item[key].astype(np.float32)[None]
        if "semantic_class" in item.files:
            arrays["semantic_class"] = item["semantic_class"].astype(np.int64)[None]
        if "bim_category" in item.files:
            arrays["bim_category"] = item["bim_category"].astype(np.int64)[None]
        _enforce_bim_depth_mask_contract(
            arrays["bim_depth"],
            arrays["bim_valid"],
            sample_id=str(record["id"]),
        )
        gt_keys = {"gt_depth", "gt_valid", "gt_weight"}
        prepared_has_ground_truth = gt_keys.issubset(item.files)
        has_ground_truth = (
            prepared_has_ground_truth
            or self.ground_truth_support == OFFICIAL_ALL_VALID_GROUND_TRUTH_SUPPORT
        )
        if self.require_ground_truth and not has_ground_truth:
            missing = sorted(gt_keys - set(item.files))
            raise RuntimeError(
                f"{record['id']}: prepared sample lacks training/evaluation "
                f"fields {missing}; use inference mode or prepare GT"
            )
        if self.ground_truth_support == OFFICIAL_ALL_VALID_GROUND_TRUTH_SUPPORT:
            gt_depth, gt_valid = load_stanford_all_valid_depth(
                official_regular_depth_path(record["image"]),
                (self.height, self.width),
            )
            arrays.update(
                {
                    "gt_depth": gt_depth.astype(np.float32)[None],
                    "gt_valid": gt_valid.astype(np.float32)[None],
                    "gt_weight": gt_valid.astype(np.float32)[None],
                }
            )
        elif prepared_has_ground_truth:
            arrays.update(
                {
                    "gt_depth": item["gt_depth"].astype(np.float32)[None],
                    "gt_valid": item["gt_valid"].astype(np.float32)[None],
                    "gt_weight": item["gt_weight"].astype(np.float32)[None],
                }
            )
        recompute_baselines = (
            bool(self.cfg.data.get("recompute_cached_baselines", False))
            or self.scale_estimator["name"] != LEGACY_SCALE_ESTIMATOR
        )
        if (
            self.require_ground_truth
            and not recompute_baselines
            and {"scaled_depth", "anchor_depth"}.issubset(item.files)
        ):
            arrays["scaled_depth"] = item["scaled_depth"].astype(np.float32)[None]
            arrays["anchor_depth"] = item["anchor_depth"].astype(np.float32)[None]
        elif (
            not self.require_ground_truth
            and not recompute_baselines
            and "scaled_depth" in item.files
        ):
            arrays["scaled_depth"] = item["scaled_depth"].astype(np.float32)[None]
        else:
            base = arrays["base_depth"][0]
            bim = arrays["bim_depth"][0]
            scaled, anchor, _, _, _ = configured_scale_and_local_features(
                base,
                bim,
                self.scale_estimator,
            )
            if self.require_ground_truth:
                arrays["anchor_depth"] = anchor[None]
            arrays["scaled_depth"] = scaled[None]

        feature_horizontal_flip = False
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
            if random.random() < float(aug.get("bim_full_dropout_probability", 0.0)):
                arrays["bim_valid"][...] = 0
                arrays["bim_depth"][...] = 0
                arrays["bim_normals"][...] = 0
                arrays["bim_edge"][...] = 1
                bim_changed = True
            if random.random() < float(aug.get("bim_depth_noise_probability", 0.0)):
                noise_std = float(aug.get("bim_depth_noise_log_std", 0.02))
                noise = np.random.normal(
                    0.0,
                    noise_std,
                    size=arrays["bim_depth"].shape,
                ).astype(np.float32)
                valid_bim = arrays["bim_valid"] > 0
                arrays["bim_depth"][valid_bim] *= np.exp(noise[valid_bim])
                bim_changed = True
            if random.random() < float(aug.get("bim_edge_dilation_probability", 0.0)):
                kernel_size = int(aug.get("bim_edge_dilation_pixels", 3))
                kernel_size = max(1, kernel_size)
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                arrays["bim_edge"][0] = cv2.dilate(
                    arrays["bim_edge"][0].astype(np.float32),
                    kernel,
                )
            if bim_changed:
                base = arrays["base_depth"][0]
                bim = arrays["bim_depth"][0]
                scaled, anchor, _, _, _ = configured_scale_and_local_features(
                    base,
                    bim,
                    self.scale_estimator,
                )
                if self.require_ground_truth or self.requires_bim_direct_residual_anchor:
                    arrays["anchor_depth"] = anchor[None]
                arrays["scaled_depth"] = scaled[None]
            if random.random() < float(aug.horizontal_flip_probability):
                rgb = rgb[..., ::-1].copy()
                for key in arrays:
                    arrays[key] = arrays[key][..., ::-1].copy()
                arrays["bim_normals"][0] *= -1
                feature_horizontal_flip = True
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

        _enforce_bim_depth_mask_contract(
            arrays["bim_depth"],
            arrays["bim_valid"],
            sample_id=str(record["id"]),
        )

        output: dict[str, Any] = {
            "rgb": torch.from_numpy(rgb.copy()),
            **{key: torch.from_numpy(value.copy()) for key, value in arrays.items()},
            "sample_id": record["id"],
            "region": record["region"],
            "image_timestamp": float(record.get("image_timestamp", index)),
            "frame_index": int(record.get("lidar_index", index)),
        }
        if self.da3_feature_fusion_enabled:
            assert self.da3_feature_cache_root is not None
            feature_mid, feature_deep = load_cached_features(
                self.da3_feature_cache_root,
                str(record["id"]),
                layers=self.da3_feature_layers,
                channels=self.da3_feature_channels,
                grid_shape=self.da3_feature_grid_shape,
            )
            if feature_horizontal_flip:
                feature_mid = feature_mid[..., ::-1].copy()
                feature_deep = feature_deep[..., ::-1].copy()
            output["da3_feature_mid"] = torch.from_numpy(feature_mid.copy())
            output["da3_feature_deep"] = torch.from_numpy(feature_deep.copy())
        if has_ground_truth:
            base = arrays["scaled_depth"]
            bim = arrays["bim_depth"]
            gt = arrays["gt_depth"]
            trust_mask = (
                (arrays["gt_valid"] > 0)
                & (arrays["bim_valid"] > 0)
                & (base > 0)
                & (bim > 0)
                & (gt > 0)
            )
            # Reliability is judged only after DA3 metric-scale recovery.
            base_error = np.abs(np.log(np.maximum(base, 1e-4)) - np.log(np.maximum(gt, 1e-4)))
            bim_error = np.abs(np.log(np.maximum(bim, 1e-4)) - np.log(np.maximum(gt, 1e-4)))
            advantage = base_error - bim_error - self.margin
            trust_logit = np.clip(advantage / self.temperature, -30.0, 30.0)
            trust_target = 1.0 / (1.0 + np.exp(-trust_logit))
            trust_target[~trust_mask] = 0.0
            output.update(
                {
                    "trust_target": torch.from_numpy(trust_target.astype(np.float32)),
                    "trust_mask": torch.from_numpy(trust_mask.astype(np.float32)),
                }
            )
        return output
