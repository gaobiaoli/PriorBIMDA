"""Dense4 dataset: existing verified split/geometry, no scale estimation."""
from __future__ import annotations

import random
import cv2
import numpy as np
import torch

from .dataset import BIMDepthDataset, _enforce_bim_depth_mask_contract
from .stanford2d3ds import load_stanford_all_valid_depth, official_regular_depth_path


def augment_dense4(arrays, aug, *, donor=None):
    """Operate on fresh arrays; shuffle BEFORE dropout, joint flip AFTER both."""
    flags = {"bim_shuffled": False, "bim_full_dropped": False, "bim_local_dropped": False, "flipped": False}
    amount = float(aug["color_jitter"])
    arrays["rgb"] = np.clip(
        arrays["rgb"] * random.uniform(1-amount, 1+amount) + random.uniform(-amount, amount), 0, 1
    )
    if donor is not None:
        arrays["bim_depth"], arrays["bim_valid"] = (a.copy() for a in donor)
        flags["bim_shuffled"] = True
    height, width = arrays["bim_depth"].shape[-2:]
    if random.random() < float(aug["bim_dropout_probability"]):
        side = max(1, int(np.sqrt(int(float(aug["bim_dropout_fraction"]) * height * width))))
        y, x = random.randint(0, max(0, height-side)), random.randint(0, max(0, width-side))
        for k in ("bim_depth", "bim_valid"):
            arrays[k][..., y:y+side, x:x+side] = 0
        flags["bim_local_dropped"] = True
    if random.random() < float(aug["bim_full_dropout_probability"]):
        for k in ("bim_depth", "bim_valid"):
            arrays[k].fill(0)
        flags["bim_full_dropped"] = True
    if random.random() < float(aug["horizontal_flip_probability"]):
        arrays = {k: a[..., ::-1].copy() for k, a in arrays.items()}
        flags["flipped"] = True
    return arrays, flags


class Dense4Dataset(BIMDepthDataset):
    def __init__(self, cfg, split, augment=None):
        # Reuse annotation SHA/fingerprint, relocation and official support checks.
        # Do not call the legacy __getitem__: no D_local/scale/trust is constructed.
        super().__init__(cfg, split, augment=False)
        self.augment = (split == "train") if augment is None else bool(augment)
        if self.augment and split != "train":
            raise ValueError("Dense4 augmentation/shuffle is train-only")
        if self.ground_truth_support != "official_all_valid" or not self.apply_da3_metric_focal_scaling:
            raise ValueError("Dense4 requires official all-valid GT and focal-corrected DA3")
        priorda_relative = any(
            cfg.model.get(name, {}).get("enabled", False)
            for name in ("priorda_relative_metric_refiner", "priorda_relative_zero_anchor_refiner", "priorda_relative_prior_identity_refiner", "priorda_relative_prior_frame_refiner", "priorda_relative_prior_frame_no_relu_refiner")
        )
        self.rgb_resize_interpolation = (
            cv2.INTER_CUBIC if priorda_relative else cv2.INTER_AREA
        )
        self.donor_indices = {
            region: [i for i, r in enumerate(self.records) if r["region"] != region]
            for region in {r["region"] for r in self.records}
        } if self.augment else {}

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(record["image"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {record['image']}")
        rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (self.width, self.height), interpolation=self.rgb_resize_interpolation)
        with np.load(record["sample"]) as item:
            intrinsic = item["intrinsic"].astype(np.float32)
            focal = float((intrinsic[0, 0] + intrinsic[1, 1]) / 2) / 300
            if not np.isfinite(focal) or focal <= 0:
                raise ValueError("Invalid DA3 metric focal correction")
            arrays = {
                "rgb": rgb.transpose(2, 0, 1).astype(np.float32)/255,
                "base_depth": item["base_depth"].astype(np.float32)[None] * focal,
                "bim_depth": item["bim_depth"].astype(np.float32)[None],
                "bim_valid": item["bim_valid"].astype(np.float32)[None],
            }
        gt, mask = load_stanford_all_valid_depth(official_regular_depth_path(record["image"]), (self.height, self.width))
        arrays.update(gt_depth=gt[None].astype(np.float32), gt_valid=mask[None].astype(np.float32))
        flags = {"bim_shuffled": False, "bim_full_dropped": False, "bim_local_dropped": False, "flipped": False}
        donor_id = ""
        if self.augment:
            donor = None
            aug = self.cfg.train.augment
            if random.random() < float(aug.bim_shuffle_probability):
                candidates = self.donor_indices[record["region"]]
                if not candidates:
                    candidates = [i for i in range(len(self.records)) if i != index]
                if not candidates:
                    raise RuntimeError("BIM shuffle requires another training sample")
                donor_record = self.records[random.choice(candidates)]
                donor_id = str(donor_record["id"])
                with np.load(donor_record["sample"]) as item:
                    donor = (item["bim_depth"].astype(np.float32)[None], item["bim_valid"].astype(np.float32)[None])
            arrays, flags = augment_dense4(arrays, aug, donor=donor)
        _enforce_bim_depth_mask_contract(arrays["bim_depth"], arrays["bim_valid"], sample_id=str(record["id"]))
        if any(a.shape[-2:] != (self.height, self.width) for a in arrays.values()):
            raise ValueError("Dense4 modalities are not aligned at the requested resolution")
        return {**{k: torch.from_numpy(np.ascontiguousarray(a)) for k, a in arrays.items()},
                **flags, "sample_id": str(record["id"]), "region": str(record["region"]), "bim_donor_id": donor_id}
