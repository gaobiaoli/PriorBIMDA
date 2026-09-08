"""Deterministic BIM-only corruption; RGB, MDE and GT are untouched."""
import cv2
import torch

from .dense4 import Dense4Dataset


def corrupt_bim(depth, mask, *, seed):
    generator = torch.Generator().manual_seed(int(seed))
    depth, mask = depth.clone(), mask.clone()
    draw = float(torch.rand((), generator=generator))
    kind = 0 if draw < .55 else 1 if draw < .75 else 2 if draw < .90 else 3
    fraction, dx, dy = 0., 0, 0
    if kind in (1, 3):
        fraction = .1 + .2 * float(torch.rand((), generator=generator))
        # A spatially contiguous neighborhood, sized by VALID pixel count,
        # not image area. Keeps the requested 10--30% support deletion exact
        # up to rounding, even with irregular BIM masks.
        coordinates = (mask[0] > 0).nonzero()
        count = coordinates.shape[0]
        if count > 1:
            center = coordinates[int(torch.randint(count, (), generator=generator))]
            distance = (coordinates - center).square().sum(1)
            remove_count = min(count - 1, max(1, round(count * fraction)))
            removed = coordinates[torch.argsort(distance, stable=True)[:remove_count]]
            depth[0, removed[:, 0], removed[:, 1]] = 0
            mask[0, removed[:, 0], removed[:, 1]] = 0
    if kind in (2, 3):
        choices = (-4, -2, 2, 4)
        dx, dy = (choices[int(torch.randint(4, (), generator=generator))] for _ in range(2))
        depth, mask = (torch.roll(v, (dy, dx), (-2, -1)) for v in (depth, mask))
        for value in (depth, mask):
            value[..., :dy if dy > 0 else 0, :] = 0
            if dy < 0:
                value[..., dy:, :] = 0
            value[..., :, :dx if dx > 0 else 0] = 0
            if dx < 0:
                value[..., :, dx:] = 0
    depth.masked_fill_(mask <= 0, 0)
    return depth, mask, {"kind": kind, "drop_fraction": fraction, "dx": dx, "dy": dy}


class BIMDomainDataset(Dense4Dataset):
    def __init__(self, cfg, split):
        # Dense4's clean data path never computes coarse/scale/trust features.
        super().__init__(cfg, split, augment=False)
        self.rgb_resize_interpolation = cv2.INTER_CUBIC
        self.epoch = 0  # zero-based epoch in the prescribed seed formula
        self.base_seed = int(cfg.experiment.seed)

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        seed = self.base_seed + self.epoch * len(self) + index
        if self.split == "train":
            batch["bim_depth"], batch["bim_valid"], metadata = corrupt_bim(
                batch["bim_depth"], batch["bim_valid"], seed=seed)
        else:
            metadata = {"kind": 0, "drop_fraction": 0., "dx": 0, "dy": 0}
        batch["augmentation_seed"] = seed
        batch["augmentation_kind"] = metadata["kind"]
        return batch
