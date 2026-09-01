from __future__ import annotations

import torch

from bim_priorda3.models.dav2_joint_scale_low import (
    joint_scale_low_loss,
    masked_area_downsample,
)


def test_masked_area_downsample_uses_only_valid_values() -> None:
    value = torch.tensor([[[[1.0, 99.0], [3.0, 5.0]]]])
    valid = torch.tensor([[[[True, False], [True, True]]]])

    target, support = masked_area_downsample(value, valid, (1, 1))

    torch.testing.assert_close(target, torch.tensor([[[[3.0]]]]))
    assert bool(support.item())


def test_two_level_target_is_laplacian_not_duplicated() -> None:
    target18 = torch.randn(2, 1, 18, 18)
    target36 = torch.nn.functional.interpolate(
        target18,
        size=(36, 36),
        mode="bilinear",
        align_corners=False,
    )
    low2_band = target36 - torch.nn.functional.interpolate(
        target18,
        size=(36, 36),
        mode="bilinear",
        align_corners=False,
    )

    assert torch.count_nonzero(low2_band) == 0


@torch.no_grad()
def test_single_native_teacher_ignores_disabled_low18() -> None:
    shape = (1, 1, 4, 4)
    gt = torch.full(shape, 2.0)
    base = torch.ones(shape)
    output = {
        "depth": gt.clone(),
        "log_scale": torch.zeros((1, 1, 1, 1)),
        "low1_log_residual_native": torch.full((1, 1, 2, 2), 9.0),
        "low2_log_residual_native": torch.zeros((1, 1, 4, 4)),
    }
    batch = {
        "gt_depth": gt,
        "gt_valid": torch.ones(shape, dtype=torch.bool),
        "base_depth": base,
    }
    for residual_mode in ("low36_only", "low72_only"):
        losses = joint_scale_low_loss(
            output,
            batch,
            pixel_weight=torch.ones(shape),
            oracle_log_scale=torch.full((1, 1, 1, 1), torch.log(torch.tensor(2.0))),
            oracle_supported=torch.ones(1, dtype=torch.bool),
            depth_weight=1.0,
            scale_teacher_weight=0.0,
            low1_teacher_weight=1.0,
            low2_teacher_weight=1.0,
            zero_mean_weight=1.0,
            teacher_beta=0.02,
            residual_mode=residual_mode,
        )

        assert float(losses["low1_teacher"]) == 0.0
        assert float(losses["low2_teacher"]) == 0.0
        assert float(losses["zero_mean"]) == 0.0
