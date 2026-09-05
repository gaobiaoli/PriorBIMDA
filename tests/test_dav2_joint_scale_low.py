from __future__ import annotations

import torch

from bim_priorda3.config import load_config
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


def test_dynamic_low2_teacher_uses_detached_predicted_low1_remainder() -> None:
    spatial_target = torch.tensor(
        [[[[ -0.15, -0.10, 0.10, 0.15],
           [ -0.15, -0.10, 0.10, 0.15],
           [ -0.15, -0.10, 0.10, 0.15],
           [ -0.15, -0.10, 0.10, 0.15]]]]
    )
    shape = tuple(spatial_target.shape)
    gt = spatial_target.exp()
    low1_target = torch.nn.functional.adaptive_avg_pool2d(spatial_target, (2, 2))
    low1_prediction = (low1_target + 0.03).requires_grad_()
    predicted_remainder = spatial_target - torch.nn.functional.interpolate(
        low1_prediction.detach(),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )
    output = {
        "depth": gt.clone(),
        "log_scale": torch.zeros((1, 1, 1, 1)),
        "low1_log_residual_native": low1_prediction,
        "low2_log_residual_native": predicted_remainder.clone().requires_grad_(),
    }
    batch = {
        "gt_depth": gt,
        "gt_valid": torch.ones(shape, dtype=torch.bool),
        "base_depth": torch.ones(shape),
    }
    arguments = {
        "pixel_weight": torch.ones(shape),
        "oracle_log_scale": torch.zeros((1, 1, 1, 1)),
        "oracle_supported": torch.ones(1, dtype=torch.bool),
        "depth_weight": 0.0,
        "scale_teacher_weight": 0.0,
        "low1_teacher_weight": 0.0,
        "low2_teacher_weight": 1.0,
        "zero_mean_weight": 0.0,
        "teacher_beta": 0.02,
        "residual_mode": "low18_low36",
    }

    dynamic = joint_scale_low_loss(
        output,
        batch,
        low2_teacher_decomposition="predicted_low18_detached",
        **arguments,
    )
    ideal_laplacian = joint_scale_low_loss(
        output,
        batch,
        low2_teacher_decomposition="oracle_low18",
        **arguments,
    )
    low1_gradient = torch.autograd.grad(
        dynamic["low2_teacher"],
        low1_prediction,
        allow_unused=True,
    )[0]

    torch.testing.assert_close(dynamic["low2_teacher"], torch.tensor(0.0))
    assert float(ideal_laplacian["low2_teacher"]) > 0.0
    assert low1_gradient is None


def test_dynamic_teacher_config_keeps_shared_detached_iteration() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_iterative_shared_"
        "geometry_r18_r36_dynamic_teacher_6epoch_continuous_"
        "full_depth_metric_da3.yaml"
    )

    geometry = cfg.model.dav2_joint_scale_low.iterative_geometry_adapters
    assert geometry.weight_sharing == "shared_trunk_separate_heads"
    assert geometry.get("detach_previous_prediction", True) is True
    assert cfg.loss.low2_teacher_decomposition == "predicted_low18_detached"
    assert cfg.train.epochs == 6


def test_uncentered_oracle_teacher_retains_residual_dc_component() -> None:
    shape = (1, 1, 4, 4)
    oracle_scale = torch.log(torch.tensor(1.5)).reshape(1, 1, 1, 1)
    residual_dc = torch.log(torch.tensor(2.0 / 1.5))
    gt = torch.full(shape, 2.0)
    output = {
        "depth": gt.clone(),
        "log_scale": oracle_scale.clone(),
        "low1_log_residual_native": torch.zeros((1, 1, 2, 2)),
        "low2_log_residual_native": torch.full((1, 1, 4, 4), residual_dc),
    }
    batch = {
        "gt_depth": gt,
        "gt_valid": torch.ones(shape, dtype=torch.bool),
        "base_depth": torch.ones(shape),
    }
    arguments = {
        "pixel_weight": torch.ones(shape),
        "oracle_log_scale": oracle_scale,
        "oracle_supported": torch.ones(1, dtype=torch.bool),
        "depth_weight": 0.0,
        "scale_teacher_weight": 0.0,
        "low1_teacher_weight": 0.0,
        "low2_teacher_weight": 1.0,
        "zero_mean_weight": 0.0,
        "teacher_beta": 0.02,
        "residual_mode": "low36_only",
    }

    uncentered = joint_scale_low_loss(
        output,
        batch,
        spatial_teacher_mean_center=False,
        **arguments,
    )
    centered = joint_scale_low_loss(
        output,
        batch,
        spatial_teacher_mean_center=True,
        **arguments,
    )

    torch.testing.assert_close(uncentered["low2_teacher"], torch.tensor(0.0))
    assert float(centered["low2_teacher"]) > 0.0


def test_uncentered_r36_teacher_config_keeps_original_anchor() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_32_64_128_decoder_"
        "128_64_32_uncentered_r36_teacher_full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    adapter = joint.calibrated_disagreement_adapter
    assert joint.residual_mode == "low36_only"
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]
    assert adapter.enabled is True
    assert adapter.hidden_channels == 32
    assert adapter.residual_blocks == 3
    assert adapter.expansion_channels == 64
    assert adapter.get("injection", "fused_f36") == "fused_f36"
    assert adapter.get("include_rgb", False) is False
    assert cfg.loss.spatial_teacher_mean_center is False
    assert cfg.loss.residual_zero_mean == 0.10
    assert cfg.train.epochs == 6


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


@torch.no_grad()
def test_direct_low18_teacher_keeps_global_dc_and_has_no_zero_mean_loss() -> None:
    shape = (1, 1, 4, 4)
    gt = torch.full(shape, 2.0)
    base = torch.ones(shape)
    output = {
        "depth": base.clone(),
        "log_scale": torch.zeros((1, 1, 1, 1)),
        "low1_log_residual_native": torch.zeros((1, 1, 2, 2)),
        "low2_log_residual_native": torch.zeros((1, 1, 2, 2)),
    }
    batch = {
        "gt_depth": gt,
        "gt_valid": torch.ones(shape, dtype=torch.bool),
        "base_depth": base,
    }
    losses = joint_scale_low_loss(
        output,
        batch,
        pixel_weight=torch.ones(shape),
        oracle_log_scale=torch.full((1, 1, 1, 1), torch.log(torch.tensor(2.0))),
        oracle_supported=torch.ones(1, dtype=torch.bool),
        depth_weight=1.0,
        scale_teacher_weight=0.0,
        low1_teacher_weight=1.0,
        low2_teacher_weight=0.0,
        zero_mean_weight=0.0,
        teacher_beta=0.02,
        residual_mode="direct_low18",
    )

    # A mean-centered target would be exactly zero for this constant scale
    # error. The positive teacher proves that direct r18 retains the DC term.
    assert float(losses["low1_teacher"]) > 0.6
    assert float(losses["scale_teacher"]) == 0.0
    assert float(losses["zero_mean"]) == 0.0
