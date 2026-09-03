from __future__ import annotations

import math

import torch

from bim_priorda3.config import load_config
from bim_priorda3.models import (
    AdapterResidualBlock,
    CalibratedDisagreementAdapter,
    ZeroInitDINOFeatureAdapter,
    ZeroInitDPTShortcutAdapter,
    build_calibrated_disagreement_condition,
    build_native_residual_head,
    rebuild_bim_condition_with_scaled_prediction,
)


def test_calibrated_disagreement_condition_uses_predicted_global_scale() -> None:
    base = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    log_scale = torch.full((1, 1, 1, 1), math.log(2.0))
    expected_z = torch.tensor([[[[1.0, -1.0], [2.0, -2.0]]]])
    bim = base * torch.exp(log_scale + expected_z)
    mask = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])

    condition = build_calibrated_disagreement_condition(
        base,
        bim,
        mask,
        log_scale,
        (1, 1),
    )

    torch.testing.assert_close(condition[:, 0], torch.tensor([[[1.0 / 3.0]]]))
    torch.testing.assert_close(condition[:, 1], torch.tensor([[[7.0 / 9.0]]]))
    torch.testing.assert_close(condition[:, 2], torch.tensor([[[0.75]]]))


def test_calibrated_disagreement_condition_zeros_unsupported_cells() -> None:
    base = torch.ones((1, 1, 2, 2))
    bim = torch.full_like(base, 2.0)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    condition = build_calibrated_disagreement_condition(
        base,
        bim,
        mask,
        torch.zeros((1, 1, 1, 1)),
        (2, 2),
    )

    normalized_log_two = math.log(2.0) / 1.5
    expected_z = torch.tensor([[[[normalized_log_two, 0.0], [0.0, normalized_log_two]]]])
    torch.testing.assert_close(condition[:, :1], expected_z)
    torch.testing.assert_close(condition[:, 1:2], expected_z.abs())
    torch.testing.assert_close(condition[:, 2:], mask)


def test_disagreement_is_clipped_before_masked_pooling() -> None:
    base = torch.ones((1, 1, 1, 2))
    bim = torch.exp(torch.tensor([[[[3.0, 0.0]]]]))
    mask = torch.ones_like(base)

    condition = build_calibrated_disagreement_condition(
        base,
        bim,
        mask,
        torch.zeros((1, 1, 1, 1)),
        (1, 1),
    )

    # Per-pixel normalized values are [1,0]. Pooling first would incorrectly
    # produce clip(mean([3,0]) / 1.5) == 1 rather than the expected 0.5.
    torch.testing.assert_close(condition[:, 0], torch.tensor([[[0.5]]]))
    torch.testing.assert_close(condition[:, 1], torch.tensor([[[0.5]]]))


def test_rgb6_condition_appends_raw_area_pooled_rgb() -> None:
    base = torch.ones((1, 1, 2, 2))
    bim = torch.full_like(base, 2.0)
    mask = torch.ones_like(base)
    rgb = torch.tensor(
        [[[[0.0, 1.0], [0.5, 0.5]], [[0.2, 0.4], [0.6, 0.8]], [[1.0, 0.0], [1.0, 0.0]]]]
    )

    condition = build_calibrated_disagreement_condition(
        base,
        bim,
        mask,
        torch.zeros((1, 1, 1, 1)),
        (1, 1),
        rgb=rgb,
    )

    assert condition.shape == (1, 6, 1, 1)
    torch.testing.assert_close(condition[:, 3:], rgb.mean(dim=(-2, -1), keepdim=True))


def test_calibrated_condition_stops_residual_gradient_into_global_scale() -> None:
    base = torch.ones((1, 1, 2, 2))
    bim = torch.full((1, 1, 2, 2), 2.0, requires_grad=True)
    log_scale = torch.zeros((1, 1, 1, 1), requires_grad=True)

    condition = build_calibrated_disagreement_condition(
        base,
        bim,
        torch.ones_like(base),
        log_scale,
        (1, 1),
    )
    bim_gradient, scale_gradient = torch.autograd.grad(
        condition[:, :2].sum(),
        (bim, log_scale),
        allow_unused=True,
    )

    assert bim_gradient is not None
    assert scale_gradient is None


def test_iterative_r36_condition_uses_and_detaches_r18_prediction() -> None:
    base = torch.ones((1, 1, 2, 2))
    log_scale = torch.full((1, 1, 1, 1), math.log(2.0), requires_grad=True)
    r18 = torch.tensor(
        [[[[math.log(1.5), 0.0], [math.log(0.5), math.log(2.0)]]]],
        requires_grad=True,
    )
    bim = base * torch.exp(log_scale.detach() + r18.detach())

    condition = build_calibrated_disagreement_condition(
        base,
        bim,
        torch.ones_like(base),
        log_scale,
        (2, 2),
        log_residual=r18,
    )

    torch.testing.assert_close(condition[:, :2], torch.zeros_like(condition[:, :2]))
    assert not condition.requires_grad
    assert log_scale.grad is None
    assert r18.grad is None


def test_second_pass_condition_uses_detached_scaled_prediction() -> None:
    base = torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
    bim = torch.tensor([[[[2.0, 2.0], [8.0, 8.0]]]], requires_grad=True)
    first_condition = torch.cat(
        (
            torch.full_like(base, 0.25),
            torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]]),
            torch.full_like(base, 0.75),
        ),
        dim=1,
    )
    log_scale = torch.full((1, 1, 1, 1), math.log(2.0), requires_grad=True)

    second_condition = rebuild_bim_condition_with_scaled_prediction(
        first_condition,
        base,
        bim,
        log_scale,
    )

    torch.testing.assert_close(second_condition[:, :2], first_condition[:, :2])
    torch.testing.assert_close(
        second_condition[:, 2:3],
        torch.tensor([[[[0.0, -math.log(2.0) / 1.5], [0.0, 0.0]]]]),
    )
    bim_gradient, scale_gradient = torch.autograd.grad(
        second_condition[:, 2:].sum(),
        (bim, log_scale),
        allow_unused=True,
    )
    assert bim_gradient is not None
    assert scale_gradient is None


def test_zero_initialized_adapter_preserves_f36_exactly() -> None:
    torch.manual_seed(11)
    adapter = CalibratedDisagreementAdapter(128, hidden_channels=32)
    condition = torch.randn(2, 3, 36, 36)
    feature36 = torch.randn(2, 128, 36, 36)

    delta = adapter(condition)

    assert adapter.input_projection.weight.shape == (32, 3, 3, 3)
    assert adapter.output_projection.weight.shape == (128, 32, 1, 1)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(feature36 + delta, feature36)


def test_residual_adapter_preserves_f36_exactly_at_zero_initialized_output() -> None:
    torch.manual_seed(13)
    adapter = CalibratedDisagreementAdapter(128, hidden_channels=32, residual_blocks=3)
    condition = torch.randn(2, 3, 36, 36)
    feature36 = torch.randn(2, 128, 36, 36)

    delta = adapter(condition)

    assert len(adapter.residual_blocks) == 3
    assert all(isinstance(block, AdapterResidualBlock) for block in adapter.residual_blocks)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(feature36 + delta, feature36)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 60_608


def test_progressive_adapter_uses_32_64_128_channels_and_stays_zero() -> None:
    torch.manual_seed(17)
    adapter = CalibratedDisagreementAdapter(
        128,
        hidden_channels=32,
        residual_blocks=3,
        expansion_channels=64,
    )
    condition = torch.randn(2, 3, 36, 36)

    delta = adapter(condition)

    assert adapter.input_projection.weight.shape == (32, 3, 3, 3)
    assert adapter.expansion_projection is not None
    assert adapter.expansion_projection.weight.shape == (64, 32, 3, 3)
    assert adapter.output_projection.weight.shape == (128, 64, 1, 1)
    assert torch.count_nonzero(delta) == 0
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 83_200


def test_rgb6_progressive_adapter_changes_only_the_input_projection() -> None:
    adapter = CalibratedDisagreementAdapter(
        128,
        input_channels=6,
        hidden_channels=32,
        residual_blocks=3,
        expansion_channels=64,
    )
    condition = torch.randn(2, 6, 36, 36)

    delta = adapter(condition)

    assert adapter.input_projection.weight.shape == (32, 6, 3, 3)
    assert adapter.expansion_projection is not None
    assert adapter.expansion_projection.weight.shape == (64, 32, 3, 3)
    assert adapter.output_projection.weight.shape == (128, 64, 1, 1)
    assert torch.count_nonzero(delta) == 0
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 84_064


def test_deeper_r36_decoder_uses_128_64_32_1_channels_and_stays_zero() -> None:
    decoder = build_native_residual_head(128, (64, 32))
    feature36 = torch.randn(2, 128, 36, 36)

    residual = decoder(feature36)

    convolutions = [module for module in decoder if isinstance(module, torch.nn.Conv2d)]
    assert [tuple(module.weight.shape) for module in convolutions] == [
        (64, 128, 3, 3),
        (32, 64, 3, 3),
        (1, 32, 1, 1),
    ]
    assert torch.count_nonzero(residual) == 0
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 92_289


def test_shared_second_pass_dino_adapter_is_zero_initialized() -> None:
    adapter = ZeroInitDINOFeatureAdapter(768, hidden_channels=64)
    second_pass_tokens = torch.randn(2, 97, 768, requires_grad=True)
    original_tokens = torch.randn_like(second_pass_tokens)

    delta = adapter(second_pass_tokens.detach())

    assert adapter.input_projection.weight.shape == (64, 768)
    assert adapter.output_projection.weight.shape == (768, 64)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(original_tokens + delta, original_tokens)
    delta.sum().backward()
    assert second_pass_tokens.grad is None
    assert adapter.output_projection.weight.grad is not None


def test_r36_shortcut_adapter_is_spatial_zero_initialized_and_detached() -> None:
    adapter = ZeroInitDPTShortcutAdapter(128, hidden_channels=64)
    second_pass_shortcut = torch.randn(2, 128, 36, 36, requires_grad=True)
    original_shortcut = torch.randn_like(second_pass_shortcut)

    delta = adapter(second_pass_shortcut.detach())

    assert adapter.input_projection.weight.shape == (64, 128, 3, 3)
    assert adapter.output_projection.weight.shape == (128, 64, 1, 1)
    assert torch.count_nonzero(delta) == 0
    assert torch.equal(original_shortcut + delta, original_shortcut)
    delta.sum().backward()
    assert second_pass_shortcut.grad is None
    assert adapter.output_projection.weight.grad is not None


def test_adapter_experiment_keeps_only_the_requested_three_channels() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_full_depth_metric_da3.yaml"
    )

    adapter = cfg.model.dav2_joint_scale_low.calibrated_disagreement_adapter
    assert adapter.enabled is True
    assert adapter.hidden_channels == 32
    assert cfg.model.dav2_joint_scale_low.residual_mode == "low36_only"
    assert cfg.train.augment.get("da3_global_scale_perturbation", {}) == {}
    assert cfg.train.augment.get("bim_condition_dropout", {}) == {}


def test_residual_adapter_experiment_adds_three_blocks() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_resblocks3_full_depth_metric_da3.yaml"
    )

    adapter = cfg.model.dav2_joint_scale_low.calibrated_disagreement_adapter
    assert adapter.enabled is True
    assert adapter.hidden_channels == 32
    assert adapter.residual_blocks == 3


def test_progressive_adapter_and_decoder_experiment_channels() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_32_64_128_decoder_"
        "128_64_32_full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    adapter = joint.calibrated_disagreement_adapter
    assert adapter.hidden_channels == 32
    assert adapter.expansion_channels == 64
    assert adapter.residual_blocks == 3
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]


def test_projected_p36_experiment_moves_only_the_adapter_injection() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_32_64_128_decoder_"
        "128_64_32_projected_p36_injection_full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    adapter = joint.calibrated_disagreement_adapter
    assert adapter.enabled is True
    assert adapter.hidden_channels == 32
    assert adapter.expansion_channels == 64
    assert adapter.residual_blocks == 3
    assert adapter.injection == "projected_p36"
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]
    assert cfg.train.epochs == 6


def test_rgb6_experiment_keeps_anchor_post_fusion_path() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_rgb6_adapter_32_64_128_decoder_"
        "128_64_32_full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    adapter = joint.calibrated_disagreement_adapter
    assert adapter.include_rgb is True
    assert adapter.injection == "fused_f36"
    assert adapter.hidden_channels == 32
    assert adapter.expansion_channels == 64
    assert adapter.residual_blocks == 3
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]
    assert cfg.train.epochs == 6


def test_iterative_geometry_experiment_has_two_three_channel_stages() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_iterative_geometry_"
        "r18_r36_6epoch_continuous_full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    geometry = joint.iterative_geometry_adapters
    assert joint.residual_mode == "low18_low36"
    assert joint.calibrated_disagreement_adapter.enabled is False
    assert geometry.enabled is True
    assert geometry.hidden_channels == 32
    assert geometry.residual_blocks == 3
    assert geometry.expansion_channels == 64
    assert list(joint.low1_decoder_hidden_channels) == [64, 32]
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]
    assert cfg.loss.low1_residual_teacher == 0.25
    assert cfg.loss.low2_residual_teacher == 0.50
    assert cfg.train.epochs == 6

    geometry18 = CalibratedDisagreementAdapter(
        128,
        hidden_channels=geometry.hidden_channels,
        residual_blocks=geometry.residual_blocks,
        expansion_channels=geometry.expansion_channels,
    )
    geometry36 = CalibratedDisagreementAdapter(
        128,
        hidden_channels=geometry.hidden_channels,
        residual_blocks=geometry.residual_blocks,
        expansion_channels=geometry.expansion_channels,
    )
    assert geometry18 is not geometry36
    assert geometry18.input_projection.weight.data_ptr() != geometry36.input_projection.weight.data_ptr()
    assert geometry18.input_projection.in_channels == 3
    assert geometry36.input_projection.in_channels == 3


def test_second_pass_adapter_experiment_changes_only_the_requested_switch() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_32_64_128_decoder_"
        "128_64_32_detached_second_pass_dino_adapter_full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    assert joint.detached_scale_second_pass_dino_adapter.enabled is True
    assert joint.detached_scale_second_pass_dino_adapter.hidden_channels == 64
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]
    assert joint.calibrated_disagreement_adapter.hidden_channels == 32
    assert joint.calibrated_disagreement_adapter.expansion_channels == 64
    assert joint.calibrated_disagreement_adapter.residual_blocks == 3
    assert cfg.train.epochs == 6


def test_second_pass_r36_shortcut_experiment_is_spatially_scoped() -> None:
    cfg = load_config(
        "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_"
        "continuous_calibrated_disagreement_adapter_32_64_128_decoder_"
        "128_64_32_detached_second_pass_r36_shortcut_adapter_"
        "full_depth_metric_da3.yaml"
    )

    joint = cfg.model.dav2_joint_scale_low
    second_pass = joint.detached_scale_second_pass_dino_adapter
    assert second_pass.enabled is True
    assert second_pass.hidden_channels == 64
    assert second_pass.scope == "r36_shortcut"
    assert joint.residual_mode == "low36_only"
    assert list(joint.low2_decoder_hidden_channels) == [64, 32]
    assert cfg.train.epochs == 6
