import math
from pathlib import Path

import pytest
import torch
from torch import nn

from bim_priorda3.config import Config, load_config
from bim_priorda3.models.priorbimda_two_stage import (
    DAV2_RELATIVE_DISPARITY_OUTPUT,
    PER_FRAME_BIM_MINMAX_NORMALIZATION,
    PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION,
    PriorBIMDAConditionStatistics,
    PriorBIMDATwoStage,
    build_priorbimda_condition,
    fixed_attention_effective_reliability,
    priorbimda_prior_min_range,
)
from scripts.model.train_priorbimda_two_stage import (
    epoch_loss_trend_audit,
    priorbimda_refinement_loss,
)


def _scale_output() -> dict[str, torch.Tensor]:
    return {
        "scale": torch.ones(1, 1, 1, 1),
        "log_scale": torch.zeros(1, 1, 1, 1),
        "attention_token_distribution": torch.tensor([[[[0.75, 0.25]], [[0.25, 0.75]]]]),
        "attention_token_valid": torch.ones(1, 1, 1, 2, dtype=torch.bool),
        "head_mixture": torch.tensor([[0.25, 0.75]]),
        "head_log_scale": torch.tensor([[0.0, math.log(2.0)]]),
    }


def test_effective_reliability_is_attention_times_final_huber_weight():
    base = torch.ones(1, 1, 1, 2)
    bim = torch.tensor([[[[1.0, 2.0]]]])
    output = fixed_attention_effective_reliability(
        base_depth=base,
        bim_depth=bim,
        bim_valid=torch.ones_like(base),
        scale_output=_scale_output(),
        huber_delta=0.5,
        ratio_min=0.2,
        ratio_max=5.0,
    )

    residual = torch.tensor([[[[0.0, math.log(2.0)]], [[-math.log(2.0), 0.0]]]])
    robust = torch.rsqrt(1.0 + (residual / 0.5).square())
    attention = _scale_output()["attention_token_distribution"]
    mixture = _scale_output()["head_mixture"][:, :, None, None]
    expected_attention = (mixture * attention).sum(dim=1, keepdim=True)
    expected_effective = (mixture * attention * robust).sum(dim=1, keepdim=True)
    torch.testing.assert_close(output["attention_token_map"], expected_attention)
    torch.testing.assert_close(output["effective_reliability_token_map"], expected_effective)
    assert bool(torch.all(output["effective_reliability"] <= output["attention_map"]))


def test_condition_keeps_global_scale_and_zeros_only_bim_channels_when_invalid():
    scaled = torch.tensor([[[[1.0, math.e]]]])
    bim = torch.tensor([[[[math.e, 0.0]]]])
    valid = torch.tensor([[[[1.0, 0.0]]]])
    reliability = torch.tensor([[[[0.25, 0.75]]]])
    condition = build_priorbimda_condition(
        scaled_depth=scaled,
        bim_depth=bim,
        bim_valid=valid,
        effective_reliability=reliability,
        depth_log_mean=0.0,
        depth_log_std=2.0,
        effective_reliability_mean=0.5,
        disagreement_clip=1.5,
    )
    torch.testing.assert_close(condition[0, 0, 0], torch.tensor([0.0, 0.5]))
    torch.testing.assert_close(condition[0, 1, 0], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(condition[0, 2, 0], torch.tensor([2.0 / 3.0, 0.0]))


def test_condition_has_no_per_frame_normalization():
    scaled = torch.tensor([[[[1.0, 1.0]]], [[[math.e, math.e]]]])
    condition = build_priorbimda_condition(
        scaled_depth=scaled,
        bim_depth=torch.ones_like(scaled),
        bim_valid=torch.ones_like(scaled),
        effective_reliability=torch.ones_like(scaled),
        depth_log_mean=0.0,
        depth_log_std=1.0,
        effective_reliability_mean=1.0,
    )
    torch.testing.assert_close(condition[0, 0], torch.zeros_like(condition[0, 0]))
    torch.testing.assert_close(condition[1, 0], torch.ones_like(condition[1, 0]))


def test_minmax_condition_and_output_anchor_are_zero_one_and_reversible():
    scaled = torch.tensor([[[[1.0, 3.0, 6.0]]]])
    bim = torch.tensor([[[[2.0, 4.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    reliability = torch.tensor([[[[0.25, 0.5, 0.0]]]])
    prior_minimum, prior_range = priorbimda_prior_min_range(
        bim_depth=bim,
        bim_valid=valid,
        fallback_depth=scaled,
    )
    torch.testing.assert_close(prior_minimum, torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(prior_range, torch.tensor([[[[2.0]]]]))

    condition = build_priorbimda_condition(
        scaled_depth=scaled,
        bim_depth=bim,
        bim_valid=valid,
        effective_reliability=reliability,
        normalization=PER_FRAME_BIM_MINMAX_NORMALIZATION,
        prior_minimum=prior_minimum,
        prior_range=prior_range,
    )
    assert bool(torch.all((condition >= 0) & (condition <= 1)))
    torch.testing.assert_close(condition[0, 0, 0], torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(condition[0, 1, 0], torch.tensor([0.5, 1.0, 0.0]))
    torch.testing.assert_close(condition[0, 2, 0, 2], torch.tensor(0.0))
    normalized_depth = torch.tensor([[[[0.0, 0.5, 1.0]]]])
    torch.testing.assert_close(
        normalized_depth * prior_range + prior_minimum,
        torch.tensor([[[[2.0, 3.0, 4.0]]]]),
    )


def test_minmax_normalization_falls_back_to_stage1_on_full_bim_dropout():
    fallback = torch.tensor([[[[1.0, 2.0, 5.0]]]])
    prior_minimum, prior_range = priorbimda_prior_min_range(
        bim_depth=torch.zeros_like(fallback),
        bim_valid=torch.zeros_like(fallback),
        fallback_depth=fallback,
    )
    torch.testing.assert_close(prior_minimum, torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(prior_range, torch.tensor([[[[4.0]]]]))


def test_union_normalization_includes_dense_stage1_range():
    stage1 = torch.tensor([[[[1.0, 3.0, 6.0]]]])
    prior_minimum, prior_range = priorbimda_prior_min_range(
        bim_depth=torch.tensor([[[[2.0, 4.0, 0.0]]]]),
        bim_valid=torch.tensor([[[[1.0, 1.0, 0.0]]]]),
        fallback_depth=stage1,
        include_fallback_in_range=True,
    )
    torch.testing.assert_close(prior_minimum, torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(prior_range, torch.tensor([[[[5.0]]]]))


def test_minmax_config_restores_registered_anchor_bim_augmentation():
    project_root = Path(__file__).resolve().parents[1]
    current = load_config(
        project_root / "configs/stanford_area1_priorbimda_stage2_dav2_metric_minmax_augmented.yaml"
    )
    anchor = load_config(
        project_root / "configs/stanford_area1_f36_anchor_1c07d65_reproduction.yaml"
    )
    keys = (
        "bim_shift_probability",
        "bim_shift_pixels",
        "bim_dropout_probability",
        "bim_dropout_fraction",
        "bim_full_dropout_probability",
        "bim_depth_noise_probability",
        "bim_depth_noise_log_std",
        "bim_edge_dilation_probability",
        "bim_edge_dilation_pixels",
    )
    assert current.model.priorbimda_condition.normalization == (PER_FRAME_BIM_MINMAX_NORMALIZATION)
    assert {key: current.train.augment[key] for key in keys} == {
        key: anchor.train.augment[key] for key in keys
    }


def test_relative_config_keeps_minmax_and_anchor_augmentation():
    project_root = Path(__file__).resolve().parents[1]
    relative = load_config(
        project_root
        / "configs/stanford_area1_priorbimda_stage2_dav2_relative_minmax_augmented.yaml"
    )
    metric = load_config(
        project_root
        / "configs/stanford_area1_priorbimda_stage2_dav2_metric_minmax_augmented.yaml"
    )
    assert relative.model.dav2.output_domain == DAV2_RELATIVE_DISPARITY_OUTPUT
    assert relative.model.dav2.model_id == "depth-anything/Depth-Anything-V2-Base-hf"
    assert relative.model.priorbimda_condition.normalization == (
        PER_FRAME_BIM_MINMAX_NORMALIZATION
    )
    assert dict(relative.train.augment) == dict(metric.train.augment)


def test_union_silog_config_enables_epoch1_trend_gate():
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        project_root
        / "configs/stanford_area1_priorbimda_stage2_dav2_relative_union_silog_augmented.yaml"
    )
    assert cfg.model.priorbimda_condition.normalization == (
        PER_FRAME_STAGE1_BIM_UNION_MINMAX_NORMALIZATION
    )
    assert cfg.loss.silog == 1.0
    assert cfg.loss.normalized_disparity == 0.0
    assert cfg.train.epoch1_loss_trend_gate.enabled is True


def test_epoch1_loss_trend_gate_accepts_descent_and_rejects_flat_loss():
    descending = epoch_loss_trend_audit([4.0, 3.8, 3.0, 2.8])
    flat = epoch_loss_trend_audit([4.0, 3.9, 4.1, 4.0])
    assert descending["pass"] is True
    assert descending["relative_decrease"] > 0.05
    assert flat["pass"] is False


def test_train_statistics_are_pixel_micro_and_fixed():
    accumulator = PriorBIMDAConditionStatistics()
    accumulator.update(
        torch.exp(torch.tensor([[[[0.0, 1.0, 2.0]]]])),
        torch.tensor([[[[0.0, 0.25, 0.75]]]]),
        torch.tensor([[[[0, 1, 1]]]], dtype=torch.bool),
    )
    stats = accumulator.compute()
    assert stats["depth_log_mean"] == pytest.approx(1.0)
    assert stats["depth_log_std"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert stats["effective_reliability_mean"] == pytest.approx(0.5)


def test_refinement_loss_is_zero_for_exact_metric_depth():
    depth = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    batch = {
        "gt_depth": depth,
        "gt_valid": torch.ones_like(depth),
        "gt_weight": torch.ones_like(depth),
        "furniture_mask": torch.zeros_like(depth),
        "bim_depth": depth,
        "bim_valid": torch.ones_like(depth),
    }
    cfg = Config(
        data=Config(min_depth=0.2, max_depth=5.0),
        loss=Config(
            robust_log_depth=1.0,
            robust_log_beta=0.02,
            depth=0.1,
            gradient=0.02,
            furniture_multiplier=1.0,
            bim_foreground_conflict_multiplier=1.0,
        ),
    )
    losses = priorbimda_refinement_loss({"depth": depth}, batch, cfg)
    for value in losses.values():
        torch.testing.assert_close(value, torch.tensor(0.0))


def test_refinement_loss_is_zero_for_exact_relative_disparity():
    depth = torch.full((1, 1, 2, 2), 3.0)
    batch = {
        "gt_depth": depth,
        "gt_valid": torch.ones_like(depth),
        "gt_weight": torch.ones_like(depth),
        "furniture_mask": torch.zeros_like(depth),
        "bim_depth": depth,
        "bim_valid": torch.ones_like(depth),
    }
    cfg = Config(
        data=Config(min_depth=0.2, max_depth=5.0),
        loss=Config(
            normalized_disparity=1.0,
            normalized_disparity_beta=0.02,
            normalized_disparity_max=1000.0,
            robust_log_depth=0.0,
            robust_log_beta=0.02,
            depth=0.1,
            gradient=0.02,
            furniture_multiplier=1.0,
            bim_foreground_conflict_multiplier=1.0,
        ),
    )
    losses = priorbimda_refinement_loss(
        {
            "depth": depth,
            "normalized_disparity": torch.full_like(depth, 2.0),
            "prior_minimum": torch.tensor([[[[2.0]]]]),
            "prior_range": torch.tensor([[[[2.0]]]]),
        },
        batch,
        cfg,
    )
    for value in losses.values():
        torch.testing.assert_close(value, torch.tensor(0.0))


def test_silog_uses_denormalized_metric_depth_without_disparity_target():
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    batch = {
        "gt_depth": target,
        "gt_valid": torch.ones_like(target),
        "gt_weight": torch.ones_like(target),
        "furniture_mask": torch.zeros_like(target),
        "bim_depth": target,
        "bim_valid": torch.ones_like(target),
    }
    cfg = Config(
        data=Config(min_depth=0.2, max_depth=5.0),
        loss=Config(
            silog=1.0,
            silog_beta=0.15,
            normalized_disparity=0.0,
            robust_log_depth=0.0,
            robust_log_beta=0.02,
            depth=0.0,
            gradient=0.0,
            furniture_multiplier=1.0,
            bim_foreground_conflict_multiplier=1.0,
        ),
    )
    losses = priorbimda_refinement_loss({"depth": target}, batch, cfg)
    assert losses["silog"] == pytest.approx(1e-5)
    assert losses["total"] == pytest.approx(1e-5)


class _FakeAttentionHead(nn.Module):
    def __init__(self, refresh: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.iterative_refresh_attention = refresh
        self.huber_delta = 0.5
        self.log_ratio_min = math.log(0.2)
        self.log_ratio_max = math.log(5.0)


class _FakeScaleSystem(nn.Module):
    def __init__(self, refresh: bool = False) -> None:
        super().__init__()
        self.attention_scale = _FakeAttentionHead(refresh)

    def _estimate_attention_scale(self, batch, base):
        return _scale_output()


class _FakeRefiner(nn.Module):
    def __init__(self, output: float = 1.0) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 1, 1)
        self.output = float(output)

    def forward(self, rgb, condition):
        return torch.full(
            (rgb.shape[0], rgb.shape[-2], rgb.shape[-1]),
            self.output,
            device=rgb.device,
        )

    def optimizer_parameter_groups(self, *, encoder_lr, decoder_lr, condition_lr):
        return [{"name": "all", "params": list(self.parameters()), "lr": condition_lr}]


def test_two_stage_wrapper_freezes_stage1_and_keeps_it_in_eval_mode():
    model = PriorBIMDATwoStage(
        _FakeScaleSystem(),
        _FakeRefiner(),
        depth_log_mean=0.0,
        depth_log_std=1.0,
        effective_reliability_mean=0.5,
    ).train()
    assert not model.scale_system.training
    assert all(not parameter.requires_grad for parameter in model.scale_system.parameters())
    assert all(parameter.requires_grad for parameter in model.refiner.parameters())

    batch = {
        "rgb": torch.zeros(1, 3, 1, 2),
        "base_depth": torch.ones(1, 1, 1, 2),
        "scaled_depth": torch.ones(1, 1, 1, 2),
        "bim_depth": torch.tensor([[[[1.0, 2.0]]]]),
        "bim_valid": torch.ones(1, 1, 1, 2),
    }
    output = model(batch)
    assert output["depth"].shape == batch["base_depth"].shape
    assert output["condition"].shape == (1, 3, 1, 2)
    assert output["scale"].requires_grad is False


def test_two_stage_rejects_refreshing_attention():
    with pytest.raises(ValueError, match="fixed attention"):
        PriorBIMDATwoStage(
            _FakeScaleSystem(refresh=True),
            _FakeRefiner(),
            depth_log_mean=0.0,
            depth_log_std=1.0,
            effective_reliability_mean=0.5,
        )


def test_two_stage_denormalizes_refiner_output_with_bim_minmax():
    model = PriorBIMDATwoStage(
        _FakeScaleSystem(),
        _FakeRefiner(output=10.0),
        condition_normalization=PER_FRAME_BIM_MINMAX_NORMALIZATION,
        output_max_depth_m=20.0,
    )
    batch = {
        "rgb": torch.zeros(1, 3, 1, 2),
        "base_depth": torch.ones(1, 1, 1, 2),
        "bim_depth": torch.tensor([[[[2.0, 4.0]]]]),
        "bim_valid": torch.ones(1, 1, 1, 2),
    }
    output = model(batch)
    torch.testing.assert_close(output["normalized_depth"], torch.full((1, 1, 1, 2), 0.5))
    torch.testing.assert_close(output["depth"], torch.full((1, 1, 1, 2), 3.0))
    assert bool(torch.all((output["condition"] >= 0) & (output["condition"] <= 1)))


def test_two_stage_inverts_relative_disparity_before_denormalizing():
    model = PriorBIMDATwoStage(
        _FakeScaleSystem(),
        _FakeRefiner(output=2.0),
        condition_normalization=PER_FRAME_BIM_MINMAX_NORMALIZATION,
        output_max_depth_m=20.0,
        refiner_output_domain=DAV2_RELATIVE_DISPARITY_OUTPUT,
    )
    batch = {
        "rgb": torch.zeros(1, 3, 1, 2),
        "base_depth": torch.ones(1, 1, 1, 2),
        "bim_depth": torch.tensor([[[[2.0, 4.0]]]]),
        "bim_valid": torch.ones(1, 1, 1, 2),
    }
    output = model(batch)
    torch.testing.assert_close(
        output["normalized_disparity"], torch.full((1, 1, 1, 2), 2.0)
    )
    torch.testing.assert_close(output["normalized_depth"], torch.full((1, 1, 1, 2), 0.5))
    torch.testing.assert_close(output["depth"], torch.full((1, 1, 1, 2), 3.0))
