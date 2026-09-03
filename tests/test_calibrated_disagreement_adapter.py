from __future__ import annotations

import math

import torch

from bim_priorda3.config import load_config
from bim_priorda3.models import (
    CalibratedDisagreementAdapter,
    build_calibrated_disagreement_condition,
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

    torch.testing.assert_close(condition[:, 0], torch.tensor([[[2.0 / 3.0]]]))
    torch.testing.assert_close(condition[:, 1], torch.tensor([[[4.0 / 3.0]]]))
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

    expected_z = torch.tensor([[[[math.log(2.0), 0.0], [0.0, math.log(2.0)]]]])
    torch.testing.assert_close(condition[:, :1], expected_z)
    torch.testing.assert_close(condition[:, 1:2], expected_z.abs())
    torch.testing.assert_close(condition[:, 2:], mask)


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
