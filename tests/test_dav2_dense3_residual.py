import math

import pytest
import torch
from torch import nn

from bim_priorda3.models.dav2_dense3_residual import (
    DenseResidualOutputHead,
    build_dense3_condition,
    dense_log_depth_loss,
)
from bim_priorda3.models.dav2_dense3_residual_aux72 import build_native_auxiliary_head


def batch():
    return {
        "base_depth": torch.full((2, 1, 4, 4), 2.0),
        "bim_depth": torch.full((2, 1, 4, 4), 6.0),
        "bim_valid": torch.ones(2, 1, 4, 4),
    }


def test_condition_order_and_da3_perturbation_information():
    item = batch()
    condition = build_dense3_condition(item)
    assert condition.shape == (2, 3, 4, 4)
    torch.testing.assert_close(condition[:, 0], torch.full_like(condition[:, 0], math.log(6)))
    torch.testing.assert_close(condition[:, 1], torch.full_like(condition[:, 1], math.log(2)))
    torch.testing.assert_close(condition[:, 2], torch.ones_like(condition[:, 2]))
    item["base_depth"] *= math.exp(0.2)
    perturbed = build_dense3_condition(item)
    torch.testing.assert_close(perturbed[:, 0], condition[:, 0])
    torch.testing.assert_close(perturbed[:, 1], condition[:, 1] + 0.2)


def test_missing_bim_is_finite_and_keeps_da3():
    item = batch()
    item["bim_depth"].fill_(float("nan"))
    item["bim_valid"].zero_()
    condition = build_dense3_condition(item)
    assert torch.isfinite(condition).all()
    assert torch.count_nonzero(condition[:, 0]) == 0
    torch.testing.assert_close(condition[:, 1], item["base_depth"][:, 0].log())
    assert torch.count_nonzero(condition[:, 2]) == 0


def test_log_depth_loss_balances_micro_and_macro():
    gt = torch.tensor([[[[1.0, 2.0]]], [[[1.0, 2.0]]]])
    pred = gt * torch.tensor([2.0, math.exp(0.2)]).view(2, 1, 1, 1)
    valid = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 1.0]]]])
    result = dense_log_depth_loss(pred.requires_grad_(), gt, valid)
    micro = (math.log(2.0) + 0.2 + 0.2) / 3
    macro = (math.log(2.0) + 0.2) / 2
    assert result["total"].item() == pytest.approx(0.5 * (micro + macro))
    result["total"].backward()
    assert torch.isfinite(pred.grad).all()


def test_zero_init_head_and_no_mean_center_constraint():
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 4, 3, padding=1)
            self.conv2 = nn.Conv2d(4, 2, 3, padding=1)

    head = DenseResidualOutputHead(Head())
    output = head(torch.randn(2, 8, 4, 4), output_size=(8, 8))
    assert output.shape == (2, 1, 8, 8)
    assert torch.count_nonzero(output) == 0
    with torch.no_grad():
        head.output_projection.bias.fill_(0.5)
    output = head(torch.randn(2, 8, 4, 4), output_size=(8, 8))
    torch.testing.assert_close(output.mean(dim=(-2, -1)), torch.full((2, 1), 0.5))


def test_auxiliary_head_matches_r18_only_structure_and_zero_init():
    head = build_native_auxiliary_head(128, 64)
    assert isinstance(head[0], nn.Conv2d)
    assert head[0].kernel_size == (3, 3)
    assert (head[0].in_channels, head[0].out_channels) == (128, 64)
    assert isinstance(head[1], nn.GELU)
    assert isinstance(head[2], nn.Conv2d)
    assert head[2].kernel_size == (1, 1)
    output = head(torch.randn(2, 128, 72, 72))
    assert output.shape == (2, 1, 72, 72)
    assert torch.count_nonzero(output) == 0
