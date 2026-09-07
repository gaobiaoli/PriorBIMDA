import pytest
import torch

from bim_priorda3.models.priorda_relative_metric_refiner import (
    bim_affine_frame,
    build_priorda_relative_condition,
    depth2disparity,
)


def test_depth2disparity_matches_priorda_exact_positive_rule():
    depth = torch.tensor([[-1.0, 0.0, 0.5, 2.0]])
    actual = depth2disparity(depth)
    assert torch.equal(actual, torch.tensor([[0.0, 0.0, 2.0, 0.5]]))


def test_bim_affine_frame_uses_only_valid_positive_bim_pixels():
    bim = torch.tensor([[[[2.0, 999.0], [4.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    valid, minimum, value_range = bim_affine_frame(bim, mask)
    assert torch.equal(valid, torch.tensor([[[[True, False], [True, False]]]]))
    assert minimum.item() == 2.0
    assert value_range.item() == 2.0


def test_condition_order_and_shared_frame_are_exact():
    batch = {
        "base_depth": torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]]),
        "bim_depth": torch.tensor([[[[2.0, 0.0], [4.0, 0.0]]]]),
        "bim_valid": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
    }
    condition, minimum, value_range = build_priorda_relative_condition(batch)
    assert minimum.item() == 2.0
    assert value_range.item() == 2.0
    # normalized DA3 depth [-0.5, 0, 0.5, 1.5] -> [0, 0, 2, 2/3]
    assert torch.allclose(condition[0, 0], torch.tensor([[0.0, 0.0], [2.0, 2.0 / 3.0]]))
    # normalized valid BIM depth [0, 1] -> both reciprocal convention outputs [0, 1].
    assert torch.equal(condition[0, 1], torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    assert torch.equal(condition[0, 2], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))


def test_bim_affine_frame_rejects_empty_support():
    with pytest.raises(ValueError, match="valid BIM support"):
        bim_affine_frame(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2))
