import pytest
import torch

from bim_priorda3.models.priorda_relative_metric_refiner import (
    bim_affine_frame,
    bim_zero_anchor_scale,
    build_prior_identity_condition,
    build_priorda_relative_condition,
    build_priorda_zero_anchor_condition,
    depth2disparity,
    fit_disparity_affine,
    prior_affine_frame,
    transform_relative_to_prior_normalized_disparity,
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


def test_zero_anchor_scale_uses_only_valid_positive_bim_maximum():
    bim = torch.tensor([[[[2.0, 999.0], [4.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])
    valid, scale = bim_zero_anchor_scale(bim, mask)
    assert torch.equal(valid, torch.tensor([[[[True, False], [True, False]]]]))
    assert scale.item() == 4.0


def test_zero_anchor_condition_preserves_depth_below_old_bim_minimum():
    base = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]])
    batch = {
        "base_depth": base,
        "bim_depth": torch.tensor([[[[2.0, 0.0], [4.0, 0.0]]]]),
        "bim_valid": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
    }
    condition, scale = build_priorda_zero_anchor_condition(batch)
    assert scale.item() == 4.0
    assert torch.allclose(
        condition[0, 0],
        torch.tensor([[4.0, 2.0], [4.0 / 3.0, 0.8]]),
    )
    assert condition[0, 0, 0, 0] > 0  # D_raw=1 m < old D_BIM_min=2 m.
    assert torch.equal(condition[0, 1], torch.tensor([[2.0, 0.0], [1.0, 0.0]]))
    assert torch.equal(condition[0, 2], torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    recovered = depth2disparity(condition[:, :1]) * scale
    assert torch.allclose(recovered, base)


def test_zero_anchor_output_roundtrip_has_no_bim_minimum_floor():
    scale = torch.tensor([[[[4.0]]]])
    metric_depth = torch.tensor([[[[0.25, 1.0], [4.0, 8.0]]]])
    normalized_disparity = scale / metric_depth
    recovered = depth2disparity(normalized_disparity) * scale
    assert torch.equal(recovered, metric_depth)


def test_prior_identity_condition_uses_dense_prior_frame():
    prior = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]])
    batch = {
        "base_depth": prior,
        "bim_depth": torch.tensor([[[[1.5, 0.0], [4.0, 0.0]]]]),
        "bim_valid": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
    }
    condition, minimum, value_range, q_prior = build_prior_identity_condition(batch)
    assert minimum.item() == 1.0
    assert value_range.item() == 4.0
    expected_prior_q = torch.tensor([[[[0.0, 4.0], [2.0, 1.0]]]])
    assert torch.equal(q_prior, expected_prior_q)
    assert torch.equal(condition[:, :1], expected_prior_q)
    assert torch.equal(condition[:, 2:], batch["bim_valid"])


def test_affine_transform_is_exact_when_inverse_prior_is_affine_in_relative_q():
    q_relative = torch.tensor([[[[0.25, 0.75], [1.25, 2.0]]]])
    slope_expected = 0.6
    intercept_expected = 0.2
    prior = (slope_expected * q_relative + intercept_expected).reciprocal()
    minimum, value_range = prior_affine_frame(prior)
    transformed, slope, intercept = transform_relative_to_prior_normalized_disparity(
        q_relative, prior, minimum, value_range
    )
    expected = depth2disparity((prior - minimum) / value_range)
    assert torch.allclose(slope, torch.tensor([[[[slope_expected]]]]), atol=1e-6)
    assert torch.allclose(intercept, torch.tensor([[[[intercept_expected]]]]), atol=1e-6)
    assert torch.allclose(transformed, expected, atol=1e-5, rtol=1e-5)


def test_affine_fit_is_only_a_best_fit_for_non_affine_prior():
    q_relative = torch.tensor([[[[0.2, 0.6], [1.1, 1.8]]]])
    prior = torch.tensor([[[[3.0, 2.1], [1.3, 0.9]]]])
    slope, intercept = fit_disparity_affine(q_relative, prior)
    fitted_inverse = slope * q_relative + intercept
    assert not torch.allclose(fitted_inverse, prior.reciprocal(), atol=1e-6, rtol=1e-6)


def test_metric_identity_residual_has_no_prior_minimum_floor():
    prior = torch.tensor([[[[1.0, 2.0], [3.0, 5.0]]]])
    zero = torch.zeros_like(prior)
    assert torch.equal(prior * torch.exp(zero), prior)
    below_minimum = prior * torch.exp(torch.full_like(prior, -0.7))
    assert below_minimum[0, 0, 0, 0] < prior.amin()
