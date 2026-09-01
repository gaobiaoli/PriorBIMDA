import torch

from bim_priorda3.models import (
    build_priorda_v11_bim_condition,
    effective_attention_top_prior,
    local_huber_log_scale_field,
)


def test_priorda_condition_matches_official_min_max_disparity_domain():
    bim = torch.tensor([[[[1.0, 2.0], [3.0, 0.0]]]])
    valid = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]])
    global_depth = torch.tensor([[[[2.0, 3.0], [1.0, 2.0]]]])
    local_depth = torch.tensor([[[[1.5, 2.0], [2.5, 3.0]]]])
    condition, minimum, span = build_priorda_v11_bim_condition(
        bim_depth=bim,
        bim_valid=valid,
        global_depth=global_depth,
        local_depth=local_depth,
    )
    torch.testing.assert_close(minimum, torch.tensor([[[[1.0]]]]))
    torch.testing.assert_close(span, torch.tensor([[[[2.0]]]]))
    torch.testing.assert_close(condition[:, 0:1], valid)
    torch.testing.assert_close(
        condition[:, 1:2],
        torch.tensor([[[[2.0, 1.0], [0.0, 2.0]]]]),
    )
    torch.testing.assert_close(
        condition[:, 2:3],
        torch.tensor([[[[4.0, 2.0], [4.0 / 3.0, 1.0]]]]),
    )


def test_priorda_condition_optionally_caps_only_the_inverse_singularity():
    bim = torch.tensor([[[[1.0, 3.0]]]])
    valid = torch.ones_like(bim)
    dense = torch.tensor([[[[1.000001, 2.0]]]])
    condition, _, _ = build_priorda_v11_bim_condition(
        bim_depth=bim,
        bim_valid=valid,
        global_depth=dense,
        local_depth=dense,
        max_disparity=100.0,
    )
    torch.testing.assert_close(condition[:, 0], valid[:, 0])
    assert float(condition[:, 1:].max()) == 100.0
    torch.testing.assert_close(condition[:, 1:, :, 1], torch.full((1, 2, 1), 2.0))


def test_local_huber_field_preserves_a_constant_ratio():
    base = torch.linspace(1.0, 4.0, 17 * 19).reshape(1, 1, 17, 19)
    bim = 1.5 * base
    valid = torch.ones_like(base)
    global_log_scale = base.new_tensor(1.5).log().view(1, 1, 1, 1)
    field, support = local_huber_log_scale_field(
        base,
        bim,
        valid,
        global_log_scale,
        kernel_size=7,
        sigma=2.0,
    )
    torch.testing.assert_close(
        field,
        global_log_scale.expand_as(field),
        atol=2e-6,
        rtol=0,
    )
    assert bool((support == 1).all())


def test_local_huber_field_falls_back_to_global_without_bim_support():
    base = torch.ones(2, 1, 9, 11)
    bim = torch.zeros_like(base)
    valid = torch.zeros_like(base)
    global_log_scale = torch.tensor([0.2, -0.1]).view(2, 1, 1, 1)
    field, support = local_huber_log_scale_field(
        base,
        bim,
        valid,
        global_log_scale,
        kernel_size=5,
        sigma=1.5,
    )
    torch.testing.assert_close(field, global_log_scale.expand_as(field))
    assert torch.count_nonzero(support) == 0


def test_effective_attention_top_prior_combines_attention_and_huber_consistency():
    base = torch.ones(1, 1, 4, 4)
    token_ratios = torch.tensor([[1.0, 1.1], [1.8, 2.5]])
    bim = token_ratios.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
    bim = bim.view(1, 1, 4, 4)
    valid = torch.ones_like(base)
    scale_output = {
        "attention_token_distribution": torch.full((1, 1, 2, 2), 0.25),
        "head_mixture": torch.ones(1, 1),
        "head_log_scale": torch.zeros(1, 1),
        "attention_token_valid": torch.ones(1, 1, 2, 2),
        "log_scale": torch.zeros(1, 1, 1, 1),
    }
    trusted, weight, distribution = effective_attention_top_prior(
        base_depth=base,
        bim_depth=bim,
        bim_valid=valid,
        scale_output=scale_output,
        huber_delta=0.15,
        top_fraction=0.25,
        residual_threshold=0.30,
        ratio_min=0.2,
        ratio_max=5.0,
    )
    expected = torch.zeros_like(base)
    expected[..., :2, :2] = 1
    torch.testing.assert_close(trusted, expected)
    assert bool((weight[trusted.bool()] > 0).all())
    assert tuple(distribution.shape) == (1, 1, 2, 2)
