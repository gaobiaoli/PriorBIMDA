from types import SimpleNamespace

import torch
from torch import nn

from bim_priorda3.models.bim_early_fusion_dav2_scale import (
    BIMEarlyFusionDAv2ScaleRegressor,
    scale_regression_loss,
)


class _FakePatchEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 768, kernel_size=14, stride=14)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values).flatten(2).transpose(1, 2)


class _FakeEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = _FakePatchEmbeddings()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.dropout = nn.Identity()

    @staticmethod
    def interpolate_pos_encoding(
        tokens: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        del height, width
        return torch.zeros_like(tokens)


class _FakeEncoder(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_: object) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=hidden_states)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _FakeEmbeddings()
        self.encoder = _FakeEncoder()
        self.layernorm = nn.LayerNorm(768)

    def gradient_checkpointing_enable(self) -> None:
        return None


class _FakeDAv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.config = SimpleNamespace(
            depth_estimation_type="metric",
            backbone_config=SimpleNamespace(patch_size=14, hidden_size=768),
        )


def test_zero_initialized_bim_projection_makes_initial_prediction_condition_invariant():
    torch.manual_seed(3)
    model = BIMEarlyFusionDAv2ScaleRegressor(_FakeDAv2())
    model.eval()
    rgb = torch.rand(2, 3, 28, 42)
    base = torch.rand(2, 1, 28, 42) + 0.5
    first = model(rgb, torch.randn(2, 3, 28, 42), base)
    second = model(rgb, torch.randn(2, 3, 28, 42), base)
    torch.testing.assert_close(first["log_scale"], second["log_scale"])
    assert torch.count_nonzero(model.bim_condition_embed.weight) == 0
    assert first["log_scale"].shape == (2, 1, 1, 1)
    assert float(first["log_scale"].detach().abs().max()) < 0.1


def test_scale_loss_trains_bim_projection_and_scalar_head_from_final_gt():
    torch.manual_seed(5)
    model = BIMEarlyFusionDAv2ScaleRegressor(_FakeDAv2())
    model.train()
    rgb = torch.rand(2, 3, 28, 28)
    condition = torch.randn(2, 3, 28, 28)
    base = torch.ones(2, 1, 28, 28)
    target = torch.full_like(base, 1.25)
    output = model(rgb, condition, base)
    oracle = torch.full((2, 1, 1, 1), torch.log(torch.tensor(1.25)))
    batch = {
        "gt_depth": target,
        "gt_valid": torch.ones_like(target),
    }
    losses = scale_regression_loss(
        output,
        batch,
        pixel_weight=torch.ones_like(target),
        oracle_log_scale=oracle,
        oracle_supported=torch.ones(2, dtype=torch.bool),
        depth_weight=1.0,
        coarse_depth_weight=0.5,
        oracle_weight=0.5,
        oracle_beta=0.02,
    )
    losses["total"].backward()
    condition_gradient = model.bim_condition_embed.weight.grad
    output_gradient = model.scale_head[-1].weight.grad
    assert condition_gradient is not None and bool((condition_gradient != 0).any())
    assert output_gradient is not None and bool((output_gradient != 0).any())


def test_optimizer_groups_cover_every_parameter_once():
    model = BIMEarlyFusionDAv2ScaleRegressor(_FakeDAv2())
    groups = model.optimizer_parameter_groups(
        encoder_lr=5e-6,
        condition_lr=5e-5,
        scale_head_lr=5e-5,
    )
    grouped = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {id(parameter) for parameter in model.parameters()}
    assert [group["name"] for group in groups] == [
        "dinov2_encoder",
        "bim_condition_projection",
        "scale_regression_head",
    ]
