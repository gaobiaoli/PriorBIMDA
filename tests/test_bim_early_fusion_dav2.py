from types import SimpleNamespace

import torch
from torch import nn

from bim_priorda3.early_fusion import dense_metric_depth_loss
from bim_priorda3.models.bim_early_fusion_dav2 import (
    BIMEarlyFusionDepthAnythingV2,
    build_bim_condition,
)


class _FakePatchEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 768, kernel_size=14, stride=14)


class _FakeEmbeddings(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embeddings = _FakePatchEmbeddings()


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _FakeEmbeddings()


class _FakeDAv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeBackbone()
        self.config = SimpleNamespace(
            depth_estimation_type="metric",
            max_depth=20.0,
            backbone_config=SimpleNamespace(patch_size=14, hidden_size=768),
        )


def test_bim_projection_is_strictly_zero_initialized():
    model = BIMEarlyFusionDepthAnythingV2(_FakeDAv2())
    condition = torch.randn(2, 3, 28, 42)
    assert torch.count_nonzero(model.bim_condition_embed(condition)) == 0
    assert torch.count_nonzero(model.bim_condition_embed.weight) == 0
    assert torch.count_nonzero(model.bim_condition_embed.bias) == 0


def test_condition_uses_metric_da3_only_in_valid_disagreement():
    batch = {
        "rgb": torch.zeros(1, 3, 2, 2),
        "base_depth": torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]]),
        "bim_depth": torch.tensor([[[[2.0, 4.0], [8.0, 16.0]]]]),
        "bim_valid": torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
    }
    condition = build_bim_condition(batch, bim_log_mean=0.0, bim_log_std=2.0)
    expected_disagreement = torch.log(torch.tensor(2.0)) / 1.5
    torch.testing.assert_close(condition[0, 2, 0, 0], expected_disagreement)
    torch.testing.assert_close(condition[0, 2, 1, 0], expected_disagreement)
    assert torch.count_nonzero(condition[0, :, 0, 1]) == 0
    assert torch.count_nonzero(condition[0, :, 1, 1]) == 0


def test_dense_loss_uses_only_valid_neighbour_pairs():
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    prediction = target[:, 0].clone()
    valid = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    result = dense_metric_depth_loss(
        prediction,
        target,
        valid,
        min_depth=0.2,
        max_depth=5.0,
        gradient_weight=0.5,
    )
    torch.testing.assert_close(result["total"], torch.tensor(0.0))
    torch.testing.assert_close(result["log_depth"], torch.tensor(0.0))
    torch.testing.assert_close(result["gradient"], torch.tensor(0.0))
    assert int(result["valid_pixels"]) == 3
