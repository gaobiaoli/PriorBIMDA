import torch

from bim_priorda3.config import load_config
from bim_priorda3.losses import BIMPriorLoss
from bim_priorda3.models import BIMPriorDA3


def test_model_forward_and_loss_are_finite() -> None:
    cfg = load_config("configs/slabim_single_frame.yaml")
    cfg.model.trust_channels = 8
    cfg.model.base_channels = 8
    cfg.model.alignment_kernel = 7
    cfg.model.alignment_downsample = 2
    model = BIMPriorDA3(cfg)
    height = width = 64
    base = torch.rand(1, 1, height, width) * 2 + 0.5
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.rand(1, 1, height, width),
        "bim_depth": base * 1.1,
        "bim_valid": torch.ones(1, 1, height, width),
        "bim_normals": torch.rand(1, 3, height, width),
        "bim_edge": torch.zeros(1, 1, height, width),
        "gt_depth": base * 1.05,
        "gt_valid": torch.ones(1, 1, height, width),
        "gt_weight": torch.ones(1, 1, height, width),
        "trust_target": torch.ones(1, 1, height, width),
        "trust_mask": torch.ones(1, 1, height, width),
    }
    output = model(batch)
    assert output["depth"].shape == base.shape
    assert torch.allclose(output["depth"], base)
    losses = BIMPriorLoss(cfg)(output, batch)
    assert all(torch.isfinite(value) for value in losses.values())


def test_strong_anchor_model_is_initialized_as_exact_baseline() -> None:
    cfg = load_config("configs/slabim_single_frame_r50_v21.yaml")
    cfg.model.trust_channels = 8
    cfg.model.base_channels = 8
    model = BIMPriorDA3(cfg)
    height = width = 64
    base = torch.rand(1, 1, height, width) * 2 + 0.5
    scaled = base * 1.15
    anchor = scaled * 1.02
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.rand(1, 1, height, width),
        "scaled_depth": scaled,
        "anchor_depth": anchor,
        "anchor_field": torch.zeros(1, 1, height, width),
        "anchor_support": torch.ones(1, 1, height, width),
        "bim_depth": anchor,
        "bim_valid": torch.ones(1, 1, height, width),
        "bim_normals": torch.rand(1, 3, height, width),
        "bim_edge": torch.zeros(1, 1, height, width),
        "gt_depth": anchor,
        "gt_valid": torch.ones(1, 1, height, width),
        "gt_weight": torch.ones(1, 1, height, width),
        "trust_target": torch.ones(1, 1, height, width),
        "trust_mask": torch.ones(1, 1, height, width),
    }
    output = model(batch)
    assert torch.allclose(output["depth"], anchor)
    assert torch.allclose(output["coarse_depth"], anchor)
    losses = BIMPriorLoss(cfg)(output, batch)
    assert all(torch.isfinite(value) for value in losses.values())


def test_candidate_fusion_stays_between_two_inputs() -> None:
    cfg = load_config("configs/slabim_single_frame_r50_v3.yaml")
    cfg.model.trust_channels = 8
    model = BIMPriorDA3(cfg)
    height = width = 64
    base = torch.rand(1, 1, height, width) * 2 + 0.5
    anchor = base * 1.1
    candidate = base * 0.9
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.rand(1, 1, height, width),
        "scaled_depth": anchor,
        "anchor_depth": anchor,
        "anchor_field": torch.zeros(1, 1, height, width),
        "anchor_support": torch.ones(1, 1, height, width),
        "candidate_depth": candidate,
        "candidate_log_variance": torch.zeros(1, 1, height, width),
        "candidate_trust": torch.ones(1, 1, height, width),
        "candidate_frame_trust": torch.ones(1),
        "bim_depth": anchor,
        "bim_valid": torch.ones(1, 1, height, width),
        "bim_normals": torch.rand(1, 3, height, width),
        "bim_edge": torch.zeros(1, 1, height, width),
        "gt_depth": base,
        "gt_valid": torch.ones(1, 1, height, width),
        "gt_weight": torch.ones(1, 1, height, width),
        "trust_target": torch.ones(1, 1, height, width),
        "trust_mask": torch.ones(1, 1, height, width),
    }
    output = model(batch)
    lower = torch.minimum(anchor, candidate)
    upper = torch.maximum(anchor, candidate)
    assert torch.all(output["depth"] >= lower)
    assert torch.all(output["depth"] <= upper)
    losses = BIMPriorLoss(cfg)(output, batch)
    assert all(torch.isfinite(value) for value in losses.values())
    model.eval()
    with torch.no_grad():
        safe_output = model(batch)
    assert torch.allclose(safe_output["depth"], anchor)
