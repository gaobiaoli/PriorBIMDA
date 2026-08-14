from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional

from bim_priorda3.baselines import (
    estimate_robust_bim_scale,
    robust_scale_and_local_features,
)
from bim_priorda3.config import Config, load_config
from bim_priorda3.losses import (
    BIMPriorLoss,
    build_depth_supervision_weight,
    build_live_trust_target,
)
from bim_priorda3.models import BIMPriorDA3


class _FakeDA3Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, images: torch.Tensor, **_: object):
        batch, views, channels, height, width = images.shape
        features = self.projection(images.reshape(batch * views, channels, height, width))
        return features.reshape(batch, views, 4, height, width), ()


class _FakeDA3Scratch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.refinenet1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.output_conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.output_conv2 = nn.Conv2d(4, 1, kernel_size=1)
        self.sky_output_conv2 = nn.Conv2d(4, 1, kernel_size=1)


class _FakeDA3Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scratch = _FakeDA3Scratch()
        self.frozen_projection = nn.Conv2d(4, 4, kernel_size=1)


class _FakeDA3Net(nn.Module):
    PATCH_SIZE = 14

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _FakeDA3Backbone()
        self.head = _FakeDA3Head()

    def _process_depth_head(
        self,
        features: torch.Tensor,
        height: int,
        width: int,
    ) -> dict[str, torch.Tensor]:
        batch, views, channels, _, _ = features.shape
        value = features.reshape(batch * views, channels, height, width)
        value = functional.relu(self.head.scratch.refinenet1(value))
        value = functional.relu(self.head.scratch.output_conv1(value))
        depth = functional.softplus(self.head.scratch.output_conv2(value)).reshape(
            batch, views, height, width
        )
        sky = self.head.scratch.sky_output_conv2(value).reshape(
            batch,
            views,
            height,
            width,
        )
        return {"depth": depth, "sky": sky}


def test_depth_supervision_weight_adds_near_furniture_and_conflict_emphasis() -> None:
    shape = (1, 1, 1, 4)
    batch = {
        "gt_weight": torch.ones(shape),
        "gt_valid": torch.ones(shape),
        "gt_depth": torch.tensor([[[[0.5, 2.0, 2.0, 2.0]]]]),
        "bim_valid": torch.ones(shape),
        "bim_depth": torch.tensor([[[[0.5, 2.0, 3.0, 2.0]]]]),
        "furniture_mask": torch.tensor([[[[0.0, 1.0, 0.0, 0.0]]]]),
    }
    weights = Config(
        {
            "near_range_boost": 2.0,
            "furniture_multiplier": 1.5,
            "bim_foreground_conflict_multiplier": 1.75,
        }
    )
    actual = build_depth_supervision_weight(batch, weights)
    expected = torch.tensor([[[[3.0, 1.5, 1.75, 1.0]]]])
    assert torch.equal(actual, expected)


def test_v5_is_initialized_as_scale_only_and_has_residual_gradients() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    height = width = 64
    base = torch.rand(2, 1, height, width) * 2 + 0.5
    scaled = base * 1.15
    bim = scaled * 1.03
    batch = {
        "rgb": torch.rand(2, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.rand(2, 1, height, width),
        "scaled_depth": scaled,
        "anchor_depth": scaled * 1.01,
        "bim_depth": bim,
        "bim_valid": torch.ones(2, 1, height, width),
        "bim_normals": torch.rand(2, 3, height, width),
        "bim_edge": torch.zeros(2, 1, height, width),
        "gt_depth": scaled * 0.95,
        "gt_valid": torch.ones(2, 1, height, width),
        "gt_weight": torch.ones(2, 1, height, width),
        "trust_target": torch.zeros(2, 1, height, width),
        "trust_mask": torch.ones(2, 1, height, width),
    }
    output = model(batch)
    assert torch.allclose(output["depth"], scaled)
    assert torch.allclose(output["coarse_depth"], scaled)
    assert torch.equal(output["refinement_anchor_depth"], scaled)
    assert output["residual_anchor_mode"] == "scaled_depth"
    assert output["residual_routing_scope"] == "frame_and_low"
    assert output["geometry_scale_channel_semantics"] == "log(scaled/base)/0.5"
    assert torch.allclose(
        output["geometry_scale_channel"],
        ((torch.log(scaled) - torch.log(base)) / 0.5).clamp(-2.0, 2.0),
    )
    assert torch.count_nonzero(output["log_residual"]) == 0

    losses = BIMPriorLoss(cfg)(output, batch)
    losses["total"].backward()
    assert model.refiner.detail_output.weight.grad is not None
    assert model.refiner.low_output.weight.grad is not None
    assert model.refiner.frame_output.weight.grad is not None
    assert torch.count_nonzero(model.refiner.detail_output.weight.grad) > 0
    assert torch.count_nonzero(model.refiner.low_output.weight.grad) > 0
    assert torch.count_nonzero(model.refiner.frame_output.weight.grad) > 0


def test_universal_model_ignores_direct_baseline_as_residual_anchor() -> None:
    cfg = load_config("configs/stanford_area1.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    height = width = 32
    base = torch.ones(1, 1, height, width)
    scaled = torch.full_like(base, 2.0)
    robust_direct = torch.full_like(base, 12.0)
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.ones_like(base),
        "scaled_depth": scaled,
        "anchor_depth": robust_direct,
        "bim_depth": torch.full_like(base, 3.0),
        "bim_valid": torch.ones_like(base),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.zeros_like(base),
        "gt_depth": scaled.clone(),
        "gt_valid": torch.ones_like(base),
        "gt_weight": torch.ones_like(base),
        "furniture_mask": torch.zeros_like(base),
        "trust_target": torch.ones_like(base),
        "trust_mask": torch.ones_like(base),
    }

    output = model(batch)

    assert torch.equal(output["coarse_depth"], scaled)
    assert torch.equal(output["refinement_anchor_depth"], scaled)
    assert torch.equal(output["depth"], scaled)
    assert output["residual_anchor_mode"] == "scaled_depth"
    assert output["geometry_scale_channel_semantics"] == "log(scaled/base)/0.5"
    assert torch.equal(
        output["geometry_scale_channel"],
        (torch.log(scaled) - torch.log(base)) / 0.5,
    )
    assert output["residual_routing_depth_semantics"] == "scaled_depth"
    assert torch.equal(output["residual_routing_depth"], scaled)
    losses = BIMPriorLoss(cfg)(output, batch)
    assert float(losses["residual_teacher"]) == 0.0
    assert float(losses["frame_residual_teacher"]) == 0.0
    assert float(losses["local_residual_teacher"]) == 0.0


def test_removed_direct_anchor_configuration_is_rejected() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.residual_anchor_mode = "robust_bim_direct"

    with pytest.raises(ValueError, match="residual_anchor_mode was removed"):
        BIMPriorDA3(cfg)


def test_v5_handles_missing_bim_and_respects_total_residual_bound() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    cfg.model.max_total_log_residual = 0.1
    model = BIMPriorDA3(cfg)
    height = width = 32
    base = torch.rand(1, 1, height, width) + 0.5
    scaled = base * 1.1
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.rand(1, 1, height, width),
        "scaled_depth": scaled,
        "anchor_depth": scaled,
        "bim_depth": torch.zeros_like(base),
        "bim_valid": torch.zeros_like(base),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.ones_like(base),
    }
    with torch.no_grad():
        model.refiner.detail_output.bias[0] = 100
        model.refiner.low_output.bias[0] = 100
        model.refiner.frame_output.bias[0] = 100
        output = model(batch)
    assert torch.all(torch.isfinite(output["depth"]))
    assert torch.all(output["bim_reliability"] == 0)
    assert float(output["log_residual"].abs().max()) <= 0.100001


def test_model_rejects_nonzero_bim_depth_outside_declared_support() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    batch = {
        "bim_depth": torch.tensor([[[[0.0, 1e-4]]]]),
        "bim_valid": torch.zeros(1, 1, 1, 2),
    }

    with pytest.raises(ValueError, match=r"atol=1e-06.*violations=1"):
        model(batch)


def test_configured_cpu_bim_direct_applies_the_explicit_bim_mask() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    base = torch.full((1, 1, 20, 20), 5e-8)
    tolerated_roundoff = torch.full_like(base, 1e-7)
    invalid = torch.zeros_like(base)
    model._validate_bim_depth_mask_contract({"bim_depth": tolerated_roundoff, "bim_valid": invalid})

    direct = model._configured_fixed_bim_direct(
        base,
        tolerated_roundoff,
        invalid,
    )

    assert torch.equal(direct, base)


def test_v5_depth_routing_suppresses_coarse_residuals_near_camera() -> None:
    cfg = load_config("configs/slabim.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    height = width = 32
    base = torch.ones(1, 1, height, width)
    scaled = base.clone()
    scaled[..., :, width // 2 :] = 2.0
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.ones_like(base),
        "scaled_depth": scaled,
        "anchor_depth": scaled,
        "bim_depth": scaled,
        "bim_valid": torch.ones_like(base),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.zeros_like(base),
    }
    with torch.no_grad():
        model.refiner.frame_output.bias[0] = 1.0
        model.refiner.low_output.bias[0] = 1.0
        output = model(batch)
    gate = output["residual_routing_gate"]
    raw_frame = output["raw_frame_log_residual"]
    effective_frame = output["frame_log_residual"]
    assert float(gate[..., 0].mean()) < float(gate[..., -1].mean())
    assert raw_frame.shape == (1, 1, 1, 1)
    assert torch.allclose(effective_frame, raw_frame * gate)
    assert torch.allclose(output["low_log_residual"], output["raw_low_log_residual"])
    assert output["residual_routing_scope"] == "frame_only"
    assert float(effective_frame[..., 0].mean()) < float(effective_frame[..., -1].mean())


def test_frame_only_routing_preserves_low_residual_capacity_near_camera() -> None:
    cfg = load_config("configs/slabim.yaml")
    cfg.model.base_channels = 4
    cfg.model.residual_routing_scope = "frame_only"
    model = BIMPriorDA3(cfg)
    height = width = 32
    base = torch.ones(1, 1, height, width)
    scaled = base.clone()
    scaled[..., :, width // 2 :] = 2.0
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.ones_like(base),
        "scaled_depth": scaled,
        "anchor_depth": scaled,
        "bim_depth": scaled,
        "bim_valid": torch.ones_like(base),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.zeros_like(base),
    }
    with torch.no_grad():
        model.refiner.frame_output.bias[0] = 1.0
        model.refiner.low_output.bias[0] = 1.0
        output = model(batch)

    gate = output["residual_routing_gate"]
    assert torch.allclose(
        output["frame_log_residual"],
        output["raw_frame_log_residual"] * gate,
    )
    assert torch.equal(
        output["low_log_residual"],
        output["raw_low_log_residual"],
    )
    assert output["residual_routing_scope"] == "frame_only"


def test_universal_model_routes_frame_by_scaled_depth() -> None:
    cfg = load_config("configs/stanford_area1.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    height = width = 32
    base = torch.ones(1, 1, height, width)
    scaled = torch.full_like(base, 2.0)
    scaled[..., :, : width // 2] = 0.5
    anchor = torch.full_like(base, 9.0)
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.ones_like(base),
        "scaled_depth": scaled,
        "anchor_depth": anchor,
        "bim_depth": scaled,
        "bim_valid": torch.ones_like(base),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.zeros_like(base),
    }
    with torch.no_grad():
        model.refiner.frame_output.bias[0] = 1.0
        output = model(batch)

    gate = output["residual_routing_gate"]
    assert float(gate[..., : width // 2].mean()) < float(gate[..., width // 2 :].mean())
    assert torch.equal(output["residual_routing_depth"], scaled)
    assert output["residual_routing_depth_semantics"] == "scaled_depth"


def test_gated_adapters_preserve_scale_initialization_and_learn_gate() -> None:
    cfg = load_config("configs/slabim_pretrain.yaml")
    cfg.model.gate_bim_adapters = True
    cfg.model.bim_adapter_gate_floor = 0.25
    cfg.loss.adapter_gate = 0.05
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    height = width = 64
    base = torch.rand(2, 1, height, width) + 0.5
    scaled = base * 1.1
    batch = {
        "rgb": torch.rand(2, 3, height, width),
        "base_depth": base,
        "base_confidence": torch.rand(2, 1, height, width),
        "scaled_depth": scaled,
        "anchor_depth": scaled,
        "bim_depth": scaled * 1.05,
        "bim_valid": torch.ones_like(base),
        "bim_normals": torch.rand(2, 3, height, width),
        "bim_edge": torch.zeros_like(base),
        "gt_depth": scaled * 0.95,
        "gt_valid": torch.ones_like(base),
        "gt_weight": torch.ones_like(base),
        "trust_target": torch.ones_like(base),
        "trust_mask": torch.ones_like(base),
    }

    output = model(batch)
    assert torch.allclose(output["depth"], scaled)
    assert torch.count_nonzero(output["log_residual"]) == 0
    assert torch.count_nonzero(output["bim_adapter_gate_logits"]) == 0

    losses = BIMPriorLoss(cfg)(output, batch)
    assert torch.isfinite(losses["total"])
    assert float(losses["adapter_gate"]) > 0
    losses["total"].backward()
    for gate in model.refiner.bim_adapter_gates:
        gate_grad = gate.weight.grad
        assert gate_grad is not None
        assert torch.count_nonzero(gate_grad) > 0


def test_gated_adapters_share_identical_common_initialization() -> None:
    plain_cfg = load_config("configs/slabim_pretrain.yaml")
    gated_cfg = load_config("configs/slabim_pretrain.yaml")
    gated_cfg.model.gate_bim_adapters = True
    gated_cfg.model.bim_adapter_gate_floor = 0.25
    gated_cfg.loss.adapter_gate = 0.05
    plain_cfg.model.base_channels = 4
    gated_cfg.model.base_channels = 4

    torch.manual_seed(1234)
    plain = BIMPriorDA3(plain_cfg)
    torch.manual_seed(1234)
    gated = BIMPriorDA3(gated_cfg)

    plain_state = plain.state_dict()
    gated_state = gated.state_dict()
    assert set(plain_state) < set(gated_state)
    for name, value in plain_state.items():
        assert torch.equal(value, gated_state[name]), name


def test_e2e_da3_last_stage_has_live_depth_scale_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    cfg.model.e2e_da3 = Config(
        {
            "enabled": True,
            "trainable_scope": "last_stage",
        }
    )
    fake_da3 = _FakeDA3Net()
    model = BIMPriorDA3(cfg, da3_model=fake_da3)
    model.train()

    height = width = 28
    cached_base = torch.full((2, 1, height, width), 9.0)
    bim = torch.full_like(cached_base, 2.0)
    batch = {
        "rgb": torch.rand(2, 3, height, width),
        "base_depth": cached_base,
        "base_confidence": torch.zeros_like(cached_base),
        "scaled_depth": cached_base,
        "anchor_depth": torch.full_like(cached_base, 2.1),
        "bim_depth": bim,
        "bim_valid": torch.ones_like(cached_base),
        "bim_normals": torch.zeros(2, 3, height, width),
        "bim_edge": torch.zeros_like(cached_base),
        "gt_depth": torch.full_like(cached_base, 1.9),
        "gt_valid": torch.ones_like(cached_base),
        "gt_weight": torch.ones_like(cached_base),
        # These deliberately stale labels must not be used in E2E mode.
        "trust_target": torch.zeros_like(cached_base),
        "trust_mask": torch.zeros_like(cached_base),
    }
    output = model(batch)
    assert output["uses_live_da3"] is True
    assert not torch.equal(output["base_depth"], cached_base)
    assert output["base_confidence"].requires_grad is False
    assert output["da3_scale"].requires_grad is False
    assert output["scaled_depth"].requires_grad is True
    assert output["live_bim_direct"].requires_grad is False
    assert torch.all(output["da3_scale_support"] == height * width)

    for sample_index in range(output["base_depth"].shape[0]):
        _, expected_direct, _, _, _ = robust_scale_and_local_features(
            output["base_depth"][sample_index, 0].detach().cpu().numpy(),
            bim[sample_index, 0].numpy(),
            q10_log_cap=float("inf"),
            q25_log_cap=0.05,
        )
        assert torch.equal(
            output["live_bim_direct"][sample_index, 0],
            torch.from_numpy(expected_direct),
        )

    groups = model.trainable_parameter_names()
    assert groups["da3"]
    assert all(
        name.startswith(
            (
                "da3.head.scratch.refinenet1.",
                "da3.head.scratch.output_conv1.",
                "da3.head.scratch.output_conv2.",
            )
        )
        for name in groups["da3"]
    )
    assert all(not parameter.requires_grad for parameter in model.da3.backbone.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.da3.head.frozen_projection.parameters()
    )

    criterion = BIMPriorLoss(cfg)
    criterion.set_epoch(int(cfg.loss.warmup_epochs))
    live_anchor_output = dict(output)
    live_anchor_output["depth"] = batch["gt_depth"] * 1.10
    live_anchor_output["live_robust_bim_direct"] = batch["gt_depth"] * 1.20
    batch["anchor_depth"] = batch["gt_depth"].clone()
    live_anchor_losses = criterion(live_anchor_output, batch)
    assert float(live_anchor_losses["degradation"]) == 0.0

    live_anchor_output["live_robust_bim_direct"] = batch["gt_depth"] * 1.05
    live_anchor_losses = criterion(live_anchor_output, batch)
    expected_degradation = torch.log(torch.tensor(1.10 / 1.05))
    assert live_anchor_losses["degradation"] == pytest.approx(float(expected_degradation))

    losses = criterion(output, batch)
    assert torch.isfinite(losses["total"])
    assert float(losses["trust"]) > 0
    losses["total"].backward()
    assert all(parameter.grad is None for parameter in model.da3.backbone.parameters())
    for module_name in (
        "head.scratch.refinenet1",
        "head.scratch.output_conv1",
        "head.scratch.output_conv2",
    ):
        module = model.da3.get_submodule(module_name)
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in module.parameters()
        )

    original_fixed_bim_direct = BIMPriorDA3._configured_fixed_bim_direct
    fixed_bim_direct_calls = 0

    def tracked_fixed_bim_direct(
        self: BIMPriorDA3,
        base_depth: torch.Tensor,
        bim_depth: torch.Tensor,
        bim_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal fixed_bim_direct_calls
        fixed_bim_direct_calls += 1
        return original_fixed_bim_direct(self, base_depth, bim_depth, bim_valid)

    monkeypatch.setattr(
        BIMPriorDA3,
        "_configured_fixed_bim_direct",
        tracked_fixed_bim_direct,
    )
    model.eval()
    inference_output = model(batch)
    assert "live_bim_direct" not in inference_output
    assert fixed_bim_direct_calls == 0

    requested_batch = dict(batch)
    requested_batch["request_live_bim_direct"] = True
    requested_output = model(requested_batch)
    assert "live_bim_direct" in requested_output
    assert fixed_bim_direct_calls == 1


def test_da3_confidence_proxy_stays_finite_inside_amp() -> None:
    depth = torch.rand(2, 1, 28, 28) + 0.5
    with torch.autocast("cpu", dtype=torch.bfloat16):
        confidence = BIMPriorDA3._depth_confidence(depth)
    assert confidence.dtype == depth.dtype
    assert torch.all(torch.isfinite(confidence))
    assert torch.all((confidence >= 0) & (confidence <= 1))


def test_e2e_robust_scale_and_live_direct_match_authoritative_cpu() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    cfg.model.scale_estimator = Config(
        {
            "name": "log_upper_cap_v1",
            "q10_log_cap": 0.20,
            "q25_log_cap": 0.05,
            "min_samples": 10,
        }
    )
    cfg.model.e2e_da3 = Config({"enabled": True, "trainable_scope": "last_stage"})
    model = BIMPriorDA3(cfg, da3_model=_FakeDA3Net()).eval()
    height = width = 28
    ratios = torch.cat(
        (
            torch.full((150,), 1.0),
            torch.full((150,), 1.1),
            torch.full((height * width - 300,), 3.0),
        )
    ).reshape(1, 1, height, width)
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": torch.ones(1, 1, height, width),
        "base_confidence": torch.ones(1, 1, height, width),
        "scaled_depth": torch.ones(1, 1, height, width),
        "anchor_depth": torch.ones(1, 1, height, width),
        "bim_depth": ratios,
        "bim_valid": torch.ones(1, 1, height, width),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.zeros(1, 1, height, width),
        "request_live_bim_direct": True,
    }
    output = model(batch)
    live_base = output["base_depth"][0, 0].detach().numpy()
    bim = ratios[0, 0].numpy()
    expected_scale = estimate_robust_bim_scale(
        live_base,
        bim,
        q10_log_cap=0.20,
        q25_log_cap=0.05,
        min_samples=10,
    )
    _, expected_direct, _, _, _ = robust_scale_and_local_features(
        live_base,
        bim,
        q10_log_cap=0.20,
        q25_log_cap=0.05,
        min_samples=10,
    )
    assert float(output["da3_scale"]) == pytest.approx(
        expected_scale.scale,
        rel=1e-5,
    )
    assert torch.allclose(
        output["live_robust_bim_direct"][0, 0],
        torch.from_numpy(expected_direct),
    )
    assert torch.equal(output["live_bim_direct"], output["live_robust_bim_direct"])


def test_e2e_uses_live_universal_scale_but_not_live_direct_as_anchor() -> None:
    cfg = load_config("configs/stanford_area1_e2e.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg, da3_model=_FakeDA3Net()).eval()
    height = width = 28
    stale_cached_anchor = torch.full((1, 1, height, width), 9.0)
    batch = {
        "rgb": torch.rand(1, 3, height, width),
        "base_depth": torch.ones(1, 1, height, width),
        "base_confidence": torch.ones(1, 1, height, width),
        "scaled_depth": torch.ones(1, 1, height, width),
        "anchor_depth": stale_cached_anchor,
        "bim_depth": torch.full((1, 1, height, width), 2.0),
        "bim_valid": torch.ones(1, 1, height, width),
        "bim_normals": torch.zeros(1, 3, height, width),
        "bim_edge": torch.zeros(1, 1, height, width),
        "request_live_bim_direct": True,
    }

    output = model(batch)

    assert "live_robust_bim_direct" in output
    assert torch.equal(
        output["live_bim_direct"],
        output["live_robust_bim_direct"],
    )
    assert torch.equal(
        output["refinement_anchor_depth"],
        output["scaled_depth"],
    )
    assert torch.equal(output["depth"], output["scaled_depth"])
    assert not torch.equal(output["depth"], stale_cached_anchor)
    batch.update(
        {
            "gt_depth": output["scaled_depth"].detach().clone(),
            "gt_valid": torch.ones_like(stale_cached_anchor),
            "gt_weight": torch.ones_like(stale_cached_anchor),
            "furniture_mask": torch.zeros_like(stale_cached_anchor),
            "trust_target": torch.zeros_like(stale_cached_anchor),
            "trust_mask": torch.ones_like(stale_cached_anchor),
        }
    )
    losses = BIMPriorLoss(cfg)(output, batch)
    assert float(losses["residual_teacher"]) == 0.0


def test_deprecated_e2e_scale_fields_are_rejected() -> None:
    cfg = load_config("configs/slabim_e2e.yaml")
    cfg.model.base_channels = 4
    cfg.model.e2e_da3.scale_quantile = 0.45
    with pytest.raises(ValueError, match="Deprecated model.e2e_da3 scale fields"):
        BIMPriorDA3(cfg, da3_model=_FakeDA3Net())


def test_live_trust_target_uses_current_scaled_depth() -> None:
    shape = (1, 1, 4, 4)
    gt = torch.full(shape, 2.0)
    bim = torch.full(shape, 2.05)
    scaled_good = torch.full(shape, 2.01)
    scaled_bad = torch.full(shape, 3.0)
    valid = torch.ones(shape)

    target_good, mask_good = build_live_trust_target(
        scaled_good,
        bim,
        valid,
        gt,
        valid,
        margin=0.005,
        temperature=0.03,
    )
    target_bad, mask_bad = build_live_trust_target(
        scaled_bad,
        bim,
        valid,
        gt,
        valid,
        margin=0.005,
        temperature=0.03,
    )
    assert torch.equal(mask_good, valid)
    assert torch.equal(mask_bad, valid)
    assert float(target_good.mean()) < 0.5
    assert float(target_bad.mean()) > 0.5
    assert target_good.requires_grad is False
    assert target_bad.requires_grad is False
