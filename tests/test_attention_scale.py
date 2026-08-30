from __future__ import annotations

import torch

from bim_priorda3.config import load_config
from bim_priorda3.losses import (
    BIMPriorLoss,
    absrel_optimal_log_scale,
    attention_scale_distribution_target_loss,
)
from bim_priorda3.models import BIMPriorDA3
from bim_priorda3.models.attention_scale import AttentiveBIMScaleHead
from bim_priorda3.models.full_regression_scale import FullRegressionIterativeScaleHead
from bim_priorda3.models.refiner import ScaleAnchoredDepthRefiner
from bim_priorda3.models.rgb_dinov2_full_regression_scale import (
    RGBDINOFullRegressionIterativeScaleHead,
)
from scripts.model.train import (
    ATTENTION_SCALE_WARMUP_PARAMETER_PREFIXES,
    ATTENTION_SCALE_WARMUP_STAGE,
    SCRATCH_ADDITIVE_STAGE,
    SCRATCH_JOINT_STAGE,
    SCRATCH_REFINER_STAGE,
    SCRATCH_SCALE_STAGE,
    apply_refiner_head_warmup_stage,
    apply_scratch_stage_module_modes,
    apply_scratch_training_stage,
    build_refiner_head_warmup_schedule,
    build_scratch_stage_schedule,
    build_optimizer,
    load_initial_model_weights,
    load_scale_continuation_weights,
    restore_optimizer_group_state,
    resolve_scratch_stage_epochs,
    snapshot_parameter_trainability,
)


def test_attentive_scale_aggregates_measured_ratios_and_falls_back() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        fallback_gate_bias=20.0,
    ).eval()
    features = torch.zeros(2, 5, 16, 16)
    log_ratio = torch.full((2, 1, 16, 16), torch.log(torch.tensor(2.0)))
    valid = torch.ones_like(log_ratio)
    valid[1, :, :, :] = 0
    valid[1, :, :2, :2] = 1
    fallback = torch.log(torch.tensor([1.25, 1.5])).view(2, 1, 1, 1)

    output = head(features, log_ratio, valid, fallback)

    assert torch.allclose(output["scale"][0], torch.tensor(2.0), atol=1e-5)
    assert torch.equal(output["scale"][1], torch.tensor([[[1.5]]]))
    assert output["fallback_gate"][0].item() > 0.999
    assert output["fallback_gate"][1].item() == 0.0
    assert output["pixel_support"].tolist() == [256, 4]
    assert output["attention_token_distribution"].shape == (2, 2, 2, 2)
    assert torch.allclose(
        output["attention_token_distribution"].flatten(2).sum(dim=-1),
        torch.ones(2, 2),
    )
    assert output["attention_token_valid"].shape == (2, 1, 2, 2)


def test_three_round_iterative_attention_starts_at_one_and_shares_reliability() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        fallback_gate_bias=20.0,
        iterative_updates=3,
        iterative_hidden_channels=5,
        iterative_initial_log_scale=0.0,
        iterative_damping=[0.5, 0.5, 0.5],
        iterative_max_log_update=0.15,
    )
    features = torch.randn(2, 5, 16, 16)
    horizontal_ratio = torch.linspace(1.1, 1.3, 16).view(1, 1, 1, 16)
    log_ratio = horizontal_ratio.log().expand(2, 1, 16, 16).clone()
    valid = torch.ones_like(log_ratio)
    valid[1] = 0
    deterministic_fallback = torch.log(torch.tensor([1.5, 1.5])).view(2, 1, 1, 1)
    output = head(features, log_ratio, valid, deterministic_fallback)

    assert output["iteration_log_scales"].shape == (2, 3, 1, 1)
    assert output["iteration_head_log_scales"].shape == (2, 3, 2)
    assert output["iteration_fallback_gates"].shape == (2, 3, 1, 1)
    assert torch.allclose(output["iteration_step_sizes"], torch.full((3,), 0.5))
    assert torch.equal(output["fallback_log_scale"], torch.zeros_like(deterministic_fallback))
    assert torch.equal(
        output["deterministic_fallback_log_scale"],
        deterministic_fallback,
    )
    assert torch.equal(output["scale"][1], torch.ones_like(output["scale"][1]))
    # A single module is recurrently reused; there are no round-specific MLPs.
    assert head.iterative_reliability is not None
    assert not any("round" in name for name, _ in head.named_parameters())

    objective = output["iteration_log_scales"][0, -1].sum()
    objective.backward()
    reliability_output = head.iterative_reliability[-1]
    assert isinstance(reliability_output, torch.nn.Conv2d)
    assert reliability_output.weight.grad is not None
    assert torch.count_nonzero(reliability_output.weight.grad) > 0


def test_full_regression_predicts_shared_bounded_residual_updates_without_damping() -> None:
    head = FullRegressionIterativeScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        iterative_updates=3,
        iterative_hidden_channels=5,
        delta_hidden_channels=7,
        iterative_max_log_update=0.15,
    ).eval()
    features = torch.randn(2, 5, 16, 16)
    log_ratio = torch.full((2, 1, 16, 16), torch.log(torch.tensor(2.0)))
    valid = torch.ones_like(log_ratio)
    valid[1] = 0

    # Saturating the shared neural output makes every direct residual update
    # exactly +Delta_max. No alpha/step-size multiplier may attenuate it.
    final = head.shared_delta_head[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(100.0)
    calls = 0

    def count_calls(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    hook = head.shared_delta_head.register_forward_hook(count_calls)
    output = head(features, log_ratio, valid)
    hook.remove()

    expected_centers = torch.tensor([0.15, 0.30, 0.45])
    assert calls == 3
    assert torch.allclose(
        output["iteration_log_scales"][0, :, 0, 0],
        expected_centers,
        atol=1e-6,
    )
    assert torch.allclose(
        output["iteration_log_scale_updates"][0, :, 0, 0],
        torch.full((3,), 0.15),
        atol=1e-6,
    )
    assert torch.allclose(
        output["scale"][0],
        torch.exp(torch.tensor(0.45)).view(1, 1, 1),
        atol=1e-6,
    )
    # Hard no-support fallback is raw DA3, never deterministic BIM direct.
    assert torch.equal(output["scale"][1], torch.ones_like(output["scale"][1]))
    assert "fallback_log_scale" not in output
    assert "deterministic_fallback_log_scale" not in output
    assert "fallback_gate" not in output
    assert "iteration_fallback_gates" not in output
    parameter_names = {name for name, _ in head.named_parameters()}
    assert not any(
        "step" in name or "damping" in name or "huber" in name
        for name in parameter_names
    )
    assert not hasattr(head, "huber_delta")


def test_rgb_dinov2_full_regression_has_separate_shared_bounded_updater() -> None:
    head = RGBDINOFullRegressionIterativeScaleHead(
        geometry_channels=7,
        rgb_base_channels=4,
        fusion_channels=8,
        dinov2_channels=6,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        iterative_updates=3,
        iterative_hidden_channels=5,
        delta_hidden_channels=7,
        iterative_max_log_update=0.15,
    ).eval()
    rgb = torch.randn(2, 3, 32, 32)
    geometry = torch.randn(2, 7, 32, 32)
    log_ratio = torch.full((2, 1, 32, 32), torch.log(torch.tensor(2.0)))
    valid = torch.ones_like(log_ratio)
    valid[1] = 0
    dino = torch.randn(2, 6, 3, 3)
    final = head.shared_delta_head[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.bias.fill_(100.0)

    output = head(rgb, geometry, log_ratio, valid, dino)

    assert torch.allclose(
        output["iteration_log_scales"][0, :, 0, 0],
        torch.tensor([0.15, 0.30, 0.45]),
        atol=1e-6,
    )
    assert torch.equal(output["scale"][1], torch.ones_like(output["scale"][1]))
    assert not any(
        term in name
        for name, _ in head.named_parameters()
        for term in ("huber", "damping", "gate", "da3_feature", "backbone")
    )
    assert not any("round" in name for name, _ in head.named_parameters())


def test_rgb_dinov2_scale_path_ignores_confidence_and_da3_latent_features() -> None:
    cfg = load_config(
        "configs/stanford_area1_full_regression_rgb_dinov2_scale_3round_3epoch_full_depth_metric_da3.yaml"
    )
    cfg.model.base_channels = 4
    cfg.model.attention_scale.rgb_base_channels = 4
    cfg.model.attention_scale.fusion_channels = 8
    cfg.model.attention_scale.iterative_hidden_channels = 5
    cfg.model.attention_scale.delta_hidden_channels = 7
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    cfg.model.dinov2_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg).eval()
    assert isinstance(model.attention_scale, RGBDINOFullRegressionIterativeScaleHead)
    assert model.da3_feature_fusion_enabled
    assert not model.da3_feature_scale_enabled
    assert model.da3_feature_refiner_enabled

    batch = _candidate_batch(size=32)
    batch["dinov2_feature"] = torch.randn(2, 6, 3, 3)
    reference = model._estimate_attention_scale(batch, batch["base_depth"])
    changed = dict(batch)
    changed["base_confidence"] = torch.rand_like(batch["base_confidence"]) * 100
    changed["da3_feature_mid"] = torch.randn(2, 13, 4, 4)
    changed["da3_feature_deep"] = torch.randn(2, 17, 2, 2)
    changed.pop("scaled_depth")
    changed.pop("anchor_depth")
    changed.pop("base_confidence")
    candidate = model._estimate_attention_scale(changed, changed["base_depth"])
    assert torch.equal(
        reference["iteration_log_scales"],
        candidate["iteration_log_scales"],
    )

    model.train()
    model.zero_grad(set_to_none=True)
    output = model._estimate_attention_scale(changed, changed["base_depth"])
    output["log_scale"].sum().backward()
    assert model.attention_scale is not None
    rgb_grads = [
        parameter.grad
        for parameter in model.attention_scale.rgb_encoder.parameters()
        if parameter.requires_grad
    ]
    assert rgb_grads and all(gradient is not None for gradient in rgb_grads)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in rgb_grads if gradient is not None)
    dino_projection_grads = [
        parameter.grad
        for parameter in model.attention_scale.dinov2_projection.parameters()
        if parameter.requires_grad
    ]
    assert dino_projection_grads and all(
        gradient is not None for gradient in dino_projection_grads
    )
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any("dinov2_backbone" in name for name in parameter_names)
    assert cfg.train.epochs == 3
    assert cfg.train.scale_only_experiment is True


def test_full_regression_config_is_three_epoch_scale_only_and_gt_supervised() -> None:
    cfg = load_config(
        "configs/stanford_area1_full_regression_scale_3round_3epoch_full_depth_metric_da3.yaml"
    )
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.iterative_hidden_channels = 5
    cfg.model.attention_scale.delta_hidden_channels = 7
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    cfg.model.da3_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg)

    assert isinstance(model.attention_scale, FullRegressionIterativeScaleHead)
    assert BIMPriorLoss(cfg).disable_degradation_anchor_access
    original = snapshot_parameter_trainability(model)
    stages = resolve_scratch_stage_epochs(
        cfg,
        e2e_enabled=model.e2e_da3_enabled,
        attention_scale_enabled=model.attention_scale_enabled,
    )
    assert stages == {"scale_only": 3}
    schedule = build_scratch_stage_schedule(model, stages, original)
    assert schedule["kind"] == "scratch_scale_only"
    assert len(schedule["phases"]) == 1
    stage = apply_scratch_training_stage(
        model,
        epoch=0,
        schedule=schedule,
        original_trainability=original,
    )
    assert stage["name"] == SCRATCH_SCALE_STAGE
    assert all(
        name.startswith("attention_scale.")
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    model.train()
    apply_scratch_stage_module_modes(model, SCRATCH_SCALE_STAGE)
    assert model.attention_scale.training
    assert not model.refiner.training

    batch = _candidate_batch(size=32)
    batch["da3_feature_mid"] = torch.randn(2, 6, 5, 5)
    batch["da3_feature_deep"] = torch.randn(2, 6, 5, 5)
    raw_only_batch = dict(batch)
    raw_only_batch.pop("scaled_depth")
    raw_only_batch.pop("anchor_depth")
    output = model(raw_only_batch)
    losses = BIMPriorLoss(cfg)(output, raw_only_batch)

    assert output["attention_iteration_log_scales"].shape == (2, 3, 1, 1)
    assert output["attention_iteration_log_scale_updates"].shape == (2, 3, 1, 1)
    assert "attention_iteration_step_sizes" not in output
    assert "attention_fallback_log_scale" not in output
    assert "attention_deterministic_fallback_log_scale" not in output
    assert "attention_fallback_gate" not in output
    assert "attention_iteration_fallback_gates" not in output
    assert losses["attention_scale_oracle"].item() >= 0
    for iteration in range(1, 4):
        assert f"attention_iteration_{iteration}_log_scale_mae" in losses
        assert f"attention_iteration_{iteration}_overshoot_rate" in losses
        assert f"attention_iteration_{iteration}_oscillation_rate" in losses
    losses["total"].backward()
    assert model.attention_scale.shared_delta_head[-1].weight.grad is not None

    # Cached deterministic robust scale and BIM-direct anchor are not scale
    # estimator inputs. Arbitrarily corrupting both must leave c1/c2/c3 exact.
    model.eval()
    with torch.no_grad():
        reference = model(raw_only_batch)["attention_iteration_log_scales"]
        restored_batch = dict(raw_only_batch)
        restored_batch["scaled_depth"] = batch["scaled_depth"] * 19.0
        restored_batch["anchor_depth"] = batch["anchor_depth"] * 23.0
        with_legacy_fields = model(restored_batch)["attention_iteration_log_scales"]
    assert torch.equal(reference, with_legacy_fields)


def test_full_regression_no_da3_feature_ablation_changes_only_scale_feature_fusion() -> None:
    baseline_cfg = load_config(
        "configs/stanford_area1_full_regression_scale_3round_3epoch_full_depth_metric_da3.yaml"
    )
    ablation_cfg = load_config(
        "configs/stanford_area1_full_regression_scale_no_da3_features_3round_3epoch_full_depth_metric_da3.yaml"
    )
    baseline_cfg.model.base_channels = 4
    ablation_cfg.model.base_channels = 4
    baseline_cfg.model.attention_scale.hidden_channels = 4
    ablation_cfg.model.attention_scale.hidden_channels = 4
    baseline_cfg.model.attention_scale.iterative_hidden_channels = 5
    ablation_cfg.model.attention_scale.iterative_hidden_channels = 5
    baseline_cfg.model.attention_scale.delta_hidden_channels = 7
    ablation_cfg.model.attention_scale.delta_hidden_channels = 7
    baseline_cfg.model.da3_feature_fusion.channels = 6
    ablation_cfg.model.da3_feature_fusion.channels = 6

    baseline = BIMPriorDA3(baseline_cfg)
    ablation = BIMPriorDA3(ablation_cfg)
    assert isinstance(baseline.attention_scale, FullRegressionIterativeScaleHead)
    assert isinstance(ablation.attention_scale, FullRegressionIterativeScaleHead)
    assert baseline.da3_feature_scale_enabled
    assert not ablation.da3_feature_scale_enabled
    assert baseline.da3_feature_refiner_enabled
    assert ablation.da3_feature_refiner_enabled
    assert baseline.attention_scale.da3_feature_channels == 6
    assert ablation.attention_scale.da3_feature_channels == 0
    assert baseline.attention_scale.encoder[0][0].in_channels == 13
    assert ablation.attention_scale.encoder[0][0].in_channels == 13
    assert baseline_cfg.model.attention_scale == ablation_cfg.model.attention_scale
    assert baseline_cfg.loss == ablation_cfg.loss
    assert baseline_cfg.train == ablation_cfg.train


def test_full_regression_no_confidence_ablation_changes_only_one_input_channel() -> None:
    reference_cfg = load_config(
        "configs/stanford_area1_full_regression_scale_no_da3_features_3round_3epoch_full_depth_metric_da3.yaml"
    )
    ablation_cfg = load_config(
        "configs/stanford_area1_full_regression_scale_no_da3_features_no_confidence_3round_3epoch_full_depth_metric_da3.yaml"
    )
    reference_cfg.model.base_channels = 4
    ablation_cfg.model.base_channels = 4
    reference_cfg.model.attention_scale.hidden_channels = 4
    ablation_cfg.model.attention_scale.hidden_channels = 4
    reference_cfg.model.attention_scale.iterative_hidden_channels = 5
    ablation_cfg.model.attention_scale.iterative_hidden_channels = 5
    reference_cfg.model.attention_scale.delta_hidden_channels = 7
    ablation_cfg.model.attention_scale.delta_hidden_channels = 7
    reference_cfg.model.da3_feature_fusion.channels = 6
    ablation_cfg.model.da3_feature_fusion.channels = 6

    reference = BIMPriorDA3(reference_cfg)
    ablation = BIMPriorDA3(ablation_cfg)
    assert isinstance(reference.attention_scale, FullRegressionIterativeScaleHead)
    assert isinstance(ablation.attention_scale, FullRegressionIterativeScaleHead)
    assert not reference.da3_feature_scale_enabled
    assert not ablation.da3_feature_scale_enabled
    assert reference.attention_scale_use_base_confidence
    assert not ablation.attention_scale_use_base_confidence
    assert reference.attention_scale.encoder[0][0].in_channels == 13
    assert ablation.attention_scale.encoder[0][0].in_channels == 12
    assert reference_cfg.loss == ablation_cfg.loss
    assert reference_cfg.train == ablation_cfg.train

    batch = _candidate_batch(size=32)
    without_confidence = dict(batch)
    without_confidence.pop("base_confidence")
    features_without, ratio_without, valid_without = (
        ablation._full_regression_scale_inputs(
            without_confidence,
            without_confidence["base_depth"],
        )
    )
    changed_confidence = dict(batch)
    changed_confidence["base_confidence"] = torch.rand_like(
        batch["base_confidence"]
    )
    features_changed, ratio_changed, valid_changed = (
        ablation._full_regression_scale_inputs(
            changed_confidence,
            changed_confidence["base_depth"],
        )
    )
    assert features_without.shape[1] == 12
    assert torch.equal(features_without, features_changed)
    assert torch.equal(ratio_without, ratio_changed)
    assert torch.equal(valid_without, valid_changed)


def test_full_regression_rgb_da3_unit_bim_has_no_bim_input_dependency() -> None:
    cfg = load_config(
        "configs/stanford_area1_full_regression_rgb_da3_unit_bim_3round_3epoch_full_depth_metric_da3.yaml"
    )
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.iterative_hidden_channels = 5
    cfg.model.attention_scale.delta_hidden_channels = 7
    model = BIMPriorDA3(cfg).eval()
    assert isinstance(model.attention_scale, FullRegressionIterativeScaleHead)
    assert model.full_regression_input_mode == model.FULL_REGRESSION_INPUT_RGB_DA3_UNIT_BIM
    assert model.attention_scale.encoder[0][0].in_channels == 4
    assert not model.attention_scale_use_base_confidence
    assert not model.da3_feature_fusion_enabled
    assert not model.use_bim_condition
    assert not model.use_frame_residual
    assert not model.use_low_residual
    assert model.refiner.max_detail_log_residual == 0.0

    batch = _candidate_batch(size=32)
    minimal_batch = {
        "rgb": batch["rgb"],
        "base_depth": batch["base_depth"],
    }
    features, log_ratio, ratio_valid = model._full_regression_scale_inputs(
        minimal_batch,
        minimal_batch["base_depth"],
    )
    expected_ratio = 1.0 / minimal_batch["base_depth"]
    expected_valid = (
        torch.isfinite(expected_ratio)
        & (expected_ratio > model.da3_scale_ratio_min)
        & (expected_ratio < model.da3_scale_ratio_max)
    )
    assert features.shape[1] == 4
    assert torch.equal(ratio_valid.bool(), expected_valid)
    assert torch.allclose(
        log_ratio[expected_valid],
        expected_ratio[expected_valid].log(),
    )

    reference = model._estimate_attention_scale(minimal_batch, batch["base_depth"])
    corrupted = dict(batch)
    corrupted["bim_depth"] = torch.rand_like(batch["bim_depth"]) * 1000.0
    corrupted["bim_valid"] = torch.zeros_like(batch["bim_valid"])
    corrupted["bim_normals"] = torch.randn_like(batch["bim_normals"])
    corrupted["bim_edge"] = torch.rand_like(batch["bim_edge"])
    corrupted["base_confidence"] = torch.rand_like(batch["base_confidence"])
    candidate = model._estimate_attention_scale(corrupted, corrupted["base_depth"])
    assert torch.equal(
        reference["iteration_log_scales"],
        candidate["iteration_log_scales"],
    )


def test_full_regression_no_bim_geometry_ablation_removes_only_four_channels() -> None:
    reference_cfg = load_config(
        "configs/stanford_area1_full_regression_scale_no_da3_features_no_confidence_3round_3epoch_full_depth_metric_da3.yaml"
    )
    ablation_cfg = load_config(
        "configs/stanford_area1_full_regression_scale_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml"
    )
    for cfg in (reference_cfg, ablation_cfg):
        cfg.model.base_channels = 4
        cfg.model.attention_scale.hidden_channels = 4
        cfg.model.attention_scale.iterative_hidden_channels = 5
        cfg.model.attention_scale.delta_hidden_channels = 7
        cfg.model.da3_feature_fusion.channels = 6

    reference = BIMPriorDA3(reference_cfg).eval()
    ablation = BIMPriorDA3(ablation_cfg).eval()
    assert isinstance(reference.attention_scale, FullRegressionIterativeScaleHead)
    assert isinstance(ablation.attention_scale, FullRegressionIterativeScaleHead)
    assert reference.attention_scale.encoder[0][0].in_channels == 12
    assert ablation.attention_scale.encoder[0][0].in_channels == 8
    assert reference.attention_scale_use_bim_normals
    assert reference.attention_scale_use_bim_edge
    assert not ablation.attention_scale_use_bim_normals
    assert not ablation.attention_scale_use_bim_edge
    assert not ablation.attention_scale_use_base_confidence
    assert not ablation.da3_feature_scale_enabled
    assert reference_cfg.loss == ablation_cfg.loss
    assert reference_cfg.train == ablation_cfg.train

    batch = _candidate_batch(size=32)
    without_geometry = dict(batch)
    without_geometry.pop("bim_normals")
    without_geometry.pop("bim_edge")
    features_without, ratio_without, valid_without = (
        ablation._full_regression_scale_inputs(
            without_geometry,
            without_geometry["base_depth"],
        )
    )
    corrupted = dict(batch)
    corrupted["bim_normals"] = torch.randn_like(batch["bim_normals"]) * 100.0
    corrupted["bim_edge"] = torch.rand_like(batch["bim_edge"])
    features_corrupted, ratio_corrupted, valid_corrupted = (
        ablation._full_regression_scale_inputs(
            corrupted,
            corrupted["base_depth"],
        )
    )
    assert features_without.shape[1] == 8
    assert torch.equal(features_without, features_corrupted)
    assert torch.equal(ratio_without, ratio_corrupted)
    assert torch.equal(valid_without, valid_corrupted)


def test_fixed_and_iterative_huber_controls_share_the_eight_channel_contract() -> None:
    config_paths = (
        "configs/stanford_area1_fixed_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml",
        "configs/stanford_area1_iterative_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml",
    )
    models = []
    for path in config_paths:
        cfg = load_config(path)
        cfg.model.base_channels = 4
        cfg.model.attention_scale.hidden_channels = 4
        cfg.model.attention_scale.iterative_hidden_channels = 5
        cfg.model.da3_feature_fusion.channels = 6
        model = BIMPriorDA3(cfg).eval()
        assert isinstance(model.attention_scale, AttentiveBIMScaleHead)
        assert model.attention_scale.encoder[0][0].in_channels == 8
        assert not model.da3_feature_scale_enabled
        assert not model.attention_scale_use_base_confidence
        assert not model.attention_scale_use_bim_normals
        assert not model.attention_scale_use_bim_edge
        assert not model.attention_scale_use_deterministic_fallback_input
        assert model.attention_scale.iterative_updates == 3
        assert model.attention_scale.fallback_gate is None
        models.append(model)

    assert not models[0].attention_scale.iterative_refresh_attention
    assert models[1].attention_scale.iterative_refresh_attention

    batch = _candidate_batch(size=32)
    minimal = dict(batch)
    minimal.pop("base_confidence")
    minimal.pop("bim_normals")
    minimal.pop("bim_edge")
    for model in models:
        reference_inputs = model._attention_scale_inputs(
            minimal,
            minimal["base_depth"],
            minimal["scaled_depth"],
        )
        corrupted = dict(batch)
        corrupted["base_confidence"] = torch.rand_like(batch["base_confidence"])
        corrupted["bim_normals"] = torch.randn_like(batch["bim_normals"]) * 100.0
        corrupted["bim_edge"] = torch.rand_like(batch["bim_edge"])
        corrupted["scaled_depth"] = batch["scaled_depth"] * 19.0
        candidate_inputs = model._attention_scale_inputs(
            corrupted,
            corrupted["base_depth"],
            corrupted["scaled_depth"],
        )
        assert reference_inputs[0].shape[1] == 8
        for reference, candidate in zip(reference_inputs, candidate_inputs):
            assert torch.equal(reference, candidate)

        with torch.no_grad():
            reference_output = model._estimate_attention_scale(
                minimal,
                minimal["base_depth"],
            )
            candidate_output = model._estimate_attention_scale(
                corrupted,
                corrupted["base_depth"],
            )
        assert torch.equal(
            reference_output["iteration_log_scales"],
            candidate_output["iteration_log_scales"],
        )


def test_static_zero_control_freezes_attention_and_disables_fallback_gate() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        iterative_updates=3,
        iterative_hidden_channels=5,
        iterative_initial_log_scale=0.0,
        iterative_damping=[0.5, 0.5, 0.5],
        iterative_refresh_attention=False,
        use_fallback_gate=False,
    ).eval()
    features = torch.randn(2, 5, 16, 16)
    log_ratio = torch.linspace(-0.3, 0.3, 16).view(1, 1, 1, 16)
    log_ratio = log_ratio.expand(2, 1, 16, 16).clone()
    valid = torch.ones_like(log_ratio)
    valid[1] = 0
    deterministic_fallback = torch.log(torch.tensor([1.5, 1.5])).view(2, 1, 1, 1)

    assert head.iterative_reliability is not None
    calls = 0

    def count_calls(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    hook = head.iterative_reliability.register_forward_hook(count_calls)
    output = head(features, log_ratio, valid, deterministic_fallback)
    hook.remove()

    assert calls == 1
    assert head.fallback_gate is None
    assert torch.equal(
        output["iteration_fallback_gates"],
        torch.ones_like(output["iteration_fallback_gates"]),
    )
    # Unsupported samples remain at z=0 without consulting deterministic BIM.
    assert torch.equal(output["scale"][1], torch.ones_like(output["scale"][1]))


def test_refresh_control_recomputes_attention_every_round() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        iterative_updates=3,
        iterative_hidden_channels=5,
        iterative_initial_log_scale=0.0,
        iterative_damping=[0.5, 0.5, 0.5],
        iterative_refresh_attention=True,
        use_fallback_gate=False,
    ).eval()
    features = torch.randn(1, 5, 16, 16)
    log_ratio = torch.linspace(-0.3, 0.3, 16).view(1, 1, 1, 16)
    log_ratio = log_ratio.expand(1, 1, 16, 16)
    valid = torch.ones_like(log_ratio)
    fallback = torch.log(torch.tensor([1.5])).view(1, 1, 1, 1)

    assert head.iterative_reliability is not None
    calls = 0

    def count_calls(_module: torch.nn.Module, _inputs: object, _output: object) -> None:
        nonlocal calls
        calls += 1

    hook = head.iterative_reliability.register_forward_hook(count_calls)
    head(features, log_ratio, valid, fallback)
    hook.remove()

    assert calls == 3


def test_matched_static_and_dynamic_configs_only_change_attention_refresh() -> None:
    static_cfg = load_config(
        "configs/stanford_area1_static_scale_3round_zero_nogate_full_depth_metric_da3.yaml"
    )
    dynamic_cfg = load_config(
        "configs/stanford_area1_dynamic_scale_3round_zero_nogate_full_depth_metric_da3.yaml"
    )

    assert static_cfg.model.attention_scale.iterative_refresh_attention is False
    assert dynamic_cfg.model.attention_scale.iterative_refresh_attention is True
    assert static_cfg.model.attention_scale.use_fallback_gate is False
    assert dynamic_cfg.model.attention_scale.use_fallback_gate is False
    assert static_cfg.model.attention_scale.iterative_updates == 3
    assert dynamic_cfg.model.attention_scale.iterative_updates == 3
    assert static_cfg.train.scratch_stage_epochs == dynamic_cfg.train.scratch_stage_epochs


def test_three_round_config_applies_iteration_weighted_oracle_loss() -> None:
    cfg = load_config(
        "configs/stanford_area1_iterative_scale_3round_full_depth_metric_da3.yaml"
    )
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.iterative_hidden_channels = 5
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    cfg.model.da3_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg)
    batch = _candidate_batch(size=32)
    batch["da3_feature_mid"] = torch.randn(2, 6, 5, 5)
    batch["da3_feature_deep"] = torch.randn(2, 6, 5, 5)

    output = model(batch)
    criterion = BIMPriorLoss(cfg)
    losses = criterion(output, batch)

    assert output["attention_iteration_log_scales"].shape == (2, 3, 1, 1)
    assert losses["attention_scale_oracle"].item() >= 0
    losses["total"].backward()
    assert model.attention_scale is not None
    assert model.attention_scale.iterative_step_logits is not None
    assert model.attention_scale.iterative_step_logits.grad is not None


def test_direct_attention_target_supervises_spatial_scale_weights() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=4,
        token_dropout_probability=0.0,
        fallback_gate_bias=20.0,
    )
    features = torch.randn(1, 5, 32, 32)
    log_ratio = torch.full((1, 1, 32, 32), torch.log(torch.tensor(1.25)))
    log_ratio[..., :16, :16] = torch.log(torch.tensor(2.0))
    valid = torch.ones_like(log_ratio)
    fallback = torch.log(torch.tensor([1.5])).view(1, 1, 1, 1)
    output = head(features, log_ratio, valid, fallback)
    output["base_depth"] = torch.ones_like(log_ratio)
    batch = {
        "bim_depth": log_ratio.exp(),
        "bim_valid": valid,
        "gt_valid": valid,
    }
    oracle = torch.log(torch.tensor([2.0])).view(1, 1, 1, 1)
    loss = attention_scale_distribution_target_loss(
        output,
        batch,
        oracle,
        torch.tensor([True]),
        temperature=0.05,
        ratio_min=0.2,
        ratio_max=5.0,
    )

    assert loss.item() > 0
    loss.backward()
    assert head.key_logits.weight.grad is not None
    assert torch.count_nonzero(head.key_logits.weight.grad) > 0


def test_bounded_scale_residual_is_zero_initialized_and_bounded() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=8,
        token_dropout_probability=0.0,
        fallback_gate_bias=20.0,
        bounded_log_scale_residual=0.05,
        residual_hidden_channels=3,
    ).eval()
    features = torch.randn(2, 5, 16, 16)
    log_ratio = torch.full((2, 1, 16, 16), torch.log(torch.tensor(2.0)))
    valid = torch.ones_like(log_ratio)
    fallback = torch.log(torch.tensor([1.25, 1.5])).view(2, 1, 1, 1)

    initial = head(features, log_ratio, valid, fallback)

    assert torch.count_nonzero(initial["bounded_log_scale_residual"]) == 0
    assert torch.equal(initial["attentive_log_scale"], initial["raw_attentive_log_scale"])
    assert head.scale_residual_mlp is not None
    final = head.scale_residual_mlp[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.bias.fill_(100.0)
    saturated = head(features, log_ratio, valid, fallback)
    assert torch.all(saturated["bounded_log_scale_residual"] <= 0.05)
    assert torch.all(saturated["bounded_log_scale_residual"] >= -0.05)
    assert torch.allclose(
        saturated["attentive_log_scale"] - saturated["raw_attentive_log_scale"],
        torch.full((2, 1, 1, 1), 0.05),
        atol=1e-6,
    )


def test_da3_features_are_native_attention_and_low_resolution_inputs() -> None:
    head = AttentiveBIMScaleHead(
        in_channels=5,
        hidden_channels=4,
        attention_heads=2,
        min_support=4,
        token_dropout_probability=0.0,
        da3_feature_channels=6,
    )
    features = torch.randn(2, 5, 32, 32)
    log_ratio = torch.linspace(-0.2, 0.2, 32).view(1, 1, 1, 32).expand(2, 1, 32, 32)
    valid = torch.ones_like(log_ratio)
    fallback = torch.zeros(2, 1, 1, 1)
    mid = torch.randn(2, 6, 5, 5, requires_grad=True)
    deep = torch.randn(2, 6, 5, 5, requires_grad=True)

    output = head(
        features,
        log_ratio,
        valid,
        fallback,
        da3_feature_mid=mid,
        da3_feature_deep=deep,
    )
    objective = output["head_log_scale"][:, 0].sum() + output["head_mixture"][:, 0].sum()
    objective.backward()

    assert output["scale"].shape == (2, 1, 1, 1)
    assert mid.grad is not None and torch.count_nonzero(mid.grad) > 0
    assert deep.grad is not None and torch.count_nonzero(deep.grad) > 0

    refiner = ScaleAnchoredDepthRefiner(
        rgb_channels=3,
        geometry_channels=4,
        bim_channels=8,
        base_channels=4,
        max_frame_log_residual=0.2,
        max_low_log_residual=0.25,
        max_detail_log_residual=0.15,
        max_total_log_residual=0.45,
        da3_feature_channels=6,
    )
    with torch.no_grad():
        refiner.low_output.weight.fill_(0.01)
        refiner.detail_output.weight[0].fill_(0.01)
    mid_refiner = torch.randn(2, 6, 5, 5, requires_grad=True)
    deep_refiner = torch.randn(2, 6, 5, 5, requires_grad=True)
    prediction = refiner(
        torch.randn(2, 3, 32, 32),
        torch.randn(2, 4, 32, 32),
        torch.randn(2, 8, 32, 32),
        da3_feature_mid=mid_refiner,
        da3_feature_deep=deep_refiner,
    )
    (
        prediction["low_log_residual"].sum()
        + prediction["detail_log_residual"].sum()
    ).backward()
    assert mid_refiner.grad is not None and torch.count_nonzero(mid_refiner.grad) > 0
    assert deep_refiner.grad is not None and torch.count_nonzero(deep_refiner.grad) > 0


def test_da3_feature_candidate_runs_end_to_end_without_live_da3() -> None:
    cfg = load_config("configs/stanford_area1_attentive_scale_da3_features.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    cfg.model.da3_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg)
    batch = _candidate_batch(size=32)
    batch["da3_feature_mid"] = torch.randn(2, 6, 5, 5)
    batch["da3_feature_deep"] = torch.randn(2, 6, 5, 5)

    output = model(batch)

    assert model.da3 is None
    assert output["depth"].shape == batch["base_depth"].shape
    assert output["attention_scale"].shape == (2, 1, 1, 1)


def test_reliability_gated_candidate_uses_live_scale_target_and_gates_detail() -> None:
    cfg = load_config("configs/stanford_area1_reliability_gated_full_depth.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    cfg.model.da3_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg)
    batch = _candidate_batch(size=32)
    batch["da3_feature_mid"] = torch.randn(2, 6, 5, 5)
    batch["da3_feature_deep"] = torch.randn(2, 6, 5, 5)
    with torch.no_grad():
        model.refiner.detail_output.bias[0] = 1.0
        model.refiner.detail_output.bias[2] = -100.0

    output = model(batch)

    assert model.refiner.bim_adapter_gate_use_rgb
    assert torch.allclose(
        output["detail_reliability_gate"],
        torch.full_like(output["detail_reliability_gate"], 0.10),
        atol=1e-6,
    )
    assert torch.allclose(
        output["detail_log_residual"],
        0.10 * output["raw_detail_log_residual"],
        atol=1e-6,
    )
    criterion = BIMPriorLoss(cfg)
    criterion.set_epoch(0)
    assert criterion.attention_entropy_factor() == 1.0
    losses = criterion(output, batch)
    assert losses["attention_weight_target"].item() >= 0
    criterion.set_epoch(3)
    assert criterion.attention_entropy_factor() == 0.0


def test_hybrid_additive_uses_da3_features_only_in_refiner_and_starts_as_noop() -> None:
    cfg = load_config("configs/stanford_area1_hybrid_additive.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    cfg.model.da3_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg)
    batch = _candidate_batch(size=32)
    batch["da3_feature_mid"] = torch.randn(2, 6, 5, 5)
    batch["da3_feature_deep"] = torch.randn(2, 6, 5, 5)

    output = model(batch)

    assert model.attention_scale is not None
    assert model.attention_scale.da3_feature_channels == 0
    assert model.refiner.da3_feature_channels == 6
    assert model.additive_residual_enabled
    assert torch.count_nonzero(output["additive_metric_residual"]) == 0
    assert torch.equal(output["depth"], output["proportional_depth"])
    assert model.refiner.additive_output is not None
    losses = BIMPriorLoss(cfg)(output, batch)
    assert losses["additive_residual_teacher"].item() > 0
    losses["total"].backward()
    assert model.refiner.additive_output.bias.grad is not None
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.refiner.additive_output.bias.fill_(100.0)
    shifted = model(batch)
    assert torch.allclose(
        shifted["additive_metric_residual"],
        torch.full_like(shifted["additive_metric_residual"], 0.20),
    )
    assert torch.allclose(
        shifted["depth"],
        shifted["proportional_depth"] + 0.20,
    )


def test_additive_head_detaches_shared_refiner_features() -> None:
    refiner = ScaleAnchoredDepthRefiner(
        rgb_channels=3,
        geometry_channels=4,
        bim_channels=8,
        base_channels=4,
        max_frame_log_residual=0.2,
        max_low_log_residual=0.25,
        max_detail_log_residual=0.15,
        max_total_log_residual=0.45,
        additive_residual_enabled=True,
        max_additive_residual_m=0.2,
        additive_detach_shared_features=True,
    )
    decoded = torch.randn(2, 4, 16, 16, requires_grad=True)
    anchor = torch.ones(2, 1, 16, 16, requires_grad=True)
    proportional = torch.zeros(2, 1, 16, 16, requires_grad=True)
    residual = refiner.predict_additive_metric_residual(decoded, anchor, proportional)
    residual.sum().backward()

    assert decoded.grad is None
    assert anchor.grad is None
    assert proportional.grad is None
    assert refiner.additive_output is not None
    assert refiner.additive_output.bias.grad is not None


def _candidate_batch(batch_size: int = 2, size: int = 32) -> dict[str, torch.Tensor]:
    base = torch.ones(batch_size, 1, size, size)
    bim = torch.ones_like(base) * 1.3
    bim[..., :, size // 2 :] = 2.2
    deterministic_scaled = base * 1.55
    return {
        "rgb": torch.rand(batch_size, 3, size, size),
        "base_depth": base,
        "base_confidence": torch.ones_like(base),
        "scaled_depth": deterministic_scaled,
        "anchor_depth": deterministic_scaled * 1.01,
        "bim_depth": bim,
        "bim_valid": torch.ones_like(base),
        "bim_normals": torch.zeros(batch_size, 3, size, size),
        "bim_edge": torch.zeros_like(base),
        "gt_depth": base * 1.75,
        "gt_valid": torch.ones_like(base),
        "gt_weight": torch.ones_like(base),
        "furniture_mask": torch.zeros_like(base),
        "trust_target": torch.ones_like(base),
        "trust_mask": torch.ones_like(base),
    }


def test_attentive_scale_candidate_is_scalar_anchored_and_gt_supervised() -> None:
    cfg = load_config("configs/stanford_area1_attentive_scale.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    model = BIMPriorDA3(cfg)
    batch = _candidate_batch()

    output = model(batch)

    assert output["attention_scale"].shape == (2, 1, 1, 1)
    assert output["attention_scale_map"].shape == batch["base_depth"].shape
    assert torch.count_nonzero(output["frame_log_residual"]) == 0
    assert torch.equal(
        output["log_residual"],
        output["low_log_residual"] + output["detail_log_residual"],
    )
    assert torch.allclose(
        output["depth"],
        output["scaled_depth"] * torch.exp(output["spatial_log_residual"]),
    )

    losses = BIMPriorLoss(cfg)(output, batch)
    losses["total"].backward()
    assert losses["coarse_depth"].item() > 0
    assert model.attention_scale is not None
    assert model.attention_scale.fallback_gate.bias.grad is not None
    assert torch.count_nonzero(model.attention_scale.fallback_gate.bias.grad) > 0
    assert model.refiner.low_output.weight.grad is not None
    assert model.refiner.detail_output.weight.grad is not None
    assert model.refiner.frame_output.weight.grad is not None
    assert torch.count_nonzero(model.refiner.frame_output.weight.grad[0]) == 0


def test_absrel_optimal_log_scale_is_exact_weighted_median() -> None:
    base = torch.ones(2, 1, 1, 4)
    target = torch.tensor([[[[1.0, 2.0, 4.0, 8.0]]], [[[2.0, 2.0, 2.0, 2.0]]]])
    valid = torch.ones_like(base)
    valid[0, ..., 3] = 0

    log_scale, supported = absrel_optimal_log_scale(
        base,
        target,
        valid,
        min_support=3,
    )

    # Ratios 1,2,4 have AbsRel weights 1,1/2,1/4, so the weighted
    # median is 1.  The constant second sample has optimum scale 2.
    assert torch.allclose(log_scale.flatten(), torch.tensor([0.0, torch.log(torch.tensor(2.0))]))
    assert supported.tolist() == [True, True]

    candidates = torch.linspace(0.25, 5.0, 19001)
    for sample_index, expected in enumerate(log_scale.exp().flatten()):
        mask = valid[sample_index] > 0
        errors = (
            (
                candidates[:, None] * base[sample_index][mask].flatten()[None]
                - target[sample_index][mask].flatten()[None]
            ).abs()
            / target[sample_index][mask].flatten()[None]
        ).mean(dim=1)
        grid_optimum = candidates[errors.argmin()]
        assert torch.isclose(expected, grid_optimum, atol=3e-4)


def test_oracle_scale_loss_directly_supervises_attention_scale() -> None:
    cfg = load_config("configs/stanford_area1_attentive_scale_oracle.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.attention_scale.token_dropout_probability = 0.0
    cfg.model.attention_scale.equivariance_probability = 0.0
    model = BIMPriorDA3(cfg)
    batch = _candidate_batch()
    output = model(batch)

    losses = BIMPriorLoss(cfg)(output, batch)

    assert losses["attention_scale_oracle"].item() > 0
    model.zero_grad(set_to_none=True)
    for key in (
        "depth",
        "coarse_depth",
        "gradient",
        "residual_teacher",
        "frame_residual_teacher",
        "local_residual_teacher",
        "low_smoothness",
        "detail_regularization",
        "trust",
        "frame_trust",
        "uncertainty",
        "degradation",
        "adapter_gate",
        "spatial_mean",
        "attention_entropy",
        "attention_scale_equivariance",
    ):
        cfg.loss[key] = 0.0
    cfg.loss.near_range_boost = 0.0
    cfg.loss.furniture_multiplier = 1.0
    cfg.loss.bim_foreground_conflict_multiplier = 1.0
    oracle_only = BIMPriorLoss(cfg)
    oracle_losses = oracle_only(output, batch)
    oracle_losses["total"].backward()
    assert model.attention_scale is not None
    assert model.attention_scale.fallback_gate.bias.grad is not None
    assert torch.count_nonzero(model.attention_scale.fallback_gate.bias.grad) > 0
    assert model.refiner.low_output.weight.grad is not None
    assert torch.count_nonzero(model.refiner.low_output.weight.grad) == 0


def test_candidate_initialization_and_first_epoch_train_only_attention_scale() -> None:
    source_cfg = load_config("configs/stanford_area1.yaml")
    source_cfg.model.base_channels = 4
    source = BIMPriorDA3(source_cfg)
    target_cfg = load_config("configs/stanford_area1_attentive_scale.yaml")
    target_cfg.model.base_channels = 4
    target_cfg.model.attention_scale.hidden_channels = 4
    target = BIMPriorDA3(target_cfg)

    receipt = load_initial_model_weights(target, source.state_dict())
    expected_missing = {name for name in target.state_dict() if name.startswith("attention_scale.")}
    assert set(receipt["missing_keys"]) == expected_missing
    assert receipt["unexpected_keys"] == []

    original = snapshot_parameter_trainability(target)
    schedule = build_refiner_head_warmup_schedule(target, 1, original)
    stage = apply_refiner_head_warmup_stage(
        target,
        epoch=0,
        warmup_epochs=1,
        original_trainability=original,
    )
    trainable = {name for name, parameter in target.named_parameters() if parameter.requires_grad}
    assert schedule["parameter_prefixes"] == list(ATTENTION_SCALE_WARMUP_PARAMETER_PREFIXES)
    assert stage["name"] == ATTENTION_SCALE_WARMUP_STAGE
    assert trainable
    assert all(name.startswith("attention_scale.") for name in trainable)


def test_scratch_scale_refiner_joint_schedule_uses_fresh_task_network() -> None:
    cfg = load_config("configs/stanford_area1_attentive_scale_mlp_scratch.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    model = BIMPriorDA3(cfg)
    original = snapshot_parameter_trainability(model)
    stages = resolve_scratch_stage_epochs(
        cfg,
        e2e_enabled=model.e2e_da3_enabled,
        attention_scale_enabled=model.attention_scale_enabled,
    )
    assert stages == {"scale_only": 3, "refiner_only": 9, "joint": 3}
    assert stages is not None
    schedule = build_scratch_stage_schedule(model, stages, original)

    scale_stage = apply_scratch_training_stage(
        model,
        epoch=0,
        schedule=schedule,
        original_trainability=original,
    )
    scale_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert scale_stage["name"] == SCRATCH_SCALE_STAGE
    assert scale_names and all(name.startswith("attention_scale.") for name in scale_names)
    model.train()
    apply_scratch_stage_module_modes(model, SCRATCH_SCALE_STAGE)
    assert model.attention_scale is not None and model.attention_scale.training
    assert not model.refiner.training

    refiner_stage = apply_scratch_training_stage(
        model,
        epoch=3,
        schedule=schedule,
        original_trainability=original,
    )
    refiner_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert refiner_stage["name"] == SCRATCH_REFINER_STAGE
    assert refiner_names and not any(name.startswith("attention_scale.") for name in refiner_names)
    model.train()
    apply_scratch_stage_module_modes(model, SCRATCH_REFINER_STAGE)
    assert not model.attention_scale.training
    assert model.refiner.training

    joint_stage = apply_scratch_training_stage(
        model,
        epoch=12,
        schedule=schedule,
        original_trainability=original,
    )
    joint_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert joint_stage["name"] == SCRATCH_JOINT_STAGE
    assert joint_names == {name for name, trainable in original.items() if trainable}


def test_reduced_refiner_continuation_restores_only_trained_scale_state() -> None:
    source_cfg = load_config(
        "configs/stanford_area1_fixed_attention_huber_no_da3_features_no_confidence_no_bim_geometry_3round_3epoch_full_depth_metric_da3.yaml"
    )
    target_cfg = load_config(
        "configs/stanford_area1_fixed_attention_huber_reduced_refiner_continuation_full_depth_metric_da3.yaml"
    )
    for cfg in (source_cfg, target_cfg):
        cfg.model.base_channels = 4
        cfg.model.attention_scale.hidden_channels = 4
        cfg.model.attention_scale.iterative_hidden_channels = 5
        cfg.model.attention_scale.token_dropout_probability = 0.0
        cfg.model.attention_scale.equivariance_probability = 0.0
    source_cfg.model.da3_feature_fusion.enabled = False
    source_cfg.model.da3_feature_fusion.refiner_enabled = False
    source = BIMPriorDA3(source_cfg)
    target = BIMPriorDA3(target_cfg)

    assert target.refiner_geometry_channels == 3
    assert target.refiner_bim_channels == 4
    assert not target.da3_feature_fusion_enabled
    with torch.no_grad():
        for parameter in source.attention_scale.parameters():
            parameter.fill_(0.125)
    receipt = load_scale_continuation_weights(target, source.state_dict())
    assert receipt["scope"] == "attention_scale_only"
    assert receipt["fresh_refiner"] is True
    assert all(
        torch.equal(target.state_dict()[name], value)
        for name, value in source.state_dict().items()
        if name.startswith("attention_scale.")
    )

    stages = resolve_scratch_stage_epochs(
        target_cfg,
        e2e_enabled=target.e2e_da3_enabled,
        attention_scale_enabled=target.attention_scale_enabled,
    )
    assert stages == {"refiner_only": 9, "joint": 3}
    original = snapshot_parameter_trainability(target)
    schedule = build_scratch_stage_schedule(target, stages, original)
    assert schedule["kind"] == "scale_checkpoint_refiner_joint"
    assert apply_scratch_training_stage(
        target,
        epoch=0,
        schedule=schedule,
        original_trainability=original,
    )["name"] == SCRATCH_REFINER_STAGE
    assert apply_scratch_training_stage(
        target,
        epoch=9,
        schedule=schedule,
        original_trainability=original,
    )["name"] == SCRATCH_JOINT_STAGE

    source_optimizer, _ = build_optimizer(
        source,
        source_cfg,
        float(source_cfg.train.learning_rate),
    )
    source_optimizer.zero_grad(set_to_none=True)
    sum(parameter.sum() for parameter in source.attention_scale.parameters()).backward()
    source_optimizer.step()
    target_optimizer, _ = build_optimizer(
        target,
        target_cfg,
        float(target_cfg.train.learning_rate),
    )
    optimizer_receipt = restore_optimizer_group_state(
        target_optimizer,
        source_optimizer.state_dict(),
        group_name="attention_scale",
    )
    assert optimizer_receipt["restored_parameter_tensors"] > 0
    assert target_optimizer.param_groups[1]["lr"] == float(
        target_cfg.train.attention_scale_learning_rate
    )


def test_hybrid_scratch_schedule_has_dedicated_additive_stage() -> None:
    cfg = load_config("configs/stanford_area1_hybrid_additive.yaml")
    cfg.model.base_channels = 4
    cfg.model.attention_scale.hidden_channels = 4
    cfg.model.da3_feature_fusion.channels = 6
    model = BIMPriorDA3(cfg)
    original = snapshot_parameter_trainability(model)
    stages = resolve_scratch_stage_epochs(
        cfg,
        e2e_enabled=model.e2e_da3_enabled,
        attention_scale_enabled=model.attention_scale_enabled,
        additive_residual_enabled=model.additive_residual_enabled,
    )
    assert stages == {
        "scale_only": 3,
        "refiner_only": 9,
        "additive_only": 3,
        "joint": 3,
    }
    assert stages is not None
    schedule = build_scratch_stage_schedule(model, stages, original)

    additive_stage = apply_scratch_training_stage(
        model,
        epoch=12,
        schedule=schedule,
        original_trainability=original,
    )
    names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert additive_stage["name"] == SCRATCH_ADDITIVE_STAGE
    assert names
    assert all(name.startswith("refiner.additive_") for name in names)
    model.train()
    apply_scratch_stage_module_modes(model, SCRATCH_ADDITIVE_STAGE)
    assert not model.attention_scale.training
    assert not model.refiner.rgb_encoder.training
    assert model.refiner.additive_body is not None and model.refiner.additive_body.training
    assert model.refiner.additive_output is not None and model.refiner.additive_output.training
