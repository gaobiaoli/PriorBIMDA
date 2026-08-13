from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from bim_priorda3.config import Config, load_config
from bim_priorda3.engine import save_checkpoint, validate
from bim_priorda3.losses import BIMPriorLoss
from bim_priorda3.models import BIMPriorDA3
from scripts.model.train import (
    INIT_POLICY_PRESERVE,
    INIT_POLICY_ZERO_RESIDUAL_HEADS,
    REFINER_FULL_STAGE,
    REFINER_HEAD_WARMUP_PARAMETER_PREFIXES,
    REFINER_HEAD_WARMUP_STAGE,
    TRAINING_ARTIFACT_NAMES,
    apply_init_checkpoint_policy,
    apply_refiner_head_warmup_stage,
    bind_split_provenance_to_subset,
    build_optimizer,
    build_refiner_head_warmup_schedule,
    is_validation_accepted,
    load_initial_model_weights,
    optimizer_parameter_names,
    reset_validation_inference_rng,
    resolve_init_checkpoint_policy,
    resolve_refiner_head_warmup_epochs,
    resolve_validation_batch_size,
    resolve_validation_inference_seed,
    snapshot_parameter_trainability,
    validate_checkpoint_selection,
    validate_fresh_output_directory,
)


class _TinyGroupedModel(nn.Module):
    def __init__(self, *, e2e_da3_enabled: bool = True) -> None:
        super().__init__()
        self.e2e_da3_enabled = e2e_da3_enabled
        self.refiner = nn.Linear(2, 2)
        self.da3 = nn.Linear(2, 2)
        self.frozen = nn.Parameter(torch.ones(1), requires_grad=False)

    def trainable_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        groups: dict[str, list[nn.Parameter]] = {
            "da3": [],
            "non_da3": [],
        }
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            group = "da3" if name.startswith("da3.") else "non_da3"
            groups[group].append(parameter)
        return groups


def _baseline_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.full_like(value, 3.0)
        for name, value in model.state_dict().items()
        if not name.startswith("da3.")
    }


def test_e2e_initialization_allows_only_missing_da3_parameters() -> None:
    model = _TinyGroupedModel()
    original_da3 = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.startswith("da3.")
    }
    checkpoint = _baseline_state(model)

    receipt = load_initial_model_weights(model, checkpoint)

    assert receipt["strict"] is False
    assert set(receipt["missing_keys"]) == {"da3.weight", "da3.bias"}
    assert receipt["unexpected_keys"] == []
    for name, value in model.state_dict().items():
        if name.startswith("da3."):
            assert torch.equal(value, original_da3[name])
        else:
            assert torch.equal(value, checkpoint[name])


def _fill_refiner_output_heads(model: BIMPriorDA3, value: float) -> None:
    with torch.no_grad():
        for module in (
            model.refiner.low_output,
            model.refiner.detail_output,
            model.refiner.frame_output,
        ):
            module.weight.fill_(value)
            module.bias.fill_(value)


def test_target_init_policy_zeros_only_exact_residual_output_slices(
    tmp_path: Path,
) -> None:
    source_cfg = load_config("configs/slabim_base.yaml")
    source_cfg.model.base_channels = 4
    source = BIMPriorDA3(source_cfg)
    _fill_refiner_output_heads(source, 3.0)

    target_cfg = load_config("configs/stanford_area1.yaml")
    target_cfg.model.base_channels = 4
    target = BIMPriorDA3(target_cfg)
    load_initial_model_weights(target, source.state_dict())
    detail_auxiliary = target.refiner.detail_output.weight[1:].detach().clone()
    detail_auxiliary_bias = target.refiner.detail_output.bias[1:].detach().clone()
    frame_trust = target.refiner.frame_output.weight[1:].detach().clone()
    frame_trust_bias = target.refiner.frame_output.bias[1:].detach().clone()

    receipt = apply_init_checkpoint_policy(
        target,
        INIT_POLICY_ZERO_RESIDUAL_HEADS,
        init_checkpoint=tmp_path / "source.pt",
        resume=None,
    )

    assert receipt["applied"] is True
    assert receipt["policy"] == INIT_POLICY_ZERO_RESIDUAL_HEADS
    reset = receipt["reset"]
    assert isinstance(reset, dict)
    assert {target_row["parameter"] for target_row in reset["targets"]} == {
        "refiner.low_output.weight",
        "refiner.low_output.bias",
        "refiner.detail_output.weight",
        "refiner.detail_output.bias",
        "refiner.frame_output.weight",
        "refiner.frame_output.bias",
    }
    assert torch.count_nonzero(target.refiner.low_output.weight) == 0
    assert torch.count_nonzero(target.refiner.low_output.bias) == 0
    assert torch.count_nonzero(target.refiner.detail_output.weight[0]) == 0
    assert torch.count_nonzero(target.refiner.detail_output.bias[0]) == 0
    assert torch.count_nonzero(target.refiner.frame_output.weight[0]) == 0
    assert torch.count_nonzero(target.refiner.frame_output.bias[0]) == 0
    assert torch.equal(target.refiner.detail_output.weight[1:], detail_auxiliary)
    assert torch.equal(target.refiner.detail_output.bias[1:], detail_auxiliary_bias)
    assert torch.equal(target.refiner.frame_output.weight[1:], frame_trust)
    assert torch.equal(target.refiner.frame_output.bias[1:], frame_trust_bias)


def test_init_policy_default_preserves_heads_and_resume_never_reapplies(
    tmp_path: Path,
) -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    _fill_refiner_output_heads(model, 2.0)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    preserved = apply_init_checkpoint_policy(
        model,
        INIT_POLICY_PRESERVE,
        init_checkpoint=tmp_path / "source.pt",
        resume=None,
    )
    assert preserved["applied"] is False
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())

    resumed = apply_init_checkpoint_policy(
        model,
        INIT_POLICY_ZERO_RESIDUAL_HEADS,
        init_checkpoint=None,
        resume=tmp_path / "last.pt",
    )
    assert resumed["applied"] is False
    assert resumed["context"] == "resume_checkpoint_restore"
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_nondefault_init_policy_requires_fresh_init_checkpoint() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)

    with pytest.raises(ValueError, match="requires a fresh --init-checkpoint"):
        apply_init_checkpoint_policy(
            model,
            INIT_POLICY_ZERO_RESIDUAL_HEADS,
            init_checkpoint=None,
            resume=None,
        )


def test_init_policy_resolution_and_checkpoint_provenance_round_trip(
    tmp_path: Path,
) -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    assert resolve_init_checkpoint_policy(cfg) == INIT_POLICY_PRESERVE
    cfg.train.init_checkpoint_policy = INIT_POLICY_ZERO_RESIDUAL_HEADS
    assert resolve_init_checkpoint_policy(cfg) == INIT_POLICY_ZERO_RESIDUAL_HEADS
    model = BIMPriorDA3(cfg)
    _fill_refiner_output_heads(model, 1.0)
    receipt = apply_init_checkpoint_policy(
        model,
        INIT_POLICY_ZERO_RESIDUAL_HEADS,
        init_checkpoint=tmp_path / "source.pt",
        resume=None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=0,
        best_metric=0.1,
        cfg=dict(cfg),
        provenance={"initialization_policy": receipt},
    )

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["provenance"]["initialization_policy"] == receipt


def _small_frozen_target_model() -> tuple[Config, BIMPriorDA3]:
    cfg = load_config("configs/stanford_area1.yaml")
    cfg.model.base_channels = 4
    return cfg, BIMPriorDA3(cfg)


def _exact_refiner_head_warmup_names() -> set[str]:
    output_names = {
        f"refiner.{module}.{field}"
        for module in ("low_output", "detail_output", "frame_output")
        for field in ("weight", "bias")
    }
    adapter_names = {
        f"refiner.bim_adapters.{index}.{field}"
        for index in range(4)
        for field in ("weight", "bias")
    }
    return output_names | adapter_names


def test_refiner_head_warmup_uses_exact_parameter_set_and_keeps_optimizer_full() -> None:
    cfg, model = _small_frozen_target_model()
    original = snapshot_parameter_trainability(model)
    original_trainable_names = {name for name, trainable in original.items() if trainable}
    optimizer, _ = build_optimizer(
        model,
        cfg,
        learning_rate=float(cfg.train.learning_rate),
    )
    optimizer_names_before = set(optimizer_parameter_names(model, optimizer))
    schedule = build_refiner_head_warmup_schedule(model, 1, original)

    warmup = apply_refiner_head_warmup_stage(
        model,
        epoch=0,
        warmup_epochs=1,
        original_trainability=original,
    )
    warmup_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = _exact_refiner_head_warmup_names()
    assert set(REFINER_HEAD_WARMUP_PARAMETER_PREFIXES) == {
        "refiner.low_output.",
        "refiner.detail_output.",
        "refiner.frame_output.",
        "refiner.bim_adapters.",
    }
    assert warmup["name"] == REFINER_HEAD_WARMUP_STAGE
    assert warmup["warmup_active"] is True
    assert warmup_names == expected
    assert set(schedule["heads_only_stage"]["trainable_non_da3_parameter_names"]) == expected
    assert optimizer_names_before == original_trainable_names
    assert set(optimizer_parameter_names(model, optimizer)) == optimizer_names_before

    full = apply_refiner_head_warmup_stage(
        model,
        epoch=1,
        warmup_epochs=1,
        original_trainability=original,
    )
    restored = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert full["name"] == REFINER_FULL_STAGE
    assert full["warmup_active"] is False
    assert restored == original_trainable_names
    assert set(optimizer_parameter_names(model, optimizer)) == optimizer_names_before


def test_refiner_head_warmup_default_zero_preserves_legacy_trainability() -> None:
    cfg = load_config("configs/slabim_base.yaml")
    cfg.model.base_channels = 4
    model = BIMPriorDA3(cfg)
    original = snapshot_parameter_trainability(model)

    assert resolve_refiner_head_warmup_epochs(cfg) == 0
    stage = apply_refiner_head_warmup_stage(
        model,
        epoch=0,
        warmup_epochs=0,
        original_trainability=original,
    )

    assert stage["name"] == REFINER_FULL_STAGE
    assert stage["warmup_active"] is False
    assert snapshot_parameter_trainability(model) == original


@pytest.mark.parametrize("value", [True, -1, 1.5])
def test_refiner_head_warmup_rejects_invalid_values(value: object) -> None:
    cfg: dict[str, object] = {"train": {"refiner_head_warmup_epochs": value}}
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        resolve_refiner_head_warmup_epochs(cfg)


def test_refiner_head_warmup_is_forbidden_for_e2e() -> None:
    cfg = {
        "train": {"refiner_head_warmup_epochs": 1},
        "model": {"e2e_da3": {"enabled": True}},
    }
    with pytest.raises(ValueError, match="must be 0 for E2E training"):
        resolve_refiner_head_warmup_epochs(cfg)


def test_refiner_head_warmup_resume_restores_stage_from_start_epoch(
    tmp_path: Path,
) -> None:
    cfg, model = _small_frozen_target_model()
    original = snapshot_parameter_trainability(model)
    schedule = build_refiner_head_warmup_schedule(model, 1, original)
    optimizer, _ = build_optimizer(
        model,
        cfg,
        learning_rate=float(cfg.train.learning_rate),
    )
    optimizer_names = set(optimizer_parameter_names(model, optimizer))
    warmup_stage = apply_refiner_head_warmup_stage(
        model,
        epoch=0,
        warmup_epochs=1,
        original_trainability=original,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint_path = tmp_path / "warmup.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=0,
        best_metric=0.1,
        cfg=dict(cfg),
        provenance={
            "refiner_head_warmup_schedule": schedule,
            "active_refiner_training_stage": warmup_stage,
        },
    )

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved["provenance"]["refiner_head_warmup_schedule"] == schedule
    assert saved["provenance"]["active_refiner_training_stage"]["name"] == (
        REFINER_HEAD_WARMUP_STAGE
    )

    resumed_cfg, resumed_model = _small_frozen_target_model()
    resumed_original = snapshot_parameter_trainability(resumed_model)
    resumed_optimizer, _ = build_optimizer(
        resumed_model,
        resumed_cfg,
        learning_rate=float(resumed_cfg.train.learning_rate),
    )
    resumed_model.load_state_dict(saved["model"], strict=True)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    start_epoch = int(saved["epoch"]) + 1
    resumed_stage = apply_refiner_head_warmup_stage(
        resumed_model,
        epoch=start_epoch,
        warmup_epochs=1,
        original_trainability=resumed_original,
    )

    assert start_epoch == 1
    assert resumed_stage["name"] == REFINER_FULL_STAGE
    assert resumed_stage["warmup_active"] is False
    assert set(optimizer_parameter_names(resumed_model, resumed_optimizer)) == (optimizer_names)
    assert snapshot_parameter_trainability(resumed_model) == resumed_original


def test_cross_dataset_flag_is_limited_to_init_checkpoint() -> None:
    with pytest.raises(
        ValueError,
        match="requires --init-checkpoint.*resume dataset validation is always strict",
    ):
        validate_checkpoint_selection(
            Path("resume.pt"),
            None,
            allow_cross_dataset_initialization=True,
        )

    with pytest.raises(ValueError, match="requires --init-checkpoint"):
        validate_checkpoint_selection(
            None,
            None,
            allow_cross_dataset_initialization=True,
        )

    validate_checkpoint_selection(
        None,
        Path("source.pt"),
        allow_cross_dataset_initialization=True,
    )


@pytest.mark.parametrize("artifact", sorted(TRAINING_ARTIFACT_NAMES))
def test_fresh_training_rejects_output_with_prior_artifacts(
    tmp_path: Path,
    artifact: str,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / artifact).write_text("old run", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Fresh training refuses"):
        validate_fresh_output_directory(output_dir, resume=None)

    validate_fresh_output_directory(output_dir, resume=output_dir / "last.pt")


def test_smoke_subset_provenance_binds_ordered_ids_and_artifacts() -> None:
    split = {
        "mode": "regions",
        "selected_regions": ["office_1"],
        "record_stride_by_region": {},
    }
    records = [
        {
            "id": "office_1/frame_1",
            "preparation_fingerprint_sha256": "1" * 64,
        },
        {
            "id": "office_1/frame_2",
            "preparation_fingerprint_sha256": "2" * 64,
        },
    ]

    first = bind_split_provenance_to_subset(
        split,
        records,
        requested_max_samples=2,
    )
    reordered = bind_split_provenance_to_subset(
        split,
        list(reversed(records)),
        requested_max_samples=2,
    )
    changed = bind_split_provenance_to_subset(
        split,
        [
            records[0],
            {
                **records[1],
                "preparation_fingerprint_sha256": "3" * 64,
            },
        ],
        requested_max_samples=2,
    )

    subset = first["runtime_subset"]
    assert subset["sample_count"] == 2
    assert subset["preparation_fingerprint_status"] == "verified"
    assert subset["fingerprint_sha256"] != reordered["runtime_subset"]["fingerprint_sha256"]
    assert subset["fingerprint_sha256"] != changed["runtime_subset"]["fingerprint_sha256"]


@pytest.mark.parametrize(
    "invalid_state",
    ["missing_non_da3", "partial_da3", "unexpected"],
)
def test_e2e_initialization_rejects_other_incompatibilities(
    invalid_state: str,
) -> None:
    model = _TinyGroupedModel()
    checkpoint = _baseline_state(model)
    if invalid_state == "missing_non_da3":
        checkpoint.pop("refiner.bias")
    elif invalid_state == "partial_da3":
        checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
        checkpoint.pop("da3.bias")
    else:
        checkpoint["legacy.unexpected"] = torch.ones(1)

    with pytest.raises(ValueError, match="E2E stage initialization is incompatible"):
        load_initial_model_weights(model, checkpoint)


def test_e2e_optimizer_uses_distinct_non_da3_and_da3_learning_rates() -> None:
    model = _TinyGroupedModel()
    cfg = Config(
        {
            "train": Config(
                {
                    "da3_learning_rate": 2.5e-6,
                    "weight_decay": 1.0e-4,
                }
            )
        }
    )

    optimizer, receipt = build_optimizer(model, cfg, learning_rate=4.0e-4)

    groups = {str(group["name"]): group for group in optimizer.param_groups}
    assert set(groups) == {"non_da3", "da3"}
    assert groups["non_da3"]["lr"] == pytest.approx(4.0e-4)
    assert groups["da3"]["lr"] == pytest.approx(2.5e-6)

    expected = model.trainable_parameter_groups()
    assert {id(parameter) for parameter in groups["non_da3"]["params"]} == {
        id(parameter) for parameter in expected["non_da3"]
    }
    assert {id(parameter) for parameter in groups["da3"]["params"]} == {
        id(parameter) for parameter in expected["da3"]
    }
    assert all(
        model.frozen is not parameter for group in groups.values() for parameter in group["params"]
    )

    receipt_by_name = {str(group["name"]): group for group in receipt}
    assert receipt_by_name["non_da3"]["learning_rate"] == pytest.approx(4.0e-4)
    assert receipt_by_name["da3"]["learning_rate"] == pytest.approx(2.5e-6)


def test_validation_batch_size_defaults_to_one_and_accepts_override() -> None:
    assert resolve_validation_batch_size({"train": {}}) == 1
    assert resolve_validation_batch_size({"train": {"val_batch_size": 8}}) == 8
    with pytest.raises(ValueError, match="must be a positive integer"):
        resolve_validation_batch_size({"train": {"val_batch_size": 0}})


def test_validation_inference_seed_defaults_to_experiment_seed_and_resets_rng() -> None:
    cfg = {"train": {}}
    assert resolve_validation_inference_seed(cfg, experiment_seed=42) == 42
    cfg["train"]["validation_inference_seed"] = 73
    assert resolve_validation_inference_seed(cfg, experiment_seed=42) == 73

    loader_generator = torch.Generator()
    assert reset_validation_inference_rng(73, loader_generator) == 90
    global_first = torch.randint(0, 100_000, (8,))
    loader_first = torch.randint(
        0,
        100_000,
        (8,),
        generator=loader_generator,
    )
    torch.rand(100)
    torch.rand(100, generator=loader_generator)
    assert reset_validation_inference_rng(73, loader_generator) == 90
    assert torch.equal(global_first, torch.randint(0, 100_000, (8,)))
    assert torch.equal(
        loader_first,
        torch.randint(0, 100_000, (8,), generator=loader_generator),
    )


@pytest.mark.parametrize("value", [True, -1, 1.5, "42"])
def test_validation_inference_seed_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        resolve_validation_inference_seed(
            {"train": {"validation_inference_seed": value}},
            experiment_seed=42,
        )


class _ValidationModel(nn.Module):
    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert batch.get("request_live_bim_direct") is True
        target = batch["gt_depth"]
        return {
            "base_depth": torch.full_like(target, 4.0),
            "scaled_depth": torch.full_like(target, 3.0),
            "live_bim_direct": torch.full_like(target, 2.4),
            "depth": torch.full_like(target, 2.2),
            "log_residual": torch.zeros_like(target),
            "uses_live_da3": True,
        }


class _ZeroCriterion(nn.Module):
    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del batch
        return {"total": output["depth"].sum() * 0.0}


def test_validate_reports_live_stages_without_replacing_fixed_anchor() -> None:
    shape = (1, 1, 2, 2)
    anchor = torch.full(shape, 2.0)
    batch = {
        "base_depth": torch.full(shape, 1.0),
        "scaled_depth": torch.full(shape, 1.5),
        "anchor_depth": anchor,
        "gt_depth": torch.full(shape, 2.0),
        "gt_valid": torch.ones(shape),
    }
    original_anchor = anchor.clone()

    metrics = validate(
        _ValidationModel(),
        [batch],
        _ZeroCriterion(),
        torch.device("cpu"),
        amp=False,
    )

    assert metrics["base_abs_rel"] == pytest.approx(0.5)
    assert metrics["scaled_abs_rel"] == pytest.approx(0.25)
    assert metrics["anchor_abs_rel"] == pytest.approx(0.0)
    assert metrics["live_da3_abs_rel"] == pytest.approx(1.0)
    assert metrics["live_scale_abs_rel"] == pytest.approx(0.5)
    assert metrics["live_bim_direct_abs_rel"] == pytest.approx(0.2)
    assert metrics["live_bim_direct_mae"] == pytest.approx(0.4)
    assert math.isnan(metrics["live_bim_direct_near_abs_rel"])
    assert metrics["refined_minus_live_bim_direct_abs_rel"] == pytest.approx(-0.1)
    assert metrics["refined_abs_rel"] == pytest.approx(0.1)
    assert metrics["anchor_count"] == 4
    assert metrics["live_da3_count"] == 4
    assert metrics["live_scale_count"] == 4
    assert metrics["live_bim_direct_count"] == 4
    assert torch.equal(batch["anchor_depth"], original_anchor)
    assert "request_live_bim_direct" not in batch


class _RobustValidationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale_estimator_config = {"name": "log_upper_cap_v1"}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        assert batch.get("request_live_bim_direct") is True
        target = batch["gt_depth"]
        return {
            "base_depth": torch.full_like(target, 4.0),
            "scaled_depth": torch.full_like(target, 3.0),
            # Deliberately wrong generic alias: robust validation must select
            # the explicit configured comparator below.
            "live_bim_direct": torch.full_like(target, 20.0),
            "live_robust_bim_direct": torch.full_like(target, 2.6),
            "depth": torch.full_like(target, 2.2),
            "log_residual": torch.zeros_like(target),
            "uses_live_da3": True,
        }


def test_robust_validation_uses_explicit_cached_and_live_comparators() -> None:
    shape = (1, 1, 2, 2)
    batch = {
        "base_depth": torch.full(shape, 1.0),
        "scaled_depth": torch.full(shape, 2.3),
        "anchor_depth": torch.full(shape, 2.4),
        "gt_depth": torch.full(shape, 2.0),
        "gt_valid": torch.ones(shape),
    }

    metrics = validate(
        _RobustValidationModel(),
        [batch],
        _ZeroCriterion(),
        torch.device("cpu"),
        amp=False,
    )

    assert metrics["robust_global_scale_abs_rel"] == pytest.approx(0.15)
    assert metrics["robust_bim_direct_abs_rel"] == pytest.approx(0.20)
    assert metrics["live_robust_bim_direct_abs_rel"] == pytest.approx(0.30)
    assert metrics["live_robust_bim_direct_mae"] == pytest.approx(0.60)
    assert "live_bim_direct_abs_rel" not in metrics
    assert metrics["refined_minus_live_robust_bim_direct_abs_rel"] == pytest.approx(-0.20)


class _InvalidValidationModel(_ValidationModel):
    def __init__(self, output_key: str, invalid_value: float) -> None:
        super().__init__()
        self.output_key = output_key
        self.invalid_value = invalid_value

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        output = super().forward(batch)
        prediction = output[self.output_key].clone()
        prediction.flatten()[0] = self.invalid_value
        output[self.output_key] = prediction
        return output


@pytest.mark.parametrize("invalid_value", [float("nan"), 0.0])
@pytest.mark.parametrize(
    ("method", "location", "key"),
    [
        ("base", "batch", "base_depth"),
        ("scaled", "batch", "scaled_depth"),
        ("anchor", "batch", "anchor_depth"),
        ("refined", "output", "depth"),
        ("live_da3", "output", "base_depth"),
        ("live_scale", "output", "scaled_depth"),
        ("live_bim_direct", "output", "live_bim_direct"),
    ],
)
def test_validate_rejects_invalid_predictions_on_fixed_gt_support(
    invalid_value: float,
    method: str,
    location: str,
    key: str,
) -> None:
    shape = (1, 1, 2, 2)
    batch = {
        "base_depth": torch.ones(shape),
        "scaled_depth": torch.ones(shape),
        "anchor_depth": torch.ones(shape),
        "gt_depth": torch.ones(shape),
        "gt_valid": torch.ones(shape),
    }
    model: nn.Module = _ValidationModel()
    if location == "batch":
        batch[key] = batch[key].clone()
        batch[key].flatten()[0] = invalid_value
    else:
        model = _InvalidValidationModel(key, invalid_value)

    with pytest.raises(RuntimeError, match=rf"Validation {method} .*fixed GT support"):
        validate(
            model,
            [batch],
            _ZeroCriterion(),
            torch.device("cpu"),
            amp=False,
        )


def test_e2e_acceptance_requires_both_cached_and_live_bim_direct_gates() -> None:
    acceptance = {"enabled": True, "near_relative_tolerance": 1.02}
    validation = {
        "fixed_gt_support_count": 4,
        "near_fixed_gt_support_count": 2,
        "base_count": 4,
        "scaled_count": 4,
        "anchor_count": 4,
        "refined_count": 4,
        "live_da3_count": 4,
        "live_scale_count": 4,
        "live_bim_direct_count": 4,
        "anchor_near_count": 2,
        "refined_near_count": 2,
        "live_bim_direct_near_count": 2,
        "refined_abs_rel": 0.08,
        "refined_mae": 0.16,
        "refined_near_abs_rel": 0.10,
        "anchor_abs_rel": 0.09,
        "anchor_mae": 0.18,
        "anchor_near_abs_rel": 0.10,
        "live_bim_direct_abs_rel": 0.10,
        "live_bim_direct_mae": 0.20,
        "live_bim_direct_near_abs_rel": 0.11,
    }
    assert is_validation_accepted(
        validation,
        acceptance,
        e2e_enabled=True,
    )

    count_failure = dict(validation, live_scale_count=3)
    with pytest.raises(ValueError, match="counts must equal the fixed GT support"):
        is_validation_accepted(
            count_failure,
            acceptance,
            e2e_enabled=True,
        )

    invalid_metric = dict(validation, refined_abs_rel=float("nan"))
    with pytest.raises(ValueError, match="acceptance metrics must be finite"):
        is_validation_accepted(
            invalid_metric,
            acceptance,
            e2e_enabled=True,
        )

    invalid_live_near = dict(
        validation,
        live_bim_direct_near_abs_rel=float("nan"),
    )
    with pytest.raises(ValueError, match="acceptance metrics must be finite"):
        is_validation_accepted(
            invalid_live_near,
            acceptance,
            e2e_enabled=True,
        )

    live_failure = dict(validation, live_bim_direct_abs_rel=0.07)
    assert not is_validation_accepted(
        live_failure,
        acceptance,
        e2e_enabled=True,
    )
    assert is_validation_accepted(
        live_failure,
        acceptance,
        e2e_enabled=False,
    )

    cached_failure = dict(validation, anchor_mae=0.15)
    assert not is_validation_accepted(
        cached_failure,
        acceptance,
        e2e_enabled=True,
    )

    live_near_failure = dict(validation, live_bim_direct_near_abs_rel=0.09)
    assert not is_validation_accepted(
        live_near_failure,
        acceptance,
        e2e_enabled=True,
    )


def test_disabled_acceptance_preserves_legacy_behavior() -> None:
    assert is_validation_accepted({}, {"enabled": False}, e2e_enabled=True)


def test_robust_acceptance_uses_explicit_robust_keys() -> None:
    validation = {
        "fixed_gt_support_count": 4,
        "near_fixed_gt_support_count": 2,
        "base_count": 4,
        "scaled_count": 4,
        "anchor_count": 4,
        "refined_count": 4,
        "robust_global_scale_count": 4,
        "robust_bim_direct_count": 4,
        "live_da3_count": 4,
        "live_robust_global_scale_count": 4,
        "live_robust_bim_direct_count": 4,
        "anchor_near_count": 2,
        "refined_near_count": 2,
        "robust_bim_direct_near_count": 2,
        "live_robust_bim_direct_near_count": 2,
        "refined_abs_rel": 0.08,
        "refined_mae": 0.16,
        "refined_near_abs_rel": 0.10,
        "robust_bim_direct_abs_rel": 0.09,
        "robust_bim_direct_mae": 0.18,
        "robust_bim_direct_near_abs_rel": 0.10,
        "live_robust_bim_direct_abs_rel": 0.10,
        "live_robust_bim_direct_mae": 0.20,
        "live_robust_bim_direct_near_abs_rel": 0.11,
        # These aliases are intentionally better than refined and must not be
        # consulted in robust mode.
        "anchor_abs_rel": 0.01,
        "anchor_mae": 0.01,
        "anchor_near_abs_rel": 0.01,
        "live_bim_direct_abs_rel": 0.01,
        "live_bim_direct_mae": 0.01,
    }
    assert is_validation_accepted(
        validation,
        {"enabled": True, "near_relative_tolerance": 1.02},
        e2e_enabled=True,
        robust_scale_enabled=True,
    )

    validation["live_robust_bim_direct_abs_rel"] = 0.07
    assert not is_validation_accepted(
        validation,
        {"enabled": True, "near_relative_tolerance": 1.02},
        e2e_enabled=True,
        robust_scale_enabled=True,
    )


def test_robust_e2e_loss_requires_explicit_live_robust_direct() -> None:
    cfg = Config(
        {
            "model": Config(
                {
                    "variant": "prior_conditioned_v4",
                    "max_total_log_residual": 0.45,
                    "scale_estimator": {
                        "name": "log_upper_cap_v1",
                        "q10_log_cap": 0.20,
                        "q25_log_cap": 0.05,
                    },
                }
            ),
            "loss": Config(
                {
                    "warmup_epochs": 0,
                    "trust_margin": 0.005,
                    "trust_temperature": 0.03,
                }
            ),
        }
    )
    criterion = BIMPriorLoss(cfg)
    shape = (1, 1, 2, 2)
    batch = {
        "gt_depth": torch.full(shape, 2.0),
        "gt_valid": torch.ones(shape),
        "gt_weight": torch.ones(shape),
    }
    output = {
        "uses_live_da3": True,
        "depth": torch.full(shape, 2.1),
        "scaled_depth": torch.full(shape, 2.0),
        "live_bim_direct": torch.full(shape, 2.0),
    }

    with pytest.raises(KeyError, match="live_robust_bim_direct"):
        criterion(output, batch)
