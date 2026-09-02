from __future__ import annotations

import pytest
import torch

from bim_priorda3.config import load_config
from bim_priorda3.data import (
    apply_bim_condition_dropout,
    apply_da3_global_scale_perturbation,
)
from bim_priorda3.losses import absrel_optimal_log_scale
from bim_priorda3.models import build_bim_condition

CONFIG = (
    "configs/stanford_area1_dav2_early_fusion_scale_low36_only_6epoch_continuous_"
    "da3_global_scale_perturb_full_depth_metric_da3.yaml"
)


def _batch() -> dict[str, torch.Tensor]:
    base = torch.tensor(
        [
            [[[1.0, 1.5], [2.0, 2.5]]],
            [[[0.8, 1.2], [1.6, 2.0]]],
        ]
    )
    return {
        "rgb": torch.zeros((2, 3, 2, 2)),
        "base_depth": base,
        "bim_depth": base * 1.4,
        "bim_valid": torch.ones_like(base),
        "gt_depth": base * torch.tensor([1.7, 0.9])[:, None, None, None],
        "gt_valid": torch.ones_like(base),
    }


def test_augmented_config_is_six_epoch_continuous_scale_r36() -> None:
    cfg = load_config(CONFIG)

    assert cfg.train.epochs == 6
    assert cfg.train.reset_optimizer_scheduler_on_resume is False
    assert cfg.train.encoder_learning_rate == pytest.approx(5e-6)
    assert cfg.train.decoder_learning_rate == pytest.approx(5e-5)
    assert cfg.model.dav2_joint_scale_low.residual_mode == "low36_only"
    assert cfg.model.dav2_joint_scale_low.residual_composition == "low36_only"
    assert cfg.train.augment.da3_global_scale_perturbation == {
        "enabled": True,
        "probability": 1.0,
        "log_range": 0.2,
    }
    assert cfg.train.augment.bim_condition_dropout == {
        "enabled": False,
        "probability": 0.15,
    }
    assert cfg.data.split_fingerprint_sha256 == (
        "87dbde4e9e454c9dca1e2f38d86fd339ae45b401145b29324ada4174d35043f7"
    )


def test_da3_global_scale_perturbation_preserves_gt_bim_and_shifts_targets() -> None:
    batch = _batch()
    original_base = batch["base_depth"].clone()
    original_oracle, supported = absrel_optimal_log_scale(
        batch["base_depth"], batch["gt_depth"], batch["gt_valid"]
    )
    original_condition = build_bim_condition(batch, bim_log_mean=0.0, bim_log_std=1.0)

    torch.manual_seed(7)
    augmented, log_q, applied = apply_da3_global_scale_perturbation(
        batch, probability=1.0, log_range=0.2
    )
    augmented_oracle, augmented_supported = absrel_optimal_log_scale(
        augmented["base_depth"], augmented["gt_depth"], augmented["gt_valid"]
    )
    augmented_condition = build_bim_condition(
        augmented, bim_log_mean=0.0, bim_log_std=1.0
    )

    assert bool(applied.all())
    assert bool(supported.all()) and torch.equal(supported, augmented_supported)
    torch.testing.assert_close(
        augmented["base_depth"], original_base * log_q.exp(), rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(augmented_oracle, original_oracle - log_q, rtol=1e-5, atol=1e-6)
    # z is the third condition channel divided by disagreement_clip (default 1.5).
    torch.testing.assert_close(
        augmented_condition[:, 2:3],
        original_condition[:, 2:3] - log_q / 1.5,
        rtol=1e-5,
        atol=1e-6,
    )
    assert augmented["gt_depth"] is batch["gt_depth"]
    assert augmented["bim_depth"] is batch["bim_depth"]
    torch.testing.assert_close(batch["base_depth"], original_base)


def test_da3_global_scale_perturbation_probability_zero_is_identity() -> None:
    batch = _batch()
    torch.manual_seed(19)
    rng_before = torch.random.get_rng_state()
    augmented, log_q, applied = apply_da3_global_scale_perturbation(
        batch, probability=0.0, log_range=0.2
    )
    assert not bool(applied.any())
    assert torch.count_nonzero(log_q) == 0
    torch.testing.assert_close(augmented["base_depth"], batch["base_depth"])
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize(
    ("probability", "log_range"),
    [(-0.1, 0.2), (1.1, 0.2), (1.0, -0.1), (1.0, float("inf"))],
)
def test_da3_global_scale_perturbation_rejects_invalid_ranges(
    probability: float, log_range: float
) -> None:
    with pytest.raises(ValueError):
        apply_da3_global_scale_perturbation(
            _batch(), probability=probability, log_range=log_range
        )


def test_bim_condition_dropout_zeros_complete_selected_conditions() -> None:
    condition = torch.arange(3 * 2 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2, 2) + 1
    original = condition.clone()
    mask = torch.tensor([True, False, True])

    dropped, returned_mask = apply_bim_condition_dropout(condition, applied=mask)

    assert returned_mask is mask
    assert torch.count_nonzero(dropped[0]) == 0
    torch.testing.assert_close(dropped[1], original[1])
    assert torch.count_nonzero(dropped[2]) == 0
    torch.testing.assert_close(condition, original)


def test_bim_condition_dropout_probability_one_drops_every_sample() -> None:
    condition = torch.ones((3, 3, 2, 2))

    dropped, mask = apply_bim_condition_dropout(condition, probability=1.0)

    assert bool(mask.all())
    assert torch.count_nonzero(dropped) == 0


def test_bim_condition_dropout_reuses_mask_for_equivariance_forward() -> None:
    main = torch.ones((4, 3, 2, 2))
    second = torch.full_like(main, 2.0)
    requested_mask = torch.tensor([True, False, True, False])

    dropped_main, mask = apply_bim_condition_dropout(main, applied=requested_mask)
    dropped_second, second_mask = apply_bim_condition_dropout(second, applied=mask)

    assert mask is requested_mask
    assert second_mask is mask
    assert torch.count_nonzero(dropped_main[mask]) == 0
    assert torch.count_nonzero(dropped_second[mask]) == 0
    torch.testing.assert_close(dropped_main[~mask], main[~mask])
    torch.testing.assert_close(dropped_second[~mask], second[~mask])


def test_disabled_bim_condition_dropout_is_identity_without_rng_change() -> None:
    condition = torch.randn((2, 3, 2, 2))
    torch.manual_seed(29)
    rng_before = torch.random.get_rng_state()

    dropped, mask = apply_bim_condition_dropout(condition, probability=0.0)

    assert dropped is condition
    assert not bool(mask.any())
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_bim_condition_dropout_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError):
        apply_bim_condition_dropout(torch.ones((2, 3, 2, 2)), probability=probability)
