import hashlib
from pathlib import Path

from bim_priorda3.checkpoints import model_config_differences
from bim_priorda3.config import load_config
from bim_priorda3.engine import semantic_config_sha256


def test_config_inheritance_deep_merges_nested_values(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        "model:\n  width: 16\n  nested:\n    keep: 1\n    replace: 2\n",
        encoding="utf-8",
    )
    child.write_text(
        "extends: parent.yaml\nmodel:\n  nested:\n    replace: 3\n",
        encoding="utf-8",
    )
    cfg = load_config(child)
    assert cfg.model.width == 16
    assert cfg.model.nested.keep == 1
    assert cfg.model.nested.replace == 3


def test_nested_config_discovers_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    nested = tmp_path / "configs" / "generated"
    nested.mkdir(parents=True)
    config = nested / "fold.yaml"
    config.write_text("model:\n  width: 16\n", encoding="utf-8")
    cfg = load_config(config)
    assert cfg.project_root == str(tmp_path)


def test_semantic_config_hash_ignores_file_location(tmp_path: Path) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "nested" / "second.yaml"
    second.parent.mkdir()
    for path in (first, second):
        path.write_text("model:\n  width: 16\n", encoding="utf-8")
    assert semantic_config_sha256(load_config(first)) == semantic_config_sha256(load_config(second))


def test_model_config_differences_reports_behavioral_overrides() -> None:
    differences = model_config_differences(
        {"base_channels": 16, "residual_routing_temperature": 0.1},
        {"base_channels": 16, "residual_routing_temperature": 0.05},
    )
    assert differences == {
        "residual_routing_temperature": {
            "checkpoint": 0.1,
            "evaluation": 0.05,
        }
    }


def test_slabim_and_stanford_share_scale_and_anchor_semantics() -> None:
    frozen_source = load_config("configs/stanford_area1_transfer.yaml")
    e2e_source = load_config("configs/stanford_area1_transfer_e2e.yaml")
    frozen_target = load_config("configs/stanford_area1.yaml")
    e2e_target = load_config("configs/stanford_area1_e2e.yaml")

    for source in (frozen_source, e2e_source):
        assert "residual_anchor_mode" not in source.model
        assert "init_checkpoint_policy" not in source.train
        assert "refiner_head_warmup_epochs" not in source.train
    all_configs = (frozen_source, e2e_source, frozen_target, e2e_target)
    for config in all_configs:
        assert "residual_anchor_mode" not in config.model
        assert config.model.residual_routing_scope == "frame_only"
        assert config.model.scale_estimator == frozen_source.model.scale_estimator
        assert config.model.scale_estimator.name == "log_upper_cap_v1"
        assert config.model.scale_estimator.q10_log_cap == "inf"
        assert config.model.scale_estimator.q25_log_cap == 0.05
    assert frozen_target.train.init_checkpoint_policy == "zero_multiplicative_residual_heads"
    assert e2e_target.train.init_checkpoint_policy == "preserve"
    assert frozen_target.train.refiner_head_warmup_epochs == 1
    assert frozen_target.train.gradient_accumulation == 2
    assert frozen_target.train.learning_rate == 4.0e-5
    assert frozen_target.train.lr_warmup_epochs == 0
    assert frozen_target.loss.furniture_multiplier == 2.0
    assert frozen_target.loss.bim_foreground_conflict_multiplier == 2.0
    assert e2e_target.train.refiner_head_warmup_epochs == 0
    assert e2e_target.train.batch_size == 4
    assert e2e_target.train.gradient_accumulation == 2
    assert e2e_target.train.learning_rate == 1.0e-5
    assert e2e_target.train.da3_learning_rate == 1.0e-6
    assert e2e_target.train.lr_warmup_epochs == 1
    assert e2e_target.loss.furniture_multiplier == 2.0
    assert e2e_target.loss.bim_foreground_conflict_multiplier == 2.0


def test_stanford_alignment_receipt_is_present_and_pinned() -> None:
    cfg = load_config("configs/stanford_area1_transfer.yaml")
    path = Path(cfg.project_root) / cfg.data.bim_alignment
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == cfg.data.bim_alignment_sha256


def test_uniform_pixel_iterative_huber_configs_disable_all_pixel_emphasis() -> None:
    scale = load_config(
        "configs/stanford_area1_iterative_attention_huber_no_da3_features_no_confidence_no_bim_geometry_uniform_pixels_3round_3epoch_full_depth_metric_da3.yaml"
    )
    continuation = load_config(
        "configs/stanford_area1_iterative_attention_huber_reduced_refiner_uniform_pixels_continuation_full_depth_metric_da3.yaml"
    )

    for cfg in (scale, continuation):
        assert cfg.loss.near_range_boost == 0.0
        assert cfg.loss.furniture_multiplier == 1.0
        assert cfg.loss.bim_foreground_conflict_multiplier == 1.0
    assert scale.train.scale_only_experiment is True
    assert scale.train.epochs == 3
    assert continuation.train.scale_only_experiment is False
    assert continuation.train.continuation_stage_epochs.refiner_only == 9
    assert continuation.train.continuation_stage_epochs.joint == 3
