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


def test_stanford_target_only_anchor_and_init_policies_do_not_leak_to_source() -> None:
    frozen_source = load_config("configs/stanford_area1_transfer.yaml")
    e2e_source = load_config("configs/stanford_area1_transfer_e2e.yaml")
    frozen_target = load_config("configs/stanford_area1.yaml")
    e2e_target = load_config("configs/stanford_area1_e2e.yaml")

    for source in (frozen_source, e2e_source):
        assert "residual_anchor_mode" not in source.model
        assert "residual_routing_scope" not in source.model
        assert "init_checkpoint_policy" not in source.train
        assert "refiner_head_warmup_epochs" not in source.train
    for target in (frozen_target, e2e_target):
        assert target.model.residual_anchor_mode == "robust_bim_direct"
        assert target.model.residual_routing_scope == "frame_only"
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
