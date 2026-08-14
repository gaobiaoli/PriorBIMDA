from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest
import torch

import scripts.model.evaluate_stanford_area1 as stanford_evaluator
from bim_priorda3.config import Config
from scripts.model.evaluate_stanford_area1 import (
    MetricSums,
    _assert_comparable_counts,
    _beats_on_absrel_and_mae,
    _bootstrap_paired_rooms,
    _coverage_fraction,
    _previous_baseline_tensors,
    _robust_selection_receipt_provenance,
)


def _metric_row(*, count: int, abs_rel: float = 0.1, mae: float = 0.2):
    return {"count": count, "abs_rel": abs_rel, "mae": mae}


def test_metric_sums_uses_fixed_support_and_ignores_values_outside_it() -> None:
    prediction = torch.tensor([[[[1.0, float("nan"), 0.0]]]])
    target = torch.tensor([[[[1.0, 2.0, 3.0]]]])
    support = torch.tensor([[[[1, 0, 0]]]], dtype=torch.bool)

    sums = MetricSums()
    sums.update(prediction, target, support, prediction_name="candidate", context="sample")

    assert sums.compute()["count"] == 1


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_metric_sums_rejects_invalid_prediction_on_fixed_support(invalid: float) -> None:
    prediction = torch.tensor([[[[1.0, invalid]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])
    support = torch.ones_like(target, dtype=torch.bool)

    with pytest.raises(RuntimeError, match="candidate has 1"):
        MetricSums().update(
            prediction,
            target,
            support,
            prediction_name="candidate",
            context="sample",
        )


def test_comparable_count_assertion_detects_support_drift() -> None:
    metrics = {
        "refined": _metric_row(count=4),
        "bim_direct": _metric_row(count=3),
    }
    with pytest.raises(RuntimeError, match="do not share the fixed support"):
        _assert_comparable_counts(metrics, 4, context="all")


def test_empty_support_comparisons_and_bootstrap_are_safe() -> None:
    empty = _metric_row(count=0, abs_rel=float("nan"), mae=float("nan"))
    assert _beats_on_absrel_and_mae(empty, empty) is None

    result = _bootstrap_paired_rooms(
        {"office_1": {"refined": empty, "bim_direct": empty}},
        candidate="refined",
        reference="bim_direct",
        metric="abs_rel",
        seed=7,
        repetitions=20,
    )
    assert result["rooms"] == 0
    assert result["room_ids"] == []
    assert math.isnan(result["mean_difference"])
    assert all(math.isnan(value) for value in result["confidence_interval_95"])


def test_paired_bootstrap_rejects_different_nonempty_supports() -> None:
    room_metrics = {
        "office_1": {
            "refined": _metric_row(count=5),
            "bim_direct": _metric_row(count=4),
        }
    }
    with pytest.raises(RuntimeError, match="support differs"):
        _bootstrap_paired_rooms(
            room_metrics,
            candidate="refined",
            reference="bim_direct",
            metric="abs_rel",
            seed=7,
            repetitions=20,
        )


def test_live_fixed_bim_direct_is_computed_from_the_supplied_live_base() -> None:
    bim = torch.full((1, 1, 12, 12), 2.0)
    cached_base = torch.full_like(bim, 1.0)
    live_base = torch.full_like(bim, 0.5)

    _, cached_direct, cached_scale = _previous_baseline_tensors(cached_base, bim)
    live_scaled, live_direct, live_scale = _previous_baseline_tensors(live_base, bim)

    assert cached_scale == pytest.approx(2.0)
    assert live_scale == pytest.approx(4.0)
    assert torch.allclose(live_scaled, bim)
    assert torch.allclose(cached_direct, bim)
    assert torch.allclose(live_direct, bim)


def test_bim_envelope_coverage_is_explicit_and_empty_safe() -> None:
    assert _coverage_fraction(3, 4) == pytest.approx(0.75)
    assert math.isnan(_coverage_fraction(0, 0))


def test_cli_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(SystemExit):
        stanford_evaluator.parse_args(
            [
                "--config",
                "config.yaml",
                "--checkpoint",
                "model.pt",
                "--batch-size",
                "0",
            ]
        )


def test_cli_rejects_negative_inference_seed() -> None:
    with pytest.raises(SystemExit):
        stanford_evaluator.parse_args(
            [
                "--config",
                "config.yaml",
                "--checkpoint",
                "model.pt",
                "--inference-seed",
                "-1",
            ]
        )


def test_cli_defaults_to_validation_not_test() -> None:
    args = stanford_evaluator.parse_args(["--config", "config.yaml", "--checkpoint", "model.pt"])
    assert args.split == "val"
    assert args.inference_seed is None


def test_cli_accepts_explicit_inference_seed() -> None:
    args = stanford_evaluator.parse_args(
        [
            "--config",
            "config.yaml",
            "--checkpoint",
            "model.pt",
            "--inference-seed",
            "17",
        ]
    )
    assert args.inference_seed == 17


def _baseline_test_batch(sample_count: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.linspace(0.8, 3.2, 16 * 16).reshape(1, 1, 16, 16)
    base = torch.cat([values * (1.0 + 0.07 * index) for index in range(sample_count)])
    ratio = torch.linspace(0.85, 1.35, 16 * 16).reshape(1, 1, 16, 16)
    bim = torch.cat(
        [base[index : index + 1] * ratio * (1.0 + 0.03 * index) for index in range(sample_count)]
    )
    bim[:, :, :, -1] = 0.0
    return base, bim


def test_ordered_baseline_map_caps_workers_and_preserves_order(monkeypatch) -> None:
    created_workers = []
    real_executor = stanford_evaluator.ThreadPoolExecutor

    class RecordingExecutor(real_executor):
        def __init__(self, *args, **kwargs):
            created_workers.append(kwargs["max_workers"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(stanford_evaluator, "ThreadPoolExecutor", RecordingExecutor)
    results = stanford_evaluator._ordered_baseline_map(lambda index: index, 11)

    assert results == list(range(11))
    assert created_workers == [8]


def test_baseline_batch_one_stays_serial(monkeypatch) -> None:
    class UnexpectedExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("batch size one must not create a thread pool")

    monkeypatch.setattr(stanford_evaluator, "ThreadPoolExecutor", UnexpectedExecutor)
    base, bim = _baseline_test_batch(sample_count=1)

    legacy_scaled, legacy_direct, legacy_estimates = (
        stanford_evaluator._previous_baseline_batch_tensors(base, bim)
    )
    robust_scaled, robust_direct, robust_estimates = (
        stanford_evaluator._robust_baseline_batch_tensors(
            base,
            bim,
            {
                "name": "log_upper_cap_v1",
                "q10_log_cap": 0.2,
                "q25_log_cap": 0.05,
                "min_samples": 10,
            },
        )
    )

    assert legacy_scaled.shape == legacy_direct.shape == base.shape
    assert robust_scaled.shape == robust_direct.shape == base.shape
    assert len(legacy_estimates) == len(robust_estimates) == 1


def test_legacy_baseline_serial_and_parallel_are_exactly_equivalent() -> None:
    base, bim = _baseline_test_batch()

    serial = stanford_evaluator._previous_baseline_batch_tensors(
        base,
        bim,
        max_workers=1,
    )
    parallel = stanford_evaluator._previous_baseline_batch_tensors(
        base,
        bim,
        max_workers=4,
    )

    assert torch.equal(serial[0], parallel[0])
    assert torch.equal(serial[1], parallel[1])
    assert serial[2] == parallel[2]
    expected_scales = [
        _previous_baseline_tensors(base[index : index + 1], bim[index : index + 1])[2]
        for index in range(base.shape[0])
    ]
    assert [estimate.scale for estimate in parallel[2]] == pytest.approx(expected_scales)


def test_robust_baseline_serial_and_parallel_are_exactly_equivalent() -> None:
    base, bim = _baseline_test_batch()
    parameters = {
        "name": "log_upper_cap_v1",
        "q10_log_cap": 0.2,
        "q25_log_cap": 0.05,
        "min_samples": 10,
    }

    serial = stanford_evaluator._robust_baseline_batch_tensors(
        base,
        bim,
        parameters,
        max_workers=1,
    )
    parallel = stanford_evaluator._robust_baseline_batch_tensors(
        base,
        bim,
        parameters,
        max_workers=4,
    )

    assert torch.equal(serial[0], parallel[0])
    assert torch.equal(serial[1], parallel[1])
    assert serial[2] == parallel[2]
    per_sample_estimates = [
        stanford_evaluator._robust_baseline_batch_tensors(
            base[index : index + 1],
            bim[index : index + 1],
            parameters,
            max_workers=1,
        )[2][0]
        for index in range(base.shape[0])
    ]
    assert parallel[2] == per_sample_estimates


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_fixture(tmp_path):
    annotation = tmp_path / "annotation.jsonl"
    annotation.write_text(
        "".join(
            json.dumps({"schema_version": 1, "id": sample_id, "split": split}) + "\n"
            for sample_id, split in (
                ("train_a/frame0", "train"),
                ("train_b/frame0", "train"),
                ("val_room/frame0", "val"),
                ("test_room/frame0", "test"),
            )
        ),
        encoding="utf-8",
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = processed / "manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"id": f"{room}/frame0", "region": room}) + "\n"
            for room in ("train_a", "train_b", "val_room", "test_room")
        ),
        encoding="utf-8",
    )
    comparator = {
        "name": "log_upper_cap_v1",
        "ratio_min": 0.2,
        "ratio_max": 5.0,
        "min_samples": 100,
        "q10_log_cap": 0.20,
        "q25_log_cap": 0.05,
    }
    protocol = json.loads(json.dumps(stanford_evaluator._ROBUST_SELECTION_PROTOCOL_V1))
    train_ids_sha256 = stanford_evaluator._canonical_sha256(["train_a/frame0", "train_b/frame0"])
    split_counts = {"train": 2, "val": 1, "test": 1, "excluded": 0}
    rooms = ("train_a", "train_b", "val_room", "test_room")
    split_region_counts = {
        split: {
            room: int(
                (split == "train" and room in {"train_a", "train_b"})
                or split == room.removesuffix("_room")
            )
            for room in rooms
        }
        for split in ("train", "val", "test", "excluded")
    }
    split_provenance = {
        "mode": "annotations",
        "fingerprint_sha256": "b" * 64,
        "manifest_preparation_fingerprint_sha256": "c" * 64,
        "ordered_ids_sha256": {
            "train": train_ids_sha256,
            "val": "4" * 64,
            "test": "5" * 64,
            "excluded": "6" * 64,
        },
        "split_counts": split_counts,
        "split_region_counts": split_region_counts,
    }
    code_files = {
        "scripts/data/select_stanford_scale_caps.py": "1" * 64,
        "src/bim_priorda3/baselines.py": "2" * 64,
        "src/bim_priorda3/data/splits.py": "3" * 64,
    }
    candidate_grid = protocol["candidate_grid"]
    receipt_payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol": protocol,
        "protocol_sha256": stanford_evaluator._canonical_sha256(protocol),
        "provenance": {
            "annotation_raw_sha256": _digest(annotation),
            "split_fingerprint_sha256": "b" * 64,
            "manifest_raw_sha256": _digest(manifest),
            "manifest_preparation_fingerprint_sha256": "c" * 64,
            "code": {
                "files_sha256": code_files,
                "composite_sha256": stanford_evaluator._canonical_sha256(code_files),
            },
        },
        "split_isolation": {
            "annotation_split_counts": split_counts,
            "room_disjoint": True,
            "train_sample_count": 2,
            "train_room_count": 2,
            "train_rooms": ["train_a", "train_b"],
            "ordered_train_ids_sha256": train_ids_sha256,
            "annotation_ordered_train_ids_sha256": train_ids_sha256,
            "selection_accessed_ids_sha256": train_ids_sha256,
            "direct_audit_accessed_ids_sha256": train_ids_sha256,
            "validation_samples_opened": 0,
            "test_samples_opened": 0,
        },
        "candidate_results": [
            {"q10_log_cap": q10, "q25_log_cap": q25}
            for q10 in candidate_grid["q10_log_cap"]
            for q25 in candidate_grid["q25_log_cap"]
        ],
        "leave_one_train_room_out": {"fold_count": 2},
        "final_selection": {
            "canonical_scale_estimator": comparator,
            "selection_scope": "train only",
        },
    }
    receipt = tmp_path / "selection.json"
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    cfg = SimpleNamespace(
        project_root=str(tmp_path),
        data=Config(
            {
                "split_annotation": str(annotation),
                "split_annotation_sha256": _digest(annotation),
                "split_fingerprint_sha256": "b" * 64,
                "processed_root": str(processed),
            }
        ),
        evaluation={
            "robust_scale_selection_receipt": str(receipt),
            "robust_scale_selection_receipt_sha256": _digest(receipt),
            "robust_scale_selection_protocol_sha256": receipt_payload["protocol_sha256"],
        },
    )
    return cfg, comparator, split_provenance, receipt_payload, receipt


def _rewrite_receipt(cfg, receipt, payload) -> None:
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    cfg.evaluation["robust_scale_selection_receipt_sha256"] = _digest(receipt)


def _verify_receipt(cfg, comparator, split_provenance, checkpoint_config=None):
    return _robust_selection_receipt_provenance(
        cfg,
        comparator,
        split_provenance,
        allow_unverified=False,
        checkpoint_config=checkpoint_config or {"model": {}},
    )


def test_robust_selection_receipt_is_bound_to_train_only_current_dataset(
    tmp_path,
) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)

    verified = _verify_receipt(cfg, comparator, split_provenance)

    assert verified["status"] == "verified"
    assert verified["formal_protocol_eligible"] is True
    assert verified["ordered_train_ids_sha256"] == (split_provenance["ordered_ids_sha256"]["train"])
    assert verified["checkpoint_binding"]["status"] == (
        "legacy_source_independent_target_train_comparator"
    )

    payload["split_isolation"]["test_samples_opened"] = 1
    _rewrite_receipt(cfg, receipt, payload)
    with pytest.raises(ValueError, match="opened test samples"):
        _verify_receipt(cfg, comparator, split_provenance)

    payload["split_isolation"]["test_samples_opened"] = 0
    payload["provenance"]["split_fingerprint_sha256"] = "e" * 64
    _rewrite_receipt(cfg, receipt, payload)
    with pytest.raises(ValueError, match="split_fingerprint_sha256 differs"):
        _verify_receipt(cfg, comparator, split_provenance)


def test_robust_receipt_accepts_pre_reorganization_selector_identity(tmp_path) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    files = payload["provenance"]["code"]["files_sha256"]
    files["scripts/select_stanford_scale_caps.py"] = files.pop(
        "scripts/data/select_stanford_scale_caps.py"
    )
    payload["provenance"]["code"]["composite_sha256"] = stanford_evaluator._canonical_sha256(files)
    _rewrite_receipt(cfg, receipt, payload)

    verified = _verify_receipt(cfg, comparator, split_provenance)

    assert verified["selector_code_identity"]["selector_path"] == (
        "scripts/select_stanford_scale_caps.py"
    )


def test_receipt_recomputes_protocol_canonical_sha256(tmp_path) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    payload["protocol"]["estimator"]["formula"] = "tampered"
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match="protocol canonical SHA256 mismatch"):
        _verify_receipt(cfg, comparator, split_provenance)


def test_receipt_rejects_unknown_formula_even_when_protocol_is_rehashed(
    tmp_path,
) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    payload["protocol"]["estimator"]["formula"] = "exp(Q45)"
    payload["protocol_sha256"] = stanford_evaluator._canonical_sha256(payload["protocol"])
    cfg.evaluation["robust_scale_selection_protocol_sha256"] = payload["protocol_sha256"]
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match="estimator formula differs"):
        _verify_receipt(cfg, comparator, split_provenance)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("name", "unknown-protocol", "protocol name"),
        ("schema_version", 2, "schema_version"),
        ("candidate_grid", {}, "fixed 48-candidate grid"),
        ("selection", {}, "objective or tie-break"),
        ("validation_and_test_policy", "samples may be opened", "policy differs"),
    ],
)
def test_receipt_rejects_unknown_registered_protocol_variants(
    tmp_path,
    field,
    replacement,
    message,
) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    payload["protocol"][field] = replacement
    payload["protocol_sha256"] = stanford_evaluator._canonical_sha256(payload["protocol"])
    cfg.evaluation["robust_scale_selection_protocol_sha256"] = payload["protocol_sha256"]
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match=message):
        _verify_receipt(cfg, comparator, split_provenance)


def test_receipt_requires_all_train_access_hashes_to_match_current_split(
    tmp_path,
) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    payload["split_isolation"]["selection_accessed_ids_sha256"] = "f" * 64
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match="access hashes do not all match"):
        _verify_receipt(cfg, comparator, split_provenance)


def test_receipt_train_hashes_must_match_current_split_provenance(tmp_path) -> None:
    cfg, comparator, split_provenance, _, _ = _receipt_fixture(tmp_path)
    split_provenance["ordered_ids_sha256"]["train"] = "f" * 64

    with pytest.raises(ValueError, match="access hashes do not all match"):
        _verify_receipt(cfg, comparator, split_provenance)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("train_sample_count", 1, "train sample count differs"),
        ("train_room_count", 1, "train room count differs"),
    ],
)
def test_receipt_train_population_counts_must_match_current_annotation(
    tmp_path,
    field,
    replacement,
    message,
) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    payload["split_isolation"][field] = replacement
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match=message):
        _verify_receipt(cfg, comparator, split_provenance)


def test_receipt_requires_complete_valid_selector_code_identity(tmp_path) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    del payload["provenance"]["code"]["files_sha256"]["src/bim_priorda3/data/splits.py"]
    files = payload["provenance"]["code"]["files_sha256"]
    payload["provenance"]["code"]["composite_sha256"] = stanford_evaluator._canonical_sha256(files)
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match="lacks required files"):
        _verify_receipt(cfg, comparator, split_provenance)


def test_receipt_rejects_invalid_selector_code_sha256(tmp_path) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    payload["provenance"]["code"]["files_sha256"]["src/bim_priorda3/baselines.py"] = "invalid"
    _rewrite_receipt(cfg, receipt, payload)

    with pytest.raises(ValueError, match="baselines.py.*lowercase hexadecimal SHA256"):
        _verify_receipt(cfg, comparator, split_provenance)


def test_configured_protocol_sha256_is_strictly_matched(tmp_path) -> None:
    cfg, comparator, split_provenance, _, _ = _receipt_fixture(tmp_path)
    cfg.evaluation["robust_scale_selection_protocol_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="Configured robust.*protocol SHA256 mismatch"):
        _verify_receipt(cfg, comparator, split_provenance)


def test_robust_target_checkpoint_must_pin_the_runtime_receipt(tmp_path) -> None:
    cfg, comparator, split_provenance, payload, receipt = _receipt_fixture(tmp_path)
    robust_checkpoint_config = {
        "model": {"scale_estimator": comparator},
        "evaluation": {
            "robust_scale_selection_receipt_sha256": _digest(receipt),
            "robust_scale_selection_protocol_sha256": payload["protocol_sha256"],
        },
    }

    verified = _verify_receipt(
        cfg,
        comparator,
        split_provenance,
        checkpoint_config=robust_checkpoint_config,
    )
    assert verified["checkpoint_binding"]["status"] == ("robust_target_checkpoint_receipt_match")

    robust_checkpoint_config["evaluation"]["robust_scale_selection_receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from the robust target checkpoint"):
        _verify_receipt(
            cfg,
            comparator,
            split_provenance,
            checkpoint_config=robust_checkpoint_config,
        )


def test_unpinned_robust_comparator_requires_explicit_exploratory_opt_out() -> None:
    cfg = SimpleNamespace(evaluation={})
    comparator = {
        "name": "log_upper_cap_v1",
        "ratio_min": 0.2,
        "ratio_max": 5.0,
        "min_samples": 100,
        "q10_log_cap": 0.20,
        "q25_log_cap": 0.05,
    }
    with pytest.raises(ValueError, match="formal Stanford protocol requires"):
        _robust_selection_receipt_provenance(
            cfg,
            comparator,
            {},
            allow_unverified=False,
            checkpoint_config={"model": {}},
        )
    receipt = _robust_selection_receipt_provenance(
        cfg,
        comparator,
        {},
        allow_unverified=True,
        checkpoint_config={"model": {}},
    )
    assert receipt["formal_protocol_eligible"] is False


def test_batch_two_preserves_per_frame_and_aggregate_results(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, _cfg, _split, augment=False):
            assert augment is False
            self.records = [
                {
                    "id": f"sample_{index}",
                    "region": "office_1" if index < 2 else "office_2",
                    "camera_uuid": f"camera_{index}",
                }
                for index in range(3)
            ]
            self.split_provenance = {"fingerprint_sha256": "fake-split"}

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            height = width = 12
            gt = torch.full((1, height, width), 2.0 + 0.2 * index)
            base = gt * (0.80 + 0.03 * index)
            bim = gt.clone()
            bim[:, :, :4] += 0.50
            bim[:, :, -1] = 0.0
            bim_valid = (bim > 0).float()
            furniture = torch.zeros_like(gt)
            furniture[:, :6] = 1.0
            non_structural = torch.zeros_like(gt)
            non_structural[:, :9] = 1.0
            return {
                "sample_id": self.records[index]["id"],
                "region": self.records[index]["region"],
                "rgb": torch.full((3, height, width), 0.1 * (index + 1)),
                "base_depth": base,
                "scaled_depth": base * 1.10,
                "bim_depth": bim,
                "bim_valid": bim_valid,
                "gt_depth": gt,
                "gt_valid": torch.ones_like(gt),
                "furniture_mask": furniture,
                "non_structural_mask": non_structural,
                "structural_mask": 1.0 - non_structural,
                "semantic_valid": torch.ones_like(gt),
            }

    class FakeModel(torch.nn.Module):
        e2e_da3_enabled = True

        def __init__(self, _cfg):
            super().__init__()

        def forward(self, batch):
            assert batch["request_live_bim_direct"] is True
            live_base = batch["base_depth"] * 0.95
            coarse, live_bim_direct, _ = stanford_evaluator._previous_baseline_batch_tensors(
                live_base,
                batch["bim_depth"],
            )
            refined = coarse * (1.0 + 0.01 * batch["rgb"][:, :1])
            return {
                "uses_live_da3": True,
                "base_depth": live_base,
                "scaled_depth": coarse,
                "coarse_depth": coarse,
                "live_bim_direct": live_bim_direct,
                "depth": refined,
            }

    cfg = SimpleNamespace(
        config_path=str(tmp_path / "fake_config.yaml"),
        model={},
        evaluation={
            "robust_scale_estimator": {
                "name": "log_upper_cap_v1",
                "q10_log_cap": 0.20,
                "q25_log_cap": 0.05,
                "min_samples": 10,
            }
        },
        data=SimpleNamespace(min_depth=0.2, max_depth=5.0),
        train=SimpleNamespace(num_workers=0),
        experiment=SimpleNamespace(output_dir="unused", seed=73),
    )
    checkpoint = tmp_path / "fake.pt"
    torch.save({"config": {"config_path": "fake"}, "model": {}}, checkpoint)
    monkeypatch.setattr(stanford_evaluator, "load_config", lambda _path: cfg)
    monkeypatch.setattr(stanford_evaluator, "BIMDepthDataset", FakeDataset)
    monkeypatch.setattr(stanford_evaluator, "BIMPriorDA3", FakeModel)
    monkeypatch.setattr(
        stanford_evaluator,
        "validate_checkpoint_evaluation_dataset_provenance",
        lambda *_args, **_kwargs: {"status": "fake"},
    )
    monkeypatch.setattr(
        stanford_evaluator,
        "validate_checkpoint_model_config",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stanford_evaluator,
        "validate_universal_scale_protocol",
        lambda *_args, **_kwargs: {"status": "fake"},
    )

    summaries = []
    csv_payloads = []
    for batch_size in (1, 2):
        output = tmp_path / f"batch_{batch_size}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_stanford_area1.py",
                "--config",
                "unused.yaml",
                "--checkpoint",
                str(checkpoint),
                "--split",
                "val",
                "--device",
                "cpu",
                "--output",
                str(output),
                "--bootstrap-repetitions",
                "20",
                "--batch-size",
                str(batch_size),
                "--allow-unverified-robust-comparator",
            ],
        )
        stanford_evaluator.main()
        summaries.append(json.loads((output / "summary.json").read_text()))
        csv_payloads.append((output / "per_frame.csv").read_bytes())

    assert csv_payloads[0] == csv_payloads[1]
    for field in (
        "aggregates",
        "per_room",
        "standalone_bim_envelope",
        "paired_room_bootstrap",
        "paired_room_bootstrap_by_reference",
        "learned_beats_bim_direct_absrel_and_mae",
        "learned_beats_bim_direct_by_reference",
    ):
        assert json.dumps(summaries[0][field], sort_keys=True) == json.dumps(
            summaries[1][field],
            sort_keys=True,
        )
    assert summaries[0]["per_frame_csv_sha256"] == summaries[1]["per_frame_csv_sha256"]
    assert summaries[0]["runtime"] == {
        "batch_size": 1,
        "num_workers": 0,
        "inference_seed": 73,
        "deterministic_algorithms": True,
        "unverified_robust_comparator_opt_out": True,
    }
    assert summaries[1]["runtime"] == {
        "batch_size": 2,
        "num_workers": 0,
        "inference_seed": 73,
        "deterministic_algorithms": True,
        "unverified_robust_comparator_opt_out": True,
    }
    scale_receipt = summaries[0]["scale_estimators"]
    assert scale_receipt["model_input"]["name"] == "legacy_q45"
    assert scale_receipt["robust_comparator"]["source"] == "evaluation.robust_scale_estimator"
    assert scale_receipt["primary_bim_direct_reference"] == "live_robust_bim_direct"
    assert set(summaries[0]["paired_room_bootstrap_by_reference"]) == {
        "live_robust_bim_direct",
        "robust_bim_direct",
    }
    aggregate_methods = set(summaries[0]["aggregates"]["all"])
    assert {
        "legacy_global_scale_q45",
        "legacy_bim_direct_q45",
        "robust_global_scale",
        "robust_bim_direct",
        "live_legacy_global_scale_q45",
        "live_legacy_bim_direct_q45",
        "live_robust_global_scale",
        "live_robust_bim_direct",
    } <= aggregate_methods
    assert "global_scale" not in aggregate_methods
    assert "bim_direct" not in aggregate_methods
    with (tmp_path / "batch_1" / "per_frame.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        first_row = next(csv.DictReader(handle))
    for prefix in (
        "legacy",
        "robust",
        "live_legacy",
        "live_robust",
    ):
        for suffix in (
            "q10",
            "q25",
            "q45",
            "support_count",
            "fallback",
            "cap_triggered",
            "scale",
        ):
            assert f"{prefix}_{suffix}" in first_row
    capsys.readouterr()
