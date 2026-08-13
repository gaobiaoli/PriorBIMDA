from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scripts.analysis.visualize_stanford_sample as visualizer


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))[None]


def _semantic_item() -> dict[str, torch.Tensor]:
    gt = np.asarray(
        [
            [0.10, 0.20, 1.0, 5.0],
            [5.10, 2.00, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    bim = np.asarray(
        [
            [0.0, 0.4, 1.3, 5.0],
            [0.0, 2.0, 0.0, 4.5],
        ],
        dtype=np.float32,
    )
    furniture = np.zeros_like(gt)
    furniture[0, 2] = 1
    non_structural = furniture.copy()
    non_structural[1, 1] = 1
    return {
        "gt_depth": _tensor(gt),
        "gt_valid": _tensor(np.ones_like(gt)),
        "bim_depth": _tensor(bim),
        "bim_valid": _tensor(bim > 0),
        "furniture_mask": _tensor(furniture),
        "non_structural_mask": _tensor(non_structural),
        "semantic_valid": _tensor(np.ones_like(gt)),
    }


def test_long_stanford_sample_id_and_modality_filename_resolution() -> None:
    long_name = "camera_14c620a723e54e8cb6847b4a0c532dca_office_1_frame_158"
    records = [
        {"id": f"office_1/{long_name}"},
        {"id": "office_2/camera_other_office_2_frame_0"},
    ]

    assert visualizer.resolve_sample_index(records, f"office_1/{long_name}") == (
        0,
        f"office_1/{long_name}",
    )
    assert visualizer.resolve_sample_index(records, long_name) == (
        0,
        f"office_1/{long_name}",
    )
    assert visualizer.resolve_sample_index(records, f"{long_name}_domain_rgb.png") == (
        0,
        f"office_1/{long_name}",
    )
    with pytest.raises(KeyError, match="never substitutes"):
        visualizer.resolve_sample_index(records, "camera_missing_frame_0")


def test_fixed_support_is_locked_to_official_depth_range_and_builds_subsets() -> None:
    support, bim_valid, subsets = visualizer.build_fixed_support_and_subsets(_semantic_item())

    assert support.tolist() == [
        [False, True, True, True],
        [False, True, True, True],
    ]
    assert int(subsets["all"].sum()) == 6
    assert int(subsets["furniture"].sum()) == 1
    assert int(subsets["non_structural"].sum()) == 2
    assert int(subsets["bim_foreground_conflict"].sum()) == 3
    assert int(subsets["bim_consistent"].sum()) == 2
    assert int(subsets["bim_no_hit"].sum()) == 1
    assert int((support & bim_valid).sum()) == 5


def test_per_method_subset_metrics_share_support_and_envelope_is_standalone() -> None:
    item = _semantic_item()
    support, bim_valid, subsets = visualizer.build_fixed_support_and_subsets(item)
    gt = visualizer.item_hw(item, "gt_depth")
    bim = visualizer.item_hw(item, "bim_depth")
    first = gt.copy()
    second = gt * 1.1
    first[0, 0] = np.nan
    methods, envelope = visualizer.evaluate_single_frame(
        predictions={"first": first, "second": second},
        gt=gt,
        bim=bim,
        bim_valid=bim_valid,
        subsets=subsets,
    )

    for subset, mask in subsets.items():
        assert methods["first"][subset]["count"] == int(mask.sum())
        assert methods["second"][subset]["count"] == int(mask.sum())
    assert methods["first"]["all"]["abs_rel"] == pytest.approx(0.0)
    assert envelope["coverage"] == {
        "fraction": pytest.approx(5 / 6),
        "hit_pixels": 5,
        "gt_pixels": 6,
        "no_hit_pixels": 1,
    }
    assert envelope["metrics_on_hit_support"]["count"] == 5
    assert support[0, 0] == 0


def test_invalid_prediction_on_fixed_support_is_rejected() -> None:
    item = _semantic_item()
    _, bim_valid, subsets = visualizer.build_fixed_support_and_subsets(item)
    gt = visualizer.item_hw(item, "gt_depth")
    prediction = gt.copy()
    prediction[0, 2] = 0.0

    with pytest.raises(RuntimeError, match="non-positive values on the fixed metric support"):
        visualizer.evaluate_single_frame(
            predictions={"bad": prediction},
            gt=gt,
            bim=visualizer.item_hw(item, "bim_depth"),
            bim_valid=bim_valid,
            subsets=subsets,
        )


@pytest.mark.parametrize("request_live_direct", [False, True])
def test_e2e_eval_requests_live_direct_only_when_explicit(
    tmp_path,
    monkeypatch,
    request_live_direct: bool,
) -> None:
    seen_batches = []

    class FakeModel(torch.nn.Module):
        e2e_da3_enabled = True

        def __init__(self, _cfg):
            super().__init__()

        def forward(self, batch):
            assert self.training is False
            seen_batches.append(batch)
            shape = batch["rgb"].shape[:1] + (1,) + batch["rgb"].shape[-2:]
            depth = torch.ones(shape, device=batch["rgb"].device)
            output = {
                "uses_live_da3": True,
                "depth": depth,
                "base_depth": depth,
                "scaled_depth": depth,
                "bim_reliability": depth * 0.5,
                "log_residual": depth * 0.1,
                "residual_routing_gate": depth,
                "da3_scale": torch.tensor(1.0, device=depth.device),
            }
            if batch.get("request_live_bim_direct", False):
                output["live_bim_direct"] = depth
            return output

    checkpoint = tmp_path / "model.pt"
    torch.save({"model": {}, "config": {"config_path": "fake"}, "epoch": 2}, checkpoint)
    monkeypatch.setattr(visualizer, "BIMPriorDA3", FakeModel)
    monkeypatch.setattr(
        visualizer,
        "validate_checkpoint_evaluation_dataset_provenance",
        lambda *_args, **_kwargs: {"status": "verified", "verified": True},
    )
    monkeypatch.setattr(
        visualizer,
        "validate_checkpoint_model_config",
        lambda *_args, **_kwargs: {},
    )
    item = {"rgb": torch.zeros((3, 2, 3))}

    output, receipt = visualizer.infer_validated_checkpoint(
        cfg=SimpleNamespace(model={}),
        checkpoint_path=checkpoint,
        split_provenance={"fake": True},
        split="val",
        item=item,
        device=torch.device("cpu"),
        expected_e2e=True,
        request_live_direct=request_live_direct,
    )

    assert ("request_live_bim_direct" in seen_batches[0]) is request_live_direct
    assert ("live_bim_direct" in output) is request_live_direct
    assert receipt["checkpoint_epoch"] == 2
    assert len(receipt["sha256"]) == 64


def test_unverified_checkpoint_dataset_provenance_is_rejected(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"model": {}, "config": {"config_path": "fake"}}, checkpoint)
    monkeypatch.setattr(
        visualizer,
        "validate_checkpoint_evaluation_dataset_provenance",
        lambda *_args, **_kwargs: {"status": "legacy_checkpoint_missing", "verified": False},
    )

    with pytest.raises(RuntimeError, match="dataset provenance is not verified"):
        visualizer.infer_validated_checkpoint(
            cfg=SimpleNamespace(model={}),
            checkpoint_path=checkpoint,
            split_provenance={"fake": True},
            split="val",
            item={"rgb": torch.zeros((3, 2, 3))},
            device=torch.device("cpu"),
            expected_e2e=False,
            request_live_direct=False,
        )


def test_json_receipt_conversion_replaces_empty_subset_nan() -> None:
    converted = visualizer._json_safe({"empty": {"abs_rel": float("nan"), "count": np.int64(0)}})
    assert converted == {"empty": {"abs_rel": None, "count": 0}}


def test_render_figure_writes_comprehensive_png(tmp_path) -> None:
    height, width = 4, 5
    gt = np.full((height, width), 2.0, dtype=np.float32)
    bim = gt.copy()
    bim[:, -1] = 0
    support = np.ones_like(gt, dtype=bool)
    bim_valid = bim > 0
    subsets = {
        "all": support,
        "furniture": np.zeros_like(support),
        "non_structural": np.zeros_like(support),
        "bim_foreground_conflict": np.zeros_like(support),
        "bim_consistent": support & bim_valid,
        "bim_no_hit": support & ~bim_valid,
    }
    predictions = {
        "raw_da3": gt * 0.9,
        "global_scale": gt * 0.95,
        "bim_direct": gt * 0.98,
        "frozen_learned": gt * 0.99,
        "e2e": gt,
    }
    metrics, _ = visualizer.evaluate_single_frame(
        predictions=predictions,
        gt=gt,
        bim=bim,
        bim_valid=bim_valid,
        subsets=subsets,
    )
    output = tmp_path / "comparison.png"

    visualizer.render_figure(
        sample_id="office_1/camera_long_office_1_frame_0",
        split="val",
        rgb=np.zeros((height, width, 3), dtype=np.float32),
        gt=gt,
        bim=bim,
        bim_valid=bim_valid,
        subsets=subsets,
        predictions=predictions,
        metrics=metrics,
        e2e_diagnostics={
            "reliability": np.ones_like(gt),
            "log_residual": np.zeros_like(gt),
            "routing_gate": np.ones_like(gt),
        },
        output_path=output,
    )

    assert output.is_file()
    assert output.stat().st_size > 10_000
