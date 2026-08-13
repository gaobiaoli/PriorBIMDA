from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import scripts.analysis.export_stanford_qualitative_panels as exporter


def _row(
    sample_id: str,
    *,
    valid: int,
    furniture: int,
    all_direct: float,
    all_refined: float,
    furniture_direct: float,
    furniture_refined: float,
) -> dict[str, str]:
    return {
        "sample_id": sample_id,
        "all_gt_pixels": str(valid),
        "furniture_gt_pixels": str(furniture),
        "all_robust_bim_direct_abs_rel": str(all_direct),
        "all_refined_abs_rel": str(all_refined),
        "furniture_robust_bim_direct_abs_rel": str(furniture_direct),
        "furniture_refined_abs_rel": str(furniture_refined),
    }


def test_parse_args_supports_repeated_explicit_validation_ids() -> None:
    args = exporter.parse_args(
        ["--sample-id", "office_1/frame_1", "--sample-id", "office_2/frame_2"]
    )
    assert args.sample_id == ["office_1/frame_1", "office_2/frame_2"]
    assert args.preset is None
    with pytest.raises(SystemExit):
        exporter.parse_args(
            ["--sample-id", "office_1/frame_1", "--preset", "three-options"]
        )


def test_three_option_rules_are_predeclared_and_deterministic() -> None:
    rows = [
        _row(
            "room/typical",
            valid=150_000,
            furniture=2_000,
            all_direct=0.12,
            all_refined=0.11,
            furniture_direct=0.3,
            furniture_refined=0.3,
        ),
        _row(
            "room/high",
            valid=100_000,
            furniture=5_000,
            all_direct=0.20,
            all_refined=0.14,
            furniture_direct=0.4,
            furniture_refined=0.35,
        ),
        _row(
            "room/furniture",
            valid=200_000,
            furniture=100_000,
            all_direct=0.20,
            all_refined=0.17,
            furniture_direct=0.5,
            furniture_refined=0.3,
        ),
        _row(
            "room/failure",
            valid=140_000,
            furniture=40_000,
            all_direct=0.10,
            all_refined=0.14,
            furniture_direct=0.3,
            furniture_refined=0.29,
        ),
        _row(
            "room/too_sparse",
            valid=90_000,
            furniture=50_000,
            all_direct=0.10,
            all_refined=0.50,
            furniture_direct=0.2,
            furniture_refined=0.19,
        ),
    ]

    selections = exporter.select_three_options(rows)

    assert [(selection.role, selection.sample_id) for selection in selections] == [
        ("option_a_typical", "room/typical"),
        ("option_b_furniture_conflict_success", "room/furniture"),
        ("option_c_failure", "room/failure"),
    ]
    assert selections[0].rule["eligible_population_median_all_absrel_gain"] == pytest.approx(
        0.01
    )
    assert selections[1].rule["furniture_fraction_times_gain"] == pytest.approx(0.1)


def test_preset_csv_population_must_equal_runtime_validation() -> None:
    rows = [{"sample_id": "val/a"}, {"sample_id": "test/leak"}]
    records = [{"id": "val/a"}, {"id": "val/b"}]
    with pytest.raises(ValueError, match="not exactly the runtime validation population"):
        exporter.validate_selection_csv_population(rows, records)

    exporter.validate_selection_csv_population(
        [{"sample_id": "val/b"}, {"sample_id": "val/a"}], records
    )


def test_validation_resolution_never_substitutes_test_or_ambiguous_frame() -> None:
    records = [
        {"id": "office_1/camera_a_office_1_frame_9"},
        {"id": "office_2/camera_b_office_2_frame_9"},
    ]
    assert exporter.resolve_validation_sample(records, "camera_a_office_1_frame_9") == (
        0,
        "office_1/camera_a_office_1_frame_9",
    )
    with pytest.raises(KeyError, match="never searches or substitutes a test frame"):
        exporter.resolve_validation_sample(records, "camera_test_frame_9")


def _asset_arrays() -> dict[str, np.ndarray]:
    shape = (exporter.PROTOCOL_HEIGHT, exporter.PROTOCOL_WIDTH)
    horizontal = np.linspace(0.2, 5.0, shape[1], dtype=np.float32)
    depth = np.repeat(horizontal[None], shape[0], axis=0)
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    rgb[..., 1] = 0.5
    zeros = np.zeros(shape, dtype=np.float32)
    ones = np.ones(shape, dtype=np.float32)
    coverage = np.ones(shape, dtype=np.int8)
    coverage[:, :20] = 0
    coverage[:20, :] = -1
    return {
        "rgb": rgb,
        "gt": depth,
        "raw_da3": depth,
        "bim_depth": depth,
        "robust_global_scale": depth,
        "robust_bim_direct": depth,
        "refined": depth,
        "raw_absrel": zeros,
        "direct_absrel": zeros,
        "refined_absrel": zeros,
        "refined_minus_direct": zeros,
        "furniture_mask": np.zeros(shape, dtype=np.uint8),
        "conflict_mask": np.zeros(shape, dtype=np.uint8),
        "bim_coverage": coverage,
        "fixed_support": np.ones(shape, dtype=np.uint8),
        "bim_valid": np.ones(shape, dtype=np.uint8),
        "local_correction_log_field": zeros,
        "local_correction_support": ones,
        "reliability": ones * 0.5,
        "routing_gate": ones,
        "total_log_residual": zeros,
        "frame_log_residual": zeros,
        "low_log_residual": zeros,
        "detail_log_residual": zeros,
    }


def test_title_free_panels_are_exact_size_and_use_fixed_display_contract(tmp_path: Path) -> None:
    arrays = _asset_arrays()
    panels = exporter.panel_images(arrays)
    assert set(exporter.DIAGNOSTIC_NAMES).issubset(panels)
    assert panels["gt"].shape == (504, 504, 3)
    assert panels["gt"].dtype == np.uint8
    assert panels["bim_coverage"][0, 100].tolist() == [0, 0, 0]
    assert panels["bim_coverage"][100, 0].tolist() == [242, 142, 43]
    assert panels["bim_coverage"][100, 100].tolist() == [89, 161, 79]

    path = tmp_path / "gt.png"
    exporter.save_title_free_png(path, panels["gt"])
    loaded = plt.imread(path)
    assert loaded.shape[:2] == (504, 504)


def test_export_sample_assets_writes_panels_arrays_metrics_and_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    arrays = _asset_arrays()
    prepared = tmp_path / "prepared.npz"
    source_rgb = tmp_path / "source.png"
    prepared.write_bytes(b"prepared")
    source_rgb.write_bytes(b"source")
    monkeypatch.setattr(
        exporter,
        "write_contact_sheet",
        lambda _panels, output, **_kwargs: output.write_bytes(b"preview"),
    )
    metrics = {
        "methods": {},
        "subset_pixels": {name: 0 for name in exporter.SUBSET_NAMES},
        "scale_estimate": {"scale": 1.0},
    }
    result = exporter.export_sample_assets(
        selection=exporter.Selection(
            role="explicit_01",
            sample_id="office_1/camera_a_frame_1",
            rule={"choice": "explicit validation ID"},
        ),
        record={
            "id": "office_1/camera_a_frame_1",
            "region": "office_1",
            "sample": str(prepared),
            "image": str(source_rgb),
            "preparation_fingerprint_sha256": "a" * 64,
        },
        arrays=arrays,
        metrics=metrics,
        output_dir=tmp_path / "export",
        common_provenance={"validation_population_identity": {"mode": "annotations"}},
    )

    sample_dir = Path(result["directory"])
    assert len(list(sample_dir.glob("*.png"))) == 21
    with np.load(sample_dir / "arrays.npz") as saved:
        assert set(exporter.DIAGNOSTIC_NAMES).issubset(saved.files)
        assert saved["rgb"].shape == (504, 504, 3)
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    assert manifest["purpose"].startswith("Validation-only")
    assert manifest["display_contract"]["all_panels_are_title_free_504x504_png"] is True
    assert len(manifest["artifacts"]["panels"]) == 20


def test_preset_csv_reproduction_check_rejects_stale_result_row() -> None:
    selection = exporter.Selection(
        role="option_a_typical",
        sample_id="room/a",
        rule={},
        csv_row=_row(
            "room/a",
            valid=100,
            furniture=10,
            all_direct=0.2,
            all_refined=0.1,
            furniture_direct=0.3,
            furniture_refined=0.2,
        ),
    )
    metrics = {
        "subset_pixels": {"all": 100, "furniture": 10},
        "methods": {
            "robust_bim_direct": {
                "all": {"abs_rel": 0.2},
                "furniture": {"abs_rel": 0.3},
            },
            "refined": {
                "all": {"abs_rel": 0.15},
                "furniture": {"abs_rel": 0.2},
            },
        },
    }
    with pytest.raises(RuntimeError, match="does not reproduce"):
        exporter._validate_preset_row_against_export(selection, metrics)
