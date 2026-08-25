from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

import scripts.analysis.export_stanford_pano_panels as exporter


def _combined_pair(
    station_id: str,
    gain: float,
    *,
    room: str = "office_1",
) -> list[dict[str, str]]:
    reference = 0.25
    common = {
        "station_id": station_id,
        "room": room,
        "support_scope": exporter.SUPPORT_SCOPE,
        "fusion_method": exporter.FSTAR_EXPECTED,
        "depth_method": exporter.RAW_METHOD,
        "fixed_support_pixels": "100",
        "gt_valid_pixels_at_eval_resolution": "200",
        "coverage_fraction": "0.5",
        "solid_angle_coverage_fraction": "0.6",
    }
    return [
        {
            **common,
            "source_set": exporter.SOURCE_REGULAR,
            "spherical_abs_rel": str(reference),
        },
        {
            **common,
            "source_set": exporter.SOURCE_JOINT14,
            "spherical_abs_rel": str(reference - gain),
        },
    ]


def _primary_summary() -> dict[str, object]:
    values = {
        "joint_weighted_log": (0.25, 0.50),
        "joint_huber": (0.24, 0.48),
        "joint_photo_huber": (0.241, 0.47),
        "joint_synchronized_huber": (0.28, 0.60),
    }
    return {
        "primary_metrics": {
            fusion: {"raw_da3": {"abs_rel": absrel, "mae": mae}}
            for fusion, (absrel, mae) in values.items()
        }
    }


def test_cli_has_no_test_split_escape_hatch() -> None:
    args = exporter.parse_args([])
    assert not hasattr(args, "split")
    with pytest.raises(SystemExit):
        exporter.parse_args(["--split", "test"])
    with pytest.raises(SystemExit):
        exporter.parse_args(["--checkpoint", "learned.pt"])
    with pytest.raises(SystemExit):
        exporter.parse_args(["--metric-atol", "0.1"])


def test_fstar_and_three_station_rules_are_preregistered_and_deterministic() -> None:
    fstar, receipt = exporter.select_fstar(_primary_summary())
    assert fstar == "joint_huber"
    assert receipt["candidate_pool"] == [
        "joint_weighted_log",
        "joint_huber",
        "joint_photo_huber",
        "joint_synchronized_huber",
    ]
    rows = []
    for index, gain in enumerate((0.01, 0.02, 0.03, 0.04, 0.05), start=1):
        rows.extend(_combined_pair(f"station_{index}", gain))

    selected = exporter.select_three_stations(rows, fstar=fstar, expected_station_count=5)

    assert [(item.role, item.station_id) for item in selected] == [
        ("A_median_gain", "station_3"),
        ("B_maximum_gain", "station_5"),
        ("C_minimum_gain_hardest", "station_1"),
    ]
    assert selected[0].rule["population_median_gain"] == pytest.approx(0.03)
    assert selected[2].gain_abs_rel > 0  # "hardest" remains an honest positive gain.


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _formal_result_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    combined = []
    for index in range(30):
        combined.extend(_combined_pair(f"station_{index:02d}", 0.001 * (index + 1)))
    _write_csv(root / "regular_pano_joint_per_station.csv", combined)
    dummy = [{"station_id": "station_00", "value": "1"}]
    for filename in exporter.FORMAL_CSV_NAMES[:-1]:
        _write_csv(root / filename, dummy)

    summary = {
        "schema_version": exporter.FORMAL_SCHEMA_VERSION,
        "protocol": exporter.FORMAL_PROTOCOL,
        "split": "val",
        "formal_protocol_eligible": True,
        "station_count": 30,
        "primary_metric_protocol": {
            "station_aggregation": "equal-station macro",
            "fixed_support_shared_by_all_methods": True,
        },
        **_primary_summary(),
    }
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    provenance: dict[str, object] = {
        "schema_version": exporter.FORMAL_SCHEMA_VERSION,
        "protocol": exporter.FORMAL_PROTOCOL,
        "split": "val",
        "formal_protocol_eligible": True,
        "test_access_explicitly_authorized": False,
        "checkpoint": {"status": "verified", "learned_method_evaluated": True},
        "route_p_tangent_inputs": {"status": "verified_formal_manifest"},
    }
    for filename in exporter.FORMAL_CSV_NAMES:
        path = root / filename
        provenance[exporter._recorded_csv_key(filename)] = {
            "path": f"/relocated/formal/{filename}",
            "sha256": exporter.file_sha256(path),
        }
    (root / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


def test_formal_loader_accepts_only_full_val_and_checks_csv_hashes(tmp_path: Path) -> None:
    result = tmp_path / "pano_val"
    _formal_result_fixture(result)
    loaded = exporter.load_formal_val_artifacts(result)
    assert loaded.fstar == "joint_huber"
    assert len(loaded.regular_rows) == 60
    assert set(exporter.FORMAL_CSV_NAMES).issubset(loaded.identities)

    with (result / "regular_pano_joint_per_station.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        exporter.load_formal_val_artifacts(result)


def test_json_paths_are_repo_relative_but_external_paths_remain_absolute(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    repo_file = exporter.PROJECT_ROOT / "configs/stanford_area1.yaml"
    external_file = tmp_path / "external.bin"
    exporter.write_json(
        output,
        {
            "repo_artifact": {"path": str(repo_file)},
            "archival": {"formal_recorded_path": str(repo_file)},
            "external_artifact": {"path": str(external_file)},
        },
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["repo_artifact"]["path"] == "configs/stanford_area1.yaml"
    assert payload["archival"]["formal_recorded_path"] == "configs/stanford_area1.yaml"
    assert payload["external_artifact"]["path"] == str(external_file.resolve())
    assert str(exporter.PROJECT_ROOT) not in output.read_text(encoding="utf-8")


def test_stanford_source_identity_is_relative_to_area_root(tmp_path: Path) -> None:
    area_root = tmp_path / "Stanford2D3DS" / "area_1"
    source = area_root / "pano" / "depth" / "camera_uuid_depth.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"official-pano-depth")

    identity = exporter._area_root_relative_identity(
        source,
        exporter.file_sha256(source),
        area_root=area_root,
        context="synthetic pano depth",
    )

    assert identity == {
        "path": "pano/depth/camera_uuid_depth.png",
        "path_base": "data.area_root",
        "sha256": exporter.file_sha256(source),
    }
    external = tmp_path / "external.bin"
    external.write_bytes(b"outside")
    with pytest.raises(RuntimeError, match="outside configured data.area_root"):
        exporter._area_root_relative_identity(
            external,
            exporter.file_sha256(external),
            area_root=area_root,
            context="external",
        )


def _metric_row(value: float = 0.1) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "fixed_support_pixels": 100,
        "gt_valid_pixels_at_eval_resolution": 200,
        "count": 100,
    }
    for name in exporter.evaluator.METRIC_NAMES:
        row[name] = value
        row[f"spherical_{name}"] = value
    return row


def test_metric_reproduction_is_strict_and_rejects_stale_formal_row() -> None:
    actual = _metric_row()
    actual["coverage_fraction"] = 0.5
    receipt = exporter.verify_metric_row(actual, dict(actual), context="station", atol=1e-8)
    assert receipt["spherical_abs_rel"] == pytest.approx(0.1)
    stale = dict(actual)
    stale["spherical_abs_rel"] = 0.11
    with pytest.raises(RuntimeError, match="does not reproduce formal CSV"):
        exporter.verify_metric_row(actual, stale, context="station", atol=1e-8)
    stale = dict(actual)
    stale["coverage_fraction"] = 0.6
    with pytest.raises(RuntimeError, match="does not reproduce formal CSV"):
        exporter.verify_metric_row(actual, stale, context="station", atol=1e-8)


def _panel_arrays() -> dict[str, np.ndarray]:
    shape = (exporter.HEIGHT, exporter.WIDTH)
    depth = np.full(shape, 2.0, dtype=np.float32)
    rgb = np.zeros((*shape, 3), dtype=np.float32)
    rgb[..., 1] = 0.5
    support = np.ones(shape, dtype=np.uint8)
    gt_valid = np.ones(shape, dtype=np.uint8)
    coverage = np.full(shape, 2, dtype=np.int8)
    support[0, 0] = 0
    gt_valid[0, 0] = 0
    coverage[0, 0] = -1
    outputs: dict[str, np.ndarray] = {
        "pano_rgb": rgb,
        "gt_range": depth,
        "gt_valid": gt_valid,
        "coverage": coverage,
        "support": support,
        "regular_coverage": np.ones(shape, dtype=np.uint8),
        "regular_plus_tangent14_coverage": np.ones(shape, dtype=np.uint8),
    }
    for name in (
        "regular_only_raw",
        "regular_plus_tangent14_raw",
        "universal_scale",
        "bim_direct",
    ):
        outputs[name] = depth.copy()
        outputs[f"{name}_absrel"] = np.zeros(shape, dtype=np.float32)
    outputs["joint_minus_regular_signed_absrel"] = np.zeros(shape, dtype=np.float32)
    return outputs


def _selection() -> exporter.Selection:
    pair = _combined_pair("station_1", 0.02)
    return exporter.Selection(
        role="A_median_gain",
        station_id="station_1",
        room="office_1",
        gain_abs_rel=0.02,
        reference_abs_rel=0.25,
        candidate_abs_rel=0.23,
        rule={"gain_definition": "reference minus candidate"},
        reference_csv_row=pair[0],
        candidate_csv_row=pair[1],
    )


def test_title_free_erp_panels_and_station_bundle_follow_display_contract(
    tmp_path: Path,
) -> None:
    arrays = _panel_arrays()
    panels = exporter.panel_images(arrays)
    assert set(panels) == set(exporter.PANEL_NAMES)
    assert "learned_refined" not in panels
    assert "learned_refined_absrel" not in panels
    assert panels["gt_range"].shape == (512, 1024, 3)
    assert panels["gt_range"].dtype == np.uint8
    assert panels["gt_range"][0, 0].tolist() == exporter.INVALID_RGB.tolist()

    metrics = {
        "signed_map_definition": "joint minus regular; negative is better",
        "formal_csv_reproduced": True,
    }
    station_dir = tmp_path / "A_median_gain__office_1__station_1"
    station_dir.mkdir()
    (station_dir / "arrays.npz").write_bytes(b"retired-process-array")
    for retired_name in exporter.RETIRED_PANEL_NAMES:
        (station_dir / f"{retired_name}.png").write_bytes(b"retired")
    exported = exporter.export_station_assets(
        output_dir=tmp_path,
        selection=_selection(),
        arrays=arrays,
        metrics=metrics,
        source_identity={"synthetic": True},
        shared_provenance={"source_split": "val"},
        colorbars={},
    )
    station_dir = Path(exported["directory"])
    assert len(list(station_dir.glob("*.png"))) == len(exporter.PANEL_NAMES)
    assert all(not (station_dir / f"{name}.png").exists() for name in exporter.RETIRED_PANEL_NAMES)
    image = cv2.imread(str(station_dir / "bim_direct.png"), cv2.IMREAD_COLOR)
    assert image is not None and image.shape == (512, 1024, 3)
    assert not (station_dir / "arrays.npz").exists()
    manifest = json.loads((station_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"]["role"] == "A_median_gain"
    assert len(manifest["artifacts"]["panels"]) == len(exporter.PANEL_NAMES)
    assert "arrays" not in manifest["artifacts"]
    assert manifest["array_export"]["status"] == "not_written"


def test_shared_colorbars_have_png_svg_and_pdf_variants(tmp_path: Path) -> None:
    outputs = exporter.write_shared_colorbars(tmp_path)
    assert set(outputs) == {"depth", "absrel", "signed_absrel"}
    for spec in outputs.values():
        assert set(spec["artifacts"]) == {"png", "svg", "pdf"}
        assert all(Path(item["path"]).is_file() for item in spec["artifacts"].values())


def test_runtime_initialization_is_strictly_training_free_after_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "accepted.pt"
    tangent = tmp_path / "val_full.json"
    for path, content in ((config, b"cfg"), (tangent, b"manifest")):
        path.write_bytes(content)
    evaluator_path = Path(exporter.evaluator.__file__).resolve()
    project_root = evaluator_path.parents[2]
    shared_geometry_paths = {
        "stanford_pano": project_root / "src/bim_priorda3/data/stanford_pano.py",
        "pano_tangent": project_root / "src/bim_priorda3/data/pano_tangent.py",
    }
    provenance = {
        "seed": 42,
        "config": {"path": str(config), "sha256": exporter.file_sha256(config)},
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": "f" * 64,
        },
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": exporter.file_sha256(evaluator_path),
        },
        "shared_geometry_code_identity": {
            name: {"path": str(path), "sha256": exporter.file_sha256(path)}
            for name, path in shared_geometry_paths.items()
        },
        "route_p_tangent_inputs": {
            "manifest": {"path": str(tangent), "sha256": exporter.file_sha256(tangent)}
        },
        "geometry": {
            "pano_shape": [512, 1024],
            "centrality_power": 4.0,
            "confidence_floor": 0.05,
        },
        "fusion": {
            "huber_log_delta": 0.08,
            "consistency_log_threshold": 0.25,
            "photometric_sigma": 0.12,
            "overlap_scale_synchronization": {
                "min_overlap": 256,
                "pair_max_deterministic_samples": 4096,
                "huber_log_delta": 0.08,
                "l2": 1e-6,
                "max_abs_log_offset": 0.5,
            },
            "station_bim_scale": {"samples_per_view_cap": 50_000},
        },
    }
    artifacts = exporter.FormalArtifacts(
        results_dir=tmp_path,
        summary={},
        provenance=provenance,
        regular_rows=(),
        route_r_rows=(),
        identities={},
        fstar="joint_huber",
        fstar_selection={},
    )
    selection = _selection()
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            min_depth=0.2,
            max_depth=5.0,
            stanford_area_root="area",
        )
    )
    dataset = SimpleNamespace(
        records=[{"camera_uuid": selection.station_id, "region": selection.room}]
    )
    station = SimpleNamespace(camera_uuid=selection.station_id, room=selection.room)
    monkeypatch.setattr(exporter.evaluator, "seed_everything", lambda _seed: None)
    monkeypatch.setattr(exporter.evaluator, "load_config", lambda _path: cfg)
    monkeypatch.setattr(exporter.evaluator, "validate_universal_scale_protocol", lambda _cfg: {})
    monkeypatch.setattr(exporter.evaluator, "BIMDepthDataset", lambda *_a, **_k: dataset)
    monkeypatch.setattr(exporter.evaluator, "_device_from_arg", lambda _value: torch.device("cpu"))
    monkeypatch.setattr(
        exporter.evaluator,
        "_load_model",
        lambda *_a, **_k: pytest.fail("training-free exporter must not load a checkpoint"),
    )
    monkeypatch.setattr(exporter.evaluator, "resolve_project_path", lambda *_a: tmp_path)
    monkeypatch.setattr(exporter.evaluator, "discover_stanford_panoramas", lambda _root: [station])
    monkeypatch.setattr(
        exporter.evaluator,
        "_validate_tangent_manifest",
        lambda *_a, **_k: SimpleNamespace(stations={selection.station_id: {}}),
    )
    args = argparse.Namespace(
        config=None,
        tangent_manifest=None,
        device="cpu",
        batch_size=8,
    )

    runtime = exporter.prepare_runtime(artifacts, [selection], args)

    assert not checkpoint.exists()
    assert runtime.model is None
    assert runtime.runtime_identity["checkpoint_load_count"] == 0
    assert runtime.runtime_identity["checkpoint"]["runtime_status"] == "not_loaded_or_hashed"
    assert set(runtime.runtime_identity["shared_geometry_code_identity"]) == {
        "stanford_pano",
        "pano_tangent",
    }
    assert runtime.stations[selection.station_id] is station
