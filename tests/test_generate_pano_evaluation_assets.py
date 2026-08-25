from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest

import scripts.analysis.generate_pano_evaluation_assets as assets


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric(
    source_set: str,
    fusion_method: str,
    support_scope: str,
    value: float,
) -> dict[str, Any]:
    return {
        "source_set": source_set,
        "fusion_method": fusion_method,
        "support_scope": support_scope,
        "depth_method": "raw_da3",
        "station_macro": {"spherical_abs_rel": value},
    }


def _tangent_metric(variant: str, value: float) -> dict[str, Any]:
    return {
        "route_variant": variant,
        "fusion_method": assets.FSTAR,
        "support_scope": "common_tangent6_tangent14",
        "depth_method": "raw_da3",
        "station_macro": {"spherical_abs_rel": value},
    }


def _contrast(
    *,
    kind: str,
    candidate: dict[str, str] | str,
    reference: dict[str, str] | str,
    reference_value: float,
    candidate_value: float,
    interval: tuple[float, float],
) -> dict[str, Any]:
    difference = candidate_value - reference_value
    return {
        "kind": kind,
        "candidate": candidate,
        "reference": reference,
        "primary_abs_rel_reference": reference_value,
        "primary_abs_rel_candidate": candidate_value,
        "room_cluster_paired_bootstrap_primary_abs_rel": {
            "mean_difference": difference,
            "confidence_interval_95": list(interval),
            "bootstrap_repetitions": 100,
            "room_count": 2,
            "station_count": 4,
        },
    }


def _write_split(root: Path, split: str, offset: float) -> None:
    directory = root / f"pano_{split}"
    directory.mkdir(parents=True)
    sources = ("regular_only", "regular_plus_tangent6", "regular_plus_tangent14")
    base_by_fusion = {
        "joint_weighted_log": 0.28 + offset,
        "joint_huber": 0.30 + offset,
        "joint_synchronized_huber": 0.33 + offset,
    }
    reductions = {
        "regular_only": 0.0,
        "regular_plus_tangent6": 0.04,
        "regular_plus_tangent14": 0.06,
    }
    route_metrics = [
        _metric(
            source,
            fusion,
            "common_regular",
            base - reductions[source],
        )
        for fusion, base in base_by_fusion.items()
        for source in sources
    ]

    regular_rows: list[dict[str, Any]] = []
    room_ids = ("room_a", "room_a", "room_b", "room_b")
    coverages = {
        "regular_only": 0.65,
        "regular_plus_tangent6": 0.98,
        "regular_plus_tangent14": 0.995,
    }
    for station_index, room in enumerate(room_ids):
        for fusion, base in base_by_fusion.items():
            for source in sources:
                value = base - reductions[source] + 0.002 * station_index
                for scope in ("common_regular", "native_union"):
                    regular_rows.append(
                        {
                            "station_id": f"station_{station_index}",
                            "room": room,
                            "source_set": source,
                            "support_scope": scope,
                            "fusion_method": fusion,
                            "depth_method": "raw_da3",
                            "spherical_abs_rel": value,
                            "solid_angle_coverage_fraction": (
                                0.65 if scope == "common_regular" else coverages[source]
                            ),
                        }
                    )
    _write_csv(directory / "regular_pano_joint_per_station.csv", regular_rows)
    dummy = [{"station_id": "station_0", "value": "1"}]
    for filename in assets.CSV_RECEIPTS:
        if filename != "regular_pano_joint_per_station.csv":
            _write_csv(directory / filename, dummy)

    common = "common_regular"
    regular = {
        "fusion_method": assets.FSTAR,
        "source_set": "regular_only",
        "support_scope": common,
    }
    plus6 = {
        "fusion_method": assets.FSTAR,
        "source_set": "regular_plus_tangent6",
        "support_scope": common,
    }
    plus14 = {
        "fusion_method": assets.FSTAR,
        "source_set": "regular_plus_tangent14",
        "support_scope": common,
    }
    combined_contrasts = [
        _contrast(
            kind="regular_plus_pano_tangent_over_regular_only_raw",
            candidate=plus14,
            reference=regular,
            reference_value=0.30 + offset,
            candidate_value=0.24 + offset,
            interval=(-0.08, -0.04),
        ),
        _contrast(
            kind="regular_plus_tangent14_over_regular_plus_tangent6",
            candidate=plus14,
            reference=plus6,
            reference_value=0.26 + offset,
            candidate_value=0.24 + offset,
            interval=(-0.03, -0.01),
        ),
    ]
    tangent6 = {
        "fusion_method": assets.FSTAR,
        "route_variant": "tangent6",
        "support_scope": "common_tangent6_tangent14",
    }
    tangent14 = {
        "fusion_method": assets.FSTAR,
        "route_variant": "tangent14",
        "support_scope": "common_tangent6_tangent14",
    }
    tangent_contrast = _contrast(
        kind="tangent14_over_tangent6_view_count_ablation",
        candidate=tangent14,
        reference=tangent6,
        reference_value=0.40 + offset,
        candidate_value=0.36 + offset,
        interval=(-0.06, -0.02),
    )
    strict_contrasts = []
    strict_values = {
        "raw_da3": (0.16, 0.26, (0.08, 0.12)),
        "universal_scale": (0.12, 0.11, (-0.03, 0.01)),
        "bim_direct": (0.125, 0.115, (-0.035, 0.015)),
    }
    for method, (reference_value, candidate_value, interval) in strict_values.items():
        strict_contrasts.append(
            _contrast(
                kind="pano_joint_over_strict_single_frame",
                candidate=f"{method}/{assets.FSTAR}",
                reference=f"{method}/strict_single_frame",
                reference_value=reference_value + offset,
                candidate_value=candidate_value + offset,
                interval=interval,
            )
        )
    primary = {
        "raw_da3": {"abs_rel": 0.30 + offset},
        "universal_scale": {"abs_rel": 0.11 + offset},
        "bim_direct": {"abs_rel": 0.112 + offset},
        # It exists in the evaluator output but must never be consumed/plotted.
        "learned_refined": {"abs_rel": 0.09 + offset},
    }
    top_contrasts = [
        _contrast(
            kind="bim_enhancement_over_raw_da3",
            candidate=f"universal_scale/{assets.FSTAR}",
            reference=f"raw_da3/{assets.FSTAR}",
            reference_value=0.30 + offset,
            candidate_value=0.11 + offset,
            interval=(-0.21, -0.17),
        ),
        _contrast(
            kind="local_bim_correction_over_scale_only",
            candidate=f"bim_direct/{assets.FSTAR}",
            reference=f"universal_scale/{assets.FSTAR}",
            reference_value=0.11 + offset,
            candidate_value=0.112 + offset,
            interval=(-0.002, 0.006),
        ),
    ]
    artifacts = {}
    provenance: dict[str, Any] = {
        "schema_version": assets.FORMAL_SCHEMA_VERSION,
        "protocol": assets.FORMAL_PROTOCOL,
        "split": split,
        "formal_protocol_eligible": True,
        "test_access_explicitly_authorized": split == "test",
    }
    for filename, receipt_key in assets.CSV_RECEIPTS.items():
        digest = _sha256(directory / filename)
        artifacts[f"{receipt_key}_sha256"] = digest
        provenance[receipt_key] = {"path": f"/formal/{split}/{filename}", "sha256": digest}
    summary = {
        "schema_version": assets.FORMAL_SCHEMA_VERSION,
        "protocol": assets.FORMAL_PROTOCOL,
        "split": split,
        "formal_protocol_eligible": True,
        "station_count": 4,
        "artifacts": artifacts,
        "route_p_regular_plus_pano_raw": {
            "metrics": route_metrics,
            "contrasts": combined_contrasts,
        },
        "route_p_tangent_only": {
            "metrics": [
                _tangent_metric("tangent6", 0.40 + offset),
                _tangent_metric("tangent14", 0.36 + offset),
            ],
            "contrasts": [tangent_contrast],
        },
        "strict_single_frame_evaluation": {"contrasts": strict_contrasts},
        "primary_metrics": {assets.FSTAR: primary},
        "contrasts": top_contrasts,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (directory / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


def _formal_fixture(root: Path) -> Path:
    results = root / "results" / "stanford_area1"
    _write_split(results, "val", 0.0)
    _write_split(results, "test", 0.01)
    _write_exploratory_validation(results)
    return results


def _write_exploratory_validation(results: Path) -> None:
    directory = results / assets.EXPLORATORY_DIRECTORY
    directory.mkdir(parents=True)
    provenance = {
        "schema_version": assets.EXPLORATORY_SCHEMA_VERSION,
        "protocol": assets.EXPLORATORY_PROTOCOL,
        "split": "val",
        "status": "exploratory_validation_only",
        "publication_status": assets.EXPLORATORY_PUBLICATION_STATUS,
        "test_access": {
            "authorized": False,
            "test_csv_opened": False,
            "test_raw_files_opened": False,
        },
    }
    provenance_path = directory / "provenance.json"
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    values = {
        "strict_single": (0.12, 0.11),
        "strict_single_plus_tangent6": (0.16, 0.99),
        "strict_single_plus_tangent14": (0.31, 0.998),
    }
    contrasts = []
    for method, interval in (
        ("strict_single_plus_tangent6", (0.02, 0.06)),
        ("strict_single_plus_tangent14", (0.14, 0.24)),
    ):
        contrast = _contrast(
            kind="strict_single_plus_pano_tangent_over_strict_single",
            candidate={"method": method},
            reference={"method": "strict_single"},
            reference_value=values["strict_single"][0],
            candidate_value=values[method][0],
            interval=interval,
        )
        contrast["room_cluster_paired_bootstrap_primary_abs_rel"]["bootstrap_repetitions"] = 10000
        contrasts.append(contrast)
    summary = {
        "schema_version": assets.EXPLORATORY_SCHEMA_VERSION,
        "protocol": assets.EXPLORATORY_PROTOCOL,
        "split": "val",
        "status": "exploratory_validation_only",
        "publication_status": assets.EXPLORATORY_PUBLICATION_STATUS,
        "formal_v3_result": False,
        "test_data_or_csv_accessed": False,
        "station_count": 30,
        "room_count": 2,
        "pano_only_excluded_count": 2,
        "methods": list(values),
        "method_scope": {
            "training_free_only": True,
            "learned": False,
            "BIM": False,
            "GT_scale": False,
            "checkpoint": False,
            "fusion_method": assets.FSTAR,
            "fusion_parameters_frozen": True,
        },
        "fixed_support_audit": {"same_count_for_all_methods_per_station": True},
        "artifacts": {"provenance_sha256": _sha256(provenance_path)},
        "primary_spherical_abs_rel": {
            method: quality for method, (quality, _coverage) in values.items()
        },
        "native_union_mean_solid_angle_coverage": {
            method: coverage for method, (_quality, coverage) in values.items()
        },
        "metrics": {
            method: {
                "station_macro": {"spherical_abs_rel": quality},
                "native_union_coverage": {"mean_solid_angle_fraction": coverage},
            }
            for method, (quality, coverage) in values.items()
        },
        "contrasts": contrasts,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_formal_loader_checks_csv_receipts_and_test_authorization(tmp_path: Path) -> None:
    results = _formal_fixture(tmp_path)
    registry = assets.SourceRegistry(tmp_path)
    loaded = assets.load_formal_split(results, "test", registry)
    assert loaded.name == "test"
    assert len(registry.records) == 7

    with (results / "pano_test" / "per_room.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        assets.load_formal_split(results, "test", assets.SourceRegistry(tmp_path))


def test_primary_payload_separates_common_quality_from_union_coverage(
    tmp_path: Path,
) -> None:
    results = _formal_fixture(tmp_path)
    test = assets.load_formal_split(results, "test", assets.SourceRegistry(tmp_path))

    payload = assets.main_test_payload(test)

    assert payload["quality_support"] == "common_regular"
    assert payload["coverage_support"] == "native_union"
    assert [row["spherical_abs_rel"] for row in payload["quality"]] == pytest.approx(
        [0.31, 0.27, 0.25]
    )
    assert [
        row["mean_solid_angle_coverage_fraction"] for row in payload["coverage"]
    ] == pytest.approx([0.65, 0.98, 0.995])


def test_room_points_are_recomputed_from_station_csv(tmp_path: Path) -> None:
    results = _formal_fixture(tmp_path)
    registry = assets.SourceRegistry(tmp_path)
    val = assets.load_formal_split(results, "val", registry)
    test = assets.load_formal_split(results, "test", registry)

    payload = assets.room_pair_payload((val, test))
    val_points = payload["splits"]["val"]["room_points_recomputed_from_csv"]

    assert [row["room"] for row in val_points] == ["room_a", "room_b"]
    assert val_points[0]["station_count"] == 2
    assert val_points[0]["regular_only"] == pytest.approx(0.301)
    assert val_points[0]["regular_plus_tangent14"] == pytest.approx(0.241)
    assert val_points[0]["candidate_minus_reference"] == pytest.approx(-0.06)


def test_generate_assets_is_traceable_training_free_and_font_independent(
    tmp_path: Path,
) -> None:
    results = _formal_fixture(tmp_path)
    output = tmp_path / "assets"

    manifest = assets.generate_assets(
        results_root=results,
        output_root=output,
        formats=("png",),
        dpi=45,
    )

    assert len(manifest["figures"]) == 10
    assert len(manifest["sources"]) == 16
    assert manifest["method_scope"]["training_free_only"] is True
    assert manifest["method_scope"]["learned_refinement_plotted"] is False
    assert manifest["method_scope"]["included_depth_methods"] == [
        "raw_da3",
        "universal_scale",
        "bim_direct",
    ]
    assert all(len(source["sha256"]) == 64 for source in manifest["sources"])
    assert all((output / figure["files"][0]["path"]).is_file() for figure in manifest["figures"])
    chain = next(
        figure for figure in manifest["figures"] if figure["id"] == "deterministic_bim_chain"
    )
    assert [row["depth_method"] for row in chain["numeric_payload"]["splits"]["test"]["chain"]] == [
        "raw_da3",
        "universal_scale",
        "bim_direct",
    ]
    validation = next(
        figure
        for figure in manifest["figures"]
        if figure["id"] == "strict_single_plus_pano_validation"
    )["numeric_payload"]
    assert validation["status"] == "exploratory_validation_only"
    assert validation["formal_v3_result"] is False
    assert [row["spherical_abs_rel"] for row in validation["methods"]] == pytest.approx(
        [0.12, 0.16, 0.31]
    )
    assert [
        row["native_union_solid_angle_coverage_fraction"] for row in validation["methods"]
    ] == pytest.approx([0.11, 0.99, 0.998])
    main_payload = next(
        figure for figure in manifest["figures"] if figure["id"] == "test_main_pano_gain"
    )["numeric_payload"]
    assert main_payload["quality"][0]["label"] == "Multi-regular only"
    assert main_payload["coverage"][0]["label"] == "Multi-regular only"

    figure, axis = plt.subplots(figsize=(2, 1))
    axis.plot([0, 1], [0, 1])
    files = assets._save_figure(figure, output, "font_test", ("svg",), 40)
    svg = (output / files[0]["path"]).read_text(encoding="utf-8")
    assert "<text" not in svg  # glyph paths avoid an external font dependency


def test_replay_mode_does_not_load_formal_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = _formal_fixture(tmp_path)
    output = tmp_path / "assets"
    assets.generate_assets(results_root=results, output_root=output, formats=("png",), dpi=40)

    def _forbid_formal_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("replay mode reopened a formal split")

    monkeypatch.setattr(assets, "load_formal_split", _forbid_formal_load)
    manifest = assets.regenerate_assets_without_formal_reads(
        results_root=results,
        output_root=output,
        replay_manifest=output / "manifest.json",
        formats=("png",),
        dpi=40,
    )

    assert manifest["generation_mode"].startswith("manifest_payload_replay")
    assert manifest["formal_sources_reopened"] is False
    assert manifest["test_csv_reopened"] is False
    assert len(manifest["figures"]) == 10
    assert len(manifest["sources"]) == 16
    serialized = json.dumps(manifest)
    assert '"label": "Regular only"' not in serialized
