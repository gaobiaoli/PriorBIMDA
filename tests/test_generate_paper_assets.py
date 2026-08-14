from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest

import scripts.analysis.generate_paper_assets as paper_assets


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _pixel(abs_rel: float, count: int = 100) -> dict[str, Any]:
    return {"pixel_micro": {"abs_rel": abs_rel, "count": count}}


def _history(anchor: float, values: list[float]) -> list[dict[str, Any]]:
    return [
        {
            "epoch": epoch,
            "val_refined_abs_rel": value,
            "val_anchor_abs_rel": anchor,
            "accepted": True,
        }
        for epoch, value in enumerate(values)
    ]


def _make_fixture(root: Path) -> tuple[Path, Path]:
    results = root / "results"
    provenance = root / "data" / "provenance" / "scale.json"
    metrics = {
        "depth_protocol_m": [0.2, 5.0],
        "slabim": {
            "methods": {
                "raw_da3": {"abs_rel": 0.20},
                "universal_global_scale": {"abs_rel": 0.09},
                "universal_bim_direct": {"abs_rel": 0.08},
                "learned_refiner": {"abs_rel": 0.07},
            }
        },
        "stanford_area1": {
            "validation": {
                "learned_abs_rel": 0.0700,
                "universal_bim_direct_abs_rel": 0.0900,
            }
        },
    }
    _write_json(results / "metrics.json", metrics)

    aggregates: dict[str, Any] = {}
    subset_values = {
        "all": (0.30, 0.078, 0.068),
        "furniture": (0.29, 0.089, 0.086),
        "bim_foreground_conflict": (0.30, 0.140, 0.142),
    }
    for subset, (raw, direct, refined) in subset_values.items():
        aggregates[subset] = {
            "raw_da3": _pixel(raw),
            "robust_bim_direct": _pixel(direct),
            "refined": _pixel(refined),
        }
    aggregates["all"]["robust_global_scale"] = _pixel(0.077)
    per_room = {
        "office_1": {
            "all": {
                "robust_bim_direct": {"abs_rel": 0.080},
                "refined": {"abs_rel": 0.065},
            }
        },
        "office_2": {
            "all": {
                "robust_bim_direct": {"abs_rel": 0.100},
                "refined": {"abs_rel": 0.105},
            }
        },
    }
    bootstrap_specs = {
        "all": (-0.005, [-0.020, -0.001], 2, 0.5),
        "furniture": (-0.003, [-0.010, -0.001], 2, 1.0),
        "bim_foreground_conflict": (0.004, [-0.005, 0.020], 2, 0.5),
    }
    paired = {}
    for subset, (mean, interval, rooms, win_fraction) in bootstrap_specs.items():
        paired[subset] = {
            "abs_rel": {
                "mean_difference": mean,
                "confidence_interval_95": interval,
                "rooms": rooms,
                "bootstrap_repetitions": 100,
                "seed": 42,
                "candidate_better_room_fraction": win_fraction,
            }
        }
    _write_json(
        results / "stanford_area1" / "test_summary.json",
        {
            "aggregates": aggregates,
            "per_room": per_room,
            "paired_room_bootstrap": paired,
        },
    )

    _write_json(results / "slabim" / "history.json", _history(0.08, [0.075, 0.070]))
    _write_json(
        results / "stanford_area1" / "history.json",
        _history(0.09, [0.082, 0.070]),
    )

    candidates = []
    objectives = {
        (0.2, 0.05): (0.105, 0.140),
        (0.2, 0.1): (0.108, 0.135),
        (float("inf"), 0.05): (0.093, 0.129),
        (float("inf"), 0.1): (0.095, 0.131),
    }
    for (q10, q25), (all_abs_rel, furniture_abs_rel) in objectives.items():
        candidates.append(
            {
                "candidate_id": f"q10={q10},q25={q25}",
                "q10_log_cap": q10,
                "q25_log_cap": q25,
                "selection_objective": {
                    "room_macro_abs_rel": all_abs_rel,
                    "furniture_room_macro_abs_rel": furniture_abs_rel,
                },
                "scale_summary": {
                    "frames": 10,
                    "q10_cap_triggered_frames": 0 if q10 == float("inf") else 2,
                    "q25_cap_triggered_frames": 3,
                },
            }
        )
    _write_json(
        provenance,
        {
            "candidate_results": candidates,
            "final_selection": {
                "canonical_scale_estimator": {
                    "q10_log_cap": float("inf"),
                    "q25_log_cap": 0.05,
                },
                "selection_scope": "train only",
            },
            "split_isolation": {
                "validation_samples_opened": 0,
                "test_samples_opened": 0,
            },
        },
    )
    return results, provenance


def test_pareto_frontier_indices_for_minimization() -> None:
    points = [(0.10, 0.20), (0.11, 0.19), (0.12, 0.22), (0.09, 0.24)]

    assert paper_assets.pareto_frontier_indices(points) == [3, 0, 1]


def test_generate_assets_is_headless_traceable_and_marks_conflict(tmp_path: Path) -> None:
    results, provenance = _make_fixture(tmp_path)
    output = tmp_path / "paper_assets"

    manifest = paper_assets.generate_assets(
        results_root=results,
        scale_selection_path=provenance,
        output_root=output,
        formats=("png",),
        dpi=45,
    )

    assert len(manifest["figures"]) == 9
    assert len(manifest["sources"]) == 5
    assert {item["availability"] for item in manifest["figures"]} == {
        "existing_result",
        "design_candidate",
    }
    assert all(len(source["sha256"]) == 64 for source in manifest["sources"])
    assert all(len(figure["files"]) == 1 for figure in manifest["figures"])
    assert all((output / figure["files"][0]["path"]).is_file() for figure in manifest["figures"])
    main = next(item for item in manifest["figures"] if item["id"] == "main_blind_test_absrel")
    slabim_methods = [row["method"] for row in main["numeric_payload"]["SLABIM"]]
    assert slabim_methods[-1] == "Learned refiner"
    subsets = next(item for item in manifest["figures"] if item["id"] == "area1_subset_absrel")
    assert subsets["numeric_payload"]["conflict_refined_minus_direct_abs_rel"] == pytest.approx(
        0.002
    )
    sensitivity = next(
        item for item in manifest["figures"] if item["id"] == "area1_train_only_scale_heatmap"
    )
    assert sensitivity["numeric_payload"]["candidate_count"] == 4
    assert sensitivity["numeric_payload"]["split_isolation"] == {
        "validation_samples_opened": 0,
        "test_samples_opened": 0,
    }
    assert all(item["planned"] for item in manifest["planned_evaluations_not_plotted"])
    on_disk = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == 1


def test_save_figure_supports_png_svg_and_pdf(tmp_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(2, 1))
    axis.plot([0, 1], [0, 1])

    files = paper_assets._save_figure(
        figure,
        tmp_path,
        "tiny",
        paper_assets.DEFAULT_FORMATS,
        40,
    )

    assert [item["format"] for item in files] == ["png", "svg", "pdf"]
    assert all((tmp_path / item["path"]).is_file() for item in files)
    assert all(len(item["sha256"]) == 64 for item in files)
