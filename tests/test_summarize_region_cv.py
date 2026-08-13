from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from bim_priorda3.region_cv import build_region_fold_plans
from scripts.pipelines.summarize_region_cv import main

REGIONS = ("A", "B", "C")
SEEDS = (11, 22)
BASELINES = {
    "A": {"base": 0.4, "global_scale": 0.25, "previous_scale_local": 0.2},
    "B": {"base": 0.5, "global_scale": 0.35, "previous_scale_local": 0.3},
    "C": {"base": 0.3, "global_scale": 0.15, "previous_scale_local": 0.1},
}
REFINED = {
    "A": {11: 0.18, 22: 0.16},
    "B": {11: 0.32, 22: 0.30},
    "C": {11: 0.08, 22: 0.06},
}


def _protocol(protocol_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "regions": list(REGIONS),
        "validation_map": {"A": "B", "B": "C", "C": "A"},
        "seeds": list(SEEDS),
    }


def _metrics(abs_rel: float) -> dict[str, float | int]:
    return {
        "abs_rel": abs_rel,
        "rmse": abs_rel * 2,
        "mae": abs_rel * 1.5,
        "delta1": 1.0 - abs_rel,
        "delta2": 1.0 - abs_rel / 2,
        "delta3": 1.0 - abs_rel / 3,
        "count": 100,
    }


def _write_evaluation(
    run_dir: Path,
    region: str,
    seed: int,
    *,
    direct_offset: float = 0.0,
    summary_region: str | None = None,
    wrong_checkpoint: bool = False,
    checkpoint_name: str = "accepted.pt",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / checkpoint_name
    checkpoint.write_bytes(f"checkpoint-{region}-{seed}".encode())
    evaluation_dir = run_dir / "evaluation_test"
    evaluation_dir.mkdir(exist_ok=True)
    evaluated_checkpoint = checkpoint
    if wrong_checkpoint:
        evaluated_checkpoint = run_dir / "different.pt"
        evaluated_checkpoint.write_bytes(b"different checkpoint")

    methods = {
        method: _metrics(value + (direct_offset if method == "previous_scale_local" else 0.0))
        for method, value in BASELINES[region].items()
    }
    methods["refined"] = _metrics(REFINED[region][seed])
    summary = {
        "checkpoint": str(evaluated_checkpoint.resolve()),
        "split": "test",
        "regions": [summary_region or region],
        "overall": methods,
    }
    (evaluation_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    row = {"sample_id": f"{region}/000000", "valid_pixels": 100}
    for method, metrics in methods.items():
        for metric, value in metrics.items():
            row[f"{method}_{metric}"] = value
    with (evaluation_dir / "per_frame.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _make_protocol_tree(
    tmp_path: Path,
    *,
    checkpoint_selection: dict[str, str] | None = None,
) -> Path:
    protocol_id = "test_protocol"
    root = tmp_path / protocol_id
    root.mkdir()
    protocol = _protocol(protocol_id)
    if checkpoint_selection is not None:
        protocol["checkpoint_selection"] = checkpoint_selection
    (root / "protocol.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )
    fold_plans = build_region_fold_plans(protocol)
    plan = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "regions": list(REGIONS),
        "validation_map": protocol["validation_map"],
        "seeds": list(SEEDS),
        "folds": [fold.to_dict() for fold in fold_plans],
    }
    if checkpoint_selection is not None:
        plan["checkpoint_selection"] = checkpoint_selection
    for fold, fold_payload in zip(fold_plans, plan["folds"]):
        region = fold.test_regions[0]
        if checkpoint_selection is not None:
            fold_payload["runs"] = []
        for seed in SEEDS:
            run_dir = root / "folds" / fold.fold_id / f"seed_{seed}"
            checkpoint_name = (
                f"{checkpoint_selection['final']}.pt"
                if checkpoint_selection is not None
                else "accepted.pt"
            )
            _write_evaluation(
                run_dir,
                region,
                seed,
                checkpoint_name=checkpoint_name,
            )
            if checkpoint_selection is not None:
                checkpoint = run_dir / checkpoint_name
                checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                fold_payload["runs"].append(
                    {
                        "seed": seed,
                        "checkpoints": {
                            "final_selected": str(checkpoint.resolve()),
                            f"final_{checkpoint_selection['final']}": str(checkpoint.resolve()),
                        },
                        "checkpoint_sha256": {
                            "final_selected": checkpoint_sha,
                            f"final_{checkpoint_selection['final']}": (checkpoint_sha),
                        },
                        "evaluation_summary": str(
                            (run_dir / "evaluation_test" / "summary.json").resolve()
                        ),
                    }
                )
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return root


def test_complete_summary_averages_seeds_before_regions(tmp_path: Path) -> None:
    root = _make_protocol_tree(tmp_path)
    assert main([str(root)]) == 0

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["planned_runs"] == 6
    assert summary["successful_runs"] == 6
    assert summary["regions_summarized"] == 3
    assert summary["region_metrics"]["A"]["refined"]["abs_rel"] == pytest.approx(0.17)
    assert summary["region_macro"]["direct_bim"]["metrics"]["abs_rel"]["mean"] == pytest.approx(0.2)
    assert summary["region_macro"]["refined"]["metrics"]["abs_rel"]["mean"] == pytest.approx(
        (0.17 + 0.31 + 0.07) / 3
    )
    comparison = summary["comparison_to_direct_bim"]
    assert comparison["regions_won"] == 2
    assert comparison["won_regions"] == ["A", "C"]
    assert comparison["worst_region"]["region"] == "B"
    assert comparison["relative_improvement"] == pytest.approx(1 / 12)

    with (root / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 24
    with (root / "region_metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12
    report = (root / "REPORT.md").read_text(encoding="utf-8")
    assert "Status: **complete**" in report
    assert "Regions won: 2/3" in report


@pytest.mark.parametrize(
    "missing_relative_path",
    [
        Path("accepted.pt"),
        Path("evaluation_test/summary.json"),
        Path("evaluation_test/per_frame.csv"),
    ],
)
def test_missing_run_is_reported_and_requires_allow_incomplete(
    tmp_path: Path,
    missing_relative_path: Path,
) -> None:
    root = _make_protocol_tree(tmp_path)
    missing = root / "folds" / "fold_00_A" / "seed_22" / missing_relative_path
    missing.unlink()

    assert main([str(root)]) == 2
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["failed_runs"] == 1
    assert summary["successful_runs"] == 5
    assert summary["failures"][0]["errors"][0]["code"] == "missing_artifact"
    assert summary["region_metrics"]["A"]["_successful_seed_count"] == 1
    assert main([str(root), "--allow-incomplete"]) == 0


def test_plan_run_seeds_must_exactly_match_the_fold(tmp_path: Path) -> None:
    root = _make_protocol_tree(tmp_path)
    plan_path = root / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["folds"][0]["runs"] = [{"seed": 11}, {"seed": 99}]
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    assert main([str(root), "--allow-incomplete"]) == 2


def test_baseline_mismatch_invalidates_the_test_fold(tmp_path: Path) -> None:
    root = _make_protocol_tree(tmp_path)
    run_dir = root / "folds" / "fold_00_A" / "seed_22"
    _write_evaluation(
        run_dir,
        "A",
        22,
        direct_offset=0.01,
    )

    assert main([str(root), "--allow-incomplete"]) == 0
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"
    assert summary["failed_runs"] == 2
    assert summary["regions_summarized"] == 2
    assert {error["code"] for failure in summary["failures"] for error in failure["errors"]} == {
        "baseline_inconsistent_across_seeds"
    }


@pytest.mark.parametrize(
    ("kwargs", "expected_fragment"),
    [
        ({"summary_region": "B"}, "planned test region"),
        ({"wrong_checkpoint": True}, "checkpoint SHA-256 differs"),
    ],
)
def test_test_region_and_checkpoint_sha_are_strictly_checked(
    tmp_path: Path,
    kwargs: dict[str, object],
    expected_fragment: str,
) -> None:
    root = _make_protocol_tree(tmp_path)
    run_dir = root / "folds" / "fold_00_A" / "seed_11"
    _write_evaluation(run_dir, "A", 11, **kwargs)

    assert main([str(root), "--allow-incomplete"]) == 0
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    failure = next(failure for failure in summary["failures"] if failure["seed"] == 11)
    assert failure["errors"][0]["code"] == "invalid_evaluation"
    assert expected_fragment in failure["errors"][0]["message"]


def test_checkpoint_sha_is_recorded_in_fold_metrics(tmp_path: Path) -> None:
    root = _make_protocol_tree(tmp_path)
    assert main([str(root)]) == 0
    checkpoint = root / "folds" / "fold_00_A" / "seed_11" / "accepted.pt"
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with (root / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(
        row
        for row in rows
        if row["fold_id"] == "fold_00_A" and row["seed"] == "11" and row["method"] == "refined"
    )
    assert row["checkpoint_sha256"] == expected


def test_best_checkpoint_selection_is_loaded_from_selected_plan_paths(
    tmp_path: Path,
) -> None:
    selection = {"pretrain": "best", "final": "best"}
    root = _make_protocol_tree(
        tmp_path,
        checkpoint_selection=selection,
    )

    assert main([str(root)]) == 0
    assert not next((root / "folds").glob("*/seed_*/accepted.pt"), None)

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["checkpoint_selection"] == selection

    with (root / "fold_metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["checkpoint_selection"] for row in rows} == {"best"}
