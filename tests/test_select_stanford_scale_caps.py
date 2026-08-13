from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from bim_priorda3.baselines import (
    estimate_robust_bim_scale,
    resolve_scale_estimator_config,
)
from bim_priorda3.data.splits import resolve_annotation_splits
from scripts.data import select_stanford_scale_caps as selector


def _write_sample(
    path: Path,
    *,
    target_depth: float,
    preparation_fingerprint: str,
) -> None:
    shape = (12, 12)
    base = np.ones(shape, dtype=np.float32)
    bim = np.linspace(1.25, 3.75, num=shape[0] * shape[1], dtype=np.float32).reshape(shape)
    gt = np.full(shape, target_depth, dtype=np.float32)
    furniture = np.zeros(shape, dtype=np.uint8)
    furniture[:4] = 1
    non_structural = np.zeros(shape, dtype=np.uint8)
    non_structural[:7] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        base_depth=base,
        bim_depth=bim,
        gt_depth=gt,
        gt_valid=np.ones(shape, dtype=np.uint8),
        semantic_class=np.zeros(shape, dtype=np.uint8),
        furniture_mask=furniture,
        non_structural_mask=non_structural,
        preparation_fingerprint_sha256=np.asarray(preparation_fingerprint),
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, list[dict[str, object]]]:
    processed = tmp_path / "processed"
    samples = processed / "samples"
    definitions = (
        ("train_a/frame0", "train_a", "train", 1.65, True),
        ("train_b/frame0", "train_b", "train", 1.95, True),
        ("val_room/frame0", "val_room", "val", 1.80, False),
        ("test_room/frame0", "test_room", "test", 1.80, False),
    )
    records: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for index, (sample_id, room, split, target, materialize) in enumerate(definitions):
        sample_path = samples / room / "frame0.npz"
        preparation_fingerprint = f"{index + 1:064x}"
        if materialize:
            _write_sample(
                sample_path,
                target_depth=target,
                preparation_fingerprint=preparation_fingerprint,
            )
        records.append(
            {
                "id": sample_id,
                "region": room,
                "sample": str(sample_path),
                "sample_relative_to_processed": str(sample_path.relative_to(processed)),
                "preparation_fingerprint_sha256": preparation_fingerprint,
            }
        )
        annotations.append({"schema_version": 1, "id": sample_id, "split": split})
    manifest = processed / "manifest.jsonl"
    processed.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    annotation = tmp_path / "split.jsonl"
    annotation.write_text(
        "".join(json.dumps(record) + "\n" for record in annotations),
        encoding="utf-8",
    )
    resolution = resolve_annotation_splits(records, annotation)
    config = tmp_path / "selector.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "processed_root": str(processed),
                    "split_annotation": str(annotation),
                    "split_annotation_sha256": resolution.provenance["annotation_raw_sha256"],
                    "split_fingerprint_sha256": resolution.provenance["fingerprint_sha256"],
                    "train_regions": [],
                    "val_regions": [],
                    "test_regions": [],
                    "record_stride_by_region": {},
                    "min_depth": 0.2,
                    "max_depth": 5.0,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config, annotation, records


def test_selector_reads_exact_train_ids_and_writes_full_atomic_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _, _ = _fixture(tmp_path)
    output = tmp_path / "receipt.json"
    opened_ids: list[str] = []
    original_loader = selector._load_prepared_sample

    def audited_loader(
        record: dict[str, object],
        processed_root: Path,
    ) -> selector.PreparedSample:
        sample_id = str(record["id"])
        assert sample_id.startswith("train_")
        opened_ids.append(sample_id)
        return original_loader(record, processed_root)

    monkeypatch.setattr(selector, "_load_prepared_sample", audited_loader)
    receipt = selector.select_scale_caps(config, output, log_every=0, workers=1)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    # One complete train pass for grid scoring and one for selected direct audit.
    assert opened_ids == [
        "train_a/frame0",
        "train_b/frame0",
        "train_a/frame0",
        "train_b/frame0",
    ]
    isolation = receipt["split_isolation"]
    assert isolation["train_sample_count"] == 2
    assert isolation["train_room_count"] == 2
    assert isolation["validation_samples_opened"] == 0
    assert isolation["test_samples_opened"] == 0
    assert (
        isolation["ordered_train_ids_sha256"]
        == isolation["annotation_ordered_train_ids_sha256"]
        == isolation["selection_accessed_ids_sha256"]
        == isolation["direct_audit_accessed_ids_sha256"]
    )
    assert len(receipt["candidate_results"]) == 8 * 6 == 48
    assert receipt["execution"] == {
        "direct_audit_workers": 1,
        "opencv_internal_threads_during_parallel_audit": None,
        "reduction_order": "annotation input order",
        "affects_protocol_or_metrics": False,
    }
    assert receipt["leave_one_train_room_out"]["fold_count"] == 2
    assert receipt["protocol"]["candidate_grid"] == {
        "q10_log_cap": [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, "inf"],
        "q25_log_cap": [0.025, 0.05, 0.075, 0.1, 0.15, "inf"],
        "cartesian_candidate_count": 48,
    }
    final = receipt["final_selection"]["canonical_scale_estimator"]
    assert final["name"] == "log_upper_cap_v1"
    assert set(final) == {
        "name",
        "q10_log_cap",
        "q25_log_cap",
        "ratio_min",
        "ratio_max",
        "min_samples",
    }
    assert resolve_scale_estimator_config(final)["name"] == "log_upper_cap_v1"
    provenance = receipt["provenance"]
    assert len(provenance["config_raw_sha256"]) == 64
    assert len(provenance["effective_config_sha256"]) == 64
    assert len(provenance["manifest_raw_sha256"]) == 64
    assert len(provenance["manifest_preparation_fingerprint_sha256"]) == 64
    assert len(provenance["code"]["composite_sha256"]) == 64
    assert set(provenance["code"]["files_sha256"]) >= {
        "scripts/data/select_stanford_scale_caps.py",
        "src/bim_priorda3/baselines.py",
    }
    assert not list(tmp_path.glob(".receipt.json.*.tmp"))


def test_selector_refuses_to_overwrite_immutable_receipt(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    output = tmp_path / "receipt.json"
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Immutable selection receipt"):
        selector.select_scale_caps(config, output, log_every=0, workers=1)

    assert output.read_text(encoding="utf-8") == "sentinel\n"


def test_selector_requires_pinned_exhaustive_annotation_config(tmp_path: Path) -> None:
    config, _, _ = _fixture(tmp_path)
    value = yaml.safe_load(config.read_text(encoding="utf-8"))
    value["data"]["train_regions"] = ["train_a"]
    config.write_text(yaml.safe_dump(value), encoding="utf-8")

    with pytest.raises(ValueError, match="region split overrides must be empty"):
        selector.select_scale_caps(
            config,
            tmp_path / "receipt.json",
            log_every=0,
            workers=1,
        )


def test_selector_detects_annotation_change_before_opening_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, annotation, _ = _fixture(tmp_path)
    annotation.write_text(
        annotation.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        selector,
        "_load_prepared_sample",
        lambda *_args, **_kwargs: pytest.fail("sample must not be opened"),
    )

    with pytest.raises(ValueError, match="split_annotation_sha256 mismatch"):
        selector.select_scale_caps(
            config,
            tmp_path / "receipt.json",
            log_every=0,
            workers=1,
        )


def test_cli_has_no_room_stride_or_sample_count_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select_stanford_scale_caps.py",
            "--config",
            "config.yaml",
            "--output",
            "receipt.json",
            "--max-samples",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        selector.parse_args()


def test_exact_objective_tie_prefers_least_restrictive_candidate() -> None:
    rooms = ["room_a", "room_b"]
    statistics: dict[selector.Candidate, selector.CandidateStatistics] = {}
    for candidate in selector.CANDIDATES:
        candidate_statistics = selector.CandidateStatistics()
        for room in rooms:
            metrics = candidate_statistics.room(room)
            metrics["all"] = selector.MetricSums(count=10, abs_rel_sum=1.0)
            metrics["furniture"] = selector.MetricSums(count=2, abs_rel_sum=0.2)
        statistics[candidate] = candidate_statistics

    selected, score = selector._select_candidate(statistics, rooms)

    assert selected.q10_log_cap == float("inf")
    assert selected.q25_log_cap == float("inf")
    assert score["primary_tie_count"] == 48
    assert score["secondary_tie_count"] == 48


def test_grid_scales_exactly_match_runtime_float32_ratio_semantics() -> None:
    rng = np.random.default_rng(93017)
    base = rng.uniform(0.31, 3.9, size=(20, 23)).astype(np.float32)
    bim = rng.uniform(0.21, 5.8, size=(20, 23)).astype(np.float32)

    grid = selector._frame_candidate_scales(base, bim)

    for candidate, (scale, fallback, q10_triggered, q25_triggered) in grid.items():
        runtime = estimate_robust_bim_scale(
            base,
            bim,
            q10_log_cap=candidate.q10_log_cap,
            q25_log_cap=candidate.q25_log_cap,
            ratio_min=selector.RATIO_MIN,
            ratio_max=selector.RATIO_MAX,
            min_samples=selector.MIN_SCALE_SAMPLES,
        )
        assert scale == pytest.approx(runtime.scale, rel=1e-15, abs=1e-15)
        assert fallback is runtime.fallback
        assert q10_triggered is runtime.q10_cap_triggered
        assert q25_triggered is runtime.q25_cap_triggered


def test_parallel_direct_audit_exactly_matches_serial_ordered_reduction(
    tmp_path: Path,
) -> None:
    _, _, records = _fixture(tmp_path)
    train_records = records[:2]
    processed_root = tmp_path / "processed"
    selected = selector.Candidate(q10_log_cap=0.20, q25_log_cap=0.05)
    original_opencv_threads = cv2.getNumThreads()

    serial, serial_ids = selector._direct_audit(
        train_records,
        processed_root,
        selected,
        log_every=0,
        workers=1,
    )
    parallel, parallel_ids = selector._direct_audit(
        train_records,
        processed_root,
        selected,
        log_every=0,
        workers=2,
    )

    assert parallel == serial
    assert parallel_ids == serial_ids == ["train_a/frame0", "train_b/frame0"]
    assert cv2.getNumThreads() == original_opencv_threads
