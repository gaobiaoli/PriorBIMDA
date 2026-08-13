from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.data.materialize_runtime_config import materialize_runtime_config


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def test_materializes_current_split_and_scale_receipt(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    processed = tmp_path / "data/processed"
    _write_jsonl(
        processed / "manifest.jsonl",
        [
            {
                "id": f"room_{index}/frame",
                "region": f"room_{index}",
                "preparation_fingerprint_sha256": f"{index + 1:064x}",
            }
            for index in range(3)
        ],
    )
    annotation = tmp_path / "data/split.jsonl"
    _write_jsonl(
        annotation,
        [
            {"schema_version": 1, "id": "room_0/frame", "split": "train"},
            {"schema_version": 1, "id": "room_1/frame", "split": "val"},
            {"schema_version": 1, "id": "room_2/frame", "split": "test"},
        ],
    )
    base = tmp_path / "configs/base.yaml"
    base.parent.mkdir()
    base.write_text(
        "data:\n"
        "  slabim_root: .\n"
        "  processed_root: data/processed\n"
        "  split_annotation: data/split.jsonl\n"
        "model:\n"
        "  scale_estimator:\n"
        "    name: log_upper_cap_v1\n"
        "evaluation: {}\n",
        encoding="utf-8",
    )
    scale_receipt = tmp_path / "data/scale.json"
    scale_payload = {
        "status": "complete",
        "protocol_sha256": "a" * 64,
        "final_selection": {
            "canonical_scale_estimator": {
                "name": "log_upper_cap_v1",
                "q10_log_cap": "inf",
                "q25_log_cap": 0.05,
                "ratio_min": 0.2,
                "ratio_max": 5.0,
                "min_samples": 100,
            }
        },
    }
    scale_receipt.write_text(json.dumps(scale_payload), encoding="utf-8")
    alignment_receipt = tmp_path / "data/alignment.json"
    alignment_receipt.write_text('{"schema_version": 2}\n', encoding="utf-8")
    output = tmp_path / "configs/local/runtime.yaml"

    result = materialize_runtime_config(
        base,
        output,
        annotation=annotation,
        scale_receipt=scale_receipt,
        alignment_receipt=alignment_receipt,
        experiment_output_dir=tmp_path / "outputs/runtime",
    )

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated["extends"] == "../base.yaml"
    assert generated["data"]["split_annotation"] == "data/split.jsonl"
    assert generated["data"]["split_annotation_sha256"] == result["annotation_raw_sha256"]
    assert generated["data"]["split_fingerprint_sha256"] == result["split_fingerprint_sha256"]
    assert generated["evaluation"]["robust_scale_selection_receipt_sha256"] == (
        hashlib.sha256(scale_receipt.read_bytes()).hexdigest()
    )
    assert generated["model"]["scale_estimator"]["q25_log_cap"] == 0.05
    assert generated["data"]["bim_alignment"] == "data/alignment.json"
    assert (
        generated["data"]["bim_alignment_sha256"]
        == hashlib.sha256(alignment_receipt.read_bytes()).hexdigest()
    )
    assert result["split_counts"] == {"train": 1, "val": 1, "test": 1, "excluded": 0}


def test_preparation_only_pins_alignment_before_manifest_exists(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    base = tmp_path / "configs/base.yaml"
    base.parent.mkdir()
    base.write_text(
        "data:\n  slabim_root: .\n  processed_root: data/not_prepared_yet\nmodel: {}\n",
        encoding="utf-8",
    )
    alignment = tmp_path / "data/alignment.local.json"
    alignment.parent.mkdir()
    alignment.write_text('{"schema_version": 2}\n', encoding="utf-8")
    output = tmp_path / "configs/local/prepare.yaml"

    result = materialize_runtime_config(
        base,
        output,
        alignment_receipt=alignment,
        preparation_only=True,
    )

    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert generated == {
        "extends": "../base.yaml",
        "data": {
            "bim_alignment": "data/alignment.local.json",
            "bim_alignment_sha256": hashlib.sha256(alignment.read_bytes()).hexdigest(),
        },
    }
    assert result["mode"] == "preparation_only"
    assert result["manifest"] is None
    assert result["split_counts"] is None
