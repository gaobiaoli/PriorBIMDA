from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from bim_priorda3.data.stanford_split import (
    assign_room_splits,
    write_room_split_annotation,
)


def _records() -> list[dict[str, object]]:
    rooms = (
        [f"office_{index}" for index in range(1, 32)]
        + [f"hallway_{index}" for index in range(1, 9)]
        + ["conferenceRoom_1", "conferenceRoom_2", "WC_1", "copyRoom_1", "pantry_1"]
    )
    records = []
    for room_index, room in enumerate(rooms):
        for frame in range(1 + room_index % 4):
            records.append(
                {
                    "id": f"{room}/frame_{frame}",
                    "region": room,
                    "camera_uuid": f"camera_{room}_{frame // 2}",
                }
            )
    return records


def test_room_split_is_exact_disjoint_and_deterministic() -> None:
    records = _records()
    first, receipt = assign_room_splits(
        records,
        train_rooms=30,
        val_rooms=7,
        test_rooms=7,
        seed=42,
        search_trials=100,
    )
    second, _ = assign_room_splits(
        records,
        train_rooms=30,
        val_rooms=7,
        test_rooms=7,
        seed=42,
        search_trials=100,
    )
    assert first == second
    assert Counter(first.values()) == {"train": 30, "val": 7, "test": 7}
    assert receipt["room_disjoint"]
    assert receipt["camera_uuid_disjoint"]


def test_write_room_split_annotation_is_exhaustive(tmp_path: Path) -> None:
    records = _records()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    annotation = tmp_path / "split.jsonl"
    receipt_path = tmp_path / "receipt.json"
    receipt = write_room_split_annotation(
        manifest,
        annotation,
        receipt_path,
        search_trials=100,
    )
    assert receipt["split_counts"] == {
        "train": sum(
            1 for record in records if receipt["room_assignment"][record["region"]] == "train"
        ),
        "val": sum(
            1 for record in records if receipt["room_assignment"][record["region"]] == "val"
        ),
        "test": sum(
            1 for record in records if receipt["room_assignment"][record["region"]] == "test"
        ),
        "excluded": 0,
    }
    assert len(annotation.read_text(encoding="utf-8").splitlines()) == len(records)
    assert receipt_path.is_file()
