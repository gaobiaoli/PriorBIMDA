from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bim_priorda3.data.splits import resolve_annotation_splits

SPLITS = ("train", "val", "test")


def room_type(room: str) -> str:
    prefix, separator, number = room.rpartition("_")
    if not separator or not prefix or not number.isdigit():
        raise ValueError(f"Room must end in an integer instance number: {room!r}")
    return prefix


def _compositions(total: int) -> list[tuple[int, int, int]]:
    return [
        (train, val, total - train - val)
        for train in range(total + 1)
        for val in range(total - train + 1)
    ]


def category_quotas(
    counts: dict[str, int],
    target_counts: dict[str, int],
) -> dict[str, dict[str, int]]:
    """Find class-stratified integer quotas with exact global room counts."""

    if sum(counts.values()) != sum(target_counts.values()):
        raise ValueError("Category and split room totals differ")
    total = sum(counts.values())
    fractions = {split: target_counts[split] / total for split in SPLITS}
    states: dict[tuple[int, int], tuple[float, list[tuple[str, tuple[int, int, int]]]]] = {
        (0, 0): (0.0, [])
    }
    for category in sorted(counts):
        count = counts[category]
        next_states: dict[
            tuple[int, int], tuple[float, list[tuple[str, tuple[int, int, int]]]]
        ] = {}
        for allocation in _compositions(count):
            allocation_cost = sum(
                (allocation[index] - count * fractions[split]) ** 2 / max(count, 1)
                for index, split in enumerate(SPLITS)
            )
            for (used_train, used_val), (cost, history) in states.items():
                key = (used_train + allocation[0], used_val + allocation[1])
                if key[0] > target_counts["train"] or key[1] > target_counts["val"]:
                    continue
                candidate = (cost + allocation_cost, history + [(category, allocation)])
                previous = next_states.get(key)
                if previous is None or candidate < previous:
                    next_states[key] = candidate
        states = next_states
    final_key = (target_counts["train"], target_counts["val"])
    if final_key not in states:
        raise RuntimeError("Could not satisfy exact room counts with category quotas")
    _, history = states[final_key]
    return {
        category: {split: allocation[index] for index, split in enumerate(SPLITS)}
        for category, allocation in history
    }


def assign_room_splits(
    records: list[dict[str, Any]],
    *,
    train_rooms: int,
    val_rooms: int,
    test_rooms: int,
    seed: int,
    search_trials: int = 20_000,
) -> tuple[dict[str, str], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["region"])].append(record)
    rooms = sorted(grouped)
    target_counts = {
        "train": int(train_rooms),
        "val": int(val_rooms),
        "test": int(test_rooms),
    }
    if sum(target_counts.values()) != len(rooms):
        raise ValueError(
            f"Requested {sum(target_counts.values())} rooms, manifest has {len(rooms)}"
        )
    if any(count <= 0 for count in target_counts.values()):
        raise ValueError("Every split must contain at least one room")

    rooms_by_type: dict[str, list[str]] = defaultdict(list)
    for room in rooms:
        rooms_by_type[room_type(room)].append(room)
    type_counts = {key: len(value) for key, value in rooms_by_type.items()}
    quotas = category_quotas(type_counts, target_counts)
    frame_count = {room: len(grouped[room]) for room in rooms}
    camera_count = {
        room: len({str(record.get("camera_uuid", "")) for record in grouped[room]})
        for room in rooms
    }
    total_frames = sum(frame_count.values())
    total_cameras = sum(camera_count.values())
    target_fractions = {split: target_counts[split] / len(rooms) for split in SPLITS}

    rng = random.Random(seed)
    best: tuple[float, tuple[tuple[str, str], ...], dict[str, str]] | None = None
    for trial in range(max(1, int(search_trials))):
        assignment: dict[str, str] = {}
        for category in sorted(rooms_by_type):
            category_rooms = list(rooms_by_type[category])
            if trial == 0:
                category_rooms.sort(
                    key=lambda room: hashlib.sha256(
                        f"stanford-room-split-v1|{seed}|{room}".encode()
                    ).digest()
                )
            else:
                rng.shuffle(category_rooms)
            position = 0
            for split in SPLITS:
                count = quotas[category][split]
                for room in category_rooms[position : position + count]:
                    assignment[room] = split
                position += count

        split_frames = {
            split: sum(frame_count[room] for room, owner in assignment.items() if owner == split)
            for split in SPLITS
        }
        split_cameras = {
            split: sum(camera_count[room] for room, owner in assignment.items() if owner == split)
            for split in SPLITS
        }
        frame_error = sum(
            (split_frames[split] / total_frames - target_fractions[split]) ** 2 for split in SPLITS
        )
        camera_error = sum(
            (split_cameras[split] / total_cameras - target_fractions[split]) ** 2
            for split in SPLITS
        )
        score = frame_error + 0.25 * camera_error
        canonical = tuple(sorted(assignment.items()))
        candidate = (score, canonical, assignment)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    assert best is not None
    assignment = best[2]

    split_rooms = {
        split: sorted(room for room, owner in assignment.items() if owner == split)
        for split in SPLITS
    }
    if {room for values in split_rooms.values() for room in values} != set(rooms):
        raise RuntimeError("Room split assignment is not exhaustive and disjoint")
    camera_owners: dict[str, set[str]] = defaultdict(set)
    for room, room_records in grouped.items():
        for record in room_records:
            camera_uuid = str(record.get("camera_uuid", ""))
            if not camera_uuid:
                raise ValueError(f"{record['id']}: manifest lacks camera_uuid")
            camera_owners[camera_uuid].add(assignment[room])
    leaked_cameras = sorted(camera for camera, owners in camera_owners.items() if len(owners) != 1)
    if leaked_cameras:
        raise ValueError(f"camera_uuid leakage across splits: {leaked_cameras[:10]}")

    receipt = {
        "schema_version": 1,
        "protocol": "stanford-area1-room-disjoint-v1",
        "seed": int(seed),
        "search_trials": int(search_trials),
        "objective": "frame-ratio-squared-error + 0.25*camera-ratio-squared-error",
        "room_counts": target_counts,
        "room_type_counts": dict(sorted(type_counts.items())),
        "room_type_quotas": quotas,
        "rooms": split_rooms,
        "frame_counts": {
            split: sum(frame_count[room] for room in split_rooms[split]) for split in SPLITS
        },
        "camera_counts": {
            split: sum(camera_count[room] for room in split_rooms[split]) for split in SPLITS
        },
        "room_assignment": dict(sorted(assignment.items())),
        "objective_value": float(best[0]),
        "room_disjoint": True,
        "camera_uuid_disjoint": True,
    }
    return assignment, receipt


def write_room_split_annotation(
    manifest_path: str | Path,
    annotation_path: str | Path,
    receipt_path: str | Path,
    *,
    train_rooms: int = 30,
    val_rooms: int = 7,
    test_rooms: int = 7,
    seed: int = 42,
    search_trials: int = 20_000,
) -> dict[str, Any]:
    manifest = Path(manifest_path).expanduser().resolve()
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Manifest is empty: {manifest}")
    ids = [str(record.get("id", "")) for record in records]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if any(not sample_id for sample_id in ids) or duplicates:
        raise ValueError(f"Manifest has invalid/duplicate IDs: {duplicates[:10]}")
    assignment, receipt = assign_room_splits(
        records,
        train_rooms=train_rooms,
        val_rooms=val_rooms,
        test_rooms=test_rooms,
        seed=seed,
        search_trials=search_trials,
    )

    annotation = Path(annotation_path).expanduser().resolve()
    annotation.parent.mkdir(parents=True, exist_ok=True)
    with annotation.open("w", encoding="utf-8") as handle:
        for record in records:
            value = {
                "schema_version": 1,
                "id": str(record["id"]),
                "split": assignment[str(record["region"])],
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    resolution = resolve_annotation_splits(records, annotation)
    receipt.update(
        {
            "manifest": str(manifest),
            "annotation": str(annotation),
            "annotation_raw_sha256": resolution.provenance["annotation_raw_sha256"],
            "split_fingerprint_sha256": resolution.provenance["fingerprint_sha256"],
            "manifest_preparation_fingerprint_status": resolution.provenance[
                "manifest_preparation_fingerprint_status"
            ],
            "manifest_preparation_fingerprint_sha256": resolution.provenance[
                "manifest_preparation_fingerprint_sha256"
            ],
            "split_counts": resolution.provenance["split_counts"],
            "ordered_ids_sha256": resolution.provenance["ordered_ids_sha256"],
        }
    )
    receipt_output = Path(receipt_path).expanduser().resolve()
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return receipt
