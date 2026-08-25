from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

DA3_FEATURE_CACHE_SCHEMA_VERSION = 1
DA3_FEATURE_KEYS = ("feature_mid", "feature_deep")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_ids_sha256(record_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in record_ids:
        digest.update(str(sample_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def feature_cache_path(root: Path, sample_id: str) -> Path:
    relative = Path(sample_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe DA3 feature-cache sample id: {sample_id!r}")
    return root / relative.parent / f"{relative.name}.npz"


def load_feature_cache_manifest(
    root: Path,
    *,
    source_manifest: Path,
    expected_record_ids: Iterable[str],
    model_name: str,
    model_revision: str,
    process_res: int,
    layers: tuple[int, int],
    channels: int,
) -> dict[str, Any]:
    expected_ids = [str(sample_id) for sample_id in expected_record_ids]
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing complete DA3 feature-cache manifest: {path}. "
            "Run scripts/data/cache_stanford_da3_features.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": DA3_FEATURE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "source_manifest_sha256": sha256_file(source_manifest),
        "record_ids_sha256": record_ids_sha256(expected_ids),
        "model_name": str(model_name),
        "model_revision": str(model_revision),
        "process_res": int(process_res),
        "layers": list(layers),
        "channels": int(channels),
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DA3 feature-cache manifest mismatch: {mismatches}")
    if int(payload.get("record_count", -1)) != len(expected_ids):
        raise ValueError("DA3 feature-cache record_count does not match the prepared manifest")
    grid_shape = payload.get("grid_shape")
    if (
        not isinstance(grid_shape, list)
        or len(grid_shape) != 2
        or any(int(value) < 1 for value in grid_shape)
    ):
        raise ValueError(f"Invalid DA3 feature-cache grid_shape: {grid_shape!r}")
    return payload


def load_cached_features(
    root: Path,
    sample_id: str,
    *,
    layers: tuple[int, int],
    channels: int,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    path = feature_cache_path(root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing DA3 feature cache for {sample_id}: {path}")
    with np.load(path, allow_pickle=False) as item:
        required = {
            *DA3_FEATURE_KEYS,
            "schema_version",
            "sample_id",
            "layers",
        }
        missing = sorted(required - set(item.files))
        if missing:
            raise ValueError(f"{path}: missing DA3 feature fields {missing}")
        if int(item["schema_version"]) != DA3_FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported DA3 feature-cache schema")
        if str(item["sample_id"].item()) != sample_id:
            raise ValueError(f"{path}: sample_id does not match {sample_id!r}")
        if tuple(int(value) for value in item["layers"].tolist()) != layers:
            raise ValueError(f"{path}: cached feature layers do not match {layers}")
        expected_shape = (channels, *grid_shape)
        features = tuple(item[key].astype(np.float32) for key in DA3_FEATURE_KEYS)
    for key, feature in zip(DA3_FEATURE_KEYS, features, strict=True):
        if feature.shape != expected_shape:
            raise ValueError(
                f"{path}: {key} has shape {feature.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(feature)):
            raise ValueError(f"{path}: {key} contains non-finite values")
    return features
