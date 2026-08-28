from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .da3_features import feature_cache_path, record_ids_sha256, sha256_file

DINOV2_FEATURE_CACHE_SCHEMA_VERSION = 1
DINOV2_FEATURE_KEY = "feature"


def load_dinov2_feature_cache_manifest(
    root: Path,
    *,
    source_manifest: Path,
    expected_record_ids: Iterable[str],
    model_name: str,
    repository: str,
    repository_revision: str,
    weights_sha256: str,
    process_res: int,
    channels: int,
    grid_shape: tuple[int, int],
) -> dict[str, Any]:
    expected_ids = [str(sample_id) for sample_id in expected_record_ids]
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing complete DINOv2 feature-cache manifest: {path}. "
            "Run scripts/data/cache_stanford_dinov2_features.py first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": DINOV2_FEATURE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "source_manifest_sha256": sha256_file(source_manifest),
        "record_ids_sha256": record_ids_sha256(expected_ids),
        "model_name": str(model_name),
        "repository": str(repository),
        "repository_revision": str(repository_revision),
        "weights_sha256": str(weights_sha256),
        "process_res": int(process_res),
        "channels": int(channels),
        "grid_shape": list(grid_shape),
        "feature_key": "last_layer_x_norm_patchtokens",
        "dtype": "float16",
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"DINOv2 feature-cache manifest mismatch: {mismatches}")
    if int(payload.get("record_count", -1)) != len(expected_ids):
        raise ValueError("DINOv2 feature-cache record_count does not match manifest")
    return payload


def load_cached_dinov2_feature(
    root: Path,
    sample_id: str,
    *,
    channels: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    path = feature_cache_path(root, sample_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing DINOv2 feature cache for {sample_id}: {path}")
    with np.load(path, allow_pickle=False) as item:
        required = {
            DINOV2_FEATURE_KEY,
            "schema_version",
            "sample_id",
            "model_name",
            "repository_revision",
        }
        missing = sorted(required - set(item.files))
        if missing:
            raise ValueError(f"{path}: missing DINOv2 feature fields {missing}")
        if int(item["schema_version"]) != DINOV2_FEATURE_CACHE_SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported DINOv2 feature-cache schema")
        if str(item["sample_id"].item()) != sample_id:
            raise ValueError(f"{path}: sample_id does not match {sample_id!r}")
        feature = item[DINOV2_FEATURE_KEY].astype(np.float32)
    expected_shape = (int(channels), *grid_shape)
    if feature.shape != expected_shape:
        raise ValueError(f"{path}: feature has shape {feature.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(feature)):
        raise ValueError(f"{path}: DINOv2 feature contains non-finite values")
    return feature
