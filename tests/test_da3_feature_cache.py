from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bim_priorda3.data.da3_features import (
    DA3_FEATURE_CACHE_SCHEMA_VERSION,
    feature_cache_path,
    load_cached_features,
    load_feature_cache_manifest,
    record_ids_sha256,
    sha256_file,
)


def test_feature_cache_manifest_and_sample_are_strictly_bound(tmp_path: Path) -> None:
    source_manifest = tmp_path / "manifest.jsonl"
    source_manifest.write_text('{"id":"room/frame"}\n', encoding="utf-8")
    root = tmp_path / "features"
    root.mkdir()
    metadata = {
        "schema_version": DA3_FEATURE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "source_manifest_sha256": sha256_file(source_manifest),
        "record_count": 1,
        "record_ids_sha256": record_ids_sha256(["room/frame"]),
        "model_name": "depth-anything/test",
        "model_revision": "abc123",
        "process_res": 28,
        "layers": [1, 3],
        "channels": 4,
        "grid_shape": [2, 2],
    }
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    path = feature_cache_path(root, "room/frame")
    path.parent.mkdir()
    np.savez(
        path,
        schema_version=np.int64(DA3_FEATURE_CACHE_SCHEMA_VERSION),
        sample_id=np.str_("room/frame"),
        layers=np.asarray((1, 3), dtype=np.int64),
        feature_mid=np.ones((4, 2, 2), dtype=np.float16),
        feature_deep=np.full((4, 2, 2), 2, dtype=np.float16),
    )

    loaded = load_feature_cache_manifest(
        root,
        source_manifest=source_manifest,
        expected_record_ids=["room/frame"],
        model_name="depth-anything/test",
        model_revision="abc123",
        process_res=28,
        layers=(1, 3),
        channels=4,
    )
    mid, deep = load_cached_features(
        root,
        "room/frame",
        layers=(1, 3),
        channels=4,
        grid_shape=tuple(loaded["grid_shape"]),
    )
    assert mid.dtype == np.float32
    np.testing.assert_array_equal(mid, np.ones((4, 2, 2), dtype=np.float32))
    np.testing.assert_array_equal(deep, np.full((4, 2, 2), 2, dtype=np.float32))

    metadata["model_revision"] = "wrong"
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_feature_cache_manifest(
            root,
            source_manifest=source_manifest,
            expected_record_ids=["room/frame"],
            model_name="depth-anything/test",
            model_revision="abc123",
            process_res=28,
            layers=(1, 3),
            channels=4,
        )


def test_feature_cache_rejects_unsafe_sample_id(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        feature_cache_path(tmp_path, "../escape")
