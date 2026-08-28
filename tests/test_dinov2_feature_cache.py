from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from bim_priorda3.data.da3_features import feature_cache_path, record_ids_sha256, sha256_file
from bim_priorda3.data.dinov2_features import (
    DINOV2_FEATURE_CACHE_SCHEMA_VERSION,
    load_cached_dinov2_feature,
    load_dinov2_feature_cache_manifest,
)
from scripts.data.cache_stanford_dinov2_features import freeze_dinov2


def test_dinov2_feature_cache_is_strictly_bound_and_loaded_as_float32(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "manifest.jsonl"
    source_manifest.write_text('{"id":"room/frame"}\n', encoding="utf-8")
    root = tmp_path / "features"
    root.mkdir()
    metadata = {
        "schema_version": DINOV2_FEATURE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "source_manifest_sha256": sha256_file(source_manifest),
        "record_count": 1,
        "record_ids_sha256": record_ids_sha256(["room/frame"]),
        "model_name": "dinov2_vitb14",
        "repository": "facebookresearch/dinov2",
        "repository_revision": "abc123",
        "weights_sha256": "weights-hash",
        "process_res": 28,
        "channels": 4,
        "grid_shape": [2, 2],
        "feature_key": "last_layer_x_norm_patchtokens",
        "dtype": "float16",
    }
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    path = feature_cache_path(root, "room/frame")
    path.parent.mkdir()
    np.savez(
        path,
        schema_version=np.int64(DINOV2_FEATURE_CACHE_SCHEMA_VERSION),
        sample_id=np.str_("room/frame"),
        model_name=np.str_("dinov2_vitb14"),
        repository_revision=np.str_("abc123"),
        feature=np.ones((4, 2, 2), dtype=np.float16),
    )

    loaded = load_dinov2_feature_cache_manifest(
        root,
        source_manifest=source_manifest,
        expected_record_ids=["room/frame"],
        model_name="dinov2_vitb14",
        repository="facebookresearch/dinov2",
        repository_revision="abc123",
        weights_sha256="weights-hash",
        process_res=28,
        channels=4,
        grid_shape=(2, 2),
    )
    feature = load_cached_dinov2_feature(
        root,
        "room/frame",
        channels=4,
        grid_shape=tuple(loaded["grid_shape"]),
    )
    assert feature.dtype == np.float32
    np.testing.assert_array_equal(feature, np.ones((4, 2, 2), dtype=np.float32))

    metadata["weights_sha256"] = "wrong"
    (root / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest mismatch"):
        load_dinov2_feature_cache_manifest(
            root,
            source_manifest=source_manifest,
            expected_record_ids=["room/frame"],
            model_name="dinov2_vitb14",
            repository="facebookresearch/dinov2",
            repository_revision="abc123",
            weights_sha256="weights-hash",
            process_res=28,
            channels=4,
            grid_shape=(2, 2),
        )


def test_freeze_dinov2_disables_gradients_and_training_mode() -> None:
    backbone = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU())
    freeze_dinov2(backbone)
    assert not backbone.training
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
