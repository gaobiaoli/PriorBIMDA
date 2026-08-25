from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import cv2
import numpy as np
import pytest

from bim_priorda3.config import Config
from bim_priorda3.data.preparation import sha256_file
from scripts.data import cache_stanford_pano_da3 as cacher


def _write_rgb(path: Path) -> None:
    height, width = 8, 16
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(width, dtype=np.uint8)[None, :] * 7
    rgb[..., 1] = np.arange(height, dtype=np.uint8)[:, None] * 13
    rgb[..., 2] = 91
    assert cv2.imwrite(str(path), rgb[..., ::-1])


def _args(config: Path, output: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "config": str(config),
        "split": "val",
        "confirm_test": False,
        "preset": "cubemap6",
        "face_resolution": 4,
        "max_stations": 1,
        "output_root": output,
        "log_every": 1,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _FakeProvider:
    calls: ClassVar[list[tuple[Path, tuple[int, int]]]] = []

    def __init__(self, _cfg: Config, _region: str, output_cache: Path) -> None:
        self.write_cache = output_cache
        self.write_cache.mkdir(parents=True, exist_ok=True)
        self.model_name = "depth-anything/test"
        self.model_revision = "0123456789abcdef"
        self.process_res = 16
        self.local_files_only = True

    def get_with_provenance(self, image_path: Path, shape: tuple[int, int]) -> SimpleNamespace:
        self.calls.append((image_path, shape))
        cache_path = self.write_cache / f"{image_path.stem}.npz"
        if not cache_path.exists():
            np.savez_compressed(
                cache_path,
                depth=np.ones(shape, dtype=np.float16),
                confidence=np.ones(shape, dtype=np.float16),
            )
        image_sha = sha256_file(image_path)
        return SimpleNamespace(
            cache_path=cache_path,
            cache_sha256=sha256_file(cache_path),
            image_sha256=image_sha,
            model_name=self.model_name,
            model_revision=self.model_revision,
            process_res=self.process_res,
            target_shape=shape,
            provenance_status="direct_inference",
        )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path, SimpleNamespace]:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    annotation_path = tmp_path / "split.jsonl"
    annotation_path.write_text('{"id":"fixture","split":"val"}\n', encoding="utf-8")
    processed = tmp_path / "processed"
    cfg = Config(
        project_root=str(tmp_path),
        data=Config(
            processed_root=str(processed),
            split_annotation=str(annotation_path),
        ),
    )
    pano_path = tmp_path / "pano.png"
    _write_rgb(pano_path)
    station = SimpleNamespace(
        camera_uuid="a" * 32,
        room="office_14",
        rgb_path=pano_path.resolve(),
    )
    split_provenance = {
        "mode": "annotations",
        "fingerprint_sha256": "b" * 64,
    }
    selection = {
        "room_source": "fixture annotation",
        "rooms": ["office_14"],
        "annotation_regular_station_count": 1,
        "split_pano_station_count": 1,
        "shared_regular_pano_station_count": 1,
        "pano_only_station_ids": [],
    }
    monkeypatch.setattr(cacher, "load_config", lambda _path: cfg)
    monkeypatch.setattr(
        cacher,
        "_select_split_stations",
        lambda _cfg, _split: ([station], split_provenance, selection),
    )
    monkeypatch.setattr(cacher, "DA3PredictionProvider", _FakeProvider)
    _FakeProvider.calls = []
    return config_path, annotation_path, station


def test_cache_run_is_atomic_provenance_bound_and_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, station = _patch_run(monkeypatch, tmp_path)
    output = tmp_path / "pano_cache"
    args = _args(config_path, output)

    first = cacher.cache_stanford_pano_da3(args)

    assert first["manifest_status"] == "generated"
    assert first["generated_pngs"] == 6
    assert first["reused_pngs"] == 0
    manifest = first["manifest"]
    assert manifest["protocol"] == cacher.PROTOCOL
    assert manifest["split"] == "val"
    assert manifest["selection"]["formal_protocol_eligible"] is False
    assert manifest["input_contract"] == {
        "pano_rgb_decoded": True,
        "pano_pose_metadata_decoded": True,
        "pano_depth_decoded": False,
        "pano_semantic_decoded": False,
        "regular_ground_truth_decoded": False,
        "output_depth_quantity": "perspective_z_depth_m",
    }
    assert manifest["stations"][0]["pano_rgb"]["sha256"] == sha256_file(station.rgb_path)
    views = manifest["stations"][0]["tangent_views"]
    assert [value["view"]["name"] for value in views] == [
        "front",
        "right",
        "back",
        "left",
        "up",
        "down",
    ]
    assert all(np.asarray(value["view"]["intrinsic"]).shape == (3, 3) for value in views)
    assert all(np.asarray(value["view"]["T_face_from_pano"]).shape == (4, 4) for value in views)
    assert all(value["view"]["horizontal_fov_degrees"] == 100.0 for value in views)
    assert all(value["da3_cache"]["model_revision"] == "0123456789abcdef" for value in views)
    assert all(value["da3_cache"]["process_res"] == 16 for value in views)
    assert all(
        value["da3_cache"]["image_sha256"] == value["tangent_rgb"]["sha256"] for value in views
    )
    assert all(path != station.rgb_path for path, _ in _FakeProvider.calls)
    assert all(shape == (4, 4) for _, shape in _FakeProvider.calls)
    assert json.loads(first["manifest_path"].read_text(encoding="utf-8")) == manifest
    assert not list(output.rglob("*.tmp"))

    _FakeProvider.calls = []
    second = cacher.cache_stanford_pano_da3(args)
    assert second["manifest_status"] == "reused"
    assert second["generated_pngs"] == 0
    assert second["reused_pngs"] == 6
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert len(_FakeProvider.calls) == 6


def test_changed_existing_tangent_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _, _ = _patch_run(monkeypatch, tmp_path)
    args = _args(config_path, tmp_path / "pano_cache")
    first = cacher.cache_stanford_pano_da3(args)
    tangent_path = Path(first["manifest"]["stations"][0]["tangent_views"][0]["tangent_rgb"]["path"])
    corrupted = cv2.imread(str(tangent_path), cv2.IMREAD_COLOR)
    assert corrupted is not None
    corrupted[0, 0] ^= 255
    assert cv2.imwrite(str(tangent_path), corrupted)

    with pytest.raises(FileExistsError, match="pixels differ"):
        cacher.cache_stanford_pano_da3(args)


def test_split_selection_includes_pano_only_station_from_annotated_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.records = [
                {"id": "a/0", "region": "office_14", "camera_uuid": "shared"},
                {"id": "a/1", "region": "office_14", "camera_uuid": "shared"},
            ]
            self.split_provenance = {"mode": "annotations"}

    stations = [
        SimpleNamespace(room="office_14", camera_uuid="shared"),
        SimpleNamespace(room="office_14", camera_uuid="pano_only"),
        SimpleNamespace(room="office_16", camera_uuid="test_station"),
    ]
    cfg = Config(
        project_root="/tmp",
        data=Config(stanford_area_root="area_1", split_annotation="split.jsonl"),
    )
    monkeypatch.setattr(cacher, "BIMDepthDataset", FakeDataset)
    monkeypatch.setattr(cacher, "discover_stanford_panoramas", lambda _root: stations)

    selected, provenance, selection = cacher._select_split_stations(cfg, "val")

    assert [station.camera_uuid for station in selected] == ["pano_only", "shared"]
    assert provenance == {"mode": "annotations"}
    assert selection["pano_only_station_ids"] == ["pano_only"]


def test_preset_namespace_changes_with_geometry() -> None:
    _, cubemap = cacher._preset_identity("cubemap6", 32)
    _, nested = cacher._preset_identity("nested14", 32)
    _, resized = cacher._preset_identity("cubemap6", 64)

    assert (
        len({cubemap["cache_namespace"], nested["cache_namespace"], resized["cache_namespace"]})
        == 3
    )
    assert nested["views"][:6] == cubemap["views"]


def test_test_split_requires_explicit_confirmation() -> None:
    with pytest.raises(SystemExit):
        cacher.parse_args(["--split", "test"])
    parsed = cacher.parse_args(["--split", "test", "--confirm-test"])
    assert parsed.split == "test"
    assert parsed.confirm_test is True

    with pytest.raises(ValueError, match="explicit"):
        cacher.cache_stanford_pano_da3(
            argparse.Namespace(
                split="test",
                confirm_test=False,
                max_stations=1,
                log_every=1,
            )
        )
