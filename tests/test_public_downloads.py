from __future__ import annotations

import copy
import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from bim_priorda3.data import public_downloads
from bim_priorda3.data.public_downloads import (
    BIMSYNC_MANIFEST_PATH,
    _normalized_tar_member,
    _selected_area1_member,
    _verify_area1_modality_pairing,
    download_sharepoint_files,
    download_url,
    extract_stanford_area1,
    load_bimsyn_manifest,
    stanford_area1_inventory,
    verify_bimsyn_model_directory,
    verify_stanford_area1_mesh,
    verify_stanford_semantic_labels,
)


def test_selected_area1_member_excludes_raw_and_pano() -> None:
    assert _selected_area1_member("area_1/data/rgb/example.png")
    assert _selected_area1_member("area_1/data/depth/example.png")
    assert _selected_area1_member("area_1/3d/semantic.obj")
    assert not _selected_area1_member("area_1/raw/example.jpg")
    assert not _selected_area1_member("area_1/pano/rgb/example.png")


def _add_bytes(bundle: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    bundle.addfile(info, io.BytesIO(value))


def test_extract_stanford_area1_rejects_incomplete_archive(tmp_path: Path) -> None:
    archive = tmp_path / "area.tar"
    with tarfile.open(archive, "w") as bundle:
        _add_bytes(bundle, "area_1/data/rgb/frame.png", b"rgb")
        _add_bytes(bundle, "area_1/raw/not-needed.jpg", b"raw")

    with pytest.raises(RuntimeError, match="expected 10327 rgb"):
        extract_stanford_area1(archive, tmp_path / "out", progress=lambda _message: None)
    assert (tmp_path / "out/area_1/data/rgb/frame.png").read_bytes() == b"rgb"
    assert not (tmp_path / "out/area_1/raw/not-needed.jpg").exists()


def test_extract_stanford_area1_rejects_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        _add_bytes(bundle, "../area_1/data/rgb/frame.png", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe path"):
        extract_stanford_area1(archive, tmp_path / "out", progress=lambda _message: None)


def test_tar_member_rejects_windows_backslash_traversal() -> None:
    with pytest.raises(ValueError, match="Windows-style"):
        _normalized_tar_member(r"..\area_1\data\rgb\frame.png")


def test_stanford_area1_inventory_counts_modalities(tmp_path: Path) -> None:
    area = tmp_path / "area_1"
    for modality, suffix in {
        "rgb": ".png",
        "depth": ".png",
        "pose": ".json",
        "semantic": ".png",
    }.items():
        directory = area / "data" / modality
        directory.mkdir(parents=True)
        (directory / f"frame{suffix}").write_bytes(b"sample")
    mesh = area / "3d"
    mesh.mkdir()
    (mesh / "semantic.obj").write_text("mesh", encoding="utf-8")

    assert stanford_area1_inventory(tmp_path) == {
        "rgb": 1,
        "depth": 1,
        "pose": 1,
        "semantic": 1,
        "mesh": 1,
    }


def test_area1_modalities_require_paired_frame_basenames(tmp_path: Path) -> None:
    area = tmp_path / "area_1"
    for modality, suffix in {
        "rgb": ".png",
        "depth": ".png",
        "pose": ".json",
        "semantic": ".png",
    }.items():
        directory = area / "data" / modality
        directory.mkdir(parents=True)
        frame = "different" if modality == "depth" else "same"
        (directory / f"{frame}_domain_{modality}{suffix}").write_bytes(b"sample")

    with pytest.raises(ValueError, match="basenames do not pair"):
        _verify_area1_modality_pairing(area)


def test_stanford_fixed_artifacts_are_sha_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = b"canonical obj"
    mtl = b"canonical mtl"
    labels = b'["wall"]\n'
    mesh = tmp_path / "area_1/3d"
    mesh.mkdir(parents=True)
    (mesh / "semantic.obj").write_bytes(obj)
    (mesh / "semantic.mtl").write_bytes(mtl)
    labels_path = tmp_path / "semantic_labels.json"
    labels_path.write_bytes(labels)
    monkeypatch.setattr(public_downloads, "STANFORD_SEMANTIC_OBJ_BYTES", len(obj))
    monkeypatch.setattr(
        public_downloads,
        "STANFORD_SEMANTIC_OBJ_SHA256",
        hashlib.sha256(obj).hexdigest(),
    )
    monkeypatch.setattr(public_downloads, "STANFORD_SEMANTIC_MTL_BYTES", len(mtl))
    monkeypatch.setattr(
        public_downloads,
        "STANFORD_SEMANTIC_MTL_SHA256",
        hashlib.sha256(mtl).hexdigest(),
    )
    monkeypatch.setattr(public_downloads, "STANFORD_LABELS_BYTES", len(labels))
    monkeypatch.setattr(
        public_downloads,
        "STANFORD_LABELS_SHA256",
        hashlib.sha256(labels).hexdigest(),
    )

    verify_stanford_area1_mesh(tmp_path)
    verify_stanford_semantic_labels(labels_path)
    labels_path.write_bytes(b'["WALL"]\n')
    with pytest.raises(ValueError, match="Unexpected sha256"):
        verify_stanford_semantic_labels(labels_path)


class _FakeResponse(io.BytesIO):
    def __init__(self, value: bytes, *, status: int = 200, content_range: str | None = None):
        super().__init__(value)
        self.status = status
        self.headers = {} if content_range is None else {"Content-Range": content_range}

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def test_download_url_validates_resume_content_range_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    partial = tmp_path / "artifact.bin.part"
    partial.write_bytes(b"abc")
    observed: dict[str, float] = {}

    def fake_urlopen(_request: object, *, timeout: float) -> _FakeResponse:
        observed["timeout"] = timeout
        return _FakeResponse(b"def", status=206, content_range="bytes 2-5/6")

    monkeypatch.setattr(public_downloads, "urlopen", fake_urlopen)
    with pytest.raises(OSError, match="starts at 2, expected 3"):
        download_url(
            "https://example.invalid/artifact.bin",
            destination,
            expected_bytes=6,
            timeout=7.5,
            progress=lambda _message: None,
        )
    assert observed["timeout"] == 7.5
    assert partial.read_bytes() == b"abc"


def test_download_url_resumes_when_content_range_matches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    (tmp_path / "artifact.bin.part").write_bytes(b"abc")

    def fake_urlopen(_request: object, *, timeout: float) -> _FakeResponse:
        assert timeout == 5.0
        return _FakeResponse(b"def", status=206, content_range="bytes 3-5/6")

    monkeypatch.setattr(public_downloads, "urlopen", fake_urlopen)
    result = download_url(
        "https://example.invalid/artifact.bin",
        destination,
        expected_bytes=6,
        expected_digest=hashlib.sha256(b"abcdef").hexdigest(),
        timeout=5.0,
        progress=lambda _message: None,
    )
    assert result.read_bytes() == b"abcdef"


def test_bimsyn_manifest_is_paired_and_detects_pairing_drift(tmp_path: Path) -> None:
    manifest = load_bimsyn_manifest()
    assert manifest["formats"]["ifc"]["count"] == 44
    assert manifest["formats"]["rvt"]["count"] == 44
    changed = copy.deepcopy(manifest)
    changed["formats"]["rvt"]["files"][0]["name"] = "unpaired_room.rvt"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="basenames are not paired"):
        load_bimsyn_manifest(changed_path)
    assert BIMSYNC_MANIFEST_PATH.is_file()


def test_bimsyn_directory_rejects_same_size_hash_corruption(tmp_path: Path) -> None:
    expected = b"correct"
    model = tmp_path / "room.ifc"
    model.write_bytes(expected)
    manifest = {
        "formats": {
            "ifc": {
                "files": [
                    {
                        "name": model.name,
                        "bytes": len(expected),
                        "sha256": hashlib.sha256(expected).hexdigest(),
                    }
                ]
            }
        }
    }
    verify_bimsyn_model_directory(tmp_path, "ifc", manifest=manifest)
    model.write_bytes(b"CORRECT")
    with pytest.raises(ValueError, match="Unexpected sha256"):
        verify_bimsyn_model_directory(tmp_path, "ifc", manifest=manifest)


def test_sharepoint_download_repairs_existing_same_size_corruption(tmp_path: Path) -> None:
    expected = b"canonical"
    target = tmp_path / "room.ifc"
    target.write_bytes(b"CORRUPTED")

    class FakeOpener:
        def __init__(self) -> None:
            self.timeout: float | None = None

        def open(self, _request: object, *, timeout: float) -> _FakeResponse:
            self.timeout = timeout
            return _FakeResponse(expected)

    opener = FakeOpener()
    outputs = download_sharepoint_files(
        opener,
        [
            {
                "name": "room.ifc",
                "bytes": len(expected),
                "sha256": hashlib.sha256(expected).hexdigest(),
                "server_relative_url": "/BIMSyn/room.ifc",
            }
        ],
        tmp_path,
        timeout=9.0,
        progress=lambda _message: None,
    )
    assert outputs == [target]
    assert target.read_bytes() == expected
    assert opener.timeout == 9.0
