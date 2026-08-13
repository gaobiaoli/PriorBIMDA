from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import pytest

from bim_priorda3.data import slabim
from bim_priorda3.data.slabim import (
    DownloadArtifact,
    DownloadManifest,
    download_file,
    download_regions,
    load_download_manifest,
    region_core_is_complete,
    region_has_rosbag,
    safe_extract,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest_for(relative_path: str, value: bytes) -> DownloadManifest:
    artifact = DownloadArtifact(relative_path, len(value), _sha256(value))
    return DownloadManifest(
        repository="https://example.invalid/resolve",
        revision=slabim.SLABIM_REVISION,
        files={relative_path: artifact},
    )


class _Response(io.BytesIO):
    def __init__(
        self,
        value: bytes,
        *,
        status: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(value)
        self.status = status
        self.headers = headers or {}


def test_pinned_manifest_covers_all_public_downloads() -> None:
    manifest = load_download_manifest()

    assert manifest.revision == "53bff75888bcf3ec7b2e1e3a65b0a278e80b65aa"
    assert len(manifest.files) == 15
    assert manifest.artifact("BIM.zip").sha256 == (
        "700a25338edd83bfd415c4446cef9e54878c5cf4468206aeecd7ae98fdd94cae"
    )
    assert manifest.artifact("sensor_data/5F_Region3.zip").bytes == 8_962_559_321
    assert manifest.artifact("calibration_files/cam_intrinsics.txt").sha256 == (
        "182f5af8d8c801ddc1ff933189e7b5ea4be2b5249427321fd3a5b2c1e48aee2d"
    )


def test_download_file_resumes_and_passes_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "sensor_data/test.zip"
    value = b"complete immutable artifact"
    destination = tmp_path / "test.zip"
    partial = tmp_path / "test.zip.part"
    partial.write_bytes(value[:9])
    captured: dict[str, object] = {}

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        captured["range"] = request.get_header("Range")  # type: ignore[attr-defined]
        captured["timeout"] = timeout
        return _Response(
            value[9:],
            status=206,
            headers={"Content-Range": f"bytes 9-{len(value) - 1}/{len(value)}"},
        )

    monkeypatch.setattr(slabim, "urlopen", fake_urlopen)
    result = download_file(
        relative,
        destination,
        manifest=_manifest_for(relative, value),
        timeout=7.5,
        progress=lambda _message: None,
    )

    assert result == destination.resolve()
    assert destination.read_bytes() == value
    assert not partial.exists()
    assert captured == {"range": "bytes=9-", "timeout": 7.5}


def test_download_file_restarts_when_server_ignores_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "BIM.zip"
    value = b"whole response"
    destination = tmp_path / "BIM.zip"
    destination.with_name("BIM.zip.part").write_bytes(b"stale prefix")
    monkeypatch.setattr(
        slabim,
        "urlopen",
        lambda _request, timeout: _Response(value, status=200),
    )

    download_file(
        relative,
        destination,
        manifest=_manifest_for(relative, value),
        progress=lambda _message: None,
    )

    assert destination.read_bytes() == value


def test_download_file_rejects_unverifiable_partial_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "BIM.zip"
    value = b"complete"
    destination = tmp_path / "BIM.zip"
    partial = destination.with_name("BIM.zip.part")
    partial.write_bytes(value[:3])
    monkeypatch.setattr(
        slabim,
        "urlopen",
        lambda _request, timeout: _Response(value[3:], status=206),
    )

    with pytest.raises(OSError, match="without a valid Content-Range"):
        download_file(
            relative,
            destination,
            manifest=_manifest_for(relative, value),
            progress=lambda _message: None,
        )

    assert partial.read_bytes() == value[:3]
    assert not destination.exists()


def test_download_file_never_replaces_destination_with_bad_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "BIM.zip"
    expected = b"good"
    destination = tmp_path / "BIM.zip"
    destination.write_bytes(b"old")
    monkeypatch.setattr(
        slabim,
        "urlopen",
        lambda _request, timeout: _Response(b"evil", status=200),
    )

    with pytest.raises(ValueError, match="Unexpected SHA256"):
        download_file(
            relative,
            destination,
            manifest=_manifest_for(relative, expected),
            progress=lambda _message: None,
        )

    assert destination.read_bytes() == b"old"


def test_download_file_reuses_verified_local_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative = "BIM.zip"
    value = b"already local"
    destination = tmp_path / "BIM.zip"
    destination.write_bytes(value)

    def unexpected_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a verified local artifact must not be downloaded again")

    monkeypatch.setattr(slabim, "urlopen", unexpected_urlopen)
    download_file(
        relative,
        destination,
        manifest=_manifest_for(relative, value),
        progress=lambda _message: None,
    )


def test_safe_extract_filters_rosbag() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "region.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("5F_Region2/images/data/000.png", b"image")
            zipped.writestr("5F_Region2/rosbag/part.bag", b"bag")
        core = root / "core"
        bag = root / "bag"
        assert safe_extract(archive, core, mode="core") == 1
        assert (core / "5F_Region2/images/data/000.png").exists()
        assert not (core / "5F_Region2/rosbag/part.bag").exists()
        assert safe_extract(archive, bag, mode="rosbag") == 1
        assert (bag / "5F_Region2/rosbag/part.bag").exists()
        assert not (bag / "5F_Region2/images/data/000.png").exists()


def test_safe_extract_rejects_parent_traversal() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        archive = root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("../escape.txt", b"unsafe")
        with pytest.raises(ValueError):
            safe_extract(archive, root / "output")


def test_safe_extract_rejects_windows_parent_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr(r"..\escape.txt", b"unsafe")

    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        safe_extract(archive, tmp_path / "output")


def test_safe_extract_rejects_symbolic_links(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        link = zipfile.ZipInfo("5F_Region2/images/data/link.png")
        link.create_system = 3
        link.external_attr = 0o120777 << 16
        zipped.writestr(link, "../../outside")

    with pytest.raises(ValueError, match="Symbolic link"):
        safe_extract(archive, tmp_path / "output")


def _core_files(region: str, *, image_value: bytes = b"image") -> dict[str, bytes]:
    prefix = f"{region}/"
    return {
        prefix + "images/data/000000.png": image_value,
        prefix + "images/timestamps.txt": b"1.0\n",
        prefix + "points/data/000000.pcd": b"pcd",
        prefix + "points/timestamps.txt": b"1.0\n",
        prefix + "points/pose_frame_to_bim.txt": b"1 0 0 0 0 0 0 1\n",
        prefix + "map/data/uncolorized.ply": b"map",
        prefix + "map/pose_map_to_bim.txt": b"pose",
        prefix + "submap/data/000000.pcd": b"submap",
        prefix + "submap/pose_submap_to_bim.txt": b"pose",
    }


def _write_zip(path: Path, files: dict[str, bytes]) -> bytes:
    with zipfile.ZipFile(path, "w") as zipped:
        for name, value in files.items():
            zipped.writestr(name, value)
    return path.read_bytes()


def _write_manifest(path: Path, relative_path: str, archive: bytes) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset": "test",
                "repository": "https://example.invalid/resolve",
                "revision": slabim.SLABIM_REVISION,
                "files": {
                    relative_path: {
                        "bytes": len(archive),
                        "sha256": _sha256(archive),
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _fake_archive_download(source: Path):
    def download(
        _relative: str,
        destination: Path,
        *_args: object,
        **_kwargs: object,
    ) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination

    return download


def _write_core(root: Path, region: str) -> Path:
    region_root = root / "sensor_data" / region
    for name, value in _core_files(region).items():
        relative = Path(name).relative_to(region)
        target = region_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    return region_root


def test_region_download_stages_then_replaces_incomplete_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = "5F_Region2"
    relative = f"sensor_data/{region}.zip"
    source_archive = tmp_path / "source.zip"
    archive = _write_zip(source_archive, _core_files(region, image_value=b"official"))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, relative, archive)
    final = tmp_path / "dataset/sensor_data" / region
    (final / "images/data").mkdir(parents=True)
    (final / "images/data/000000.png").write_bytes(b"incomplete")
    (final / "user-note.txt").write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(slabim, "download_shared_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(slabim, "download_file", _fake_archive_download(source_archive))

    download_regions(
        tmp_path / "dataset",
        [region],
        manifest_path=manifest,
        progress=lambda _message: None,
    )

    assert region_core_is_complete(final)
    assert (final / "images/data/000000.png").read_bytes() == b"official"
    assert (final / "user-note.txt").read_text(encoding="utf-8") == "preserve me"
    assert not (final.parent / f".{region}.bim_priorda3_previous").exists()
    (final / "points/data/000000.pcd").unlink()
    assert not region_core_is_complete(final)


def test_failed_staged_validation_keeps_existing_region_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = "5F_Region2"
    relative = f"sensor_data/{region}.zip"
    source_archive = tmp_path / "source.zip"
    archive = _write_zip(
        source_archive,
        {
            f"{region}/images/data/000000.png": b"only one image",
            f"{region}/points/data/000000.pcd": b"only one point cloud",
        },
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, relative, archive)
    final = tmp_path / "dataset/sensor_data" / region
    final.mkdir(parents=True)
    sentinel = final / "do-not-touch.txt"
    sentinel.write_text("original", encoding="utf-8")
    monkeypatch.setattr(slabim, "download_shared_files", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(slabim, "download_file", _fake_archive_download(source_archive))

    with pytest.raises(RuntimeError, match="staged core data validation failed"):
        download_regions(
            tmp_path / "dataset",
            [region],
            manifest_path=manifest,
            progress=lambda _message: None,
        )

    assert sentinel.read_text(encoding="utf-8") == "original"


def test_rosbag_only_is_atomically_added_to_complete_core(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    region = "5F_Region2"
    relative = f"sensor_data/{region}.zip"
    source_archive = tmp_path / "source.zip"
    archive = _write_zip(
        source_archive,
        {f"{region}/rosbag/data_0.bag": b"complete bag"},
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, relative, archive)
    dataset = tmp_path / "dataset"
    final = _write_core(dataset, region)
    stale = final / "rosbag/partial.bag"
    stale.parent.mkdir()
    stale.write_bytes(b"")
    monkeypatch.setattr(slabim, "download_file", _fake_archive_download(source_archive))

    download_regions(
        dataset,
        [region],
        rosbag_only=True,
        manifest_path=manifest,
        progress=lambda _message: None,
    )

    assert region_core_is_complete(final)
    assert region_has_rosbag(final)
    assert (final / "rosbag/data_0.bag").read_bytes() == b"complete bag"
    assert not stale.exists()
    (final / "rosbag/data_0.bag").write_bytes(b"truncated")
    assert not region_has_rosbag(final)
