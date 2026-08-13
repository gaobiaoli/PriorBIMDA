from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

SLABIM_REVISION = "53bff75888bcf3ec7b2e1e3a65b0a278e80b65aa"
SLABIM_REPOSITORY_ROOT = "https://huggingface.co/datasets/BobH62/SLABIM/resolve"
SLABIM_REPOSITORY = f"{SLABIM_REPOSITORY_ROOT}/{SLABIM_REVISION}"
SLABIM_MANIFEST_PATH = Path(__file__).with_name("slabim_download_manifest.json")
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
DEFAULT_REGIONS = (
    "3F_Region2",
    "3F_Region3",
    "4F_Region2",
    "4F_Region3",
    "5F_Region2",
    "5F_Region3",
)
ALL_REGIONS = (
    "1F_Region1",
    "1F_Region2",
    "1F_Region3",
    "3F_Region1",
    "3F_Region2",
    "3F_Region3",
    "4F_Region1",
    "4F_Region2",
    "4F_Region3",
    "5F_Region1",
    "5F_Region2",
    "5F_Region3",
)

_USER_AGENT = "Mozilla/5.0 (compatible; BIM-PriorDA3 reproducibility downloader)"
_CORE_RECEIPT = ".bim_priorda3_core.json"
_ROSBAG_RECEIPT = ".bim_priorda3_rosbag.json"
_BIM_RECEIPT = ".bim_priorda3_source.json"
_CONTENT_RANGE_PATTERN = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class DownloadArtifact:
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DownloadManifest:
    repository: str
    revision: str
    files: Mapping[str, DownloadArtifact]

    def artifact(self, relative_path: str) -> DownloadArtifact:
        try:
            return self.files[relative_path]
        except KeyError as exc:
            raise KeyError(
                f"SLABIM download is not pinned in the manifest: {relative_path}"
            ) from exc


def _safe_relative_path(value: str, *, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.parts[0].endswith(":")
    ):
        raise ValueError(f"Unsafe {context} path: {value!r}")
    return path


def load_download_manifest(path: Path = SLABIM_MANIFEST_PATH) -> DownloadManifest:
    """Load and strictly validate the pinned public SLABIM artifact manifest."""

    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported SLABIM manifest schema in {manifest_path}")
    revision = payload.get("revision")
    if revision != SLABIM_REVISION:
        raise ValueError(
            f"SLABIM manifest revision {revision!r} does not match pinned {SLABIM_REVISION}"
        )
    repository = payload.get("repository")
    if not isinstance(repository, str) or not repository.startswith("https://"):
        raise ValueError(f"Invalid SLABIM repository in {manifest_path}")
    raw_files = payload.get("files")
    if not isinstance(raw_files, dict) or not raw_files:
        raise ValueError(f"SLABIM manifest has no files: {manifest_path}")

    files: dict[str, DownloadArtifact] = {}
    for relative_path, raw in raw_files.items():
        if not isinstance(relative_path, str) or not isinstance(raw, dict):
            raise TypeError(f"Malformed SLABIM file record in {manifest_path}")
        _safe_relative_path(relative_path, context="manifest")
        byte_count = raw.get("bytes")
        digest = raw.get("sha256")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
            raise ValueError(f"Invalid byte count for {relative_path}: {byte_count!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid SHA256 for {relative_path}: {digest!r}")
        files[relative_path] = DownloadArtifact(relative_path, byte_count, digest)
    return DownloadManifest(repository=repository, revision=revision, files=files)


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_download(path: Path, artifact: DownloadArtifact) -> None:
    """Validate an artifact using the same size and SHA256 authority as download."""

    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    if actual_bytes != artifact.bytes:
        raise ValueError(
            f"Unexpected size for {artifact.relative_path}: "
            f"{actual_bytes} != {artifact.bytes} bytes"
        )
    actual_sha256 = file_sha256(path)
    if actual_sha256 != artifact.sha256:
        raise ValueError(
            f"Unexpected SHA256 for {artifact.relative_path}: {actual_sha256} != {artifact.sha256}"
        )


def _url(relative_path: str, repository: str) -> str:
    return f"{repository.rstrip('/')}/{quote(relative_path, safe='/')}"


def _response_status(response: Any) -> int | None:
    status = getattr(response, "status", None)
    if status is None and hasattr(response, "getcode"):
        status = response.getcode()
    return status


def download_file(
    relative_path: str,
    destination: Path,
    repository: str = SLABIM_REPOSITORY,
    chunk_size: int = 16 * 1024 * 1024,
    *,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    manifest: DownloadManifest | None = None,
    progress: Callable[[str], None] = print,
) -> Path:
    """Resume, verify, and atomically publish one immutable SLABIM artifact."""

    manifest = manifest or load_download_manifest()
    artifact = manifest.artifact(relative_path)
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            verify_download(destination, artifact)
        except (FileNotFoundError, ValueError):
            progress(f"[repair] unverified local artifact: {destination}")
        else:
            progress(f"[skip] verified {destination}")
            return destination

    partial = destination.with_name(destination.name + ".part")
    if partial.exists() and partial.stat().st_size >= artifact.bytes:
        try:
            verify_download(partial, artifact)
        except (FileNotFoundError, ValueError):
            partial.unlink()
        else:
            os.replace(partial, destination)
            progress(f"[publish] verified existing partial {destination}")
            return destination

    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(_url(relative_path, repository), headers={"User-Agent": _USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urlopen(request, timeout=timeout) as response:
        status = _response_status(response)
        append = offset > 0 and status == 206
        if append:
            content_range = response.headers.get("Content-Range")
            match = (
                _CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
                if isinstance(content_range, str)
                else None
            )
            if match is None:
                raise OSError(
                    f"HTTP 206 without a valid Content-Range for {relative_path}: {content_range!r}"
                )
            start, end, total = (int(value) for value in match.groups())
            if start != offset or end < start or total != artifact.bytes or end >= total:
                raise OSError(f"Unexpected Content-Range for {relative_path}: {content_range!r}")
        else:
            offset = 0
        mode = "ab" if append else "wb"
        copied = offset
        with partial.open(mode) as output:
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                output.write(block)
                copied += len(block)
                progress(f"  {relative_path}: {copied / 1024**3:.2f} GiB")
            output.flush()
            os.fsync(output.fileno())
    verify_download(partial, artifact)
    os.replace(partial, destination)
    progress(f"[publish] verified {destination}")
    return destination


def _member_selected(path: PurePosixPath, mode: str) -> bool:
    has_rosbag = any(part.lower() == "rosbag" for part in path.parts)
    if mode == "core":
        return not has_rosbag
    if mode == "rosbag":
        return has_rosbag
    if mode == "all":
        return True
    raise ValueError(f"Unknown extraction mode: {mode}")


def safe_extract(archive: Path, destination: Path, mode: str = "all") -> int:
    """Safely extract regular all, core-only, or rosbag-only ZIP members."""

    if mode not in {"all", "core", "rosbag"}:
        raise ValueError(f"Unknown extraction mode: {mode}")
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    seen: set[PurePosixPath] = set()
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        for index, member in enumerate(members, 1):
            path = _safe_relative_path(member.filename, context="ZIP")
            if not _member_selected(path, mode):
                continue
            if path in seen:
                raise ValueError(f"Duplicate path in ZIP: {member.filename}")
            seen.add(path)
            member_mode = member.external_attr >> 16
            if stat.S_ISLNK(member_mode):
                raise ValueError(f"Symbolic link is not allowed in ZIP: {member.filename}")
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                with zipped.open(member) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
                os.replace(temporary, target)
                extracted += 1
            if index % 1000 == 0:
                print(f"  scanned {index}/{len(members)} ZIP entries", flush=True)
    return extracted


def _find_region_root(extract_root: Path, region: str, mode: str) -> Path:
    required = ("rosbag",) if mode == "rosbag" else ("images", "points")
    candidates = [
        path
        for path in extract_root.rglob(region)
        if path.is_dir() and all((path / name).is_dir() for name in required)
    ]
    if candidates:
        return min(candidates, key=lambda path: len(path.parts))
    if all((extract_root / name).is_dir() for name in required):
        return extract_root
    raise FileNotFoundError(
        f"Could not locate extracted {region} root containing {', '.join(required)}"
    )


def _find_bim_root(extract_root: Path) -> Path:
    candidates = [
        path
        for path in (extract_root, *extract_root.rglob("BIM"))
        if path.is_dir()
        and all((path / floor).is_dir() for floor in ("1F", "2F", "3F", "4F", "5F"))
    ]
    if not candidates:
        raise FileNotFoundError("Could not locate the extracted SLABIM BIM directory")
    return min(candidates, key=lambda path: len(path.parts))


def _nonempty_line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(bool(line.strip()) for line in handle)
    except (OSError, UnicodeError):
        return 0


def _directory_has_file(path: Path, pattern: str = "*") -> bool:
    return path.is_dir() and any(candidate.is_file() for candidate in path.glob(pattern))


def _receipt_is_complete(root: Path, name: str, mode: str) -> bool:
    receipt = root / name
    if not receipt.is_file():
        return True  # Legacy extraction: structural checks below remain authoritative.
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("revision") != SLABIM_REVISION:
            return False
        records = payload.get("files")
        if not isinstance(records, list) or not records:
            return False
        for record in records:
            if not isinstance(record, dict):
                return False
            relative = _safe_relative_path(str(record.get("path", "")), context="receipt")
            if mode != "all" and not _member_selected(relative, mode):
                return False
            path = root.joinpath(*relative.parts)
            if not path.is_file() or path.stat().st_size != record.get("bytes"):
                return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def region_core_is_complete(region_root: Path) -> bool:
    image_timestamps = _nonempty_line_count(region_root / "images/timestamps.txt")
    point_timestamps = _nonempty_line_count(region_root / "points/timestamps.txt")
    pose_rows = _nonempty_line_count(region_root / "points/pose_frame_to_bim.txt")
    image_count = sum(1 for _ in (region_root / "images/data").glob("*.png"))
    point_count = sum(1 for _ in (region_root / "points/data").glob("*.pcd"))
    required = (
        region_root / "map/pose_map_to_bim.txt",
        region_root / "submap/pose_submap_to_bim.txt",
    )
    return (
        image_count > 0
        and image_count == image_timestamps
        and point_count > 0
        and point_count == point_timestamps == pose_rows
        and all(path.is_file() and path.stat().st_size > 0 for path in required)
        and _directory_has_file(region_root / "map/data")
        and _directory_has_file(region_root / "submap/data", "*.pcd")
        and _receipt_is_complete(region_root, _CORE_RECEIPT, "core")
    )


def region_has_rosbag(region_root: Path) -> bool:
    bags = list((region_root / "rosbag").glob("*.bag"))
    return (
        bool(bags)
        and all(path.is_file() and path.stat().st_size > 0 for path in bags)
        and _receipt_is_complete(region_root, _ROSBAG_RECEIPT, "rosbag")
    )


def bim_is_complete(bim_root: Path) -> bool:
    required = []
    for floor in ("1F", "2F", "3F", "4F", "5F"):
        required.append(bim_root / floor / "CAD" / f"{floor}.dxf")
        required.extend(
            bim_root / floor / "mesh" / f"{component}.ply"
            for component in ("columns", "doors", "floors", "walls")
        )
    return all(path.is_file() and path.stat().st_size > 0 for path in required) and (
        _receipt_is_complete(bim_root, _BIM_RECEIPT, "all")
    )


def _receipt_payload(root: Path, artifact: DownloadArtifact, mode: str) -> dict[str, Any]:
    files = []
    receipt_names = {_CORE_RECEIPT, _ROSBAG_RECEIPT, _BIM_RECEIPT}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in receipt_names:
            continue
        relative = PurePosixPath(path.relative_to(root).as_posix())
        if mode != "all" and not _member_selected(relative, mode):
            continue
        files.append({"path": relative.as_posix(), "bytes": path.stat().st_size})
    if not files:
        raise RuntimeError(f"No extracted files found for receipt mode={mode} in {root}")
    return {
        "schema_version": 1,
        "revision": SLABIM_REVISION,
        "source": {
            "path": artifact.relative_path,
            "bytes": artifact.bytes,
            "sha256": artifact.sha256,
        },
        "mode": mode,
        "files": files,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_missing_files(source: Path, destination: Path) -> None:
    """Preserve user/generated files that are absent from a fresh official extraction."""

    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        target = destination / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _publish_directory(candidate: Path, destination: Path) -> None:
    """Swap a staged directory into place, rolling back if publication fails."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.bim_priorda3_previous")
    if backup.exists() and not destination.exists():
        os.replace(backup, destination)
    if backup.exists():
        raise RuntimeError(
            f"Found an unfinished SLABIM publication backup: {backup}. Inspect it before retrying."
        )
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(candidate, destination)
    except BaseException:
        if had_destination and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _staging_directory(root: Path, prefix: str) -> Path:
    parent = root / ".extracting"
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=parent))


def download_shared_files(
    root: Path,
    repository: str = SLABIM_REPOSITORY,
    keep_archives: bool = False,
    *,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    manifest: DownloadManifest | None = None,
    progress: Callable[[str], None] = print,
) -> None:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = manifest or load_download_manifest()
    archives = root / ".downloads"
    bim_root = root / "BIM"
    if not bim_is_complete(bim_root):
        artifact = manifest.artifact("BIM.zip")
        archive = download_file(
            "BIM.zip",
            archives / "BIM.zip",
            repository,
            timeout=timeout,
            manifest=manifest,
            progress=progress,
        )
        staging = _staging_directory(root, "BIM")
        try:
            if not safe_extract(archive, staging, mode="core"):
                raise RuntimeError("BIM.zip contained no regular files")
            candidate = _find_bim_root(staging)
            if not bim_is_complete(candidate):
                raise RuntimeError("SLABIM BIM validation failed after extraction")
            _write_json_atomic(
                candidate / _BIM_RECEIPT,
                _receipt_payload(candidate, artifact, "all"),
            )
            _copy_missing_files(bim_root, candidate)
            _publish_directory(candidate, bim_root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if not keep_archives:
            archive.unlink()

    calibration = root / "calibration_files"
    for name in ("cam_intrinsics.txt", "cam_to_lidar.txt"):
        relative = f"calibration_files/{name}"
        download_file(
            relative,
            calibration / name,
            repository,
            timeout=timeout,
            manifest=manifest,
            progress=progress,
        )


def download_regions(
    root: Path,
    regions: Iterable[str],
    include_rosbag: bool = False,
    rosbag_only: bool = False,
    repository: str = SLABIM_REPOSITORY,
    keep_archives: bool = False,
    progress: Callable[[str], None] = print,
    *,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    manifest_path: Path = SLABIM_MANIFEST_PATH,
) -> None:
    """Download verified regions through staging and publish complete trees."""

    root = root.expanduser().resolve()
    requested = tuple(dict.fromkeys(regions))
    unknown = [region for region in requested if region not in ALL_REGIONS]
    if unknown:
        raise ValueError(f"Unknown SLABIM region(s): {', '.join(unknown)}")
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_download_manifest(manifest_path)
    if not rosbag_only:
        download_shared_files(
            root,
            repository,
            keep_archives,
            timeout=timeout,
            manifest=manifest,
            progress=progress,
        )

    for order, region in enumerate(requested, 1):
        final_root = root / "sensor_data" / region
        core_complete = region_core_is_complete(final_root)
        if rosbag_only and not core_complete:
            raise FileNotFoundError(
                f"{region}: --rosbag-only requires an already complete core extraction"
            )
        bag_complete = region_has_rosbag(final_root)
        need_core = not rosbag_only and not core_complete
        need_bag = (include_rosbag or rosbag_only) and not bag_complete
        if not need_core and not need_bag:
            progress(f"[{order}/{len(requested)}] {region}: already complete")
            continue
        mode = "all" if need_core and need_bag else ("core" if need_core else "rosbag")
        relative = f"sensor_data/{region}.zip"
        artifact = manifest.artifact(relative)
        archive = root / ".downloads" / f"{region}.zip"
        progress(f"[{order}/{len(requested)}] {region}: acquiring verified archive")
        download_file(
            relative,
            archive,
            repository,
            timeout=timeout,
            manifest=manifest,
            progress=progress,
        )
        staging = _staging_directory(root, region)
        try:
            progress(f"[{order}/{len(requested)}] {region}: extracting mode={mode}")
            count = safe_extract(archive, staging, mode=mode)
            if not count:
                raise RuntimeError(f"{region}: archive had no files for extraction mode={mode}")
            candidate = _find_region_root(staging, region, mode)
            core_receipt: dict[str, Any] | None = None
            bag_receipt: dict[str, Any] | None = None
            if need_core:
                if not region_core_is_complete(candidate):
                    raise RuntimeError(f"{region}: staged core data validation failed")
                core_receipt = _receipt_payload(candidate, artifact, "core")
                _write_json_atomic(candidate / _CORE_RECEIPT, core_receipt)
            if need_bag:
                if not region_has_rosbag(candidate):
                    raise RuntimeError(f"{region}: staged rosbag validation failed")
                bag_receipt = _receipt_payload(candidate, artifact, "rosbag")

            if need_core:
                if bag_receipt is not None:
                    _write_json_atomic(candidate / _ROSBAG_RECEIPT, bag_receipt)
                _copy_missing_files(final_root, candidate)
                _publish_directory(candidate, final_root)
            else:
                assert bag_receipt is not None
                _publish_directory(candidate / "rosbag", final_root / "rosbag")
                _write_json_atomic(final_root / _ROSBAG_RECEIPT, bag_receipt)

            if need_core and not region_core_is_complete(final_root):
                raise RuntimeError(f"{region}: core data validation failed after publication")
            if need_bag and not region_has_rosbag(final_root):
                raise RuntimeError(f"{region}: rosbag validation failed after publication")
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        if not keep_archives:
            archive.unlink()
        progress(f"[{order}/{len(requested)}] {region}: complete")
