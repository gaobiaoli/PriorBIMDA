from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable, Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile


SLABIM_REPOSITORY = "https://huggingface.co/datasets/BobH62/SLABIM/resolve/main"
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


def _url(relative_path: str, repository: str) -> str:
    return f"{repository.rstrip('/')}/{quote(relative_path, safe='/')}"


def download_file(
    relative_path: str,
    destination: Path,
    repository: str = SLABIM_REPOSITORY,
    chunk_size: int = 16 * 1024 * 1024,
) -> Path:
    """Download with a resumable .part file and atomically publish on success."""
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(_url(relative_path, repository))
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urlopen(request) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if append else "wb"
        if offset and not append:
            offset = 0
        expected = response.headers.get("Content-Length")
        with partial.open(mode) as output:
            copied = offset
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                output.write(block)
                copied += len(block)
                print(
                    f"\r  {relative_path}: {copied / 1024**3:.2f} GiB",
                    end="",
                    flush=True,
                )
        print(flush=True)
        if expected is not None:
            expected_total = offset + int(expected)
            if partial.stat().st_size != expected_total:
                raise OSError(
                    f"Incomplete download for {relative_path}: "
                    f"{partial.stat().st_size} != {expected_total} bytes"
                )
    os.replace(partial, destination)
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
    """Safely extract all, core-only, or rosbag-only ZIP members."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        for index, member in enumerate(members, 1):
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe path in ZIP: {member.filename}")
            if not _member_selected(path, mode):
                continue
            target = destination.joinpath(*path.parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
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


def region_core_is_complete(region_root: Path) -> bool:
    required = (
        region_root / "images/timestamps.txt",
        region_root / "points/timestamps.txt",
        region_root / "points/pose_frame_to_bim.txt",
    )
    return (
        all(path.is_file() for path in required)
        and any((region_root / "images/data").glob("*.png"))
        and any((region_root / "points/data").glob("*.pcd"))
    )


def region_has_rosbag(region_root: Path) -> bool:
    return any((region_root / "rosbag").glob("*.bag"))


def _merge_region(staging: Path, output: Path, region: str, mode: str) -> None:
    source = _find_region_root(staging, region, mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, output, dirs_exist_ok=True)


def download_shared_files(
    root: Path,
    repository: str = SLABIM_REPOSITORY,
    keep_archives: bool = False,
) -> None:
    root = root.resolve()
    archives = root / ".downloads"
    if not (root / "BIM").is_dir():
        archive = download_file("BIM.zip", archives / "BIM.zip", repository)
        safe_extract(archive, root, mode="core")
        if not keep_archives:
            archive.unlink()
    calibration = root / "calibration_files"
    calibration.mkdir(parents=True, exist_ok=True)
    for name in ("cam_intrinsics.txt", "cam_to_lidar.txt"):
        target = calibration / name
        if not target.exists():
            download_file(f"calibration_files/{name}", target, repository)


def download_regions(
    root: Path,
    regions: Iterable[str],
    include_rosbag: bool = False,
    rosbag_only: bool = False,
    repository: str = SLABIM_REPOSITORY,
    keep_archives: bool = False,
    progress: Callable[[str], None] = print,
) -> None:
    """Download selected regions without deleting or replacing existing user data."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not rosbag_only:
        download_shared_files(root, repository, keep_archives)
    requested = tuple(regions)
    for order, region in enumerate(requested, 1):
        if region not in ALL_REGIONS:
            raise ValueError(f"Unknown SLABIM region: {region}")
        final_root = root / "sensor_data" / region
        core_complete = region_core_is_complete(final_root)
        bag_complete = region_has_rosbag(final_root)
        need_core = not rosbag_only and not core_complete
        need_bag = (include_rosbag or rosbag_only) and not bag_complete
        if not need_core and not need_bag:
            progress(f"[{order}/{len(requested)}] {region}: already complete")
            continue
        mode = "all" if need_core and need_bag else ("core" if need_core else "rosbag")
        archive = root / ".downloads" / f"{region}.zip"
        staging = root / ".extracting" / region
        if staging.exists():
            shutil.rmtree(staging)
        progress(f"[{order}/{len(requested)}] {region}: downloading archive")
        download_file(f"sensor_data/{region}.zip", archive, repository)
        progress(f"[{order}/{len(requested)}] {region}: extracting mode={mode}")
        count = safe_extract(archive, staging, mode=mode)
        if not count:
            raise RuntimeError(f"{region}: archive had no files for extraction mode={mode}")
        _merge_region(staging, final_root, region, mode)
        shutil.rmtree(staging)
        if not keep_archives:
            archive.unlink()
        if need_core and not region_core_is_complete(final_root):
            raise RuntimeError(f"{region}: core data validation failed after extraction")
        if need_bag and not region_has_rosbag(final_root):
            raise RuntimeError(f"{region}: rosbag validation failed after extraction")
        progress(f"[{order}/{len(requested)}] {region}: complete")
