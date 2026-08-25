from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import tarfile
from collections.abc import Callable, Iterable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

STANFORD_AREA1_URL = "https://cvg-data.inf.ethz.ch/2d3ds/no_xyz/area_1_no_xyz.tar"
STANFORD_AREA1_BYTES = 32_684_605_440
STANFORD_AREA1_MD5 = "21098fbe93b561e30e79197a95fa4fd2"
STANFORD_LICENSE_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLScFR0U8WEUtb7tgjOhhnl31OrkEs73-Y8bQwPeXgebqVKNMpQ/viewform"
)
STANFORD_METADATA_REVISION = "54a532959b20203ea4c3fcc26f5c8bf678d6fdb4"
STANFORD_LABELS_URL = (
    "https://raw.githubusercontent.com/alexsax/2D-3D-Semantics/"
    f"{STANFORD_METADATA_REVISION}/assets/semantic_labels.json"
)
STANFORD_LABELS_BYTES = 250_159
STANFORD_LABELS_SHA256 = "735765d68387ce324fc3d5c1cd49c32791c5ac0ab179ff06fd35defdaab08110"
STANFORD_SEMANTIC_OBJ_BYTES = 22_822_093
STANFORD_SEMANTIC_OBJ_SHA256 = "7983cb80bf8bbcec05639f9e1acf9439f1220a14737a7fb9ee6c66ce279ef5c5"
STANFORD_SEMANTIC_MTL_BYTES = 327_876
STANFORD_SEMANTIC_MTL_SHA256 = "875d3f2205ca5305d860ff3db18f49fb0c3b2503b44025d66dfea3c193c8cf86"

BIMSYNC_SHARE_URL = (
    "https://szueducn-my.sharepoint.com/:f:/g/personal/"
    "shengjuntang_szu_edu_cn/"
    "EgpOf3leEiFOjtfQTl-k7GgB9NHyLKhaRCnaUhD-jD0-yw?e=Zo37wP"
)
BIMSYNC_SITE = "https://szueducn-my.sharepoint.com"
BIMSYNC_API_ROOT = f"{BIMSYNC_SITE}/personal/shengjuntang_szu_edu_cn/_api/web"
BIMSYNC_MODEL_ROOT = "/personal/shengjuntang_szu_edu_cn/Documents/BIMSyn/BIM model"
BIMSYNC_EXPECTED = {
    "ifc": {"count": 44, "bytes": 125_546_311},
    "rvt": {"count": 44, "bytes": 603_906_048},
}
BIMSYNC_MANIFEST_PATH = Path(__file__).with_name("bimsyn_models_manifest.json")

_USER_AGENT = "Mozilla/5.0 (compatible; BIM-PriorDA3 reproducibility downloader)"
_HTTP_TIMEOUT_SECONDS = 60.0
_AREA1_SELECTED_PREFIXES = (
    "area_1/data/rgb/",
    "area_1/data/depth/",
    "area_1/data/pose/",
    "area_1/data/semantic/",
)
_AREA1_PANO_PREFIXES = (
    "area_1/pano/rgb/",
    "area_1/pano/depth/",
    "area_1/pano/pose/",
    "area_1/pano/semantic/",
)
_AREA1_SELECTED_FILES = {
    "area_1/3d/semantic.obj",
    "area_1/3d/semantic.mtl",
}
_AREA1_MODALITY_PATTERNS = {
    "rgb": "*.png",
    "depth": "*.png",
    "pose": "*.json",
    "semantic": "*.png",
}
_AREA1_PANO_MODALITY_PATTERNS = dict(_AREA1_MODALITY_PATTERNS)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTENT_RANGE_PATTERN = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_file(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_digest: str | None,
    digest_algorithm: str,
) -> None:
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        raise ValueError(
            f"Unexpected size for {path}: {path.stat().st_size} != {expected_bytes} bytes"
        )
    if expected_digest is not None:
        actual = file_digest(path, digest_algorithm)
        if actual != expected_digest:
            raise ValueError(
                f"Unexpected {digest_algorithm} for {path}: {actual} != {expected_digest}"
            )


def load_bimsyn_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load and structurally validate the committed BIMSyn byte/hash manifest."""

    manifest_path = (path or BIMSYNC_MANIFEST_PATH).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"BIMSyn integrity manifest is missing: {manifest_path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported BIMSyn manifest schema in {manifest_path}")
    formats = manifest.get("formats")
    if not isinstance(formats, dict) or set(formats) != {"ifc", "rvt"}:
        raise ValueError("BIMSyn manifest must contain exactly IFC and RVT inventories")

    stems: dict[str, set[str]] = {}
    for extension in ("ifc", "rvt"):
        section = formats.get(extension)
        if not isinstance(section, dict) or not isinstance(section.get("files"), list):
            raise TypeError(f"Malformed BIMSyn {extension.upper()} manifest section")
        files = section["files"]
        names: set[str] = set()
        total_bytes = 0
        format_stems: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise TypeError(f"Malformed BIMSyn {extension.upper()} file record")
            name = item.get("name")
            size = item.get("bytes")
            sha256 = item.get("sha256")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or "\\" in name
                or Path(name).suffix.lower() != f".{extension}"
            ):
                raise ValueError(f"Unsafe BIMSyn manifest filename: {name!r}")
            if name in names:
                raise ValueError(f"Duplicate BIMSyn manifest filename: {name}")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"Invalid byte size for BIMSyn file {name}: {size!r}")
            if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
                raise ValueError(f"Invalid SHA256 for BIMSyn file {name}")
            names.add(name)
            format_stems.add(Path(name).stem)
            total_bytes += size
        expected = BIMSYNC_EXPECTED[extension]
        if len(files) != expected["count"] or total_bytes != expected["bytes"]:
            raise ValueError(
                f"BIMSyn {extension.upper()} manifest aggregate mismatch: "
                f"count={len(files)}, bytes={total_bytes}"
            )
        if section.get("count") != len(files) or section.get("total_bytes") != total_bytes:
            raise ValueError(f"BIMSyn {extension.upper()} manifest summary is inconsistent")
        stems[extension] = format_stems
    if stems["ifc"] != stems["rvt"]:
        missing_rvt = sorted(stems["ifc"] - stems["rvt"])
        missing_ifc = sorted(stems["rvt"] - stems["ifc"])
        raise ValueError(
            "BIMSyn manifest IFC/RVT basenames are not paired one-to-one: "
            f"missing_rvt={missing_rvt}, missing_ifc={missing_ifc}"
        )
    return manifest


def bimsyn_manifest_files(
    extension: str,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extension = extension.lower()
    if extension not in BIMSYNC_EXPECTED:
        raise ValueError(f"Unsupported BIMSyn model extension: {extension}")
    loaded = load_bimsyn_manifest() if manifest is None else manifest
    files = loaded["formats"][extension]["files"]
    return [dict(item) for item in files]


def verify_bimsyn_model_directory(
    directory: Path,
    extension: str,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Verify an IFC or RVT folder against every committed size and SHA256."""

    extension = extension.lower()
    loaded = load_bimsyn_manifest() if manifest is None else manifest
    if manifest is not None:
        formats = manifest.get("formats") if isinstance(manifest, dict) else None
        if not isinstance(formats, dict) or extension not in formats:
            raise ValueError(f"Malformed BIMSyn manifest for {extension}")
    expected = bimsyn_manifest_files(extension, manifest=loaded)
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    actual_paths = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() == f".{extension}"
        ),
        key=lambda path: path.name,
    )
    expected_names = {str(item["name"]) for item in expected}
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        raise ValueError(
            f"BIMSyn {extension.upper()} filenames differ from the canonical manifest: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for item in expected:
        _validate_file(
            directory / str(item["name"]),
            expected_bytes=int(item["bytes"]),
            expected_digest=str(item["sha256"]),
            digest_algorithm="sha256",
        )
    return expected


def download_url(
    url: str,
    destination: Path,
    *,
    expected_bytes: int | None = None,
    expected_digest: str | None = None,
    digest_algorithm: str = "sha256",
    chunk_size: int = 16 * 1024 * 1024,
    timeout: float = _HTTP_TIMEOUT_SECONDS,
    progress: Callable[[str], None] = print,
) -> Path:
    """Resume an HTTP download and atomically publish a verified artifact."""

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_file(
            destination,
            expected_bytes=expected_bytes,
            expected_digest=expected_digest,
            digest_algorithm=digest_algorithm,
        )
        progress(f"[skip] verified {destination}")
        return destination

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urlopen(request, timeout=timeout) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        if append:
            content_range = response.headers.get("Content-Range")
            match = (
                _CONTENT_RANGE_PATTERN.fullmatch(content_range.strip())
                if isinstance(content_range, str)
                else None
            )
            if match is None:
                raise OSError(
                    f"Server returned HTTP 206 without a valid Content-Range for {destination}"
                )
            range_start = int(match.group(1))
            if range_start != offset:
                raise OSError(
                    f"Resume response for {destination} starts at {range_start}, expected {offset}"
                )
            total = match.group(3)
            if expected_bytes is not None and total != "*" and int(total) != expected_bytes:
                raise OSError(
                    f"Resume response for {destination} reports {total} total bytes, "
                    f"expected {expected_bytes}"
                )
        if not append:
            offset = 0
        mode = "ab" if append else "wb"
        with partial.open(mode) as output:
            copied = offset
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                output.write(block)
                copied += len(block)
                progress(f"  {destination.name}: {copied / 1024**3:.2f} GiB")
    _validate_file(
        partial,
        expected_bytes=expected_bytes,
        expected_digest=expected_digest,
        digest_algorithm=digest_algorithm,
    )
    os.replace(partial, destination)
    return destination


def _normalized_tar_member(name: str) -> PurePosixPath:
    if "\\" in name:
        raise ValueError(f"Unsafe Windows-style path in Stanford TAR: {name}")
    raw = PurePosixPath(name)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"Unsafe path in Stanford TAR: {name}")
    parts = tuple(part for part in raw.parts if part != ".")
    return PurePosixPath(*parts)


def _selected_area1_member(name: str, *, include_pano: bool = False) -> bool:
    normalized = _normalized_tar_member(name).as_posix()
    selected_prefixes = (
        _AREA1_SELECTED_PREFIXES + _AREA1_PANO_PREFIXES
        if include_pano
        else _AREA1_SELECTED_PREFIXES
    )
    return normalized in _AREA1_SELECTED_FILES or normalized.startswith(selected_prefixes)


def stanford_area1_inventory(root: Path) -> dict[str, int]:
    area = root.expanduser().resolve() / "area_1"
    counts = {
        modality: len(list((area / "data" / modality).glob(pattern)))
        for modality, pattern in _AREA1_MODALITY_PATTERNS.items()
    }
    counts["mesh"] = sum(
        (area / "3d" / name).is_file() for name in ("semantic.obj", "semantic.mtl")
    )
    return counts


def stanford_area1_pano_inventory(root: Path) -> dict[str, int]:
    """Count the optional equirectangular modalities used by pano evaluation."""

    pano = root.expanduser().resolve() / "area_1" / "pano"
    return {
        modality: len(list((pano / modality).glob(pattern)))
        for modality, pattern in _AREA1_PANO_MODALITY_PATTERNS.items()
    }


def _area1_modality_frame_keys(
    area: Path,
    modality: str,
    *,
    projection: str = "data",
) -> set[str]:
    patterns = _AREA1_MODALITY_PATTERNS if projection == "data" else _AREA1_PANO_MODALITY_PATTERNS
    if projection not in {"data", "pano"}:
        raise ValueError(f"Unknown Area_1 projection directory: {projection}")
    pattern = patterns[modality]
    paths = (area / projection / modality).glob(pattern)
    suffix = f"_domain_{modality}"
    frame_keys: set[str] = set()
    for path in paths:
        base = path.name[: -len(path.suffix)]
        if not base.endswith(suffix):
            raise ValueError(
                f"Unexpected Area_1 {modality} filename (missing {suffix!r}): {path.name}"
            )
        frame_key = base[: -len(suffix)]
        if frame_key in frame_keys:
            raise ValueError(f"Duplicate Area_1 {modality} frame basename: {frame_key}")
        frame_keys.add(frame_key)
    return frame_keys


def _verify_area1_modality_pairing(area: Path, *, projection: str = "data") -> None:
    frame_keys = {
        modality: _area1_modality_frame_keys(area, modality, projection=projection)
        for modality in _AREA1_MODALITY_PATTERNS
    }
    reference = frame_keys["rgb"]
    for modality in ("depth", "pose", "semantic"):
        if frame_keys[modality] != reference:
            raise ValueError(
                f"Area_1 {modality} basenames do not pair one-to-one with RGB: "
                f"missing={sorted(reference - frame_keys[modality])[:5]}, "
                f"unexpected={sorted(frame_keys[modality] - reference)[:5]}"
            )


def verify_stanford_area1_mesh(root: Path) -> dict[str, dict[str, Any]]:
    root = root.expanduser().resolve()
    mesh_specs = {
        "semantic.obj": (STANFORD_SEMANTIC_OBJ_BYTES, STANFORD_SEMANTIC_OBJ_SHA256),
        "semantic.mtl": (STANFORD_SEMANTIC_MTL_BYTES, STANFORD_SEMANTIC_MTL_SHA256),
    }
    audit: dict[str, dict[str, Any]] = {}
    for name, (expected_bytes, expected_sha256) in mesh_specs.items():
        path = root / "area_1" / "3d" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        _validate_file(
            path,
            expected_bytes=expected_bytes,
            expected_digest=expected_sha256,
            digest_algorithm="sha256",
        )
        audit[name] = {
            "path": str(path),
            "bytes": expected_bytes,
            "sha256": expected_sha256,
        }
    return audit


def verify_stanford_area1_extraction(root: Path) -> dict[str, int]:
    """Verify selected Area_1 files, mesh hashes, and cross-modality pairing."""

    root = root.expanduser().resolve()
    area = root / "area_1"
    counts = stanford_area1_inventory(root)
    for modality in _AREA1_MODALITY_PATTERNS:
        if counts[modality] != 10_327:
            raise ValueError(
                f"Area_1 extraction expected 10327 {modality} files, got {counts[modality]}"
            )
    verify_stanford_area1_mesh(root)
    _verify_area1_modality_pairing(area)
    return counts


def verify_stanford_area1_pano_extraction(root: Path) -> dict[str, int]:
    """Verify the optional 190-station equirectangular RGB/depth/pose/semantic set."""

    root = root.expanduser().resolve()
    area = root / "area_1"
    counts = stanford_area1_pano_inventory(root)
    for modality in _AREA1_PANO_MODALITY_PATTERNS:
        if counts[modality] != 190:
            raise ValueError(
                f"Area_1 pano extraction expected 190 {modality} files, got {counts[modality]}"
            )
    _verify_area1_modality_pairing(area, projection="pano")
    return counts


def verify_stanford_semantic_labels(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    _validate_file(
        path,
        expected_bytes=STANFORD_LABELS_BYTES,
        expected_digest=STANFORD_LABELS_SHA256,
        digest_algorithm="sha256",
    )
    return {
        "path": str(path),
        "bytes": STANFORD_LABELS_BYTES,
        "sha256": STANFORD_LABELS_SHA256,
        "revision": STANFORD_METADATA_REVISION,
    }


def stanford_area1_is_complete(root: Path, *, require_pano: bool = False) -> bool:
    try:
        verify_stanford_area1_extraction(root)
        if require_pano:
            verify_stanford_area1_pano_extraction(root)
    except (OSError, ValueError):
        return False
    return True


def extract_stanford_area1(
    archive: Path,
    destination: Path,
    *,
    include_pano: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Extract the regular benchmark inputs and, optionally, pano modalities."""

    archive = archive.expanduser().resolve()
    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    regular_complete = stanford_area1_is_complete(destination)
    pano_complete = False
    if include_pano:
        try:
            verify_stanford_area1_pano_extraction(destination)
        except (OSError, ValueError):
            pano_complete = False
        else:
            pano_complete = True
    if regular_complete and (not include_pano or pano_complete):
        counts = stanford_area1_inventory(destination)
        if include_pano:
            counts.update(
                {
                    f"pano_{key}": value
                    for key, value in stanford_area1_pano_inventory(destination).items()
                }
            )
        progress(f"[skip] verified Area_1 extraction at {destination / 'area_1'}")
        return counts
    with tarfile.open(archive, mode="r:") as bundle:
        for index, member in enumerate(bundle, 1):
            path = _normalized_tar_member(member.name)
            name = path.as_posix()
            if not member.isfile() or not _selected_area1_member(
                name,
                include_pano=include_pano,
            ):
                continue
            is_pano = name.startswith("area_1/pano/")
            if (is_pano and pano_complete) or (not is_pano and regular_complete):
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read selected TAR member: {member.name}")
            target = destination.joinpath(*path.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
            os.replace(temporary, target)
            if index % 10_000 == 0:
                progress(f"  scanned {index} TAR entries")
    try:
        counts = verify_stanford_area1_extraction(destination)
        if include_pano:
            counts.update(
                {
                    f"pano_{key}": value
                    for key, value in verify_stanford_area1_pano_extraction(destination).items()
                }
            )
        return counts
    except ValueError as error:
        raise RuntimeError(str(error)) from error


def _sharepoint_opener(*, timeout: float = _HTTP_TIMEOUT_SECONDS) -> Any:
    jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))
    opener.open(
        Request(BIMSYNC_SHARE_URL, headers={"User-Agent": _USER_AGENT}),
        timeout=timeout,
    ).close()
    return opener


def _odata_folder_url(folder: str) -> str:
    encoded = quote(folder, safe="/")
    return (
        f"{BIMSYNC_API_ROOT}/GetFolderByServerRelativeUrl('{encoded}')/Files"
        "?$select=Name,Length,ServerRelativeUrl"
    )


def _odata_file_url(server_relative_url: str) -> str:
    encoded = quote(server_relative_url, safe="/")
    return f"{BIMSYNC_API_ROOT}/GetFileByServerRelativeUrl('{encoded}')/$value"


def sharepoint_folder_files(
    opener: Any,
    folder: str,
    *,
    timeout: float = _HTTP_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    request = Request(
        _odata_folder_url(folder),
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "application/json;odata=nometadata",
        },
    )
    with opener.open(request, timeout=timeout) as response:
        payload = json.load(response)
    raw = payload.get("value")
    if raw is None and isinstance(payload.get("d"), dict):
        raw = payload["d"].get("results")
    if not isinstance(raw, list):
        raise TypeError(f"Unexpected SharePoint folder response for {folder}")
    files: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError(f"Malformed SharePoint file record in {folder}")
        name = str(item.get("Name", ""))
        relative = str(item.get("ServerRelativeUrl", ""))
        length = int(item.get("Length", -1))
        if not name or not relative or length < 0 or Path(name).name != name:
            raise RuntimeError(f"Unsafe or incomplete SharePoint file record: {item}")
        files.append({"name": name, "bytes": length, "server_relative_url": relative})
    return sorted(files, key=lambda item: item["name"])


def download_sharepoint_files(
    opener: Any,
    files: Iterable[dict[str, Any]],
    destination: Path,
    *,
    timeout: float = _HTTP_TIMEOUT_SECONDS,
    progress: Callable[[str], None] = print,
) -> list[Path]:
    """Download and SHA-verify SharePoint files; this endpoint does not support Range."""

    destination = destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for index, item in enumerate(files, 1):
        name = str(item["name"])
        if not name or Path(name).name != name or "\\" in name:
            raise ValueError(f"Unsafe SharePoint download filename: {name!r}")
        target = destination / name
        expected = int(item["bytes"])
        expected_sha256 = str(item.get("sha256", ""))
        if _SHA256_PATTERN.fullmatch(expected_sha256) is None:
            raise ValueError(f"Missing canonical SHA256 for SharePoint file {target.name}")
        if target.exists():
            try:
                _validate_file(
                    target,
                    expected_bytes=expected,
                    expected_digest=expected_sha256,
                    digest_algorithm="sha256",
                )
            except ValueError:
                progress(f"[repair] integrity mismatch for {target.name}")
            else:
                progress(f"[skip] verified {target.name}")
                outputs.append(target)
                continue
        temporary = target.with_name(target.name + ".part")
        request = Request(
            _odata_file_url(str(item["server_relative_url"])),
            headers={"User-Agent": _USER_AGENT},
        )
        with opener.open(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=16 * 1024 * 1024)
        _validate_file(
            temporary,
            expected_bytes=expected,
            expected_digest=expected_sha256,
            digest_algorithm="sha256",
        )
        os.replace(temporary, target)
        outputs.append(target)
        progress(f"[{index}] downloaded {target.name}")
    return outputs


def download_bimsyn_models(
    root: Path,
    *,
    include_rvt: bool,
    timeout: float = _HTTP_TIMEOUT_SECONDS,
    progress: Callable[[str], None] = print,
) -> dict[str, list[Path]]:
    manifest = load_bimsyn_manifest()
    opener = _sharepoint_opener(timeout=timeout)
    result: dict[str, list[Path]] = {}
    for extension in ("ifc", "rvt"):
        if extension == "rvt" and not include_rvt:
            continue
        folder = f"{BIMSYNC_MODEL_ROOT}/{extension}"
        remote_files = [
            item
            for item in sharepoint_folder_files(opener, folder, timeout=timeout)
            if str(item["name"]).lower().endswith(f".{extension}")
        ]
        expected_files = bimsyn_manifest_files(extension, manifest=manifest)
        expected_by_name = {str(item["name"]): item for item in expected_files}
        remote_by_name = {str(item["name"]): item for item in remote_files}
        if set(remote_by_name) != set(expected_by_name):
            raise RuntimeError(
                f"BIMSyn {extension.upper()} listing filenames changed: "
                f"missing={sorted(set(expected_by_name) - set(remote_by_name))}, "
                f"unexpected={sorted(set(remote_by_name) - set(expected_by_name))}"
            )
        files: list[dict[str, Any]] = []
        for name, expected in expected_by_name.items():
            remote = remote_by_name[name]
            if int(remote["bytes"]) != int(expected["bytes"]):
                raise RuntimeError(
                    f"BIMSyn {extension.upper()} listing size changed for {name}: "
                    f"{remote['bytes']} != {expected['bytes']}"
                )
            files.append({**remote, "sha256": str(expected["sha256"])})
        outputs = download_sharepoint_files(
            opener,
            files,
            root / "BIM_model" / extension,
            timeout=timeout,
            progress=progress,
        )
        verify_bimsyn_model_directory(
            root / "BIM_model" / extension,
            extension,
            manifest=manifest,
        )
        result[extension] = outputs
    return result
