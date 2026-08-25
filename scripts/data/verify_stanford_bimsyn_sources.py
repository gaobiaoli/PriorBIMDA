#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bim_priorda3.data.public_downloads import (
    BIMSYNC_MANIFEST_PATH,
    BIMSYNC_SHARE_URL,
    STANFORD_AREA1_BYTES,
    STANFORD_AREA1_MD5,
    STANFORD_AREA1_URL,
    STANFORD_LICENSE_URL,
    STANFORD_SEMANTIC_MTL_SHA256,
    STANFORD_SEMANTIC_OBJ_SHA256,
    file_digest,
    load_bimsyn_manifest,
    verify_bimsyn_model_directory,
    verify_stanford_area1_extraction,
    verify_stanford_area1_pano_extraction,
    verify_stanford_semantic_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify downloaded Area_1 and BIMSyn BIM files")
    parser.add_argument(
        "--area-tar",
        type=Path,
        help="Optional original TAR; verifies the official size and MD5 when retained",
    )
    parser.add_argument(
        "--area-root",
        required=True,
        type=Path,
        help="Extracted directory containing area_1/data and area_1/3d",
    )
    parser.add_argument(
        "--semantic-labels",
        type=Path,
        help=(
            "Pinned semantic_labels.json; defaults to "
            "<area-root>/../metadata/assets/semantic_labels.json"
        ),
    )
    parser.add_argument("--ifc-root", required=True, type=Path)
    parser.add_argument(
        "--require-pano",
        action="store_true",
        help="Also require and verify the 190 paired equirectangular stations",
    )
    parser.add_argument(
        "--rvt-root",
        type=Path,
        help="Optional RVT directory; IFC alone is sufficient for the computational pipeline",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    area_tar = args.area_tar.expanduser().resolve() if args.area_tar else None
    extracted_root = args.area_root.expanduser().resolve()
    area_root = extracted_root / "area_1"
    semantic_labels = (
        args.semantic_labels.expanduser().resolve()
        if args.semantic_labels is not None
        else extracted_root.parent / "metadata/assets/semantic_labels.json"
    )
    ifc_root = args.ifc_root.expanduser().resolve()
    rvt_root = args.rvt_root.expanduser().resolve() if args.rvt_root else None
    area_size: int | None = None
    area_md5: str | None = None
    if area_tar is not None:
        if not area_tar.is_file():
            raise FileNotFoundError(area_tar)
        area_size = area_tar.stat().st_size
        area_md5 = file_digest(area_tar, "md5")
        if area_size != STANFORD_AREA1_BYTES or area_md5 != STANFORD_AREA1_MD5:
            raise ValueError(
                "Area_1 archive does not match the official size/MD5: "
                f"size={area_size}, md5={area_md5}"
            )
    modality_counts = verify_stanford_area1_extraction(extracted_root)
    pano_modality_counts = (
        verify_stanford_area1_pano_extraction(extracted_root) if args.require_pano else None
    )
    labels_audit = verify_stanford_semantic_labels(semantic_labels)

    manifest = load_bimsyn_manifest()
    ifcs = verify_bimsyn_model_directory(ifc_root, "ifc", manifest=manifest)
    rvts = (
        verify_bimsyn_model_directory(rvt_root, "rvt", manifest=manifest)
        if rvt_root is not None
        else []
    )
    if rvt_root is not None and {Path(str(item["name"])).stem for item in ifcs} != {
        Path(str(item["name"])).stem for item in rvts
    }:
        raise ValueError("BIMSyn IFC/RVT room stems are not paired one-to-one")
    for item in ifcs:
        with (ifc_root / item["name"]).open("rb") as handle:
            signature = handle.read(13)
        if signature != b"ISO-10303-21;":
            raise ValueError(f"IFC file has an invalid STEP header: {item['name']}")
    ole_magic = bytes.fromhex("d0cf11e0a1b11ae1")
    for item in rvts:
        if rvt_root is None:
            raise AssertionError("RVT inventory exists without an RVT root")
        with (rvt_root / item["name"]).open("rb") as handle:
            if handle.read(8) != ole_magic:
                raise ValueError(f"RVT file has an invalid compound-file header: {item['name']}")

    receipt = {
        "schema_version": 2,
        "stanford_area1": {
            "archive": str(area_tar) if area_tar is not None else None,
            "bytes": area_size,
            "md5": area_md5,
            "official_md5": STANFORD_AREA1_MD5,
            "extracted_root": str(area_root),
            "modality_counts": modality_counts,
            "pano_modality_counts": pano_modality_counts,
            "semantic_obj_sha256": STANFORD_SEMANTIC_OBJ_SHA256,
            "semantic_mtl_sha256": STANFORD_SEMANTIC_MTL_SHA256,
            "semantic_labels": labels_audit,
            "source": STANFORD_AREA1_URL,
            "license_registration": STANFORD_LICENSE_URL,
        },
        "bimsyn": {
            "source": BIMSYNC_SHARE_URL,
            "manifest": str(BIMSYNC_MANIFEST_PATH.resolve()),
            "manifest_sha256": file_digest(BIMSYNC_MANIFEST_PATH, "sha256"),
            "ifc_root": str(ifc_root),
            "rvt_root": str(rvt_root) if rvt_root is not None else None,
            "room_count": len(ifcs),
            "ifc_total_bytes": sum(item["bytes"] for item in ifcs),
            "rvt_total_bytes": sum(item["bytes"] for item in rvts),
            "paired_stems": sorted(Path(str(item["name"])).stem for item in ifcs),
            "ifc_files": ifcs,
            "rvt_files": rvts,
        },
        "verification": {
            "area_official_size_and_md5": area_tar is not None,
            "area_extracted_modalities": True,
            "area_modality_basenames_one_to_one": True,
            "area_pano_modalities": args.require_pano,
            "area_semantic_obj_sha256": True,
            "area_semantic_labels_sha256": True,
            "bimsyn_ifc_per_file_size_sha256": True,
            "bimsyn_rvt_per_file_size_sha256": rvt_root is not None,
            "ifc_rvt_one_to_one": rvt_root is not None,
            "file_signature_checks": True,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt["verification"], indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
