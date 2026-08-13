#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from bim_priorda3.data.public_downloads import (
    BIMSYNC_SHARE_URL,
    STANFORD_AREA1_BYTES,
    STANFORD_AREA1_MD5,
    STANFORD_AREA1_URL,
    STANFORD_LABELS_SHA256,
    STANFORD_LABELS_URL,
    STANFORD_LICENSE_URL,
    download_bimsyn_models,
    download_url,
    extract_stanford_area1,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download licensed Stanford Area_1 inputs and public BIMSyn BIM files"
    )
    parser.add_argument("--stanford-root", type=Path, default=Path("../Stanford2D3DS"))
    parser.add_argument("--bimsyn-root", type=Path, default=Path("../BIMSyn"))
    parser.add_argument(
        "--accept-stanford-license",
        action="store_true",
        help=f"Confirm that you registered for and accepted {STANFORD_LICENSE_URL}",
    )
    parser.add_argument(
        "--acknowledge-bimsyn-license",
        "--accept-bimsyn-terms",
        dest="acknowledge_bimsyn_license",
        action="store_true",
        help=(
            "Acknowledge that BIMSyn has no clear redistribution license in the "
            f"shared folder and that you independently confirmed authorized use: {BIMSYNC_SHARE_URL}"
        ),
    )
    parser.add_argument("--skip-stanford", action="store_true")
    parser.add_argument("--skip-bimsyn", action="store_true")
    parser.add_argument(
        "--include-rvt",
        action="store_true",
        help="Also download 44 RVT source files; IFC alone is sufficient for computation",
    )
    parser.add_argument(
        "--delete-area-archive",
        action="store_true",
        help="Delete the verified 30.44 GiB TAR after successful selective extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_stanford and args.skip_bimsyn:
        raise ValueError("Nothing selected: both --skip-stanford and --skip-bimsyn were set")
    if not args.skip_stanford and not args.accept_stanford_license:
        raise PermissionError(
            "Stanford Area_1 requires prior license registration; read the URL in --help "
            "and rerun with --accept-stanford-license"
        )
    if not args.skip_bimsyn and not args.acknowledge_bimsyn_license:
        raise PermissionError(
            "BIMSyn redistribution terms are unclear; independently confirm authorized "
            "use and rerun with --acknowledge-bimsyn-license"
        )

    if not args.skip_stanford:
        stanford_root = args.stanford_root.expanduser().resolve()
        no_xyz = stanford_root / "no_xyz"
        archive = download_url(
            STANFORD_AREA1_URL,
            no_xyz / "area_1_no_xyz.tar",
            expected_bytes=STANFORD_AREA1_BYTES,
            expected_digest=STANFORD_AREA1_MD5,
            digest_algorithm="md5",
        )
        download_url(
            STANFORD_LABELS_URL,
            stanford_root / "metadata/assets/semantic_labels.json",
            expected_digest=STANFORD_LABELS_SHA256,
        )
        counts = extract_stanford_area1(archive, no_xyz)
        print(f"Area_1 selective extraction complete: {counts}")
        if args.delete_area_archive:
            archive.unlink()
            print(f"Removed verified archive after extraction: {archive}")

    if not args.skip_bimsyn:
        outputs = download_bimsyn_models(
            args.bimsyn_root,
            include_rvt=args.include_rvt,
        )
        print(
            "BIMSyn download complete: "
            + ", ".join(f"{name}={len(paths)}" for name, paths in outputs.items())
        )


if __name__ == "__main__":
    main()
