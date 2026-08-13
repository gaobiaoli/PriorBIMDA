#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.preparation import DA3PredictionProvider
from bim_priorda3.data.stanford2d3ds import discover_stanford_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache single-frame DA3 for Stanford Area_1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rooms", nargs="*")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--finalize-legacy",
        action="store_true",
        help=(
            "Validate and atomically add schema/provenance to existing legacy "
            "depth+confidence NPZ files; never runs inference"
        ),
    )
    parser.add_argument(
        "--legacy-generation-attestation",
        help=(
            "Required free-text statement identifying how the legacy caches were "
            "generated and why the configured model/revision/process resolution apply"
        ),
    )
    parser.add_argument(
        "--migration-receipt",
        help="JSON receipt path for --finalize-legacy (defaults inside the cache directory)",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    area_root = resolve_project_path(cfg, cfg.data.stanford_area_root)
    ifc_root = resolve_project_path(cfg, cfg.data.bimsyn_ifc_root)
    ifc_rooms = {path.stem for path in ifc_root.glob("*.ifc")}
    frames = discover_stanford_frames(
        area_root,
        available_bim_rooms=ifc_rooms,
    )
    if args.rooms:
        selected = set(args.rooms)
        unknown = sorted(selected - {frame.room for frame in frames})
        if unknown:
            raise ValueError(f"Unknown Area_1 rooms: {unknown}")
        frames = [frame for frame in frames if frame.room in selected]
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        frames = frames[: args.max_frames]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards>=1 and 0<=shard_index<num_shards")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    if args.finalize_legacy:
        if not args.legacy_generation_attestation or not args.legacy_generation_attestation.strip():
            raise ValueError("--finalize-legacy requires --legacy-generation-attestation")
    elif args.legacy_generation_attestation or args.migration_receipt:
        raise ValueError(
            "--legacy-generation-attestation/--migration-receipt require --finalize-legacy"
        )
    frames = frames[args.shard_index :: args.num_shards]

    output_root = resolve_project_path(cfg, cfg.data.processed_root)
    provider = DA3PredictionProvider(cfg, "area_1", output_root / "da3_cache" / "area_1")
    target_shape = (int(cfg.data.target_height), int(cfg.data.target_width))
    if args.finalize_legacy:
        # Preflight the complete selection before changing a single artifact. This
        # catches missing, malformed, or concurrently incomplete cache files.
        inspections = [
            provider.inspect_legacy_cache(frame.rgb_path, target_shape) for frame in frames
        ]
        results: list[dict[str, Any]] = []
        start = time.perf_counter()
        for index, (frame, inspection) in enumerate(
            zip(frames, inspections, strict=True),
            start=1,
        ):
            result = provider.finalize_legacy_cache(
                frame.rgb_path,
                target_shape,
                generation_attestation=str(args.legacy_generation_attestation),
                inspection=inspection,
            )
            results.append(
                {
                    "id": frame.sample_id,
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in result.items()
                    },
                }
            )
            if index == 1 or index % args.log_every == 0 or index == len(frames):
                print(
                    f"[finalize {index}/{len(frames)}] {frame.sample_id}: {result['status']}",
                    flush=True,
                )
        config_path = Path(args.config).expanduser().resolve()
        receipt_path = (
            Path(args.migration_receipt).expanduser().resolve()
            if args.migration_receipt
            else provider.write_cache / "legacy_finalization_receipt.json"
        )
        receipt = {
            "schema_version": 1,
            "operation": "stanford-da3-legacy-cache-finalization-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "warning": (
                "Legacy predictions were not rerun. Model identity is a user-supplied "
                "generation attestation, not cryptographic proof of the original process."
            ),
            "generation_attestation": str(args.legacy_generation_attestation).strip(),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "model_name": provider.model_name,
            "model_revision": provider.model_revision,
            "local_files_only": provider.local_files_only,
            "process_res": provider.process_res,
            "target_shape": list(target_shape),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "frames": len(results),
            "status_counts": {
                status: sum(result["status"] == status for result in results)
                for status in sorted({str(result["status"]) for result in results})
            },
            "artifacts": results,
            "elapsed_s": time.perf_counter() - start,
        }
        _atomic_write_json(receipt_path, receipt)
        print(f"Wrote auditable migration receipt: {receipt_path}", flush=True)
        return

    start = time.perf_counter()
    for index, frame in enumerate(frames, start=1):
        prediction = provider.get_with_provenance(frame.rgb_path, target_shape)
        elapsed = time.perf_counter() - start
        remaining = elapsed / index * (len(frames) - index)
        if index == 1 or index % args.log_every == 0 or index == len(frames):
            print(
                f"[shard {args.shard_index}/{args.num_shards} "
                f"{index}/{len(frames)}] {frame.sample_id}: {prediction.source}; "
                f"elapsed={elapsed:.1f}s ETA={remaining / 60:.1f}min",
                flush=True,
            )


if __name__ == "__main__":
    main()
