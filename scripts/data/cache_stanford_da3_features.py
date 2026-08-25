#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.da3_features import (
    DA3_FEATURE_CACHE_SCHEMA_VERSION,
    feature_cache_path,
    record_ids_sha256,
    sha256_file,
)
from bim_priorda3.data.dataset import _read_manifest, relocate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen DA3 encoder features for prepared Stanford Area_1 records"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _atomic_save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _cached_file_matches(
    path: Path,
    *,
    sample_id: str,
    image_sha256: str,
    model_name: str,
    model_revision: str,
    process_res: int,
    layers: tuple[int, int],
    expected_shape: tuple[int, int, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as item:
            return (
                int(item["schema_version"]) == DA3_FEATURE_CACHE_SCHEMA_VERSION
                and str(item["sample_id"].item()) == sample_id
                and str(item["image_sha256"].item()) == image_sha256
                and str(item["model_name"].item()) == model_name
                and str(item["model_revision"].item()) == model_revision
                and int(item["process_res"]) == process_res
                and tuple(int(value) for value in item["layers"].tolist()) == layers
                and item["feature_mid"].shape == expected_shape
                and item["feature_deep"].shape == expected_shape
            )
    except (KeyError, OSError, ValueError):
        return False


def _reshape_feature(
    feature: torch.Tensor,
    *,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    if feature.ndim != 4 or feature.shape[1] != 1:
        raise RuntimeError(
            "DA3 auxiliary feature must have shape [B, 1, tokens, channels]; "
            f"got {tuple(feature.shape)}"
        )
    height, width = grid_shape
    if feature.shape[2] != height * width:
        raise RuntimeError(
            f"DA3 feature token count {feature.shape[2]} does not match grid {grid_shape}"
        )
    return (
        feature[:, 0]
        .transpose(1, 2)
        .reshape(feature.shape[0], feature.shape[-1], height, width)
        .detach()
        .to(device="cpu", dtype=torch.float16)
        .numpy()
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.log_every < 1:
        raise ValueError("--log-every must be positive")
    cfg = load_config(args.config)
    fusion = cfg.model.get("da3_feature_fusion", {})
    if not bool(fusion.get("enabled", False)):
        raise ValueError("model.da3_feature_fusion.enabled must be true")
    layers = tuple(int(value) for value in fusion.get("layers", (11, 23)))
    if len(layers) != 2 or layers[0] >= layers[1]:
        raise ValueError("model.da3_feature_fusion.layers must contain two increasing layers")
    channels = int(fusion.get("channels", 1024))
    process_res = int(cfg.data.da3_process_res)
    patch_size = 14
    if process_res % patch_size:
        raise ValueError("data.da3_process_res must be divisible by DA3 patch size 14")
    grid_shape = (process_res // patch_size, process_res // patch_size)
    expected_shape = (channels, *grid_shape)

    processed_root = resolve_project_path(cfg, cfg.data.processed_root)
    source_manifest = processed_root / "manifest.jsonl"
    records = _read_manifest(source_manifest)
    source_value = cfg.data.get("source_root")
    source_root = resolve_project_path(cfg, source_value) if source_value else None
    slabim_value = cfg.data.get("slabim_root")
    slabim_root = resolve_project_path(cfg, slabim_value) if slabim_value else source_root
    if slabim_root is None:
        raise ValueError("data.source_root or data.slabim_root must be configured")
    records = [
        relocate_record(record, processed_root, slabim_root, source_root) for record in records
    ]
    cache_root = resolve_project_path(cfg, cfg.data.da3_feature_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    model_name = str(cfg.data.da3_model)
    model_revision = str(cfg.data.da3_revision)
    if not model_revision:
        raise ValueError("A pinned data.da3_revision is required")
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise RuntimeError("Install the optional depth-anything-3 dependency") from exc
    api = DepthAnything3.from_pretrained(
        model_name,
        revision=model_revision,
        local_files_only=bool(cfg.data.get("local_files_only", False)),
    ).to(args.device).eval()

    start = time.perf_counter()
    generated = 0
    reused = 0
    for offset in range(0, len(records), args.batch_size):
        selected = records[offset : offset + args.batch_size]
        pending: list[tuple[dict[str, Any], str, Path]] = []
        for record in selected:
            image_path = Path(record["image"])
            image_hash = sha256_file(image_path)
            output_path = feature_cache_path(cache_root, str(record["id"]))
            if not args.overwrite and _cached_file_matches(
                output_path,
                sample_id=str(record["id"]),
                image_sha256=image_hash,
                model_name=model_name,
                model_revision=model_revision,
                process_res=process_res,
                layers=layers,
                expected_shape=expected_shape,
            ):
                reused += 1
            else:
                pending.append((record, image_hash, output_path))
        if pending:
            processed_images, _, _ = api._preprocess_inputs(
                [str(record["image"]) for record, _, _ in pending],
                process_res=process_res,
                process_res_method="upper_bound_resize",
            )
            expected_image_shape = (len(pending), 3, process_res, process_res)
            if tuple(processed_images.shape) != expected_image_shape:
                raise RuntimeError(
                    "Unexpected DA3 preprocessing shape "
                    f"{tuple(processed_images.shape)}; expected {expected_image_shape}"
                )
            images = processed_images.unsqueeze(1).to(args.device)
            use_amp = torch.device(args.device).type == "cuda"
            amp_dtype = (
                torch.bfloat16
                if use_amp and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.inference_mode(), torch.autocast(
                device_type=torch.device(args.device).type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                _, auxiliary = api.model.backbone(
                    images,
                    cam_token=None,
                    export_feat_layers=list(layers),
                    ref_view_strategy="saddle_balanced",
                )
            if len(auxiliary) != 2:
                raise RuntimeError(f"Expected two DA3 auxiliary features, got {len(auxiliary)}")
            mid = _reshape_feature(auxiliary[0], grid_shape=grid_shape)
            deep = _reshape_feature(auxiliary[1], grid_shape=grid_shape)
            if mid.shape[1:] != expected_shape or deep.shape[1:] != expected_shape:
                raise RuntimeError(
                    f"Unexpected DA3 feature shapes: mid={mid.shape}, deep={deep.shape}"
                )
            for index, (record, image_hash, output_path) in enumerate(pending):
                _atomic_save_npz(
                    output_path,
                    {
                        "schema_version": np.int64(DA3_FEATURE_CACHE_SCHEMA_VERSION),
                        "sample_id": np.str_(record["id"]),
                        "image_sha256": np.str_(image_hash),
                        "model_name": np.str_(model_name),
                        "model_revision": np.str_(model_revision),
                        "process_res": np.int64(process_res),
                        "layers": np.asarray(layers, dtype=np.int64),
                        "feature_mid": mid[index],
                        "feature_deep": deep[index],
                    },
                )
                generated += 1
        completed = min(offset + len(selected), len(records))
        if offset == 0 or completed % args.log_every < args.batch_size or completed == len(records):
            elapsed = time.perf_counter() - start
            remaining = elapsed / completed * (len(records) - completed)
            print(
                f"[{completed}/{len(records)}] generated={generated} reused={reused}; "
                f"elapsed={elapsed / 60:.1f}min ETA={remaining / 60:.1f}min",
                flush=True,
            )

    manifest = {
        "schema_version": DA3_FEATURE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "record_count": len(records),
        "record_ids_sha256": record_ids_sha256(record["id"] for record in records),
        "model_name": model_name,
        "model_revision": model_revision,
        "process_res": process_res,
        "process_res_method": "upper_bound_resize",
        "layers": list(layers),
        "channels": channels,
        "grid_shape": list(grid_shape),
        "dtype": "float16",
        "generated": generated,
        "reused": reused,
        "elapsed_s": time.perf_counter() - start,
    }
    _atomic_write_json(cache_root / "manifest.json", manifest)
    print(f"Wrote complete cache manifest: {cache_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
