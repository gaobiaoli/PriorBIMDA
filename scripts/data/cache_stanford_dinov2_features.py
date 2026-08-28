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

import cv2
import numpy as np
import torch
from torch import nn

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.da3_features import (
    feature_cache_path,
    record_ids_sha256,
    sha256_file,
)
from bim_priorda3.data.dataset import _read_manifest, relocate_record
from bim_priorda3.data.dinov2_features import DINOV2_FEATURE_CACHE_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache frozen official DINOv2-B/14 patch tokens for Stanford Area_1"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def freeze_dinov2(model: nn.Module) -> nn.Module:
    model.eval()
    model.requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("DINOv2 backbone did not freeze completely")
    return model


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
    repository_revision: str,
    process_res: int,
    expected_shape: tuple[int, int, int],
) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as item:
            return (
                int(item["schema_version"]) == DINOV2_FEATURE_CACHE_SCHEMA_VERSION
                and str(item["sample_id"].item()) == sample_id
                and str(item["image_sha256"].item()) == image_sha256
                and str(item["model_name"].item()) == model_name
                and str(item["repository_revision"].item()) == repository_revision
                and int(item["process_res"]) == process_res
                and item["feature"].shape == expected_shape
                and item["feature"].dtype == np.float16
            )
    except (KeyError, OSError, ValueError):
        return False


def _load_rgb(path: Path, process_res: int) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (process_res, process_res), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div_(255.0)
    mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = tensor.new_tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    return (tensor - mean) / std


def _checkpoint_path(model_name: str) -> Path:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}_pretrain.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Official DINOv2 checkpoint was not found after torch.hub load: {checkpoint}"
        )
    return checkpoint


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.log_every < 1:
        raise ValueError("--batch-size and --log-every must be positive")
    cfg = load_config(args.config)
    fusion = cfg.model.get("dinov2_feature_fusion", {})
    if not bool(fusion.get("enabled", False)):
        raise ValueError("model.dinov2_feature_fusion.enabled must be true")

    model_name = str(cfg.data.dinov2_model)
    repository = str(cfg.data.dinov2_repository)
    repository_revision = str(cfg.data.dinov2_revision)
    expected_weights_sha256 = str(cfg.data.dinov2_weights_sha256)
    process_res = int(cfg.data.dinov2_process_res)
    channels = int(fusion.get("channels", 768))
    patch_size = 14
    if process_res % patch_size:
        raise ValueError("data.dinov2_process_res must be divisible by patch size 14")
    grid_shape = (process_res // patch_size, process_res // patch_size)
    expected_shape = (channels, *grid_shape)
    if model_name != "dinov2_vitb14" or channels != 768:
        raise ValueError("This experiment is fixed to official dinov2_vitb14 with 768 channels")
    if not repository_revision:
        raise ValueError("A pinned data.dinov2_revision is required")

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
    cache_root = resolve_project_path(cfg, cfg.data.dinov2_feature_cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    hub_identifier = f"{repository}:{repository_revision}"
    # The environment may contain an ABI-incompatible xFormers wheel.  The
    # official implementation has an exact PyTorch SDPA fallback, which is the
    # reproducible path used by this cache job.
    os.environ["XFORMERS_DISABLED"] = "1"
    backbone = torch.hub.load(
        hub_identifier,
        model_name,
        source="github",
        pretrained=True,
        trust_repo=True,
        verbose=True,
    )
    backbone = freeze_dinov2(backbone).to(args.device)
    checkpoint_path = _checkpoint_path(model_name)
    weights_sha256 = sha256_file(checkpoint_path)
    if weights_sha256 != expected_weights_sha256:
        raise ValueError(
            "DINOv2 checkpoint SHA256 mismatch: "
            f"expected={expected_weights_sha256}, actual={weights_sha256}"
        )

    use_amp = torch.device(args.device).type == "cuda"
    cuda_supports_bfloat16 = use_amp and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if cuda_supports_bfloat16 else torch.float16
    generated = 0
    reused = 0
    start = time.perf_counter()
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
                repository_revision=repository_revision,
                process_res=process_res,
                expected_shape=expected_shape,
            ):
                reused += 1
            else:
                pending.append((record, image_hash, output_path))
        if pending:
            images = torch.stack(
                [_load_rgb(Path(record["image"]), process_res) for record, _, _ in pending]
            ).to(args.device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=torch.device(args.device).type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ),
            ):
                feature_dict = backbone.forward_features(images)
                tokens = feature_dict["x_norm_patchtokens"]
            expected_tokens = grid_shape[0] * grid_shape[1]
            if tuple(tokens.shape) != (len(pending), expected_tokens, channels):
                raise RuntimeError(
                    f"Unexpected DINOv2 tokens {tuple(tokens.shape)}; expected "
                    f"{(len(pending), expected_tokens, channels)}"
                )
            features = (
                tokens.transpose(1, 2)
                .reshape(len(pending), channels, *grid_shape)
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .numpy()
            )
            for index, (record, image_hash, output_path) in enumerate(pending):
                _atomic_save_npz(
                    output_path,
                    {
                        "schema_version": np.int64(DINOV2_FEATURE_CACHE_SCHEMA_VERSION),
                        "sample_id": np.str_(record["id"]),
                        "image_sha256": np.str_(image_hash),
                        "model_name": np.str_(model_name),
                        "repository_revision": np.str_(repository_revision),
                        "process_res": np.int64(process_res),
                        "feature": features[index],
                    },
                )
                generated += 1
        completed = min(offset + len(selected), len(records))
        if offset == 0 or completed % args.log_every < args.batch_size or completed == len(records):
            elapsed = time.perf_counter() - start
            print(
                f"cached={completed}/{len(records)} generated={generated} reused={reused} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    manifest = {
        "schema_version": DINOV2_FEATURE_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "record_count": len(records),
        "record_ids_sha256": record_ids_sha256(str(record["id"]) for record in records),
        "model_name": model_name,
        "repository": repository,
        "repository_revision": repository_revision,
        "weights_path": str(checkpoint_path),
        "weights_sha256": weights_sha256,
        "process_res": process_res,
        "patch_size": patch_size,
        "channels": channels,
        "grid_shape": list(grid_shape),
        "feature_key": "last_layer_x_norm_patchtokens",
        "dtype": "float16",
        "preprocessing": {
            "resize": "OpenCV INTER_AREA to 504x504",
            "range": "RGB float32 [0,1]",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "augmentation": "none; horizontal flips are applied to cached grids by dataset",
        },
    }
    _atomic_write_json(cache_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
