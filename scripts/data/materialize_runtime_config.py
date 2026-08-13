#!/usr/bin/env python3
"""Pin a generated manifest/split/scale receipt into a portable child config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data.splits import resolve_annotation_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local child config whose split and optional robust-scale "
            "receipt are bound to freshly prepared data"
        )
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--annotation",
        type=Path,
        help="Defaults to data.split_annotation from the base config",
    )
    parser.add_argument("--scale-receipt", type=Path)
    parser.add_argument(
        "--alignment-receipt",
        type=Path,
        help="Optional BIM-to-Area alignment receipt to pin in data.*",
    )
    parser.add_argument(
        "--experiment-output-dir",
        type=Path,
        help="Optional output directory for training with the generated config",
    )
    parser.add_argument(
        "--preparation-only",
        action="store_true",
        help=(
            "Create an alignment-pinned child config before a manifest exists; "
            "requires --alignment-receipt and forbids split/scale/output overrides"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    if not records:
        raise RuntimeError(f"Manifest contains no records: {path}")
    return records


def _portable_path(path: Path, project_root: Path) -> str:
    path = path.expanduser().resolve()
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _atomic_write_yaml(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def materialize_runtime_config(
    base_config: Path,
    output: Path,
    *,
    annotation: Path | None = None,
    scale_receipt: Path | None = None,
    alignment_receipt: Path | None = None,
    experiment_output_dir: Path | None = None,
    preparation_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    base_config = base_config.expanduser().resolve()
    output = output.expanduser().resolve()
    cfg = load_config(base_config)
    project_root = Path(cfg.project_root).resolve()
    parent = Path(os.path.relpath(base_config, output.parent)).as_posix()
    if preparation_only:
        if alignment_receipt is None:
            raise ValueError("--preparation-only requires --alignment-receipt")
        if annotation is not None or scale_receipt is not None or experiment_output_dir is not None:
            raise ValueError(
                "--preparation-only forbids annotation, scale receipt, and experiment output"
            )
        alignment_receipt = alignment_receipt.expanduser().resolve()
        alignment_raw = alignment_receipt.read_bytes()
        payload = {
            "extends": parent,
            "data": {
                "bim_alignment": _portable_path(alignment_receipt, project_root),
                "bim_alignment_sha256": hashlib.sha256(alignment_raw).hexdigest(),
            },
        }
        _atomic_write_yaml(output, payload, overwrite=overwrite)
        return {
            "mode": "preparation_only",
            "output": str(output),
            "manifest": None,
            "annotation": None,
            "annotation_raw_sha256": None,
            "split_fingerprint_sha256": None,
            "split_counts": None,
            "scale_receipt": None,
            "alignment_receipt": str(alignment_receipt),
        }

    manifest = resolve_project_path(cfg, cfg.data.processed_root) / "manifest.jsonl"
    if annotation is None:
        annotation_value = cfg.data.get("split_annotation")
        if not annotation_value:
            raise ValueError("Base config has no data.split_annotation; pass --annotation")
        annotation = resolve_project_path(cfg, annotation_value)
    else:
        annotation = annotation.expanduser().resolve()

    resolution = resolve_annotation_splits(_read_jsonl(manifest), annotation)
    provenance = resolution.provenance
    payload: dict[str, Any] = {
        "extends": parent,
        "data": {
            "split_annotation": _portable_path(annotation, project_root),
            "split_annotation_sha256": provenance["annotation_raw_sha256"],
            "split_fingerprint_sha256": provenance["fingerprint_sha256"],
        },
    }
    if alignment_receipt is not None:
        alignment_receipt = alignment_receipt.expanduser().resolve()
        alignment_raw = alignment_receipt.read_bytes()
        payload["data"].update(
            {
                "bim_alignment": _portable_path(alignment_receipt, project_root),
                "bim_alignment_sha256": hashlib.sha256(alignment_raw).hexdigest(),
            }
        )
    if experiment_output_dir is not None:
        payload["experiment"] = {
            "name": output.stem,
            "output_dir": _portable_path(experiment_output_dir, project_root),
        }

    if scale_receipt is not None:
        scale_receipt = scale_receipt.expanduser().resolve()
        raw_receipt = scale_receipt.read_bytes()
        receipt = json.loads(raw_receipt.decode("utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError("Robust-scale receipt is not complete")
        selection = receipt.get("final_selection")
        if not isinstance(selection, dict):
            raise TypeError("Robust-scale receipt has no final_selection mapping")
        estimator = selection.get("canonical_scale_estimator")
        if not isinstance(estimator, dict) or estimator.get("name") != "log_upper_cap_v1":
            raise ValueError("Robust-scale receipt has no canonical log_upper_cap_v1 estimator")
        protocol_sha256 = receipt.get("protocol_sha256")
        if not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64:
            raise ValueError("Robust-scale receipt has an invalid protocol_sha256")
        receipt_sha256 = hashlib.sha256(raw_receipt).hexdigest()
        payload["evaluation"] = {
            "robust_scale_estimator": estimator,
            "robust_scale_selection_receipt": _portable_path(scale_receipt, project_root),
            "robust_scale_selection_receipt_sha256": receipt_sha256,
            "robust_scale_selection_protocol_sha256": protocol_sha256,
        }
        configured_model_estimator = cfg.model.get("scale_estimator")
        if isinstance(configured_model_estimator, dict):
            if configured_model_estimator.get("name") != "log_upper_cap_v1":
                raise ValueError(
                    "Cannot replace a non-robust model.scale_estimator from a robust receipt"
                )
            payload["model"] = {"scale_estimator": estimator}

    _atomic_write_yaml(output, payload, overwrite=overwrite)
    return {
        "output": str(output),
        "manifest": str(manifest),
        "annotation": str(annotation),
        "annotation_raw_sha256": provenance["annotation_raw_sha256"],
        "split_fingerprint_sha256": provenance["fingerprint_sha256"],
        "split_counts": provenance["split_counts"],
        "scale_receipt": str(scale_receipt) if scale_receipt is not None else None,
        "alignment_receipt": (str(alignment_receipt) if alignment_receipt is not None else None),
    }


def main() -> None:
    args = parse_args()
    receipt = materialize_runtime_config(
        args.base_config,
        args.output,
        annotation=args.annotation,
        scale_receipt=args.scale_receipt,
        alignment_receipt=args.alignment_receipt,
        experiment_output_dir=args.experiment_output_dir,
        preparation_only=args.preparation_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
