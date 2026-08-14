from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bim_priorda3.baselines import resolve_scale_estimator_config
from bim_priorda3.config import Config, resolve_project_path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_universal_scale_protocol(cfg: Config) -> dict[str, Any]:
    """Verify that training/evaluation use the one frozen public scale rule."""

    raw_path = cfg.evaluation.get("universal_scale_protocol")
    expected_sha256 = cfg.evaluation.get("universal_scale_protocol_sha256")
    if not raw_path or not expected_sha256:
        raise ValueError("Universal scale protocol path and SHA256 are required")
    path = resolve_project_path(cfg, raw_path)
    actual_sha256 = file_sha256(path)
    if actual_sha256 != str(expected_sha256):
        raise ValueError(
            "Universal scale protocol SHA256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("status") != "frozen":
        raise ValueError("Universal scale protocol must be frozen schema v1")
    if payload.get("per_dataset_overrides_allowed") is not False:
        raise ValueError("Universal scale protocol permits per-dataset overrides")
    receipt_estimator = payload.get("estimator")
    if not isinstance(receipt_estimator, dict):
        raise TypeError("Universal scale protocol estimator must be a mapping")
    estimator_keys = (
        "name",
        "q10_log_cap",
        "q25_log_cap",
        "ratio_min",
        "ratio_max",
        "min_samples",
    )
    try:
        receipt_parameters = {key: receipt_estimator[key] for key in estimator_keys}
    except KeyError as error:
        raise ValueError(f"Universal scale protocol estimator lacks {error.args[0]!r}") from error
    configured = resolve_scale_estimator_config(cfg.model.get("scale_estimator"))
    receipt = resolve_scale_estimator_config(receipt_parameters)
    if configured != receipt:
        raise ValueError("model.scale_estimator differs from the universal protocol")
    contract = payload.get("model_contract")
    if not isinstance(contract, dict) or contract.get("refinement_anchor") != (
        "universally scaled DA3 depth"
    ):
        raise ValueError("Universal scale protocol has an unknown model anchor contract")
    if contract.get("bim_direct_is_model_anchor") is not False:
        raise ValueError("Universal scale protocol incorrectly uses BIM-direct as model anchor")
    return {
        "status": "verified",
        "path": str(path),
        "sha256": actual_sha256,
        "scale_estimator": configured,
        "refinement_anchor": contract["refinement_anchor"],
        "per_dataset_overrides_allowed": False,
    }
