from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from bim_priorda3.data.splits import LEGACY_PREPARATION_FINGERPRINT_SHA256

INFERENCE_CALIBRATION_KEYS = {
    "residual_routing_depth",
    "residual_routing_temperature",
}
DATASET_PROVENANCE_SCHEMA_VERSION = 1


def _required_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} lacks required non-empty string {key!r}")
    return value


def _normalized_regions(value: Any, context: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{context} selected_regions must be a sequence of strings")
    regions = list(value)
    if any(not isinstance(region, str) or not region for region in regions):
        raise ValueError(f"{context} selected_regions contains an invalid region")
    if len(set(regions)) != len(regions):
        raise ValueError(f"{context} selected_regions contains duplicates")
    return sorted(regions)


def _manifest_preparation_identity(
    split_provenance: Mapping[str, Any],
    context: str,
) -> dict[str, str]:
    status = split_provenance.get(
        "manifest_preparation_fingerprint_status",
        "legacy_missing",
    )
    if status not in {"verified", "legacy_missing"}:
        raise ValueError(f"{context} has unknown manifest preparation status {status!r}")
    fingerprint = split_provenance.get("manifest_preparation_fingerprint_sha256")
    if fingerprint is None and status == "legacy_missing":
        fingerprint = LEGACY_PREPARATION_FINGERPRINT_SHA256
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"{context} lacks a manifest preparation fingerprint")
    if status == "legacy_missing" and fingerprint != LEGACY_PREPARATION_FINGERPRINT_SHA256:
        raise ValueError(f"{context} has an invalid legacy preparation fallback")
    return {"status": str(status), "fingerprint_sha256": fingerprint}


def _runtime_subset_identity(
    split_provenance: Mapping[str, Any],
    context: str,
) -> dict[str, Any] | None:
    value = split_provenance.get("runtime_subset")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} runtime_subset must be a mapping")
    if value.get("schema_version") != 1:
        raise ValueError(f"{context} runtime_subset has an unsupported schema version")
    if value.get("selection") != "ordered_prefix":
        raise ValueError(f"{context} runtime_subset has an unsupported selection policy")
    requested = value.get("requested_max_samples")
    count = value.get("sample_count")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError(f"{context} runtime_subset requested_max_samples must be positive")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{context} runtime_subset sample_count must be positive")
    if count > requested:
        raise ValueError(f"{context} runtime_subset sample_count exceeds requested_max_samples")
    status = value.get("preparation_fingerprint_status")
    if status not in {"verified", "legacy_missing"}:
        raise ValueError(f"{context} runtime_subset has an invalid preparation fingerprint status")
    hashes: dict[str, str] = {}
    for key in (
        "ordered_sample_ids_sha256",
        "ordered_sample_preparation_fingerprints_sha256",
        "fingerprint_sha256",
    ):
        digest = value.get(key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{context} runtime_subset has invalid {key}")
        hashes[key] = digest
    return {
        "schema_version": 1,
        "selection": "ordered_prefix",
        "requested_max_samples": requested,
        "sample_count": count,
        "preparation_fingerprint_status": status,
        **hashes,
    }


def dataset_split_identity(split_provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Return a relocation-stable identity for one resolved dataset protocol."""
    runtime_subset = _runtime_subset_identity(
        split_provenance,
        "dataset split provenance",
    )
    mode = split_provenance.get("mode")
    if mode == "annotations":
        identity = {
            "mode": "annotations",
            "fingerprint_sha256": _required_string(
                split_provenance,
                "fingerprint_sha256",
                "annotation split provenance",
            ),
            "annotation_raw_sha256": _required_string(
                split_provenance,
                "annotation_raw_sha256",
                "annotation split provenance",
            ),
            "canonical_assignment_sha256": _required_string(
                split_provenance,
                "canonical_assignment_sha256",
                "annotation split provenance",
            ),
            "manifest_ordered_ids_sha256": _required_string(
                split_provenance,
                "manifest_ordered_ids_sha256",
                "annotation split provenance",
            ),
            "selected_regions": _normalized_regions(
                split_provenance.get("selected_regions"),
                "annotation split provenance",
            ),
            "manifest_preparation": _manifest_preparation_identity(
                split_provenance,
                "annotation split provenance",
            ),
        }
        if runtime_subset is not None:
            identity["runtime_subset"] = runtime_subset
        return identity
    if mode == "regions":
        stride_value = split_provenance.get("record_stride_by_region", {})
        if not isinstance(stride_value, Mapping):
            raise TypeError("region split provenance record_stride_by_region must be a mapping")
        stride_by_region = {
            str(region): int(stride) for region, stride in sorted(stride_value.items())
        }
        if any(stride < 1 for stride in stride_by_region.values()):
            raise ValueError("region split provenance record strides must be positive")
        identity = {
            "mode": "regions",
            "selected_regions": _normalized_regions(
                split_provenance.get("selected_regions"),
                "region split provenance",
            ),
            "record_stride_by_region": stride_by_region,
            "manifest_preparation": _manifest_preparation_identity(
                split_provenance,
                "region split provenance",
            ),
        }
        if runtime_subset is not None:
            identity["runtime_subset"] = runtime_subset
        return identity
    raise ValueError(f"Unknown dataset split provenance mode: {mode!r}")


def make_training_dataset_provenance(
    train_split_provenance: Mapping[str, Any],
    val_split_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the checkpoint payload for the exact train/validation population."""
    split_provenance = {
        "train": dict(train_split_provenance),
        "val": dict(val_split_provenance),
    }
    return {
        "schema_version": DATASET_PROVENANCE_SCHEMA_VERSION,
        "split_provenance": split_provenance,
        "split_identities": {
            split: dataset_split_identity(provenance)
            for split, provenance in split_provenance.items()
        },
    }


def _checkpoint_dataset_provenance(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    checkpoint_provenance = checkpoint.get("provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        return None
    dataset_provenance = checkpoint_provenance.get("dataset")
    if dataset_provenance is None:
        return None
    if not isinstance(dataset_provenance, Mapping):
        raise TypeError("Checkpoint provenance.dataset must be a mapping")
    return dataset_provenance


def _training_split_identities(
    dataset_provenance: Mapping[str, Any],
    context: str,
) -> dict[str, dict[str, Any]]:
    split_value = dataset_provenance.get("split_provenance")
    if not isinstance(split_value, Mapping):
        raise TypeError(f"{context}.split_provenance must be a mapping")
    identities: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        provenance = split_value.get(split)
        if not isinstance(provenance, Mapping):
            raise TypeError(f"{context}.split_provenance[{split!r}] must be a mapping")
        identities[split] = dataset_split_identity(provenance)
    return identities


def validate_checkpoint_training_dataset_provenance(
    checkpoint: Mapping[str, Any],
    runtime_dataset_provenance: Mapping[str, Any],
    *,
    allow_cross_dataset: bool = False,
) -> dict[str, Any]:
    """Validate resume/stage initialization without comparing machine paths.

    ``allow_cross_dataset`` is intentionally narrow: it accepts only a dataset
    identity mismatch. Callers remain responsible for validating the model
    configuration and loading the model state with their normal strict policy.
    """
    runtime_identities = _training_split_identities(
        runtime_dataset_provenance,
        "runtime provenance.dataset",
    )
    target = {
        "kind": "runtime_training_dataset",
        "dataset_provenance": dict(runtime_dataset_provenance),
        "split_identities": runtime_identities,
    }
    checkpoint_dataset = _checkpoint_dataset_provenance(checkpoint)
    if checkpoint_dataset is None:
        return {
            "status": "legacy_checkpoint_missing",
            "accepted": True,
            "verified": False,
            "dataset_match": None,
            "cross_dataset": None,
            "explicit_cross_dataset_opt_in": allow_cross_dataset,
            "source": {
                "kind": "checkpoint_training_dataset",
                "dataset_provenance": None,
                "split_identities": None,
            },
            "target": target,
            "runtime_split_identities": runtime_identities,
        }

    checkpoint_identities = _training_split_identities(
        checkpoint_dataset,
        "checkpoint provenance.dataset",
    )
    identities_match = checkpoint_identities == runtime_identities
    source = {
        "kind": "checkpoint_training_dataset",
        "dataset_provenance": dict(checkpoint_dataset),
        "split_identities": checkpoint_identities,
    }
    if not identities_match and not allow_cross_dataset:
        raise ValueError(
            "Checkpoint dataset split provenance does not match the runtime "
            f"training population: checkpoint={checkpoint_identities}, "
            f"runtime={runtime_identities}"
        )
    return {
        "status": "verified" if identities_match else "accepted_cross_dataset",
        "accepted": True,
        "verified": identities_match,
        "dataset_match": identities_match,
        "cross_dataset": not identities_match,
        "explicit_cross_dataset_opt_in": allow_cross_dataset,
        "source": source,
        "target": target,
        "checkpoint_split_identities": checkpoint_identities,
        "runtime_split_identities": runtime_identities,
    }


def validate_checkpoint_evaluation_dataset_provenance(
    checkpoint: Mapping[str, Any],
    runtime_split_provenance: Mapping[str, Any],
    *,
    split: str,
    allow_cross_dataset: bool = False,
) -> dict[str, Any]:
    """Require the checkpoint protocol unless cross-dataset use is explicit.

    The opt-in affects only the source/target dataset identity comparison. A
    malformed checkpoint provenance payload still fails validation.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(
            "Evaluation dataset provenance requires split 'train', 'val', or 'test'"
        )
    runtime_identity = dataset_split_identity(runtime_split_provenance)
    target = {
        "kind": "runtime_evaluation_dataset",
        "split": split,
        "split_provenance": dict(runtime_split_provenance),
        "split_identity": runtime_identity,
    }
    checkpoint_dataset = _checkpoint_dataset_provenance(checkpoint)
    if checkpoint_dataset is None:
        return {
            "status": "legacy_checkpoint_missing",
            "accepted": True,
            "verified": False,
            "dataset_match": None,
            "cross_dataset": None,
            "explicit_cross_dataset_opt_in": allow_cross_dataset,
            "split": split,
            "source": {
                "kind": "checkpoint_training_dataset",
                "dataset_provenance": None,
                "training_split_identities": None,
                "expected_evaluation_split_identity": None,
            },
            "target": target,
            "runtime_split_identity": runtime_identity,
        }

    checkpoint_identities = _training_split_identities(
        checkpoint_dataset,
        "checkpoint provenance.dataset",
    )
    checkpoint_modes = {identity["mode"] for identity in checkpoint_identities.values()}
    if len(checkpoint_modes) != 1:
        raise ValueError(
            "Checkpoint train/val dataset provenance uses inconsistent modes: "
            f"{sorted(checkpoint_modes)}"
        )
    checkpoint_mode = next(iter(checkpoint_modes))

    if checkpoint_mode == "annotations":
        expected_identity = checkpoint_identities["train"]
        if checkpoint_identities["val"] != expected_identity:
            raise ValueError(
                "Checkpoint train/val annotations do not resolve to one dataset fingerprint"
            )
    else:
        checkpoint_cfg = checkpoint.get("config")
        checkpoint_data = (
            checkpoint_cfg.get("data") if isinstance(checkpoint_cfg, Mapping) else None
        )
        if not isinstance(checkpoint_data, Mapping):
            raise TypeError("Checkpoint training config does not contain a data mapping")
        expected_regions = checkpoint_data.get(f"{split}_regions")
        expected_stride = checkpoint_data.get("record_stride_by_region", {})
        expected_identity = dataset_split_identity(
            {
                "mode": "regions",
                "selected_regions": expected_regions,
                "record_stride_by_region": expected_stride,
            }
        )

    identities_match = runtime_identity == expected_identity
    source = {
        "kind": "checkpoint_training_dataset",
        "dataset_provenance": dict(checkpoint_dataset),
        "training_split_identities": checkpoint_identities,
        "expected_evaluation_split_identity": expected_identity,
    }
    if not identities_match and not allow_cross_dataset:
        if runtime_identity["mode"] != checkpoint_mode:
            raise ValueError(
                "Evaluation dataset split mode differs from the checkpoint: "
                f"checkpoint={checkpoint_mode!r}, "
                f"runtime={runtime_identity['mode']!r}"
            )
        raise ValueError(
            "Evaluation dataset split provenance does not match the checkpoint: "
            f"checkpoint={expected_identity}, runtime={runtime_identity}"
        )
    return {
        "status": "verified" if identities_match else "accepted_cross_dataset",
        "accepted": True,
        "verified": identities_match,
        "dataset_match": identities_match,
        "cross_dataset": not identities_match,
        "explicit_cross_dataset_opt_in": allow_cross_dataset,
        "split": split,
        "source": source,
        "target": target,
        "checkpoint_split_identity": expected_identity,
        "runtime_split_identity": runtime_identity,
    }


def model_config_differences(
    checkpoint_model: Mapping[str, Any],
    runtime_model: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    """Return every behavioral model-config difference."""
    differences = {}
    for key in sorted(set(checkpoint_model) | set(runtime_model)):
        checkpoint_value = checkpoint_model.get(key)
        runtime_value = runtime_model.get(key)
        if checkpoint_value != runtime_value:
            differences[key] = {
                "checkpoint": checkpoint_value,
                "evaluation": runtime_value,
            }
    return differences


def validate_checkpoint_model_config(
    checkpoint: Mapping[str, Any],
    runtime_model: Mapping[str, Any],
    *,
    allow_inference_calibration: bool = False,
) -> dict[str, dict[str, object]]:
    """Reject a checkpoint whose architecture differs from the runtime config."""
    checkpoint_cfg = checkpoint.get("config")
    if not isinstance(checkpoint_cfg, Mapping):
        raise TypeError("Checkpoint does not contain a training config")
    checkpoint_model = checkpoint_cfg.get("model")
    if not isinstance(checkpoint_model, Mapping):
        raise TypeError("Checkpoint training config does not contain a model mapping")

    differences = model_config_differences(checkpoint_model, runtime_model)
    disallowed = set(differences) - INFERENCE_CALIBRATION_KEYS
    if disallowed:
        raise ValueError(
            "Runtime model config is incompatible with the checkpoint; "
            f"differing keys: {sorted(disallowed)}"
        )
    if differences and not allow_inference_calibration:
        raise ValueError(
            "Routing calibration differs from the checkpoint. Explicitly allow "
            f"and record the inference override: {differences}"
        )
    return differences
