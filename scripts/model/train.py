#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from bim_priorda3.baselines import (
    ROBUST_LOG_CAP_SCALE_ESTIMATOR,
    resolve_scale_estimator_config,
)
from bim_priorda3.checkpoints import (
    make_training_dataset_provenance,
    validate_checkpoint_training_dataset_provenance,
)
from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.engine import (
    build_loader,
    move_batch,
    resolved_config_sha256,
    save_checkpoint,
    seed_everything,
    semantic_config_sha256,
    training_source_sha256,
    validate,
    write_history,
)
from bim_priorda3.losses import BIMPriorLoss
from bim_priorda3.models import BIMPriorDA3
from bim_priorda3.models.refiner import ScaleAnchoredDepthRefiner
from bim_priorda3.scale_protocol import validate_universal_scale_protocol

TRAINING_ARTIFACT_NAMES = frozenset(
    {
        "history.json",
        "run_state.json",
        "best.pt",
        "accepted.pt",
        "last.pt",
        "oom_state.pt",
        "OOM_README.txt",
    }
)
INIT_POLICY_PRESERVE = "preserve"
INIT_POLICY_ZERO_RESIDUAL_HEADS = "zero_multiplicative_residual_heads"
INIT_POLICIES = frozenset({INIT_POLICY_PRESERVE, INIT_POLICY_ZERO_RESIDUAL_HEADS})
REFINER_HEAD_WARMUP_PARAMETER_PREFIXES = (
    "refiner.low_output.",
    "refiner.detail_output.",
    "refiner.frame_output.",
    "refiner.bim_adapters.",
)
ATTENTION_SCALE_WARMUP_PARAMETER_PREFIXES = ("attention_scale.",)
REFINER_HEAD_WARMUP_STAGE = "refiner_heads_and_bim_adapters"
ATTENTION_SCALE_WARMUP_STAGE = "attention_scale_head"
REFINER_FULL_STAGE = "full_original_non_da3"
SCRATCH_SCALE_STAGE = "scratch_scale_only"
SCRATCH_REFINER_STAGE = "scratch_refiner_only"
SCRATCH_ADDITIVE_STAGE = "scratch_additive_only"
SCRATCH_JOINT_STAGE = "scratch_low_lr_joint"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_fresh_output_directory(
    output_dir: Path,
    *,
    resume: Path | None,
) -> None:
    """Prevent a fresh run from inheriting checkpoints or state from an old run."""
    if resume is not None or not output_dir.exists():
        return
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Training output path is not a directory: {output_dir}")
    existing = sorted(name for name in TRAINING_ARTIFACT_NAMES if (output_dir / name).exists())
    if existing:
        raise FileExistsError(
            "Fresh training refuses to reuse an output directory containing prior "
            f"run artifacts: {existing}. Choose a new --output-dir or use --resume."
        )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bind_split_provenance_to_subset(
    split_provenance: Mapping[str, Any],
    records: list[dict[str, Any]],
    *,
    requested_max_samples: int,
) -> dict[str, Any]:
    """Bind smoke-run provenance to the exact ordered sample artifacts used."""
    if requested_max_samples < 1:
        raise ValueError("Smoke-test sample limits must be positive")
    if not records:
        raise ValueError("A smoke-test subset must contain at least one sample")
    entries: list[dict[str, str | None]] = []
    fingerprint_present: list[bool] = []
    for record in records:
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Smoke-test subset records require non-empty sample IDs")
        fingerprint = record.get("preparation_fingerprint_sha256")
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(
                f"{sample_id!r}: invalid preparation_fingerprint_sha256 in smoke subset"
            )
        fingerprint_present.append(fingerprint is not None)
        entries.append(
            {
                "id": sample_id,
                "preparation_fingerprint_sha256": fingerprint,
            }
        )
    if any(fingerprint_present) and not all(fingerprint_present):
        raise ValueError(
            "Smoke-test subset mixes records with and without preparation fingerprints"
        )
    sample_ids = [entry["id"] for entry in entries]
    sample_fingerprints = [entry["preparation_fingerprint_sha256"] for entry in entries]
    subset_identity = {
        "schema_version": 1,
        "selection": "ordered_prefix",
        "requested_max_samples": int(requested_max_samples),
        "sample_count": len(entries),
        "preparation_fingerprint_status": (
            "verified" if all(fingerprint_present) else "legacy_missing"
        ),
        "ordered_sample_ids_sha256": _canonical_sha256(sample_ids),
        "ordered_sample_preparation_fingerprints_sha256": _canonical_sha256(sample_fingerprints),
        "fingerprint_sha256": _canonical_sha256({"schema_version": 1, "ordered_samples": entries}),
    }
    return {**dict(split_provenance), "runtime_subset": subset_identity}


def load_initial_model_weights(
    model: BIMPriorDA3,
    checkpoint_model: dict[str, torch.Tensor],
    checkpoint_model_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Initialize a stage while preserving freshly loaded E2E DA3 weights."""
    attention_scale_enabled = bool(getattr(model, "attention_scale_enabled", False))
    if not model.e2e_da3_enabled and not attention_scale_enabled:
        model.load_state_dict(checkpoint_model, strict=True)
        return {
            "strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
        }

    incompatible = model.load_state_dict(checkpoint_model, strict=False)
    model_da3_keys = {name for name in model.state_dict() if name.startswith("da3.")}
    checkpoint_da3_keys = {name for name in checkpoint_model if name.startswith("da3.")}
    model_attention_keys = {
        name for name in model.state_dict() if name.startswith("attention_scale.")
    }
    checkpoint_attention_keys = {
        name for name in checkpoint_model if name.startswith("attention_scale.")
    }
    missing_keys = set(incompatible.missing_keys)
    expected_missing = set()
    if model.e2e_da3_enabled and not checkpoint_da3_keys:
        expected_missing.update(model_da3_keys)
    if attention_scale_enabled and not checkpoint_attention_keys:
        expected_missing.update(model_attention_keys)
    if missing_keys != expected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "E2E stage initialization is incompatible with the checkpoint: "
            f"missing={sorted(missing_keys)}, "
            f"expected_missing={sorted(expected_missing)}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    if checkpoint_da3_keys and checkpoint_model_config is not None:
        checkpoint_e2e = checkpoint_model_config.get("e2e_da3", {})
        if not isinstance(checkpoint_e2e, Mapping):
            raise TypeError("Checkpoint model.e2e_da3 must be a mapping")
        for key in ("model_name", "revision"):
            checkpoint_value = checkpoint_e2e.get(key)
            runtime_value = model.e2e_da3_config.get(key)
            if checkpoint_value != runtime_value:
                raise ValueError(
                    f"E2E DA3 {key} mismatch during initialization: "
                    f"checkpoint={checkpoint_value!r}, "
                    f"runtime={runtime_value!r}"
                )
    return {
        "strict": False,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "policy": ("only freshly loaded da3.* and/or attention_scale.* parameters may be absent"),
    }


def resolve_init_checkpoint_policy(cfg: Mapping[str, Any]) -> str:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, Mapping):
        raise TypeError("train config must be a mapping")
    policy = str(train_cfg.get("init_checkpoint_policy", INIT_POLICY_PRESERVE))
    if policy not in INIT_POLICIES:
        raise ValueError(
            "train.init_checkpoint_policy must be 'preserve' or "
            "'zero_multiplicative_residual_heads'"
        )
    return policy


def apply_init_checkpoint_policy(
    model: BIMPriorDA3,
    policy: str,
    *,
    init_checkpoint: Path | None,
    resume: Path | None,
) -> dict[str, object]:
    """Apply a one-time, explicit policy after fresh checkpoint loading."""

    if policy not in INIT_POLICIES:
        raise ValueError(f"Unknown initialization policy: {policy!r}")
    if init_checkpoint is not None and resume is not None:
        raise ValueError("Initialization policy cannot combine init and resume")
    if resume is not None:
        return {
            "schema_version": 1,
            "policy": policy,
            "applied": False,
            "context": "resume_checkpoint_restore",
            "reason": "one-time initialization policies are never reapplied on resume",
        }
    if init_checkpoint is None:
        if policy != INIT_POLICY_PRESERVE:
            raise ValueError(
                f"train.init_checkpoint_policy={policy} requires a fresh --init-checkpoint run"
            )
        return {
            "schema_version": 1,
            "policy": policy,
            "applied": False,
            "context": "fresh_without_init_checkpoint",
            "reason": "default model initialization was preserved",
        }
    if policy == INIT_POLICY_PRESERVE:
        return {
            "schema_version": 1,
            "policy": policy,
            "applied": False,
            "context": "fresh_init_checkpoint",
            "reason": "loaded checkpoint heads were preserved",
        }
    if not isinstance(model.refiner, ScaleAnchoredDepthRefiner):
        raise TypeError("zero_multiplicative_residual_heads requires ScaleAnchoredDepthRefiner")
    reset_receipt = model.refiner.zero_multiplicative_residual_heads()
    return {
        "schema_version": 1,
        "policy": policy,
        "applied": True,
        "context": "fresh_init_checkpoint",
        "checkpoint": str(init_checkpoint.resolve()),
        "reset": reset_receipt,
    }


def resolve_refiner_head_warmup_epochs(
    cfg: Mapping[str, Any],
    *,
    e2e_enabled: bool | None = None,
) -> int:
    """Resolve the opt-in target-refiner warmup without changing old configs."""

    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, Mapping):
        raise TypeError("train config must be a mapping")
    value = train_cfg.get("refiner_head_warmup_epochs", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("train.refiner_head_warmup_epochs must be a non-negative integer")
    if e2e_enabled is None:
        model_cfg = cfg.get("model", {})
        if not isinstance(model_cfg, Mapping):
            raise TypeError("model config must be a mapping")
        e2e_cfg = model_cfg.get("e2e_da3", {})
        if not isinstance(e2e_cfg, Mapping):
            raise TypeError("model.e2e_da3 config must be a mapping")
        e2e_enabled = bool(e2e_cfg.get("enabled", False))
    if e2e_enabled and value != 0:
        raise ValueError(
            "train.refiner_head_warmup_epochs must be 0 for E2E training; "
            "DA3 trainability is controlled only by model.e2e_da3.trainable_scope"
        )
    return value


def snapshot_parameter_trainability(
    model: torch.nn.Module,
) -> dict[str, bool]:
    """Capture the configuration-defined trainability before epoch staging."""

    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def resolve_scratch_stage_epochs(
    cfg: Mapping[str, Any],
    *,
    e2e_enabled: bool,
    attention_scale_enabled: bool,
    additive_residual_enabled: bool = False,
) -> dict[str, int] | None:
    """Resolve the optional scratch scale/refiner/joint curriculum."""

    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, Mapping):
        raise TypeError("train config must be a mapping")
    raw = train_cfg.get("scratch_stage_epochs")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("train.scratch_stage_epochs must be a mapping")
    if e2e_enabled:
        raise ValueError("train.scratch_stage_epochs requires frozen/cached DA3")
    if not attention_scale_enabled:
        raise ValueError("train.scratch_stage_epochs requires model.attention_scale.enabled=true")
    expected = {"scale_only", "refiner_only", "joint"}
    ordered_names = ("scale_only", "refiner_only", "joint")
    if additive_residual_enabled:
        expected.add("additive_only")
        ordered_names = ("scale_only", "refiner_only", "additive_only", "joint")
    if set(raw) != expected:
        raise ValueError(
            "train.scratch_stage_epochs has the wrong stages; expected "
            f"{sorted(expected)}"
        )
    stages: dict[str, int] = {}
    for name in ordered_names:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"train.scratch_stage_epochs.{name} must be a positive integer")
        stages[name] = value
    total_epochs = train_cfg.get("epochs")
    if isinstance(total_epochs, bool) or not isinstance(total_epochs, int):
        raise TypeError("train.epochs must be an integer for scratch staged training")
    if sum(stages.values()) != total_epochs:
        raise ValueError(
            "train.scratch_stage_epochs must sum to train.epochs: "
            f"stages={sum(stages.values())}, epochs={total_epochs}"
        )
    warmup = train_cfg.get("refiner_head_warmup_epochs", 0)
    if warmup != 0:
        raise ValueError(
            "train.refiner_head_warmup_epochs must be 0 when scratch_stage_epochs is configured"
        )
    return stages


def build_scratch_stage_schedule(
    model: torch.nn.Module,
    stage_epochs: Mapping[str, int],
    original_trainability: Mapping[str, bool],
) -> dict[str, object]:
    """Build an auditable three-stage schedule for task-network scratch training."""

    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(original_trainability):
        raise ValueError("Original parameter trainability does not match the current model")
    originally_trainable = tuple(
        name for name in named_parameters if bool(original_trainability[name])
    )
    non_da3 = tuple(name for name in originally_trainable if not name.startswith("da3."))
    scale = tuple(name for name in non_da3 if name.startswith("attention_scale."))
    additive = tuple(name for name in non_da3 if name.startswith("refiner.additive_"))
    refiner = tuple(
        name
        for name in non_da3
        if not name.startswith("attention_scale.") and name not in additive
    )
    if not scale or not refiner:
        raise RuntimeError(
            "Scratch staged training requires trainable scale and refiner parameters"
        )
    scale_end = int(stage_epochs["scale_only"])
    refiner_end = scale_end + int(stage_epochs["refiner_only"])
    if "additive_only" in stage_epochs:
        if not additive:
            raise RuntimeError(
                "Scratch additive stage requires refiner additive-head parameters"
            )
        additive_end = refiner_end + int(stage_epochs["additive_only"])
        joint_end = additive_end + int(stage_epochs["joint"])
        phases = (
            (SCRATCH_SCALE_STAGE, 0, scale_end, scale),
            (SCRATCH_REFINER_STAGE, scale_end, refiner_end, refiner),
            (SCRATCH_ADDITIVE_STAGE, refiner_end, additive_end, additive),
            (SCRATCH_JOINT_STAGE, additive_end, joint_end, non_da3),
        )
        schedule_kind = "scratch_scale_refiner_additive_joint"
    else:
        if additive:
            raise RuntimeError(
                "An enabled additive head requires train.scratch_stage_epochs.additive_only"
            )
        joint_end = refiner_end + int(stage_epochs["joint"])
        phases = (
            (SCRATCH_SCALE_STAGE, 0, scale_end, scale),
            (SCRATCH_REFINER_STAGE, scale_end, refiner_end, refiner),
            (SCRATCH_JOINT_STAGE, refiner_end, joint_end, non_da3),
        )
        schedule_kind = "scratch_scale_refiner_joint"
    return {
        "schema_version": 2,
        "kind": schedule_kind,
        "initialization": "fresh_task_network_without_init_checkpoint",
        "phases": [
            {
                "name": name,
                "start_epoch_inclusive": start,
                "end_epoch_exclusive": end,
                "trainable_non_da3_parameter_names": list(names),
            }
            for name, start, end, names in phases
        ],
        "total_epochs": joint_end,
        "da3_policy": "frozen_cached",
        "optimizer_parameter_policy": (
            "one optimizer contains all originally trainable tensors; requires_grad "
            "selects each stage and the global cosine schedule makes the final joint "
            "stage low learning rate"
        ),
        "optimizer_parameter_tensors": len(originally_trainable),
        "optimizer_parameter_names_sha256": _canonical_sha256(list(originally_trainable)),
    }


def apply_scratch_training_stage(
    model: torch.nn.Module,
    *,
    epoch: int,
    schedule: Mapping[str, Any],
    original_trainability: Mapping[str, bool],
) -> dict[str, object]:
    """Activate exactly one scratch-training phase without rebuilding the optimizer."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    phases = schedule.get("phases")
    if not isinstance(phases, list):
        raise TypeError("Scratch schedule phases must be a list")
    selected = next(
        (
            phase
            for phase in phases
            if int(phase["start_epoch_inclusive"]) <= epoch < int(phase["end_epoch_exclusive"])
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"Epoch {epoch} is outside the configured scratch stage schedule")
    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(original_trainability):
        raise ValueError("Original parameter trainability does not match the current model")
    selected_names = set(selected["trainable_non_da3_parameter_names"])
    for name, parameter in named_parameters.items():
        if name.startswith("da3."):
            desired = bool(original_trainability[name])
        else:
            desired = bool(original_trainability[name]) and name in selected_names
        parameter.requires_grad_(desired)
    actual = tuple(
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad and not name.startswith("da3.")
    )
    expected = tuple(name for name in named_parameters if name in selected_names)
    if actual != expected:
        raise RuntimeError("Scratch training stage produced an unexpected parameter set")
    return {
        "schema_version": 1,
        "epoch": epoch,
        "name": str(selected["name"]),
        "start_epoch_inclusive": int(selected["start_epoch_inclusive"]),
        "end_epoch_exclusive": int(selected["end_epoch_exclusive"]),
        "trainable_non_da3_parameter_tensors": len(actual),
        "trainable_non_da3_parameters": sum(named_parameters[name].numel() for name in actual),
        "trainable_non_da3_parameter_names": list(actual),
        "da3_trainability_unchanged": True,
    }


def apply_scratch_stage_module_modes(
    model: BIMPriorDA3,
    stage_name: str,
) -> None:
    """Keep each frozen phase deterministic after the parent enters train mode."""

    if model.attention_scale is None:
        raise RuntimeError("Scratch staged training requires an attention-scale module")
    if stage_name == SCRATCH_SCALE_STAGE:
        model.attention_scale.train()
        model.refiner.eval()
    elif stage_name == SCRATCH_REFINER_STAGE:
        model.attention_scale.eval()
        model.refiner.train()
        if model.refiner.additive_body is not None:
            model.refiner.additive_body.eval()
        if model.refiner.additive_output is not None:
            model.refiner.additive_output.eval()
    elif stage_name == SCRATCH_ADDITIVE_STAGE:
        model.attention_scale.eval()
        model.refiner.eval()
        if model.refiner.additive_body is None or model.refiner.additive_output is None:
            raise RuntimeError("Scratch additive stage requires an enabled additive head")
        model.refiner.additive_body.train()
        model.refiner.additive_output.train()
    elif stage_name == SCRATCH_JOINT_STAGE:
        model.attention_scale.train()
        model.refiner.train()
    else:
        raise ValueError(f"Unknown scratch training stage: {stage_name!r}")


def _warmup_parameter_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    if bool(getattr(model, "attention_scale_enabled", False)):
        return ATTENTION_SCALE_WARMUP_PARAMETER_PREFIXES
    return REFINER_HEAD_WARMUP_PARAMETER_PREFIXES


def _warmup_stage_name(model: torch.nn.Module) -> str:
    if bool(getattr(model, "attention_scale_enabled", False)):
        return ATTENTION_SCALE_WARMUP_STAGE
    return REFINER_HEAD_WARMUP_STAGE


def _is_refiner_head_warmup_parameter(
    name: str,
    prefixes: tuple[str, ...] = REFINER_HEAD_WARMUP_PARAMETER_PREFIXES,
) -> bool:
    return name.startswith(prefixes)


def build_refiner_head_warmup_schedule(
    model: torch.nn.Module,
    warmup_epochs: int,
    original_trainability: Mapping[str, bool],
) -> dict[str, object]:
    """Build an immutable receipt for the optimizer-preserving stage schedule."""

    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
        raise TypeError("warmup_epochs must be an integer")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(original_trainability):
        raise ValueError("Original parameter trainability does not match the current model")
    originally_trainable = tuple(
        name for name in named_parameters if bool(original_trainability[name])
    )
    warmup_prefixes = _warmup_parameter_prefixes(model)
    full_non_da3 = tuple(name for name in originally_trainable if not name.startswith("da3."))
    heads_only = tuple(
        name for name in full_non_da3 if _is_refiner_head_warmup_parameter(name, warmup_prefixes)
    )
    if warmup_epochs and not heads_only:
        raise RuntimeError(
            "Refiner head warmup requested, but the model exposes no trainable "
            "residual heads or BIM adapters"
        )
    return {
        "schema_version": 1,
        "configured_epochs": warmup_epochs,
        "default_when_omitted": 0,
        "parameter_prefixes": list(warmup_prefixes),
        "heads_only_stage": {
            "name": _warmup_stage_name(model),
            "start_epoch_inclusive": 0,
            "end_epoch_exclusive": warmup_epochs,
            "trainable_non_da3_parameter_names": list(heads_only),
        },
        "full_stage": {
            "name": REFINER_FULL_STAGE,
            "start_epoch_inclusive": warmup_epochs,
            "trainable_non_da3_parameter_names": list(full_non_da3),
        },
        "da3_policy": "unchanged from model.e2e_da3.trainable_scope",
        "optimizer_parameter_policy": (
            "optimizer is constructed once from every originally trainable "
            "parameter before requires_grad staging"
        ),
        "optimizer_parameter_tensors": len(originally_trainable),
        "optimizer_parameter_names_sha256": _canonical_sha256(list(originally_trainable)),
    }


def optimizer_parameter_names(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[str, ...]:
    """Return and validate the optimizer's immutable model-parameter set."""

    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names: list[str] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id not in names_by_id:
                raise ValueError("Optimizer contains a parameter outside the model")
            if parameter_id in seen:
                raise ValueError("Optimizer contains the same parameter more than once")
            seen.add(parameter_id)
            names.append(names_by_id[parameter_id])
    return tuple(names)


def apply_refiner_head_warmup_stage(
    model: torch.nn.Module,
    *,
    epoch: int,
    warmup_epochs: int,
    original_trainability: Mapping[str, bool],
) -> dict[str, object]:
    """Switch one epoch's gradients while preserving the optimizer parameter set."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(warmup_epochs, bool) or not isinstance(warmup_epochs, int):
        raise TypeError("warmup_epochs must be an integer")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    named_parameters = dict(model.named_parameters())
    if set(named_parameters) != set(original_trainability):
        raise ValueError("Original parameter trainability does not match the current model")
    warmup_active = epoch < warmup_epochs
    warmup_prefixes = _warmup_parameter_prefixes(model)
    for name, parameter in named_parameters.items():
        originally_trainable = bool(original_trainability[name])
        if name.startswith("da3."):
            desired = originally_trainable
        elif warmup_active:
            desired = originally_trainable and _is_refiner_head_warmup_parameter(
                name,
                warmup_prefixes,
            )
        else:
            desired = originally_trainable
        parameter.requires_grad_(desired)

    trainable_non_da3_names = tuple(
        name
        for name, parameter in named_parameters.items()
        if parameter.requires_grad and not name.startswith("da3.")
    )
    expected_non_da3_names = tuple(
        name
        for name in named_parameters
        if bool(original_trainability[name])
        and not name.startswith("da3.")
        and (not warmup_active or _is_refiner_head_warmup_parameter(name, warmup_prefixes))
    )
    if trainable_non_da3_names != expected_non_da3_names:
        raise RuntimeError("Refiner head warmup produced an unexpected non-DA3 parameter set")
    da3_changed = [
        name
        for name, parameter in named_parameters.items()
        if name.startswith("da3.")
        and bool(parameter.requires_grad) != bool(original_trainability[name])
    ]
    if da3_changed:
        raise RuntimeError(f"Refiner head warmup changed DA3 trainability: {da3_changed}")
    return {
        "schema_version": 1,
        "epoch": epoch,
        "name": (_warmup_stage_name(model) if warmup_active else REFINER_FULL_STAGE),
        "warmup_active": warmup_active,
        "configured_warmup_epochs": warmup_epochs,
        "trainable_non_da3_parameter_tensors": len(trainable_non_da3_names),
        "trainable_non_da3_parameters": sum(
            named_parameters[name].numel() for name in trainable_non_da3_names
        ),
        "trainable_non_da3_parameter_names": list(trainable_non_da3_names),
        "da3_trainability_unchanged": True,
    }


def build_optimizer(
    model: BIMPriorDA3,
    cfg,
    learning_rate: float,
) -> tuple[torch.optim.AdamW, list[dict[str, object]]]:
    grouped_parameters = model.trainable_parameter_groups()
    parameter_groups = []
    receipt = []
    for name, configured_lr in (
        ("non_da3", learning_rate),
        (
            "attention_scale",
            float(cfg.train.get("attention_scale_learning_rate", learning_rate)),
        ),
        (
            "additive_residual",
            float(cfg.train.get("additive_residual_learning_rate", learning_rate)),
        ),
        (
            "da3",
            float(cfg.train.get("da3_learning_rate", learning_rate)),
        ),
    ):
        parameters = grouped_parameters.get(name, [])
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": configured_lr,
                "name": name,
            }
        )
        receipt.append(
            {
                "name": name,
                "learning_rate": configured_lr,
                "parameter_tensors": len(parameters),
                "parameters": sum(parameter.numel() for parameter in parameters),
            }
        )
    if not parameter_groups:
        raise RuntimeError("Model exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=float(cfg.train.weight_decay),
    )
    return optimizer, receipt


def resolve_validation_batch_size(cfg: Mapping[str, Any]) -> int:
    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, Mapping):
        raise TypeError("train config must be a mapping")
    value = train_cfg.get("val_batch_size", 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("train.val_batch_size must be a positive integer")
    return value


def resolve_validation_inference_seed(
    cfg: Mapping[str, Any],
    *,
    experiment_seed: int,
) -> int:
    """Resolve an epoch-independent seed for validation model inference."""

    train_cfg = cfg.get("train", {})
    if not isinstance(train_cfg, Mapping):
        raise TypeError("train config must be a mapping")
    value = train_cfg.get("validation_inference_seed", experiment_seed)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("train.validation_inference_seed must be a non-negative integer")
    return value


def reset_validation_inference_rng(
    inference_seed: int,
    loader_generator: torch.Generator,
) -> int:
    """Reset global inference RNG and the validation loader RNG together."""

    if (
        isinstance(inference_seed, bool)
        or not isinstance(inference_seed, int)
        or inference_seed < 0
    ):
        raise ValueError("validation inference seed must be a non-negative integer")
    loader_seed = inference_seed + 17
    seed_everything(inference_seed)
    loader_generator.manual_seed(loader_seed)
    return loader_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-frame BIM-PriorDA3 refiner")
    parser.add_argument(
        "--config",
        required=True,
        help="Training config; V5 uses separate pretrain and stage-two configs",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load model weights only and start a fresh optimizer/schedule",
    )
    parser.add_argument(
        "--allow-cross-dataset-initialization",
        action="store_true",
        help=(
            "Allow --init-checkpoint to come from a different dataset. Only "
            "dataset provenance matching is relaxed; this never applies to "
            "--resume."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, help="Override configured epoch count")
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=(
            "Override the configured non-DA3/refiner learning rate for a fresh "
            "optimizer; DA3 keeps train.da3_learning_rate"
        ),
    )
    parser.add_argument("--output-dir", type=Path, help="Override experiment output directory")
    parser.add_argument("--max-train-samples", type=int, help="Smoke-test subset size")
    parser.add_argument("--max-val-samples", type=int, help="Smoke-test subset size")
    return parser.parse_args()


def validate_checkpoint_selection(
    resume: Path | None,
    init_checkpoint: Path | None,
    allow_cross_dataset_initialization: bool,
) -> None:
    """Keep cross-dataset transfer exclusive to fresh initialization."""
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if allow_cross_dataset_initialization and init_checkpoint is None:
        raise ValueError(
            "--allow-cross-dataset-initialization requires --init-checkpoint; "
            "resume dataset validation is always strict"
        )


def is_validation_accepted(
    validation: Mapping[str, float],
    acceptance: Mapping[str, object],
    *,
    e2e_enabled: bool,
    robust_scale_enabled: bool = False,
) -> bool:
    """Apply the configured cached and live BIM-direct acceptance gates."""
    if not bool(acceptance.get("enabled", False)):
        return True
    near_tolerance = float(acceptance.get("near_relative_tolerance", 1.02))
    cached_prefix = "robust_bim_direct" if robust_scale_enabled else "anchor"
    live_prefix = "live_robust_bim_direct" if robust_scale_enabled else "live_bim_direct"
    count_prefixes = ["base", "scaled", "anchor", "refined"]
    require_learned_scale = bool(
        acceptance.get("require_learned_scale_better_than_universal", False)
    )
    if require_learned_scale:
        count_prefixes.append("learned_scale")
    if robust_scale_enabled:
        count_prefixes.extend(["robust_global_scale", "robust_bim_direct"])
    if e2e_enabled:
        count_prefixes.extend(
            [
                "live_da3",
                ("live_robust_global_scale" if robust_scale_enabled else "live_scale"),
                live_prefix,
            ]
        )
    fixed_count_value = float(validation["fixed_gt_support_count"])
    if (
        not math.isfinite(fixed_count_value)
        or not fixed_count_value.is_integer()
        or fixed_count_value < 1
    ):
        raise ValueError(
            "Validation fixed_gt_support_count must be a positive integer; "
            f"got {validation['fixed_gt_support_count']!r}"
        )
    fixed_count = int(fixed_count_value)
    count_mismatches: dict[str, object] = {}
    for prefix in count_prefixes:
        key = f"{prefix}_count"
        value = validation[key]
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer() or int(numeric) != fixed_count:
            count_mismatches[key] = value
    if count_mismatches:
        raise ValueError(
            "Validation comparator counts must equal the fixed GT support "
            f"count {fixed_count}; mismatches={count_mismatches}"
        )
    near_count_value = float(validation["near_fixed_gt_support_count"])
    if (
        not math.isfinite(near_count_value)
        or not near_count_value.is_integer()
        or near_count_value < 1
    ):
        raise ValueError(
            "Validation near_fixed_gt_support_count must be a positive integer; "
            f"got {validation['near_fixed_gt_support_count']!r}"
        )
    near_count = int(near_count_value)
    near_prefixes = ["anchor", "refined"]
    if robust_scale_enabled:
        near_prefixes.append("robust_bim_direct")
    if e2e_enabled:
        near_prefixes.append(live_prefix)
    near_count_mismatches: dict[str, object] = {}
    for prefix in near_prefixes:
        key = f"{prefix}_near_count"
        value = validation[key]
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer() or int(numeric) != near_count:
            near_count_mismatches[key] = value
    if near_count_mismatches:
        raise ValueError(
            "Validation near comparator counts must equal the fixed near-GT "
            f"support count {near_count}; mismatches={near_count_mismatches}"
        )

    metric_keys = [
        "refined_abs_rel",
        "refined_mae",
        "refined_near_abs_rel",
        f"{cached_prefix}_abs_rel",
        f"{cached_prefix}_mae",
        f"{cached_prefix}_near_abs_rel",
    ]
    if require_learned_scale:
        metric_keys.extend(
            [
                "learned_scale_abs_rel",
                "learned_scale_mae",
                "scaled_abs_rel",
                "scaled_mae",
            ]
        )
    if e2e_enabled:
        metric_keys.extend(
            [
                f"{live_prefix}_abs_rel",
                f"{live_prefix}_mae",
                f"{live_prefix}_near_abs_rel",
            ]
        )
    invalid_metrics = {
        key: validation[key] for key in metric_keys if not math.isfinite(float(validation[key]))
    }
    if invalid_metrics:
        raise ValueError(
            "Validation acceptance metrics must be finite on the fixed GT "
            f"support; invalid={invalid_metrics}"
        )
    accepted = (
        validation["refined_abs_rel"] < validation[f"{cached_prefix}_abs_rel"]
        and validation["refined_mae"] < validation[f"{cached_prefix}_mae"]
        and validation["refined_near_abs_rel"]
        <= validation[f"{cached_prefix}_near_abs_rel"] * near_tolerance
    )
    if require_learned_scale:
        accepted = accepted and (
            validation["learned_scale_abs_rel"] < validation["scaled_abs_rel"]
            and validation["learned_scale_mae"] < validation["scaled_mae"]
        )
    if e2e_enabled:
        accepted = accepted and (
            validation["refined_abs_rel"] < validation[f"{live_prefix}_abs_rel"]
            and validation["refined_mae"] < validation[f"{live_prefix}_mae"]
            and validation["refined_near_abs_rel"]
            <= validation[f"{live_prefix}_near_abs_rel"] * near_tolerance
        )
    return accepted


def main() -> None:
    args = parse_args()
    validate_checkpoint_selection(
        args.resume,
        args.init_checkpoint,
        args.allow_cross_dataset_initialization,
    )
    for option, value in (
        ("--max-train-samples", args.max_train_samples),
        ("--max-val-samples", args.max_val_samples),
    ):
        if value is not None and value < 1:
            raise ValueError(f"{option} must be positive")
    if args.resume:
        forbidden_resume_overrides = {
            name: value
            for name, value in {
                "--epochs": args.epochs,
                "--learning-rate": args.learning_rate,
                "--output-dir": args.output_dir,
                "--max-train-samples": args.max_train_samples,
                "--max-val-samples": args.max_val_samples,
            }.items()
            if value is not None
        }
        if forbidden_resume_overrides:
            raise ValueError(
                f"Resume does not allow runtime overrides: {sorted(forbidden_resume_overrides)}"
            )
    cfg = load_config(args.config)
    universal_scale_protocol = validate_universal_scale_protocol(cfg)
    init_checkpoint_policy = resolve_init_checkpoint_policy(cfg)
    if (
        init_checkpoint_policy != INIT_POLICY_PRESERVE
        and args.init_checkpoint is None
        and args.resume is None
    ):
        raise ValueError(
            "A non-default train.init_checkpoint_policy requires --init-checkpoint for a fresh run"
        )
    e2e_enabled = bool(cfg.model.get("e2e_da3", {}).get("enabled", False))
    refiner_head_warmup_epochs = resolve_refiner_head_warmup_epochs(
        cfg,
        e2e_enabled=e2e_enabled,
    )
    scratch_stage_epochs = resolve_scratch_stage_epochs(
        cfg,
        e2e_enabled=e2e_enabled,
        attention_scale_enabled=bool(cfg.model.get("attention_scale", {}).get("enabled", False)),
        additive_residual_enabled=bool(
            cfg.model.get("additive_residual", {}).get("enabled", False)
        ),
    )
    if scratch_stage_epochs is not None and args.init_checkpoint is not None:
        raise ValueError(
            "Scratch staged training forbids --init-checkpoint; initialize the task network "
            "from the configured deterministic defaults"
        )
    robust_scale_enabled = (
        resolve_scale_estimator_config(cfg.model.get("scale_estimator"))["name"]
        == ROBUST_LOG_CAP_SCALE_ESTIMATOR
    )
    if e2e_enabled and not bool(cfg.data.get("recompute_cached_baselines", False)):
        raise ValueError(
            "E2E DA3 requires data.recompute_cached_baselines=true so the "
            "frozen BIM-direct acceptance baseline is recomputed consistently"
        )
    experiment_seed = int(cfg.experiment.seed)
    validation_inference_seed = resolve_validation_inference_seed(
        cfg,
        experiment_seed=experiment_seed,
    )
    validation_loader_generator_seed = validation_inference_seed + 17
    seed_everything(experiment_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else resolve_project_path(cfg, cfg.experiment.output_dir)
    )
    validate_fresh_output_directory(output_dir, resume=args.resume)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_set = BIMDepthDataset(cfg, "train")
    val_set = BIMDepthDataset(cfg, "val", augment=False)
    train_split_provenance = dict(train_set.split_provenance)
    val_split_provenance = dict(val_set.split_provenance)
    if args.max_train_samples is not None:
        train_indices = range(min(args.max_train_samples, len(train_set)))
        train_split_provenance = bind_split_provenance_to_subset(
            train_split_provenance,
            [train_set.records[index] for index in train_indices],
            requested_max_samples=args.max_train_samples,
        )
        train_set = Subset(train_set, train_indices)
    if args.max_val_samples is not None:
        val_indices = range(min(args.max_val_samples, len(val_set)))
        val_split_provenance = bind_split_provenance_to_subset(
            val_split_provenance,
            [val_set.records[index] for index in val_indices],
            requested_max_samples=args.max_val_samples,
        )
        val_set = Subset(val_set, val_indices)
    dataset_provenance = make_training_dataset_provenance(
        train_split_provenance,
        val_split_provenance,
    )
    configured_samples_per_epoch = cfg.train.get("samples_per_epoch")
    effective_samples_per_epoch = configured_samples_per_epoch
    if args.max_train_samples is not None and effective_samples_per_epoch is not None:
        effective_samples_per_epoch = min(
            int(effective_samples_per_epoch),
            len(train_set),
        )
    train_generator = torch.Generator()
    val_generator = torch.Generator().manual_seed(validation_loader_generator_seed)
    val_batch_size = resolve_validation_batch_size(cfg)
    train_loader = build_loader(
        train_set,
        int(cfg.train.batch_size),
        int(cfg.train.num_workers),
        shuffle=True,
        region_balanced=bool(cfg.train.get("region_balanced_sampling", False)),
        region_balance_exponent=float(cfg.train.get("region_balance_exponent", 1.0)),
        samples_per_epoch=effective_samples_per_epoch,
        generator=train_generator,
        persistent_workers=False,
    )
    val_loader = build_loader(
        val_set,
        val_batch_size,
        int(cfg.train.num_workers),
        shuffle=False,
        generator=val_generator,
        persistent_workers=False,
    )
    model = BIMPriorDA3(cfg).to(device)
    provenance = {
        "training_config": str(Path(cfg.config_path).resolve()),
        "resolved_config_sha256": resolved_config_sha256(cfg),
        "semantic_config_sha256": semantic_config_sha256(cfg),
        "training_source_sha256": training_source_sha256(Path(cfg.project_root)),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "dataset": dataset_provenance,
        "universal_scale_protocol": universal_scale_protocol,
        "configured_init_checkpoint_policy": init_checkpoint_policy,
    }
    if args.init_checkpoint:
        # Initialization only needs model weights.  Keep optimizer/scheduler state
        # from large E2E checkpoints on CPU to avoid a needless GPU-memory spike.
        state = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        initialized_cfg = state.get("config", {})
        initialized_model_cfg = initialized_cfg.get("model", {})
        initialized_dataset_validation = validate_checkpoint_training_dataset_provenance(
            state,
            dataset_provenance,
            allow_cross_dataset=(args.allow_cross_dataset_initialization),
        )
        initialization_receipt = load_initial_model_weights(
            model,
            state["model"],
            initialized_model_cfg,
        )
        init_policy_receipt = apply_init_checkpoint_policy(
            model,
            init_checkpoint_policy,
            init_checkpoint=args.init_checkpoint,
            resume=None,
        )
        initialized_model_differences = {
            key: {
                "initialized": initialized_model_cfg.get(key),
                "training": cfg.model.get(key),
            }
            for key in sorted(set(initialized_model_cfg) | set(cfg.model))
            if initialized_model_cfg.get(key) != cfg.model.get(key)
        }
        provenance.update(
            {
                "initialized_from": str(args.init_checkpoint.resolve()),
                "initialized_from_sha256": file_sha256(args.init_checkpoint),
                "initialized_from_epoch": state.get("epoch"),
                "initialized_from_training_config": initialized_cfg.get("config_path"),
                "initialized_cross_dataset_opt_in": (args.allow_cross_dataset_initialization),
                "initialized_model_config_differences": (initialized_model_differences),
                "initialized_dataset_split_validation": (initialized_dataset_validation),
                "initialized_model_state_validation": initialization_receipt,
                "initialization_policy": init_policy_receipt,
            }
        )
        print(f"Initialized model weights from {args.init_checkpoint}")
        del state
    else:
        provenance["initialization_policy"] = apply_init_checkpoint_policy(
            model,
            init_checkpoint_policy,
            init_checkpoint=None,
            resume=args.resume,
        )
    original_parameter_trainability = snapshot_parameter_trainability(model)
    refiner_head_warmup_schedule = build_refiner_head_warmup_schedule(
        model,
        refiner_head_warmup_epochs,
        original_parameter_trainability,
    )
    provenance["refiner_head_warmup_schedule"] = refiner_head_warmup_schedule
    parameter_stage_schedule = (
        build_scratch_stage_schedule(
            model,
            scratch_stage_epochs,
            original_parameter_trainability,
        )
        if scratch_stage_epochs is not None
        else refiner_head_warmup_schedule
    )
    provenance["parameter_stage_schedule"] = parameter_stage_schedule
    criterion = BIMPriorLoss(cfg)
    learning_rate = (
        float(args.learning_rate)
        if args.learning_rate is not None
        else float(cfg.train.learning_rate)
    )
    optimizer, optimizer_group_receipt = build_optimizer(
        model,
        cfg,
        learning_rate,
    )
    expected_optimizer_parameter_names = tuple(
        name for name, trainable in original_parameter_trainability.items() if trainable
    )
    actual_optimizer_parameter_names = optimizer_parameter_names(
        model,
        optimizer,
    )
    if set(actual_optimizer_parameter_names) != set(expected_optimizer_parameter_names):
        raise RuntimeError(
            "Optimizer must contain every and only originally trainable "
            "parameter before refiner head warmup staging"
        )
    provenance["optimizer_parameter_groups"] = optimizer_group_receipt
    total_epochs = args.epochs or int(cfg.train.epochs)
    warmup_epochs = min(
        int(cfg.train.get("lr_warmup_epochs", 0)),
        max(total_epochs - 1, 0),
    )
    provenance["effective_runtime"] = {
        "epochs": total_epochs,
        "learning_rate": learning_rate,
        "attention_scale_learning_rate": (
            float(cfg.train.get("attention_scale_learning_rate", learning_rate))
            if model.attention_scale_enabled
            else None
        ),
        "additive_residual_learning_rate": (
            float(cfg.train.get("additive_residual_learning_rate", learning_rate))
            if model.additive_residual_enabled
            else None
        ),
        "da3_learning_rate": (
            float(cfg.train.get("da3_learning_rate", learning_rate))
            if model.e2e_da3_enabled
            else None
        ),
        "amp_init_scale": float(cfg.train.get("amp_init_scale", 65536.0)),
        "lr_warmup_epochs": warmup_epochs,
        "refiner_head_warmup_epochs": refiner_head_warmup_epochs,
        "output_dir": str(output_dir),
        "batch_size": int(cfg.train.batch_size),
        "val_batch_size": val_batch_size,
        "train_samples": len(train_set),
        "samples_per_epoch": int(
            effective_samples_per_epoch
            if effective_samples_per_epoch is not None
            else len(train_set)
        ),
        "region_balanced_sampling": bool(cfg.train.get("region_balanced_sampling", False)),
        "region_balance_exponent": float(cfg.train.get("region_balance_exponent", 1.0)),
        "actual_samples_per_epoch": (
            len(train_loader) * int(cfg.train.batch_size)
            if train_loader.drop_last
            else len(train_loader.sampler)
        ),
        "batches_per_epoch": len(train_loader),
        "optimizer_steps_per_epoch": (len(train_loader) + int(cfg.train.gradient_accumulation) - 1)
        // int(cfg.train.gradient_accumulation),
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "epoch_seed_schedule": "experiment_seed + epoch * 1000003 (mod 2^32)",
        "validation_inference_seed": validation_inference_seed,
        "validation_loader_generator_seed": validation_loader_generator_seed,
        "validation_rng_policy": (
            "global inference RNG is reset before every validation pass; "
            "validation DataLoader generator is reset to inference_seed + 17"
        ),
    }
    provenance["cli_overrides"] = {
        key: value
        for key, value in {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "allow_cross_dataset_initialization": (
                True if args.allow_cross_dataset_initialization else None
            ),
        }.items()
        if value is not None
    }
    if warmup_epochs:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=warmup_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_epochs - warmup_epochs, 1),
                ),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
        )
    start_epoch, best_metric = 0, float("inf")
    resume_state = None
    if args.resume:
        resume_state = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        checkpoint_cfg = resume_state.get("config")
        if checkpoint_cfg != dict(cfg):
            raise ValueError(
                "Resume requires the exact checkpoint training config; use "
                "--init-checkpoint for a new stage or changed configuration"
            )
        checkpoint_provenance = resume_state.get("provenance", {})
        checkpoint_head_warmup_schedule = checkpoint_provenance.get("refiner_head_warmup_schedule")
        if checkpoint_head_warmup_schedule != refiner_head_warmup_schedule:
            raise ValueError(
                "Resume refiner head warmup schedule differs from the current model/configuration"
            )
        checkpoint_parameter_stage_schedule = checkpoint_provenance.get(
            "parameter_stage_schedule",
            checkpoint_head_warmup_schedule,
        )
        if checkpoint_parameter_stage_schedule != parameter_stage_schedule:
            raise ValueError(
                "Resume parameter stage schedule differs from the current model/configuration"
            )
        for hash_name in (
            "resolved_config_sha256",
            "semantic_config_sha256",
            "training_source_sha256",
        ):
            checkpoint_hash = checkpoint_provenance.get(hash_name)
            current_hash = provenance[hash_name]
            if checkpoint_hash is None:
                raise ValueError(
                    f"Resume checkpoint lacks required {hash_name}; start a "
                    "fresh run or use --init-checkpoint for a new stage"
                )
            if checkpoint_hash != current_hash:
                raise ValueError(
                    f"Resume {hash_name} mismatch: checkpoint="
                    f"{checkpoint_hash}, current={current_hash}"
                )
        resume_dataset_validation = validate_checkpoint_training_dataset_provenance(
            resume_state,
            dataset_provenance,
        )
        prior_runtime = resume_state.get("provenance", {}).get(
            "effective_runtime",
            {},
        )
        prior_epochs = prior_runtime.get("epochs")
        if prior_epochs is not None and int(prior_epochs) != total_epochs:
            raise ValueError(
                "Resume cannot change the scheduler epoch budget; use "
                "--init-checkpoint for a new training stage"
            )
        model.load_state_dict(resume_state["model"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch = int(resume_state["epoch"]) + 1
        best_metric = float(resume_state["best_metric"])
        resumed_provenance = dict(resume_state.get("provenance", {}))
        resume_events = list(resumed_provenance.get("resume_events", []))
        resume_events.append(str(args.resume.resolve()))
        resumed_provenance["resume_events"] = resume_events
        resumed_provenance["effective_runtime"] = provenance["effective_runtime"]
        resumed_provenance["cli_overrides"] = provenance["cli_overrides"]
        resumed_provenance["dataset"] = dataset_provenance
        resumed_provenance["refiner_head_warmup_schedule"] = refiner_head_warmup_schedule
        resumed_provenance["parameter_stage_schedule"] = parameter_stage_schedule
        init_policy_resume_events = list(
            resumed_provenance.get("initialization_policy_resume_events", [])
        )
        init_policy_resume_events.append(
            {
                "checkpoint": str(args.resume.resolve()),
                "configured_policy": init_checkpoint_policy,
                "applied": False,
                "reason": "one-time initialization policy was not reapplied",
            }
        )
        resumed_provenance["initialization_policy_resume_events"] = init_policy_resume_events
        dataset_validation_events = list(
            resumed_provenance.get("dataset_split_validation_events", [])
        )
        dataset_validation_events.append(
            {
                "operation": "resume",
                "checkpoint": str(args.resume.resolve()),
                **resume_dataset_validation,
            }
        )
        resumed_provenance["dataset_split_validation_events"] = dataset_validation_events
        provenance = resumed_provenance

    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
        init_scale=float(cfg.train.get("amp_init_scale", 65536.0)),
    )
    if resume_state is not None:
        if "scaler" not in resume_state:
            raise ValueError(
                "Resume checkpoint lacks GradScaler state; start a fresh run "
                "or use --init-checkpoint for a new stage"
            )
        scaler.load_state_dict(resume_state["scaler"])
    accumulation = int(cfg.train.gradient_accumulation)
    history_path = output_dir / "history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if args.resume and history_path.exists()
        else []
    )
    if args.resume and (not history or int(history[-1].get("epoch", -1)) != start_epoch - 1):
        raise ValueError("Resume history.json must exist and end at the checkpoint epoch")
    if history:
        best_history_index = min(
            range(len(history)),
            key=lambda index: history[index]["val_refined_abs_rel"],
        )
        stale_epochs = len(history) - 1 - best_history_index
        accepted_metrics = [
            row["val_refined_abs_rel"] for row in history if row.get("accepted", False)
        ]
        best_accepted_metric = min(accepted_metrics) if accepted_metrics else float("inf")
    else:
        stale_epochs = 0
        best_accepted_metric = float("inf")
    successful_optimizer_steps = int(history[-1].get("optimizer_steps", 0)) if history else 0
    attempted_optimizer_steps = int(history[-1].get("optimizer_step_attempts", 0)) if history else 0
    active_refiner_training_stage = (
        apply_scratch_training_stage(
            model,
            epoch=start_epoch,
            schedule=parameter_stage_schedule,
            original_trainability=original_parameter_trainability,
        )
        if scratch_stage_epochs is not None
        else apply_refiner_head_warmup_stage(
            model,
            epoch=start_epoch,
            warmup_epochs=refiner_head_warmup_epochs,
            original_trainability=original_parameter_trainability,
        )
    )
    provenance["active_refiner_training_stage"] = active_refiner_training_stage
    provenance["active_parameter_stage"] = active_refiner_training_stage
    print(
        f"device={device}, train={len(train_set)}, val={len(val_set)}, "
        f"parameters={sum(p.numel() for p in model.parameters()):,}"
    )
    run_state_path = output_dir / "run_state.json"
    run_state = {
        "status": "running",
        "training_config": str(Path(cfg.config_path).resolve()),
        "resolved_config_sha256": provenance["resolved_config_sha256"],
        "semantic_config_sha256": provenance["semantic_config_sha256"],
        "training_source_sha256": provenance["training_source_sha256"],
        "effective_runtime": provenance["effective_runtime"],
        "dataset": provenance["dataset"],
        "initialization_policy": provenance.get("initialization_policy"),
        "initialization_policy_resume_events": provenance.get(
            "initialization_policy_resume_events",
            [],
        ),
        "refiner_head_warmup_schedule": refiner_head_warmup_schedule,
        "parameter_stage_schedule": parameter_stage_schedule,
        "active_refiner_training_stage": active_refiner_training_stage,
        "active_parameter_stage": active_refiner_training_stage,
    }
    if "initialized_dataset_split_validation" in provenance:
        run_state["checkpoint_initialization"] = {
            "checkpoint": provenance["initialized_from"],
            "checkpoint_sha256": provenance["initialized_from_sha256"],
            "cross_dataset_opt_in": provenance["initialized_cross_dataset_opt_in"],
            "dataset_validation": provenance["initialized_dataset_split_validation"],
            "model_state_validation": provenance["initialized_model_state_validation"],
            "initialization_policy": provenance["initialization_policy"],
        }
    run_state_path.write_text(
        json.dumps(run_state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    try:
        for epoch in range(start_epoch, total_epochs):
            previous_stage_name = str(active_refiner_training_stage["name"])
            active_refiner_training_stage = (
                apply_scratch_training_stage(
                    model,
                    epoch=epoch,
                    schedule=parameter_stage_schedule,
                    original_trainability=original_parameter_trainability,
                )
                if scratch_stage_epochs is not None
                else apply_refiner_head_warmup_stage(
                    model,
                    epoch=epoch,
                    warmup_epochs=refiner_head_warmup_epochs,
                    original_trainability=original_parameter_trainability,
                )
            )
            if str(active_refiner_training_stage["name"]) != previous_stage_name:
                # A new phase optimizes a disjoint parameter family.  Do not
                # let a plateau in the prior phase prematurely stop it.
                stale_epochs = 0
            provenance["active_refiner_training_stage"] = active_refiner_training_stage
            provenance["active_parameter_stage"] = active_refiner_training_stage
            run_state["active_refiner_training_stage"] = active_refiner_training_stage
            run_state["active_parameter_stage"] = active_refiner_training_stage
            run_state["last_started_epoch"] = epoch
            run_state_path.write_text(
                json.dumps(run_state, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            epoch_seed = (experiment_seed + epoch * 1_000_003) % 2**32
            seed_everything(epoch_seed)
            train_generator.manual_seed(epoch_seed)
            model.train()
            if scratch_stage_epochs is not None:
                apply_scratch_stage_module_modes(
                    model,
                    str(active_refiner_training_stage["name"]),
                )
            criterion.set_epoch(epoch)
            epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
            epoch_learning_rates = {
                str(group.get("name", index)): float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            epoch_successful_steps = 0
            epoch_attempted_steps = 0
            for step, batch in enumerate(train_loader, 1):
                batch = move_batch(batch, device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output = model(batch)
                    losses = criterion(output, batch)
                    scaled_loss = losses["total"] / accumulation
                scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.train.gradient_clip_norm)
                    )
                    scale_before_step = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_attempted_steps += 1
                    if not use_amp or scaler.get_scale() >= scale_before_step:
                        epoch_successful_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                for key, value in losses.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach())
                if step % int(cfg.train.log_every) == 0:
                    print(
                        f"epoch={epoch + 1}/{total_epochs} step={step}/{len(train_loader)} "
                        f"loss={sums['total'] / step:.5f}",
                        flush=True,
                    )
            successful_optimizer_steps += epoch_successful_steps
            attempted_optimizer_steps += epoch_attempted_steps
            if epoch_successful_steps:
                scheduler.step()
            else:
                print(
                    f"epoch={epoch + 1}: all optimizer steps were skipped; "
                    "the learning-rate scheduler was not advanced",
                    flush=True,
                )
            applied_validation_loader_seed = reset_validation_inference_rng(
                validation_inference_seed,
                val_generator,
            )
            validation = validate(model, val_loader, criterion, device, use_amp)
            row = {
                "epoch": epoch,
                "validation_inference_seed": validation_inference_seed,
                "validation_loader_generator_seed": (applied_validation_loader_seed),
                "refiner_training_stage": active_refiner_training_stage,
                "parameter_training_stage": active_refiner_training_stage,
                "learning_rate": epoch_learning_rate,
                "learning_rates": epoch_learning_rates,
                "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "next_learning_rates": {
                    str(group.get("name", index)): float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                },
                "optimizer_steps": successful_optimizer_steps,
                "optimizer_step_attempts": attempted_optimizer_steps,
                "epoch_optimizer_steps": epoch_successful_steps,
                "epoch_optimizer_step_attempts": epoch_attempted_steps,
                "peak_cuda_allocated_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else 0.0
                ),
                **{f"train_{key}": value / len(train_loader) for key, value in sums.items()},
                **{f"val_{key}": value for key, value in validation.items()},
            }
            acceptance = cfg.train.get("acceptance", {})
            accepted = is_validation_accepted(
                validation,
                acceptance,
                e2e_enabled=e2e_enabled,
                robust_scale_enabled=robust_scale_enabled,
            )
            row["accepted"] = accepted
            history.append(row)
            write_history(history_path, history)
            improved = validation["refined_abs_rel"] < best_metric
            if improved:
                best_metric = validation["refined_abs_rel"]
                stale_epochs = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_metric,
                    cfg,
                    provenance,
                )
            else:
                stale_epochs += 1
            if accepted and validation["refined_abs_rel"] < best_accepted_metric:
                best_accepted_metric = validation["refined_abs_rel"]
                save_checkpoint(
                    output_dir / "accepted.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_accepted_metric,
                    cfg,
                    provenance,
                )
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_metric,
                cfg,
                provenance,
            )
            run_state["completed_epochs"] = len(history)
            run_state["last_completed_epoch"] = epoch
            run_state_path.write_text(
                json.dumps(run_state, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if stale_epochs >= int(cfg.train.early_stopping_patience):
                print("Early stopping: validation AbsRel stopped improving")
                break
    except torch.cuda.OutOfMemoryError:
        run_state["status"] = "oom"
        run_state_path.write_text(
            json.dumps(run_state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        emergency = output_dir / "oom_state.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "config": dict(cfg),
                "provenance": provenance,
            },
            emergency,
        )
        message = (
            f"CUDA out of memory. State saved to {emergency}. "
            "Reduce target resolution/channels or move this project to the cloud server."
        )
        (output_dir / "OOM_README.txt").write_text(message + "\n", encoding="utf-8")
        print(message)
        raise SystemExit(2)
    except BaseException as error:
        run_state["status"] = "interrupted"
        run_state["error_type"] = type(error).__name__
        run_state_path.write_text(
            json.dumps(run_state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise
    else:
        run_state["status"] = "complete"
        run_state["completed_epochs"] = len(history)
        run_state_path.write_text(
            json.dumps(run_state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
