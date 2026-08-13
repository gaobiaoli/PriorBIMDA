#!/usr/bin/env python3
"""Generate and execute a registered leave-one-region-out protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from bim_priorda3.config import load_config, resolve_project_path
from bim_priorda3.engine import (
    resolved_config_sha256,
    semantic_config_sha256,
    training_source_sha256,
)
from bim_priorda3.region_cv import (
    RegionFoldPlan,
    build_region_fold_plans,
    dataset_fingerprint_from_manifest,
    parse_region_cv_protocol,
)

PROTOCOL_SCHEMA_VERSION = 1
PLAN_SCHEMA_VERSION = 1
EXECUTION_STATE_SCHEMA_VERSION = 1
STAGE_ORDER = ("plan", "pretrain", "finetune", "evaluate")
TRAINING_ARTIFACT_NAMES = (
    "accepted.pt",
    "best.pt",
    "history.json",
    "last.pt",
    "oom_state.pt",
    "run_state.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
    )


def project_path(project: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def display_path(project: Path, path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(project.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} contains invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or run a registered leave-one-region-out experiment. "
            "The safe default only materializes the plan and configs."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/slabim_region_cv.yaml"),
        help="Region-CV protocol YAML (default: configs/slabim_region_cv.yaml)",
    )
    parser.add_argument(
        "--fold",
        nargs="+",
        help="Fold index, fold ID, or held-out test region; default is every fold",
    )
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        help="Registered seed(s) to execute; default is every protocol seed",
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=("all",) + STAGE_ORDER,
        default=("plan",),
        help=(
            "Stages to execute. Default is 'plan'; 'all' means pretrain, "
            "finetune, and test evaluation."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print decisions without writing files or starting subprocesses",
    )
    return parser.parse_args(argv)


def normalize_stages(values: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(values))
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("--stage all cannot be combined with another stage")
        return ("pretrain", "finetune", "evaluate")
    return tuple(stage for stage in STAGE_ORDER if stage in requested)


def select_folds(
    plans: Sequence[RegionFoldPlan],
    selectors: Sequence[str] | None,
) -> tuple[RegionFoldPlan, ...]:
    if not selectors:
        return tuple(plans)
    selected: list[RegionFoldPlan] = []
    for selector in selectors:
        matches = [
            plan
            for plan in plans
            if selector
            in {
                str(plan.fold_index),
                plan.fold_id,
                plan.test_regions[0],
            }
        ]
        if not matches:
            choices = ", ".join(f"{plan.fold_index}:{plan.test_regions[0]}" for plan in plans)
            raise ValueError(f"Unknown --fold {selector!r}; choices are {choices}")
        plan = matches[0]
        if plan not in selected:
            selected.append(plan)
    return tuple(selected)


def select_seeds(registered: Sequence[int], requested: Sequence[int] | None) -> tuple[int, ...]:
    if not requested:
        return tuple(registered)
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        raise ValueError(
            f"Unregistered --seed values {unknown}; registered seeds are {list(registered)}"
        )
    return tuple(dict.fromkeys(requested))


@dataclass(frozen=True)
class ProtocolContext:
    project: Path
    source_path: Path
    protocol_id: str
    protocol_sha256: str
    normalized: dict[str, Any]
    parsed: Any
    folds: tuple[RegionFoldPlan, ...]
    templates: dict[str, Path]
    template_configs: dict[str, dict[str, Any]]
    output_root: Path
    checkpoint_selection: dict[str, str]
    stride_by_region: dict[str, int]
    samples_per_epoch: int
    manifest_path: Path
    dataset_fingerprint: dict[str, Any] | None
    training_source_sha256: str


def load_protocol_context(
    project: Path,
    protocol_path: Path,
    *,
    allow_missing_manifest: bool,
) -> ProtocolContext:
    source_path = project_path(project, protocol_path)
    with source_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"{source_path} must contain a YAML mapping")
    if raw.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"{source_path}: schema_version must be {PROTOCOL_SCHEMA_VERSION}")

    protocol_id = raw.get("protocol_id")
    if not isinstance(protocol_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*", protocol_id
    ):
        raise ValueError("protocol_id must be a filesystem-safe non-empty identifier")
    parsed = parse_region_cv_protocol(raw)
    if "5F_Region1" in parsed.regions:
        raise ValueError("5F_Region1 is an external holdout and cannot enter region CV")

    raw_checkpoint_selection = raw.get("checkpoint_selection")
    if raw_checkpoint_selection is None:
        checkpoint_selection = {"pretrain": "accepted", "final": "accepted"}
    else:
        if not isinstance(raw_checkpoint_selection, Mapping):
            raise TypeError("checkpoint_selection must be a mapping")
        if set(raw_checkpoint_selection) != {"pretrain", "final"}:
            raise ValueError("checkpoint_selection keys must be exactly ['pretrain', 'final']")
        checkpoint_selection = {}
        for stage in ("pretrain", "final"):
            selected = raw_checkpoint_selection[stage]
            if selected not in {"accepted", "best"}:
                raise ValueError(f"checkpoint_selection.{stage} must be 'accepted' or 'best'")
            checkpoint_selection[stage] = str(selected)

    raw_templates = raw.get("templates")
    if not isinstance(raw_templates, Mapping):
        raise TypeError("templates must contain pretrain and final paths")
    if set(raw_templates) != {"pretrain", "final"}:
        raise ValueError("templates keys must be exactly ['pretrain', 'final']")
    templates = {
        stage: project_path(project, str(raw_templates[stage])) for stage in ("pretrain", "final")
    }
    for stage, path in templates.items():
        if not path.is_file():
            raise FileNotFoundError(f"{stage} template does not exist: {path}")

    loaded_templates = {stage: load_config(path) for stage, path in templates.items()}
    template_configs = {
        stage: {
            key: value
            for key, value in dict(config).items()
            if key not in {"config_path", "project_root"}
        }
        for stage, config in loaded_templates.items()
    }
    configured_regions = tuple(loaded_templates["pretrain"].data.regions)
    missing_template_regions = sorted(set(parsed.regions) - set(configured_regions))
    if missing_template_regions:
        raise ValueError(
            f"pretrain template does not cover protocol regions: {missing_template_regions}"
        )

    raw_data = raw.get("data")
    if not isinstance(raw_data, Mapping):
        raise TypeError("data must be a mapping")
    raw_stride_by_region = raw_data.get("record_stride_by_region")
    if not isinstance(raw_stride_by_region, Mapping):
        raise TypeError("data.record_stride_by_region must be a mapping")
    stride_by_region = {str(region): int(stride) for region, stride in raw_stride_by_region.items()}
    if stride_by_region != {"5F_Region2": 2}:
        raise ValueError(
            "registered region CV requires default stride 1 with only "
            "data.record_stride_by_region.5F_Region2=2"
        )

    raw_train = raw.get("train")
    if not isinstance(raw_train, Mapping):
        raise TypeError("train must be a mapping")
    samples_per_epoch = raw_train.get("samples_per_epoch")
    if isinstance(samples_per_epoch, bool) or samples_per_epoch != 480:
        raise ValueError("registered region CV requires train.samples_per_epoch=480")

    raw_output_root = raw.get("output_root")
    if not isinstance(raw_output_root, str) or not raw_output_root.strip():
        raise ValueError("output_root must be a non-empty path")
    output_root = project_path(project, raw_output_root)
    if output_root == project.resolve():
        raise ValueError("output_root cannot be the project root")

    pretrain_cfg = loaded_templates["pretrain"]
    final_cfg = loaded_templates["final"]
    pretrain_processed = resolve_project_path(pretrain_cfg, pretrain_cfg.data.processed_root)
    final_processed = resolve_project_path(final_cfg, final_cfg.data.processed_root)
    if pretrain_processed != final_processed:
        raise ValueError("pretrain and final templates must use the same processed_root")
    manifest_path = pretrain_processed / "manifest.jsonl"
    dataset_fingerprint = None
    if manifest_path.is_file():
        dataset_fingerprint = dataset_fingerprint_from_manifest(
            manifest_path,
            regions=parsed.regions,
            stride=1,
            stride_by_region=stride_by_region,
        ).to_dict()
    elif not allow_missing_manifest:
        raise FileNotFoundError(f"CV provenance requires the prepared manifest: {manifest_path}")

    template_provenance = {
        stage: {
            "path": display_path(project, templates[stage]),
            "semantic_config_sha256": semantic_config_sha256(loaded_templates[stage]),
        }
        for stage in ("pretrain", "final")
    }
    normalized = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "regions": list(parsed.regions),
        "validation_map": parsed.validation_map,
        "seeds": list(parsed.seeds),
        "templates": template_provenance,
        "output_root": display_path(project, output_root),
        "data": {
            "manifest": display_path(project, manifest_path),
            "record_stride": 1,
            "record_stride_by_region": stride_by_region,
        },
        "train": {"samples_per_epoch": int(samples_per_epoch)},
    }
    # Keep hashes of already registered legacy protocols stable. Explicit
    # checkpoint-selection policies are nevertheless part of the normalized
    # protocol and therefore its hash.
    if raw_checkpoint_selection is not None:
        normalized["checkpoint_selection"] = dict(checkpoint_selection)
    protocol_sha = canonical_sha256(normalized)
    return ProtocolContext(
        project=project,
        source_path=source_path,
        protocol_id=protocol_id,
        protocol_sha256=protocol_sha,
        normalized=normalized,
        parsed=parsed,
        folds=build_region_fold_plans(parsed),
        templates=templates,
        template_configs=template_configs,
        output_root=output_root,
        checkpoint_selection=checkpoint_selection,
        stride_by_region=stride_by_region,
        samples_per_epoch=int(samples_per_epoch),
        manifest_path=manifest_path,
        dataset_fingerprint=dataset_fingerprint,
        training_source_sha256=training_source_sha256(project),
    )


@dataclass(frozen=True)
class RunPaths:
    fold_id: str
    seed: int
    run_dir: Path
    pretrain_config: Path
    final_config: Path
    pretrain_output: Path
    final_output: Path
    evaluation_output: Path

    @property
    def pretrain_accepted(self) -> Path:
        return self.pretrain_output / "accepted.pt"

    @property
    def pretrain_best(self) -> Path:
        return self.pretrain_output / "best.pt"

    @property
    def final_accepted(self) -> Path:
        return self.final_output / "accepted.pt"

    @property
    def final_best(self) -> Path:
        return self.final_output / "best.pt"

    def pretrain_checkpoint(self, selection: str) -> Path:
        if selection not in {"accepted", "best"}:
            raise ValueError(f"Unsupported pretrain checkpoint selection: {selection}")
        return self.pretrain_output / f"{selection}.pt"

    def final_checkpoint(self, selection: str) -> Path:
        if selection not in {"accepted", "best"}:
            raise ValueError(f"Unsupported final checkpoint selection: {selection}")
        return self.final_output / f"{selection}.pt"

    @property
    def evaluation_summary(self) -> Path:
        return self.evaluation_output / "summary.json"

    @property
    def evaluation_receipt(self) -> Path:
        return self.evaluation_output / "runner_provenance.json"


def run_paths(context: ProtocolContext, fold: RegionFoldPlan, seed: int) -> RunPaths:
    run_dir = context.output_root / "folds" / fold.fold_id / f"seed_{seed}"
    config_dir = context.output_root / "configs"
    stem = f"{fold.fold_id}__seed_{seed}"
    return RunPaths(
        fold_id=fold.fold_id,
        seed=seed,
        run_dir=run_dir,
        pretrain_config=config_dir / f"{stem}__pretrain.yaml",
        final_config=config_dir / f"{stem}__final.yaml",
        pretrain_output=run_dir / "pretrain",
        final_output=run_dir,
        evaluation_output=run_dir / "evaluation_test",
    )


def generated_override(
    context: ProtocolContext,
    fold: RegionFoldPlan,
    seed: int,
    stage: str,
    output_dir: Path,
) -> dict[str, Any]:
    dataset_sha = (
        context.dataset_fingerprint["sha256"]
        if context.dataset_fingerprint is not None
        else "unavailable-in-dry-run"
    )
    return {
        "experiment": {
            "name": f"{context.protocol_id}_{fold.fold_id}_seed_{seed}_{stage}",
            "seed": seed,
            "output_dir": display_path(context.project, output_dir),
        },
        "data": {
            "regions": list(context.parsed.regions),
            "train_regions": list(fold.train_regions),
            "val_regions": list(fold.val_regions),
            "test_regions": list(fold.test_regions),
            "record_stride_by_region": dict(context.stride_by_region),
        },
        "train": {"samples_per_epoch": context.samples_per_epoch},
        "region_cv": {
            "protocol_id": context.protocol_id,
            "protocol_sha256": context.protocol_sha256,
            "dataset_fingerprint_sha256": dataset_sha,
            "fold_index": fold.fold_index,
            "fold_id": fold.fold_id,
            "seed": seed,
            "stage": stage,
        },
    }


def expected_config(
    context: ProtocolContext,
    fold: RegionFoldPlan,
    seed: int,
    stage: str,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    merged = deep_merge(
        context.template_configs[stage],
        generated_override(context, fold, seed, stage, output_dir),
    )
    merged["config_path"] = str(config_path.resolve())
    merged["project_root"] = str(context.project.resolve())
    return merged


def render_generated_config(
    context: ProtocolContext,
    fold: RegionFoldPlan,
    seed: int,
    stage: str,
    config_path: Path,
    output_dir: Path,
) -> str:
    relative_template = Path(
        os.path.relpath(context.templates[stage], config_path.parent)
    ).as_posix()
    value = {
        "extends": relative_template,
        **generated_override(context, fold, seed, stage, output_dir),
    }
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def materialize_generated_config(path: Path, content: str) -> None:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != content:
            raise RuntimeError(f"Refusing to overwrite a changed generated config: {path}")
        return
    atomic_write_text(path, content)


def build_protocol_artifact(context: ProtocolContext) -> dict[str, Any]:
    return {
        **context.normalized,
        "protocol_sha256": context.protocol_sha256,
        "source_config": display_path(context.project, context.source_path),
        "dataset_fingerprint": context.dataset_fingerprint,
        "training_source_sha256_at_registration": context.training_source_sha256,
    }


def _old_run_metadata(
    existing_plan: Mapping[str, Any] | None,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    if not existing_plan:
        return {}
    old: dict[tuple[str, int], Mapping[str, Any]] = {}
    for fold in existing_plan.get("folds", []):
        if not isinstance(fold, Mapping):
            continue
        fold_id = fold.get("fold_id")
        for run in fold.get("runs", []):
            if (
                isinstance(fold_id, str)
                and isinstance(run, Mapping)
                and isinstance(run.get("seed"), int)
            ):
                old[(fold_id, int(run["seed"]))] = run
    return old


def build_plan(
    context: ProtocolContext,
    existing_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old_runs = _old_run_metadata(existing_plan)
    folds = []
    for fold in context.folds:
        fold_value = fold.to_dict()
        fold_value["runs"] = []
        for seed in context.parsed.seeds:
            paths = run_paths(context, fold, seed)
            old_run = old_runs.get((fold.fold_id, seed), {})
            run = {
                "seed": seed,
                "config_paths": {
                    "pretrain": display_path(context.project, paths.pretrain_config),
                    "final": display_path(context.project, paths.final_config),
                },
                "output_dirs": {
                    "pretrain": display_path(context.project, paths.pretrain_output),
                    "final": display_path(context.project, paths.final_output),
                    "evaluation": display_path(context.project, paths.evaluation_output),
                },
                "checkpoints": {
                    "pretrain_accepted": display_path(context.project, paths.pretrain_accepted),
                    "pretrain_best": display_path(context.project, paths.pretrain_best),
                    "final_accepted": display_path(context.project, paths.final_accepted),
                    "final_best": display_path(context.project, paths.final_best),
                    "pretrain_selected": display_path(
                        context.project,
                        paths.pretrain_checkpoint(context.checkpoint_selection["pretrain"]),
                    ),
                    "final_selected": display_path(
                        context.project,
                        paths.final_checkpoint(context.checkpoint_selection["final"]),
                    ),
                },
                "checkpoint_sha256": dict(old_run.get("checkpoint_sha256", {})),
                "evaluation_summary": display_path(context.project, paths.evaluation_summary),
                "evaluation_summary_sha256": old_run.get("evaluation_summary_sha256"),
                "evaluation_receipt": display_path(context.project, paths.evaluation_receipt),
            }
            fold_value["runs"].append(run)
        folds.append(fold_value)
    created_at = existing_plan.get("created_at_utc") if existing_plan else None
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol_id": context.protocol_id,
        "protocol_sha256": context.protocol_sha256,
        "created_at_utc": created_at or utc_now(),
        "updated_at_utc": utc_now(),
        "protocol_path": display_path(context.project, context.output_root / "protocol.json"),
        "source_protocol_path": display_path(context.project, context.source_path),
        "output_root": display_path(context.project, context.output_root),
        "template_paths": {
            stage: display_path(context.project, path) for stage, path in context.templates.items()
        },
        "regions": list(context.parsed.regions),
        "validation_map": context.parsed.validation_map,
        "seeds": list(context.parsed.seeds),
        "checkpoint_selection": dict(context.checkpoint_selection),
        "data": {
            "manifest": display_path(context.project, context.manifest_path),
            "record_stride": 1,
            "record_stride_by_region": dict(context.stride_by_region),
            "dataset_fingerprint": context.dataset_fingerprint,
        },
        "train": {"samples_per_epoch": context.samples_per_epoch},
        "folds": folds,
    }


def find_plan_run(
    plan: dict[str, Any],
    fold_id: str,
    seed: int,
) -> dict[str, Any]:
    for fold in plan["folds"]:
        if fold["fold_id"] != fold_id:
            continue
        for run in fold["runs"]:
            if run["seed"] == seed:
                return run
    raise KeyError(f"Plan does not contain {fold_id}/seed_{seed}")


def validate_existing_registration(
    context: ProtocolContext,
    protocol_path: Path,
    plan_path: Path,
) -> dict[str, Any] | None:
    if protocol_path.exists():
        registered = load_json_object(protocol_path)
        if registered.get("protocol_sha256") != context.protocol_sha256:
            raise RuntimeError(
                f"{protocol_path} belongs to a different protocol; "
                "choose a new protocol_id/output_root"
            )
        registered_fingerprint = registered.get("dataset_fingerprint")
        if registered_fingerprint != context.dataset_fingerprint:
            raise RuntimeError(
                f"{protocol_path} dataset fingerprint differs from the current manifest"
            )
    if not plan_path.exists():
        return None
    existing_plan = load_json_object(plan_path)
    if existing_plan.get("protocol_sha256") != context.protocol_sha256:
        raise RuntimeError(f"{plan_path} belongs to a different protocol")
    return existing_plan


def materialize_registration(
    context: ProtocolContext,
    plan: dict[str, Any],
) -> None:
    protocol_path = context.output_root / "protocol.json"
    plan_path = context.output_root / "plan.json"
    atomic_write_json(protocol_path, build_protocol_artifact(context))
    for fold in context.folds:
        for seed in context.parsed.seeds:
            paths = run_paths(context, fold, seed)
            materialize_generated_config(
                paths.pretrain_config,
                render_generated_config(
                    context,
                    fold,
                    seed,
                    "pretrain",
                    paths.pretrain_config,
                    paths.pretrain_output,
                ),
            )
            materialize_generated_config(
                paths.final_config,
                render_generated_config(
                    context,
                    fold,
                    seed,
                    "final",
                    paths.final_config,
                    paths.final_output,
                ),
            )
    atomic_write_json(plan_path, plan)


def validate_run_state(
    path: Path,
    expected_cfg: Mapping[str, Any],
    current_source_sha: str,
) -> dict[str, Any]:
    state = load_json_object(path)
    expected_semantic = semantic_config_sha256(expected_cfg)
    expected_resolved = resolved_config_sha256(expected_cfg)
    if state.get("semantic_config_sha256") != expected_semantic:
        raise RuntimeError(f"{path}: semantic config provenance mismatch")
    if state.get("resolved_config_sha256") != expected_resolved:
        raise RuntimeError(f"{path}: resolved config provenance mismatch")
    if state.get("training_source_sha256") != current_source_sha:
        raise RuntimeError(f"{path}: training source provenance mismatch")
    return state


def load_and_validate_checkpoint(
    path: Path,
    expected_cfg: Mapping[str, Any],
    current_source_sha: str,
    *,
    require_exact_config: bool,
    expected_initialized_from_sha256: str | None,
) -> tuple[dict[str, Any], str]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"Unable to read checkpoint {path}: {error}") from error
    if not isinstance(state, dict):
        raise TypeError(f"{path}: checkpoint must contain a mapping")
    checkpoint_cfg = state.get("config")
    if not isinstance(checkpoint_cfg, Mapping):
        raise TypeError(f"{path}: checkpoint has no training config")
    if semantic_config_sha256(checkpoint_cfg) != semantic_config_sha256(expected_cfg):
        raise RuntimeError(f"{path}: checkpoint semantic config mismatch")
    if require_exact_config and dict(checkpoint_cfg) != dict(expected_cfg):
        raise RuntimeError(f"{path}: exact config mismatch prevents optimizer/scheduler resume")
    provenance = state.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"{path}: checkpoint has no provenance mapping")
    if provenance.get("semantic_config_sha256") != semantic_config_sha256(expected_cfg):
        raise RuntimeError(f"{path}: checkpoint provenance config hash mismatch")
    if provenance.get("training_source_sha256") != current_source_sha:
        raise RuntimeError(f"{path}: checkpoint training source hash mismatch")
    if (
        expected_initialized_from_sha256 is not None
        and provenance.get("initialized_from_sha256") != expected_initialized_from_sha256
    ):
        raise RuntimeError(f"{path}: stage-two initialization checkpoint hash mismatch")
    return state, file_sha256(path)


@dataclass(frozen=True)
class TrainingDecision:
    action: str
    resume_checkpoint: Path | None = None
    selected_checkpoint: Path | None = None
    selected_sha256: str | None = None


def decide_training(
    output_dir: Path,
    expected_cfg: Mapping[str, Any],
    current_source_sha: str,
    *,
    expected_initialized_from_sha256: str | None,
    checkpoint_selection: str = "accepted",
) -> TrainingDecision:
    if checkpoint_selection not in {"accepted", "best"}:
        raise ValueError(f"Unsupported checkpoint selection: {checkpoint_selection!r}")
    run_state_path = output_dir / "run_state.json"
    selected_path = output_dir / f"{checkpoint_selection}.pt"
    last_path = output_dir / "last.pt"
    present = [
        output_dir / name for name in TRAINING_ARTIFACT_NAMES if (output_dir / name).exists()
    ]
    if not run_state_path.exists():
        if present:
            raise RuntimeError(f"{output_dir}: training artifacts exist without run_state.json")
        return TrainingDecision("fresh")

    run_state = validate_run_state(
        run_state_path,
        expected_cfg,
        current_source_sha,
    )
    status = run_state.get("status")
    if status == "complete":
        if not selected_path.is_file():
            raise RuntimeError(
                f"{output_dir}: training completed but the registered "
                f"{checkpoint_selection}.pt checkpoint is missing"
            )
        _, selected_sha = load_and_validate_checkpoint(
            selected_path,
            expected_cfg,
            current_source_sha,
            require_exact_config=False,
            expected_initialized_from_sha256=expected_initialized_from_sha256,
        )
        return TrainingDecision(
            "skip",
            selected_checkpoint=selected_path,
            selected_sha256=selected_sha,
        )

    if not last_path.is_file():
        raise RuntimeError(
            f"{output_dir}: status={status!r} but no last.pt is available for resume"
        )
    checkpoint, _ = load_and_validate_checkpoint(
        last_path,
        expected_cfg,
        current_source_sha,
        require_exact_config=True,
        expected_initialized_from_sha256=expected_initialized_from_sha256,
    )
    history_path = output_dir / "history.json"
    if not history_path.is_file():
        raise RuntimeError(f"{output_dir}: resume requires history.json")
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if (
        not isinstance(history, list)
        or not history
        or history[-1].get("epoch") != checkpoint.get("epoch")
    ):
        raise RuntimeError(f"{output_dir}: history.json does not end at the last checkpoint epoch")
    if selected_path.exists():
        load_and_validate_checkpoint(
            selected_path,
            expected_cfg,
            current_source_sha,
            require_exact_config=False,
            expected_initialized_from_sha256=expected_initialized_from_sha256,
        )
    return TrainingDecision("resume", resume_checkpoint=last_path)


def training_command(
    context: ProtocolContext,
    config_path: Path,
    decision: TrainingDecision,
    *,
    device: str,
    init_checkpoint: Path | None,
) -> list[str]:
    command = [
        sys.executable,
        str(context.project / "scripts" / "model" / "train.py"),
        "--config",
        str(config_path),
        "--device",
        device,
    ]
    if decision.action == "resume":
        command.extend(("--resume", str(decision.resume_checkpoint)))
    elif init_checkpoint is not None:
        command.extend(("--init-checkpoint", str(init_checkpoint)))
    return command


def update_execution_state(
    context: ProtocolContext,
    key: str,
    status: str,
    command: Sequence[str] | None = None,
    *,
    error: str | None = None,
    started_at: str | None = None,
) -> None:
    path = context.output_root / "runner_state.json"
    state = (
        load_json_object(path)
        if path.exists()
        else {
            "schema_version": EXECUTION_STATE_SCHEMA_VERSION,
            "protocol_id": context.protocol_id,
            "protocol_sha256": context.protocol_sha256,
            "runs": {},
        }
    )
    if state.get("protocol_sha256") != context.protocol_sha256:
        raise RuntimeError(f"{path} belongs to a different protocol")
    value = {
        "status": status,
        "command": list(command) if command is not None else None,
        "started_at_utc": started_at,
        "updated_at_utc": utc_now(),
    }
    if error is not None:
        value["error"] = error
    state["runs"][key] = value
    atomic_write_json(path, state)


def execute_command(
    context: ProtocolContext,
    key: str,
    command: list[str],
) -> None:
    print(f"[run] {key}: {shlex.join(command)}", flush=True)
    started = utc_now()
    update_execution_state(
        context,
        key,
        "running",
        command,
        started_at=started,
    )
    try:
        subprocess.run(command, cwd=context.project, check=True)
    except BaseException as error:
        update_execution_state(
            context,
            key,
            "failed",
            command,
            error=f"{type(error).__name__}: {error}",
            started_at=started,
        )
        raise
    update_execution_state(
        context,
        key,
        "completed",
        command,
        started_at=started,
    )


def validate_evaluation_summary(
    summary_path: Path,
    *,
    expected_checkpoint: Path,
    expected_config: Path,
    expected_test_regions: Sequence[str],
    expected_cfg: Mapping[str, Any],
    current_source_sha: str,
    expected_initialized_from_sha256: str,
) -> dict[str, Any]:
    summary = load_json_object(summary_path)
    if Path(str(summary.get("checkpoint"))).resolve() != expected_checkpoint.resolve():
        raise RuntimeError(f"{summary_path}: evaluated checkpoint path mismatch")
    if Path(str(summary.get("evaluation_config"))).resolve() != expected_config.resolve():
        raise RuntimeError(f"{summary_path}: evaluation config path mismatch")
    if summary.get("split") != "test":
        raise RuntimeError(f"{summary_path}: split must be test")
    if list(summary.get("regions", [])) != list(expected_test_regions):
        raise RuntimeError(f"{summary_path}: held-out test region mismatch")
    provenance = summary.get("checkpoint_provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError(f"{summary_path}: checkpoint provenance is missing")
    if provenance.get("semantic_config_sha256") != semantic_config_sha256(expected_cfg):
        raise RuntimeError(f"{summary_path}: checkpoint config provenance mismatch")
    if provenance.get("training_source_sha256") != current_source_sha:
        raise RuntimeError(f"{summary_path}: training source provenance mismatch")
    if provenance.get("initialized_from_sha256") != expected_initialized_from_sha256:
        raise RuntimeError(f"{summary_path}: initialization provenance mismatch")
    return summary


def evaluation_is_complete(
    context: ProtocolContext,
    fold: RegionFoldPlan,
    paths: RunPaths,
    expected_final_cfg: Mapping[str, Any],
    *,
    final_checkpoint: Path,
    final_checkpoint_sha256: str,
    pretrain_checkpoint_sha256: str,
) -> bool:
    summary_path = paths.evaluation_summary
    per_frame_path = paths.evaluation_output / "per_frame.csv"
    receipt_path = paths.evaluation_receipt
    if not summary_path.exists() and not per_frame_path.exists() and not receipt_path.exists():
        return False
    if summary_path.exists() and per_frame_path.exists() and not receipt_path.exists():
        print(
            f"[rerun] {fold.fold_id}/seed_{paths.seed}/evaluate: "
            "legacy evaluation has no checkpoint-hash receipt",
            flush=True,
        )
        return False
    if not summary_path.is_file() or not per_frame_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"{paths.evaluation_output}: incomplete evaluation artifact set")

    validate_evaluation_summary(
        summary_path,
        expected_checkpoint=final_checkpoint,
        expected_config=paths.final_config,
        expected_test_regions=fold.test_regions,
        expected_cfg=expected_final_cfg,
        current_source_sha=context.training_source_sha256,
        expected_initialized_from_sha256=pretrain_checkpoint_sha256,
    )
    receipt = load_json_object(receipt_path)
    expected = {
        "schema_version": 1,
        "protocol_id": context.protocol_id,
        "protocol_sha256": context.protocol_sha256,
        "fold_id": fold.fold_id,
        "seed": paths.seed,
        "checkpoint": display_path(context.project, final_checkpoint),
        "checkpoint_sha256": final_checkpoint_sha256,
        "summary": display_path(context.project, summary_path),
        "summary_sha256": file_sha256(summary_path),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"{receipt_path}: {key} provenance mismatch")
    return True


def write_evaluation_receipt(
    context: ProtocolContext,
    fold: RegionFoldPlan,
    paths: RunPaths,
    *,
    final_checkpoint: Path,
    final_checkpoint_sha256: str,
) -> str:
    summary_sha = file_sha256(paths.evaluation_summary)
    receipt = {
        "schema_version": 1,
        "protocol_id": context.protocol_id,
        "protocol_sha256": context.protocol_sha256,
        "fold_id": fold.fold_id,
        "seed": paths.seed,
        "checkpoint": display_path(context.project, final_checkpoint),
        "checkpoint_sha256": final_checkpoint_sha256,
        "summary": display_path(context.project, paths.evaluation_summary),
        "summary_sha256": summary_sha,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(paths.evaluation_receipt, receipt)
    return summary_sha


def update_plan_artifact(
    plan: dict[str, Any],
    context: ProtocolContext,
    fold: RegionFoldPlan,
    seed: int,
    *,
    pretrain_sha: str | None = None,
    final_sha: str | None = None,
    evaluation_summary_sha: str | None = None,
) -> None:
    run = find_plan_run(plan, fold.fold_id, seed)
    if pretrain_sha is not None:
        selection = context.checkpoint_selection["pretrain"]
        run["checkpoint_sha256"]["pretrain_selected"] = pretrain_sha
        run["checkpoint_sha256"][f"pretrain_{selection}"] = pretrain_sha
    if final_sha is not None:
        selection = context.checkpoint_selection["final"]
        run["checkpoint_sha256"]["final_selected"] = final_sha
        run["checkpoint_sha256"][f"final_{selection}"] = final_sha
    if evaluation_summary_sha is not None:
        run["evaluation_summary_sha256"] = evaluation_summary_sha
    plan["updated_at_utc"] = utc_now()
    atomic_write_json(context.output_root / "plan.json", plan)


def print_plan_summary(
    context: ProtocolContext,
    selected_folds: Sequence[RegionFoldPlan],
    selected_seeds: Sequence[int],
    stages: Sequence[str],
    *,
    dry_run: bool,
) -> None:
    mode = "dry-run" if dry_run else "execute"
    fingerprint = (
        context.dataset_fingerprint["sha256"]
        if context.dataset_fingerprint is not None
        else "unavailable"
    )
    print(
        f"[plan] mode={mode} protocol={context.protocol_id} "
        f"protocol_sha256={context.protocol_sha256}",
        flush=True,
    )
    print(
        f"[plan] output={display_path(context.project, context.output_root)} "
        f"dataset_sha256={fingerprint} samples_per_epoch={context.samples_per_epoch}",
        flush=True,
    )
    print(
        f"[plan] folds={len(selected_folds)}/{len(context.folds)} "
        f"seeds={list(selected_seeds)} stages={list(stages)}",
        flush=True,
    )
    for fold in selected_folds:
        print(
            f"[fold] {fold.fold_index} {fold.fold_id}: "
            f"train={list(fold.train_regions)} val={list(fold.val_regions)} "
            f"test={list(fold.test_regions)}",
            flush=True,
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project = Path(__file__).resolve().parents[2]
    stages = normalize_stages(args.stage)
    context = load_protocol_context(
        project,
        args.protocol,
        allow_missing_manifest=args.dry_run,
    )
    selected_folds = select_folds(context.folds, args.fold)
    selected_seeds = select_seeds(context.parsed.seeds, args.seed)
    print_plan_summary(
        context,
        selected_folds,
        selected_seeds,
        stages,
        dry_run=args.dry_run,
    )
    if context.dataset_fingerprint is None:
        print(
            f"[warning] manifest is unavailable: {context.manifest_path}; "
            "commands are preview-only",
            flush=True,
        )

    protocol_path = context.output_root / "protocol.json"
    plan_path = context.output_root / "plan.json"
    existing_plan = validate_existing_registration(context, protocol_path, plan_path)
    plan = build_plan(context, existing_plan)

    if args.dry_run:
        print("[dry-run] no configs, plans, checkpoints, or results were written", flush=True)
    else:
        materialize_registration(context, plan)
        print(
            f"[ready] protocol={display_path(project, protocol_path)} "
            f"plan={display_path(project, plan_path)}",
            flush=True,
        )

    if stages == ("plan",):
        return

    for fold in selected_folds:
        for seed in selected_seeds:
            paths = run_paths(context, fold, seed)
            pretrain_selection = context.checkpoint_selection["pretrain"]
            final_selection = context.checkpoint_selection["final"]
            pretrain_checkpoint = paths.pretrain_checkpoint(pretrain_selection)
            final_checkpoint = paths.final_checkpoint(final_selection)
            pretrain_cfg = expected_config(
                context,
                fold,
                seed,
                "pretrain",
                paths.pretrain_config,
                paths.pretrain_output,
            )
            final_cfg = expected_config(
                context,
                fold,
                seed,
                "final",
                paths.final_config,
                paths.final_output,
            )
            pretrain_sha: str | None = None
            final_sha: str | None = None

            if "pretrain" in stages:
                decision = decide_training(
                    paths.pretrain_output,
                    pretrain_cfg,
                    context.training_source_sha256,
                    expected_initialized_from_sha256=None,
                    checkpoint_selection=pretrain_selection,
                )
                key = f"{fold.fold_id}/seed_{seed}/pretrain"
                command = training_command(
                    context,
                    paths.pretrain_config,
                    decision,
                    device=args.device,
                    init_checkpoint=None,
                )
                if decision.action == "skip":
                    pretrain_sha = decision.selected_sha256
                    print(
                        f"[skip] {key}: validated {pretrain_checkpoint.name}",
                        flush=True,
                    )
                    if not args.dry_run:
                        update_execution_state(context, key, "skipped", command)
                elif args.dry_run:
                    print(
                        f"[{decision.action}] {key}: {shlex.join(command)}",
                        flush=True,
                    )
                else:
                    execute_command(context, key, command)
                    completed = decide_training(
                        paths.pretrain_output,
                        pretrain_cfg,
                        context.training_source_sha256,
                        expected_initialized_from_sha256=None,
                        checkpoint_selection=pretrain_selection,
                    )
                    if completed.action != "skip":
                        raise RuntimeError(f"{key}: training did not complete")
                    pretrain_sha = completed.selected_sha256
                if not args.dry_run and pretrain_sha is not None:
                    update_plan_artifact(
                        plan,
                        context,
                        fold,
                        seed,
                        pretrain_sha=pretrain_sha,
                    )

            if pretrain_sha is None and pretrain_checkpoint.is_file():
                _, pretrain_sha = load_and_validate_checkpoint(
                    pretrain_checkpoint,
                    pretrain_cfg,
                    context.training_source_sha256,
                    require_exact_config=False,
                    expected_initialized_from_sha256=None,
                )

            if "finetune" in stages:
                if pretrain_sha is None and not args.dry_run:
                    raise FileNotFoundError(f"Finetuning requires {pretrain_checkpoint}")
                decision = decide_training(
                    paths.final_output,
                    final_cfg,
                    context.training_source_sha256,
                    expected_initialized_from_sha256=pretrain_sha,
                    checkpoint_selection=final_selection,
                )
                key = f"{fold.fold_id}/seed_{seed}/finetune"
                command = training_command(
                    context,
                    paths.final_config,
                    decision,
                    device=args.device,
                    init_checkpoint=pretrain_checkpoint,
                )
                if decision.action == "skip":
                    final_sha = decision.selected_sha256
                    print(
                        f"[skip] {key}: validated {final_checkpoint.name}",
                        flush=True,
                    )
                    if not args.dry_run:
                        update_execution_state(context, key, "skipped", command)
                elif args.dry_run:
                    prerequisite = (
                        "" if pretrain_sha is not None else " (after registered pretrain completes)"
                    )
                    print(
                        f"[{decision.action}] {key}{prerequisite}: {shlex.join(command)}",
                        flush=True,
                    )
                else:
                    execute_command(context, key, command)
                    completed = decide_training(
                        paths.final_output,
                        final_cfg,
                        context.training_source_sha256,
                        expected_initialized_from_sha256=pretrain_sha,
                        checkpoint_selection=final_selection,
                    )
                    if completed.action != "skip":
                        raise RuntimeError(f"{key}: training did not complete")
                    final_sha = completed.selected_sha256
                if not args.dry_run and final_sha is not None:
                    update_plan_artifact(
                        plan,
                        context,
                        fold,
                        seed,
                        pretrain_sha=pretrain_sha,
                        final_sha=final_sha,
                    )

            if final_sha is None and final_checkpoint.is_file():
                if pretrain_sha is None:
                    raise RuntimeError(
                        f"Cannot validate {final_checkpoint} without its "
                        "registered pretrain checkpoint"
                    )
                _, final_sha = load_and_validate_checkpoint(
                    final_checkpoint,
                    final_cfg,
                    context.training_source_sha256,
                    require_exact_config=False,
                    expected_initialized_from_sha256=pretrain_sha,
                )

            if "evaluate" not in stages:
                continue
            key = f"{fold.fold_id}/seed_{seed}/evaluate"
            command = [
                sys.executable,
                str(context.project / "scripts" / "model" / "evaluate.py"),
                "--config",
                str(paths.final_config),
                "--checkpoint",
                str(final_checkpoint),
                "--split",
                "test",
                "--output",
                str(paths.evaluation_output),
                "--device",
                args.device,
            ]
            if final_sha is None or pretrain_sha is None:
                if not args.dry_run:
                    raise FileNotFoundError(
                        f"Evaluation requires {final_checkpoint} and {pretrain_checkpoint}"
                    )
                print(
                    f"[fresh] {key} (after registered finetune completes): {shlex.join(command)}",
                    flush=True,
                )
                continue

            complete = evaluation_is_complete(
                context,
                fold,
                paths,
                final_cfg,
                final_checkpoint=final_checkpoint,
                final_checkpoint_sha256=final_sha,
                pretrain_checkpoint_sha256=pretrain_sha,
            )
            if complete:
                print(f"[skip] {key}: validated summary and checkpoint receipt", flush=True)
                if not args.dry_run:
                    update_execution_state(context, key, "skipped", command)
                    update_plan_artifact(
                        plan,
                        context,
                        fold,
                        seed,
                        pretrain_sha=pretrain_sha,
                        final_sha=final_sha,
                        evaluation_summary_sha=file_sha256(paths.evaluation_summary),
                    )
                continue
            if args.dry_run:
                print(f"[fresh] {key}: {shlex.join(command)}", flush=True)
                continue
            execute_command(context, key, command)
            validate_evaluation_summary(
                paths.evaluation_summary,
                expected_checkpoint=final_checkpoint,
                expected_config=paths.final_config,
                expected_test_regions=fold.test_regions,
                expected_cfg=final_cfg,
                current_source_sha=context.training_source_sha256,
                expected_initialized_from_sha256=pretrain_sha,
            )
            summary_sha = write_evaluation_receipt(
                context,
                fold,
                paths,
                final_checkpoint=final_checkpoint,
                final_checkpoint_sha256=final_sha,
            )
            update_plan_artifact(
                plan,
                context,
                fold,
                seed,
                pretrain_sha=pretrain_sha,
                final_sha=final_sha,
                evaluation_summary_sha=summary_sha,
            )


if __name__ == "__main__":
    main()
