#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bim_priorda3.region_cv import (
    RegionCVProtocol,
    build_region_fold_plans,
    parse_region_cv_protocol,
    region_macro_summary,
)

SCHEMA_VERSION = 1
METRICS = ("abs_rel", "rmse", "mae", "delta1", "delta2", "delta3")
METHOD_ALIASES = {
    "base": ("base",),
    "scaled": ("scaled", "global_scale"),
    "direct_bim": ("direct_bim", "anchor", "previous_scale_local"),
    "refined": ("refined",),
}
BASELINE_METHODS = ("base", "scaled", "direct_bim")
COMPARISON_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PlannedRun:
    fold_index: int
    fold_id: str
    test_region: str
    seed: int
    checkpoint_selection: str
    selected_checkpoint: Path
    evaluation_summary: Path
    per_frame_csv: Path
    expected_checkpoint_sha256: str | None = None


@dataclass
class RunObservation:
    plan: PlannedRun
    status: str = "failed"
    errors: list[dict[str, str]] = field(default_factory=list)
    checkpoint_sha256: str = ""
    method_sources: dict[str, str] = field(default_factory=dict)
    method_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    per_frame_baselines: dict[str, tuple[tuple[Any, ...], ...]] = field(default_factory=dict)
    valid_for_aggregation: bool = False

    def fail(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})
        self.status = "failed"
        self.valid_for_aggregation = False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Required file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _protocol_body(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = payload.get("protocol")
    if isinstance(nested, Mapping) and {
        "regions",
        "validation_map",
        "seeds",
    }.issubset(nested):
        return nested
    return payload


def _resolve_project_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _optional_string(mapping: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _parse_checkpoint_selection(
    payload: Mapping[str, Any],
    *,
    context: str,
) -> dict[str, str]:
    raw = payload.get("checkpoint_selection")
    if raw is None:
        return {"pretrain": "accepted", "final": "accepted"}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{context} checkpoint_selection must be an object")
    if set(raw) != {"pretrain", "final"}:
        raise ValueError(
            f"{context} checkpoint_selection keys must be exactly ['pretrain', 'final']"
        )
    selection = {}
    for stage in ("pretrain", "final"):
        value = raw[stage]
        if value not in {"accepted", "best"}:
            raise ValueError(f"{context} checkpoint_selection.{stage} must be 'accepted' or 'best'")
        selection[stage] = str(value)
    return selection


def _plan_run_paths(
    protocol_dir: Path,
    project_root: Path,
    fold_id: str,
    seed: int,
    run_spec: Mapping[str, Any] | None,
    final_checkpoint_selection: str,
) -> tuple[Path, Path, Path, str | None]:
    run_dir = protocol_dir / "folds" / fold_id / f"seed_{seed}"
    checkpoint = run_dir / f"{final_checkpoint_selection}.pt"
    summary = run_dir / "evaluation_test" / "summary.json"
    per_frame = run_dir / "evaluation_test" / "per_frame.csv"
    expected_sha = None
    if run_spec is None:
        return checkpoint, summary, per_frame, expected_sha

    checkpoints = run_spec.get("checkpoints", {})
    if isinstance(checkpoints, Mapping):
        checkpoint_keys = [
            "final_selected",
            f"final_{final_checkpoint_selection}",
        ]
        if final_checkpoint_selection == "accepted":
            checkpoint_keys.extend(("final_accepted", "accepted"))
        checkpoint_value = _optional_string(
            checkpoints,
            *checkpoint_keys,
        )
        checkpoint_sha_keys = [
            "final_selected_sha256",
            f"final_{final_checkpoint_selection}_sha256",
        ]
        if final_checkpoint_selection == "accepted":
            checkpoint_sha_keys.extend(("final_accepted_sha256", "accepted_sha256"))
        checkpoint_sha_keys.append("sha256")
        expected_sha = _optional_string(
            checkpoints,
            *checkpoint_sha_keys,
        )
        if checkpoint_value:
            checkpoint = _resolve_project_path(checkpoint_value, project_root)

    checkpoint_sha = run_spec.get("checkpoint_sha256")
    if isinstance(checkpoint_sha, Mapping):
        sha_keys = [
            "final_selected",
            f"final_{final_checkpoint_selection}",
        ]
        if final_checkpoint_selection == "accepted":
            sha_keys.extend(("final_accepted", "accepted"))
        expected_sha = _optional_string(checkpoint_sha, *sha_keys) or expected_sha
    elif isinstance(checkpoint_sha, str) and checkpoint_sha:
        expected_sha = checkpoint_sha
    elif checkpoint_sha is not None:
        raise TypeError("run checkpoint_sha256 must be a string or object")
    direct_sha_keys = ["selected_checkpoint_sha256"]
    if final_checkpoint_selection == "accepted":
        direct_sha_keys.append("accepted_checkpoint_sha256")
    expected_sha = _optional_string(run_spec, *direct_sha_keys) or expected_sha

    summary_value = _optional_string(
        run_spec,
        "evaluation_summary",
        "summary",
    )
    if summary_value:
        summary = _resolve_project_path(summary_value, project_root)
        per_frame = summary.parent / "per_frame.csv"
    output_dirs = run_spec.get("output_dirs", {})
    if isinstance(output_dirs, Mapping) and not summary_value:
        evaluation_dir = _optional_string(output_dirs, "evaluation", "evaluation_test")
        if evaluation_dir:
            evaluation_path = _resolve_project_path(evaluation_dir, project_root)
            summary = evaluation_path / "summary.json"
            per_frame = evaluation_path / "per_frame.csv"
    per_frame_value = _optional_string(run_spec, "per_frame_csv")
    if per_frame_value:
        per_frame = _resolve_project_path(per_frame_value, project_root)
    return checkpoint, summary, per_frame, expected_sha


def _extract_run_spec(
    fold_payload: Mapping[str, Any],
    seed: int,
) -> Mapping[str, Any] | None:
    raw_runs = fold_payload.get("runs")
    if raw_runs is None:
        return None
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise TypeError(f"{fold_payload.get('fold_id')}: runs must be a list")
    matches = [run for run in raw_runs if isinstance(run, Mapping) and run.get("seed") == seed]
    if len(matches) > 1:
        raise ValueError(f"{fold_payload.get('fold_id')}: duplicate run entries for seed {seed}")
    return matches[0] if matches else None


def _validate_run_specs(
    fold_payload: Mapping[str, Any],
    expected_seeds: Sequence[int],
) -> None:
    raw_runs = fold_payload.get("runs")
    if raw_runs is None:
        return
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise TypeError(f"{fold_payload.get('fold_id')}: runs must be a list")
    if any(not isinstance(run, Mapping) for run in raw_runs):
        raise TypeError(f"{fold_payload.get('fold_id')}: every run must be an object")
    run_seeds = [run.get("seed") for run in raw_runs]
    if len(run_seeds) != len(set(run_seeds)):
        raise ValueError(f"{fold_payload.get('fold_id')}: duplicate run seeds")
    if set(run_seeds) != set(expected_seeds):
        raise ValueError(
            f"{fold_payload.get('fold_id')}: run seeds do not match fold seeds; "
            f"expected={list(expected_seeds)}, actual={run_seeds}"
        )


def _validate_and_expand_plan(
    protocol_dir: Path,
    project_root: Path,
    protocol_payload: Mapping[str, Any],
    plan_payload: Mapping[str, Any],
) -> tuple[str, RegionCVProtocol, dict[str, str], list[PlannedRun]]:
    protocol_body = _protocol_body(protocol_payload)
    protocol = parse_region_cv_protocol(protocol_body)
    protocol_selection_source = (
        protocol_body if "checkpoint_selection" in protocol_body else protocol_payload
    )
    checkpoint_selection = _parse_checkpoint_selection(
        protocol_selection_source,
        context="protocol.json",
    )
    plan_checkpoint_selection = _parse_checkpoint_selection(
        plan_payload,
        context="plan.json",
    )
    if plan_checkpoint_selection != checkpoint_selection:
        raise ValueError("plan.json checkpoint_selection does not match protocol.json")
    protocol_id = str(
        plan_payload.get(
            "protocol_id",
            protocol_payload.get("protocol_id", protocol_dir.name),
        )
    )
    if not protocol_id:
        raise ValueError("protocol_id must not be empty")
    payload_protocol_id = protocol_payload.get("protocol_id")
    if payload_protocol_id is not None and str(payload_protocol_id) != protocol_id:
        raise ValueError(
            "protocol.json and plan.json disagree on protocol_id: "
            f"{payload_protocol_id!r} != {protocol_id!r}"
        )
    if protocol_dir.name != protocol_id:
        raise ValueError(
            f"protocol directory name {protocol_dir.name!r} does not match "
            f"protocol_id {protocol_id!r}"
        )

    for key, expected in (
        ("regions", list(protocol.regions)),
        ("validation_map", protocol.validation_map),
        ("seeds", list(protocol.seeds)),
    ):
        if key in plan_payload and plan_payload[key] != expected:
            raise ValueError(f"plan.json {key} does not match protocol.json")

    expected_hash = plan_payload.get("protocol_sha256")
    if expected_hash:
        accepted_hashes = {
            _canonical_json_sha256(protocol_payload),
            _canonical_json_sha256(protocol_body),
            _file_sha256(protocol_dir / "protocol.json"),
        }
        embedded_hash = protocol_payload.get("protocol_sha256")
        if isinstance(embedded_hash, str):
            accepted_hashes.add(embedded_hash)
        if expected_hash not in accepted_hashes:
            raise ValueError("plan.json protocol_sha256 does not match protocol.json")

    raw_folds = plan_payload.get("folds")
    if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, (str, bytes)):
        raise TypeError("plan.json folds must be a list")
    actual_folds: dict[str, Mapping[str, Any]] = {}
    for fold in raw_folds:
        if not isinstance(fold, Mapping):
            raise TypeError("each plan fold must be an object")
        fold_id = fold.get("fold_id")
        if not isinstance(fold_id, str) or not fold_id:
            raise ValueError("each plan fold must have a non-empty fold_id")
        if fold_id in actual_folds:
            raise ValueError(f"duplicate plan fold_id: {fold_id}")
        actual_folds[fold_id] = fold

    expected_folds = build_region_fold_plans(protocol)
    expected_ids = {fold.fold_id for fold in expected_folds}
    if set(actual_folds) != expected_ids:
        raise ValueError(
            "plan folds do not match protocol; "
            f"missing={sorted(expected_ids - set(actual_folds))}, "
            f"extra={sorted(set(actual_folds) - expected_ids)}"
        )

    planned_runs = []
    for expected_fold in expected_folds:
        actual = actual_folds[expected_fold.fold_id]
        for key, expected in (
            ("fold_index", expected_fold.fold_index),
            ("train_regions", list(expected_fold.train_regions)),
            ("val_regions", list(expected_fold.val_regions)),
            ("test_regions", list(expected_fold.test_regions)),
            ("seeds", list(expected_fold.seeds)),
        ):
            if actual.get(key) != expected:
                raise ValueError(f"{expected_fold.fold_id}: {key} does not match protocol")
        _validate_run_specs(actual, expected_fold.seeds)
        test_region = expected_fold.test_regions[0]
        for seed in expected_fold.seeds:
            run_spec = _extract_run_spec(actual, seed)
            checkpoint, summary, per_frame, expected_sha = _plan_run_paths(
                protocol_dir,
                project_root,
                expected_fold.fold_id,
                seed,
                run_spec,
                checkpoint_selection["final"],
            )
            planned_runs.append(
                PlannedRun(
                    fold_index=expected_fold.fold_index,
                    fold_id=expected_fold.fold_id,
                    test_region=test_region,
                    seed=seed,
                    checkpoint_selection=checkpoint_selection["final"],
                    selected_checkpoint=checkpoint,
                    evaluation_summary=summary,
                    per_frame_csv=per_frame,
                    expected_checkpoint_sha256=expected_sha,
                )
            )
    return protocol_id, protocol, checkpoint_selection, planned_runs


def _finite_metric(
    value: object,
    *,
    context: str,
    allow_zero: bool = True,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{context} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{context} must be numeric") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{context} must be finite")
    if not allow_zero and normalized <= 0:
        raise ValueError(f"{context} must be positive")
    return normalized


def _extract_methods(
    summary: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    overall = summary.get("overall")
    if not isinstance(overall, Mapping):
        raise TypeError("evaluation summary overall must be an object")
    sources = {}
    methods = {}
    for canonical, aliases in METHOD_ALIASES.items():
        available = [alias for alias in aliases if alias in overall]
        if len(available) != 1:
            raise ValueError(
                f"evaluation summary must contain exactly one {canonical} method "
                f"from {aliases}; found {available}"
            )
        source = available[0]
        raw_metrics = overall[source]
        if not isinstance(raw_metrics, Mapping):
            raise TypeError(f"overall.{source} must be an object")
        metrics = {
            name: _finite_metric(
                raw_metrics.get(name),
                context=f"overall.{source}.{name}",
            )
            for name in METRICS
        }
        count = _finite_metric(
            raw_metrics.get("count"),
            context=f"overall.{source}.count",
            allow_zero=False,
        )
        metrics["count"] = count
        sources[canonical] = source
        methods[canonical] = metrics

    counts = {round(metrics["count"]) for metrics in methods.values()}
    if len(counts) != 1:
        raise ValueError("evaluation methods use different valid-pixel counts")
    return sources, methods


def _read_per_frame_baselines(
    path: Path,
    test_region: str,
    method_sources: Mapping[str, str],
) -> dict[str, tuple[tuple[Any, ...], ...]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: per_frame.csv has no rows")
    sample_ids = [row.get("sample_id", "") for row in rows]
    expected_prefix = f"{test_region}/"
    invalid_ids = [
        sample_id for sample_id in sample_ids if not sample_id.startswith(expected_prefix)
    ]
    if invalid_ids:
        raise ValueError(f"{path}: sample IDs do not belong to test region {test_region!r}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{path}: duplicate sample IDs")

    signatures = {}
    for canonical in BASELINE_METHODS:
        source = method_sources[canonical]
        signature = []
        for row_index, row in enumerate(rows, 2):
            values: list[Any] = [row["sample_id"]]
            for metric in (*METRICS, "count"):
                column = f"{source}_{metric}"
                if column not in row:
                    raise ValueError(f"{path}:{row_index}: missing column {column}")
                values.append(
                    _finite_metric(
                        row[column],
                        context=f"{path}:{row_index}:{column}",
                    )
                )
            signature.append(tuple(values))
        signatures[canonical] = tuple(signature)
    return signatures


def _load_run(observation: RunObservation, project_root: Path) -> None:
    plan = observation.plan
    missing = []
    for label, path in (
        ("selected checkpoint", plan.selected_checkpoint),
        ("evaluation summary", plan.evaluation_summary),
        ("per-frame evaluation", plan.per_frame_csv),
    ):
        if not path.is_file():
            missing.append(f"{label}: {path}")
    if missing:
        observation.fail("missing_artifact", "; ".join(missing))
        return

    selected_sha = _file_sha256(plan.selected_checkpoint)
    observation.checkpoint_sha256 = selected_sha
    if plan.expected_checkpoint_sha256 and plan.expected_checkpoint_sha256 != selected_sha:
        observation.fail(
            "checkpoint_sha_mismatch",
            "selected checkpoint SHA-256 differs from plan: "
            f"{selected_sha} != {plan.expected_checkpoint_sha256}",
        )
        return

    try:
        summary = _read_json(plan.evaluation_summary)
        if summary.get("split") != "test":
            raise ValueError("evaluation summary split must be 'test'")
        if summary.get("regions") != [plan.test_region]:
            raise ValueError("evaluation summary regions do not match the planned test region")
        summary_checkpoint = summary.get("checkpoint")
        if not isinstance(summary_checkpoint, str) or not summary_checkpoint:
            raise ValueError("evaluation summary has no checkpoint path")
        evaluated_checkpoint = _resolve_project_path(
            summary_checkpoint,
            project_root,
        )
        if not evaluated_checkpoint.is_file():
            raise ValueError(f"evaluation checkpoint does not exist: {evaluated_checkpoint}")
        evaluated_sha = _file_sha256(evaluated_checkpoint)
        if evaluated_sha != selected_sha:
            raise ValueError(
                "evaluation checkpoint SHA-256 differs from selected checkpoint: "
                f"{evaluated_sha} != {selected_sha}"
            )
        recorded_sha = summary.get("checkpoint_sha256")
        if recorded_sha is not None and recorded_sha != selected_sha:
            raise ValueError(
                "evaluation summary checkpoint_sha256 differs from selected "
                f"checkpoint: {recorded_sha} != {selected_sha}"
            )
        sources, methods = _extract_methods(summary)
        per_frame = _read_per_frame_baselines(
            plan.per_frame_csv,
            plan.test_region,
            sources,
        )
    except (OSError, TypeError, ValueError) as error:
        observation.fail("invalid_evaluation", str(error))
        return

    observation.method_sources = sources
    observation.method_metrics = methods
    observation.per_frame_baselines = per_frame
    observation.status = "success"
    observation.valid_for_aggregation = True


def _numbers_equal(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=COMPARISON_TOLERANCE,
    )


def _baseline_difference(
    reference: RunObservation,
    candidate: RunObservation,
) -> str | None:
    for method in BASELINE_METHODS:
        for metric in (*METRICS, "count"):
            left = reference.method_metrics[method][metric]
            right = candidate.method_metrics[method][metric]
            if not _numbers_equal(left, right):
                return (
                    f"{method}.{metric} differs between seeds "
                    f"{reference.plan.seed} and {candidate.plan.seed}: "
                    f"{left} != {right}"
                )
        left_rows = reference.per_frame_baselines[method]
        right_rows = candidate.per_frame_baselines[method]
        if len(left_rows) != len(right_rows):
            return (
                f"{method} per-frame row count differs between seeds "
                f"{reference.plan.seed} and {candidate.plan.seed}"
            )
        for row_index, (left_row, right_row) in enumerate(
            zip(left_rows, right_rows),
            1,
        ):
            if left_row[0] != right_row[0]:
                return (
                    f"{method} sample order differs at row {row_index} between "
                    f"seeds {reference.plan.seed} and {candidate.plan.seed}"
                )
            if any(
                not _numbers_equal(float(left), float(right))
                for left, right in zip(left_row[1:], right_row[1:])
            ):
                return (
                    f"{method} per-frame metrics differ for {left_row[0]} "
                    f"between seeds {reference.plan.seed} and "
                    f"{candidate.plan.seed}"
                )
    return None


def _enforce_fold_baseline_consistency(
    observations: Sequence[RunObservation],
) -> None:
    by_fold: dict[str, list[RunObservation]] = {}
    for observation in observations:
        if observation.status == "success":
            by_fold.setdefault(observation.plan.fold_id, []).append(observation)
    for fold_observations in by_fold.values():
        if len(fold_observations) < 2:
            continue
        reference = fold_observations[0]
        inconsistency = next(
            (
                difference
                for candidate in fold_observations[1:]
                if (difference := _baseline_difference(reference, candidate))
            ),
            None,
        )
        if inconsistency is None:
            continue
        for observation in fold_observations:
            observation.fail("baseline_inconsistent_across_seeds", inconsistency)


def _run_failures(
    observations: Sequence[RunObservation],
) -> list[dict[str, Any]]:
    return [
        {
            "fold_id": observation.plan.fold_id,
            "test_region": observation.plan.test_region,
            "seed": observation.plan.seed,
            "errors": observation.errors,
        }
        for observation in observations
        if observation.status != "success"
    ]


def _fold_metric_rows(
    observations: Sequence[RunObservation],
) -> list[dict[str, Any]]:
    rows = []
    for observation in observations:
        base = {
            "fold_index": observation.plan.fold_index,
            "fold_id": observation.plan.fold_id,
            "test_region": observation.plan.test_region,
            "seed": observation.plan.seed,
            "status": observation.status,
            "failure_codes": ";".join(error["code"] for error in observation.errors),
            "failure_messages": " | ".join(error["message"] for error in observation.errors),
            "checkpoint_selection": observation.plan.checkpoint_selection,
            "checkpoint_sha256": observation.checkpoint_sha256,
        }
        if observation.status != "success":
            rows.append(
                {
                    **base,
                    "method": "",
                    "source_method": "",
                    **{metric: "" for metric in (*METRICS, "count")},
                }
            )
            continue
        for method in METHOD_ALIASES:
            rows.append(
                {
                    **base,
                    "method": method,
                    "source_method": observation.method_sources[method],
                    **observation.method_metrics[method],
                }
            )
    return rows


def _aggregate_regions(
    protocol: RegionCVProtocol,
    observations: Sequence[RunObservation],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, float]]]]:
    valid_by_region: dict[str, list[RunObservation]] = {}
    for observation in observations:
        if observation.valid_for_aggregation:
            valid_by_region.setdefault(
                observation.plan.test_region,
                [],
            ).append(observation)

    rows = []
    per_region: dict[str, dict[str, dict[str, float]]] = {}
    planned_seed_count = len(protocol.seeds)
    for region in protocol.regions:
        valid = sorted(
            valid_by_region.get(region, []),
            key=lambda observation: observation.plan.seed,
        )
        per_region[region] = {}
        for method in METHOD_ALIASES:
            row: dict[str, Any] = {
                "test_region": region,
                "method": method,
                "status": ("complete" if len(valid) == planned_seed_count else "incomplete"),
                "successful_seed_count": len(valid),
                "planned_seed_count": planned_seed_count,
                "successful_seeds": ";".join(str(observation.plan.seed) for observation in valid),
            }
            if not valid:
                row.update(
                    {
                        **{metric: "" for metric in (*METRICS, "count")},
                        **{f"{metric}_seed_std": "" for metric in METRICS},
                    }
                )
                rows.append(row)
                continue
            averaged = {}
            for metric in METRICS:
                values = [observation.method_metrics[method][metric] for observation in valid]
                averaged[metric] = sum(values) / len(values)
                row[f"{metric}_seed_std"] = (
                    math.sqrt(
                        sum((value - averaged[metric]) ** 2 for value in values) / (len(values) - 1)
                    )
                    if len(values) > 1
                    else 0.0
                )
            counts = {observation.method_metrics[method]["count"] for observation in valid}
            if len(counts) != 1:
                raise ValueError(f"{region}/{method}: valid-pixel count varies across seeds")
            averaged["count"] = counts.pop()
            row.update(averaged)
            rows.append(row)
            per_region[region][method] = averaged
    return rows, per_region


def _macro_metrics(
    per_region: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    macro = {}
    for method in METHOD_ALIASES:
        available = {
            region: metrics[method] for region, metrics in per_region.items() if method in metrics
        }
        macro[method] = (
            region_macro_summary(available, metric_names=METRICS)
            if available
            else {
                "aggregation": "region_macro",
                "region_count": 0,
                "regions": [],
                "metrics": {},
            }
        )
    return macro


def _comparison_summary(
    per_region: Mapping[str, Mapping[str, Mapping[str, float]]],
    macro: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    comparable = {
        region: metrics
        for region, metrics in per_region.items()
        if "direct_bim" in metrics and "refined" in metrics
    }
    won = []
    lost = []
    tied = []
    relative_by_region = {}
    for region, metrics in comparable.items():
        direct = metrics["direct_bim"]["abs_rel"]
        refined = metrics["refined"]["abs_rel"]
        relative_by_region[region] = (direct - refined) / direct if direct > 0 else float("nan")
        if _numbers_equal(direct, refined):
            tied.append(region)
        elif refined < direct:
            won.append(region)
        else:
            lost.append(region)

    direct_macro_metrics = macro["direct_bim"]["metrics"]
    refined_macro_metrics = macro["refined"]["metrics"]
    if "abs_rel" in direct_macro_metrics and "abs_rel" in refined_macro_metrics:
        direct_macro = direct_macro_metrics["abs_rel"]["mean"]
        refined_macro = refined_macro_metrics["abs_rel"]["mean"]
        absolute_improvement = direct_macro - refined_macro
        relative_improvement = (
            absolute_improvement / direct_macro if direct_macro > 0 else float("nan")
        )
    else:
        direct_macro = None
        refined_macro = None
        absolute_improvement = None
        relative_improvement = None

    worst_region = None
    if comparable:
        name = max(
            comparable,
            key=lambda region: comparable[region]["refined"]["abs_rel"],
        )
        worst_region = {
            "region": name,
            "refined_abs_rel": comparable[name]["refined"]["abs_rel"],
            "direct_bim_abs_rel": comparable[name]["direct_bim"]["abs_rel"],
            "relative_improvement": relative_by_region[name],
        }
    return {
        "baseline": "direct_bim",
        "metric": "abs_rel",
        "regions_compared": len(comparable),
        "regions_won": len(won),
        "regions_lost": len(lost),
        "regions_tied": len(tied),
        "won_regions": won,
        "lost_regions": lost,
        "tied_regions": tied,
        "relative_improvement_by_region": relative_by_region,
        "direct_bim_region_macro_abs_rel": direct_macro,
        "refined_region_macro_abs_rel": refined_macro,
        "absolute_improvement": absolute_improvement,
        "relative_improvement": relative_improvement,
        "worst_region": worst_region,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: object, digits: int = 6) -> str:
    return "—" if value is None or value == "" else f"{float(value):.{digits}f}"


def _render_report(summary: Mapping[str, Any]) -> str:
    lines = [
        f"# Region Cross-Validation Report: {summary['protocol_id']}",
        "",
        f"Status: **{summary['status']}**",
        "",
        (
            f"Completed runs: {summary['successful_runs']}/"
            f"{summary['planned_runs']}; summarized regions: "
            f"{summary['regions_summarized']}/{summary['planned_regions']}."
        ),
        "",
        (
            "Checkpoint selection: pretrain="
            f"`{summary['checkpoint_selection']['pretrain']}`, final="
            f"`{summary['checkpoint_selection']['final']}`."
        ),
        "",
        (
            "Aggregation order: average seeds within each held-out region, then "
            "average regions with equal weight."
        ),
        "",
    ]
    if summary["status"] != "complete":
        lines.extend(
            [
                (
                    "> WARNING: This is an incomplete result. Missing or invalid "
                    "runs were recorded as failures and were not silently discarded."
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Region-macro metrics",
            "",
            "| Method | AbsRel ↓ | RMSE ↓ | MAE ↓ | δ1 ↑ | Regions |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in METHOD_ALIASES:
        method_summary = summary["region_macro"][method]
        metrics = method_summary["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _format_metric(metrics.get("abs_rel", {}).get("mean")),
                    _format_metric(metrics.get("rmse", {}).get("mean")),
                    _format_metric(metrics.get("mae", {}).get("mean")),
                    _format_metric(metrics.get("delta1", {}).get("mean")),
                    str(method_summary["region_count"]),
                ]
            )
            + " |"
        )

    comparison = summary["comparison_to_direct_bim"]
    lines.extend(
        [
            "",
            "## Learned versus direct BIM",
            "",
            (
                f"- Regions won: {comparison['regions_won']}/"
                f"{comparison['regions_compared']} "
                f"({', '.join(comparison['won_regions']) or 'none'})."
            ),
            (
                "- Region-macro AbsRel relative improvement: "
                f"{_format_metric(comparison['relative_improvement'] * 100 if comparison['relative_improvement'] is not None else None, 3)}%."
            ),
        ]
    )
    if comparison["worst_region"]:
        worst = comparison["worst_region"]
        lines.append(
            f"- Worst region by refined AbsRel: {worst['region']} "
            f"({_format_metric(worst['refined_abs_rel'])})."
        )

    lines.extend(
        [
            "",
            "## Per-region AbsRel",
            "",
            "| Region | Seeds | Base | Scaled | Direct BIM | Refined | Relative improvement |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    region_rows = summary["region_metrics"]
    for region in summary["regions"]:
        metrics = region_rows.get(region, {})
        seed_count = metrics.get("_successful_seed_count", 0)
        direct = metrics.get("direct_bim", {}).get("abs_rel")
        refined = metrics.get("refined", {}).get("abs_rel")
        relative = (
            (direct - refined) / direct if direct not in (None, 0) and refined is not None else None
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    region,
                    str(seed_count),
                    _format_metric(metrics.get("base", {}).get("abs_rel")),
                    _format_metric(metrics.get("scaled", {}).get("abs_rel")),
                    _format_metric(metrics.get("direct_bim", {}).get("abs_rel")),
                    _format_metric(metrics.get("refined", {}).get("abs_rel")),
                    (f"{_format_metric(relative * 100, 3)}%" if relative is not None else "—"),
                ]
            )
            + " |"
        )

    failures = summary["failures"]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Fold | Region | Seed | Code | Detail |",
                "|---|---|---:|---|---|",
            ]
        )
        for failure in failures:
            codes = ", ".join(error["code"] for error in failure["errors"])
            details = " / ".join(
                error["message"].replace("|", "\\|") for error in failure["errors"]
            )
            lines.append(
                f"| {failure['fold_id']} | {failure['test_region']} | "
                f"{failure['seed']} | {codes} | {details} |"
            )
    return "\n".join(lines) + "\n"


def summarize_region_cv(protocol_dir: Path) -> dict[str, Any]:
    protocol_dir = protocol_dir.resolve()
    project_root = Path(__file__).resolve().parents[2]
    protocol_path = protocol_dir / "protocol.json"
    plan_path = protocol_dir / "plan.json"
    protocol_payload = _read_json(protocol_path)
    plan_payload = _read_json(plan_path)
    protocol_id, protocol, checkpoint_selection, planned_runs = _validate_and_expand_plan(
        protocol_dir,
        project_root,
        protocol_payload,
        plan_payload,
    )

    observations = [RunObservation(plan=plan) for plan in planned_runs]
    for observation in observations:
        _load_run(observation, project_root)
    _enforce_fold_baseline_consistency(observations)

    fold_rows = _fold_metric_rows(observations)
    region_rows, per_region = _aggregate_regions(protocol, observations)
    macro = _macro_metrics(per_region)
    failures = _run_failures(observations)
    successful_runs = sum(observation.status == "success" for observation in observations)
    summarized_regions = sum("refined" in region_metrics for region_metrics in per_region.values())
    complete = (
        not failures
        and successful_runs == len(planned_runs)
        and summarized_regions == len(protocol.regions)
    )

    serialized_region_metrics: dict[str, Any] = {}
    for region in protocol.regions:
        valid = [
            observation
            for observation in observations
            if observation.valid_for_aggregation and observation.plan.test_region == region
        ]
        serialized_region_metrics[region] = {
            "_successful_seed_count": len(valid),
            "_planned_seed_count": len(protocol.seeds),
            **per_region[region],
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "protocol_sha256": _canonical_json_sha256(protocol_payload),
        "status": "complete" if complete else "incomplete",
        "aggregation": "seed_mean_then_region_macro",
        "checkpoint_selection": checkpoint_selection,
        "regions": list(protocol.regions),
        "seeds": list(protocol.seeds),
        "planned_regions": len(protocol.regions),
        "regions_summarized": summarized_regions,
        "planned_runs": len(planned_runs),
        "successful_runs": successful_runs,
        "failed_runs": len(planned_runs) - successful_runs,
        "failures": failures,
        "region_macro": macro,
        "region_metrics": serialized_region_metrics,
        "comparison_to_direct_bim": _comparison_summary(per_region, macro),
        "artifacts": {
            "fold_metrics_csv": "fold_metrics.csv",
            "region_metrics_csv": "region_metrics.csv",
            "report": "REPORT.md",
        },
    }
    _write_csv(protocol_dir / "fold_metrics.csv", fold_rows)
    _write_csv(protocol_dir / "region_metrics.csv", region_rows)
    (protocol_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (protocol_dir / "REPORT.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and summarize region cross-validation outputs."
    )
    parser.add_argument(
        "protocol_dir",
        type=Path,
        help="outputs/region_cv/<protocol_id> directory",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Return success after writing a clearly marked partial summary.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = summarize_region_cv(args.protocol_dir)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}")
        return 2
    print(
        f"{summary['status']}: {summary['successful_runs']}/"
        f"{summary['planned_runs']} runs, "
        f"{summary['regions_summarized']}/{summary['planned_regions']} regions"
    )
    if summary["status"] != "complete" and not args.allow_incomplete:
        print(
            "ERROR: incomplete protocol; rerun with --allow-incomplete only "
            "for diagnostic partial reporting"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
