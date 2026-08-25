#!/usr/bin/env python3
"""Generate traceable, training-free panorama evaluation figures.

This script is intentionally a compact-result consumer.  It reads only the
formal validation/test JSON receipts and CSV tables written by
``evaluate_stanford_pano.py`` plus the compact exploratory validation receipt
written by ``evaluate_stanford_pano_single_plus_tangent.py``; it never opens
source RGB, depth, semantic, BIM, or model-cache files.  Every plotted scalar
and every source SHA256 is retained in ``manifest.json`` so the figures can be
audited without rerunning a model.

The figure set deliberately excludes learned refinement.  BIM evidence is
limited to the deterministic ``universal_scale`` and ``bim_direct`` branches,
while the primary analysis concerns panorama tangent views and fusion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

GENERATOR = "scripts/analysis/generate_pano_evaluation_assets.py"
FORMAL_PROTOCOL = "stanford-area1-regular-and-pano-tangent-depth-v3"
FORMAL_SCHEMA_VERSION = 3
EXPLORATORY_PROTOCOL = "stanford-area1-val-strict-single-plus-pano-tangent-v4-candidate-v1"
EXPLORATORY_SCHEMA_VERSION = 1
EXPLORATORY_DIRECTORY = "pano_val_single_plus_tangent"
EXPLORATORY_PUBLICATION_STATUS = "candidate_for_future_preregistration"
FSTAR = "joint_huber"
DEFAULT_FORMATS = ("png", "svg", "pdf")
ALLOWED_FORMATS = frozenset(DEFAULT_FORMATS)
CSV_RECEIPTS = {
    "per_room.csv": "per_room_csv",
    "per_station.csv": "per_station_csv",
    "regular_pano_joint_per_station.csv": "regular_pano_joint_per_station_csv",
    "strict_single_per_station.csv": "strict_single_per_station_csv",
    "tangent_per_station.csv": "tangent_per_station_csv",
}

FUSIONS = (
    ("joint_weighted_log", "Weighted log"),
    ("joint_huber", "Huber (F*)"),
    ("joint_synchronized_huber", "Synchronized"),
)
SOURCES = (
    ("regular_only", "Multi-regular only"),
    ("regular_plus_tangent6", "+ tangent6"),
    ("regular_plus_tangent14", "+ tangent14"),
)
DEPTH_METHODS = (
    ("raw_da3", "Raw DA3"),
    ("universal_scale", "Universal scale"),
    ("bim_direct", "BIM direct"),
)

COLORS = {
    "ink": "#203040",
    "muted": "#66788A",
    "grid": "#DDE5EC",
    "regular": "#6B7280",
    "tangent6": "#4C78A8",
    "tangent14": "#2A9D8F",
    "weighted": "#4C78A8",
    "huber": "#2A9D8F",
    "sync": "#A06CD5",
    "raw": "#7A7A7A",
    "scale": "#4C78A8",
    "bim": "#F2A541",
    "worse": "#D65F5F",
    "better": "#2A9D8F",
    "light_blue": "#E8F1FA",
    "light_green": "#E7F5F1",
    "light_orange": "#FFF3DF",
    "light_gray": "#F2F4F7",
}


@dataclass(frozen=True)
class FormalSplit:
    """Verified compact artifacts for one formal split."""

    name: str
    summary: Mapping[str, Any]
    provenance: Mapping[str, Any]
    csv_rows: Mapping[str, tuple[dict[str, str], ...]]


class SourceRegistry:
    """Load compact result files and retain immutable source identities."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self._records: dict[Path, dict[str, Any]] = {}

    def _read(self, path: Path) -> bytes:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required formal result is missing: {resolved}")
        raw = resolved.read_bytes()
        self._records.setdefault(
            resolved,
            {
                "path": _display_path(resolved, self.root),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            },
        )
        return raw

    def load_json(self, path: Path) -> Mapping[str, Any]:
        payload = json.loads(self._read(path).decode("utf-8"), parse_constant=_reject_nan)
        if not isinstance(payload, Mapping):
            raise TypeError(f"{path}: expected a JSON object")
        return payload

    def load_csv(self, path: Path) -> tuple[dict[str, str], ...]:
        raw = self._read(path).decode("utf-8")
        rows = tuple(dict(row) for row in csv.DictReader(raw.splitlines()))
        if not rows:
            raise ValueError(f"{path}: formal CSV is empty")
        return rows

    def sha256(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._records:
            raise KeyError(f"Source was not loaded: {resolved}")
        return str(self._records[resolved]["sha256"])

    @property
    def records(self) -> list[dict[str, Any]]:
        return sorted(self._records.values(), key=lambda item: str(item["path"]))


def _reject_nan(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is not allowed: {value}")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paper/PPT panorama figures from formal scalar artifacts only. "
            "No model, raw image, or ground-truth file is opened."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/stanford_area1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/pano_evaluation/quantitative"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=sorted(ALLOWED_FORMATS),
        default=list(DEFAULT_FORMATS),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--replay-manifest-without-formal-reads",
        type=Path,
        default=None,
        help=(
            "Regenerate figures from an existing quantitative manifest and read only "
            "the new exploratory validation summary/provenance. This audit mode does "
            "not reopen formal val/test artifacts."
        ),
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#526273",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Convert glyphs to paths so the SVG has no external font dependency.
            "svg.fonttype": "path",
            "svg.hashsalt": "PriorBIMDA-pano-evaluation-assets-v1",
        }
    )


def _save_figure(
    figure: Figure,
    output_root: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for extension in formats:
        if extension not in ALLOWED_FORMATS:
            raise ValueError(f"Unsupported output format: {extension}")
        path = output_root / f"{stem}.{extension}"
        if extension == "pdf":
            metadata: dict[str, Any] = {
                "Creator": GENERATOR,
                "CreationDate": None,
                "ModDate": None,
            }
        elif extension == "svg":
            metadata = {"Creator": GENERATOR, "Date": None}
        else:
            metadata = {"Software": GENERATOR}
        figure.savefig(
            path,
            format=extension,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        if extension == "svg":
            normalized = "\n".join(
                line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()
            )
            path.write_text(normalized + "\n", encoding="utf-8")
        outputs.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "format": extension,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    plt.close(figure)
    return outputs


def load_formal_split(
    results_root: Path,
    split: str,
    registry: SourceRegistry,
) -> FormalSplit:
    """Load a formal result split and verify every CSV receipt."""

    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported formal split: {split}")
    directory = results_root / f"pano_{split}"
    summary_path = directory / "summary.json"
    provenance_path = directory / "provenance.json"
    summary = registry.load_json(summary_path)
    provenance = registry.load_json(provenance_path)
    for label, payload in (("summary", summary), ("provenance", provenance)):
        if payload.get("schema_version") != FORMAL_SCHEMA_VERSION:
            raise RuntimeError(f"{split} {label}: expected schema v3")
        if payload.get("protocol") != FORMAL_PROTOCOL:
            raise RuntimeError(f"{split} {label}: unexpected formal protocol")
        if payload.get("split") != split:
            raise RuntimeError(f"{split} {label}: split identity mismatch")
        if payload.get("formal_protocol_eligible") is not True:
            raise RuntimeError(f"{split} {label}: result is not formal-protocol eligible")
    if split == "test" and provenance.get("test_access_explicitly_authorized") is not True:
        raise RuntimeError("Formal test receipt does not record explicit authorization")

    csv_rows: dict[str, tuple[dict[str, str], ...]] = {}
    summary_artifacts = summary.get("artifacts")
    for filename, receipt_key in CSV_RECEIPTS.items():
        path = directory / filename
        csv_rows[filename] = registry.load_csv(path)
        actual_sha = registry.sha256(path)
        receipt = provenance.get(receipt_key)
        if not isinstance(receipt, Mapping) or receipt.get("sha256") != actual_sha:
            raise RuntimeError(f"{split} {filename}: SHA256 differs from provenance receipt")
        summary_sha_key = f"{receipt_key}_sha256"
        if isinstance(summary_artifacts, Mapping):
            recorded = summary_artifacts.get(summary_sha_key)
            if recorded is not None and recorded != actual_sha:
                raise RuntimeError(f"{split} {filename}: SHA256 differs from summary receipt")

    return FormalSplit(
        name=split,
        summary=summary,
        provenance=provenance,
        csv_rows=csv_rows,
    )


def load_exploratory_validation(
    results_root: Path,
    registry: SourceRegistry,
) -> Mapping[str, Any]:
    """Load the strict-single-plus-tangent validation receipt without its CSVs."""

    directory = results_root / EXPLORATORY_DIRECTORY
    summary_path = directory / "summary.json"
    provenance_path = directory / "provenance.json"
    summary = registry.load_json(summary_path)
    provenance = registry.load_json(provenance_path)
    for label, payload in (("summary", summary), ("provenance", provenance)):
        if payload.get("schema_version") != EXPLORATORY_SCHEMA_VERSION:
            raise RuntimeError(f"exploratory validation {label}: expected schema v1")
        if payload.get("protocol") != EXPLORATORY_PROTOCOL:
            raise RuntimeError(f"exploratory validation {label}: unexpected protocol")
        if payload.get("split") != "val":
            raise RuntimeError(f"exploratory validation {label}: split must be val")
        if payload.get("status") != "exploratory_validation_only":
            raise RuntimeError(f"exploratory validation {label}: validation-only status is missing")
        if payload.get("publication_status") != EXPLORATORY_PUBLICATION_STATUS:
            raise RuntimeError(
                f"exploratory validation {label}: publication status is not protocol-v4 candidate"
            )
    if summary.get("formal_v3_result") is not False:
        raise RuntimeError("exploratory validation must not be marked as a formal v3 result")
    if summary.get("test_data_or_csv_accessed") is not False:
        raise RuntimeError("exploratory validation receipt records test access")
    test_access = provenance.get("test_access")
    if not isinstance(test_access, Mapping) or any(
        test_access.get(key) is not False
        for key in ("authorized", "test_csv_opened", "test_raw_files_opened")
    ):
        raise RuntimeError("exploratory validation provenance does not exclude test access")
    method_scope = summary.get("method_scope")
    required_scope = {
        "training_free_only": True,
        "learned": False,
        "BIM": False,
        "GT_scale": False,
        "checkpoint": False,
        "fusion_method": FSTAR,
        "fusion_parameters_frozen": True,
    }
    if not isinstance(method_scope, Mapping) or not _matches(method_scope, required_scope):
        raise RuntimeError("exploratory validation method scope is not the frozen raw protocol")
    expected_methods = [
        "strict_single",
        "strict_single_plus_tangent6",
        "strict_single_plus_tangent14",
    ]
    if summary.get("methods") != expected_methods:
        raise RuntimeError("exploratory validation method order/identity differs")
    if summary.get("station_count") != 30 or summary.get("pano_only_excluded_count") != 2:
        raise RuntimeError("exploratory validation expected 30 paired and 2 pano-only stations")
    fixed_support = summary.get("fixed_support_audit")
    if (
        not isinstance(fixed_support, Mapping)
        or fixed_support.get("same_count_for_all_methods_per_station") is not True
    ):
        raise RuntimeError("exploratory validation lacks a common fixed-support audit")
    artifacts = summary.get("artifacts")
    actual_provenance_sha = registry.sha256(provenance_path)
    if (
        not isinstance(artifacts, Mapping)
        or artifacts.get("provenance_sha256") != actual_provenance_sha
    ):
        raise RuntimeError("exploratory validation provenance SHA256 differs from summary")
    return {"summary": summary, "provenance": provenance}


def _matches(record: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(record.get(key) == value for key, value in expected.items())


def _unique_record(
    records: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Any],
    *,
    context: str,
) -> Mapping[str, Any]:
    matches = [record for record in records if _matches(record, expected)]
    if len(matches) != 1:
        raise RuntimeError(
            f"{context}: expected one record for {dict(expected)}, found {len(matches)}"
        )
    return matches[0]


def route_metric(
    split: FormalSplit,
    route: str,
    *,
    support_scope: str,
    **expected: Any,
) -> float:
    section = split.summary.get(route)
    if not isinstance(section, Mapping):
        raise TypeError(f"{split.name}: missing or malformed {route}")
    metrics = section.get("metrics")
    if not isinstance(metrics, Sequence):
        raise TypeError(f"{split.name}: malformed {route}.metrics")
    record = _unique_record(
        metrics,
        {**expected, "support_scope": support_scope},
        context=f"{split.name}/{route}",
    )
    station_macro = record.get("station_macro")
    if not isinstance(station_macro, Mapping):
        raise TypeError(f"{split.name}/{route}: missing station_macro")
    return float(station_macro["spherical_abs_rel"])


def find_contrast(
    records: Sequence[Mapping[str, Any]],
    *,
    kind: str,
    candidate: Mapping[str, Any] | str,
    reference: Mapping[str, Any] | str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in records
        if row.get("kind") == kind
        and row.get("candidate") == candidate
        and row.get("reference") == reference
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {kind} contrast for {candidate} vs {reference}, found {len(matches)}"
        )
    return matches[0]


def _contrast_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    bootstrap = record.get("room_cluster_paired_bootstrap_primary_abs_rel")
    if not isinstance(bootstrap, Mapping):
        raise TypeError("Formal contrast is missing room-cluster bootstrap")
    interval = bootstrap.get("confidence_interval_95")
    if not isinstance(interval, Sequence) or len(interval) != 2:
        raise RuntimeError("Formal contrast has malformed confidence interval")
    return {
        "reference_abs_rel": float(record["primary_abs_rel_reference"]),
        "candidate_abs_rel": float(record["primary_abs_rel_candidate"]),
        "candidate_minus_reference": float(bootstrap["mean_difference"]),
        "confidence_interval_95": [float(interval[0]), float(interval[1])],
        "bootstrap_repetitions": int(bootstrap["bootstrap_repetitions"]),
        "room_count": int(bootstrap["room_count"]),
        "station_count": int(bootstrap["station_count"]),
    }


def _mean_csv_value(
    rows: Sequence[Mapping[str, str]],
    expected: Mapping[str, str],
    column: str,
) -> float:
    selected = [float(row[column]) for row in rows if _matches(row, expected)]
    if not selected:
        raise RuntimeError(f"No CSV rows for {dict(expected)}")
    return float(np.mean(np.asarray(selected, dtype=np.float64)))


def main_test_payload(test: FormalSplit) -> dict[str, Any]:
    """Primary test numbers: fixed-support quality plus native-union coverage."""

    route = "route_p_regular_plus_pano_raw"
    quality: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    rows = test.csv_rows["regular_pano_joint_per_station.csv"]
    for source_set, label in SOURCES:
        quality.append(
            {
                "source_set": source_set,
                "label": label,
                "spherical_abs_rel": route_metric(
                    test,
                    route,
                    support_scope="common_regular",
                    source_set=source_set,
                    fusion_method=FSTAR,
                    depth_method="raw_da3",
                ),
            }
        )
        coverage.append(
            {
                "source_set": source_set,
                "label": label,
                "mean_solid_angle_coverage_fraction": _mean_csv_value(
                    rows,
                    {
                        "source_set": source_set,
                        "support_scope": "native_union",
                        "fusion_method": FSTAR,
                        "depth_method": "raw_da3",
                    },
                    "solid_angle_coverage_fraction",
                ),
            }
        )
    contrasts = test.summary[route]["contrasts"]
    combined = find_contrast(
        contrasts,
        kind="regular_plus_pano_tangent_over_regular_only_raw",
        candidate={
            "fusion_method": FSTAR,
            "source_set": "regular_plus_tangent14",
            "support_scope": "common_regular",
        },
        reference={
            "fusion_method": FSTAR,
            "source_set": "regular_only",
            "support_scope": "common_regular",
        },
    )
    view_count = find_contrast(
        contrasts,
        kind="regular_plus_tangent14_over_regular_plus_tangent6",
        candidate={
            "fusion_method": FSTAR,
            "source_set": "regular_plus_tangent14",
            "support_scope": "common_regular",
        },
        reference={
            "fusion_method": FSTAR,
            "source_set": "regular_plus_tangent6",
            "support_scope": "common_regular",
        },
    )
    return {
        "split": "test",
        "fusion_method": FSTAR,
        "depth_method": "raw_da3",
        "quality_support": "common_regular",
        "coverage_support": "native_union",
        "quality": quality,
        "coverage": coverage,
        "regular_plus_tangent14_vs_regular": _contrast_payload(combined),
        "tangent14_vs_tangent6": _contrast_payload(view_count),
    }


def fusion_sensitivity_payload(splits: Sequence[FormalSplit]) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        rows: list[dict[str, Any]] = []
        for fusion, fusion_label in FUSIONS:
            for source_set, source_label in SOURCES:
                rows.append(
                    {
                        "fusion_method": fusion,
                        "fusion_label": fusion_label,
                        "source_set": source_set,
                        "source_label": source_label,
                        "spherical_abs_rel": route_metric(
                            split,
                            "route_p_regular_plus_pano_raw",
                            support_scope="common_regular",
                            source_set=source_set,
                            fusion_method=fusion,
                            depth_method="raw_da3",
                        ),
                    }
                )
        values[split.name] = rows
    return {
        "depth_method": "raw_da3",
        "quality_support": "common_regular",
        "selection_note": "F* was frozen on validation before test; alternatives are sensitivity only.",
        "splits": values,
    }


def tangent_ablation_payload(splits: Sequence[FormalSplit]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for split in splits:
        values = {}
        for variant in ("tangent6", "tangent14"):
            values[variant] = route_metric(
                split,
                "route_p_tangent_only",
                support_scope="common_tangent6_tangent14",
                route_variant=variant,
                fusion_method=FSTAR,
                depth_method="raw_da3",
            )
        contrast = find_contrast(
            split.summary["route_p_tangent_only"]["contrasts"],
            kind="tangent14_over_tangent6_view_count_ablation",
            candidate={
                "fusion_method": FSTAR,
                "route_variant": "tangent14",
                "support_scope": "common_tangent6_tangent14",
            },
            reference={
                "fusion_method": FSTAR,
                "route_variant": "tangent6",
                "support_scope": "common_tangent6_tangent14",
            },
        )
        output[split.name] = {"values": values, "contrast": _contrast_payload(contrast)}
    return {
        "fusion_method": FSTAR,
        "depth_method": "raw_da3",
        "quality_support": "common_tangent6_tangent14",
        "splits": output,
    }


def strict_single_payload(splits: Sequence[FormalSplit]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for split in splits:
        contrasts = split.summary["strict_single_frame_evaluation"]["contrasts"]
        methods: list[dict[str, Any]] = []
        for depth_method, label in DEPTH_METHODS:
            contrast = find_contrast(
                contrasts,
                kind="pano_joint_over_strict_single_frame",
                candidate=f"{depth_method}/{FSTAR}",
                reference=f"{depth_method}/strict_single_frame",
            )
            methods.append(
                {
                    "depth_method": depth_method,
                    "label": label,
                    **_contrast_payload(contrast),
                }
            )
        values[split.name] = methods
    return {
        "joint_fusion_method": FSTAR,
        "joint_route": "multi_regular_route_r",
        "panorama_tangents_included": False,
        "comparison_support": "strict selected-frame support",
        "splits": values,
    }


def strict_single_plus_pano_validation_payload(
    exploratory: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the validation-only negative control from its compact summary."""

    summary = exploratory.get("summary")
    if not isinstance(summary, Mapping):
        raise TypeError("exploratory validation summary is missing")
    quality = summary.get("primary_spherical_abs_rel")
    coverage = summary.get("native_union_mean_solid_angle_coverage")
    contrasts = summary.get("contrasts")
    if not isinstance(quality, Mapping) or not isinstance(coverage, Mapping):
        raise TypeError("exploratory validation quality/coverage is malformed")
    if not isinstance(contrasts, Sequence):
        raise TypeError("exploratory validation contrasts are malformed")

    method_specs = (
        ("strict_single", "Single regular"),
        ("strict_single_plus_tangent6", "+ tangent6"),
        ("strict_single_plus_tangent14", "+ tangent14"),
    )
    rows: list[dict[str, Any]] = []
    for method, label in method_specs:
        row: dict[str, Any] = {
            "method": method,
            "label": label,
            "spherical_abs_rel": float(quality[method]),
            "native_union_solid_angle_coverage_fraction": float(coverage[method]),
        }
        if method != "strict_single":
            record = find_contrast(
                contrasts,
                kind="strict_single_plus_pano_tangent_over_strict_single",
                candidate={"method": method},
                reference={"method": "strict_single"},
            )
            row["contrast_vs_strict_single"] = _contrast_payload(record)
        rows.append(row)

    metrics = summary.get("metrics")
    if not isinstance(metrics, Mapping):
        raise TypeError("exploratory validation metrics are malformed")
    for row in rows:
        method_metrics = metrics.get(row["method"])
        if not isinstance(method_metrics, Mapping):
            raise TypeError(f"exploratory validation lacks metrics for {row['method']}")
        station_macro = method_metrics.get("station_macro")
        native_union = method_metrics.get("native_union_coverage")
        if not isinstance(station_macro, Mapping) or not isinstance(native_union, Mapping):
            raise TypeError(f"exploratory validation metrics are incomplete for {row['method']}")
        if not math.isclose(
            row["spherical_abs_rel"],
            float(station_macro["spherical_abs_rel"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"exploratory validation quality disagrees for {row['method']}")
        if not math.isclose(
            row["native_union_solid_angle_coverage_fraction"],
            float(native_union["mean_solid_angle_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"exploratory validation coverage disagrees for {row['method']}")

    return {
        "split": "val",
        "status": "exploratory_validation_only",
        "publication_status": EXPLORATORY_PUBLICATION_STATUS,
        "formal_v3_result": False,
        "training_free_only": True,
        "depth_method": "raw_da3",
        "fusion_method": FSTAR,
        "fusion_parameters_frozen": True,
        "quality_support": "common strict selected-frame support",
        "coverage_support": "per-method native union",
        "station_count": int(summary["station_count"]),
        "room_count": int(summary["room_count"]),
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 42,
        "methods": rows,
        "interpretation": "coverage gain but accuracy regression",
    }


def bim_chain_payload(splits: Sequence[FormalSplit]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for split in splits:
        primary = split.summary["primary_metrics"][FSTAR]
        chain = [
            {
                "depth_method": method,
                "label": label,
                "spherical_abs_rel": float(primary[method]["abs_rel"]),
            }
            for method, label in DEPTH_METHODS
        ]
        local = find_contrast(
            split.summary["contrasts"],
            kind="local_bim_correction_over_scale_only",
            candidate=f"bim_direct/{FSTAR}",
            reference=f"universal_scale/{FSTAR}",
        )
        scale = find_contrast(
            split.summary["contrasts"],
            kind="bim_enhancement_over_raw_da3",
            candidate=f"universal_scale/{FSTAR}",
            reference=f"raw_da3/{FSTAR}",
        )
        values[split.name] = {
            "chain": chain,
            "universal_scale_vs_raw": _contrast_payload(scale),
            "bim_direct_vs_universal_scale": _contrast_payload(local),
        }
    return {
        "fusion_method": FSTAR,
        "training_free_only": True,
        "splits": values,
    }


def room_pair_payload(splits: Sequence[FormalSplit]) -> dict[str, Any]:
    """Recompute every room point directly from formal per-station CSV rows."""

    output: dict[str, Any] = {}
    for split in splits:
        rows = [
            row
            for row in split.csv_rows["regular_pano_joint_per_station.csv"]
            if row["support_scope"] == "common_regular"
            and row["fusion_method"] == FSTAR
            and row["depth_method"] == "raw_da3"
            and row["source_set"] in {"regular_only", "regular_plus_tangent14"}
        ]
        grouped: defaultdict[str, defaultdict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        stations: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            grouped[row["room"]][row["source_set"]].append(float(row["spherical_abs_rel"]))
            stations[row["room"]].add(row["station_id"])
        room_points = []
        for room in sorted(grouped):
            sources = grouped[room]
            if set(sources) != {"regular_only", "regular_plus_tangent14"}:
                raise RuntimeError(f"{split.name}/{room}: incomplete paired room rows")
            reference = float(np.mean(sources["regular_only"]))
            candidate = float(np.mean(sources["regular_plus_tangent14"]))
            room_points.append(
                {
                    "room": room,
                    "station_count": len(stations[room]),
                    "regular_only": reference,
                    "regular_plus_tangent14": candidate,
                    "candidate_minus_reference": candidate - reference,
                }
            )
        contrast = find_contrast(
            split.summary["route_p_regular_plus_pano_raw"]["contrasts"],
            kind="regular_plus_pano_tangent_over_regular_only_raw",
            candidate={
                "fusion_method": FSTAR,
                "source_set": "regular_plus_tangent14",
                "support_scope": "common_regular",
            },
            reference={
                "fusion_method": FSTAR,
                "source_set": "regular_only",
                "support_scope": "common_regular",
            },
        )
        output[split.name] = {
            "room_points_recomputed_from_csv": room_points,
            "formal_station_macro_contrast": _contrast_payload(contrast),
        }
    return {
        "fusion_method": FSTAR,
        "depth_method": "raw_da3",
        "support": "common_regular",
        "room_aggregation": "mean station spherical AbsRel within each room",
        "splits": output,
    }


def _panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.05,
        1.12,
        label,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def plot_main_test(payload: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    colors = [COLORS["regular"], COLORS["tangent6"], COLORS["tangent14"]]
    quality = payload["quality"]
    coverage = payload["coverage"]
    x = np.arange(3)
    bars = axes[0].bar(
        x,
        [row["spherical_abs_rel"] for row in quality],
        color=colors,
        width=0.66,
    )
    axes[0].set_xticks(x, [row["label"] for row in quality])
    axes[0].set_ylabel("Spherical AbsRel (lower is better)")
    axes[0].set_title("Test quality on identical regular support")
    axes[0].grid(axis="y")
    axes[0].set_ylim(0, max(bar.get_height() for bar in bars) * 1.28)
    for bar in bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    contrast = payload["regular_plus_tangent14_vs_regular"]
    axes[0].text(
        0.98,
        0.94,
        (
            f"+14 vs regular: {100 * (1 - contrast['candidate_abs_rel'] / contrast['reference_abs_rel']):.1f}% lower\n"
            f"95% CI of Δ: [{contrast['confidence_interval_95'][0]:+.3f}, "
            f"{contrast['confidence_interval_95'][1]:+.3f}]"
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8.7,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": COLORS["grid"]},
    )

    coverage_values = [row["mean_solid_angle_coverage_fraction"] for row in coverage]
    bars = axes[1].bar(x, np.asarray(coverage_values) * 100, color=colors, width=0.66)
    axes[1].set_xticks(x, [row["label"] for row in coverage])
    axes[1].set_ylabel("Solid-angle coverage (%)")
    axes[1].set_title("Test deployment coverage on native union")
    axes[1].set_ylim(0, 108)
    axes[1].grid(axis="y")
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{bar.get_height():.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    figure.suptitle("Training-free panorama contribution (F* = joint Huber)", y=1.02)
    figure.tight_layout()
    return figure


def plot_fusion_sensitivity(payload: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharey=True)
    color_map = {
        "joint_weighted_log": COLORS["weighted"],
        "joint_huber": COLORS["huber"],
        "joint_synchronized_huber": COLORS["sync"],
    }
    for index, split_name in enumerate(("val", "test")):
        axis = axes[index]
        rows = payload["splits"][split_name]
        for fusion, label in FUSIONS:
            selected = [row for row in rows if row["fusion_method"] == fusion]
            selected.sort(key=lambda row: [item[0] for item in SOURCES].index(row["source_set"]))
            axis.plot(
                np.arange(3),
                [row["spherical_abs_rel"] for row in selected],
                marker="o",
                linewidth=2,
                markersize=5,
                label=label,
                color=color_map[fusion],
            )
        axis.set_xticks(np.arange(3), [item[1] for item in SOURCES])
        axis.set_title(f"{split_name.capitalize()} sensitivity")
        axis.grid(axis="y")
        _panel_label(axis, chr(ord("a") + index))
    axes[0].set_ylabel("Spherical AbsRel on common regular support")
    axes[1].legend(frameon=False, loc="best")
    figure.suptitle("Fusion sensitivity: regular views with 6 or 14 pano tangents", y=1.02)
    figure.tight_layout()
    return figure


def plot_tangent_ablation(payload: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    split_colors = {"val": COLORS["weighted"], "test": COLORS["huber"]}
    for split_name in ("val", "test"):
        values = payload["splits"][split_name]["values"]
        axes[0].plot(
            [6, 14],
            [values["tangent6"], values["tangent14"]],
            marker="o",
            markersize=6,
            linewidth=2,
            label=split_name.capitalize(),
            color=split_colors[split_name],
        )
    axes[0].set_xticks([6, 14])
    axes[0].set_xlabel("Tangent view count")
    axes[0].set_ylabel("Spherical AbsRel")
    axes[0].set_title("Tangent-only DA3 on identical support")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)

    y = np.asarray([1, 0], dtype=float)
    for ypos, split_name in zip(y, ("val", "test"), strict=True):
        contrast = payload["splits"][split_name]["contrast"]
        mean = contrast["candidate_minus_reference"]
        low, high = contrast["confidence_interval_95"]
        axes[1].errorbar(
            mean,
            ypos,
            xerr=[[mean - low], [high - mean]],
            fmt="o",
            color=split_colors[split_name],
            capsize=4,
            markersize=6,
        )
        axes[1].text(
            high + 0.002,
            ypos,
            f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]",
            va="center",
            fontsize=8.5,
        )
    axes[1].axvline(0, color=COLORS["ink"], linewidth=0.9)
    axes[1].set_yticks(y, ["Validation", "Test"])
    axes[1].set_xlabel("Δ AbsRel: tangent14 − tangent6")
    axes[1].set_title("Room-cluster paired 95% CI")
    axes[1].grid(axis="x")
    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    figure.suptitle("View-count ablation for panorama-only tangent inference", y=1.02)
    figure.tight_layout()
    return figure


def plot_strict_single(payload: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))
    method_colors = {
        "raw_da3": COLORS["raw"],
        "universal_scale": COLORS["scale"],
        "bim_direct": COLORS["bim"],
    }
    for column, split_name in enumerate(("val", "test")):
        rows = payload["splits"][split_name]
        top = axes[0, column]
        for row in rows:
            top.plot(
                [0, 1],
                [row["reference_abs_rel"], row["candidate_abs_rel"]],
                marker="o",
                linewidth=2,
                label=row["label"],
                color=method_colors[row["depth_method"]],
            )
        top.set_xticks([0, 1], ["Strict single\nregular", "Multi-regular\nRoute-R joint"])
        top.set_ylabel("Spherical AbsRel" if column == 0 else "")
        top.set_title(f"{split_name.capitalize()}: absolute error")
        top.grid(axis="y")
        if column == 1:
            top.legend(frameon=False, fontsize=8.5)

        bottom = axes[1, column]
        ypos = np.arange(len(rows))[::-1]
        for y, row in zip(ypos, rows, strict=True):
            mean = row["candidate_minus_reference"]
            low, high = row["confidence_interval_95"]
            crosses = low <= 0 <= high
            color = COLORS["worse"] if mean > 0 else COLORS["better"]
            bottom.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=6,
            )
            if crosses:
                bottom.text(
                    high,
                    y + 0.20,
                    "CI crosses 0",
                    color=COLORS["muted"],
                    fontsize=7.8,
                    ha="right",
                )
        bottom.axvline(0, color=COLORS["ink"], linewidth=0.9)
        bottom.set_yticks(ypos, [row["label"] for row in rows])
        bottom.set_xlabel("Δ AbsRel: Route-R joint − strict (negative is better)")
        bottom.set_title(f"{split_name.capitalize()}: paired room-cluster CI")
        bottom.grid(axis="x")
    for index, axis in enumerate(axes.flat):
        _panel_label(axis, chr(ord("a") + index))
    figure.suptitle(
        "Strict single regular versus multi-regular Route-R joint (training-free)", y=1.01
    )
    figure.tight_layout()
    return figure


def plot_strict_single_plus_pano_validation(payload: Mapping[str, Any]) -> Figure:
    """Show the validation-only coverage/accuracy trade-off without implying a test claim."""

    rows = payload["methods"]
    x = np.arange(len(rows))
    labels = [row["label"] for row in rows]
    colors = [COLORS["regular"], COLORS["tangent6"], COLORS["tangent14"]]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    quality = [row["spherical_abs_rel"] for row in rows]
    bars = axes[0].bar(x, quality, color=colors, width=0.66)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Spherical AbsRel (lower is better)")
    axes[0].set_title("Accuracy regression on common strict support")
    axes[0].grid(axis="y")
    axes[0].set_ylim(0, max(quality) * 1.35)
    for index, (bar, value) in enumerate(zip(bars, quality, strict=True)):
        annotation = f"{value:.4f}"
        if index:
            contrast = rows[index]["contrast_vs_strict_single"]
            low, high = contrast["confidence_interval_95"]
            annotation += (
                f"\nΔ {contrast['candidate_minus_reference']:+.4f}"
                f"\n95% CI [{low:+.4f}, {high:+.4f}]"
            )
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            annotation,
            ha="center",
            va="bottom",
            fontsize=8.3,
            color=COLORS["worse"] if index else COLORS["ink"],
        )

    coverage = [100.0 * row["native_union_solid_angle_coverage_fraction"] for row in rows]
    bars = axes[1].bar(x, coverage, color=colors, width=0.66)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Native solid-angle coverage (%)")
    axes[1].set_title("Coverage gain on each method's native union")
    axes[1].set_ylim(0, 112)
    axes[1].grid(axis="y")
    for bar, value in zip(bars, coverage, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.0,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _panel_label(axes[0], "a")
    _panel_label(axes[1], "b")
    figure.suptitle(
        "Exploratory validation-only: coverage gain, but accuracy regression",
        y=1.03,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.005,
        (
            "Raw DA3 · frozen joint Huber · 30 paired val stations · "
            "not a test result or formal v3 claim"
        ),
        ha="center",
        color=COLORS["muted"],
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96))
    return figure


def plot_bim_chain(payload: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    colors = [COLORS["raw"], COLORS["scale"], COLORS["bim"]]
    for index, split_name in enumerate(("val", "test")):
        axis = axes[index]
        split_payload = payload["splits"][split_name]
        chain = split_payload["chain"]
        bars = axis.bar(
            np.arange(3),
            [row["spherical_abs_rel"] for row in chain],
            color=colors,
            width=0.66,
        )
        axis.set_xticks(np.arange(3), [row["label"] for row in chain])
        axis.set_title(split_name.capitalize())
        axis.grid(axis="y")
        axis.set_ylim(0, 0.31)
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{bar.get_height():.3f}",
                ha="center",
                fontsize=8.7,
            )
        local = split_payload["bim_direct_vs_universal_scale"]
        low, high = local["confidence_interval_95"]
        axis.text(
            0.98,
            0.94,
            (
                f"Direct − scale = {local['candidate_minus_reference']:+.4f}\n"
                f"95% CI [{low:+.4f}, {high:+.4f}]\n"
                "local direct is not supported"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.3,
            color=COLORS["worse"],
            bbox={"boxstyle": "round,pad=0.32", "fc": "white", "ec": COLORS["grid"]},
        )
        _panel_label(axis, chr(ord("a") + index))
    axes[0].set_ylabel("Spherical AbsRel (joint Huber)")
    figure.suptitle("Deterministic BIM chain: scale helps; local direct does not", y=1.02)
    figure.tight_layout()
    return figure


def plot_room_pairs(payload: Mapping[str, Any]) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for row_index, split_name in enumerate(("val", "test")):
        split_payload = payload["splits"][split_name]
        points = split_payload["room_points_recomputed_from_csv"]
        slope = axes[row_index, 0]
        forest = axes[row_index, 1]
        for point in points:
            color = COLORS["better"] if point["candidate_minus_reference"] < 0 else COLORS["worse"]
            slope.plot(
                [0, 1],
                [point["regular_only"], point["regular_plus_tangent14"]],
                marker="o",
                linewidth=1.4,
                alpha=0.85,
                color=color,
            )
        slope.set_xticks([0, 1], ["Multi-regular\nRoute-R", "+ tangent14"])
        slope.set_ylabel("Room-mean spherical AbsRel")
        slope.set_title(f"{split_name.capitalize()}: paired room slopes")
        slope.grid(axis="y")

        ordered = sorted(points, key=lambda item: item["candidate_minus_reference"])
        y = np.arange(len(ordered))[::-1]
        deltas = [point["candidate_minus_reference"] for point in ordered]
        colors = [COLORS["better"] if value < 0 else COLORS["worse"] for value in deltas]
        forest.scatter(deltas, y, c=colors, s=32, zorder=3)
        for ypos, point in zip(y, ordered, strict=True):
            forest.plot([0, point["candidate_minus_reference"]], [ypos, ypos], color=COLORS["grid"])
        aggregate = split_payload["formal_station_macro_contrast"]
        mean = aggregate["candidate_minus_reference"]
        low, high = aggregate["confidence_interval_95"]
        aggregate_y = -1.2
        forest.errorbar(
            mean,
            aggregate_y,
            xerr=[[mean - low], [high - mean]],
            fmt="D",
            color=COLORS["ink"],
            capsize=4,
            markersize=5,
        )
        labels = [f"{point['room']} (n={point['station_count']})" for point in ordered]
        forest.set_yticks([*y, aggregate_y], [*labels, "Formal aggregate 95% CI"])
        forest.axvline(0, color=COLORS["ink"], linewidth=0.9)
        forest.set_xlabel("Δ AbsRel: +tangent14 − regular")
        forest.set_title(f"{split_name.capitalize()}: room points from formal CSV")
        forest.grid(axis="x")
    for index, axis in enumerate(axes.flat):
        _panel_label(axis, chr(ord("a") + index))
    figure.suptitle("Room-level audit of the selected panorama claim", y=1.01)
    figure.tight_layout()
    return figure


def _diagram_axis(figsize: tuple[float, float] = (11.5, 3.7)) -> tuple[Figure, Axes]:
    figure, axis = plt.subplots(figsize=figsize)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return figure, axis


def _box(
    axis: Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    detail: str,
    *,
    facecolor: str,
    edgecolor: str = "#6B7C8F",
) -> None:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.2,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height * 0.63, title, ha="center", va="center", fontweight="bold")
    axis.text(
        x + width / 2,
        y + height * 0.29,
        detail,
        ha="center",
        va="center",
        fontsize=8.2,
        color=COLORS["muted"],
        linespacing=1.2,
    )


def _arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color=COLORS["ink"],
        )
    )


def plot_process_a() -> Figure:
    figure, axis = _diagram_axis()
    stages = (
        ("ERP panorama", "4096×2048 RGB\nknown pose", COLORS["light_blue"]),
        ("nested14", "6 cube faces +\n8 cube corners", COLORS["light_green"]),
        ("DA3 inference", "shared frozen model\n14 tangent depths", COLORS["light_orange"]),
        ("ERP reprojection", "z-depth → radial range\nsolid-angle geometry", COLORS["light_blue"]),
        ("Robust fusion", "F*: joint Huber\nno GT scale", COLORS["light_green"]),
        ("Pano depth", "0.2–5.0 m protocol\ncoverage + uncertainty", COLORS["light_orange"]),
    )
    width, gap, left, y = 0.135, 0.024, 0.025, 0.35
    for index, (title, detail, color) in enumerate(stages):
        x = left + index * (width + gap)
        _box(axis, (x, y), (width, 0.30), title, detail, facecolor=color)
        if index < len(stages) - 1:
            _arrow(axis, (x + width + 0.003, y + 0.15), (x + width + gap - 0.003, y + 0.15))
    axis.text(
        0.025,
        0.87,
        "Candidate A — training-free ERP tangent pipeline",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        0.025,
        0.78,
        "Panorama supplies viewing directions and coverage; DA3 remains unchanged.",
        color=COLORS["muted"],
    )
    axis.text(
        0.5,
        0.15,
        "No learned refiner • no test-time fitting • deterministic tangent geometry",
        ha="center",
        color=COLORS["ink"],
        bbox={"boxstyle": "round,pad=0.38", "fc": COLORS["light_gray"], "ec": "none"},
    )
    return figure


def plot_process_b() -> Figure:
    figure, axis = _diagram_axis((10.8, 5.5))
    axis.text(
        0.04, 0.94, "Candidate B — orthogonal evaluation matrix", fontsize=15, fontweight="bold"
    )
    axis.text(
        0.04,
        0.87,
        "Separate panorama evidence from deterministic BIM evidence on fixed support.",
        color=COLORS["muted"],
    )
    columns = ("Multi-regular\nonly", "+ tangent6", "+ tangent14")
    rows = ("No BIM\n(raw DA3)", "Universal\nscale", "BIM direct\n(local)")
    cell_colors = (COLORS["light_gray"], COLORS["light_blue"], COLORS["light_green"])
    x0, y0, width, height = 0.27, 0.17, 0.22, 0.18
    for col, label in enumerate(columns):
        axis.text(
            x0 + col * width + width / 2, 0.78, label, ha="center", va="center", fontweight="bold"
        )
    for row, label in enumerate(rows):
        y = y0 + (2 - row) * height
        axis.text(0.18, y + height / 2, label, ha="center", va="center", fontweight="bold")
        for col in range(3):
            x = x0 + col * width
            axis.add_patch(
                Rectangle(
                    (x, y),
                    width - 0.012,
                    height - 0.012,
                    facecolor=cell_colors[row],
                    edgecolor="white",
                    linewidth=2,
                )
            )
            axis.text(
                x + (width - 0.012) / 2,
                y + (height - 0.012) / 2,
                f"P{col} × B{row}",
                ha="center",
                va="center",
                color=COLORS["ink"],
            )
    axis.text(
        0.38, 0.08, "Panorama effect →", ha="center", color=COLORS["weighted"], fontweight="bold"
    )
    axis.text(
        0.90, 0.45, "BIM effect ↑", rotation=90, ha="center", color=COLORS["bim"], fontweight="bold"
    )
    axis.text(
        0.04,
        0.05,
        "Primary claim: P2B0 − P0B0   |   BIM chain: P0B1/P0B2 − P0B0   |   learned branch excluded",
        fontsize=8.8,
        color=COLORS["muted"],
    )
    return figure


def plot_process_c(main_payload: Mapping[str, Any]) -> Figure:
    figure, axis = _diagram_axis((11.5, 4.2))
    coverage = {
        row["source_set"]: 100 * row["mean_solid_angle_coverage_fraction"]
        for row in main_payload["coverage"]
    }
    stages = (
        (
            "Registered inputs",
            "regular RGB + ERP pano\nexact same camera center",
            COLORS["light_blue"],
        ),
        (
            "Multi-regular route",
            f"all regular projections\n{coverage['regular_only']:.2f}% coverage",
            COLORS["light_gray"],
        ),
        ("Panorama route", "cubemap6 / nested14\nDA3 tangent inference", COLORS["light_green"]),
        (
            "Coverage union",
            f"+6: {coverage['regular_plus_tangent6']:.2f}%\n+14: {coverage['regular_plus_tangent14']:.2f}%",
            COLORS["light_orange"],
        ),
        ("Robust depth", "joint Huber F*\nquality on common support", COLORS["light_green"]),
    )
    width, gap, left, y = 0.16, 0.032, 0.03, 0.34
    for index, (title, detail, color) in enumerate(stages):
        x = left + index * (width + gap)
        _box(axis, (x, y), (width, 0.31), title, detail, facecolor=color)
        if index < len(stages) - 1:
            _arrow(axis, (x + width + 0.004, y + 0.155), (x + width + gap - 0.004, y + 0.155))
    axis.text(
        0.03, 0.89, "Candidate C — deployment and coverage flow", fontsize=15, fontweight="bold"
    )
    axis.text(
        0.03,
        0.79,
        "The views are same-center observations: fusion improves angular evidence, not baseline parallax.",
        color=COLORS["muted"],
    )
    axis.text(
        0.5,
        0.13,
        "SAME CENTER  →  NO TRIANGULATION / NO MVS CLAIM",
        ha="center",
        fontweight="bold",
        color=COLORS["worse"],
        bbox={"boxstyle": "round,pad=0.40", "fc": "#FFF0F0", "ec": "#F1B5B5"},
    )
    return figure


FIGURE_DESCRIPTIONS = {
    "test_main_pano_gain": "Primary test panorama quality and coverage result.",
    "fusion_sensitivity": "Validation/test fusion sensitivity; alternatives do not redefine F*.",
    "tangent_view_count_ablation": "Six-versus-fourteen tangent-view ablation with paired CI.",
    "strict_single_vs_joint": (
        "Strict single regular versus all-regular Route-R joint; no pano tangents."
    ),
    "strict_single_plus_pano_validation": (
        "Exploratory validation-only negative control: pano tangents gain coverage but regress "
        "accuracy on common strict support; not a formal/test claim."
    ),
    "deterministic_bim_chain": (
        "Training-free raw→scale→direct chain; local direct correction is negative/null."
    ),
    "room_level_paired_pano_claim": (
        "Room points recomputed exactly from formal per-station CSV rows."
    ),
    "process_candidate_a_erp_tangent_pipeline": "Candidate A: method pipeline.",
    "process_candidate_b_orthogonal_matrix": "Candidate B: orthogonal evaluation design.",
    "process_candidate_c_deployment_coverage": (
        "Candidate C: same-center deployment coverage flow."
    ),
}


def _render_figures(
    payloads: Mapping[str, Any],
    output_root: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[dict[str, Any]]:
    plotters = {
        "test_main_pano_gain": lambda: plot_main_test(payloads["test_main_pano_gain"]),
        "fusion_sensitivity": lambda: plot_fusion_sensitivity(payloads["fusion_sensitivity"]),
        "tangent_view_count_ablation": lambda: plot_tangent_ablation(
            payloads["tangent_view_count_ablation"]
        ),
        "strict_single_vs_joint": lambda: plot_strict_single(payloads["strict_single_vs_joint"]),
        "strict_single_plus_pano_validation": lambda: plot_strict_single_plus_pano_validation(
            payloads["strict_single_plus_pano_validation"]
        ),
        "deterministic_bim_chain": lambda: plot_bim_chain(payloads["deterministic_bim_chain"]),
        "room_level_paired_pano_claim": lambda: plot_room_pairs(
            payloads["room_level_paired_pano_claim"]
        ),
        "process_candidate_a_erp_tangent_pipeline": plot_process_a,
        "process_candidate_b_orthogonal_matrix": plot_process_b,
        "process_candidate_c_deployment_coverage": lambda: plot_process_c(
            payloads["test_main_pano_gain"]
        ),
    }
    if set(payloads) != set(plotters):
        missing = sorted(set(plotters) - set(payloads))
        extra = sorted(set(payloads) - set(plotters))
        raise RuntimeError(f"Figure payload identities differ: missing={missing}, extra={extra}")
    figures: list[dict[str, Any]] = []
    for figure_id, plotter in plotters.items():
        figure = plotter()
        files = _save_figure(figure, output_root, figure_id, formats, dpi)
        figures.append(
            {
                "id": figure_id,
                "description": FIGURE_DESCRIPTIONS[figure_id],
                "numeric_payload": payloads[figure_id],
                "files": files,
            }
        )
    return figures


def generate_assets(
    *,
    results_root: Path,
    output_root: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 300,
) -> dict[str, Any]:
    """Load compact scalar artifacts, generate ten figures, and write a manifest."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not formats or any(item not in ALLOWED_FORMATS for item in formats):
        raise ValueError(f"formats must be selected from {sorted(ALLOWED_FORMATS)}")
    _configure_matplotlib()
    project_root = Path(__file__).resolve().parents[2]
    registry = SourceRegistry(project_root)
    val = load_formal_split(results_root, "val", registry)
    test = load_formal_split(results_root, "test", registry)
    exploratory = load_exploratory_validation(results_root, registry)
    splits = (val, test)

    payloads = {
        "test_main_pano_gain": main_test_payload(test),
        "fusion_sensitivity": fusion_sensitivity_payload(splits),
        "tangent_view_count_ablation": tangent_ablation_payload(splits),
        "strict_single_vs_joint": strict_single_payload(splits),
        "strict_single_plus_pano_validation": strict_single_plus_pano_validation_payload(
            exploratory
        ),
        "deterministic_bim_chain": bim_chain_payload(splits),
        "room_level_paired_pano_claim": room_pair_payload(splits),
        "process_candidate_a_erp_tangent_pipeline": {
            "stages": ["ERP", "nested14", "DA3", "ERP reprojection", "joint Huber", "pano depth"],
            "training_free": True,
        },
        "process_candidate_b_orthogonal_matrix": {
            "panorama_axis": ["regular_only", "regular_plus_tangent6", "regular_plus_tangent14"],
            "bim_axis": ["raw_da3", "universal_scale", "bim_direct"],
            "learned_refinement_included": False,
        },
        "process_candidate_c_deployment_coverage": {
            "same_camera_center": True,
            "triangulation_claim": False,
            "test_coverage": main_test_payload(test)["coverage"],
        },
    }
    figures = _render_figures(payloads, output_root, formats, dpi)

    manifest = {
        "schema_version": 1,
        "generator": GENERATOR,
        "generator_sha256": sha256_file(Path(__file__)),
        "formal_protocol": FORMAL_PROTOCOL,
        "formal_schema_version": FORMAL_SCHEMA_VERSION,
        "exploratory_validation_protocol": EXPLORATORY_PROTOCOL,
        "generation_mode": "full_compact_sources",
        "primary_fusion": {
            "id": FSTAR,
            "label": "joint Huber",
            "status": "frozen on validation before formal test",
        },
        "method_scope": {
            "training_free_only": True,
            "included_depth_methods": [item[0] for item in DEPTH_METHODS],
            "learned_refinement_plotted": False,
            "station-local_BIM_variants_plotted": False,
            "quality_range_m": [0.2, 5.0],
            "same_camera_center": True,
            "triangulation_or_mvs_claim": False,
        },
        "source_access_contract": (
            "Formal summary/CSV/provenance receipts plus exploratory val summary/provenance only; "
            "no raw images, GT, BIM, prediction caches, checkpoints, or model execution."
        ),
        "sources": registry.records,
        "figures": figures,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def regenerate_assets_without_formal_reads(
    *,
    results_root: Path,
    output_root: Path,
    replay_manifest: Path,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = 300,
) -> dict[str, Any]:
    """Add the exploratory val figure while replaying already-audited formal payloads.

    This mode exists for embargo-safe figure maintenance: it deliberately does
    not reopen any formal validation/test result, including the formal test CSVs.
    The prior manifest supplies their numeric payloads and recorded source
    identities; only the exploratory validation summary/provenance are opened.
    """

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not formats or any(item not in ALLOWED_FORMATS for item in formats):
        raise ValueError(f"formats must be selected from {sorted(ALLOWED_FORMATS)}")
    raw_manifest = replay_manifest.read_bytes()
    previous = json.loads(raw_manifest.decode("utf-8"), parse_constant=_reject_nan)
    if not isinstance(previous, Mapping) or previous.get("schema_version") != 1:
        raise RuntimeError("Replay input is not a quantitative manifest schema v1")
    if previous.get("formal_protocol") != FORMAL_PROTOCOL:
        raise RuntimeError("Replay input formal protocol differs")
    previous_figures = previous.get("figures")
    previous_sources = previous.get("sources")
    if not isinstance(previous_figures, Sequence) or not isinstance(previous_sources, Sequence):
        raise TypeError("Replay input lacks figures or source identities")
    previous_payloads: dict[str, Any] = {}
    for record in previous_figures:
        if not isinstance(record, Mapping):
            raise TypeError("Replay input has a malformed figure record")
        figure_id = str(record.get("id"))
        if figure_id in previous_payloads:
            raise RuntimeError(f"Replay input repeats figure {figure_id}")
        previous_payloads[figure_id] = record.get("numeric_payload")
    exploratory_id = "strict_single_plus_pano_validation"
    expected_prior = set(FIGURE_DESCRIPTIONS) - {exploratory_id}
    if not expected_prior.issubset(previous_payloads):
        missing = sorted(expected_prior - set(previous_payloads))
        raise RuntimeError(f"Replay input is missing prior figure payloads: {missing}")
    unexpected = set(previous_payloads) - set(FIGURE_DESCRIPTIONS)
    if unexpected:
        raise RuntimeError(f"Replay input has unknown figure payloads: {sorted(unexpected)}")

    project_root = Path(__file__).resolve().parents[2]
    registry = SourceRegistry(project_root)
    exploratory = load_exploratory_validation(results_root, registry)
    payloads = {figure_id: previous_payloads[figure_id] for figure_id in expected_prior}
    main_payload = payloads["test_main_pano_gain"]
    if not isinstance(main_payload, Mapping):
        raise TypeError("Replay main-test payload is malformed")
    for collection_name in ("quality", "coverage"):
        collection = main_payload.get(collection_name)
        if not isinstance(collection, Sequence):
            raise TypeError(f"Replay main-test {collection_name} payload is malformed")
        for row in collection:
            if isinstance(row, dict) and row.get("source_set") == "regular_only":
                row["label"] = "Multi-regular only"
    process_c_payload = payloads["process_candidate_c_deployment_coverage"]
    if not isinstance(process_c_payload, Mapping):
        raise TypeError("Replay process-C payload is malformed")
    process_c_coverage = process_c_payload.get("test_coverage")
    if not isinstance(process_c_coverage, Sequence):
        raise TypeError("Replay process-C coverage payload is malformed")
    for row in process_c_coverage:
        if isinstance(row, dict) and row.get("source_set") == "regular_only":
            row["label"] = "Multi-regular only"
    sensitivity_payload = payloads["fusion_sensitivity"]
    if not isinstance(sensitivity_payload, Mapping):
        raise TypeError("Replay fusion-sensitivity payload is malformed")
    sensitivity_splits = sensitivity_payload.get("splits")
    if not isinstance(sensitivity_splits, Mapping):
        raise TypeError("Replay fusion-sensitivity split payload is malformed")
    for rows in sensitivity_splits.values():
        if not isinstance(rows, Sequence):
            raise TypeError("Replay fusion-sensitivity rows are malformed")
        for row in rows:
            if isinstance(row, dict) and row.get("source_set") == "regular_only":
                row["source_label"] = "Multi-regular only"
    strict_payload = payloads["strict_single_vs_joint"]
    if not isinstance(strict_payload, Mapping):
        raise TypeError("Replay strict-single payload is malformed")
    payloads["strict_single_vs_joint"] = {
        **strict_payload,
        "joint_route": "multi_regular_route_r",
        "panorama_tangents_included": False,
    }
    payloads[exploratory_id] = strict_single_plus_pano_validation_payload(exploratory)

    source_by_path: dict[str, dict[str, Any]] = {}
    for record in previous_sources:
        if not isinstance(record, Mapping):
            raise TypeError("Replay input has a malformed source identity")
        path = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("Replay input has an invalid source identity")
        source_by_path[path] = dict(record)
    for record in registry.records:
        source_by_path[str(record["path"])] = record
    sources = [source_by_path[path] for path in sorted(source_by_path)]

    formal_payload_digest = hashlib.sha256(
        json.dumps(
            _json_safe({figure_id: payloads[figure_id] for figure_id in sorted(expected_prior)}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _configure_matplotlib()
    figures = _render_figures(payloads, output_root, formats, dpi)
    manifest = {
        "schema_version": 1,
        "generator": GENERATOR,
        "generator_sha256": sha256_file(Path(__file__)),
        "formal_protocol": FORMAL_PROTOCOL,
        "formal_schema_version": FORMAL_SCHEMA_VERSION,
        "exploratory_validation_protocol": EXPLORATORY_PROTOCOL,
        "generation_mode": "manifest_payload_replay_plus_exploratory_validation_receipt",
        "formal_numeric_payload_sha256": formal_payload_digest,
        "formal_sources_reopened": False,
        "test_csv_reopened": False,
        "primary_fusion": {
            "id": FSTAR,
            "label": "joint Huber",
            "status": "frozen on validation before formal test",
        },
        "method_scope": {
            "training_free_only": True,
            "included_depth_methods": [item[0] for item in DEPTH_METHODS],
            "learned_refinement_plotted": False,
            "station-local_BIM_variants_plotted": False,
            "quality_range_m": [0.2, 5.0],
            "same_camera_center": True,
            "triangulation_or_mvs_claim": False,
        },
        "source_access_contract": (
            "Formal numeric payloads/source identities replayed from the prior quantitative "
            "manifest without reopening formal artifacts; only exploratory val summary/provenance "
            "opened. No test CSV, raw images, GT, BIM, prediction caches, checkpoints, or model "
            "execution."
        ),
        "sources": sources,
        "figures": figures,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.replay_manifest_without_formal_reads is None:
        manifest = generate_assets(
            results_root=args.results_root,
            output_root=args.output,
            formats=tuple(args.formats),
            dpi=args.dpi,
        )
    else:
        manifest = regenerate_assets_without_formal_reads(
            results_root=args.results_root,
            output_root=args.output,
            replay_manifest=args.replay_manifest_without_formal_reads,
            formats=tuple(args.formats),
            dpi=args.dpi,
        )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "figures": len(manifest["figures"]),
                "sources": len(manifest["sources"]),
                "training_free_only": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
