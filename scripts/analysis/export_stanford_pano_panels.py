#!/usr/bin/env python3
"""Export validation-only Stanford pano panels from the frozen v3 evaluator.

Selection is performed exclusively from the formal validation summary/CSVs.
Only after the three station UUIDs are frozen does this script initialize the
dataset, open images, or recompute predictions.  Geometry, projection, fusion,
GT loading, and metric calculation are delegated to the exact evaluator whose
SHA256 is recorded by the formal result provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, TwoSlopeNorm

from bim_priorda3.config import Config
from bim_priorda3.data import BIMDepthDataset
from bim_priorda3.data.stanford_pano import StanfordPanorama

# Direct ``python scripts/analysis/...py`` execution otherwise exposes only the
# analysis directory on sys.path.  Keep the repository script package importable
# without changing the frozen evaluator or relying on an editable source root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.model import evaluate_stanford_pano as evaluator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMAL_SCHEMA_VERSION = 3
FORMAL_PROTOCOL = "stanford-area1-regular-and-pano-tangent-depth-v3"
FORMAL_SPLIT = "val"
FSTAR_EXPECTED = "joint_huber"
RAW_METHOD = "raw_da3"
EXPORTED_ROUTE_R_METHODS = ("universal_scale", "bim_direct")
SOURCE_REGULAR = "regular_only"
SOURCE_JOINT14 = "regular_plus_tangent14"
SUPPORT_SCOPE = "common_regular"
HEIGHT = 512
WIDTH = 1024
DEPTH_MIN_M = 0.2
DEPTH_MAX_M = 5.0
ABSREL_MAX = 0.5
SIGNED_LIMIT = 0.25
INVALID_RGB = np.asarray((48, 48, 48), dtype=np.uint8)

FORMAL_CSV_NAMES = (
    "per_station.csv",
    "strict_single_per_station.csv",
    "per_room.csv",
    "tangent_per_station.csv",
    "regular_pano_joint_per_station.csv",
)

PANEL_NAMES = (
    "pano_rgb",
    "gt_range",
    "regular_only_raw",
    "regular_plus_tangent14_raw",
    "universal_scale",
    "bim_direct",
    "regular_only_raw_absrel",
    "regular_plus_tangent14_raw_absrel",
    "universal_scale_absrel",
    "bim_direct_absrel",
    "joint_minus_regular_signed_absrel",
    "coverage",
    "support",
)
RETIRED_PANEL_NAMES = ("learned_refined", "learned_refined_absrel")


@dataclass(frozen=True)
class Selection:
    """One preselected formal-validation station."""

    role: str
    station_id: str
    room: str
    gain_abs_rel: float
    reference_abs_rel: float
    candidate_abs_rel: float
    rule: Mapping[str, Any]
    reference_csv_row: Mapping[str, str]
    candidate_csv_row: Mapping[str, str]


@dataclass(frozen=True)
class FormalArtifacts:
    """Validated scalar-only formal validation artifacts."""

    results_dir: Path
    summary: Mapping[str, Any]
    provenance: Mapping[str, Any]
    regular_rows: tuple[dict[str, str], ...]
    route_r_rows: tuple[dict[str, str], ...]
    identities: Mapping[str, Mapping[str, Any]]
    fstar: str
    fstar_selection: Mapping[str, Any]


@dataclass(frozen=True)
class RuntimeContext:
    """Objects initialized once after validation-only station selection."""

    cfg: Config
    dataset: BIMDepthDataset
    device: torch.device
    model: torch.nn.Module | None
    evaluator_args: argparse.Namespace
    tangent_bundle: evaluator.TangentManifestBundle
    stations: Mapping[str, StanfordPanorama]
    records_by_station: Mapping[str, tuple[Mapping[str, Any], ...]]
    runtime_identity: Mapping[str, Any]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _metric_tolerance(value: str) -> float:
    parsed = _positive_float(value)
    if parsed > 1e-5:
        raise argparse.ArgumentTypeError("metric reproduction tolerance may not exceed 1e-5")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and export three auditable Stanford Area_1 panorama examples from "
            "formal validation artifacts only. There is intentionally no test-split option."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/stanford_area1/pano_val"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/pano_evaluation/qualitative"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional relocated config; its SHA256 must equal formal provenance.",
    )
    parser.add_argument(
        "--tangent-manifest",
        type=Path,
        help="Optional relocated val tangent manifest; its SHA256 must match provenance.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--metric-atol", type=_metric_tolerance, default=2e-6)
    parser.add_argument(
        "--preview-sheet",
        action="store_true",
        help="Also write a labelled contact sheet; independent panel PNGs remain title-free.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(value: str | Path) -> str:
    """Represent repository-owned artifacts as relocatable POSIX paths."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _json_safe(value: Any, *, field_name: str | None = None) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, field_name=field_name) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item(), field_name=field_name)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return _portable_path(value)
    if isinstance(value, str) and (
        field_name
        in {
            "path",
            "directory",
            "annotation_file",
            "formal_recorded_path",
            "manifest_recorded_path",
            "resolved_path",
        }
        or (field_name is not None and field_name.endswith("_path"))
    ):
        return _portable_path(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-finite JSON constant {value!r}")

    payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path}: expected a JSON object")
    return payload


def _read_csv(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: CSV is empty")
    return rows


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    return value


def _required_float(row: Mapping[str, Any], key: str, context: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{context}: missing/non-numeric {key!r}") from error
    if not math.isfinite(value):
        raise ValueError(f"{context}: {key!r} must be finite")
    return value


def _recorded_csv_key(filename: str) -> str:
    return {
        "per_station.csv": "per_station_csv",
        "strict_single_per_station.csv": "strict_single_per_station_csv",
        "per_room.csv": "per_room_csv",
        "tangent_per_station.csv": "tangent_per_station_csv",
        "regular_pano_joint_per_station.csv": "regular_pano_joint_per_station_csv",
    }[filename]


def _require_identity(path: Path, expected_sha256: str, context: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} is missing: {resolved}")
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"{context} SHA256 differs: expected={expected_sha256}, actual={actual}")
    return {"path": str(resolved), "sha256": actual}


def _area_root_relative_identity(
    path: Path,
    expected_sha256: str,
    *,
    area_root: Path,
    context: str,
) -> dict[str, Any]:
    """Hash a Stanford source while publishing only its area-root-relative path."""

    identity = _require_identity(path, expected_sha256, context)
    resolved_path = Path(identity["path"])
    resolved_root = area_root.expanduser().resolve()
    try:
        logical_path = resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise RuntimeError(f"{context} is outside configured data.area_root") from error
    return {
        "path": logical_path,
        "path_base": "data.area_root",
        "sha256": identity["sha256"],
    }


def select_fstar(summary: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Apply the preregistered raw-val station-macro AbsRel/MAE selection."""

    primary = _mapping(summary.get("primary_metrics"), "summary.primary_metrics")
    pool = tuple(evaluator.FUSION_METHODS[1:])
    candidates = []
    for fusion in pool:
        fusion_metrics = _mapping(primary.get(fusion), f"primary_metrics.{fusion}")
        raw_metrics = _mapping(fusion_metrics.get(RAW_METHOD), f"{fusion}.{RAW_METHOD}")
        candidates.append(
            {
                "fusion_method": fusion,
                "spherical_equal_station_abs_rel": _required_float(
                    raw_metrics, "abs_rel", f"{fusion}/{RAW_METHOD}"
                ),
                "spherical_equal_station_mae_m": _required_float(
                    raw_metrics, "mae", f"{fusion}/{RAW_METHOD}"
                ),
            }
        )
    winner = min(
        candidates,
        key=lambda item: (
            item["spherical_equal_station_abs_rel"],
            item["spherical_equal_station_mae_m"],
            item["fusion_method"],
        ),
    )["fusion_method"]
    receipt = {
        "name": "F*",
        "selection_split": FORMAL_SPLIT,
        "selection_depth_method": RAW_METHOD,
        "candidate_pool": list(pool),
        "excluded": [
            "strict_single_frame",
            "single_best_view/per-ray mosaic",
            "Route-P source variants",
            "all BIM/scale/learned depth methods",
        ],
        "primary": "exact-solid-angle equal-station macro AbsRel",
        "tie_break": "exact-solid-angle equal-station macro MAE, then fusion name",
        "candidates": candidates,
        "winner": winner,
    }
    return str(winner), receipt


def load_formal_val_artifacts(results_dir: Path) -> FormalArtifacts:
    """Validate formal val scalar artifacts without touching any image/depth data."""

    root = results_dir.expanduser().resolve()
    summary_path = root / "summary.json"
    provenance_path = root / "provenance.json"
    if not summary_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(f"Formal summary/provenance is missing under {root}")
    summary = _read_json(summary_path)
    provenance = _read_json(provenance_path)
    for name, payload in (("summary", summary), ("provenance", provenance)):
        if payload.get("schema_version") != FORMAL_SCHEMA_VERSION:
            raise RuntimeError(f"{name}: expected schema {FORMAL_SCHEMA_VERSION}")
        if payload.get("protocol") != FORMAL_PROTOCOL:
            raise RuntimeError(f"{name}: unexpected protocol {payload.get('protocol')!r}")
        if payload.get("split") != FORMAL_SPLIT:
            raise RuntimeError(f"{name}: only formal val artifacts are permitted")
        if payload.get("formal_protocol_eligible") is not True:
            raise RuntimeError(f"{name}: artifact is not formal-protocol eligible")
    if provenance.get("test_access_explicitly_authorized") is not False:
        raise RuntimeError("Provenance indicates test access; qualitative selection refuses it")
    if summary.get("station_count") != 30:
        raise RuntimeError("Formal paired validation population must contain exactly 30 stations")
    primary_protocol = _mapping(
        summary.get("primary_metric_protocol"), "summary.primary_metric_protocol"
    )
    if primary_protocol.get("station_aggregation") != "equal-station macro":
        raise RuntimeError("Formal primary is not the preregistered equal-station macro")
    if primary_protocol.get("fixed_support_shared_by_all_methods") is not True:
        raise RuntimeError("Formal result does not declare fixed support across methods")

    identities: dict[str, Mapping[str, Any]] = {
        "summary.json": {
            "path": str(summary_path),
            "sha256": file_sha256(summary_path),
        },
        "provenance.json": {
            "path": str(provenance_path),
            "sha256": file_sha256(provenance_path),
        },
    }
    for filename in FORMAL_CSV_NAMES:
        local_path = root / filename
        recorded = _mapping(
            provenance.get(_recorded_csv_key(filename)), f"provenance identity for {filename}"
        )
        if Path(str(recorded.get("path", ""))).name != filename:
            raise RuntimeError(f"Provenance path for {filename} has a different basename")
        identities[filename] = _require_identity(
            local_path, str(recorded.get("sha256", "")), f"formal {filename}"
        )

    tangent = _mapping(provenance.get("route_p_tangent_inputs"), "route_p_tangent_inputs")
    if tangent.get("status") != "verified_formal_manifest":
        raise RuntimeError("Formal val lacks a verified Route-P tangent manifest")

    regular_rows = _read_csv(root / "regular_pano_joint_per_station.csv")
    route_r_rows = _read_csv(root / "per_station.csv")
    fstar, fstar_receipt = select_fstar(summary)
    if fstar != FSTAR_EXPECTED:
        raise RuntimeError(
            f"Frozen F* was expected to be {FSTAR_EXPECTED!r}, but formal val selects {fstar!r}"
        )
    return FormalArtifacts(
        results_dir=root,
        summary=summary,
        provenance=provenance,
        regular_rows=regular_rows,
        route_r_rows=route_r_rows,
        identities=identities,
        fstar=fstar,
        fstar_selection=fstar_receipt,
    )


def select_three_stations(
    rows: Sequence[Mapping[str, str]],
    *,
    fstar: str,
    expected_station_count: int,
) -> tuple[Selection, Selection, Selection]:
    """Select median, maximum, and minimum raw gain before images are opened."""

    relevant = [
        dict(row)
        for row in rows
        if row.get("support_scope") == SUPPORT_SCOPE
        and row.get("fusion_method") == fstar
        and row.get("depth_method") == RAW_METHOD
        and row.get("source_set") in (SOURCE_REGULAR, SOURCE_JOINT14)
    ]
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in relevant:
        station_id = str(row.get("station_id", ""))
        source = str(row.get("source_set", ""))
        if not station_id or source in grouped[station_id]:
            raise RuntimeError(f"Duplicate or empty formal selection row: {station_id}/{source}")
        grouped[station_id][source] = row
    if len(grouped) != expected_station_count:
        raise RuntimeError(
            "Selection matrix does not match formal station population: "
            f"expected={expected_station_count}, actual={len(grouped)}"
        )

    candidates = []
    for station_id, source_rows in grouped.items():
        if set(source_rows) != {SOURCE_REGULAR, SOURCE_JOINT14}:
            raise RuntimeError(f"{station_id}: incomplete regular/tangent14 selection pair")
        reference = source_rows[SOURCE_REGULAR]
        candidate = source_rows[SOURCE_JOINT14]
        if reference.get("room") != candidate.get("room"):
            raise RuntimeError(f"{station_id}: paired selection rows disagree on room")
        support_keys = (
            "fixed_support_pixels",
            "gt_valid_pixels_at_eval_resolution",
            "coverage_fraction",
            "solid_angle_coverage_fraction",
        )
        for key in support_keys:
            if _required_float(reference, key, station_id) != _required_float(
                candidate, key, station_id
            ):
                raise RuntimeError(f"{station_id}: fixed-support field {key!r} differs")
        reference_absrel = _required_float(reference, "spherical_abs_rel", station_id)
        candidate_absrel = _required_float(candidate, "spherical_abs_rel", station_id)
        candidates.append(
            {
                "station_id": station_id,
                "room": str(reference["room"]),
                "gain": reference_absrel - candidate_absrel,
                "reference_absrel": reference_absrel,
                "candidate_absrel": candidate_absrel,
                "reference": reference,
                "candidate": candidate,
            }
        )
    gains = [float(item["gain"]) for item in candidates]
    median_gain = float(statistics.median(gains))
    median_choice = min(
        candidates,
        key=lambda item: (abs(float(item["gain"]) - median_gain), item["station_id"]),
    )
    maximum_choice = min(candidates, key=lambda item: (-float(item["gain"]), item["station_id"]))
    minimum_choice = min(candidates, key=lambda item: (float(item["gain"]), item["station_id"]))
    choices = (median_choice, maximum_choice, minimum_choice)
    if len({item["station_id"] for item in choices}) != 3:
        raise RuntimeError(
            "Preregistered median/maximum/minimum rules did not yield three stations"
        )

    common = {
        "selection_population": (
            f"all {expected_station_count} formal val shared stations with paired "
            f"{SOURCE_REGULAR}/{SOURCE_JOINT14} rows"
        ),
        "support_scope": SUPPORT_SCOPE,
        "fusion_method": fstar,
        "depth_method": RAW_METHOD,
        "gain_definition": (
            "regular_only spherical_abs_rel minus regular_plus_tangent14 spherical_abs_rel; "
            "positive is better"
        ),
        "population_median_gain": median_gain,
        "population_minimum_gain": float(min(gains)),
        "population_maximum_gain": float(max(gains)),
        "tie_break": "station_id ascending",
        "image_or_gt_opened_during_selection": False,
    }
    roles = (
        (
            "A_median_gain",
            {
                **common,
                "score": "absolute distance to the 30-station median gain",
                "choice": "minimum score",
            },
        ),
        (
            "B_maximum_gain",
            {**common, "score": "gain", "choice": "maximum gain"},
        ),
        (
            "C_minimum_gain_hardest",
            {
                **common,
                "score": "gain",
                "choice": "minimum gain, reported honestly even when positive",
            },
        ),
    )
    outputs = []
    for (role, rule), choice in zip(roles, choices):
        outputs.append(
            Selection(
                role=role,
                station_id=str(choice["station_id"]),
                room=str(choice["room"]),
                gain_abs_rel=float(choice["gain"]),
                reference_abs_rel=float(choice["reference_absrel"]),
                candidate_abs_rel=float(choice["candidate_absrel"]),
                rule=rule,
                reference_csv_row=dict(choice["reference"]),
                candidate_csv_row=dict(choice["candidate"]),
            )
        )
    return tuple(outputs)  # type: ignore[return-value]


def _resolve_recorded_artifact(
    override: Path | None,
    recorded: Mapping[str, Any],
    context: str,
) -> tuple[Path, dict[str, Any]]:
    recorded_path = Path(str(recorded.get("path", ""))).expanduser()
    path = override.expanduser() if override is not None else recorded_path
    identity = _require_identity(path, str(recorded.get("sha256", "")), context)
    return Path(identity["path"]), {
        **identity,
        "relocated_override": override is not None,
        "archival": {"formal_recorded_path": str(recorded_path)},
    }


def _evaluator_argv(
    *,
    config: Path,
    tangent_manifest: Path,
    provenance: Mapping[str, Any],
    device: str,
    batch_size: int,
) -> list[str]:
    geometry = _mapping(provenance.get("geometry"), "provenance.geometry")
    fusion = _mapping(provenance.get("fusion"), "provenance.fusion")
    synchronization = _mapping(
        fusion.get("overlap_scale_synchronization"), "fusion.overlap_scale_synchronization"
    )
    station_scale = _mapping(fusion.get("station_bim_scale"), "fusion.station_bim_scale")
    pano_shape = geometry.get("pano_shape")
    if pano_shape != [HEIGHT, WIDTH]:
        raise RuntimeError(f"Formal panorama shape differs from {[HEIGHT, WIDTH]}: {pano_shape}")
    return [
        "--config",
        str(config),
        "--tangent-manifest",
        str(tangent_manifest),
        "--split",
        FORMAL_SPLIT,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--pano-height",
        str(HEIGHT),
        "--seed",
        str(int(provenance["seed"])),
        "--centrality-power",
        str(float(geometry["centrality_power"])),
        "--confidence-floor",
        str(float(geometry["confidence_floor"])),
        "--huber-log-delta",
        str(float(fusion["huber_log_delta"])),
        "--consistency-log-threshold",
        str(float(fusion["consistency_log_threshold"])),
        "--photo-sigma",
        str(float(fusion["photometric_sigma"])),
        "--sync-min-overlap",
        str(int(synchronization["min_overlap"])),
        "--sync-pair-max-samples",
        str(int(synchronization["pair_max_deterministic_samples"])),
        "--sync-huber-log-delta",
        str(float(synchronization["huber_log_delta"])),
        "--sync-l2",
        str(float(synchronization["l2"])),
        "--sync-max-abs-offset",
        str(float(synchronization["max_abs_log_offset"])),
        "--station-scale-samples-per-view",
        str(int(station_scale["samples_per_view_cap"])),
    ]


def prepare_runtime(
    artifacts: FormalArtifacts,
    selections: Sequence[Selection],
    args: argparse.Namespace,
) -> RuntimeContext:
    """Initialize val runtime once; this is called only after selection is frozen."""

    provenance = artifacts.provenance
    config_recorded = _mapping(provenance.get("config"), "provenance.config")
    checkpoint_recorded = _mapping(provenance.get("checkpoint"), "provenance.checkpoint")
    tangent_recorded = _mapping(
        _mapping(provenance.get("route_p_tangent_inputs"), "route_p_tangent_inputs").get(
            "manifest"
        ),
        "route_p_tangent_inputs.manifest",
    )
    config_path, config_identity = _resolve_recorded_artifact(
        args.config, config_recorded, "formal config"
    )
    tangent_path, tangent_identity = _resolve_recorded_artifact(
        args.tangent_manifest, tangent_recorded, "formal tangent manifest"
    )
    evaluator_recorded = _mapping(provenance.get("evaluator"), "provenance.evaluator")
    evaluator_identity = _require_identity(
        Path(evaluator.__file__).resolve(),
        str(evaluator_recorded.get("sha256", "")),
        "frozen evaluator",
    )
    geometry_recorded = _mapping(
        provenance.get("shared_geometry_code_identity"),
        "provenance.shared_geometry_code_identity",
    )
    geometry_runtime_paths = {
        "stanford_pano": PROJECT_ROOT / "src/bim_priorda3/data/stanford_pano.py",
        "pano_tangent": PROJECT_ROOT / "src/bim_priorda3/data/pano_tangent.py",
    }
    geometry_identities = {}
    for name, runtime_path in geometry_runtime_paths.items():
        recorded = _mapping(geometry_recorded.get(name), f"shared geometry {name}")
        identity = _require_identity(
            runtime_path,
            str(recorded.get("sha256", "")),
            f"frozen shared geometry {name}",
        )
        geometry_identities[name] = {
            **identity,
            "archival": {"formal_recorded_path": str(recorded.get("path", ""))},
        }
    eval_args = evaluator.parse_args(
        _evaluator_argv(
            config=config_path,
            tangent_manifest=tangent_path,
            provenance=provenance,
            device=str(args.device),
            batch_size=int(args.batch_size),
        )
    )
    evaluator.seed_everything(int(eval_args.seed))
    cfg = evaluator.load_config(eval_args.config)
    evaluator.validate_universal_scale_protocol(cfg)
    if (float(cfg.data.min_depth), float(cfg.data.max_depth)) != (
        DEPTH_MIN_M,
        DEPTH_MAX_M,
    ):
        raise RuntimeError("Qualitative display contract requires the formal 0.2--5.0 m range")
    dataset = evaluator.BIMDepthDataset(
        cfg,
        FORMAL_SPLIT,
        augment=False,
        require_ground_truth=False,
    )
    device = evaluator._device_from_arg(str(eval_args.device))
    # The qualitative protocol is deliberately training-free.  The formal
    # summary may retain archival learned rows, but no checkpoint is opened and
    # the evaluator receives model=None.
    model = None

    area_root = evaluator.resolve_project_path(cfg, cfg.data.stanford_area_root)
    all_stations = evaluator.discover_stanford_panoramas(area_root)
    records_mutable: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in dataset.records:
        records_mutable[str(record["camera_uuid"])].append(record)
    selected_rooms = {str(record["region"]) for record in dataset.records}
    split_panos = [station for station in all_stations if station.room in selected_rooms]
    tangent_bundle = evaluator._validate_tangent_manifest(
        tangent_path,
        cfg=cfg,
        dataset=dataset,
        split=FORMAL_SPLIT,
        confirm_test=False,
        split_panoramas=split_panos,
    )
    stations = {station.camera_uuid: station for station in split_panos}
    for selection in selections:
        if selection.station_id not in stations or selection.station_id not in records_mutable:
            raise RuntimeError(
                f"Selected station is not a shared val station: {selection.station_id}"
            )
        if stations[selection.station_id].room != selection.room:
            raise RuntimeError(f"{selection.station_id}: selected room differs from val metadata")
    records = {station_id: tuple(values) for station_id, values in records_mutable.items()}
    return RuntimeContext(
        cfg=cfg,
        dataset=dataset,
        device=device,
        model=model,
        evaluator_args=eval_args,
        tangent_bundle=tangent_bundle,
        stations=stations,
        records_by_station=records,
        runtime_identity={
            "config": config_identity,
            "checkpoint": {
                "formal_recorded_sha256": str(checkpoint_recorded.get("sha256", "")),
                "runtime_status": "not_loaded_or_hashed",
                "role": "archival formal-result provenance only",
                "archival": {"formal_recorded_path": str(checkpoint_recorded.get("path", ""))},
            },
            "tangent_manifest": tangent_identity,
            "evaluator": evaluator_identity,
            "shared_geometry_code_identity": geometry_identities,
            "checkpoint_load_count": 0,
        },
    )


def _find_unique_row(
    rows: Sequence[Mapping[str, Any]],
    selectors: Mapping[str, str],
    context: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in selectors.items())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"{context}: expected one row for {dict(selectors)}, got {len(matches)}")
    return matches[0]


def verify_metric_row(
    actual: Mapping[str, Any],
    formal: Mapping[str, Any],
    *,
    context: str,
    atol: float,
) -> dict[str, float | int]:
    """Fail if recomputed station metrics do not reproduce the formal CSV row."""

    integer_keys = ("fixed_support_pixels", "gt_valid_pixels_at_eval_resolution", "count")
    float_keys = (
        "abs_rel",
        "mae",
        "rmse",
        "delta1",
        "delta2",
        "delta3",
        "spherical_abs_rel",
        "spherical_mae",
        "spherical_rmse",
        "spherical_delta1",
        "spherical_delta2",
        "spherical_delta3",
    )
    metadata_float_keys = (
        "coverage_fraction",
        "regular_coverage_fraction",
        "solid_angle_coverage_fraction",
        "multi_view_overlap_fraction",
        "mean_contributors_on_support",
    )
    receipt: dict[str, float | int] = {}
    for key in integer_keys:
        actual_value = int(actual[key])
        formal_value = int(float(formal[key]))
        if actual_value != formal_value:
            raise RuntimeError(
                f"{context}: {key} does not reproduce formal CSV ({actual_value} != {formal_value})"
            )
        receipt[key] = actual_value
    for key in float_keys:
        actual_value = _required_float(actual, key, context)
        formal_value = _required_float(formal, key, context)
        if not math.isclose(actual_value, formal_value, rel_tol=1e-7, abs_tol=atol):
            raise RuntimeError(
                f"{context}: {key} does not reproduce formal CSV "
                f"({actual_value:.12g} != {formal_value:.12g}, atol={atol})"
            )
        receipt[key] = actual_value
    for key in metadata_float_keys:
        if key not in actual and key not in formal:
            continue
        if key not in actual or key not in formal:
            raise RuntimeError(f"{context}: formal/runtime metadata field {key!r} differs")
        actual_value = _required_float(actual, key, context)
        formal_value = _required_float(formal, key, context)
        if not math.isclose(actual_value, formal_value, rel_tol=1e-7, abs_tol=atol):
            raise RuntimeError(
                f"{context}: {key} does not reproduce formal CSV "
                f"({actual_value:.12g} != {formal_value:.12g}, atol={atol})"
            )
        receipt[key] = actual_value
    return receipt


def _masked_absrel(
    prediction: np.ndarray,
    gt: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    output = np.full(gt.shape, np.nan, dtype=np.float32)
    output[support] = np.abs(prediction[support] - gt[support]) / gt[support]
    return output


def recompute_station(
    context: RuntimeContext,
    artifacts: FormalArtifacts,
    selection: Selection,
    *,
    metric_atol: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    """Recompute one selected station through the frozen evaluator high-level path."""

    station = context.stations[selection.station_id]
    tangent = evaluator._prepare_tangent_predictions(
        context.tangent_bundle,
        station,
        context.evaluator_args,
    )
    (
        route_r_rows,
        route_r_arrays,
        _strict_rows,
        _strict_arrays,
        station_info,
        route_outputs,
    ) = evaluator._evaluate_station(
        station,
        context.records_by_station[selection.station_id],
        context.cfg,
        context.model,
        context.device,
        context.evaluator_args,
        tangent,
    )
    combined_rows = route_outputs["combined_rows"]
    combined_arrays = route_outputs["combined_arrays"]
    fstar = artifacts.fstar
    regular_key = (SUPPORT_SCOPE, SOURCE_REGULAR, fstar)
    joint_key = (SUPPORT_SCOPE, SOURCE_JOINT14, fstar)
    regular_prediction, gt, support = combined_arrays[regular_key]
    joint_prediction, joint_gt, joint_support = combined_arrays[joint_key]
    if not np.array_equal(gt, joint_gt) or not np.array_equal(support, joint_support):
        raise RuntimeError(f"{selection.station_id}: combined comparison support/GT differs")

    predictions: dict[str, np.ndarray] = {
        "regular_only_raw": regular_prediction.copy(),
        "regular_plus_tangent14_raw": joint_prediction.copy(),
    }
    for method in EXPORTED_ROUTE_R_METHODS:
        prediction, method_gt, method_support = route_r_arrays[(fstar, method)]
        if not np.array_equal(gt, method_gt) or not np.array_equal(support, method_support):
            raise RuntimeError(
                f"{selection.station_id}: Route-R {method} differs from common regular support"
            )
        predictions[method] = prediction.copy()

    actual_regular = _find_unique_row(
        combined_rows,
        {
            "source_set": SOURCE_REGULAR,
            "support_scope": SUPPORT_SCOPE,
            "fusion_method": fstar,
            "depth_method": RAW_METHOD,
        },
        selection.station_id,
    )
    actual_joint = _find_unique_row(
        combined_rows,
        {
            "source_set": SOURCE_JOINT14,
            "support_scope": SUPPORT_SCOPE,
            "fusion_method": fstar,
            "depth_method": RAW_METHOD,
        },
        selection.station_id,
    )
    formal_route_rows = [
        row for row in artifacts.regular_rows if row.get("station_id") == selection.station_id
    ]
    formal_regular = _find_unique_row(
        formal_route_rows,
        {
            "source_set": SOURCE_REGULAR,
            "support_scope": SUPPORT_SCOPE,
            "fusion_method": fstar,
            "depth_method": RAW_METHOD,
        },
        f"formal/{selection.station_id}",
    )
    formal_joint = _find_unique_row(
        formal_route_rows,
        {
            "source_set": SOURCE_JOINT14,
            "support_scope": SUPPORT_SCOPE,
            "fusion_method": fstar,
            "depth_method": RAW_METHOD,
        },
        f"formal/{selection.station_id}",
    )
    reproductions = {
        "regular_only_raw": verify_metric_row(
            actual_regular,
            formal_regular,
            context=f"{selection.station_id}/regular_only_raw",
            atol=metric_atol,
        ),
        "regular_plus_tangent14_raw": verify_metric_row(
            actual_joint,
            formal_joint,
            context=f"{selection.station_id}/regular_plus_tangent14_raw",
            atol=metric_atol,
        ),
    }
    formal_route_r_rows = [
        row for row in artifacts.route_r_rows if row.get("station_id") == selection.station_id
    ]
    for method in EXPORTED_ROUTE_R_METHODS:
        actual = _find_unique_row(
            route_r_rows,
            {"fusion_method": fstar, "depth_method": method},
            selection.station_id,
        )
        formal = _find_unique_row(
            formal_route_r_rows,
            {"fusion_method": fstar, "depth_method": method},
            f"formal/{selection.station_id}",
        )
        reproductions[method] = verify_metric_row(
            actual,
            formal,
            context=f"{selection.station_id}/{method}",
            atol=metric_atol,
        )

    instant_gain = float(reproductions["regular_only_raw"]["spherical_abs_rel"]) - float(
        reproductions["regular_plus_tangent14_raw"]["spherical_abs_rel"]
    )
    if not math.isclose(instant_gain, selection.gain_abs_rel, rel_tol=1e-7, abs_tol=metric_atol):
        raise RuntimeError(
            f"{selection.station_id}: recomputed gain {instant_gain} differs from selected "
            f"formal gain {selection.gain_abs_rel}"
        )

    gt_valid = np.isfinite(gt) & (gt >= DEPTH_MIN_M) & (gt <= DEPTH_MAX_M)
    regular_coverage = np.isfinite(regular_prediction) & (regular_prediction > 0)
    joint_coverage = np.isfinite(joint_prediction) & (joint_prediction > 0)
    if np.any(regular_coverage & ~joint_coverage):
        raise RuntimeError(f"{selection.station_id}: tangent14 union lost regular coverage")
    coverage = np.full(gt.shape, -1, dtype=np.int8)
    coverage[gt_valid] = 0
    coverage[gt_valid & joint_coverage] = 1
    coverage[gt_valid & regular_coverage] = 2
    pano_rgb = evaluator._load_pano_rgb(station.rgb_path, (HEIGHT, WIDTH))
    errors = {
        f"{name}_absrel": _masked_absrel(prediction, gt, support)
        for name, prediction in predictions.items()
    }
    arrays: dict[str, np.ndarray] = {
        "pano_rgb": pano_rgb.astype(np.float32),
        "gt_range": gt.copy().astype(np.float32),
        "gt_valid": gt_valid.astype(np.uint8),
        **{key: value.astype(np.float32) for key, value in predictions.items()},
        **errors,
        "joint_minus_regular_signed_absrel": (
            errors["regular_plus_tangent14_raw_absrel"] - errors["regular_only_raw_absrel"]
        ).astype(np.float32),
        "regular_coverage": regular_coverage.astype(np.uint8),
        "regular_plus_tangent14_coverage": joint_coverage.astype(np.uint8),
        "coverage": coverage,
        "support": support.astype(np.uint8),
    }
    metrics = {
        "schema_version": 1,
        "station_id": selection.station_id,
        "room": selection.room,
        "fusion_method": fstar,
        "support_scope": SUPPORT_SCOPE,
        "fixed_support_pixels": int(np.count_nonzero(support)),
        "gt_valid_pixels": int(np.count_nonzero(gt_valid)),
        "fixed_support_fraction": float(np.mean(support[gt_valid])),
        "formal_selected_gain_abs_rel": selection.gain_abs_rel,
        "recomputed_gain_abs_rel": instant_gain,
        "gain_definition": selection.rule["gain_definition"],
        "signed_map_definition": (
            "regular_plus_tangent14 per-pixel AbsRel minus regular_only per-pixel AbsRel; "
            "negative values indicate improvement"
        ),
        "formal_csv_reproduction_atol": metric_atol,
        "formal_csv_reproduced": True,
        "methods": {
            name: reproductions[name]
            for name in (
                "regular_only_raw",
                "regular_plus_tangent14_raw",
                *EXPORTED_ROUTE_R_METHODS,
            )
        },
        "training_free_contract": {
            "checkpoint_loaded": False,
            "learned_refined_computed_or_reproduced": False,
            "formal_source_summary_contains_archival_learned_rows": (
                "learned_refined" in artifacts.summary.get("depth_methods", [])
            ),
        },
        "coverage": {
            "regular_gt_valid_fraction": float(np.mean(regular_coverage[gt_valid])),
            "regular_plus_tangent14_gt_valid_fraction": float(np.mean(joint_coverage[gt_valid])),
        },
    }
    area_root = evaluator.resolve_project_path(context.cfg, context.cfg.data.stanford_area_root)
    source_identity = {
        "pano_rgb": _area_root_relative_identity(
            station.rgb_path,
            str(
                _mapping(
                    artifacts.provenance["selected_pano_files"][f"{selection.station_id}/rgb"],
                    "selected pano RGB",
                )["sha256"]
            ),
            area_root=area_root,
            context=f"{selection.station_id} pano RGB",
        ),
        "pano_depth_evaluation_target": _area_root_relative_identity(
            station.depth_path,
            str(
                _mapping(
                    artifacts.provenance["selected_pano_files"][f"{selection.station_id}/depth"],
                    "selected pano depth",
                )["sha256"]
            ),
            area_root=area_root,
            context=f"{selection.station_id} pano depth",
        ),
        "regular_view_count": int(station_info["regular_view_count"]),
    }
    return arrays, metrics, source_identity


def _colorize(
    values: np.ndarray,
    *,
    cmap_name: str,
    norm: Normalize,
    valid: np.ndarray,
) -> np.ndarray:
    if values.shape != (HEIGHT, WIDTH) or valid.shape != values.shape:
        raise ValueError("Scalar panorama panels must be 512x1024")
    safe = np.nan_to_num(values, nan=0.0, posinf=norm.vmax, neginf=norm.vmin)
    rgba = matplotlib.colormaps[cmap_name](norm(safe), bytes=True)
    rgb = rgba[..., :3].copy()
    rgb[~valid] = INVALID_RGB
    return rgb.astype(np.uint8)


def panel_images(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Create all title-free fixed-contract panel RGB arrays."""

    support = arrays["support"].astype(bool)
    gt_valid = arrays["gt_valid"].astype(bool)
    depth_norm = Normalize(vmin=DEPTH_MIN_M, vmax=DEPTH_MAX_M, clip=True)
    absrel_norm = Normalize(vmin=0.0, vmax=ABSREL_MAX, clip=True)
    signed_norm = TwoSlopeNorm(vmin=-SIGNED_LIMIT, vcenter=0.0, vmax=SIGNED_LIMIT)
    outputs: dict[str, np.ndarray] = {
        "pano_rgb": np.clip(arrays["pano_rgb"] * 255.0, 0, 255).round().astype(np.uint8),
        "gt_range": _colorize(
            arrays["gt_range"], cmap_name="turbo", norm=depth_norm, valid=gt_valid
        ),
    }
    depth_names = (
        "regular_only_raw",
        "regular_plus_tangent14_raw",
        "universal_scale",
        "bim_direct",
    )
    for name in depth_names:
        outputs[name] = _colorize(
            arrays[name],
            cmap_name="turbo",
            norm=depth_norm,
            valid=np.isfinite(arrays[name]) & (arrays[name] > 0),
        )
        outputs[f"{name}_absrel"] = _colorize(
            arrays[f"{name}_absrel"],
            cmap_name="magma",
            norm=absrel_norm,
            valid=support,
        )
    outputs["joint_minus_regular_signed_absrel"] = _colorize(
        arrays["joint_minus_regular_signed_absrel"],
        cmap_name="coolwarm",
        norm=signed_norm,
        valid=support,
    )

    coverage_values = arrays["coverage"]
    coverage_rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    coverage_rgb[coverage_values == -1] = INVALID_RGB
    coverage_rgb[coverage_values == 0] = (0, 0, 0)
    coverage_rgb[coverage_values == 1] = (50, 180, 210)  # tangent14-only union support
    coverage_rgb[coverage_values == 2] = (80, 190, 100)  # common regular support
    outputs["coverage"] = coverage_rgb
    support_rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    support_rgb[~gt_valid] = INVALID_RGB
    support_rgb[support] = (245, 245, 245)
    outputs["support"] = support_rgb
    if set(outputs) != set(PANEL_NAMES):
        raise AssertionError(f"Panel contract differs: {sorted(set(PANEL_NAMES) ^ set(outputs))}")
    for name, image in outputs.items():
        if image.shape != (HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
            raise AssertionError(f"{name}: invalid rendered panel {image.shape}/{image.dtype}")
    return outputs


def save_title_free_png(path: Path, rgb: np.ndarray) -> None:
    if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
        raise ValueError("Title-free panel must be uint8 RGB with shape 512x1024")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(path),
        rgb[..., ::-1],
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    ):
        raise RuntimeError(f"Could not write PNG: {path}")


def write_shared_colorbars(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Write reusable depth/error/signed colorbars in PNG, SVG, and PDF."""

    specs: dict[str, tuple[str, Normalize, str]] = {
        "depth": (
            "turbo",
            Normalize(DEPTH_MIN_M, DEPTH_MAX_M),
            "Radial depth / range (m)",
        ),
        "absrel": ("magma", Normalize(0.0, ABSREL_MAX), "Per-pixel AbsRel"),
        "signed_absrel": (
            "coolwarm",
            TwoSlopeNorm(vmin=-SIGNED_LIMIT, vcenter=0.0, vmax=SIGNED_LIMIT),
            "Joint minus regular per-pixel AbsRel (negative is better)",
        ),
    }
    colorbar_dir = output_dir / "colorbars"
    colorbar_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, Any]] = {}
    for name, (cmap_name, norm, label) in specs.items():
        fig = plt.figure(figsize=(6.4, 1.2))
        axis = fig.add_axes((0.08, 0.52, 0.84, 0.24))
        colorbar = ColorbarBase(
            axis,
            cmap=matplotlib.colormaps[cmap_name],
            norm=norm,
            orientation="horizontal",
        )
        colorbar.set_label(label, fontsize=9)
        paths = {}
        for suffix in ("png", "svg", "pdf"):
            path = colorbar_dir / f"{name}.{suffix}"
            fig.savefig(path, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.05)
            paths[suffix] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        plt.close(fig)
        outputs[name] = {
            "cmap": cmap_name,
            "vmin": float(norm.vmin),
            "vmax": float(norm.vmax),
            "vcenter": 0.0 if isinstance(norm, TwoSlopeNorm) else None,
            "label": label,
            "artifacts": paths,
        }
    return outputs


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )


def _shared_manifest_payload(
    artifacts: FormalArtifacts,
    runtime: RuntimeContext,
) -> dict[str, Any]:
    provenance = artifacts.provenance
    return {
        "source_split": FORMAL_SPLIT,
        "test_data_read": False,
        "formal_schema_version": FORMAL_SCHEMA_VERSION,
        "formal_protocol": FORMAL_PROTOCOL,
        "formal_artifacts": dict(artifacts.identities),
        "config": runtime.runtime_identity["config"],
        "checkpoint": runtime.runtime_identity["checkpoint"],
        "evaluator": runtime.runtime_identity["evaluator"],
        "shared_geometry_code_identity": runtime.runtime_identity["shared_geometry_code_identity"],
        "tangent_manifest": runtime.runtime_identity["tangent_manifest"],
        "split_annotation": provenance.get("split_annotation"),
        "prepared_manifest": provenance.get("prepared_manifest"),
        "fstar_selection": artifacts.fstar_selection,
        "calculation_reuse": {
            "module": "scripts.model.evaluate_stanford_pano",
            "station_entrypoint": "_evaluate_station",
            "tangent_entrypoint": "_prepare_tangent_predictions",
            "geometry_or_fusion_reimplemented": False,
            "checkpoint_load_count": runtime.runtime_identity["checkpoint_load_count"],
            "learned_model_forward_count": 0,
        },
        "figure_method_scope": {
            "focus": "training-free panorama joint estimation and deterministic BIM enhancement",
            "exported_depth_methods": [
                "regular_only_raw",
                "regular_plus_tangent14_raw",
                *EXPORTED_ROUTE_R_METHODS,
            ],
            "learned_refined_exported": False,
            "checkpoint_role": "not loaded; archival formal-result identity only",
            "formal_archival_learned_rows_recomputed": False,
        },
        "display_contract": {
            "panel_shape": [HEIGHT, WIDTH],
            "aspect_ratio": "2:1 equirectangular",
            "title_free_individual_png": True,
            "depth": {"range_m": [DEPTH_MIN_M, DEPTH_MAX_M], "cmap": "turbo"},
            "absrel": {"range": [0.0, ABSREL_MAX], "cmap": "magma"},
            "signed_absrel": {
                "range": [-SIGNED_LIMIT, SIGNED_LIMIT],
                "cmap": "coolwarm",
            },
            "invalid_rgb_uint8": INVALID_RGB.tolist(),
        },
    }


def export_station_assets(
    *,
    output_dir: Path,
    selection: Selection,
    arrays: Mapping[str, np.ndarray],
    metrics: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    shared_provenance: Mapping[str, Any],
    colorbars: Mapping[str, Any],
) -> dict[str, Any]:
    station_dir = output_dir / f"{selection.role}__{_slug(selection.room)}__{selection.station_id}"
    station_dir.mkdir(parents=True, exist_ok=True)
    # Remove only the two explicitly retired learned assets when refreshing an
    # output produced by the earlier mixed learned/training-free figure scope.
    for retired_name in RETIRED_PANEL_NAMES:
        retired_path = station_dir / f"{retired_name}.png"
        if retired_path.is_file():
            retired_path.unlink()
    retired_arrays_path = station_dir / "arrays.npz"
    if retired_arrays_path.is_file():
        retired_arrays_path.unlink()
    rendered = panel_images(arrays)
    panel_artifacts = {}
    for name in PANEL_NAMES:
        path = station_dir / f"{name}.png"
        save_title_free_png(path, rendered[name])
        panel_artifacts[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}

    metrics_path = station_dir / "metrics.json"
    write_json(metrics_path, dict(metrics))
    manifest = {
        "schema_version": 1,
        "purpose": "Validation-only title-free panorama material for paper/PPT assembly",
        "selection": {
            "role": selection.role,
            "station_id": selection.station_id,
            "room": selection.room,
            "gain_abs_rel": selection.gain_abs_rel,
            "reference_abs_rel": selection.reference_abs_rel,
            "candidate_abs_rel": selection.candidate_abs_rel,
            "rule": dict(selection.rule),
            "formal_reference_csv_row": dict(selection.reference_csv_row),
            "formal_candidate_csv_row": dict(selection.candidate_csv_row),
        },
        "source_identity": dict(source_identity),
        "shared_provenance": dict(shared_provenance),
        "signed_map_definition": metrics["signed_map_definition"],
        "coverage_legend": {
            "-1": "invalid/out-of-range official pano GT (dark gray)",
            "0": "valid GT but no regular+tangent14 prediction (black)",
            "1": "regular+tangent14 coverage outside regular-only coverage (cyan)",
            "2": "common regular support (green)",
        },
        "shared_colorbars": dict(colorbars),
        "artifacts": {
            "panels": panel_artifacts,
            "metrics": {
                "path": str(metrics_path.resolve()),
                "sha256": file_sha256(metrics_path),
            },
        },
        "array_export": {
            "status": "not_written",
            "reason": "paper/PPT release contains rendered assets only",
        },
    }
    manifest_path = station_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "role": selection.role,
        "station_id": selection.station_id,
        "room": selection.room,
        "directory": str(station_dir.resolve()),
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
        },
        "panels": panel_artifacts,
    }


def write_preview_sheet(exports: Sequence[Mapping[str, Any]], output: Path) -> None:
    names = (
        "pano_rgb",
        "gt_range",
        "regular_only_raw",
        "regular_plus_tangent14_raw",
        "universal_scale",
        "bim_direct",
        "joint_minus_regular_signed_absrel",
    )
    fig, axes = plt.subplots(len(exports), len(names), figsize=(18, 3.4 * len(exports)))
    axes_array = np.atleast_2d(axes)
    for row_index, export in enumerate(exports):
        for column_index, name in enumerate(names):
            path = Path(export["panels"][name]["path"])
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Cannot build preview from {path}")
            axis = axes_array[row_index, column_index]
            axis.imshow(bgr[..., ::-1])
            axis.axis("off")
            if row_index == 0:
                axis.set_title(name.replace("_", " "), fontsize=9)
            if column_index == 0:
                axis.text(
                    -0.035,
                    0.5,
                    f"{export['role']}\n{export['room']}\n{export['station_id'][:8]}",
                    transform=axis.transAxes,
                    horizontalalignment="right",
                    verticalalignment="center",
                    fontsize=9,
                    clip_on=False,
                )
    fig.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = load_formal_val_artifacts(args.results_dir)
    selections = select_three_stations(
        artifacts.regular_rows,
        fstar=artifacts.fstar,
        expected_station_count=int(artifacts.summary["station_count"]),
    )
    print(
        "Frozen validation selections (no image/depth opened): "
        + ", ".join(
            f"{item.role}={item.room}/{item.station_id} gain={item.gain_abs_rel:.6f}"
            for item in selections
        ),
        flush=True,
    )

    # Dataset, caches, RGB, and GT are intentionally touched only now; the
    # archival learned checkpoint is never opened by this training-free exporter.
    runtime = prepare_runtime(artifacts, selections, args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    colorbars = write_shared_colorbars(output_dir)
    shared_provenance = _shared_manifest_payload(artifacts, runtime)
    exports = []
    for index, selection in enumerate(selections, start=1):
        print(
            f"[{index}/{len(selections)}] recompute {selection.room}/{selection.station_id}",
            flush=True,
        )
        arrays, metrics, source_identity = recompute_station(
            runtime,
            artifacts,
            selection,
            metric_atol=float(args.metric_atol),
        )
        exports.append(
            export_station_assets(
                output_dir=output_dir,
                selection=selection,
                arrays=arrays,
                metrics=metrics,
                source_identity=source_identity,
                shared_provenance=shared_provenance,
                colorbars=colorbars,
            )
        )
    preview_identity = None
    if args.preview_sheet:
        preview_path = output_dir / "preview_sheet.png"
        write_preview_sheet(exports, preview_path)
        preview_identity = {"path": str(preview_path), "sha256": file_sha256(preview_path)}

    exporter_path = Path(__file__).resolve()
    root_manifest = {
        "schema_version": 1,
        "purpose": "Three preregistered formal-val panorama qualitative alternatives",
        "shared_provenance": shared_provenance,
        "selection_order": [selection.role for selection in selections],
        "selections": [
            {
                "role": selection.role,
                "station_id": selection.station_id,
                "room": selection.room,
                "gain_abs_rel": selection.gain_abs_rel,
                "rule": dict(selection.rule),
            }
            for selection in selections
        ],
        "exporter": {"path": str(exporter_path), "sha256": file_sha256(exporter_path)},
        "shared_colorbars": colorbars,
        "station_exports": exports,
        "preview_sheet": preview_identity,
    }
    write_json(output_dir / "manifest.json", root_manifest)
    print(f"Wrote {len(exports)} station alternatives to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
