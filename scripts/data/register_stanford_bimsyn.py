#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

from bim_priorda3.data.stanford_registration import (
    RegistrationParameters,
    build_registration_audit,
    write_registration_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate one fixed, geometry-only BIMSyn-room-to-Stanford-Area transform "
            "per room. Registration is constrained to unit scale, Z-up yaw and XYZ "
            "translation; RGB/depth frames are never read."
        )
    )
    parser.add_argument(
        "--semantic-obj",
        type=Path,
        required=True,
        help="Area_1/3d/semantic.obj with per-face usemtl labels",
    )
    parser.add_argument(
        "--ifc-dir",
        type=Path,
        required=True,
        help="BIMSyn BIM_model/ifc directory (one <room>.ifc per room)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Audit JSON output")
    parser.add_argument(
        "--rooms",
        nargs="*",
        default=None,
        help="Optional exact room stems; default requires the union of OBJ and IFC rooms",
    )
    parser.add_argument("--expected-area", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--sample-points", type=int, default=20_000)
    parser.add_argument("--coarse-points", type=int, default=2_500)
    parser.add_argument("--yaw-starts", type=int, default=72)
    parser.add_argument("--refine-candidates", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=35)
    parser.add_argument(
        "--correspondence-distances-m",
        nargs="+",
        type=float,
        default=(1.5, 0.75, 0.35, 0.18),
        metavar="METRES",
        help="Strictly decreasing coarse-to-fine ICP correspondence thresholds",
    )
    parser.add_argument("--trim-fraction", type=float, default=0.85)
    parser.add_argument("--huber-delta-m", type=float, default=0.12)
    parser.add_argument("--metric-threshold-m", type=float, default=0.20)
    parser.add_argument("--min-fitness", type=float, default=0.55)
    parser.add_argument("--max-rmse-m", type=float, default=0.15)
    parser.add_argument("--min-points", type=int, default=100)
    parser.add_argument("--semantic-clip-distance-m", type=float, default=0.75)
    parser.add_argument("--semantic-trim-fraction", type=float, default=0.90)
    parser.add_argument("--semantic-min-points-per-class", type=int, default=24)
    parser.add_argument("--semantic-geometric-tolerance-m", type=float, default=0.03)
    parser.add_argument("--semantic-min-improvement-m", type=float, default=0.02)
    parser.add_argument("--semantic-discriminative-weight", type=float, default=3.0)
    parser.add_argument(
        "--semantic-min-yaw-separation-deg",
        type=float,
        default=30.0,
        help="Minimum yaw difference for a class-aware alternative candidate",
    )
    parser.add_argument(
        "--semantic-discriminative-classes",
        nargs="+",
        default=("door", "window", "beam", "column"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parameters = RegistrationParameters(
        seed=args.seed,
        sample_points=args.sample_points,
        coarse_points=args.coarse_points,
        yaw_starts=args.yaw_starts,
        refine_candidates=args.refine_candidates,
        max_iterations=args.max_iterations,
        correspondence_distances_m=tuple(args.correspondence_distances_m),
        trim_fraction=args.trim_fraction,
        huber_delta_m=args.huber_delta_m,
        metric_threshold_m=args.metric_threshold_m,
        min_fitness=args.min_fitness,
        max_rmse_m=args.max_rmse_m,
        min_points=args.min_points,
        semantic_clip_distance_m=args.semantic_clip_distance_m,
        semantic_trim_fraction=args.semantic_trim_fraction,
        semantic_min_points_per_class=args.semantic_min_points_per_class,
        semantic_geometric_tolerance_m=args.semantic_geometric_tolerance_m,
        semantic_min_improvement_m=args.semantic_min_improvement_m,
        semantic_discriminative_weight=args.semantic_discriminative_weight,
        semantic_min_yaw_separation_rad=math.radians(args.semantic_min_yaw_separation_deg),
        semantic_discriminative_classes=tuple(args.semantic_discriminative_classes),
    )
    payload = build_registration_audit(
        args.semantic_obj,
        args.ifc_dir,
        parameters=parameters,
        rooms=args.rooms,
        expected_area=args.expected_area,
    )
    output = write_registration_audit(payload, args.output, overwrite=args.overwrite)
    summary = payload["summary"]
    print(
        json.dumps(
            {
                "output": str(output),
                **summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if payload["failures"]:
        print(
            json.dumps({"failures": payload["failures"]}, indent=2, sort_keys=True),
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
