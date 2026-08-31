#!/usr/bin/env python3
"""Prepare or execute the joint scale-gradient-isolation experiment.

The default is a safe dry run: commands are printed but not executed. Pass
``--execute`` only after the experiment definition has been reviewed.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(
    "configs/stanford_area1_iterative_attention_huber_reduced_refiner_"
    "joint_scale_final_only_full_depth_metric_da3.yaml"
)
DEFAULT_INITIAL_CHECKPOINT = Path(
    "outputs/stanford_area1_iterative_attention_huber_no_da3_features_"
    "no_confidence_no_bim_geometry_3round_3epoch/accepted.pt"
)
DEFAULT_OUTPUT = Path(
    "outputs/stanford_area1_iterative_attention_huber_reduced_refiner_"
    "joint_scale_final_only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("train", "area-test", "zero-shot"),
        default=("train", "area-test", "zero-shot"),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--init-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_OUTPUT / "last.pt",
        help="Checkpoint for evaluation; defaults to the final joint-stage checkpoint",
    )
    parser.add_argument(
        "--area-output",
        type=Path,
        default=Path(
            "results/stanford_area1/iterative_attention_huber_reduced_refiner_"
            "joint_scale_final_only_last_test"
        ),
    )
    parser.add_argument("--matterport-root", type=Path, default=Path("/home/bgao491/Matterport3D"))
    parser.add_argument("--bimnet-root", type=Path, default=Path("/home/bgao491/BIMNet_release"))
    parser.add_argument(
        "--toolkit-root", type=Path, default=Path("/home/bgao491/S3-SAM3D-ToolKit")
    )
    parser.add_argument("--bimnet-scene", default="hxp")
    parser.add_argument(
        "--zero-shot-output",
        type=Path,
        default=Path(
            "results/matterport3d/hxp_iterative_attention_huber_refiner_"
            "joint_scale_final_only_zero_shot"
        ),
    )
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def command(script: str, *args: object) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script), *(str(value) for value in args)]


def main() -> None:
    args = parse_args()
    config = project_path(args.config)
    initial_checkpoint = project_path(args.init_checkpoint)
    checkpoint = project_path(args.checkpoint)
    commands: dict[str, list[str]] = {
        "train": command(
            "scripts/model/train.py",
            "--config",
            config,
            "--init-checkpoint",
            initial_checkpoint,
            "--device",
            args.device,
        ),
        "area-test": command(
            "scripts/model/evaluate_stanford_area1.py",
            "--config",
            config,
            "--checkpoint",
            checkpoint,
            "--split",
            "test",
            "--depth-support",
            "all-valid",
            "--batch-size",
            8,
            "--device",
            args.device,
            "--allow-unverified-robust-comparator",
            "--output",
            project_path(args.area_output),
        ),
        "zero-shot": command(
            "scripts/model/evaluate_matterport_bimnet_scale_refiner.py",
            "--matterport-root",
            project_path(args.matterport_root),
            "--bimnet-root",
            project_path(args.bimnet_root),
            "--toolkit-root",
            project_path(args.toolkit_root),
            "--bimnet-scene",
            args.bimnet_scene,
            "--config",
            config,
            "--checkpoint",
            checkpoint,
            "--output-dir",
            project_path(args.zero_shot_output),
            "--process-res",
            504,
            "--device",
            args.device,
            "--progress-every",
            100,
        ),
    }

    if args.execute:
        if not config.is_file():
            raise FileNotFoundError(config)
        if "train" in args.stages and not initial_checkpoint.is_file():
            raise FileNotFoundError(initial_checkpoint)

    for stage in args.stages:
        stage_command = commands[stage]
        print(f"[{stage}] {shlex.join(stage_command)}", flush=True)
        if not args.execute:
            continue
        if stage != "train" and not checkpoint.is_file():
            raise FileNotFoundError(
                f"{stage} requires {checkpoint}; run the train stage first or pass --checkpoint"
            )
        subprocess.run(stage_command, cwd=PROJECT_ROOT, check=True)

    if not args.execute:
        print("Dry run only. Add --execute after reviewing these commands.", flush=True)


if __name__ == "__main__":
    main()
