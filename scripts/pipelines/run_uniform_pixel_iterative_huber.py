#!/usr/bin/env python3
"""Run the pixel-uniform iterative-Huber scale/refiner experiment."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCALE_CONFIG = Path(
    "configs/stanford_area1_iterative_attention_huber_no_da3_features_"
    "no_confidence_no_bim_geometry_uniform_pixels_3round_3epoch_"
    "full_depth_metric_da3.yaml"
)
CONTINUATION_CONFIG = Path(
    "configs/stanford_area1_iterative_attention_huber_reduced_refiner_"
    "uniform_pixels_continuation_full_depth_metric_da3.yaml"
)
SCALE_OUTPUT = Path(
    "outputs/stanford_area1_iterative_attention_huber_uniform_pixels_3round_3epoch"
)
CONTINUATION_OUTPUT = Path(
    "outputs/stanford_area1_iterative_attention_huber_reduced_refiner_"
    "uniform_pixels_continuation"
)
TEST_OUTPUT = Path(
    "results/stanford_area1/iterative_attention_huber_reduced_refiner_"
    "uniform_pixels_best_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("scale", "continuation", "test"),
        default=("scale", "continuation", "test"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wait-for-pid",
        type=int,
        help="Wait for an already-running scale trainer before selected stages",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return (PROJECT_ROOT / path).resolve()


def python_command(script: str, *arguments: object) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script), *(str(arg) for arg in arguments)]


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    args = parse_args()
    scale_checkpoint = absolute(SCALE_OUTPUT / "accepted.pt")
    final_checkpoint = absolute(CONTINUATION_OUTPUT / "best.pt")
    commands = {
        "scale": python_command(
            "scripts/model/train.py",
            "--config",
            absolute(SCALE_CONFIG),
            "--device",
            args.device,
        ),
        "continuation": python_command(
            "scripts/model/train.py",
            "--config",
            absolute(CONTINUATION_CONFIG),
            "--init-checkpoint",
            scale_checkpoint,
            "--device",
            args.device,
        ),
        "test": python_command(
            "scripts/model/evaluate_stanford_area1.py",
            "--config",
            absolute(CONTINUATION_CONFIG),
            "--checkpoint",
            final_checkpoint,
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
            absolute(TEST_OUTPUT),
        ),
    }

    for stage in args.stages:
        print(f"[{stage}] {shlex.join(commands[stage])}", flush=True)
    if not args.execute:
        print("Dry run only; pass --execute to run.", flush=True)
        return

    if args.wait_for_pid is not None:
        print(f"Waiting for PID {args.wait_for_pid} to finish...", flush=True)
        while process_exists(args.wait_for_pid):
            time.sleep(60)
        print(f"PID {args.wait_for_pid} finished.", flush=True)

    for stage in args.stages:
        if stage == "continuation" and not scale_checkpoint.is_file():
            raise FileNotFoundError(f"Successful scale checkpoint missing: {scale_checkpoint}")
        if stage == "test" and not final_checkpoint.is_file():
            raise FileNotFoundError(f"Validation-selected checkpoint missing: {final_checkpoint}")
        subprocess.run(commands[stage], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
