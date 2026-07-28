#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root


STAGES = (
    "download",
    "poses",
    "verify",
    "prepare",
    "audit",
    "anchors",
    "train-v1",
    "eval-v1",
    "cache-candidates",
    "train-v3",
    "eval-v3",
    "reconstruct",
    "report",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable SLABIM -> poses -> GT -> training -> 2D/3D evaluation pipeline"
    )
    parser.add_argument("--v1-config", type=Path, default=Path("configs/slabim_single_frame_r50.yaml"))
    parser.add_argument("--v3-config", type=Path, default=Path("configs/slabim_single_frame_r50_v3.yaml"))
    parser.add_argument("--slabim-root", type=Path)
    parser.add_argument("--regions", nargs="+")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all",) + STAGES,
        default=("verify",),
        help="Use --stages all for the complete experiment; default is a read-only check",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--train-val-stride",
        type=int,
        default=2,
        help="Historical fixed protocol: subsample train/validation RGB by 2",
    )
    parser.add_argument(
        "--test-stride",
        type=int,
        default=1,
        help="Historical fixed protocol: evaluate every test RGB frame",
    )
    parser.add_argument("--max-frames", type=int, help="Smoke-test preparation/reconstruction")
    parser.add_argument("--keep-rosbags", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--state", type=Path, default=Path("outputs/pipeline_state.json"))
    return parser.parse_args()


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project = Path(__file__).resolve().parents[1]
        self.v1_config = self._path(args.v1_config)
        self.v3_config = self._path(args.v3_config)
        self.v1_cfg = load_config(self.v1_config)
        self.v3_cfg = load_config(self.v3_config)
        self.slabim = (
            args.slabim_root.expanduser().resolve()
            if args.slabim_root
            else resolve_slabim_root(self.v1_cfg)
        )
        self.regions = args.regions or list(self.v1_cfg.data.regions)
        self.state_path = self._path(args.state)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = (
            json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state_path.exists()
            else {"stages": {}}
        )
        self.environment = os.environ.copy()
        self.environment["BIM_PRIORDA3_SLABIM_ROOT"] = str(self.slabim)

    def _path(self, path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (self.project / path).resolve()

    def command(self, script: str, *arguments: object) -> list[str]:
        command = [sys.executable, str(self.project / "scripts" / script)]
        command.extend(str(value) for value in arguments if value is not None)
        return command

    def run(self, stage: str, command: list[str], complete: Callable[[], bool] | None = None) -> None:
        if complete and complete() and not self.args.force:
            print(f"[skip] {stage}: expected outputs already exist", flush=True)
            if not self.args.dry_run:
                self._record(stage, "skipped", command)
            return
        print(f"[run] {stage}: {' '.join(command)}", flush=True)
        if self.args.dry_run:
            return
        started = datetime.now(timezone.utc).isoformat()
        try:
            subprocess.run(
                command,
                cwd=self.project,
                env=self.environment,
                check=True,
            )
        except subprocess.CalledProcessError:
            self._record(stage, "failed", command, started)
            raise
        self._record(stage, "completed", command, started)

    def _record(
        self,
        stage: str,
        status: str,
        command: list[str],
        started: str | None = None,
    ) -> None:
        self.state["slabim_root"] = str(self.slabim)
        self.state["regions"] = self.regions
        self.state["stages"][stage] = {
            "status": status,
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
        }
        self.state_path.write_text(
            json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def execute(self, stages: list[str]) -> None:
        v1_output = resolve_project_path(self.v1_cfg, self.v1_cfg.experiment.output_dir)
        v3_output = resolve_project_path(self.v3_cfg, self.v3_cfg.experiment.output_dir)
        processed = resolve_project_path(self.v1_cfg, self.v1_cfg.data.processed_root)
        region_args = ["--regions", *self.regions]
        if "download" in stages:
            pose_missing = [
                region
                for region in self.regions
                if not (
                    self.slabim
                    / "sensor_data"
                    / region
                    / "points/lidar_pose_local_to_bim_from_rosbag.txt"
                ).exists()
            ]
            pose_ready = [region for region in self.regions if region not in pose_missing]
            if pose_missing:
                self.run(
                    "download-with-rosbags",
                    self.command(
                        "download_slabim.py",
                        "--root",
                        self.slabim,
                        "--regions",
                        *pose_missing,
                        "--include-rosbag",
                    ),
                )
            if pose_ready:
                self.run(
                    "download-core",
                    self.command(
                        "download_slabim.py",
                        "--root",
                        self.slabim,
                        "--regions",
                        *pose_ready,
                    ),
                )
        if "poses" in stages:
            extra = [] if self.args.keep_rosbags else ["--delete-rosbags"]
            self.run(
                "poses",
                self.command(
                    "recover_slabim_poses.py",
                    "--root",
                    self.slabim,
                    *region_args,
                    *extra,
                    *(["--overwrite"] if self.args.force else []),
                ),
                complete=lambda: all(
                    (
                        self.slabim
                        / "sensor_data"
                        / region
                        / "points/lidar_pose_local_to_bim_from_rosbag.txt"
                    ).exists()
                    and (
                        self.args.keep_rosbags
                        or not any(
                            (
                                self.slabim
                                / "sensor_data"
                                / region
                                / "rosbag"
                            ).glob("*.bag")
                        )
                    )
                    for region in self.regions
                ),
            )
        if "verify" in stages:
            self.run(
                "verify",
                self.command(
                    "verify_slabim_dataset.py",
                    "--root",
                    self.slabim,
                    *region_args,
                    "--output",
                    self.project / "outputs/dataset_verification.json",
                ),
            )
        if "prepare" in stages:
            train_val_regions = list(
                dict.fromkeys(
                    [
                        *self.v1_cfg.data.train_regions,
                        *self.v1_cfg.data.val_regions,
                    ]
                )
            )
            test_regions = list(self.v1_cfg.data.test_regions)
            selected = set(self.regions)
            groups = (
                (
                    "prepare-train-val",
                    [region for region in train_val_regions if region in selected],
                    self.args.train_val_stride,
                ),
                (
                    "prepare-test",
                    [region for region in test_regions if region in selected],
                    self.args.test_stride,
                ),
            )
            for stage_name, regions, stride in groups:
                if not regions:
                    continue
                arguments: list[object] = [
                    "--config",
                    self.v1_config,
                    "--regions",
                    *regions,
                    "--stride",
                    stride,
                    "--replace-regions-in-manifest",
                ]
                if self.args.max_frames:
                    arguments.extend(("--max-frames", self.args.max_frames))
                if self.args.force:
                    arguments.append("--overwrite")
                self.run(stage_name, self.command("prepare_dataset.py", *arguments))
        if "audit" in stages:
            self.run(
                "audit",
                self.command(
                    "audit_dataset.py",
                    "--config",
                    self.v1_config,
                    "--output",
                    processed / "audit.json",
                ),
            )
        if "anchors" in stages:
            self.run(
                "anchors",
                self.command(
                    "prepare_strong_anchors.py",
                    "--config",
                    self.v1_config,
                    *(["--overwrite"] if self.args.force else []),
                ),
            )
        if "train-v1" in stages:
            self.run(
                "train-v1",
                self.command("train.py", "--config", self.v1_config, "--device", self.args.device),
                complete=lambda: (v1_output / "best.pt").exists(),
            )
        if "eval-v1" in stages:
            for split in ("val", "test"):
                self.run(
                    f"eval-v1-{split}",
                    self.command(
                        "evaluate.py",
                        "--config",
                        self.v1_config,
                        "--checkpoint",
                        v1_output / "best.pt",
                        "--device",
                        self.args.device,
                        "--split",
                        split,
                        "--output",
                        v1_output / f"evaluation_{split}",
                    ),
                )
        if "cache-candidates" in stages:
            self.run(
                "cache-candidates",
                self.command(
                    "cache_candidate_predictions.py",
                    "--config",
                    self.v1_config,
                    "--checkpoint",
                    v1_output / "best.pt",
                    "--device",
                    self.args.device,
                    *(["--overwrite"] if self.args.force else []),
                ),
            )
        if "train-v3" in stages:
            self.run(
                "train-v3",
                self.command("train.py", "--config", self.v3_config, "--device", self.args.device),
                complete=lambda: (v3_output / "best.pt").exists(),
            )
        if "eval-v3" in stages:
            for split in ("val", "test"):
                self.run(
                    f"eval-v3-{split}",
                    self.command(
                        "evaluate.py",
                        "--config",
                        self.v3_config,
                        "--checkpoint",
                        v3_output / "best.pt",
                        "--device",
                        self.args.device,
                        "--split",
                        split,
                        "--output",
                        v3_output / f"evaluation_{split}",
                    ),
                )
        if "reconstruct" in stages:
            arguments = [
                "--config",
                self.v3_config,
                "--checkpoint",
                v3_output / "best.pt",
                "--device",
                self.args.device,
                "--split",
                "test",
                "--output",
                v3_output / "reconstruction_test",
                "--save-clouds",
            ]
            if self.args.max_frames:
                arguments.extend(("--max-frames", self.args.max_frames))
            self.run("reconstruct", self.command("evaluate_reconstruction.py", *arguments))
        if "report" in stages:
            self.run(
                "report",
                self.command(
                    "summarize_experiments.py",
                    "--project-root",
                    self.project,
                    "--v1-output",
                    v1_output,
                    "--v3-output",
                    v3_output,
                    "--processed-root",
                    processed,
                ),
            )


def main() -> None:
    args = parse_args()
    stages = list(STAGES) if "all" in args.stages else list(args.stages)
    Pipeline(args).execute(stages)


if __name__ == "__main__":
    main()
