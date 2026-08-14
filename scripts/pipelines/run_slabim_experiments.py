#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from bim_priorda3.config import load_config, resolve_project_path, resolve_slabim_root

STAGES = (
    "download",
    "poses",
    "verify",
    "prepare",
    "audit",
    "pretrain",
    "finetune",
    "evaluate",
    "reconstruct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable public SLABIM data preparation and two-stage training pipeline"
    )
    parser.add_argument(
        "--pretrain-config",
        type=Path,
        default=Path("configs/slabim_pretrain.yaml"),
    )
    parser.add_argument(
        "--final-config",
        type=Path,
        default=Path("configs/slabim.yaml"),
    )
    parser.add_argument("--slabim-root", type=Path)
    parser.add_argument("--regions", nargs="+")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("all",) + STAGES,
        default=("verify",),
        help="Default is a read-only dataset check; use 'all' explicitly",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-val-stride", type=int, default=2)
    parser.add_argument("--test-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--keep-rosbags", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("outputs/pipeline_state_slabim.json"),
    )
    return parser.parse_args()


def annotation_region_strides(path: Path) -> dict[str, int]:
    """Infer each region's source stride from an exhaustive split annotation.

    The pooled SLABIM protocol ships the exact active *and excluded* population
    in one JSONL file.  Deriving preparation strides from that authority avoids
    silently falling back to the older region-role split when reproducing the
    public experiment.
    """

    frames_by_region: dict[str, list[int]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or sample_id.count("/") != 1:
            raise ValueError(f"{path}:{line_number}: invalid annotation ID {sample_id!r}")
        region, frame_text = sample_id.split("/")
        if not frame_text.isdigit():
            raise ValueError(
                f"{path}:{line_number}: SLABIM frame ID must be numeric, got {sample_id!r}"
            )
        frames_by_region[region].append(int(frame_text))
    if not frames_by_region:
        raise ValueError(f"Split annotation is empty: {path}")

    strides: dict[str, int] = {}
    for region, raw_frames in sorted(frames_by_region.items()):
        frames = sorted(raw_frames)
        if len(frames) != len(set(frames)):
            raise ValueError(f"{path}: duplicate frame IDs in {region}")
        if frames[0] != 0:
            raise ValueError(f"{path}: {region} population must start at frame 0")
        if len(frames) == 1:
            strides[region] = 1
            continue
        differences = {right - left for left, right in zip(frames, frames[1:])}
        if len(differences) != 1 or next(iter(differences)) < 1:
            raise ValueError(f"{path}: {region} is not an exhaustive constant-stride population")
        strides[region] = differences.pop()
    return strides


def regions_requiring_pose_rosbags(
    slabim_root: Path,
    regions: list[str],
    *,
    force_pose_refresh: bool,
) -> list[str]:
    """Return regions whose pose stage must have the source rosbags restored."""

    if force_pose_refresh:
        return list(regions)
    return [
        region
        for region in regions
        if not (
            slabim_root / "sensor_data" / region / "points/lidar_pose_local_to_bim_from_rosbag.txt"
        ).exists()
    ]


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project = Path(__file__).resolve().parents[2]
        self.pretrain_config = self._path(args.pretrain_config)
        self.final_config = self._path(args.final_config)
        self.pretrain_cfg = load_config(self.pretrain_config)
        self.final_cfg = load_config(self.final_config)
        self.slabim = (
            args.slabim_root.expanduser().resolve()
            if args.slabim_root
            else resolve_slabim_root(self.pretrain_cfg)
        )
        self.regions = args.regions or list(self.pretrain_cfg.data.regions)
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

    def run(
        self,
        stage: str,
        command: list[str],
        complete: Callable[[], bool] | None = None,
    ) -> None:
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
            json.dumps(self.state, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _require(self, path: Path, purpose: str) -> None:
        if not path.exists() and not self.args.dry_run:
            raise FileNotFoundError(f"{purpose} requires {path}")

    def execute(self, stages: list[str]) -> None:
        pretrain_output = resolve_project_path(
            self.pretrain_cfg,
            self.pretrain_cfg.experiment.output_dir,
        )
        final_output = resolve_project_path(
            self.final_cfg,
            self.final_cfg.experiment.output_dir,
        )
        processed = resolve_project_path(
            self.pretrain_cfg,
            self.pretrain_cfg.data.processed_root,
        )
        region_args = ["--regions", *self.regions]

        if "download" in stages:
            pose_missing = regions_requiring_pose_rosbags(
                self.slabim,
                self.regions,
                force_pose_refresh=self.args.force and "poses" in stages,
            )
            pose_ready = [region for region in self.regions if region not in pose_missing]
            if pose_missing:
                self.run(
                    "download-with-rosbags",
                    self.command(
                        "data/download_slabim.py",
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
                        "data/download_slabim.py",
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
                    "data/recover_slabim_poses.py",
                    "--root",
                    self.slabim,
                    *region_args,
                    *extra,
                    *(["--overwrite"] if self.args.force else []),
                ),
            )

        if "verify" in stages:
            self.run(
                "verify",
                self.command(
                    "data/verify_slabim_dataset.py",
                    "--root",
                    self.slabim,
                    *region_args,
                    "--output",
                    self.project / "outputs/dataset_verification.json",
                ),
            )

        if "prepare" in stages:
            selected = set(self.regions)
            annotation_value = self.pretrain_cfg.data.get("split_annotation")
            if annotation_value:
                annotation = resolve_project_path(self.pretrain_cfg, annotation_value)
                self._require(annotation, "annotation-driven SLABIM preparation")
                stride_groups: dict[int, list[str]] = defaultdict(list)
                if annotation.exists():
                    for region, stride in annotation_region_strides(annotation).items():
                        if region in selected:
                            stride_groups[stride].append(region)
                groups = tuple(
                    (f"prepare-stride-{stride}", regions, stride)
                    for stride, regions in sorted(stride_groups.items())
                )
            else:
                train_val_regions = list(
                    dict.fromkeys(
                        [
                            *self.pretrain_cfg.data.train_regions,
                            *self.pretrain_cfg.data.val_regions,
                        ]
                    )
                )
                groups = (
                    (
                        "prepare-train-val",
                        [region for region in train_val_regions if region in selected],
                        self.args.train_val_stride,
                    ),
                    (
                        "prepare-test",
                        [
                            region
                            for region in self.pretrain_cfg.data.test_regions
                            if region in selected
                        ],
                        self.args.test_stride,
                    ),
                )
            for stage_name, regions, stride in groups:
                if not regions:
                    continue
                arguments: list[object] = [
                    "--config",
                    self.pretrain_config,
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
                self.run(
                    stage_name,
                    self.command("data/prepare_dataset.py", *arguments),
                )

        if "audit" in stages:
            self.run(
                "audit",
                self.command(
                    "data/audit_dataset.py",
                    "--config",
                    self.pretrain_config,
                    "--output",
                    processed / "audit.json",
                ),
            )

        pretrain_checkpoint = pretrain_output / "accepted.pt"
        if "pretrain" in stages:
            self.run(
                "pretrain",
                self.command(
                    "model/train.py",
                    "--config",
                    self.pretrain_config,
                    "--device",
                    self.args.device,
                ),
                complete=pretrain_checkpoint.exists,
            )

        final_checkpoint = final_output / "accepted.pt"
        if "finetune" in stages:
            self._require(pretrain_checkpoint, "V5 stage-two fine-tuning")
            self.run(
                "finetune",
                self.command(
                    "model/train.py",
                    "--config",
                    self.final_config,
                    "--init-checkpoint",
                    pretrain_checkpoint,
                    "--device",
                    self.args.device,
                ),
                complete=final_checkpoint.exists,
            )

        if "evaluate" in stages:
            self._require(final_checkpoint, "V5 evaluation")
            for split in ("val", "test"):
                self.run(
                    f"evaluate-{split}",
                    self.command(
                        "model/evaluate.py",
                        "--config",
                        self.final_config,
                        "--checkpoint",
                        final_checkpoint,
                        "--device",
                        self.args.device,
                        "--split",
                        split,
                        "--output",
                        final_output / f"evaluation_{split}",
                    ),
                )

        if "reconstruct" in stages:
            self._require(final_checkpoint, "V5 reconstruction")
            arguments: list[object] = [
                "--config",
                self.final_config,
                "--checkpoint",
                final_checkpoint,
                "--device",
                self.args.device,
                "--split",
                "test",
                "--output",
                final_output / "reconstruction_test",
                "--save-clouds",
            ]
            if self.args.max_frames:
                arguments.extend(("--max-frames", self.args.max_frames))
            self.run(
                "reconstruct",
                self.command("model/evaluate_reconstruction.py", *arguments),
            )


def main() -> None:
    args = parse_args()
    stages = list(STAGES) if "all" in args.stages else list(args.stages)
    Pipeline(args).execute(stages)


if __name__ == "__main__":
    main()
