#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect dataset, 2D depth, and 3D reconstruction results into one report"
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--v1-output", type=Path, default=Path("outputs/slabim_single_frame_r50"))
    parser.add_argument("--v3-output", type=Path, default=Path("outputs/slabim_single_frame_r50_v3"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/slabim_504_r50"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_summary"))
    return parser.parse_args()


def _absolute(project: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (project / path).resolve()


def _load_optional(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _metric_row(label: str, summary: dict[str, Any] | None) -> str:
    if summary is None:
        return f"| {label} | 未运行 | — | — | — |"
    overall = summary.get("overall", {})
    base = overall.get("base", {})
    refined = overall.get("refined", {})
    return (
        f"| {label} | {summary.get('split', '?')} | "
        f"{base.get('abs_rel', float('nan')):.5f} | "
        f"{refined.get('abs_rel', float('nan')):.5f} | "
        f"{refined.get('rmse', float('nan')):.4f} |"
    )


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    v1 = _absolute(project, args.v1_output)
    v3 = _absolute(project, args.v3_output)
    processed = _absolute(project, args.processed_root)
    output = _absolute(project, args.output)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": processed / "audit.json",
        "v1_val": v1 / "evaluation_val" / "summary.json",
        "v1_test": v1 / "evaluation_test" / "summary.json",
        "v3_val": v3 / "evaluation_val" / "summary.json",
        "v3_test": v3 / "evaluation_test" / "summary.json",
        "v3_reconstruction_test": v3 / "reconstruction_test" / "summary.json",
    }
    results = {name: _load_optional(path) for name, path in paths.items()}
    bundle = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project),
        "sources": {name: str(path) for name, path in paths.items()},
        "results": results,
        "protocol": {
            "train_regions": ["3F_Region2", "3F_Region3", "4F_Region2", "4F_Region3"],
            "validation_region": ["5F_Region3"],
            "test_region": ["5F_Region2"],
            "test_policy": "fixed hyperparameters; do not select checkpoints or gates on test",
            "inference_inputs": "one RGB image + DA3 prediction + BIM render + calibration/pose",
            "pcd_role": "training/evaluation GT only; never a model inference input",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    reconstruction = results["v3_reconstruction_test"]
    reconstruction_lines = ["三维重建尚未运行。"]
    if reconstruction:
        reconstruction_lines = [
            "| 方法 | Chamfer-L1 (m) | Accuracy mean (m) | Completeness mean (m) |",
            "|---|---:|---:|---:|",
        ]
        for method, metrics in reconstruction.get("aggregate", {}).items():
            reconstruction_lines.append(
                f"| {method} | {metrics['chamfer_l1_m']:.4f} | "
                f"{metrics['accuracy_pred_to_gt']['mean_m']:.4f} | "
                f"{metrics['completeness_gt_to_pred']['mean_m']:.4f} |"
            )
    markdown = "\n".join(
        [
            "# SLABIM 完整实验汇总",
            "",
            "本报告由固定流水线自动生成。验证集用于模型选择，5F_Region2 测试集只用于",
            "固定方案的最终报告；PCD 仅用于生成 GT 和评测，不进入推理网络。",
            "",
            "## 单帧深度",
            "",
            "| 实验 | 划分 | 原始 DA3 AbsRel | 最终 AbsRel | 最终 RMSE (m) |",
            "|---|---|---:|---:|---:|",
            _metric_row("V1 learned refiner", results["v1_val"]),
            _metric_row("V1 learned refiner", results["v1_test"]),
            _metric_row("V3 candidate fusion", results["v3_val"]),
            _metric_row("V3 candidate fusion", results["v3_test"]),
            "",
            "每个 `evaluation_*/summary.json` 同时包含 global scale、固定",
            "scale+local BIM、coarse、refined，以及各距离区间结果。",
            "",
            "## 三维重建",
            "",
            *reconstruction_lines,
            "",
            "重建在 BIM 坐标系中直接融合，不进行评测时 ICP。默认 `prediction_mask=all`",
            "会同时受到稀疏 GT 覆盖率影响；`prediction_mask=gt` 仅作为同像素诊断。",
            "",
            "## 可复现性说明",
            "",
            "- 位姿由 rosbag 原始 Livox 扫描配准到官方 SLAM-global PCD，并保留诊断。",
            "- GT 是相邻 ±50 帧逐扫描 z-buffer 后的最前方一致簇。",
            "- 推理输入只有 RGB、DA3、BIM、内外参与恢复位姿，不读取 PCD/GT。",
            "- 若任何条目显示“未运行”，执行统一流水线对应阶段后重新汇总。",
            "",
        ]
    )
    (output / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote {output / 'summary.json'}")
    print(f"Wrote {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()

