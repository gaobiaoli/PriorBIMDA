#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/bgao491/miniconda3/envs/priorbimda/bin/python}"
config="$project_root/outputs/stanford_area1_priorbimda_stage2_dav2_metric/config.yaml"
checkpoint="$project_root/outputs/stanford_area1_priorbimda_stage2_dav2_metric/best.pt"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
log_path="$project_root/tmp/priorbimda_two_stage_zero_shot_20260906.log"
status_path="$project_root/tmp/priorbimda_two_stage_zero_shot_20260906.status"

mkdir -p "$project_root/tmp" "$cache_root"
test -f "$config"
test -f "$checkpoint"

for scene in hxp 759 1px; do
  result_dir="$project_root/results/matterport3d/${scene}_priorbimda_two_stage_20260906_zero_shot"
  if [[ -e "$result_dir/per_frame.csv" || -e "$result_dir/summary.json" ]]; then
    echo "Refusing existing zero-shot result: $result_dir" >&2
    exit 2
  fi
done

exec >>"$log_path" 2>&1

finish() {
  local exit_code=$1
  local state=FAILED
  if [[ $exit_code -eq 0 ]]; then state=COMPLETED; fi
  {
    echo "state=$state"
    echo "exit_code=$exit_code"
    echo "finished_at=$(date -Ins)"
    echo "log=$log_path"
  } >"$status_path"
}
trap 'finish $?' EXIT

{
  echo "state=RUNNING"
  echo "started_at=$(date -Ins)"
  echo "pid=$$"
  echo "checkpoint=$checkpoint"
  echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
  echo "protocol=frozen_three_rule_selected_frames_no_alignment"
} >"$status_path"

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/scripts/model"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

for scene in hxp 759 1px; do
  result_dir="$project_root/results/matterport3d/${scene}_priorbimda_two_stage_20260906_zero_shot"
  echo "zero_shot_scene_started=$scene time=$(date -Ins)"
  "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
    --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
    --bimnet-root /home/bgao491/BIMNet_release \
    --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
    --bimnet-scene "$scene" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$result_dir" \
    --selection-audit "$project_root/data/provenance/matterport_bimnet_three_rule_v1.json" \
    --evaluate-selected-only \
    --process-res 504 \
    --device cuda \
    --progress-every 100 \
    --no-resume \
    --da3-cache-dir "$cache_root"
  echo "zero_shot_scene_completed=$scene time=$(date -Ins)"
done

echo "zero_shot_pipeline_completed=$(date -Ins)"
