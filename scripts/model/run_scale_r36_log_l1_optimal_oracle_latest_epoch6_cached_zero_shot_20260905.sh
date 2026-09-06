#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
train_run=stanford_area1_scale_r36_log_l1_optimal_oracle_effective_batch16_6epoch_20260905
config="$project_root/outputs/$train_run/config.yaml"
checkpoint="$project_root/outputs/$train_run/latest.pt"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
run_name=scale_r36_log_l1_optimal_oracle_latest_epoch6_cached_zero_shot_20260905
tmp_dir=/home/bgao491/tmp
log_path="$tmp_dir/${run_name}.log"
status_path="$tmp_dir/${run_name}.status"

mkdir -p "$tmp_dir" "$cache_root"
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite existing log: $log_path" >&2
  exit 2
fi
for scene in hxp 759 1px; do
  output_dir="$project_root/results/matterport3d/${scene}_scale_r36_log_l1_optimal_oracle_effective_batch16_6epoch_20260905_latest_epoch6_zero_shot"
  if [[ -e "$output_dir/per_frame.csv" || -e "$output_dir/summary.json" ]]; then
    echo "Refusing to overwrite existing result: $output_dir" >&2
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
    echo "checkpoint=$checkpoint"
    echo "log=$log_path"
  } >"$status_path"
  echo "pipeline_state=$state exit_code=$exit_code finished_at=$(date -Ins)"
}
trap 'finish $?' EXIT

run_command() {
  printf 'CMD'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/scripts/model"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

{
  echo "state=RUNNING"
  echo "started_at=$(date -Ins)"
  echo "pid=$$"
  echo "checkpoint=$checkpoint"
  echo "checkpoint_epoch=6"
  echo "config=$config"
  echo "cache_root=$cache_root"
  echo "log=$log_path"
} >"$status_path"

echo "pipeline_started=$(date -Ins)"
echo "checkpoint=$checkpoint"
echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
for scene in hxp 759 1px; do
  output_dir="$project_root/results/matterport3d/${scene}_scale_r36_log_l1_optimal_oracle_effective_batch16_6epoch_20260905_latest_epoch6_zero_shot"
  echo "zero_shot_scene_started=$scene time=$(date -Ins)"
  run_command "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
    --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
    --bimnet-root /home/bgao491/BIMNet_release \
    --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
    --bimnet-scene "$scene" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --selection-audit "$project_root/data/provenance/matterport_bimnet_three_rule_v1.json" \
    --evaluate-selected-only \
    --process-res 504 \
    --device cuda \
    --progress-every 100 \
    --no-resume \
    --da3-cache-dir "$cache_root"
  echo "zero_shot_scene_completed=$scene time=$(date -Ins)"
done

echo "pipeline_completed=$(date -Ins)"
