#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
run_name=stanford_area1_scale_r36_final_depth_scale_supervision_effective_batch16_6epoch_20260905
config="$project_root/configs/${run_name}.yaml"
output_dir="$project_root/outputs/$run_name"
checkpoint="$output_dir/best.pt"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
tmp_dir=/home/bgao491/tmp
log_path="$tmp_dir/${run_name}.log"
status_path="$tmp_dir/${run_name}.status"

mkdir -p "$tmp_dir" "$cache_root"
if [[ -e "$output_dir/best.pt" || -e "$output_dir/latest.pt" || -e "$output_dir/training_history.csv" ]]; then
  echo "Refusing to overwrite existing training output: $output_dir" >&2
  exit 2
fi
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite existing log: $log_path" >&2
  exit 2
fi
for scene in hxp 759 1px; do
  zero_shot_dir="$project_root/results/matterport3d/${scene}_scale_r36_final_depth_scale_supervision_effective_batch16_6epoch_20260905_zero_shot"
  if [[ -e "$zero_shot_dir/per_frame.csv" || -e "$zero_shot_dir/summary.json" ]]; then
    echo "Refusing to overwrite existing result: $zero_shot_dir" >&2
    exit 2
  fi
done

exec >>"$log_path" 2>&1

finish() {
  local exit_code=$1
  local final_state=FAILED
  if [[ $exit_code -eq 0 ]]; then final_state=COMPLETED; fi
  {
    echo "state=$final_state"
    echo "exit_code=$exit_code"
    echo "finished_at=$(date -Ins)"
    echo "log=$log_path"
    echo "output_dir=$output_dir"
    echo "checkpoint=$checkpoint"
    echo "cache_root=$cache_root"
  } >"$status_path"
  echo "pipeline_state=$final_state exit_code=$exit_code finished_at=$(date -Ins)"
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
  echo "git_commit=$(git rev-parse HEAD)"
  echo "config=$config"
  echo "architecture=original_scale_r36_no_post_scale_adapter"
  echo "scale_oracle_teacher_weight=0"
  echo "scale_equivariance_weight=0"
  echo "r36_teacher_weight=0.5"
  echo "r36_zero_mean_weight=0.1"
  echo "micro_batch=2"
  echo "gradient_accumulation=8"
  echo "effective_batch=16"
  echo "cache_root=$cache_root"
  echo "log=$log_path"
} >"$status_path"

echo "pipeline_started=$(date -Ins)"
echo "git_commit=$(git rev-parse HEAD)"
echo "config_sha256=$(sha256sum "$config" | awk '{print $1}')"
echo "architecture=original_scale_r36_no_post_scale_adapter"
echo "scale_supervision=final_depth_only effective_batch=16"

run_command "$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$config" \
  --device cuda

echo "stanford_training_val_test_completed=$(date -Ins)"
for scene in hxp 759 1px; do
  zero_shot_dir="$project_root/results/matterport3d/${scene}_scale_r36_final_depth_scale_supervision_effective_batch16_6epoch_20260905_zero_shot"
  echo "zero_shot_scene_started=$scene time=$(date -Ins)"
  run_command "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
    --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
    --bimnet-root /home/bgao491/BIMNet_release \
    --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
    --bimnet-scene "$scene" \
    --config "$output_dir/config.yaml" \
    --checkpoint "$checkpoint" \
    --output-dir "$zero_shot_dir" \
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
