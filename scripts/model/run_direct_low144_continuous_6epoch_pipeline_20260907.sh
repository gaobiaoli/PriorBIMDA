#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
run_name=stanford_area1_direct_low144_continuous_6epoch_20260907
config="$project_root/configs/$run_name.yaml"
output_dir="/mnt/priorbimda-data/PriorBIMDA-Outputs/$run_name"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
log_path=/home/bgao491/tmp/direct_low144_continuous_6epoch_pipeline_20260907.log
status_path=/home/bgao491/tmp/direct_low144_continuous_6epoch_pipeline_20260907.status
commands_path=/home/bgao491/tmp/direct_low144_continuous_6epoch_pipeline_20260907.cmd

mkdir -p /home/bgao491/tmp "$cache_root"
if [[ -e "$output_dir/best.pt" || -e "$output_dir/latest.pt" || -e "$output_dir/training_history.csv" ]]; then
  echo "Refusing to overwrite existing training output: $output_dir" >&2
  exit 2
fi
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite existing log: $log_path" >&2
  exit 2
fi
for checkpoint_label in best latest; do
  for scene in hxp 759 1px; do
    result_dir="$project_root/results/matterport3d/${scene}_direct_low144_continuous6_${checkpoint_label}_20260907_zero_shot"
    if [[ -e "$result_dir/per_frame.csv" || -e "$result_dir/summary.json" ]]; then
      echo "Refusing to overwrite existing result: $result_dir" >&2
      exit 2
    fi
  done
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
    echo "output_dir=$output_dir"
    echo "best_checkpoint=$output_dir/best.pt"
    echo "latest_checkpoint=$output_dir/latest.pt"
    echo "cache_root=$cache_root"
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
  echo "git_commit=$(git rev-parse HEAD)"
  echo "architecture=direct_F144_native_r144_bilinear_to_504_no_global_scale"
  echo "training=fresh_uninterrupted_epochs_1_to_6"
  echo "scheduler=single_cosine_schedule_over_all_6_epochs"
  echo "micro_batch=2"
  echo "gradient_accumulation=8"
  echo "effective_batch=16"
  echo "runtime_flags=original_no_deterministic_no_tf32_override_no_fast_sdp_override"
  echo "evaluate_checkpoints=best,latest"
  echo "log=$log_path"
} >"$status_path"
{
  printf 'nohup setsid bash %q >/dev/null 2>&1 &\n' "$project_root/scripts/model/run_direct_low144_continuous_6epoch_pipeline_20260907.sh"
  printf 'tail -f %q\n' "$log_path"
  printf 'cat %q\n' "$status_path"
} >"$commands_path"

echo "pipeline_started=$(date -Ins)"
echo "git_commit=$(git rev-parse HEAD)"
echo "config_sha256=$(sha256sum "$config" | awk '{print $1}')"
echo "training=fresh_uninterrupted_epochs_1_to_6 single_cosine_schedule"
echo "micro_batch=2 gradient_accumulation=8 effective_batch=16"
run_command nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
run_command "$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$config" \
  --device cuda
echo "stanford_training_val_test_completed=$(date -Ins)"

for checkpoint_label in best latest; do
  checkpoint="$output_dir/${checkpoint_label}.pt"
  echo "zero_shot_checkpoint_started=$checkpoint_label checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}') time=$(date -Ins)"
  for scene in hxp 759 1px; do
    result_dir="$project_root/results/matterport3d/${scene}_direct_low144_continuous6_${checkpoint_label}_20260907_zero_shot"
    echo "zero_shot_scene_started=$scene checkpoint=$checkpoint_label time=$(date -Ins)"
    run_command "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
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
    echo "zero_shot_scene_completed=$scene checkpoint=$checkpoint_label time=$(date -Ins)"
  done
done
echo "pipeline_completed=$(date -Ins)"
