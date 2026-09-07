#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
stage1_name=stanford_area1_direct_low18_retrain_stage1_3epoch_20260907
stage2_name=stanford_area1_direct_low18_retrain_stage2_6epoch_20260907
stage1_config="$project_root/configs/$stage1_name.yaml"
stage2_config="$project_root/configs/$stage2_name.yaml"
stage1_output="/mnt/priorbimda-data/PriorBIMDA-Outputs/$stage1_name"
stage2_output="/mnt/priorbimda-data/PriorBIMDA-Outputs/$stage2_name"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
log_path=/home/bgao491/tmp/direct_low18_exact_retrain_pipeline_20260907.log
status_path=/home/bgao491/tmp/direct_low18_exact_retrain_pipeline_20260907.status
commands_path=/home/bgao491/tmp/direct_low18_exact_retrain_pipeline_20260907.cmd

mkdir -p /home/bgao491/tmp "$cache_root"
for output_dir in "$stage1_output" "$stage2_output"; do
  if [[ -e "$output_dir/best.pt" || -e "$output_dir/latest.pt" || -e "$output_dir/training_history.csv" ]]; then
    echo "Refusing to overwrite existing training output: $output_dir" >&2
    exit 2
  fi
done
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite existing log: $log_path" >&2
  exit 2
fi
for checkpoint_label in best latest; do
  for scene in hxp 759 1px; do
    result_dir="$project_root/results/matterport3d/${scene}_direct_low18_exact_retrain_${checkpoint_label}_20260907_zero_shot"
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
    echo "stage1_output=$stage1_output"
    echo "stage2_output=$stage2_output"
    echo "best_checkpoint=$stage2_output/best.pt"
    echo "latest_checkpoint=$stage2_output/latest.pt"
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
  echo "architecture=direct_F18_r18_upsampled_to_504_no_global_scale"
  echo "stage1=fresh_epochs_1_to_3_original_learning_rates"
  echo "stage2=resume_fresh_stage1_best_epoch_reset_optimizer_lower_learning_rates_to_epoch_6"
  echo "micro_batch=2"
  echo "gradient_accumulation=8"
  echo "effective_batch=16"
  echo "runtime_flags=original_no_deterministic_no_tf32_override_no_fast_sdp_override"
  echo "evaluate_checkpoints=best,latest"
  echo "log=$log_path"
} >"$status_path"

{
  printf 'nohup setsid bash %q >/dev/null 2>&1 &\n' "$project_root/scripts/model/run_direct_low18_exact_retrain_pipeline_20260907.sh"
  printf 'tail -f %q\n' "$log_path"
  printf 'cat %q\n' "$status_path"
} >"$commands_path"

echo "pipeline_started=$(date -Ins)"
echo "git_commit=$(git rev-parse HEAD)"
echo "stage1_config_sha256=$(sha256sum "$stage1_config" | awk '{print $1}')"
echo "stage2_config_sha256=$(sha256sum "$stage2_config" | awk '{print $1}')"
echo "micro_batch=2 gradient_accumulation=8 effective_batch=16"
run_command nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

echo "stage1_started=$(date -Ins)"
run_command "$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$stage1_config" \
  --device cuda
echo "stage1_completed=$(date -Ins) checkpoint=$stage1_output/best.pt"

echo "stage2_started=$(date -Ins) resume=$stage1_output/best.pt"
run_command "$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$stage2_config" \
  --device cuda \
  --resume "$stage1_output/best.pt"
echo "stanford_training_val_test_completed=$(date -Ins)"

for checkpoint_label in best latest; do
  checkpoint="$stage2_output/${checkpoint_label}.pt"
  echo "zero_shot_checkpoint_started=$checkpoint_label checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}') time=$(date -Ins)"
  for scene in hxp 759 1px; do
    result_dir="$project_root/results/matterport3d/${scene}_direct_low18_exact_retrain_${checkpoint_label}_20260907_zero_shot"
    echo "zero_shot_scene_started=$scene checkpoint=$checkpoint_label time=$(date -Ins)"
    run_command "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
      --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
      --bimnet-root /home/bgao491/BIMNet_release \
      --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
      --bimnet-scene "$scene" \
      --config "$stage2_config" \
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
  echo "zero_shot_checkpoint_completed=$checkpoint_label time=$(date -Ins)"
done

echo "pipeline_completed=$(date -Ins)"
