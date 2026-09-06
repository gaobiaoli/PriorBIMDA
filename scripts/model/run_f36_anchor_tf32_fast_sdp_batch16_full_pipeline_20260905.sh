#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
run_name=stanford_area1_f36_anchor_tf32_fast_sdp_batch16_6epoch_20260905
config="$project_root/configs/${run_name}.yaml"
output_dir="$project_root/outputs/$run_name"
checkpoint="$output_dir/best.pt"
tmp_dir=/home/bgao491/tmp
log_path="$tmp_dir/${run_name}.log"
status_path="$tmp_dir/${run_name}.status"
commands_path="$tmp_dir/${run_name}.cmd"

mkdir -p "$tmp_dir"
if [[ -e "$output_dir/best.pt" || -e "$output_dir/latest.pt" || -e "$output_dir/training_history.csv" ]]; then
  echo "Refusing to overwrite an existing training run: $output_dir" >&2
  exit 2
fi
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite an existing pipeline log: $log_path" >&2
  exit 2
fi

exec >>"$log_path" 2>&1

finish() {
  local exit_code=$1
  local final_state=FAILED
  if [[ $exit_code -eq 0 ]]; then
    final_state=COMPLETED
  fi
  {
    echo "state=$final_state"
    echo "exit_code=$exit_code"
    echo "finished_at=$(date -Ins)"
    echo "log=$log_path"
    echo "output_dir=$output_dir"
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
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=0

{
  echo "state=RUNNING"
  echo "started_at=$(date -Ins)"
  echo "pid=$$"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "config=$config"
  echo "batch_size=16"
  echo "gradient_accumulation=1"
  echo "effective_batch_size=16"
  echo "log_every=20"
  echo "deterministic_warn_only=true"
  echo "native_cuda_resize_backward=true"
  echo "tf32=true"
  echo "flash_sdp=true"
  echo "memory_efficient_sdp=true"
  echo "math_sdp=true"
  echo "cudnn_sdp=true"
  echo "source_snapshot=false"
  echo "log=$log_path"
} >"$status_path"

{
  printf 'nohup setsid bash %q >/dev/null 2>&1 &\n' "$project_root/scripts/model/run_f36_anchor_tf32_fast_sdp_batch16_full_pipeline_20260905.sh"
  printf 'tail -f %q\n' "$log_path"
  printf 'cat %q\n' "$status_path"
} >"$commands_path"

echo "pipeline_started=$(date -Ins)"
echo "git_commit=$(git rev-parse HEAD)"
echo "config_sha256=$(sha256sum "$config" | awk '{print $1}')"
echo "deterministic_warn_only=true native_cuda_resize_backward=true seed=42"
echo "tf32=true flash_sdp=true memory_efficient_sdp=true math_sdp=true cudnn_sdp=true"
echo "micro_batch=16 gradient_accumulation=1 effective_batch=16 log_every=20 source_snapshot=false"
run_command nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader

run_command "$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$config" \
  --device cuda \
  --deterministic \
  --allow-tf32 \
  --allow-fast-sdp

echo "stanford_training_val_test_completed=$(date -Ins)"

for scene in hxp 759 1px; do
  zero_shot_dir="$project_root/results/matterport3d/${scene}_f36_anchor_tf32_fast_sdp_batch16_6epoch_20260905_zero_shot"
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
    --no-resume
  echo "zero_shot_scene_completed=$scene time=$(date -Ins)"
done

echo "pipeline_completed=$(date -Ins)"
