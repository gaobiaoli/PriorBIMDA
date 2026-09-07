#!/usr/bin/env bash
set -Eeuo pipefail

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
train_run=stanford_area1_f36_adapter_resblocks3_main_path_da3_scale_perturb_no_equivariance_effective_batch16_6epoch_20260906
config="$project_root/outputs/$train_run/config.yaml"
checkpoint="$project_root/outputs/$train_run/latest.pt"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
result_suffix=f36_adapter_resblocks3_main_path_da3_scale_perturb_latest_epoch6_20260906_zero_shot
tmp_dir=/home/bgao491/tmp
log_path="$tmp_dir/${result_suffix}.log"
status_path="$tmp_dir/${result_suffix}.status"

mkdir -p "$tmp_dir" "$cache_root"
if [[ ! -f "$checkpoint" ]]; then
  echo "Missing checkpoint: $checkpoint" >&2
  exit 2
fi
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite existing log: $log_path" >&2
  exit 2
fi
for scene in hxp 759 1px; do
  result_dir="$project_root/results/matterport3d/${scene}_${result_suffix}"
  if [[ -e "$result_dir/per_frame.csv" || -e "$result_dir/summary.json" ]]; then
    echo "Refusing to overwrite existing result: $result_dir" >&2
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
    echo "checkpoint=$checkpoint"
    echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
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
export CUDA_VISIBLE_DEVICES=0

{
  echo "state=RUNNING"
  echo "started_at=$(date -Ins)"
  echo "pid=$$"
  echo "checkpoint=$checkpoint"
  echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
  echo "checkpoint_epoch=6"
  echo "config=$config"
  echo "cache_root=$cache_root"
  echo "log=$log_path"
} >"$status_path"

echo "pipeline_started=$(date -Ins)"
for scene in hxp 759 1px; do
  result_dir="$project_root/results/matterport3d/${scene}_${result_suffix}"
  echo "zero_shot_scene_started=$scene time=$(date -Ins)"
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
  echo "zero_shot_scene_completed=$scene time=$(date -Ins)"
done

echo "pipeline_completed=$(date -Ins)"
