#!/usr/bin/env bash
set -Eeuo pipefail

evaluation_epoch=${1:?usage: $0 EPOCH}
case "$evaluation_epoch" in
  3|4|5|6) ;;
  *) echo "EPOCH must be one of 3, 4, 5, 6" >&2; exit 2 ;;
esac

project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
model_run=stanford_area1_f36_anchor_scale_gradient_tf32_fast_sdp_batch32_6epoch_20260905
epoch_tag=$(printf '%03d' "$evaluation_epoch")
run_name=f36_anchor_scale_gradient_epoch${evaluation_epoch}_cached_zero_shot_20260905
config="$project_root/outputs/$model_run/config.yaml"
checkpoint="$project_root/outputs/$model_run/epoch_${epoch_tag}.pt"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
tmp_dir=/home/bgao491/tmp
log_path="$tmp_dir/${run_name}.log"
status_path="$tmp_dir/${run_name}.status"

mkdir -p "$tmp_dir" "$cache_root"
test -s "$checkpoint"
if [[ -e "$log_path" ]]; then
  echo "Refusing to overwrite existing log: $log_path" >&2
  exit 2
fi
for scene in hxp 759 1px; do
  output_dir="$project_root/results/matterport3d/${scene}_${run_name}"
  if [[ -e "$output_dir/per_frame.csv" || -e "$output_dir/summary.json" ]]; then
    echo "Refusing to overwrite existing result: $output_dir" >&2
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
export CUDA_VISIBLE_DEVICES=0

{
  echo "state=RUNNING"
  echo "started_at=$(date -Ins)"
  echo "pid=$$"
  echo "epoch=$evaluation_epoch"
  echo "checkpoint=$checkpoint"
  echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
  echo "cache_root=$cache_root"
  echo "process_res=504"
  echo "scenes=hxp,759,1px"
  echo "log=$log_path"
} >"$status_path"

echo "pipeline_started=$(date -Ins) epoch=$evaluation_epoch"
echo "checkpoint_sha256=$(sha256sum "$checkpoint" | awk '{print $1}')"
for scene in hxp 759 1px; do
  output_dir="$project_root/results/matterport3d/${scene}_${run_name}"
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
