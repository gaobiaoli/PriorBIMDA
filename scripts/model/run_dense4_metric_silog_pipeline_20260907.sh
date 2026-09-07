#!/usr/bin/env bash
set -Eeuo pipefail
project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
run_name=stanford_area1_dense4_metric_silog_effective_batch16_6epoch_20260907
config="$project_root/configs/$run_name.yaml"
output_dir="/mnt/priorbimda-data/PriorBIMDA-Outputs/$run_name"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
log_path="/home/bgao491/tmp/$run_name.log"
status_path="/home/bgao491/tmp/$run_name.status"
commands_path="/home/bgao491/tmp/$run_name.cmd"
if [[ -e "$log_path" || -e "$output_dir/receipt.json" ]]; then
  echo "Refusing to overwrite dense4 experiment" >&2
  exit 2
fi
mkdir -p /home/bgao491/tmp "$output_dir" "$cache_root"
exec >>"$log_path" 2>&1
finish() {
  local rc=$1
  local state=FAILED
  if [[ $rc -eq 0 ]]; then state=COMPLETED; fi
  printf 'state=%s\nexit_code=%s\nfinished_at=%s\nlog=%s\noutput_dir=%s\n' \
    "$state" "$rc" "$(date -Ins)" "$log_path" "$output_dir" >"$status_path"
  echo "pipeline_state=$state exit_code=$rc finished_at=$(date -Ins)"
}
trap 'finish $?' EXIT
run_command() {
  { printf 'CMD'; printf ' %q' "$@"; printf '\n'; } | tee -a "$commands_path"
  "$@"
}
cd "$project_root"
export PYTHONPATH="$project_root/src:$project_root/scripts/model"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=42 CUDA_VISIBLE_DEVICES=0
printf 'state=RUNNING\npid=%s\nstarted_at=%s\nlog=%s\noutput_dir=%s\n' \
  "$$" "$(date -Ins)" "$log_path" "$output_dir" >"$status_path"
echo "pipeline_started=$(date -Ins) git_commit=$(git rev-parse HEAD)"
echo "architecture=direct_metric_dense4_no_scale_no_r36_no_F36_adapter"
echo "micro_batch=2 accumulation=8 effective_batch=16 epochs=6 source_snapshot=false"
echo "SILog=10sqrt(var_g_correction1+0.15mean_g_squared) fixed_all_valid_GT=true"
echo "official_metric_head_max=20m no_GT_cutoff=true"
run_command sha256sum "$config"
run_command "$python_bin" scripts/model/train_dav2_dense4.py --config "$config" --device cuda
for checkpoint_name in best latest; do
  for scene in hxp 759 1px; do
    echo "ZERO_SHOT checkpoint=$checkpoint_name scene=$scene started=$(date -Ins)"
    run_command "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
      --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
      --bimnet-root /home/bgao491/BIMNet_release \
      --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
      --bimnet-scene "$scene" --config "$config" --checkpoint "$output_dir/$checkpoint_name.pt" \
      --output-dir "$project_root/results/matterport3d/${scene}_dense4_metric_silog_${checkpoint_name}_20260907_zero_shot" \
      --selection-audit "$project_root/data/provenance/matterport_bimnet_three_rule_v1.json" \
      --evaluate-selected-only --process-res 504 --device cuda --progress-every 100 \
      --no-resume --da3-cache-dir "$cache_root"
  done
done
