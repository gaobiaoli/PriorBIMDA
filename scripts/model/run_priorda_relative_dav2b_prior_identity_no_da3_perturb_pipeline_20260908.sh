#!/usr/bin/env bash
set -Eeuo pipefail
project_root=/home/bgao491/PriorBIMDA
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
run_name=stanford_area1_priorda_relative_dav2b_prior_identity_no_da3_perturb_effective_batch16_6epoch_20260908
config="$project_root/configs/$run_name.yaml"
output_dir="/mnt/priorbimda-data/PriorBIMDA-Outputs/$run_name"
cache_root=/mnt/priorbimda-data/PriorBIMDA-Caches/matterport3d_da3_raw_v1
log_path="/home/bgao491/tmp/$run_name.log"
status_path="/home/bgao491/tmp/$run_name.status"
commands_path="/home/bgao491/tmp/$run_name.cmd"
if [[ -e "$log_path" || -e "$output_dir/receipt.json" ]]; then
  echo "Refusing to overwrite exact-prior-identity controlled experiment" >&2
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
echo "architecture=DAv2-relative-ViT-B14+DPT+zero-init-condition-and-exact-prior-identity-head"
echo "condition=q_prior_normalized,q_BIM_in_prior_frame,BIM_mask;auxiliary_signal=T(q_relative)"
echo "normalization=D_prior_min_range_for_condition_and_T_signal"
echo "output=D_prior*exp(bounded_zero_init_log_depth_residual) bitwise_identity_at_step0 no_prior_minimum_floor"
echo "micro_batch=2 accumulation=8 effective_batch=16 epochs=6"
echo "optimizer=AdamW encoder_lr=5e-6 decoder_and_condition_lr=5e-5 scheduler=cosine"
echo "mixed_precision=bfloat16 gradient_scaler=false exact_positive_reciprocal_condition"
echo "loss=official_ZoeDepth_SILog_10sqrt(var_g_correction1+0.15mean_g_squared)"
echo "augment=rgb_jitter,flip,bim_local_dropout;full_dropout=false;shuffle=false;da3_scale_perturb=false"
run_command sha256sum "$config"
run_command "$python_bin" -m pytest -q tests/test_priorda_relative_metric_refiner.py tests/test_dav2_dense4.py
run_command "$python_bin" scripts/model/train_dav2_dense4.py --config "$config" --device cuda
for checkpoint_name in best latest; do
  for scene in hxp 759 1px; do
    echo "PRIOR_IDENTITY_ZERO_SHOT checkpoint=$checkpoint_name scene=$scene started=$(date -Ins)"
    run_command "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
      --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
      --bimnet-root /home/bgao491/BIMNet_release \
      --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
      --bimnet-scene "$scene" --config "$config" --checkpoint "$output_dir/$checkpoint_name.pt" \
      --output-dir "$project_root/results/matterport3d/${scene}_priorda_relative_dav2b_prior_identity_no_da3_perturb_${checkpoint_name}_20260908_zero_shot" \
      --selection-audit "$project_root/data/provenance/matterport_bimnet_three_rule_v1.json" \
      --evaluate-selected-only --process-res 504 --device cuda --progress-every 100 \
      --no-resume --da3-cache-dir "$cache_root"
  done
done
