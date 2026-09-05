#!/usr/bin/env bash
set -euo pipefail

project_root=/home/bgao491/PriorBIMDA
run_name=stanford_area1_dav2_early_fusion_scale_low36_only_12epoch_deterministic_repro2_interrupt_after_epoch1
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
config="configs/${run_name}.yaml"
checkpoint="outputs/${run_name}/best.pt"
run_output="$project_root/outputs/$run_name"
eval_log="$run_output/post_interrupt_evaluation.log"

cd "$project_root"
export PYTHONPATH="$project_root/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=0

exec >>"$eval_log" 2>&1
echo "evaluation_started=$(date -Ins)"

"$python_bin" scripts/model/evaluate_dav2_joint_scale_low_checkpoint.py \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --split val \
  --output "results/stanford_area1/${run_name}/interrupted_epoch1_val_summary.json" \
  --device cuda \
  --deterministic
echo "stanford_val_completed=$(date -Ins)"

"$python_bin" scripts/model/evaluate_dav2_joint_scale_low_checkpoint.py \
  --config "$config" \
  --checkpoint "$checkpoint" \
  --split test \
  --output "results/stanford_area1/${run_name}/interrupted_epoch1_test_summary.json" \
  --device cuda \
  --deterministic
echo "stanford_test_completed=$(date -Ins)"

for scene in hxp 759 1px; do
  echo "zero_shot_scene_started=$scene time=$(date -Ins)"
  "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
    --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
    --bimnet-root /home/bgao491/BIMNet_release \
    --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
    --bimnet-scene "$scene" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "results/matterport3d/${scene}_dav2_r36_repro2_interrupted_epoch1_zero_shot" \
    --selection-audit data/provenance/matterport_bimnet_three_rule_v1.json \
    --evaluate-selected-only \
    --process-res 504 \
    --device cuda \
    --progress-every 100
  echo "zero_shot_scene_completed=$scene time=$(date -Ins)"
done

echo "evaluation_completed=$(date -Ins)"
