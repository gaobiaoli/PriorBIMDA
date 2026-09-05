#!/usr/bin/env bash
set -euo pipefail

project_root=/home/bgao491/PriorBIMDA
run_name=stanford_area1_f36_anchor_1c07d65_seed42_batch16_retrain
run_output="$project_root/outputs/$run_name"
pipeline_log="$run_output/pipeline.log"
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
config="configs/${run_name}.yaml"
checkpoint="outputs/${run_name}/best.pt"

mkdir -p "$run_output"
exec >>"$pipeline_log" 2>&1

cd "$project_root"
export PYTHONPATH="$project_root/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0

echo "pipeline_started=$(date -Ins)"
echo "random_seed=42 deterministic_algorithms=false micro_batch=16 gradient_accumulation=1 effective_batch=16"
"$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$config" \
  --device cuda

echo "training_and_test_completed=$(date -Ins)"
for scene in hxp 759 1px; do
  echo "zero_shot_scene_started=$scene time=$(date -Ins)"
  "$python_bin" scripts/model/evaluate_matterport_bimnet_scale_refiner.py \
    --matterport-root /mnt/priorbimda-data/PriorBIMDA-Datasets/Matterport3D \
    --bimnet-root /home/bgao491/BIMNet_release \
    --toolkit-root /home/bgao491/S3-SAM3D-ToolKit \
    --bimnet-scene "$scene" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "results/matterport3d/${scene}_f36_anchor_1c07d65_seed42_batch16_retrain_zero_shot" \
    --selection-audit data/provenance/matterport_bimnet_three_rule_v1.json \
    --evaluate-selected-only \
    --process-res 504 \
    --device cuda \
    --progress-every 100
  echo "zero_shot_scene_completed=$scene time=$(date -Ins)"
done

echo "pipeline_completed=$(date -Ins)"
