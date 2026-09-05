#!/usr/bin/env bash
set -uo pipefail

project_root=/home/bgao491/PriorBIMDA
run_name=stanford_area1_dav2_early_fusion_scale_low36_only_12epoch_deterministic_repro2_interrupt_after_epoch1
run_output="$project_root/outputs/$run_name"
run_results="$project_root/results/stanford_area1/dav2_early_fusion_scale_low36_only_12epoch_deterministic_repro2_interrupt_after_epoch1"
run_log="$run_output/interrupt_run.log"
python_bin=/home/bgao491/miniconda3/envs/priorbimda/bin/python
config="configs/${run_name}.yaml"

mkdir -p "$run_output" "$run_results"
cd "$project_root" || exit 1

export PYTHONPATH="$project_root/src"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export NVIDIA_TF32_OVERRIDE=0
export CUDA_VISIBLE_DEVICES=0

echo "run_started=$(date -Ins)" >>"$run_log"
"$python_bin" scripts/model/train_dav2_joint_scale_low.py \
  --config "$config" \
  --device cuda \
  --deterministic >>"$run_log" 2>&1 &
train_pid=$!
echo "$train_pid" >"$run_output/train.pid"
echo "train_pid=$train_pid" >>"$run_log"

interrupted=0
while kill -0 "$train_pid" 2>/dev/null; do
  if [[ -s "$run_output/training_history.csv" \
        && -s "$run_output/latest.pt" \
        && -s "$run_output/best.pt" ]]; then
    completed_epochs=$(($(wc -l <"$run_output/training_history.csv") - 1))
    if (( completed_epochs >= 1 )); then
      interrupted=1
      echo "interrupt_requested_after_epoch=$completed_epochs time=$(date -Ins)" >>"$run_log"
      kill -INT "$train_pid" 2>/dev/null || true
      break
    fi
  fi
  sleep 2
done

if (( interrupted == 1 )); then
  for _ in $(seq 1 30); do
    kill -0 "$train_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$train_pid" 2>/dev/null; then
    echo "sigint_timeout_sending_sigterm=$(date -Ins)" >>"$run_log"
    kill -TERM "$train_pid" 2>/dev/null || true
  fi
fi

wait "$train_pid"
exit_code=$?
echo "train_exit_code=$exit_code time=$(date -Ins)" >>"$run_log"
echo "$exit_code" >"$run_output/train.exit_code"
exit 0
