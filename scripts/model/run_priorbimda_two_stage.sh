#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/bgao491/miniconda3/envs/priorbimda/bin/python}"
DEVICE="${DEVICE:-cuda}"

cd "${PROJECT_ROOT}"

stage1_checkpoint=outputs/stanford_area1_priorbimda_stage1_fixed_attention_huber_3epoch/best.pt
if [[ ! -f "${stage1_checkpoint}" ]]; then
  "${PYTHON_BIN}" scripts/model/train.py \
    --config configs/stanford_area1_priorbimda_stage1_fixed_attention_huber_3epoch.yaml \
    --device "${DEVICE}"
fi

test -f "${stage1_checkpoint}"

"${PYTHON_BIN}" scripts/model/train_priorbimda_two_stage.py \
  --config configs/stanford_area1_priorbimda_stage2_dav2_metric_minmax_augmented.yaml \
  --device "${DEVICE}"
