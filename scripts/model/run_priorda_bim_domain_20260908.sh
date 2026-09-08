#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/bgao491/PriorBIMDA
export PYTHONPATH=/home/bgao491/PriorBIMDA/src:/home/bgao491/PriorBIMDA/scripts/model
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=42
exec /home/bgao491/miniconda3/envs/priorbimda/bin/python -u \
  scripts/model/train_priorda_bim_domain.py \
  --config configs/stanford_area1_priorda_v11_bim_domain_6plus12epoch_20260908.yaml
