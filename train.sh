#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1

CUDA_VISIBLE_DEVICES=0,1 python train_net.py \
  --num-gpus 2 \
  --config-file configs/convnext_large_ovusis_a.yaml
