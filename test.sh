#!/usr/bin/env sh
cd "$(dirname "$0")" || exit 1

CUDA_VISIBLE_DEVICES=1 python -X faulthandler -u train_net.py \
  --num-gpus 1 \
  --eval-only \
  --config-file configs/convnext_large_ovusis_a.yaml \
  MODEL.WEIGHTS output/convnext_l_a/model_final.pth
