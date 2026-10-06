#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

mkdir -p pretrained

wget -O pretrained/convnext_b.bin \
  https://huggingface.co/laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K-augreg/resolve/main/open_clip_pytorch_model.bin

wget -O pretrained/convnext_l.bin \
  https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/main/open_clip_pytorch_model.bin

wget -O pretrained/depth_anything_v2_vits.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth

wget -O pretrained/depth_anything_v2_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

wget -O pretrained/depth_anything_v2_vitl.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

wget -O pretrained/bioclip.bin \
  https://huggingface.co/imageomics/bioclip/resolve/main/open_clip_pytorch_model.bin
