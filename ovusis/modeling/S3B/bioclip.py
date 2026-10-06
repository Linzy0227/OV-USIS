"""
S3B with DepthAnything structure features.

This module extracts structure cues using a DepthAnything backbone (if available),
projects them to match the visual feature channels, and fuses with the visual feature
map via a lightweight residual/MLP fusion.
"""
import sys
sys.path.append('./')  # DepthAnythingV2 根目录
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from depth_anything_v2.dpt import DepthAnythingV2
import open_clip


class BioCLIPContextGenerator(nn.Module):
    def __init__(
        self,
        local_weight_path='pretrained/bioclip.bin',
        model_name='ViT-B-16',
        target_channels=256,
    ):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None
        )

        checkpoint = torch.load(local_weight_path, map_location='cpu')
        print(f"Loading BioCLIP weights from {local_weight_path}...")
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        self.visual = model.visual
        self.visual.eval()
        for p in self.visual.parameters():
            p.requires_grad = False

        self.embed_dim = self.visual.conv1.out_channels  # 768
        self.proj_layers = nn.Conv2d(self.embed_dim, target_channels, kernel_size=1, bias=False)
    
    
    def forward(self) -> torch.Tensor:
        """
        """
    
    def extract_biofeature(self, x_rgb, input_hw=(224, 224)):
        """
        Args:
            x_rgb: [B, 3, H, W]
        Returns:
            spatial_feat: [B, target_channels, H/16, W/16]
            cls_token:    [B, target_channels]
        """

        if input_hw is not None:
            x_rgb = F.interpolate(x_rgb, size=input_hw, mode='bilinear', align_corners=False)

        B, _, H, W = x_rgb.shape
        patch_h, patch_w = H // 16, W // 16
        with torch.no_grad():
            # ---- 1. patch embedding ----
            x = self.visual.conv1(x_rgb)                  # [B, 768, H/16, W/16]
            
            B, C, H, W = x.shape
            
            x = x.reshape(B, self.embed_dim, -1)
            x = x.permute(0, 2, 1)                         # [B, N, 768]

            # ---- 2. add CLS token ----
            cls_token = self.visual.class_embedding.unsqueeze(0).expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)           # [B, 1+N, 768]

            pos_embed = self.visual.positional_embedding # [N_orig, C]
            
            if pos_embed.shape[0] != x.shape[1]:
                pos_embed = pos_embed.unsqueeze(0) # [1, 197, 768]
                cls_pos = pos_embed[:, 0:1, :]
                grid_pos = pos_embed[:, 1:, :] # [1, 196, 768]
                
                orig_size = int(grid_pos.shape[1] ** 0.5)
                
                grid_pos = grid_pos.permute(0, 2, 1).reshape(1, C, orig_size, orig_size)
                
                grid_pos = F.interpolate(grid_pos, size=(H, W), mode='bicubic', align_corners=False)
                
                grid_pos = grid_pos.flatten(2).permute(0, 2, 1)
                
                pos_embed = torch.cat([cls_pos, grid_pos], dim=1)
                
            x = x + pos_embed

            # ---- 4. transformer ----
            x = self.visual.ln_pre(x)
            x = x.permute(1, 0, 2)                          # [L, B, C]
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)                          # [B, L, C]
            x = self.visual.ln_post(x)

            # ---- 5. split ----
            cls_token = x[:, 0]                             # [B, 768]
            patch_tokens = x[:, 1:]                         # [B, N, 768]

            spatial_feat = (
                patch_tokens
                .permute(0, 2, 1)
                .reshape(B, self.embed_dim, patch_h, patch_w)
            )
        spatial_feat = self.proj_layers(spatial_feat)      # [B, target_channels, h, w]

        return spatial_feat, cls_token



if __name__ == "__main__":
    model = BioCLIPContextGenerator()
    x = torch.randn(2, 3, 1024, 1024)
    depth_list, cls_token = model.extract_biofeature(x, input_hw=(512, 512))
    print(depth_list.shape, cls_token.shape)