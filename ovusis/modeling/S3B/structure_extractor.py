"""S3B structure feature extraction and injection."""

import sys
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

sys.path.append("./")

from depth_anything_v2.dpt import DepthAnythingV2


def _build_depthanything(model_type: str = "vits") -> Optional[nn.Module]:
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    model = DepthAnythingV2(**model_configs[model_type])
    model.load_state_dict(torch.load(f"pretrained/depth_anything_v2_{model_type}.pth", map_location="cpu"))
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class SaliencyStructureFusion(nn.Module):
    """Saliency bootstrapping followed by structure injection."""

    def __init__(
        self,
        rgb_dim: int = 256,
        structure_dim: int = 512,
        drop: float = 0.0,
        eps: float = 1e-6,
        learnable_tau: bool = True,
        init_tau: float = 1.0,
        detach_assign: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.detach_assign = detach_assign

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(float(init_tau)))
        else:
            self.register_buffer("tau", torch.tensor(float(init_tau)), persistent=False)

        self.proj_rgb = nn.Conv2d(rgb_dim, rgb_dim, kernel_size=1)
        self.proj_structure = nn.Sequential(
            nn.Conv2d(structure_dim, rgb_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(rgb_dim, rgb_dim, kernel_size=3, padding=1),
        )
        self.structure_gate = nn.Conv2d(rgb_dim, 1, kernel_size=1, bias=True)
        self.refine = nn.Sequential(
            nn.Conv2d(rgb_dim, rgb_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(drop) if drop > 0 else nn.Identity(),
            nn.Conv2d(rgb_dim, rgb_dim, kernel_size=1),
        )

    def _get_prototype(self, x: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        assignment = assignment.view(batch, channels, -1)
        x_flat = x.view(batch, channels, -1)
        return torch.bmm(assignment, x_flat.transpose(1, 2))

    def _get_correlation_map(self, x: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        prototype = prototype / prototype.norm(dim=2, keepdim=True).clamp_min(self.eps)
        x_flat = x.view(batch, channels, -1)
        x_flat = x_flat / x_flat.norm(dim=1, keepdim=True).clamp_min(self.eps)
        return torch.bmm(prototype, x_flat).view(batch, channels, height, width)

    def _saliency_bootstrap(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.size()
        logits = x.view(batch, channels, -1)
        tau = F.softplus(self.tau) + self.eps
        assignment_logits = logits.detach() if self.detach_assign else logits
        assignment = F.softmax(assignment_logits / tau, dim=2).view(batch, channels, height, width)
        prototype = self._get_prototype(x, assignment)
        correlation = self._get_correlation_map(x, prototype)
        saliency = torch.sigmoid(correlation.mean(dim=1, keepdim=True))
        return x * saliency

    def forward(self, rgb_feat: torch.Tensor, structure_feat: torch.Tensor) -> torch.Tensor:
        rgb_feat = self.proj_rgb(rgb_feat)
        structure_feat = self.proj_structure(structure_feat)

        if structure_feat.shape[2:] != rgb_feat.shape[2:]:
            structure_feat = F.interpolate(
                structure_feat,
                size=rgb_feat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        bootstrapped_rgb = self._saliency_bootstrap(rgb_feat)
        gate = torch.sigmoid(self.structure_gate(bootstrapped_rgb + structure_feat))
        return rgb_feat + self.refine(bootstrapped_rgb + gate * structure_feat)


class StructureFeatureExtractor(nn.Module):
    def __init__(
        self,
        feat_channels: int,
        struct_channels: list[int],
        depthanything_model_type: str = "vits",
    ) -> None:
        super().__init__()
        self.depth_model = _build_depthanything(depthanything_model_type)
        self.fusion_blocks = nn.ModuleList(
            [
                SaliencyStructureFusion(rgb_dim=feat_channels, structure_dim=structure_channels)
                for structure_channels in struct_channels
            ]
        )

    def forward_features_extra(self, x: torch.Tensor):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        out_features = self.depth_model.pretrained.get_intermediate_layers(
            x,
            self.depth_model.intermediate_layer_idx[self.depth_model.encoder],
            return_class_token=True,
        )

        out_feats = []
        cls_token_map = None
        for token, cls_token in out_features:
            cls_token = cls_token.unsqueeze(1).expand_as(token)
            token = token.permute(0, 2, 1).reshape((token.shape[0], token.shape[-1], patch_h, patch_w))
            cls_token_map = cls_token.permute(0, 2, 1).reshape(
                (cls_token.shape[0], cls_token.shape[-1], patch_h, patch_w)
            )
            out_feats.append(token)
        return out_feats, cls_token_map

    @torch.no_grad()
    def extract_structure_features(self, x_rgb: torch.Tensor, input_hw: int = None):
        if self.depth_model is None:
            return None, None
        if input_hw is not None:
            x_rgb = F.interpolate(x_rgb, size=input_hw, mode="bilinear", align_corners=False)
        structure_features, cls_token = self.forward_features_extra(x_rgb)
        return structure_features[1:], cls_token

    def inject_structure(self, structure_features, visual_features):
        return [
            self.fusion_blocks[i](visual_features[i], structure_features[i])
            for i in range(len(structure_features))
        ]
