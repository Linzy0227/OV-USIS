"""
This file may have been modified by Bytedance Ltd. and/or its affiliates ("Bytedance's Modifications").
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/meta_arch/mask_former_head.py
"""

from typing import Dict, Optional

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .DGIC.ovusis_transformer_decoder import build_transformer_decoder
from .S3B.msdeformattn import build_pixel_decoder
from .S3B.structure_extractor import StructureFeatureExtractor


@SEM_SEG_HEADS_REGISTRY.register()
class OVUSISHead(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        transformer_predictor: nn.Module,
        mask_dim:int,
        transformer_in_feature: str,
        S3B_cfg: Optional[Dict] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
            S3B_cfg: configuration for S3B module
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature
        
        self.num_classes = num_classes

        da_type = S3B_cfg.get("DA_TYPE", "vitl")

        if da_type == "vitl":
            struct_channels = [1024, 1024, 1024]   
        elif da_type == "vitb":
            struct_channels = [768, 768, 768]
        elif da_type == "vits":
            struct_channels = [384, 384, 384]

        self.proj_cls = nn.Conv2d(struct_channels[-1], mask_dim, kernel_size=1)
        self.structure_extractor = None
        if S3B_cfg is not None and S3B_cfg.get("ENABLED", False):
            self.structure_extractor = StructureFeatureExtractor(
                feat_channels=256,
                struct_channels=struct_channels,
                depthanything_model_type=da_type,
            )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            raise NotImplementedError

        S3B_cfg = {
            "ENABLED": hasattr(cfg.MODEL, 'S3B') and cfg.MODEL.S3B.ENABLED,
            "DA_TYPE": getattr(cfg.MODEL.S3B, 'DEPTHANYTHING_MODEL', 'vits') if hasattr(cfg.MODEL, 'S3B') else 'vits',
        }

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "mask_dim": cfg.MODEL.SEM_SEG_HEAD.MASK_DIM,
            "S3B_cfg": S3B_cfg,
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        
        cls_token = None
        if self.structure_extractor is not None and 'input_rgb' in features:
            structure_features, cls_token = self.structure_extractor.extract_structure_features(
                features['input_rgb'],
                input_hw=518,
            )
            if structure_features is not None:
                cls_token = self.proj_cls(cls_token)
                multi_scale_features = self.structure_extractor.inject_structure(
                    structure_features,
                    multi_scale_features,
                )
        
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, 
                                        mask_features, 
                                        cls_token,
                                        mask,
                                        text_classifier=features["text_classifier"], 
                                        num_templates=features["num_templates"],
                                        biofeat_map=features["bio_feat_map"],
                                        bio_cls_token=features["bio_cls_token"]
                                        )
        else:
            raise NotImplementedError
        return predictions
