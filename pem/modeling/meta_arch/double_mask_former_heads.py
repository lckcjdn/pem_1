import torch
import torch.nn as nn
from typing import Dict
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.utils.registry import Registry
import fvcore.nn.weight_init as weight_init

from ..pixel_decoder.pem_pixel_decoder import build_pixel_decoder
from ..transformer_decoder.pem_transformer_decoder import MultiScaleMaskedTransformerDecoder

# 创建双分割头专用注册器
DOUBLE_SEM_SEG_HEADS_REGISTRY = Registry("DOUBLE_SEM_SEG_HEADS")

def build_sem_seg_head(cfg, input_shape, head_id=1):
    """
    根据head_id构建不同配置的语义分割头
    """
    name = cfg.MODEL.DOUBLE_MASK_FORMER[f"HEAD{head_id}"].NAME
    return DOUBLE_SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape, head_id)

@DOUBLE_SEM_SEG_HEADS_REGISTRY.register()
class DoubleMaskFormerHead(nn.Module):
    """
    用于双分割头架构的MaskFormer头
    可以为每个头部配置不同的参数
    """

    @configurable
    def __init__(
        self,
        *,
        cfg,
        input_shape: Dict[str, ShapeSpec],
        num_classes: int,
        mask_dim: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        enforce_input_proj: bool,
        norm: str = "BN",
        ignore_value: int = 255,
        loss_weight: float = 1.0,
        convs_dim: int = 128,
        # head specific
        head_id: int = 1,
        transformer_in_feature: str = "res5",
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.head_id = head_id
        self.num_classes = num_classes
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim  # 保存hidden_dim为实例变量
        self.deep_supervision = deep_supervision
        self.cfg = cfg  # 保存配置对象用于后续使用
        self.ignore_value = ignore_value  # 添加ignore_value参数
        self.loss_weight = loss_weight  # 添加loss_weight参数
        self.convs_dim = convs_dim  # 添加convs_dim参数
        
        # Pixel decoder (为每个头独立构建)
        self.pixel_decoder = build_pixel_decoder(cfg, input_shape)
        
        # Transformer decoder配置
        self.transformer_encoder_feature_name = transformer_in_feature
        
        # 直接创建MultiScaleMaskedTransformerDecoder实例
        transformer_decoder = MultiScaleMaskedTransformerDecoder(
            in_channels=mask_dim,
            mask_classification=True,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_proj
        )
        
        self.predictor = transformer_decoder

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec], head_id: int = 1):
        head_config = cfg.MODEL.DOUBLE_MASK_FORMER[f"HEAD{head_id}"]
        
        return {
            "cfg": cfg,  # 返回cfg参数供__init__方法使用
            "input_shape": input_shape,
            "num_classes": head_config.NUM_CLASSES,
            "mask_dim": head_config.MASK_DIM,
            "hidden_dim": head_config.HIDDEN_DIM, 
            "num_queries": head_config.NUM_OBJECT_QUERIES,
            "nheads": head_config.NHEADS,
            "dim_feedforward": head_config.DIM_FEEDFORWARD,
            "dec_layers": head_config.DEC_LAYERS,
            "pre_norm": head_config.PRE_NORM,
            "enforce_input_proj": head_config.ENFORCE_INPUT_PROJ,
            "norm": head_config.NORM,
            "ignore_value": getattr(head_config, "IGNORE_VALUE", 255),  # 添加ignore_value配置
            "loss_weight": getattr(head_config, "LOSS_WEIGHT", 1.0),  # 添加loss_weight配置
            "convs_dim": getattr(head_config, "CONVS_DIM", 128),  # 添加convs_dim配置
            "head_id": head_id,
            "transformer_in_feature": head_config.TRANSFORMER_IN_FEATURE,
            "deep_supervision": head_config.DEEP_SUPERVISION,
        }

    def forward(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        
        # 动态添加通道适配层解决通道不匹配问题
        channel_adapters = nn.ModuleList()
        # 获取特征图的数据类型，确保适配层使用相同类型
        if multi_scale_features:
            feature_dtype = multi_scale_features[0].dtype
        else:
            feature_dtype = torch.float32  # 默认为float32
        
        for i in range(len(multi_scale_features)):
            # 获取当前特征图的通道数
            current_channels = multi_scale_features[i].shape[1]
            # 如果通道数不匹配，添加1x1卷积进行调整
            if current_channels != self.hidden_dim:
                adapter = Conv2d(current_channels, self.hidden_dim, kernel_size=1)
                weight_init.c2_msra_fill(adapter)
                # 将适配器移至与特征图相同的设备和数据类型
                adapter = adapter.to(device=multi_scale_features[i].device, dtype=feature_dtype)
                channel_adapters.append(adapter)
            else:
                # 通道数匹配时使用恒等映射
                channel_adapters.append(nn.Identity())
        
        # 应用通道适配
        adapted_features = [channel_adapters[i](multi_scale_features[i]) 
                           for i in range(len(multi_scale_features))]
        
        # 传递适配后的特征给predictor，并包含mask参数
        outputs = self.predictor(
            adapted_features,
            mask_features,
            mask
        )
        
        return outputs