# Copyright (c) Facebook, Inc. and its affiliates.
from .mask_former_instance_dataset_mapper import MaskFormerInstanceDatasetMapper
from .mask_former_panoptic_dataset_mapper import MaskFormerPanopticDatasetMapper
from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper
from .double_mask_former_semantic_dataset_mapper import DoubleMaskFormerSemanticDatasetMapper

__all__ = [
    "MaskFormerInstanceDatasetMapper",
    "MaskFormerPanopticDatasetMapper",
    "MaskFormerSemanticDatasetMapper",
    "DoubleMaskFormerSemanticDatasetMapper",
]