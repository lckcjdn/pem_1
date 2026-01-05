# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances

from .mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

__all__ = ["DoubleMaskFormerSemanticDatasetMapper"]


class DoubleMaskFormerSemanticDatasetMapper(MaskFormerSemanticDatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DoubleMaskFormer for semantic segmentation.
    
    This mapper extends MaskFormerSemanticDatasetMapper by adding dataset_id
    support for handling multiple datasets with independent supervision signals.
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        is_evaluation=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
            is_evaluation: whether to enable special evaluation mode with label comparison
        """
        # Call parent constructor
        super().__init__(
            is_train=is_train,
            augmentations=augmentations,
            image_format=image_format,
            ignore_label=ignore_label,
            size_divisibility=size_divisibility,
        )
        
        # Add is_evaluation flag
        self.is_evaluation = is_evaluation
        
        logger = logging.getLogger(__name__)
        logger.info(f"[{self.__class__.__name__}] is_evaluation: {is_evaluation}")

    @classmethod
    def from_config(cls, cfg, is_train=True, is_evaluation=False):
        # Build configuration similar to parent class
        ret = super().from_config(cfg, is_train)
        # Add is_evaluation to the configuration
        ret["is_evaluation"] = is_evaluation
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept, with added dataset_id and is_evaluation flag
        """
        # Make a copy to avoid modifying the original
        dataset_dict_copy = dataset_dict.copy()
        
        # Extract dataset_id from the input dict if present
        dataset_id = dataset_dict_copy.pop("dataset_id", 0)  # Default to 0 if not present
        
        # Get the processed dataset_dict from parent class
        result = super().__call__(dataset_dict_copy)
        
        # Add dataset_id to the result
        result["dataset_id"] = dataset_id
        
        # Add is_evaluation flag if enabled
        if self.is_evaluation:
            result["is_evaluation"] = True
        
        return result