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

from pem.data.datasets.labels_double import get_label_mappings


class DoubleMaskFormerSemanticDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.
    
    This version supports double datasets by adding dataset_id and dataset_type identifiers.
    0 indicates the first dataset (Apollo), 1 indicates the second dataset (Vestas).
    
    The callable currently does the following:
    
    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    5. Add dataset identifier information
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
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.logger = logging.getLogger(__name__)
        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        
        # 配置两个数据集的标识
        self.dataset_1_name = "cityscapes_apollo"
        self.dataset_2_name = "cityscapes_vestas"

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")
        
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation based on mode
        if is_train:
            # Training mode augmentations
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                        cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    )
                )
            if cfg.INPUT.COLOR_AUG_SSD:
                augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT))
            augs.append(T.RandomFlip())
        else:
            # Evaluation mode augmentations - no randomness
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TEST,
                    cfg.INPUT.MAX_SIZE_TEST,
                    "choice",  # Use deterministic resize
                )
            ]
        
        # Get metadata from appropriate dataset based on mode
        if is_train and hasattr(cfg.DATASETS, "TRAIN") and len(cfg.DATASETS.TRAIN) > 0:
            dataset_names = cfg.DATASETS.TRAIN
        elif hasattr(cfg.DATASETS, "TEST") and len(cfg.DATASETS.TEST) > 0:
            dataset_names = cfg.DATASETS.TEST
        else:
            # Fallback to training set if no appropriate dataset is found
            dataset_names = getattr(cfg.DATASETS, "TRAIN", [])
            if not dataset_names:
                raise ValueError("No datasets available for configuration")
                
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        
        # 确定数据集ID和类型
        # 优先通过dataset_name判断数据集来源
        dataset_id = 0  # 默认使用Apollo数据集
        dataset_type = "apollo"
        
        if hasattr(dataset_dict, "dataset_name"):
            if self.dataset_2_name in dataset_dict.dataset_name:
                dataset_id = 1
                dataset_type = "vestas"
        # 其次通过file_name判断
        elif "file_name" in dataset_dict:
            file_path = dataset_dict["file_name"]
            if "vestas" in file_path.lower():
                dataset_id = 1  # Vestas数据集
                dataset_type = "vestas"
        
        # 设置数据集标识
        dataset_dict["dataset_id"] = dataset_id
        dataset_dict["dataset_type"] = dataset_type

        # 读取图像
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(dataset_dict, image)
        except Exception as e:
            self.logger.error(f"Error reading image {dataset_dict['file_name']} from {dataset_type} dataset: {e}")
            # 创建一个空图像作为占位符，避免训练中断
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            dataset_dict["height"] = 512
            dataset_dict["width"] = 512

        if "sem_seg_file_name" in dataset_dict:
            # Use float32 instead of float64 to reduce memory usage
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("float32")
            
            # 应用数据集特定的标签映射
            try:
                label_mappings = get_label_mappings(f"cityscapes_{dataset_type}")
                id_to_train_id = label_mappings.get('id_to_train_id', {})
                
                # 应用id到trainId的映射
                if id_to_train_id:
                    # 创建映射数组
                    max_id = max(id_to_train_id.keys()) if id_to_train_id else 0
                    mapping_array = np.zeros(max_id + 1, dtype=np.int32)
                    
                    # 填充映射数组
                    for original_id, train_id in id_to_train_id.items():
                        mapping_array[original_id] = train_id
                    
                    # 应用映射，对于超出范围的id保持不变
                    sem_seg_gt_mapped = np.zeros_like(sem_seg_gt)
                    valid_mask = sem_seg_gt <= max_id
                    sem_seg_gt_mapped[valid_mask] = mapping_array[sem_seg_gt[valid_mask].astype(int)]
                    sem_seg_gt_mapped[~valid_mask] = sem_seg_gt[~valid_mask]  # 对于超出范围的id保持不变
                    sem_seg_gt = sem_seg_gt_mapped
            except Exception as e:
                self.logger.error(f"Error applying label mapping for {dataset_type} dataset: {e}")
        else:
            sem_seg_gt = None

        if sem_seg_gt is None:
            raise ValueError(
                "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {} from {} dataset.".format(
                    dataset_dict["file_name"], dataset_type
                )
            )

        # 应用数据增强变换
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        aug_input, transforms = T.apply_transform_gens(self.tfm_gens, aug_input)
        image = aug_input.image
        sem_seg_gt = aug_input.sem_seg

        # 转换为张量
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

        # 填充图像和分割标签到size_divisibility的整数倍
        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if sem_seg_gt is not None:
                sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

        image_shape = (image.shape[-2], image.shape[-1])  # h, w

        # 将数据添加到dataset_dict
        dataset_dict["image"] = image
        
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            del dataset_dict["annotations"]

        # 准备每类的二值掩码
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # 移除被忽略的区域
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # 某些图像可能没有标注（全部被忽略）
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict