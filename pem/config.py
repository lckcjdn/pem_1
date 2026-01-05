# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_double_maskformer_config(cfg):
    """
    Add config for DOUBLE_MASK_FORMER.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    # learning rate schedule
    cfg.SOLVER.BASE_LR_END = 0.0
    # learning rate scheduler
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    # warmup configuration
    cfg.SOLVER.WARMUP_FACTOR = 1.0
    cfg.SOLVER.WARMUP_ITERS = 0
    # gradient clipping
    cfg.SOLVER.CLIP_GRADIENTS = CN()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.01
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    # AMP configuration
    cfg.SOLVER.AMP = CN()
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.AMP.INITIAL_SCALE_POWER = 32.0
    # head-specific learning rate multipliers
    cfg.SOLVER.HEAD1_LR_MULTIPLIER = 1.0
    cfg.SOLVER.HEAD2_LR_MULTIPLIER = 1.0
    # gradient accumulation steps
    cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 1

    # double mask_former model config
    cfg.MODEL.DOUBLE_MASK_FORMER = CN()

    # PIXEL_DECODER
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER = CN()
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.NAME = "PEM_Pixel_Decoder"
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.PROJECT_CHANNELS = [48]
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.CONVS_DIM = 128
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.COMMON_STRIDE = 4
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.NORM = "BN"
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.TRANSFORMER_ENC_LAYERS = 6
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.ASPP_CHANNELS = 256
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.ASPP_DROPOUT = 0.1
    cfg.MODEL.DOUBLE_MASK_FORMER.PIXEL_DECODER.USE_DEPTHWISE_SEPARABLE_CONV = False

    # FUSION module
    cfg.MODEL.DOUBLE_MASK_FORMER.FUSION = CN()
    cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.METHOD = "weighted_sum"
    cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.HEAD1_WEIGHT = 0.6
    cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.HEAD2_WEIGHT = 0.4

    # HEAD1 configuration
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1 = CN()
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NAME = "DoubleMaskFormerHead"
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NUM_CLASSES = 16
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.LOSS_WEIGHT = 1.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CONVS_DIM = 128
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.MASK_DIM = 256
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.HIDDEN_DIM = 256
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NHEADS = 8
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.PRE_NORM = True
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.ENFORCE_INPUT_PROJ = True
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NORM = "BN"
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CLASS_WEIGHT = 2.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CLASS_WEIGHTS = None
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.MASK_WEIGHT = 5.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DICE_WEIGHT = 5.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.TRAIN_NUM_POINTS = 12544
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DEEP_SUPERVISION = True
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DEC_LAYERS = 7
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DROPOUT = 0.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.ENC_LAYERS = 0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.IGNORE_VALUE = 255

    # HEAD2 configuration - similar to HEAD1
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2 = CN()
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NAME = "DoubleMaskFormerHead"
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NUM_CLASSES = 9
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.LOSS_WEIGHT = 1.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CONVS_DIM = 128
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.MASK_DIM = 256
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.HIDDEN_DIM = 256
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NHEADS = 8
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.PRE_NORM = True
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.ENFORCE_INPUT_PROJ = True
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NORM = "BN"
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.TRANSFORMER_IN_FEATURE = "multi_scale_pixel_decoder"
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CLASS_WEIGHT = 2.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CLASS_WEIGHTS = None
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.MASK_WEIGHT = 5.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DICE_WEIGHT = 5.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.TRAIN_NUM_POINTS = 12544
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DEEP_SUPERVISION = True
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DEC_LAYERS = 7
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DROPOUT = 0.0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.ENC_LAYERS = 0
    cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.IGNORE_VALUE = 255

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.DOUBLE_MASK_FORMER.SIZE_DIVISIBILITY = 32

    # TEST configuration
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST = CN()
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.DOUBLE_MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.DOUBLE_MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.DOUBLE_MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.DOUBLE_MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # RESNETS default configuration
    if not hasattr(cfg.MODEL, "RESNETS"):
        cfg.MODEL.RESNETS = CN()
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STEM_TYPE = "basic"
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NORM = "BN"
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]
    
    # BACKBONE default configuration
    if not hasattr(cfg.MODEL, "BACKBONE"):
        cfg.MODEL.BACKBONE = CN()
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    
    # META_ARCHITECTURE
    cfg.MODEL.META_ARCHITECTURE = "DoubleMaskFormer"
    
    # 确保INPUT结构存在
    if not hasattr(cfg, "INPUT"):
        cfg.INPUT = CN()
    
    # 确保DATASETS结构存在
    if not hasattr(cfg, "DATASETS"):
        cfg.DATASETS = CN()
        cfg.DATASETS.TRAIN = []
        cfg.DATASETS.TEST = []
        cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
        cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
        cfg.DATASETS.PROPOSAL_FILES_TEST = []
        cfg.DATASETS.PROPOSAL_FILES_TRAIN = []
    
    # 确保TEST结构存在
    if not hasattr(cfg, "TEST"):
        cfg.TEST = CN()
    # TEST AMP configuration
    cfg.TEST.AMP = CN()
    cfg.TEST.AMP.ENABLED = True
    # TEST augmentation configuration
    cfg.TEST.AUG = CN()
    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.AUG.MIN_SIZES = [512, 768, 1024, 1280, 1536, 1792]
    cfg.TEST.AUG.MAX_SIZE = 4096
    cfg.TEST.AUG.FLIP = True
    # TEST evaluation period
    cfg.TEST.EVAL_PERIOD = 5000
    # Additional TEST configurations
    cfg.TEST.PRECOMPUTED_PROPOSAL_TOPK_TEST = 1000
    cfg.TEST.PRECOMPUTED_PROPOSAL_TOPK_TRAIN = 2000
    cfg.TEST.PROPOSAL_FILES_TEST = []
    cfg.TEST.PROPOSAL_FILES_TRAIN = []
    
    # 确保SOLVER结构存在
    if not hasattr(cfg, "SOLVER"):
        cfg.SOLVER = CN()
    
    # 确保DATALOADER结构存在
    if not hasattr(cfg, "DATALOADER"):
        cfg.DATALOADER = CN()
    # DATALOADER configurations
    cfg.DATALOADER.ASPECT_RATIO_GROUPING = True
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.DATALOADER.REPEAT_SQRT = True
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.0
    cfg.DATALOADER.SAMPLER_TRAIN = "TrainingSampler"
    cfg.DATALOADER.PIN_MEMORY = True
    
    # 确保SEGMENTATION结构存在
    if not hasattr(cfg, "SEGMENTATION"):
        cfg.SEGMENTATION = CN()
    # SEGMENTATION configurations
    cfg.SEGMENTATION.APOLLO_CLASS_WEIGHTS = []
    
    # 确保TRAINING结构存在
    if not hasattr(cfg, "TRAINING"):
        cfg.TRAINING = CN()
    # TRAINING configurations
    cfg.TRAINING.FREEZE_BACKBONE = False
    cfg.TRAINING.FREEZE_HEAD2 = False
    cfg.TRAINING.FREEZE_PIXEL_DECODER = False
    cfg.TRAINING.FREEZE_TRANSFORMER_DECODER = False
    cfg.TRAINING.TRAIN_HEAD1 = True
    cfg.TRAINING.TRAIN_HEAD2 = True
    
    # 确保OPTIMIZATION结构存在
    if not hasattr(cfg, "OPTIMIZATION"):
        cfg.OPTIMIZATION = CN()
    # OPTIMIZATION configurations
    cfg.OPTIMIZATION.GRADIENT_ACCUMULATION = CN()
    cfg.OPTIMIZATION.GRADIENT_ACCUMULATION.ENABLED = False
    cfg.OPTIMIZATION.GRADIENT_ACCUMULATION.STEPS = 1
    cfg.OPTIMIZATION.MEMORY_OPTIMIZATION = CN()
    cfg.OPTIMIZATION.MEMORY_OPTIMIZATION.ENABLED = False
    cfg.OPTIMIZATION.MEMORY_OPTIMIZATION.MAX_SPLIT_SIZE_MB = 128
    
    # OUTPUT directory
    cfg.OUTPUT_DIR = "output/default"
    
    # VERSION
    cfg.VERSION = 2
    
    # 确保INPUT结构存在
    if not hasattr(cfg, "INPUT"):
        cfg.INPUT = CN()
    # INPUT configurations
    cfg.INPUT.MIN_SIZE_TRAIN = [512, 614, 716, 819, 921, 1024, 1126, 1228, 1331, 1433, 1536, 1638, 1740, 1843, 1945, 2048]
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = "choice"
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TRAIN = 4096
    cfg.INPUT.MAX_SIZE_TEST = 2048
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.TYPE = "absolute"
    cfg.INPUT.CROP.SIZE = (512, 1024)
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.COLOR_AUG_SSD = True
    cfg.INPUT.SIZE_DIVISIBILITY = -1
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.DATASET_MAPPER_NAME = "double_mask_former_semantic"
    cfg.INPUT.RANDOM_FLIP = "horizontal"


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # stdc backbone
    cfg.MODEL.STDC = CN()
    cfg.MODEL.STDC.LAYERS = [2, 2, 2]
    cfg.MODEL.STDC.BLOCK_NUM = 4
    cfg.MODEL.STDC.BLOCK_TIPE = "cat"
    cfg.MODEL.STDC.USE_CONV_LAST = False
    cfg.MODEL.STDC.NORM = "BN"
    cfg.MODEL.STDC.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
