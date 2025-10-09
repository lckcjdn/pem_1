# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings('ignore')
except:
    warnings.filterwarnings('ignore')
    pass

import copy
import itertools
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# MaskFormer
from pem import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
)


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    @classmethod
    def save_inference_results(cls, cfg, model):
        """
        对输入文件夹中的图片进行语义分割并保存结果
        """
        import numpy as np
        import cv2
        import glob
        from tqdm import tqdm
        import os
        from detectron2.data.detection_utils import read_image
        from detectron2.data.transforms import ResizeShortestEdge
        from collections import namedtuple

        Label = namedtuple('Label', [
            'name', 'id', 'trainId', 'category', 'catId',
            'hasInstances', 'ignoreInEval', 'color'
        ])
        labels = [
            #           name          id  trainId      category     catId hasInstances ignoreInEval      color
            Label( 'background' ,     0 ,     0 ,       'void' ,      0 ,      False ,      False , (  0,   0,   0) ),
            Label(     's_w_d' ,      1 ,     1 ,   'dividing' ,      1 ,      False ,      False , ( 70, 130, 180) ),
            Label(     's_y_d' ,      2 ,     2 ,   'dividing' ,      1 ,      False ,      False , (220,  20,  60) ),
            Label(   'ds_y_dn' ,      3 ,     3 ,   'dividing' ,      1 ,      False ,      False , (255,   0,   0) ),
            Label(   'sb_w_do' ,      4 ,     4 ,   'dividing' ,      1 ,      False ,      False , (  0,   0,  60) ),
            Label(   'sb_y_do' ,      5 ,     5 ,   'dividing' ,      1 ,      False ,      False , (  0,  60, 100) ),
            Label(      'b_w_g' ,     6 ,     6 ,    'guiding' ,      2 ,      False ,      False , (  0,   0, 142) ),
            Label(      's_w_s' ,     7 ,     7 ,   'stopping' ,      3 ,      False ,      False , (220, 220,   0) ),
            Label(      's_w_c' ,     8 ,     8 ,    'chevron' ,      4 ,      False ,      False , (102, 102, 156) ),
            Label(      's_y_c' ,     9 ,     9 ,    'chevron' ,      4 ,      False ,      False , (128,   0,   0) ),
            Label(      's_w_p' ,    10 ,    10 ,    'parking' ,      5 ,      False ,      False , (128,  64, 128) ),
            Label(      'c_wy_z' ,   11 ,    11 ,      'zebra' ,      6 ,      False ,      False , (190, 153, 153) ),
            Label(       'a_w_u',    12 ,    12 ,  'thru/turn' ,      7 ,      False ,      False , (  0,   0, 230) ),
            Label(       'a_w_t',    13 ,    13 ,  'thru/turn' ,      7 ,      False ,      False , (128, 128,   0) ),
            Label(      'a_w_tl',    14 ,    14 ,  'thru/turn' ,      7 ,      False ,      False , (128,  78, 160) ),
            Label(      'a_w_tr',    15 ,    15 ,  'thru/turn' ,      7 ,      False ,      False , (150, 100, 100) ),
            Label(       'a_w_l',    16 ,    16 ,  'thru/turn' ,      7 ,      False ,      False , (180, 165, 180) ),
            Label(       'a_w_r',    17 ,    17 ,  'thru/turn' ,      7 ,      False ,      False , (107, 142,  35) ),
            Label(      'a_n_lu',    18 ,    18 ,  'thru/turn' ,      7 ,      False ,      False , (  0, 191, 255) ),
            Label(      'b_n_sr',    19 ,    19 ,  'reduction' ,      8 ,      False ,      False , (255, 128,   0) ),
            Label(     'd_wy_za',    20 ,    20 ,  'attention' ,      9 ,      False ,      False , (  0, 255, 255) ),
            Label(      'r_wy_np',   21 ,    21 , 'no parking' ,     10 ,      False ,      False , (178, 132, 190) ),
            Label(     'vom_wy_n',   22 ,    22 ,     'others' ,     11 ,      False ,      False , (128, 128,  64) ),
            Label(       'om_n_n',   23 ,    23 ,     'others' ,     11 ,      False ,      False , (102,   0, 204) ),
        ]

        color_map = np.zeros((256, 3), dtype=np.uint8)
        for label in labels:
            r, g, b = label.color
            color_map[label.trainId] = [b, g, r]
        
        logger = logging.getLogger("detectron2.trainer")
        logger.info("开始对输入图片进行语义分割...")
        
        model.eval()
        save_dir = os.path.join(cfg.OUTPUT_DIR, "segmentation")
        os.makedirs(save_dir, exist_ok=True)
        
        # input_dir = "/home/guitu/Data/ytj/data/AR_GOPRO/AR/"
        # input_dir = "AR"
        input_dir = "/home/guitu/Data/ytj/data/AR_GOPRO/AR/"

        if not os.path.exists(input_dir):
            logger.warning(f"输入目录 {input_dir} 不存在，请确保创建该目录并放入图片")
            return
            
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        logger.info(f"找到 {len(image_paths)} 张图片进行处理")
        
        aug = ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        
        with torch.no_grad():
            for image_path in tqdm(image_paths, desc="处理图片"):
                original_image = read_image(image_path, format="BGR")
                
                height, width = original_image.shape[:2]
                image = aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                
                inputs = {"image": image, "height": height, "width": width}
                outputs = model([inputs])[0]
 
                if "sem_seg" in outputs:
                    sem_seg = outputs["sem_seg"].argmax(dim=0).cpu().numpy()

                    if sem_seg.shape[0] != height or sem_seg.shape[1] != width:
                        sem_seg = cv2.resize(
                            sem_seg.astype(np.float32), 
                            (width, height), 
                            interpolation=cv2.INTER_NEAREST
                        ).astype(np.int64)

                    colored_seg = np.zeros((height, width, 3), dtype=np.uint8)
                    for train_id in range(len(labels)):
                        colored_seg[sem_seg == train_id] = color_map[train_id]
                    
                    blended = cv2.addWeighted(original_image, 0.3, colored_seg, 0.7, 0)
                    
                    h1, w1 = original_image.shape[:2]
                    h2, w2 = blended.shape[:2]
                    if w1 != w2:
                        blended = cv2.resize(blended, (w1, int(h2 * w1 / w2)))
                    
                    stacked_img = cv2.vconcat([original_image, blended])
                    
                    cv2.imwrite(
                        os.path.join(save_dir, f"{file_name}_result.png"), 
                        stacked_img
                    )
        
        logger.info(f"分割结果已保存到 {save_dir}")

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=False,
                    output_dir=output_folder,
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=False, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=False, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )       
        
        Trainer.save_inference_results(cfg, model)
        return {}
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.distributed.init_process_group(backend='nccl',init_method='tcp://localhost:12345', world_size=1, rank=0)
    comm.create_local_process_group(1)

    # main(args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url='auto',
        args=(args,),
    )
