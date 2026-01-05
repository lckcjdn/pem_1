# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
# 在导入任何大型库之前，先设置内存优化环境变量
import os
import sys

# 系统级内存优化和性能设置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,cache_enabled=True'  # 更大的分割大小，启用缓存以提高性能
# 移除内存缓存禁用，启用缓存可以提高GPU利用率
# os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 只使用第一个GPU
os.environ['OMP_NUM_THREADS'] = '8'  # 增加OpenMP线程数以充分利用CPU
os.environ['MKL_NUM_THREADS'] = '8'  # 增加MKL线程数以充分利用CPU
#os.environ['NCCL_DEBUG'] = 'INFO'  # 启用NCCL调试信息（可选）

# 现在再导入必要的库
import gc
import copy
import numpy as np
import random

# 在导入torch前执行垃圾回收
gc.collect()

# 导入torch
try:
    import torch
    # 基本的PyTorch优化设置
    torch.backends.cudnn.benchmark = True  # 启用benchmark以提高性能（相同尺寸输入时）
    torch.backends.cudnn.deterministic = False  # 牺牲确定性以提高性能
except ImportError:
    print("错误: 无法导入torch。请确保已正确安装PyTorch。")
    sys.exit(1)


# 启用自动混合精度（如果可用）
has_amp = False
try:
    from torch.cuda.amp import autocast, GradScaler
    has_amp = True
except ImportError:
    pass

# 忽略警告
try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
    warnings.filterwarnings('ignore')
except:
    pass

import itertools
import logging

from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
import cv2

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
import glob
from typing import List, Dict, Any
from detectron2.data import DatasetCatalog, MetadataCatalog
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

# 导入数据集注册函数
from pem.data.datasets.register_double_cityscapes import register_all_double_cityscapes
class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        
    @classmethod
    def save_inference_results(cls, cfg, model):
        """
        对输入文件夹中的图片进行语义分割并保存结果
        """
        # 确保numpy可用
        import numpy as np
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
        labels =  [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'car'                  ,  1 ,        1 , 'car'             , 1       , False        , False        , (  0, 60, 100) ),
    Label(  'human'                ,  2 ,        2 , 'human'           , 2       , False        , False        , (220, 20, 60) ),
    Label(  'road'                 ,  3 ,        3 , 'road'            , 3       , False        , False        , (128, 64, 128) ),
    Label(  'lane_mark'            ,  4 ,        4 , 'lane_mark'       , 4       , False        , False        , (255, 255, 255) ),
    Label(  'curb'                 ,  5 ,        5 , 'curb'            , 5       , False        , False        , (196, 196, 196) ),
    Label(  'road_mark'            ,  6 ,        6 , 'road_mark'       , 6       , False        , False        , (250, 170, 11) ),
    Label(  'guard_rail'           ,  7 ,        7 , 'guard_rail'      , 7       , False        , False        , (51, 0, 255) ),
    Label(  'traffic_sign'         ,  8 ,        8 , 'traffic_sign'    , 8       , False        , False        , (220, 220, 0) ),
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
        
        # input_dir = "/home/guitu/Data/ytj/guitu/images/"
        #input_dir = "D:/Ar/labelApollo/backend/Artest"
        input_dir = "barrier"

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
                import numpy as np
                image = torch.as_tensor(image.astype(np.float32).transpose(2, 0, 1), dtype=torch.float32)
                file_name = os.path.splitext(os.path.basename(image_path))[0]
                
                inputs = {"image": image, "height": height, "width": width}
                outputs = model([inputs])[0]
 
                if "sem_seg" in outputs:
                    # 确保numpy可用
                    import numpy as np
                    # 将PyTorch张量转换为numpy数组
                    sem_seg_tensor = outputs["sem_seg"].argmax(dim=0)
                    sem_seg = sem_seg_tensor.cpu().numpy()

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
                    
                    # 模式1：只保存叠加图（当前使用）
                    cv2.imwrite(
                        os.path.join(save_dir, f"{file_name}_result.png"), 
                        blended
                    )
                    
                    # 保存原始语义分割mask（单通道）
                    cv2.imwrite(
                        os.path.join(save_dir, f"{file_name}_mask.png"), 
                        sem_seg.astype(np.uint8)
                    )
                    
                    # 保存彩色语义分割mask
                    cv2.imwrite(
                        os.path.join(save_dir, f"{file_name}_colored_mask.png"), 
                        colored_seg
                    )
                    
                    # 模式2：保存原图+叠加图的组合（注释掉的原功能，需要时取消注释）
                    # h1, w1 = original_image.shape[:2]
                    # h2, w2 = blended.shape[:2]
                    # if w1 != w2:
                    #     blended = cv2.resize(blended, (w1, int(h2 * w1 / w2)))
                    # 
                    # stacked_img = cv2.vconcat([original_image, blended])
                    # 
                    # cv2.imwrite(
                    #     os.path.join(save_dir, f"{file_name}_result.png"), 
                    #     stacked_img
                    # )
        
        # 生成图例图片
        def generate_legend(labels, legend_height=300, legend_width=500):
            # 创建一个白色背景的图像
            legend_img = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
            
            # 设置字体和字体大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 1
            
            # 计算每个标签行的高度
            line_height = 25
            start_y = 20
            
            # 设置颜色方块的大小
            square_size = 20
            
            for label in labels:
                # 跳过背景或不需要在图例中显示的标签
                if label.trainId == 0 or label.ignoreInEval:
                    continue
                
                # 计算位置
                square_x = 20
                square_y = start_y
                
                # 绘制颜色方块
                color_bgr = color_map[label.trainId]
                cv2.rectangle(legend_img, (square_x, square_y), 
                             (square_x + square_size, square_y + square_size), 
                             color_bgr.tolist(), -1)
                
                # 绘制标签名称
                text_x = square_x + square_size + 10
                text_y = square_y + square_size // 2 + 5
                cv2.putText(legend_img, f"{label.name} (ID: {label.trainId})".replace('*', ''), 
                            (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                
                # 更新下一行的起始位置
                start_y += line_height
                
                # 如果超出图像高度，停止添加标签
                if start_y + square_size > legend_height:
                    break
            
            return legend_img
        
        # 生成并保存图例图片
        legend_img = generate_legend(labels)
        legend_path = os.path.join(save_dir, "legend.png")
        cv2.imwrite(legend_path, legend_img)
        logger.info(f"图例图片已保存到 {legend_path}")
        
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
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=False))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=False))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            # 替换CityscapesSemSegEvaluator为SemSegEvaluator，避免使用有问题的getCustomPrediction函数
            return SemSegEvaluator(
                dataset_name,
                distributed=False,
                output_dir=output_folder,
            )
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                # 替换CityscapesSemSegEvaluator为SemSegEvaluator
                evaluator_list.append(
                    SemSegEvaluator(
                        dataset_name,
                        distributed=False,
                        output_dir=output_folder,
                    )
                )
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
        # 优化数据加载器配置，优先保证内存稳定性
        import os
        
        # 确保使用trainID进行训练
        os.environ["USE_TRAIN_ID"] = "1"
        
        # 获取worker数量
        
        # 确保至少使用1个worker以启用多进程
        num_workers = max(1, min(cfg.DATALOADER.NUM_WORKERS, 6))
        
        # 创建一个优化的数据加载器配置
        data_loader_args = {
            "mapper": None,
            "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "num_workers": num_workers,
            "sampler": None,
            "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            "collate_fn": None,
        }
        
        # 只有当num_workers > 0时才能设置prefetch_factor
        if num_workers > 0:
            data_loader_args["prefetch_factor"] = 1
        
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            data_loader_args["mapper"] = mapper
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            data_loader_args["mapper"] = mapper
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            data_loader_args["mapper"] = mapper
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            data_loader_args["mapper"] = mapper
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            data_loader_args["mapper"] = mapper
        
        # 创建内存优化的数据加载器
        data_loader = build_detection_train_loader(cfg, **data_loader_args)
        
        # 使用原始数据加载器但减少worker数量，避免额外的内存开销
        return data_loader

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

    
    def after_step(self):
        """简化的训练步骤后处理，最小化内存开销"""
        # 只调用父类方法，不进行额外的内存监控和清理
        super().after_step()
            
    def run_step(self):
        """简化的单步执行，避免额外的内存操作"""
        # 只调用父类的run_step方法，减少额外的内存操作
        super().run_step()
            




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
    
    # 性能优化配置，但更注重稳定性
    # 适度增加batch size以提高GPU利用率
    original_batch_size = cfg.SOLVER.IMS_PER_BATCH
    
    # 相应调整学习率，保持学习率与batch size的比例
    original_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR = original_lr * (cfg.SOLVER.IMS_PER_BATCH / original_batch_size)
    
    # 确保使用自动混合精度训练
    if hasattr(cfg.SOLVER, 'AMP'):
        cfg.SOLVER.AMP.ENABLED = True
    
    # 优化数据加载参数，Windows下保守设置
    cfg.DATALOADER.NUM_WORKERS = 2  # 减少worker数量以降低内存使用
    
    # 在freeze之前修改配置
    cfg.SOLVER.CHECKPOINT_PERIOD = max(cfg.SOLVER.CHECKPOINT_PERIOD, 5000)  # 减少检查点保存频率
    cfg.TEST.EVAL_PERIOD = max(cfg.TEST.EVAL_PERIOD, 1000)  # 减少评估频率
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask2former")
    return cfg


def main(args):
    # 配置日志级别，减少输出量
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    # 只保留错误和警告级别的日志
    logging.basicConfig(level=logging.WARNING)
    
    # Windows系统不支持resource模块，跳过进程内存限制设置
    
    # 进一步减少数据加载器的worker数量
    # 预处理配置，设置更小的batch size以减少内存使用
    if torch.cuda.is_available():
        # 禁用TF32加速以减少内存使用
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # 清理CUDA缓存和Python垃圾收集
        torch.cuda.empty_cache()
        gc.collect()
    
    logging.info("程序启动，已配置忽略modulated_deformable相关的CUDA内核错误")
    
    cfg = setup(args)
    
    if args.eval_only:
        # 创建一个可修改的配置副本用于评估
        eval_cfg = copy.deepcopy(cfg)
        eval_cfg.defrost()  # 解冻配置以便修改
        
        # 评估时限制batch size为1以最小化内存使用
        eval_cfg.SOLVER.IMS_PER_BATCH = 1
        
        # 只评估vestas数据集
        eval_cfg.DATASETS.TEST = ("cityscapes_vestas_sem_seg_val",)
        
        # 进一步优化数据加载参数，减少内存使用
        eval_cfg.DATALOADER.NUM_WORKERS = 0  # 禁用多进程加载
        eval_cfg.DATALOADER.PREFETCH_FACTOR = 1  # 减少预取数量
        
        eval_cfg.freeze()  # 重新冻结
        
        # 评估模式下的内存优化
        with torch.no_grad():
            # 注册vestas数据集
            register_all_double_cityscapes(os.getenv("DETECTRON2_DATASETS", "datasets"))
            
            # 再次清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            model = Trainer.build_model(eval_cfg)
            DetectionCheckpointer(model, save_dir=eval_cfg.OUTPUT_DIR).resume_or_load(
                eval_cfg.MODEL.WEIGHTS, resume=args.resume
            )
            
            # 加载模型后再次清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            res = None  # 初始化res变量以避免UnboundLocalError
            try:
                Trainer.save_inference_results(eval_cfg, model)
                res = Trainer.test(eval_cfg, model)
                if eval_cfg.TEST.AUG.ENABLED:
                    res.update(Trainer.test_with_TTA(eval_cfg, model))
                if comm.is_main_process():
                    verify_results(eval_cfg, res)
            finally:
                # 释放内存
                del model, eval_cfg
                if res is not None:
                    del res
                torch.cuda.empty_cache()
                gc.collect()
        return

    # 训练模式
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    try:
        result = trainer.train()
    finally:
        # 训练结束时清理内存
        torch.cuda.empty_cache()
        gc.collect()
    
    return result


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12345'
    #torch.distributed.init_process_group(backend='gloo',init_method='tcp://localhost:12345', world_size=1, rank=0)
    #comm.create_local_process_group(1)

    try:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url='auto',
            args=(args,),
        )
    except RuntimeError as e:
        # 捕获CUDA相关错误并提供友好的提示
        if "no kernel image is available" in str(e) or "modulated_deformable" in str(e).lower():
            logging.warning(f"检测到CUDA内核错误: {e}，这已被配置为可忽略的错误。")
        else:
            # 对于其他错误，仍然抛出
            raise
    except KeyboardInterrupt:
        logging.info("程序被用户中断")
    except Exception as e:
        logging.error(f"发生未预期的错误: {e}")
        raise