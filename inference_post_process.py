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
        import math
        from skimage.morphology import skeletonize

        def split_and_filter(mask_bin,
                                   win=31, ori_tol_deg=20,
                                   aspect_thresh=6, min_pixels=150):
            h, w = mask_bin.shape
            half = win // 2
            tol = math.radians(ori_tol_deg)

            # 1) 骨架化
            skel = skeletonize((mask_bin>0)).astype(np.uint8)
            # cv2.imwrite(os.path.join(save_dir, "skel.png"),skel * 255)

            # 2) 缓存方向
            ori_cache = {}
            def local_ori(y, x):
                # 只在骨架上取样
                if (y, x) in ori_cache:
                    return ori_cache[(y, x)]
                y0, y1 = max(0, y-half), min(h, y+half+1)
                x0, x1 = max(0, x-half), min(w, x+half+1)
                ys, xs = np.nonzero(skel[y0:y1, x0:x1])
                if len(xs) < 5:
                    ori_cache[(y, x)] = None
                    return None
                xs = xs + x0;  ys = ys + y0
                pts = np.column_stack((xs, ys)).astype(np.float32)
                cov = np.cov((pts - pts.mean(0)).T)
                eig_vals, eig_vecs = np.linalg.eig(cov)
                v = eig_vecs[:, eig_vals.argmax()]
                theta = math.atan2(v[1], v[0]) % math.pi
                ori_cache[(y, x)] = theta
                return theta

            def ang_diff(a, b):
                d = abs(a - b) % math.pi
                return d if d < math.pi/2 else math.pi - d

            # 3) 区域生长
            visited = np.zeros_like(mask_bin, np.uint8)
            labels  = np.zeros((h, w), np.int32)
            label_id = 0

            for y in range(h):
                for x in range(w):
                    if mask_bin[y, x] == 0 or visited[y, x]:
                        continue
                    theta0 = local_ori(y, x)
                    if theta0 is None:
                        visited[y, x] = 1
                        continue

                    label_id += 1
                    stk = [(y, x)]
                    visited[y, x] = 1
                    labels[y, x]  = label_id
                    while stk:
                        cy, cx = stk.pop()
                        for ny in range(max(0, cy-1), min(h, cy+2)):
                            for nx in range(max(0, cx-1), min(w, cx+2)):
                                if mask_bin[ny, nx] and not visited[ny, nx]:
                                    t = local_ori(ny, nx)
                                    if t is None:
                                        visited[ny, nx] = 1
                                        continue
                                    if ang_diff(t, theta0) < tol:
                                        visited[ny, nx] = 1
                                        labels[ny, nx]  = label_id
                                        stk.append((ny, nx))

           
            # 4) 长宽比过滤剔除斑马线
            lane_mask = np.zeros_like(mask_bin)
            for lbl in range(1, label_id+1):
                ys, xs = np.nonzero(labels==lbl)
                if len(xs) < min_pixels:
                    labels[labels==lbl] = 0
                    continue
                # rect = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
                # w_rect, h_rect = rect[1]
                # if w_rect < h_rect:
                #     w_rect, h_rect = h_rect, w_rect
                # if w_rect / (h_rect + 1e-5) >= aspect_thresh:
                #     lane_mask[labels==lbl] = 255

            return lane_mask, labels

        




        Label = namedtuple('Label', [
            'name', 'id', 'trainId', 'category', 'catId',
            'hasInstances', 'ignoreInEval', 'color'
        ])
        labels = [
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
        
        # input_dir = "/home/guitu/Data/ytj/data/AR_GOPRO/AR"
        input_dir = "AR"

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

                    # 车道线连通性分析
                    # lane_mask = (sem_seg == 4).astype(np.uint8)
                    # num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(lane_mask, connectivity=8)
                    # lane_cluster_vis = np.zeros((height, width, 3), dtype=np.uint8)
                    # for i in range(1, num_labels): # 为每个连通组件（车道线段）分配不同颜色
                    #     color = np.random.randint(0, 255, size=3).tolist()
                    #     lane_cluster_vis[labels_im == i] = color
                    
                    lane_mask_raw = (sem_seg == 4).astype(np.uint8)
                    lane_mask, labels_im = split_and_filter(lane_mask_raw)
                    
                    SOLID_OCC_THRESH = 0.65  # Otsu 二值化后占有率 > 0.40 ⇒ 实线
                    YELLOW_B_MAX = 180 # B 通道均值 < 180 且 R,G > B ⇒ 黄线；否则白线
                    SAT_THRESH = 50                    # ≳ 50 视为黄线

                    # 可视化每条车道线实例
                    lane_cluster_vis = np.zeros((height, width, 3), dtype=np.uint8)
                    for inst_id in np.unique(labels_im):
                        if inst_id == 0:
                            continue
                        color = np.random.randint(0, 255, size=3).tolist()
                        lane_cluster_vis[labels_im == inst_id] = color
                        
                        mask = (labels_im == inst_id)           # bool, 整幅图尺寸
                        if mask.sum() == 0:
                            continue

                        # ——— ① 灰度图 + Otsu ———
                        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

                        # 仅裁剪最小包围盒，减少背景干扰
                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                        roi_gray   = gray [y:y+h, x:x+w]
                        roi_mask   = mask [y:y+h, x:x+w]

                        # 自动阈值
                        _, bin_roi = cv2.threshold(roi_gray, 160, 255, cv2.THRESH_BINARY)
                        bin_roi    = bin_roi.astype(bool)

                        # 仅保留聚类内部的阈值结果
                        core_mask = bin_roi & roi_mask
                        core_cnt  = core_mask.sum()
                        total_cnt = roi_mask.sum()

                        # 覆盖率 —— 用于实 / 虚线判定
                        occ_rate  = core_cnt / (total_cnt + 1e-6)
                        line_type = "solid" if occ_rate > SOLID_OCC_THRESH else "dashed"

                        # ——— ② 计算平均颜色（仅对 core_mask==True 的像素）———
                        if core_cnt > 0:
                            pixels    = original_image[y:y+h, x:x+w][core_mask]
                        else:  # 极端失败回退，用整条 mask
                            pixels    = original_image[mask]

                        # mean_bgr  = pixels.mean(axis=0).astype(int)
                        # mean_b, mean_g, mean_r = mean_bgr

                        # # 黄 / 白线判定
                        # is_yellow  = (mean_b < YELLOW_B_MAX) and (mean_r > mean_b) and (mean_g > mean_b)
                        mean_s = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_BGR2HSV) \
                                [:,:,1].mean()          # 仅 S 通道
                        color_type = "yellow" if mean_s >= SAT_THRESH else "white"
                        
                        # ——— ③ 将聚类可视化，并写文字 ———
                        rand_col = np.random.randint(0, 255, size=3).tolist()
                        lane_cluster_vis[mask] = rand_col

                        center_y, center_x = int(np.mean(np.where(mask)[0])), int(np.mean(np.where(mask)[1]))
                        label_text = (f"{line_type}-{color_type} "
                                    f"occ={occ_rate:.2f} "
                                    f"hsv=({mean_s:.2f})")
                        cv2.putText(lane_cluster_vis, label_text,
                                    (center_x, center_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                        
                    cv2.imwrite(
                        os.path.join(save_dir, f"{file_name}_lane_clusters.png"),
                        lane_cluster_vis
                    )
                    
                    
                    # 语义分割结果保存
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
