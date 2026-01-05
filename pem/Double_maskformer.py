import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.data import MetadataCatalog
from detectron2.utils.registry import Registry

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from detectron2.utils.events import get_event_storage
from .modeling.meta_arch.double_mask_former_heads import build_sem_seg_head



@META_ARCH_REGISTRY.register()
class DoubleMaskFormer(nn.Module):
    """
    Dual MaskFormer architecture with two parallel segmentation heads
    for multi-task or hierarchical segmentation.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: nn.Module,
        sem_seg_head1: nn.Module,  # First MaskFormer head
        sem_seg_head2: nn.Module,  # Second MaskFormer head  
        criterion1: nn.Module,
        criterion2: nn.Module,
        num_queries1: int,
        num_queries2: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        # dual head specific
        fusion_method: str = "weighted_sum",  # "weighted_sum", "concat", "attention"
        head1_weight: float = 0.5,
        head2_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head1 = sem_seg_head1
        self.sem_seg_head2 = sem_seg_head2
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        
        self.num_queries1 = num_queries1
        self.num_queries2 = num_queries2
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        # 将pixel_mean和pixel_std转换为tensor并移至模型所在设备
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # inference flags
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        
        # dual head specific
        self.fusion_method = fusion_method
        self.head1_weight = head1_weight
        self.head2_weight = head2_weight
        
        # Fusion modules
        if fusion_method == "attention":
            self.fusion_attention = nn.Sequential(
                nn.Conv2d(2 * sem_seg_head1.num_classes, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, 3, padding=1),
                nn.Softmax(dim=1)
            )
        
        # 模型设备和数据类型由Detectron2框架自动管理

    @classmethod
    def from_config(cls, cfg):
        # 强制检查CUDA可用性，因为可变形卷积只能在GPU上运行
        if not torch.cuda.is_available():
            raise RuntimeError("此模型包含可变形卷积，必须在GPU上运行！CUDA不可用。")
        
        # 构建模型组件
        # 注意：不要尝试修改cfg，因为CfgNode是不可变的
        backbone = build_backbone(cfg)
        
        # Build first segmentation head
        sem_seg_head1 = build_sem_seg_head(cfg, backbone.output_shape(), head_id=1)
        # Build second segmentation head  
        sem_seg_head2 = build_sem_seg_head(cfg, backbone.output_shape(), head_id=2)
        
        # 立即将所有组件移动到GPU上
        device = torch.device("cuda")
        backbone = backbone.to(device)
        sem_seg_head1 = sem_seg_head1.to(device)
        sem_seg_head2 = sem_seg_head2.to(device)

        # Loss parameters for head 1
        matcher1 = HungarianMatcher(
            cost_class=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CLASS_WEIGHT,
            cost_mask=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.MASK_WEIGHT,
            cost_dice=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DICE_WEIGHT,
            num_points=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.TRAIN_NUM_POINTS,
        )

        weight_dict1 = {
            "loss_ce": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CLASS_WEIGHT,
            "loss_mask": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.MASK_WEIGHT, 
            "loss_dice": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DICE_WEIGHT
        }

        if cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict1.items()})
            weight_dict1.update(aux_weight_dict)

        # 检查配置中是否有类别权重设置，如果没有或为空则为None
        class_weights1 = getattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1, "CLASS_WEIGHTS", None)
        # 如果提供了类别权重且不为空，则转换为tensor
        if class_weights1 is not None and len(class_weights1) > 0:
            class_weights1 = torch.tensor(class_weights1, dtype=torch.float, device=device)
        else:
            class_weights1 = None
        
        criterion1 = SetCriterion(
            sem_seg_head1.num_classes,
            matcher=matcher1,
            weight_dict=weight_dict1,
            eos_coef=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NO_OBJECT_WEIGHT,
            losses=["labels", "masks"],
            num_points=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.IMPORTANCE_SAMPLE_RATIO,
            class_weights=class_weights1,
        )

        # Loss parameters for head 2
        matcher2 = HungarianMatcher(
            cost_class=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CLASS_WEIGHT,
            cost_mask=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.MASK_WEIGHT,
            cost_dice=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DICE_WEIGHT,
            num_points=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.TRAIN_NUM_POINTS,
        )

        weight_dict2 = {
            "loss_ce": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CLASS_WEIGHT,
            "loss_mask": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.MASK_WEIGHT,
            "loss_dice": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DICE_WEIGHT
        }

        if cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DEEP_SUPERVISION:
            dec_layers = cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict2.items()})
            weight_dict2.update(aux_weight_dict)

        # 检查配置中是否有类别权重设置，如果没有则为None
        class_weights2 = getattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2, "CLASS_WEIGHTS", None)
        # 如果提供了类别权重且不为空，则转换为tensor
        if class_weights2 is not None and len(class_weights2) > 0:
            class_weights2 = torch.tensor(class_weights2, dtype=torch.float, device=device)
        else:
            class_weights2 = None
        
        criterion2 = SetCriterion(
            sem_seg_head2.num_classes,
            matcher=matcher2,
            weight_dict=weight_dict2,
            eos_coef=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NO_OBJECT_WEIGHT,
            losses=["labels", "masks"],
            num_points=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.IMPORTANCE_SAMPLE_RATIO,
            class_weights=class_weights2,
        )

        return {
            "backbone": backbone,
            "sem_seg_head1": sem_seg_head1,
            "sem_seg_head2": sem_seg_head2,
            "criterion1": criterion1,
            "criterion2": criterion2,
            "num_queries1": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.NUM_OBJECT_QUERIES,
            "num_queries2": cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.DOUBLE_MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.DOUBLE_MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.DOUBLE_MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.DOUBLE_MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.DOUBLE_MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.DOUBLE_MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.DOUBLE_MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.DOUBLE_MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.DOUBLE_MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "fusion_method": cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.METHOD,
            "head1_weight": cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.HEAD1_WEIGHT,
            "head2_weight": cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.HEAD2_WEIGHT,
        }

    def forward(self, batched_inputs):
        # 首先检查CUDA是否可用，因为模型包含可变形卷积，必须在GPU上运行
        if not torch.cuda.is_available():
            raise RuntimeError("此模型包含可变形卷积，必须在GPU上运行！CUDA不可用。")
        
        # 统一设备管理，确保所有操作在同一个设备上执行
        device = torch.device("cuda")
        
        # 确保所有模型组件都在GPU上
        # 注意：在forward方法中不应该重复移动模型到GPU，这会影响梯度流动
        # 模型的设备应该在初始化时就设置好，或者在trainer中统一管理
        # 如果需要检查设备，可以保留以下代码但注释掉实际的to(device)调用
        # 只在必要时进行设备检查
        model_device = next(self.parameters()).device
        # 只比较设备类型（cpu或cuda）而不比较具体的设备索引（如cuda:0）
        if model_device.type != device.type:
            raise RuntimeError(f"模型设备类型不匹配: 期望{device.type}，实际{model_device.type}")
        
        # 预处理图像 - 确保所有操作都在同一设备上进行
        try:
            images = [x["image"].to(device, non_blocking=True) for x in batched_inputs]
            # 确保pixel_mean和pixel_std在正确的设备上
            pixel_mean = self.pixel_mean.to(device, non_blocking=True)
            pixel_std = self.pixel_std.to(device, non_blocking=True)
            images = [(x - pixel_mean) / pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            
            # 确保images.tensor在GPU上
            images_tensor = images.tensor.to(device, non_blocking=True)
        except Exception as e:
            print(f"图像预处理错误: {e}")
            # 尝试减小批量大小或清理缓存
            torch.cuda.empty_cache()
            raise e
        
        # 提取特征
        try:
            features = self.backbone(images_tensor)
            # 确保所有特征都在GPU上
            for k, v in features.items():
                features[k] = v.to(device, non_blocking=True)
            
            # Forward through both heads
            # 添加梯度累积和内存优化
            torch.cuda.empty_cache()  # 在模型前向传播前清理缓存
            outputs1 = self.sem_seg_head1(features)
            # 释放中间特征以节省内存
            torch.cuda.empty_cache()
            outputs2 = self.sem_seg_head2(features)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"CUDA内存不足错误: {e}")
                torch.cuda.empty_cache()
                # 尝试使用混合精度训练以减少内存使用
                with torch.cuda.amp.autocast():
                    features = self.backbone(images_tensor)
                    for k, v in features.items():
                        features[k] = v.to(device, non_blocking=True)
                    outputs1 = self.sem_seg_head1(features)
                    outputs2 = self.sem_seg_head2(features)
            else:
                print(f"特征提取或模型前向传播错误: {e}")
                raise e

        if self.training:
            # Apply weight dictionary and combine losses
            combined_losses = {}
            
            # 尝试处理目标数据，并添加错误处理
            all_targets = None
            try:
                # Check if we have instance annotations
                if "instances" in batched_inputs[0]:
                    # Prepare targets for both heads
                    gt_instances = [x["instances"].to(device, non_blocking=True) for x in batched_inputs]
                    all_targets = self.prepare_targets(gt_instances, images, device=device)
                else:
                    all_targets = None
            except Exception as e:
                print(f"目标数据处理错误: {e}")
                # 如果目标数据处理失败，使用空的目标列表继续
                all_targets = [{} for _ in range(len(batched_inputs))]

            # 数据集分离逻辑 - 添加安全检查
            try:
                has_dataset_id = all(["dataset_id" in x for x in batched_inputs])
            except Exception as e:
                print(f"数据集ID检查错误: {e}")
                has_dataset_id = False
            
            # 初始化损失变量
            total_loss_head1 = 0.0
            total_loss_head2 = 0.0
            losses1 = {}
            losses2 = {}
            
            if has_dataset_id:
                # Separate targets by dataset_id
                dataset1_indices = [i for i, x in enumerate(batched_inputs) if x["dataset_id"] == 0]
                dataset2_indices = [i for i, x in enumerate(batched_inputs) if x["dataset_id"] == 1]
                
                # Compute losses for head1 using only dataset1
                if dataset1_indices:
                    dataset1_targets = [all_targets[i] for i in dataset1_indices]
                    dataset1_outputs = {
                        "pred_logits": outputs1["pred_logits"][dataset1_indices],
                        "pred_masks": outputs1["pred_masks"][dataset1_indices]
                    }
                    losses1 = self.criterion1(dataset1_outputs, dataset1_targets)
                    for k in list(losses1.keys()):
                        if k in self.criterion1.weight_dict:
                            weighted_loss = losses1[k] * self.criterion1.weight_dict[k]
                            combined_losses[k + "_head1"] = weighted_loss
                            total_loss_head1 += weighted_loss
                
                # Compute losses for head2 using only dataset2
                if dataset2_indices:
                    dataset2_targets = [all_targets[i] for i in dataset2_indices]
                    dataset2_outputs = {
                        "pred_logits": outputs2["pred_logits"][dataset2_indices],
                        "pred_masks": outputs2["pred_masks"][dataset2_indices]
                    }
                    losses2 = self.criterion2(dataset2_outputs, dataset2_targets)
                    for k in list(losses2.keys()):
                        if k in self.criterion2.weight_dict:
                            weighted_loss = losses2[k] * self.criterion2.weight_dict[k]
                            combined_losses[k + "_head2"] = weighted_loss
                            total_loss_head2 += weighted_loss
            else:
                # Fallback to original behavior if no dataset_id is provided
                losses1 = self.criterion1(outputs1, all_targets)
                losses2 = self.criterion2(outputs2, all_targets)
                
                for k in list(losses1.keys()):
                    if k in self.criterion1.weight_dict:
                        weighted_loss = losses1[k] * self.criterion1.weight_dict[k]
                        combined_losses[k + "_head1"] = weighted_loss
                        total_loss_head1 += weighted_loss
                
                for k in list(losses2.keys()):
                    if k in self.criterion2.weight_dict:
                        weighted_loss = losses2[k] * self.criterion2.weight_dict[k]
                        combined_losses[k + "_head2"] = weighted_loss
                        total_loss_head2 += weighted_loss
            
            # 由于total_loss_head1和total_loss_head2已经包含了各自的权重，直接将它们相加
            # 避免重复应用权重导致的梯度问题
            total_loss = total_loss_head1 + total_loss_head2
            
            # 添加正则化项以防止过拟合
            # L2正则化已经通过优化器的weight_decay处理，这里不再需要额外的正则化项
            # 但我们可以添加梯度范数监控，以帮助诊断训练问题
            if self.training and torch.is_grad_enabled():
                # 定期记录梯度范数信息
                storage = get_event_storage()
                if storage.iter % 100 == 0:
                    # 计算总梯度范数
                    total_grad_norm = 0.0
                    for param in self.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** 0.5
                    storage.put_scalar("gradients/total_grad_norm", total_grad_norm)
            
            # 添加总损失到输出
            combined_losses["total_loss_head1"] = total_loss_head1
            combined_losses["total_loss_head2"] = total_loss_head2
            # 移除所有权重记录，避免在计算图中引入额外操作
            combined_losses["loss"] = total_loss  # detectron2期望的总损失键名
            
            # 记录损失信息到事件存储
            storage = get_event_storage()
            storage.put_scalar("losses/total_loss_head1", total_loss_head1)
            storage.put_scalar("losses/total_loss_head2", total_loss_head2)
            
            return combined_losses
        else:
            # Check if we're in special evaluation mode
            is_evaluation_mode = any(["is_evaluation" in x and x["is_evaluation"] for x in batched_inputs])
            
            # Inference - get outputs from both heads
            mask_cls_results1 = outputs1["pred_logits"]
            mask_pred_results1 = outputs1["pred_masks"]
            mask_cls_results2 = outputs2["pred_logits"] 
            mask_pred_results2 = outputs2["pred_masks"]

            # 获取当前特征图尺寸
            h_input, w_input = images.tensor.shape[-2:]
            
            # Upsample masks to match input size
            mask_pred_results1 = F.interpolate(
                mask_pred_results1,
                size=(h_input, w_input),
                mode="bilinear",
                align_corners=False,
            )
            mask_pred_results2 = F.interpolate(
                mask_pred_results2,
                size=(h_input, w_input),
                mode="bilinear", 
                align_corners=False,
            )

            processed_results = []
            for i, (input_per_image, image_size) in enumerate(zip(batched_inputs, images.image_sizes)):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                orig_h, orig_w = image_size
                target_h, target_w = height, width
                
                # Get individual results
                mask_cls1 = mask_cls_results1[i]
                mask_pred1 = mask_pred_results1[i]
                mask_cls2 = mask_cls_results2[i]
                mask_pred2 = mask_pred_results2[i]

                # Pre-inference postprocessing if enabled
                if self.sem_seg_postprocess_before_inference:
                    mask_pred1 = sem_seg_postprocess(mask_pred1, (orig_h, orig_w), target_h, target_w)
                    mask_pred2 = sem_seg_postprocess(mask_pred2, (orig_h, orig_w), target_h, target_w)

                # Get semantic predictions from both heads
                semseg1 = self.semantic_inference(mask_cls1, mask_pred1)
                semseg2 = self.semantic_inference(mask_cls2, mask_pred2)
                
                if is_evaluation_mode:
                    # Special evaluation mode with separate head outputs based on dataset
                    processed_result = self._process_evaluation_outputs(
                        semseg1, semseg2, 
                        image_size=(orig_h, orig_w),
                        height=target_h,
                        width=target_w,
                        already_postprocessed=self.sem_seg_postprocess_before_inference
                    )
                    # Use dataset-specific head output instead of merged result
                    # Check if dataset_id is present in input
                    dataset_id = input_per_image.get("dataset_id", 0)
                    if dataset_id == 0:  # First dataset (e.g., Apollo)
                        processed_result["sem_seg"] = processed_result["sem_seg_head1"]
                    else:  # Second dataset (e.g., Vestas)
                        processed_result["sem_seg"] = processed_result["sem_seg_head2"]
                else:
                    # Regular inference - use dataset-specific head output to avoid dimension mismatch
                    # Get dataset_id from input_per_image
                    dataset_id = 0  # Default to first dataset
                    if "dataset_id" in input_per_image:
                        dataset_id = input_per_image["dataset_id"]
                    
                    # Select the appropriate head output based on dataset
                    if dataset_id == 0:  # First dataset (e.g., Apollo)
                        fused_semseg = semseg1
                    else:  # Second dataset (e.g., Vestas)
                        fused_semseg = semseg2

                    # Apply postprocessing if needed
                    if not self.sem_seg_postprocess_before_inference:
                        fused_semseg = sem_seg_postprocess(fused_semseg, (orig_h, orig_w), target_h, target_w)
                    
                    # Ensure correct output size
                    if fused_semseg.shape[-2:] != (target_h, target_w):
                        fused_semseg = F.interpolate(
                            fused_semseg.unsqueeze(0), 
                            size=(target_h, target_w), 
                            mode="bilinear", 
                            align_corners=False
                        ).squeeze(0)

                    processed_result = {"sem_seg": fused_semseg}

                processed_results.append(processed_result)

            return processed_results
    
    def _process_evaluation_outputs(self, semseg1, semseg2, image_size, height, width, 
                                   already_postprocessed=False):
        """
        Process outputs from both heads for evaluation with independent mode only.
        Each head's output remains independent without modifying each other's outputs.
        
        Args:
            semseg1: output from head1 with shape (C1, H, W)
            semseg2: output from head2 with shape (C2, H, W)
            image_size: original image size (height, width)
            height: target height
            width: target width
            already_postprocessed: whether semseg1 and semseg2 are already postprocessed
            
        Returns:
            dict with processed sem_seg_head1 and sem_seg_head2
        """
        # Remove batch dimension if present
        if semseg1.dim() == 4:
            semseg1 = semseg1.squeeze(0)
        if semseg2.dim() == 4:
            semseg2 = semseg2.squeeze(0)
        
        # Clone to avoid modifying originals
        processed_semseg1 = semseg1.clone()
        processed_semseg2 = semseg2.clone()
        
        # Process original image size
        orig_h, orig_w = image_size[0], image_size[1] if isinstance(image_size, tuple) and len(image_size) >= 2 else (height, width)
        
        # Apply postprocessing if needed
        if not already_postprocessed:
            processed_semseg1 = sem_seg_postprocess(processed_semseg1, (orig_h, orig_w), height, width)
            processed_semseg2 = sem_seg_postprocess(processed_semseg2, (orig_h, orig_w), height, width)
            
            # Ensure correct dimensions
            if processed_semseg1.shape[-2:] != (height, width):
                processed_semseg1 = F.interpolate(
                    processed_semseg1.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
                ).squeeze(0)
            
            if processed_semseg2.shape[-2:] != (height, width):
                processed_semseg2 = F.interpolate(
                    processed_semseg2.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
                ).squeeze(0)
        
        # Final dimension verification
        def ensure_correct_size(tensor, target_h, target_w):
            if tensor.shape[-2:] != (target_h, target_w):
                return F.interpolate(
                    tensor.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
                ).squeeze(0)
            return tensor
        
        # Apply size correction
        processed_semseg1 = ensure_correct_size(processed_semseg1, height, width)
        processed_semseg2 = ensure_correct_size(processed_semseg2, height, width)
        
        return {
                "sem_seg_head1": processed_semseg1,
                "sem_seg_head2": processed_semseg2
            }

    def fuse_with_attention(self, semseg1, semseg2):
        """Fuse two semantic segmentations using attention mechanism"""
        if hasattr(self, 'fusion_attention'):
            # Concatenate along channel dimension
            concat = torch.cat([semseg1, semseg2], dim=0).unsqueeze(0)
            # Compute attention weights
            attention_weights = self.fusion_attention(concat)
            # Apply attention weights
            fused = attention_weights[:, 0:1] * semseg1 + attention_weights[:, 1:2] * semseg2
            return fused.squeeze(0)
        else:
            # Fallback to weighted sum
            return 0.5 * semseg1 + 0.5 * semseg2

    def prepare_targets(self, targets, images, device=None):
        """Same as original MaskFormer but ensure targets are on the correct device"""
        # 如果没有提供device参数，使用self.device作为备选
        if device is None:
            device = self.device
        
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            # Ensure masks are on the correct device
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks.to(device)
            new_targets.append({
                "labels": targets_per_image.gt_classes.to(device),
                "masks": padded_masks,
            })
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        """Same as original MaskFormer"""
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    @property
    def device(self):
        """返回模型所在设备，确保在GPU上运行"""
        # 首先尝试从参数获取设备
        param = next(self.parameters(), None)
        if param is not None:
            return param.device
        
        # 如果没有参数，强制返回cuda设备（因为代码仅支持GPU）
        if torch.cuda.is_available():
            return torch.device("cuda")
        
        # 如果没有GPU可用，抛出异常
        raise RuntimeError("此模型仅支持GPU训练，但CUDA不可用")


