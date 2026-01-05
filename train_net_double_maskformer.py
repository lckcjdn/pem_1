#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

# 直接在此文件中修复PIL.Image.LINEAR兼容性问题
from PIL import Image
# 添加LINEAR属性，映射到BILINEAR
if not hasattr(Image, 'LINEAR'):
    Image.LINEAR = Image.BILINEAR

import logging
import os
from collections import OrderedDict
import torch
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (CityscapesInstanceEvaluator,
                                 CityscapesSemSegEvaluator,
                                 COCOEvaluator,
                                 COCOPanopticEvaluator,
                                 DatasetEvaluators,
                                 LVISEvaluator,
                                 PascalVOCDetectionEvaluator,
                                 SemSegEvaluator,
                                 verify_results,
                                 DatasetEvaluator)
# 导入自定义的双分割头评估器
from pem.custom_double_head_evaluator import CustomDoubleHeadEvaluator
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import get_event_storage
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

# Import our double maskformer model
from pem.Double_maskformer import DoubleMaskFormer
from pem.data.dataset_mappers.double_maskformer_semantic_dataset_mapper import DoubleMaskFormerSemanticDatasetMapper
# Import our custom dataset registration
from pem.data.datasets.register_double_cityscapes_with_bdd100k import register_all_double_cityscapes

# Import configuration function
from pem.config import add_double_maskformer_config

logger = logging.getLogger("detectron2")


# add_double_maskformer_config function is now defined in pem/config.py


class DoubleMaskFormerTrainer(DefaultTrainer):
    """
    Trainer for Double MaskFormer model with two datasets.
    仅支持GPU训练
    """
    def __init__(self, cfg):
        # 确保在GPU上运行
        if not torch.cuda.is_available():
            raise RuntimeError("此代码仅支持GPU训练，请确保CUDA可用")
        logger.info("训练模式: GPU (CUDA)")
        
        # 内存优化配置
        # 启用CUDA内存分配器优化
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = False  # 对于动态尺寸输入，关闭benchmark可能更节省内存
        
        # 初始化梯度累积步数
        # 安全地获取梯度累积步数，避免配置项不存在的错误
        if hasattr(cfg, 'SOLVER') and hasattr(cfg.SOLVER, 'GRADIENT_ACCUMULATION_STEPS'):
            self.gradient_accumulation_steps = cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
        else:
            self.gradient_accumulation_steps = 2
        self.accumulated_grads = 0
        
        super().__init__(cfg)
        # 确保_train_loader属性正确初始化
        if not hasattr(self, '_train_loader'):
            self._train_loader = self.build_train_loader(cfg)
        
        # 强制将模型移动到GPU上
        device = torch.device("cuda")
        self.model = self.model.to(device)
        # 确保所有子模块都在GPU上
        for module in self.model.modules():
            if isinstance(module, torch.nn.Module):
                module = module.to(device)
        
        # 打印模型信息
        logger.info("模型:")
        logger.info(f"  {self.model}")
        self.print_parameter_status()
    
    def print_parameter_status(self):
        """
        打印模型各部分参数的可训练状态
        """
        backbone_trainable = 0
        backbone_total = 0
        
        head1_trainable = 0
        head1_total = 0
        
        head2_trainable = 0
        head2_total = 0
        
        decoder_trainable = 0
        decoder_total = 0
        
        other_trainable = 0
        other_total = 0
        
        # 遍历所有参数并统计
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            
            if "backbone" in name:
                backbone_total += param_count
                if param.requires_grad:
                    backbone_trainable += param_count
            elif "sem_seg_head1" in name:
                head1_total += param_count
                if param.requires_grad:
                    head1_trainable += param_count
            elif "sem_seg_head2" in name:
                head2_total += param_count
                if param.requires_grad:
                    head2_trainable += param_count
            elif "pixel_decoder" in name or "transformer_decoder" in name:
                decoder_total += param_count
                if param.requires_grad:
                    decoder_trainable += param_count
            else:
                other_total += param_count
                if param.requires_grad:
                    other_trainable += param_count
        
        logger.info("模型参数详细统计:")
        logger.info(f"  骨干网络: {backbone_trainable/1e6:.2f}M/{backbone_total/1e6:.2f}M 可训练")
        logger.info(f"  分割头1: {head1_trainable/1e6:.2f}M/{head1_total/1e6:.2f}M 可训练")
        logger.info(f"  分割头2: {head2_trainable/1e6:.2f}M/{head2_total/1e6:.2f}M 可训练")
        logger.info(f"  解码器: {decoder_trainable/1e6:.2f}M/{decoder_total/1e6:.2f}M 可训练")
        logger.info(f"  其他部分: {other_trainable/1e6:.2f}M/{other_total/1e6:.2f}M 可训练")
    
    def after_step(self):
        """
        自定义训练步骤后处理，避免访问可能不存在的_total_timer属性，并过滤掉学习率调度器相关的hooks
        """
        # 只执行必要的操作，避免引用不存在的属性
        try:
            # 简单实现after_step，不依赖_total_timer
            # 过滤掉学习率调度器相关的hooks，因为我们在run_step中已经手动处理了调度器更新
            if hasattr(self, '_hooks'):
                for h in self._hooks:
                    # 检查hook是否是学习率调度器相关的
                    hook_name = str(type(h).__name__)
                    # 跳过任何与学习率调度相关的hook，避免重复更新导致的边界条件错误
                    if 'LRScheduler' in hook_name or 'ParamScheduler' in hook_name:
                        continue
                    
                    if hasattr(h, 'after_step'):
                        try:
                            h.after_step()
                        except Exception as e:
                            # 记录错误但不中断训练
                            if hasattr(logger, 'warning'):
                                logger.warning(f"Hook after_step error ({hook_name}): {e}")
        except Exception as e:
            # 记录错误但不中断训练
            if hasattr(logger, 'warning'):
                logger.warning(f"After step error: {e}")
    
    def run_step(self):
        """
        重写单步执行逻辑，确保正确的执行顺序：
        1. optimizer.step() 在 lr_scheduler.step() 之前调用
        2. 避免PyTorch的学习率调度器顺序警告
        3. 添加梯度累积以减少内存使用
        4. 正确管理计时器状态
        """
        assert self.model.training, f"[{self.__class__.__name__}.run_step] model was changed to eval mode!"
        
        # 正确处理计时器状态 - 暂停计时器
        if hasattr(self, '_total_timer') and hasattr(self._total_timer, 'pause'):
            try:
                self._total_timer.pause()
            except (ValueError, RuntimeError):
                # 忽略计时器已经暂停或不存在的错误
                pass
        
        # 正确获取数据加载器迭代器
        if not hasattr(self, "_data_loader_iter"):
            self._data_loader_iter = iter(self._train_loader)
        
        try:
            # 获取数据
            data = next(self._data_loader_iter)
        except StopIteration:
            # 数据集迭代完成，重新创建迭代器
            self._data_loader_iter = iter(self._train_loader)
            data = next(self._data_loader_iter)
        
        # 确保数据移动到模型所在设备
        device = next(self.model.parameters()).device
        data = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in data]
        
        # 启用梯度累积
        if self.accumulated_grads == 0:
            self.optimizer.zero_grad()
        
        # 前向传播和反向传播
        loss_dict = self.model(data)
        
        # 只使用主要损失项进行梯度计算，避免包含不需要梯度的辅助项
        # 使用'loss'键（detectron2期望的总损失键名）
        if 'loss' in loss_dict:
            main_loss = loss_dict['loss']
        else:
            # 备选方案：只使用包含'loss'的项，排除'weight'相关项
            main_loss = sum(v for k, v in loss_dict.items() if 'loss' in k and 'weight' not in k)
        
        # 缩放损失以适应梯度累积
        scaled_loss = main_loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_grads += 1
        
        # 只在累积了足够的梯度后更新参数
        if self.accumulated_grads % self.gradient_accumulation_steps == 0:
            # 添加梯度裁剪以防止梯度爆炸
            if hasattr(self.cfg.SOLVER, 'CLIP_GRADIENTS') and self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
                # 增加梯度裁剪阈值，从0.01提高到1.0或更高，避免过度限制梯度更新
                clip_value = max(self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE, 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_value)
                # 记录实际使用的裁剪值
                if self.accumulated_grads % self.gradient_accumulation_steps == 0:
                    storage = get_event_storage()
                    storage.put_scalar("gradients/clip_value", clip_value)
            
            self.optimizer.step()
            # 优化器更新后立即更新学习率调度器
            # 确保optimizer.step()在scheduler.step()之前执行，以符合PyTorch的建议
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                # 检查是否已经达到或超过最大迭代次数，避免调度器参数超出范围
                try:
                    # 优化的当前迭代比例计算，使用更安全的方式获取max_iter
                    max_iter = getattr(self, 'max_iter', float('inf'))
                    # 从配置中获取max_iter作为备选
                    if hasattr(self, 'cfg') and hasattr(self.cfg, 'SOLVER') and hasattr(self.cfg.SOLVER, 'MAX_ITER'):
                        max_iter = min(max_iter, self.cfg.SOLVER.MAX_ITER)
                    
                    # 确保current_ratio严格小于1.0，防止调度器参数超出范围
                    current_ratio = min(self.iter / max_iter, 0.999)
                    
                    # 安全地更新学习率调度器
                    if hasattr(self.scheduler, '_max_iter'):
                        # 对于param_scheduler类型的调度器
                        # 使用try-except块安全地执行更新
                        try:
                            self.scheduler.step()
                        except Exception as inner_e:
                            if hasattr(logger, 'warning'):
                                logger.warning(f"调度器更新异常但继续训练: {inner_e}, 迭代={self.iter}/{max_iter}")
                    else:
                        # 对于标准PyTorch调度器
                        # 检查是否为PyTorch原生调度器，避免不必要的错误
                        import inspect
                        if inspect.ismethod(self.scheduler.step):
                            self.scheduler.step()
                except RuntimeError as e:
                    error_msg = str(e)
                    if "where in ParamScheduler must be in [0, 1]" in error_msg:
                        # 捕获边界条件错误，使用更详细的日志
                        current_value = None
                        # 尝试提取错误中的具体值
                        import re
                        match = re.search(r'got (\d+\.?\d*)', error_msg)
                        if match:
                            current_value = match.group(1)
                        
                        if hasattr(logger, 'warning'):
                            logger.warning(f"调度器边界条件错误，跳过更新: 当前值={current_value}, 最大允许值=1.0, 迭代={self.iter}/{max_iter}")
                    else:
                        # 捕获任何调度器相关的错误，确保训练可以继续
                        if hasattr(logger, 'warning'):
                            logger.warning(f"Scheduler step error: {e}")
                except Exception as e:
                    # 捕获任何调度器相关的错误，确保训练可以继续
                    if hasattr(logger, 'warning'):
                        logger.warning(f"Scheduler step error: {e}")
                
                # 记录当前学习率，帮助监控训练过程
                if hasattr(logger, 'info') and self.iter % 100 == 0:
                    try:
                        # 获取当前学习率
                        if hasattr(self.optimizer, 'param_groups'):
                            lr = self.optimizer.param_groups[0]['lr']
                            logger.info(f"Iteration {self.iter}, Learning Rate: {lr:.8f}")
                    except Exception:
                        pass
            self.accumulated_grads = 0
        
        # 定期清理缓存以释放内存
        if self.iter % 100 == 0:
            torch.cuda.empty_cache()
        
        # 记录损失 - 使用更安全的方式，避免上下文管理器错误
        try:
            if hasattr(self, 'storage'):
                # 尝试不同的记录方式，兼容不同版本的Detectron2
                if hasattr(self.storage, 'put_scalar'):
                    # 直接使用put_scalar方法（更简单的方式）
                    self.storage.put_scalar('total_loss', main_loss.detach().item())
                elif hasattr(self.storage, 'put_scalars'):
                    # 如果必须使用put_scalars，尝试不使用上下文管理器
                    put_scalar_result = self.storage.put_scalars(total_loss=main_loss.detach().item())
                    # 如果需要手动关闭上下文管理器
                    if hasattr(put_scalar_result, '__exit__'):
                        put_scalar_result.__exit__(None, None, None)
        except Exception as e:
            # 记录错误但不中断训练
            if hasattr(logger, 'warning'):
                logger.warning(f"Failed to record loss: {e}")
        
        # 记录学习率
        if self.optimizer is not None:
            # 获取第一个参数组的学习率作为代表性学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            storage = get_event_storage()
            storage.put_scalar("lr", current_lr)
        
        # 更新迭代计数
        self.iter += 1
        
        # run_step方法已在前面完整实现，这里不再需要额外代码
        
    @classmethod
    def save_inference_results(cls, cfg, model):
        """
        对输入文件夹中的图片进行语义分割并保存结果
        同时处理双分割头模型的两个分割头输出，并分别保存
        """
        import numpy as np
        import cv2
        import glob
        from tqdm import tqdm
        import os
        from detectron2.data.detection_utils import read_image
        from detectron2.data.transforms import ResizeShortestEdge
        from collections import namedtuple

        # 导入labels_double.py中的标签定义
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'pem', 'data', 'datasets'))
        from labels_double import get_label_mappings, apollo_labels, vestas_labels
        from labels_bdd100k import get_label_mappings, bdd100k_labels, bdd100k_0_labels
        
        # 对于双分割头模型，我们根据分割头的特性选择不同的标签集
        # 分割头1使用Apollo标签集，分割头2使用Vestas标签集
        # 这样可以确保每个分割头的输出都能得到正确的颜色映射
        # 注意：在后续处理中，我们会为每个分割头分别使用对应的标签集
        
        # 为不同的分割头创建对应的颜色映射
        # 分割头1使用Apollo标签的颜色映射
        apollo_color_map = {label.trainId: label.color for label in apollo_labels}
        bdd100k_color_map = {label.trainId: label.color for label in bdd100k_labels}
        bdd100k_0_color_map = {label.trainId: label.color for label in bdd100k_0_labels}

        # 分割头2使用Vestas标签的颜色映射
        vestas_color_map = {label.trainId: label.color for label in vestas_labels}
        
        # 设置输入输出目录
        input_dir = os.path.join(cfg.OUTPUT_DIR, "input_images")
        output_dir_head1 = os.path.join(cfg.OUTPUT_DIR, "segmentation_head1")
        output_dir_head2 = os.path.join(cfg.OUTPUT_DIR, "segmentation_head2")
        
        # 创建输出目录
        os.makedirs(output_dir_head1, exist_ok=True)
        os.makedirs(output_dir_head2, exist_ok=True)
        
        # 获取图像列表
        image_files = glob.glob(os.path.join(input_dir, "*.jpg")) + \
                      glob.glob(os.path.join(input_dir, "*.png")) + \
                      glob.glob(os.path.join(input_dir, "*.jpeg"))
        
        if not image_files:
            logger.warning(f"在{input_dir}目录中未找到图像文件")
            return
        
        # 设置图像预处理
        # 检查MIN_SIZE_TEST的类型，如果是列表且只有一个元素，就直接使用该元素值
        if isinstance(cfg.INPUT.MIN_SIZE_TEST, list) and len(cfg.INPUT.MIN_SIZE_TEST) == 1:
            min_size = cfg.INPUT.MIN_SIZE_TEST[0]
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            
        transform = ResizeShortestEdge(
            min_size, cfg.INPUT.MAX_SIZE_TEST,
            sample_style="choice"  # 使用'choice'采样风格，可以接受单个值
        )
        
        # 设置模型为评估模式
        model.eval()
        
        # 处理每张图像
        for image_file in tqdm(image_files):
            # 读取图像
            img = read_image(image_file, format="RGB")
            original_height, original_width = img.shape[:2]
            
            # 预处理
            image = transform.get_transform(img).apply_image(img)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            # 创建batch输入
            inputs = [
                {
                    "image": image,
                    "height": original_height,
                    "width": original_width,
                    # 添加评估模式标志
                    "is_evaluation": True
                }
            ]
            
            # 模型推理
            with torch.no_grad():
                outputs = model(inputs)
            
            # 获取分割结果
            output = outputs[0]
            sem_seg_head1 = output.get("sem_seg_head1")
            sem_seg_head2 = output.get("sem_seg_head2")
            
            # 确保两个分割头的结果都存在
            if sem_seg_head1 is None or sem_seg_head2 is None:
                logger.warning(f"无法获取图像{image_file}的分割结果")
                continue
            
            # 将结果移到CPU并转换为numpy数组
            sem_seg_head1 = sem_seg_head1.cpu().numpy()
            sem_seg_head2 = sem_seg_head2.cpu().numpy()
            
            # 获取预测类别
            pred_head1 = np.argmax(sem_seg_head1, axis=0)
            pred_head2 = np.argmax(sem_seg_head2, axis=0)
            
            # 确保预测结果尺寸与原始图像一致
            if pred_head1.shape != (original_height, original_width):
                pred_head1 = cv2.resize(pred_head1, (original_width, original_height), 
                                      interpolation=cv2.INTER_NEAREST)
            if pred_head2.shape != (original_height, original_width):
                pred_head2 = cv2.resize(pred_head2, (original_width, original_height), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # 创建彩色分割图
            color_mask_head1 = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            color_mask_head2 = np.zeros((original_height, original_width, 3), dtype=np.uint8)
            
            # 为分割头1应用Apollo标签颜色映射
            for label in bdd100k_labels:
                mask_head1 = pred_head1 == label.trainId
                color_mask_head1[mask_head1] = label.color
            
            # 为分割头2应用Vestas标签颜色映射
            for label in vestas_labels:
                mask_head2 = pred_head2 == label.trainId
                color_mask_head2[mask_head2] = label.color
            
            # 创建叠加图
            overlay_head1 = cv2.addWeighted(img, 0.7, color_mask_head1, 0.3, 0)
            overlay_head2 = cv2.addWeighted(img, 0.7, color_mask_head2, 0.3, 0)
            
            # 保存结果
            filename = os.path.basename(image_file)
            color_mask_head1 = cv2.cvtColor(color_mask_head1, cv2.COLOR_RGB2BGR)
            color_mask_head2 = cv2.cvtColor(color_mask_head2, cv2.COLOR_RGB2BGR)
            overlay_head1 = cv2.cvtColor(overlay_head1, cv2.COLOR_RGB2BGR)
            overlay_head2 = cv2.cvtColor(overlay_head2, cv2.COLOR_RGB2BGR)
            # 保存分割图
            cv2.imwrite(os.path.join(output_dir_head1, f"{os.path.splitext(filename)[0]}_seg.png"), color_mask_head1)
            cv2.imwrite(os.path.join(output_dir_head2, f"{os.path.splitext(filename)[0]}_seg.png"), color_mask_head2)
            
            # 保存叠加图
            cv2.imwrite(os.path.join(output_dir_head1, f"{os.path.splitext(filename)[0]}_overlay.png"), overlay_head1)
            cv2.imwrite(os.path.join(output_dir_head2, f"{os.path.splitext(filename)[0]}_overlay.png"), overlay_head2)
            
        # 生成图例图片函数
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
                cv2.rectangle(legend_img, (square_x, square_y), 
                             (square_x + square_size, square_y + square_size), 
                             label.color, -1)
                
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
        
        # 为分割头1生成并保存Apollo图例
        bdd100k_legend_img = generate_legend(bdd100k_labels)
        bdd100k_legend_path = os.path.join(output_dir_head1, "legend.png")
        bdd100k_legend_img_bgr = cv2.cvtColor(bdd100k_legend_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(bdd100k_legend_path, bdd100k_legend_img_bgr)
        logger.info(f"BDD100K图例已保存到 {bdd100k_legend_path}")
        
        # 为分割头2生成并保存Vestas图例
        vestas_legend_img = generate_legend(vestas_labels)
        vestas_legend_path = os.path.join(output_dir_head2, "legend.png")
        vestas_legend_img_bgr = cv2.cvtColor(vestas_legend_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(vestas_legend_path, vestas_legend_img_bgr)
        logger.info(f"Vestas图例已保存到 {vestas_legend_path}")
        
        logger.info(f"分割完成！结果已保存到 {output_dir_head1} 和 {output_dir_head2}")
    
    @classmethod
    def build_model(cls, cfg):
        """
        Build the Double MaskFormer model.
        """
        model = DoubleMaskFormer(cfg)
        return model
    
    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build the optimizer with support for full_model gradient clipping.
        This is a custom implementation since detectron2 doesn't support full_model gradient clipping natively.
        """
        from detectron2.solver.build import maybe_add_gradient_clipping
        import torch.optim as optim
        import itertools
        
        # 自定义全模型梯度裁剪函数
        def maybe_add_full_model_gradient_clipping(optim_cls):
            # detectron2 doesn't have full model gradient clipping now
            # 提高梯度裁剪阈值到一个更合理的值
            clip_norm_val = max(cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE, 1.0)
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim_cls):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim_cls
        
        # 获取参数列表，为不同组件设置不同的学习率
        params = []
        
        # 为双分割头添加配置参数支持
        # 如果配置中没有，可以使用默认值
        head1_multiplier = getattr(cfg.SOLVER, "HEAD1_LR_MULTIPLIER", 1.0)
        head2_multiplier = getattr(cfg.SOLVER, "HEAD2_LR_MULTIPLIER", 1.0)
        
        # 统计不同部分的参数数量
        backbone_params = 0
        head1_params = 0
        head2_params = 0
        other_params = 0
        
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
                
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            
            # 根据参数名称设置不同的学习率
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
                backbone_params += value.numel()
            elif "sem_seg_head1" in key:
                lr = lr * head1_multiplier
                head1_params += value.numel()
            elif "sem_seg_head2" in key:
                lr = lr * head2_multiplier
                head2_params += value.numel()
            else:
                # 其他参数使用基础学习率
                other_params += value.numel()
            
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
        # 记录参数统计信息
        total_params = backbone_params + head1_params + head2_params + other_params
        logger.info(f"参数学习率配置:")
        logger.info(f"  Backbone: {backbone_params / 1e6:.2f}M 参数, 学习率乘数: {cfg.SOLVER.BACKBONE_MULTIPLIER}")
        logger.info(f"  分割头1: {head1_params / 1e6:.2f}M 参数, 学习率乘数: {head1_multiplier}")
        logger.info(f"  分割头2: {head2_params / 1e6:.2f}M 参数, 学习率乘数: {head2_multiplier}")
        logger.info(f"  其他部分: {other_params / 1e6:.2f}M 参数")
        logger.info(f"  总计: {total_params / 1e6:.2f}M 参数")
        
        # 根据优化器类型创建优化器
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
        
        # 如果不是full_model类型，使用detectron2的默认梯度裁剪
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        
        return optimizer
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Build evaluators for both datasets, ensuring correct head selection based on dataset type.
        For double head evaluation, use the appropriate number of classes based on the dataset.
        """
        # 创建数据集特定的输出文件夹
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        # 根据数据集名称创建专门的子文件夹，避免结果混淆
        dataset_specific_output = os.path.join(output_folder, dataset_name)
        os.makedirs(dataset_specific_output, exist_ok=True)
        
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        
        # 对于语义分割任务，使用自定义的双分割头评估器
        # 它会根据数据集类型自动选择对应的分割头结果
        if evaluator_type in ["sem_seg", "cityscapes_sem_seg", "cityscapes_instance"]:
            return CustomDoubleHeadEvaluator(
                dataset_name,
                distributed=False,  # 禁用分布式评估以减少通信开销
                output_dir=dataset_specific_output,
                # 类别数会在CustomDoubleHeadEvaluator内部根据数据集名称自动设置
            )
        
        # 对于其他类型的任务，使用默认评估器
        evaluator_list = []
        if evaluator_type == "coco":
            evaluator_list.append(
                COCOEvaluator(dataset_name, output_dir=dataset_specific_output)
            )
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name)
        elif evaluator_type == "coco_panoptic":
            return COCOPanopticEvaluator(dataset_name)
        
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
    def test(cls, cfg, model):
        """
        静态方法：在评估模式下测试模型性能
        兼容detectron2的评估流程
        """
        from detectron2.evaluation import inference_on_dataset
        from detectron2.utils.logger import log_first_n
        import logging
        
        # 确保模型在正确的设备上
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # 获取测试数据集
        dataset_names = cfg.DATASETS.TEST
        results = {}
        
        for dataset_name in dataset_names:
            logger.info(f"评估数据集: {dataset_name}")
            
            # 构建测试加载器
            data_loader = cls.build_test_loader(cfg, dataset_name)
            
            # 构建评估器
            evaluator = cls.build_evaluator(cfg, dataset_name)
            
            # 执行推理和评估
            logger.info(f"开始在数据集 {dataset_name} 上执行评估")
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            
            if comm.is_main_process():
                logger.info(f"数据集 {dataset_name} 的评估结果:")
                for k, v in results_i.items():
                    logger.info(f"  {k}: {v}")
        
        return results
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Build test loaders for both datasets with dataset_id support.
        内存优化版本
        """
        from detectron2.data import build_detection_test_loader
        from detectron2.data.dataset_mapper import DatasetMapper
        
        # 确保数据集已注册，但避免重复注册
        from detectron2.data import DatasetCatalog
        # 检查是否已注册，只在需要时注册
        if "cityscapes_apollo_sem_seg_train" not in DatasetCatalog:
            register_all_double_cityscapes(os.getenv("DETECTRON2_DATASETS", "datasets"))
        
        # Create dataset mapper with dataset_id support
        mapper = DoubleMaskFormerSemanticDatasetMapper(cfg, is_train=False)
        
        # 内存优化参数
        num_workers = 1  # 测试时使用最少的worker以节省内存
        
        return build_detection_test_loader(
            cfg, 
            dataset_name=dataset_name,
            mapper=mapper,
            num_workers=num_workers,
            batch_size=1  # 测试时使用batch size=1以最小化内存使用
        )
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build train loaders for both datasets with dataset_id support.
        内存优化版本
        """
        from detectron2.data import build_detection_train_loader
        from detectron2.data.common import DatasetFromList, MapDataset
        from detectron2.data.dataset_mapper import DatasetMapper
        
        # 确保数据集已注册，但避免重复注册
        from detectron2.data import DatasetCatalog
        # 检查是否已注册，只在需要时注册
        if "cityscapes_apollo_sem_seg_train" not in DatasetCatalog:
            register_all_double_cityscapes(os.getenv("DETECTRON2_DATASETS", "datasets"))
        
        # Get dataset names
        dataset_names = cfg.DATASETS.TRAIN
        
        # Create a combined dataset with dataset_id
        combined_dataset = []
        for dataset_id, dataset_name in enumerate(dataset_names):
            dataset = DatasetCatalog.get(dataset_name)
            print(f"加载数据集 {dataset_name}，包含 {len(dataset)} 个样本")
            for record in dataset:
                # Create a copy of the record and add dataset_id
                record_copy = record.copy()
                record_copy["dataset_id"] = dataset_id
                # 确保数据集名称也被记录
                record_copy["dataset_name"] = dataset_name
                combined_dataset.append(record_copy)
        
        print(f"合并后的数据集总样本数: {len(combined_dataset)}")
        
        # Create dataset mapper with dataset_id support
        mapper = DoubleMaskFormerSemanticDatasetMapper(cfg, is_train=True)
        
        # 内存优化参数
        # 减小worker数量以减少CPU内存占用
        num_workers = max(2, min(os.cpu_count() // 4, 4))
        
        # Build the loader with optimized parameters for memory usage
        return build_detection_train_loader(
            cfg, 
            mapper=mapper,
            dataset=combined_dataset,
            num_workers=num_workers,  
            pin_memory=False,  # Disable pin_memory to reduce CPU memory usage
            batch_size=cfg.SOLVER.IMS_PER_BATCH  
        )
    



def setup(args):
    """
    Create configs and perform basic setups.
    """
    # 导入必要的模块
    from detectron2.config import CfgNode
    cfg = get_cfg()
    # 添加DeepLab配置（如果需要）
    add_deeplab_config(cfg)
    # 添加双分割头配置
    add_double_maskformer_config(cfg)
    
    # 记录是否处于评估模式
    logger.info(f"Setup called with eval_only={args.eval_only}")
    
    # 在加载配置文件之前，确保所有必要的配置组都已存在
    # 1. 确保DATALOADER配置组存在
    if not hasattr(cfg, 'DATALOADER'):
        cfg.DATALOADER = CfgNode()
    
    # 确保PIN_MEMORY配置项存在
    if not hasattr(cfg.DATALOADER, 'PIN_MEMORY'):
        cfg.DATALOADER.PIN_MEMORY = False
    
    # 2. 确保MODEL.DOUBLE_MASK_FORMER配置组存在，避免加载配置文件时出错
    if not hasattr(cfg, 'MODEL'):
        cfg.MODEL = CfgNode()
    if not hasattr(cfg.MODEL, 'DOUBLE_MASK_FORMER'):
        cfg.MODEL.DOUBLE_MASK_FORMER = CfgNode()
    
    # 确保HEAD1配置组存在并添加必要的配置项
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER, 'HEAD1'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1 = CfgNode()
    # 确保CLASS_WEIGHTS配置项存在
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1, 'CLASS_WEIGHTS'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CLASS_WEIGHTS = None
    # 确保其他关键配置项存在
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1, 'IGNORE_VALUE'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.IGNORE_VALUE = 255
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1, 'CONVS_DIM'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.CONVS_DIM = 256
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1, 'LOSS_WEIGHT'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD1.LOSS_WEIGHT = 1.0
    
    # 确保HEAD2配置组存在并添加必要的配置项
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER, 'HEAD2'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2 = CfgNode()
    # 确保CLASS_WEIGHTS配置项存在
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2, 'CLASS_WEIGHTS'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CLASS_WEIGHTS = None
    # 确保其他关键配置项存在
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2, 'IGNORE_VALUE'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.IGNORE_VALUE = 255
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2, 'CONVS_DIM'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.CONVS_DIM = 256
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2, 'LOSS_WEIGHT'):
        cfg.MODEL.DOUBLE_MASK_FORMER.HEAD2.LOSS_WEIGHT = 1.0
    
    # 确保FUSION配置组存在
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER, 'FUSION'):
        cfg.MODEL.DOUBLE_MASK_FORMER.FUSION = CfgNode()
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.FUSION, 'METHOD'):
        cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.METHOD = "weighted_sum"
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.FUSION, 'HEAD1_WEIGHT'):
        cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.HEAD1_WEIGHT = 0.5
    if not hasattr(cfg.MODEL.DOUBLE_MASK_FORMER.FUSION, 'HEAD2_WEIGHT'):
        cfg.MODEL.DOUBLE_MASK_FORMER.FUSION.HEAD2_WEIGHT = 0.5
    
    # 3. 确保SOLVER配置组存在
    if not hasattr(cfg, 'SOLVER'):
        cfg.SOLVER = CfgNode()
    
    # 确保GRADIENT_ACCUMULATION_STEPS配置项存在
    if not hasattr(cfg.SOLVER, 'GRADIENT_ACCUMULATION_STEPS'):
        cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS = 4
    
    # 预定义学习率乘数配置项，避免配置加载错误
    if not hasattr(cfg.SOLVER, 'BACKBONE_MULTIPLIER'):
        cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1
    if not hasattr(cfg.SOLVER, 'HEAD1_LR_MULTIPLIER'):
        cfg.SOLVER.HEAD1_LR_MULTIPLIER = 1.0
    if not hasattr(cfg.SOLVER, 'HEAD2_LR_MULTIPLIER'):
        cfg.SOLVER.HEAD2_LR_MULTIPLIER = 1.0
    if not hasattr(cfg.SOLVER, 'WARMUP_FACTOR'):
        cfg.SOLVER.WARMUP_FACTOR = 0.01
    if not hasattr(cfg.SOLVER, 'WARMUP_ITERS'):
        cfg.SOLVER.WARMUP_ITERS = 10000
    if not hasattr(cfg.SOLVER, 'WARMUP_METHOD'):
        cfg.SOLVER.WARMUP_METHOD = "linear"
    
    # 确保AMP配置组及其INITIAL_SCALE_POWER配置项存在
    if not hasattr(cfg.SOLVER, 'AMP'):
        cfg.SOLVER.AMP = CfgNode()
    if not hasattr(cfg.SOLVER.AMP, 'INITIAL_SCALE_POWER'):
        cfg.SOLVER.AMP.INITIAL_SCALE_POWER = 32.0
    
    # 3. 确保TEST配置组存在
    if not hasattr(cfg, 'TEST'):
        cfg.TEST = CfgNode()
    
    # 确保TEST.AMP配置组及其ENABLED配置项存在
    if not hasattr(cfg.TEST, 'AMP'):
        cfg.TEST.AMP = CfgNode()
    if not hasattr(cfg.TEST.AMP, 'ENABLED'):
        cfg.TEST.AMP.ENABLED = False
    
    # 确保OPTIMIZATION配置组存在，避免加载配置文件时出错
    if not hasattr(cfg, 'OPTIMIZATION'):
        from detectron2.config import CfgNode
        cfg.OPTIMIZATION = CfgNode()
    
    # 从配置文件加载配置
    # 此时配置组已经存在，不会出现Non-existent config key错误
    cfg.merge_from_file(args.config_file)
    # 从命令行参数合并配置
    cfg.merge_from_list(args.opts)
    
    # 内存优化配置
    # 1. 调整batch size以降低GPU内存使用，但确保至少为2以支持BatchNorm层正常工作
    cfg.SOLVER.IMS_PER_BATCH = max(2, cfg.SOLVER.IMS_PER_BATCH // 2)  # 确保batch size至少为2
    print(f"优化后batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    
    # 2. 梯度累积步数已在配置加载前创建，但仍可以在这里调整值
    print(f"梯度累积步数: {cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS}")
    
    # 3. 配置梯度裁剪以提高稳定性并优化内存使用
    if not hasattr(cfg.SOLVER, 'CLIP_GRADIENTS'):
        cfg.SOLVER.CLIP_GRADIENTS = CfgNode()
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
    
    # 4. 在非评估模式下使用自动混合精度训练以节省内存
    if not args.eval_only:
        from detectron2.config import CfgNode
        # 确保SOLVER配置组存在
        if not hasattr(cfg, 'SOLVER'):
            cfg.SOLVER = CfgNode()
        # 确保AMP配置组存在
        if not hasattr(cfg.SOLVER, 'AMP'):
            cfg.SOLVER.AMP = CfgNode()
        # 设置AMP相关配置
        cfg.SOLVER.AMP.ENABLED = True
        # 确保INITIAL_SCALE_POWER配置项存在
        if not hasattr(cfg.SOLVER.AMP, 'INITIAL_SCALE_POWER'):
            cfg.SOLVER.AMP.INITIAL_SCALE_POWER = 32.0  # 增加初始缩放值以提高数值稳定性
        print("自动混合精度训练已启用，使用增强的数值稳定性设置")
    else:
        logger.info("评估模式下禁用训练相关的AMP设置")
    
    # 5. 为推理阶段启用自动混合精度，提高速度
    if not hasattr(cfg, 'TEST'):
        from detectron2.config import CfgNode
        cfg.TEST = CfgNode()
    if not hasattr(cfg.TEST, 'AMP'):
        cfg.TEST.AMP = CfgNode()
    cfg.TEST.AMP.ENABLED = True
    
    # 6. 设置CUDA内存分配器环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    print("CUDA内存分配器优化已配置")
    
    # 在评估模式下禁用训练相关设置
    if args.eval_only:
        logger.info("评估模式下优化配置参数")
        # 设置较小的批处理大小以加速推理
        cfg.DATALOADER.NUM_WORKERS = 1  # 评估模式下减少worker
        # 为评估模式设置适合的参数
        if hasattr(cfg.TEST, 'IMS_PER_BATCH'):
            cfg.TEST.IMS_PER_BATCH = 1  # 单图像推理
        # 确保不会保存训练检查点
        if hasattr(cfg.SOLVER, 'CHECKPOINT_PERIOD'):
            cfg.SOLVER.CHECKPOINT_PERIOD = 0
    else:
        # 仅在训练模式下调整学习率
        effective_batch_size = cfg.SOLVER.IMS_PER_BATCH * cfg.SOLVER.GRADIENT_ACCUMULATION_STEPS
        original_lr = getattr(cfg.SOLVER, 'BASE_LR', 0.0001)
        # 假设原始batch size为4（一个常见值）
        original_batch_size = 4
        cfg.SOLVER.BASE_LR = original_lr * (effective_batch_size / original_batch_size)
        print(f"调整后学习率: {cfg.SOLVER.BASE_LR}")
        
        # 训练模式下的数据加载优化
        cfg.DATALOADER.NUM_WORKERS = 2  # 减少worker数量以降低内存使用
        
        # 启用CUDA流，提高并行性能
        torch.backends.cudnn.benchmark = True  # 启用CUDNN自动调优
        
        # 在freeze之前修改配置
        cfg.SOLVER.CHECKPOINT_PERIOD = max(cfg.SOLVER.CHECKPOINT_PERIOD, 5000)  # 检查点保存频率
        cfg.TEST.EVAL_PERIOD = 0  # 设置为0完全禁用评估步骤
    
    # 确保数据集已注册
    register_all_double_cityscapes(os.getenv("DETECTRON2_DATASETS", "datasets"))
    
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    # 添加命令行参数解析日志
    logger.info(f"Command line arguments parsed: eval_only={args.eval_only}, save_results={hasattr(args, 'save_results') and args.save_results}")
    
    # 先创建配置
    cfg = setup(args)
    
    # 从命令行参数中提取MODEL.WEIGHTS
    weights_from_args = None
    # 检查是否有MODEL.WEIGHTS参数
    for i, arg in enumerate(args.opts):
        if arg == "MODEL.WEIGHTS" and i+1 < len(args.opts):
            weights_from_args = args.opts[i+1]
            logger.info(f"从命令行参数中提取MODEL.WEIGHTS: {weights_from_args}")
            # 如果从命令行参数中提取到权重，直接更新到cfg中
            # 注意：这里需要先解冻配置，修改后再冻结
            cfg.defrost()
            cfg.MODEL.WEIGHTS = weights_from_args
            cfg.freeze()
    
    # 检查是否在评估模式下
    if args.eval_only:
        logger.info("进入评估模式，将不会进行训练")
        
        # 使用Trainer的build_model方法构建模型，与train_net_apo_finetune.py保持一致
        model = DoubleMaskFormerTrainer.build_model(cfg)
        
        # 使用DetectionCheckpointer的resume_or_load方法，与train_net_apo_finetune.py保持一致
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        logger.info(f"开始加载模型权重: {cfg.MODEL.WEIGHTS}")
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        logger.info(f"模型权重加载完成")
        
        # 评估模式下的主要流程：
        # 1. 先检查是否需要保存推理结果（--save-results参数）
        if hasattr(args, 'save_results') and args.save_results:
            logger.info("开始保存推理结果模式")
            DoubleMaskFormerTrainer.save_inference_results(cfg, model)
            logger.info("推理结果保存完成")
        
        # 2. 无论是否保存了推理结果，都执行模型评估
        logger.info("执行模型评估")
        try:
            # 直接调用静态方法进行评估
            res = DoubleMaskFormerTrainer.test(cfg, model)
            if comm.is_main_process():
                verify_results(cfg, res)
            logger.info(f"评估结果: {res}")
            return res
        except Exception as e:
            logger.error(f"评估过程中出现错误: {e}")
            raise
    else:
        # 非评估模式下，记录训练警告
        logger.warning("未设置--eval-only参数，将开始模型训练")
        # 只有在非评估模式下才创建trainer并开始训练
        trainer = DoubleMaskFormerTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        return trainer.train()


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = default_argument_parser()
    # 添加保存推理结果的参数
    parser.add_argument(
        "--save-results",
        "--save_results",
        action="store_true",
        help="保存分割结果到OUTPUT_DIR/segmentation_head1和OUTPUT_DIR/segmentation_head2目录"
    )
    
    args = parser.parse_args()
    logger.info("Command line arguments: " + str(args))
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )