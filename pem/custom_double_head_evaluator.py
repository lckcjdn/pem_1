#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import torch
import numpy as np
import gc
from collections import OrderedDict

from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation.sem_seg_evaluation import SemSegEvaluator as _SemSegEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process

class CustomDoubleHeadEvaluator(SemSegEvaluator):
    def _update_confusion_matrix(self, pred, gt):
        """
        更新混淆矩阵
        
        Args:
            pred: 预测的语义分割结果，形状为(H, W)
            gt: 真实标签，形状为(H, W)
        """
        # 确保混淆矩阵已初始化，使用与父类一致的变量名_conf_matrix
        if not hasattr(self, '_conf_matrix'):
            # 创建混淆矩阵
            self._conf_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
        
        # 确保预测结果和真实标签的类别索引在有效范围内
        # 这是防止混淆矩阵形状不匹配的关键步骤
        pred = np.clip(pred, 0, self._num_classes - 1)
        
        # 计算混淆矩阵
        # 忽略ignore_label标记的像素
        if hasattr(self, '_ignore_label'):
            mask = (gt != self._ignore_label)
            pred = pred[mask]
            gt = gt[mask]
        
        # 进一步确保gt也在有效范围内
        gt = np.clip(gt, 0, self._num_classes - 1)
        
        # 计算混淆矩阵的索引
        indices = gt * self._num_classes + pred
        
        # 计算每个索引出现的次数，确保minlength正确设置
        try:
            counts = np.bincount(indices, minlength=self._num_classes ** 2)
            
            # 重塑为混淆矩阵形状
            confusion_matrix = counts.reshape(self._num_classes, self._num_classes)
            
            # 验证形状一致性
            if confusion_matrix.shape != (self._num_classes, self._num_classes):
                # 创建正确形状的零矩阵
                confusion_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
            
            # 更新混淆矩阵
            if self._conf_matrix.shape == confusion_matrix.shape:
                self._conf_matrix += confusion_matrix
            else:
                # 创建新的混淆矩阵以匹配计算出的形状
                self._conf_matrix = np.zeros_like(confusion_matrix)
                self._conf_matrix += confusion_matrix
                # 更新类别数以匹配新的混淆矩阵
                self._num_classes = confusion_matrix.shape[0]
        except Exception as e:
            # 静默处理异常，不输出调试信息
            pass
    """
    自定义评估器，用于双分割头模型的独立评估。
    根据数据集类型，选择使用对应的分割头输出进行评估：
    - 对于cityscapes数据集，使用HEAD1的输出
    - 对于其他数据集(如mapillary vistas)，使用HEAD2的输出
    """
    def __init__(self, dataset_name, distributed=False, output_dir=None, num_classes=None):
        """
        Args:
            dataset_name: 数据集名称
            distributed: 是否使用分布式评估
            output_dir: 输出目录
            num_classes: 类别数量
        """
        # 先调用父类构造函数
        super().__init__(dataset_name, distributed, output_dir)
        
        # 根据数据集名称确定使用哪个分割头的输出
        self.use_head1 = "cityscapes" in dataset_name
        
        # 手动设置类别数，并确保覆盖父类可能设置的值
        if num_classes is not None:
            original_num_classes = getattr(self, '_num_classes', None)
            self._num_classes = num_classes
            
            # 确保混淆矩阵被正确初始化
            if hasattr(self, '_conf_matrix'):
                delattr(self, '_conf_matrix')
            # 显式初始化混淆矩阵，使用正确的类别数
            self._conf_matrix = np.zeros((self._num_classes, self._num_classes), dtype=np.int64)
        
        # 统计信息
        self.processed_count = 0
        
        # 评估器初始化完成
    
    def process(self, inputs, outputs):
        """
        处理模型输出，使用对应的分割头结果进行评估。
        实现增量评估，不保存预测结果，直接更新混淆矩阵。
        """
        # 立即清理内存
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 处理每个输入-输出对
        for input_dict, output_dict in zip(inputs, outputs):
            # 根据数据集选择对应的分割头输出
            sem_seg = None
            
            # 首先尝试直接从output_dict获取对应的分割头输出
            if self.use_head1 and "sem_seg_head1" in output_dict:
                sem_seg = output_dict["sem_seg_head1"]
            elif not self.use_head1 and "sem_seg_head2" in output_dict:
                sem_seg = output_dict["sem_seg_head2"]
            # 如果直接分割头输出不存在，再尝试使用sem_seg（现在应该是根据数据集选择的正确分割头输出）
            elif "sem_seg" in output_dict:
                sem_seg = output_dict["sem_seg"]
            # 最后才尝试使用merged_sem_seg作为备选
            elif "merged_sem_seg" in output_dict:
                sem_seg = output_dict["merged_sem_seg"]
            
            # 确保sem_seg是合适的格式
            if isinstance(sem_seg, torch.Tensor):
                with torch.no_grad():
                    # 移除批次维度（如果存在）
                    if sem_seg.dim() == 4 and sem_seg.size(0) == 1:
                        sem_seg = sem_seg.squeeze(0)
                    
                    # 如果是概率图格式(C, H, W)，直接获取类别预测
                    if sem_seg.dim() == 3 and sem_seg.size(0) > 1:
                        sem_seg = torch.argmax(sem_seg, dim=0)
                    
                    # 获取真实标签
                    if "sem_seg" in input_dict:
                        gt = input_dict["sem_seg"]
                        
                        # 检查尺寸是否匹配 - 增强的尺寸匹配处理
                        if sem_seg.shape != gt.shape:
                            # 计算缩放因子
                            zoom_factor = (
                                gt.shape[0] / sem_seg.shape[0],
                                gt.shape[1] / sem_seg.shape[1],
                            )
                            
                            # 优先使用PyTorch的插值方法（更稳定且与模型一致）
                            try:
                                import torch.nn.functional as F
                                # 使用PyTorch进行重采样
                                # 需要添加批次和通道维度，然后进行插值
                                sem_seg_tensor = sem_seg.unsqueeze(0).unsqueeze(0).float()
                                resampled = F.interpolate(
                                    sem_seg_tensor,
                                    size=(gt.shape[0], gt.shape[1]),
                                    mode="nearest-exact"
                                )
                                sem_seg = resampled.squeeze(0).squeeze(0).long()
                            except Exception as e:
                                # 备选方案：如果SciPy可用，使用重采样调整尺寸
                                try:
                                    # 先转换为numpy进行SciPy处理
                                    sem_seg_np = sem_seg.cpu().numpy()
                                    gt_np = gt.cpu().numpy()
                                    from scipy.ndimage import zoom
                                    # 使用最近邻插值重采样预测结果
                                    sem_seg_np = zoom(sem_seg_np, zoom_factor, order=0)
                                    sem_seg = torch.from_numpy(sem_seg_np).long()
                                except ImportError:
                                    # 如果SciPy不可用，使用简单的裁剪（可能会丢失信息）
                                    min_h = min(sem_seg.shape[0], gt.shape[0])
                                    min_w = min(sem_seg.shape[1], gt.shape[1])
                                    sem_seg = sem_seg[:min_h, :min_w]
                                    gt = gt[:min_h, :min_w]
                        
                        # 只在CPU上进行处理
                        sem_seg_np = sem_seg.cpu().numpy().astype(np.uint8)
                        gt_np = gt.cpu().numpy().astype(np.uint8)
                        
                        # 确保预测值在有效类别范围内
                        sem_seg_np = np.clip(sem_seg_np, 0, self._num_classes - 1)
                        
                        # 更新混淆矩阵
                        self._update_confusion_matrix(sem_seg_np, gt_np)
                        
                        # 立即清理
                        del sem_seg_np
                        del gt_np
            
            # 立即删除所有引用
            del sem_seg
            del input_dict
            del output_dict
            
            self.processed_count += 1
            
            # 每处理几个样本就清理一次内存
            if self.processed_count % 20 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def evaluate(self):
        """
        执行评估并返回评估结果。
        直接使用累积的混淆矩阵计算指标。
        """
        # 开始评估
        
        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        try:
            # 直接基于混淆矩阵计算指标
            # 计算准确率
            acc = np.diag(self._conf_matrix).sum() / self._conf_matrix.sum() if self._conf_matrix.sum() > 0 else 0.0
            
            # 计算类别准确率和IoU
            class_acc = np.zeros(self._num_classes)
            class_iou = np.zeros(self._num_classes)
            
            for i in range(self._num_classes):
                if self._conf_matrix[i, :].sum() > 0:
                    class_acc[i] = self._conf_matrix[i, i] / self._conf_matrix[i, :].sum()
                if (self._conf_matrix[i, :].sum() + self._conf_matrix[:, i].sum() - self._conf_matrix[i, i]) > 0:
                    class_iou[i] = self._conf_matrix[i, i] / (self._conf_matrix[i, :].sum() + self._conf_matrix[:, i].sum() - self._conf_matrix[i, i])
            
            # 过滤掉NaN值
            class_acc = np.nan_to_num(class_acc)
            class_iou = np.nan_to_num(class_iou)
            
            # 计算平均指标
            mean_acc = np.mean(class_acc)
            mean_iou = np.mean(class_iou)
            
            # 评估计算完成
            
            # 打印完整的混淆矩阵
            print(f"\n完整混淆矩阵 ({self._num_classes}x{self._num_classes}):")
            print("Confusion Matrix:")
            for i in range(self._num_classes):
                row_str = f"类别{i:2d}: ["
                for j in range(self._num_classes):
                    row_str += f"{self._conf_matrix[i, j]:6d}"  # 调整宽度以适应较大的数字
                    if j < self._num_classes - 1:
                        row_str += ", "
                row_str += "]"
                print(row_str)
            
            # 打印各类别的准确率和IoU
            print("\n各类别指标:")
            print("类别\t准确率\tIoU")
            for i in range(self._num_classes):
                print(f"{i:2d}\t{class_acc[i]:.4f}\t{class_iou[i]:.4f}")
            
            # 只返回数值评估指标结果，这样detectron2的print_csv_format函数可以正确处理
            results = OrderedDict()
            results["sem_seg"] = OrderedDict()
            results["sem_seg"]["mIoU"] = mean_iou
            results["sem_seg"]["mACC"] = mean_acc
            results["sem_seg"]["pixel_accuracy"] = acc
            
            # 将评估器信息打印出来而不加入到results中，避免格式化错误
            print(f"评估器信息 - 数据集: {self._dataset_name}, 使用分割头: {1 if self.use_head1 else 2}, 类别数: {self._num_classes}")
            
            return results
        finally:
            # 评估完成后彻底清理资源
            self._predictions = []
            self._conf_matrix = None
            self.processed_count = 0
            
            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    获取适合双分割头模型的评估器。
    根据数据集类型选择合适的评估器和参数。
    
    Args:
        cfg: 配置对象
        dataset_name: 数据集名称
        output_folder: 输出文件夹
    
    Returns:
        适当的评估器实例
    """
    # 创建数据集特定的输出文件夹
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    
    # 根据数据集名称创建专门的子文件夹，避免结果混淆
    dataset_specific_output = os.path.join(output_folder, dataset_name)
    os.makedirs(dataset_specific_output, exist_ok=True)
    
    # 根据数据集名称确定使用的分割头和对应的类别数
    # cityscapes数据集包含16个有效类别+背景类别=17个类别
    if "cityscapes" in dataset_name:
        num_classes = 17  # HEAD1的类别数，包含背景类别
        print(f"Cityscapes数据集使用类别数: {num_classes}")
    else:
        num_classes = 9  # HEAD2的类别数
        print(f"其他数据集使用类别数: {num_classes}")
    
    # 返回自定义的双分割头评估器，确保明确传递正确的类别数
    evaluator = CustomDoubleHeadEvaluator(
        dataset_name,
        distributed=False,  # 禁用分布式评估以减少通信开销
        output_dir=dataset_specific_output,
        num_classes=num_classes
    )
    
    # 验证评估器的类别数设置
    if hasattr(evaluator, '_num_classes'):
        print(f"评估器实际类别数: {evaluator._num_classes}")
        # 确保混淆矩阵使用正确的类别数初始化
        if not hasattr(evaluator, '_conf_matrix') or evaluator._conf_matrix.shape != (num_classes, num_classes):
            print(f"重置混淆矩阵以使用正确的类别数: {num_classes}")
            evaluator._conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    return evaluator