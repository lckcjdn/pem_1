#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双分割头后处理融合脚本

功能：
1. 将两个分割头的分割结果映射回各自的trainId
2. 对分割结果进行形态学处理，减少细小噪声
3. 融合两个分割结果，替换分割头二的lane_mark类别为分割头一结果
4. 创建融合后的标签和ID对应表
5. 可视化融合结果

作者：AI Assistant
日期：2024年
"""

import os
import sys
import numpy as np
import cv2
from collections import namedtuple
import argparse
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 导入项目中的标签定义
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pem', 'data', 'datasets'))
from labels_double import apollo_labels, vestas_labels, Label

# 导入车道线实例化优化模块
try:
    from lane_instance_optimization import LaneInstanceOptimizer
except ImportError:
    print("警告: 无法导入车道线实例化优化模块")
    LaneInstanceOptimizer = None


class PostProcessFusion:
    """双分割头后处理融合类"""
    
    def __init__(self, use_instance_optimization: bool = False):
        """
        初始化后处理融合类
        
        Args:
            use_instance_optimization: 是否使用实例化车道线优化功能
        """
        self.use_instance_optimization = use_instance_optimization
        
        # 创建标签映射
        self.apollo_trainId2label = {label.trainId: label for label in apollo_labels}
        self.vestas_trainId2label = {label.trainId: label for label in vestas_labels}
        
        # 创建融合后的标签定义
        self.fused_labels = self._create_fused_labels()
        self.fused_trainId2label = {label.trainId: label for label in self.fused_labels}
        
        # 形态学处理参数
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # 实例化优化器（如果需要）
        self.lane_optimizer = None
        if self.use_instance_optimization and LaneInstanceOptimizer is not None:
            self.lane_optimizer = LaneInstanceOptimizer()
            print("车道线实例化优化器已初始化")
        elif self.use_instance_optimization:
            print("警告: 无法创建车道线实例化优化器")
        
        print("后处理融合类初始化完成")
        print(f"融合标签数量: {len(self.fused_labels)}")
        print(f"使用实例化优化: {use_instance_optimization}")
    
    def _create_fused_labels(self) -> List[Label]:
        """创建融合后的标签定义"""
        fused_labels = []
        
        # 复制Apollo标签（分割头一的结果）
        for label in apollo_labels:
            if label.trainId == 0:  # 背景
                fused_labels.append(label)
            else:
                # 保持Apollo标签不变
                fused_labels.append(label)
        
        # 添加Vestas标签（除了lane_mark和背景）
        vestas_start_id = len(apollo_labels)  # 从Apollo标签之后开始
        
        # 为Vestas类别分配连续的trainId
        vestas_trainId_counter = 0
        
        for label in vestas_labels:
            if label.trainId == 0:  # 背景，已经包含
                continue
            elif label.name == 'lane_mark':  # lane_mark将被替换，不添加到融合标签
                continue
            else:
                # 创建新的标签，trainId从Apollo标签之后开始
                new_trainId = vestas_start_id + vestas_trainId_counter
                fused_labels.append(Label(
                    name=label.name,
                    id=label.id + 100,  # 避免ID冲突
                    trainId=new_trainId,
                    category=label.category,
                    categoryId=label.categoryId + 10,  # 避免类别ID冲突
                    hasInstances=label.hasInstances,
                    ignoreInEval=label.ignoreInEval,
                    color=label.color
                ))
                vestas_trainId_counter += 1
        
        return fused_labels
    
    def map_to_trainId(self, seg_mask: np.ndarray, is_head1: bool = True) -> np.ndarray:
        """
        将分割结果映射回trainId
        
        Args:
            seg_mask: 分割结果图像（RGB或灰度）
            is_head1: 是否为分割头一的结果
            
        Returns:
            trainId映射结果
        """
        if len(seg_mask.shape) == 3:  # RGB图像
            # 转换为灰度图进行颜色匹配
            if seg_mask.shape[2] == 3:
                # 使用颜色到trainId的映射
                return self._color_to_trainId(seg_mask, is_head1)
            else:
                raise ValueError("不支持的图像格式")
        else:  # 灰度图，直接返回
            return seg_mask
    
    def _color_to_trainId(self, color_mask: np.ndarray, is_head1: bool) -> np.ndarray:
        """将颜色图像映射到trainId"""
        height, width = color_mask.shape[:2]
        trainId_mask = np.zeros((height, width), dtype=np.uint8)
        
        # 转换颜色空间：从BGR到RGB
        color_mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)
        
        # 选择对应的标签集
        labels = apollo_labels if is_head1 else vestas_labels
        
        for label in labels:
            # 创建颜色匹配掩码
            target_color = np.array(label.color, dtype=np.uint8)
            color_diff = np.abs(color_mask_rgb.astype(np.int16) - target_color)
            
            # 计算颜色差异（RGB空间）
            color_distance = np.sum(color_diff, axis=2)
            
            # 找到颜色匹配的像素
            match_mask = color_distance < 20  # 降低阈值以提高匹配精度
            trainId_mask[match_mask] = label.trainId
        
        return trainId_mask
    
    def morphological_processing(self, trainId_mask: np.ndarray) -> np.ndarray:
        """
        形态学处理，减少细小噪声
        
        Args:
            trainId_mask: trainId映射结果
            
        Returns:
            处理后的trainId_mask
        """
        processed_mask = trainId_mask.copy()
        
        # 对每个非背景类别进行形态学处理
        unique_ids = np.unique(trainId_mask)
        
        for trainId in unique_ids:
            if trainId == 0:  # 背景，不处理
                continue
            
            # 创建当前类别的二值掩码
            class_mask = (trainId_mask == trainId).astype(np.uint8)
            
            # 开运算去除小噪声
            opened = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, self.morph_kernel)
            
            # 闭运算填充小孔洞
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)
            
            # 更新处理后的掩码
            processed_mask[class_mask == 1] = 0  # 先移除原类别
            processed_mask[closed == 1] = trainId  # 添加处理后的类别
        
        return processed_mask
    
    def fuse_segmentations(self, head1_mask: np.ndarray, head2_mask: np.ndarray, 
                          original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        融合两个分割结果
        
        融合逻辑：
        1. 如果使用实例化优化，则使用优化后的车道线结果
        2. 否则，替换分割头二的lane_mark类别为分割头一结果
        3. 保留分割头二的其他类别
        4. 忽略背景（trainId=0）
        
        Args:
            head1_mask: 分割头一结果（trainId格式）
            head2_mask: 分割头二结果（trainId格式）
            original_image: 原始图像（用于实例化优化）
            
        Returns:
            融合后的trainId掩码
        """
        if self.use_instance_optimization and self.lane_optimizer is not None:
            # 使用实例化车道线优化
            return self._fuse_with_instance_optimization(head1_mask, head2_mask, original_image)
        else:
            # 使用基础融合策略
            return self._fuse_basic(head1_mask, head2_mask)
    
    def _fuse_basic(self, head1_mask: np.ndarray, head2_mask: np.ndarray) -> np.ndarray:
        """基础融合策略"""
        # 初始化融合结果，使用分割头二的结果作为基础
        fused_mask = head2_mask.copy()
        
        # 找到分割头二中的lane_mark区域 (Vestas中lane_mark的trainId为4)
        lane_mark_mask = (head2_mask == 4)
        
        # 完全替换分割头二的lane_mark区域为分割头一的结果（包括背景）
        if lane_mark_mask.any():
            fused_mask[lane_mark_mask] = head1_mask[lane_mark_mask]
        
        return fused_mask
    
    def _fuse_with_instance_optimization(self, head1_mask: np.ndarray, head2_mask: np.ndarray,
                                       original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """使用实例化车道线优化的融合策略"""
        print("使用实例化车道线优化进行融合...")
        
        # 执行车道线实例化优化
        optimization_result = self.lane_optimizer.optimize_lane_mark(head2_mask, head1_mask)
        
        # 获取优化后的车道线地图
        optimized_lane_map = optimization_result['lane_map']
        
        # 初始化融合结果，使用分割头一的结果作为基础
        fused_mask = head1_mask.copy()
        
        # 将车道线优化图与分割头1进行与操作
        # 找到优化后的车道线区域（非零区域）
        optimized_lane_region = optimized_lane_map > 0
        
        # 在优化后的车道线区域使用优化结果
        fused_mask[optimized_lane_region] = optimized_lane_map[optimized_lane_region]
        
        # 添加分割头二的所有非背景、非lane_mark类别
        vestas_special_categories = {'curb', 'guard_rail'}  # Vestas特有类别
        vestas_common_categories = {'car', 'human', 'road', 'road_mark', 'traffic_sign'}  # 与Apollo共有的类别
        
        for label in vestas_labels:
            if label.trainId == 0:  # 背景
                continue
            elif label.name == 'lane_mark':  # lane_mark已经被优化结果替换
                continue
            elif label.name in vestas_special_categories:
                # Vestas特有类别：只在Apollo对应区域为背景时使用
                # 使用Vestas的原始trainId来查找类别
                fused_trainId = self._get_fused_trainId(label.name)
                if fused_trainId == 0:  # 如果找不到融合标签，跳过
                    continue
                vestas_class_mask = (head2_mask == label.trainId)  # 使用Vestas原始trainId
                apollo_background_mask = (head1_mask == 0)  # Apollo背景区域
                
                # 只在Apollo背景区域使用Vestas特有类别
                valid_mask = vestas_class_mask & apollo_background_mask
                
                if valid_mask.any():
                    fused_mask[valid_mask] = fused_trainId
            elif label.name in vestas_common_categories:
                # 与Apollo共有的类别：直接使用Vestas的结果（优先级更高）
                # 使用Vestas的原始trainId来查找类别
                fused_trainId = self._get_fused_trainId(label.name)
                if fused_trainId == 0:  # 如果找不到融合标签，跳过
                    continue
                vestas_class_mask = (head2_mask == label.trainId)  # 使用Vestas原始trainId
                
                if vestas_class_mask.any():
                    fused_mask[vestas_class_mask] = fused_trainId
        
        # 保存优化结果信息
        self.optimization_result = optimization_result
        
        return fused_mask
    
    def _get_fused_trainId(self, label_name: str) -> int:
        """获取融合标签中的trainId"""
        for label in self.fused_labels:
            if label.name == label_name:
                return label.trainId
        
        # 如果找不到，返回背景
        return 0
    
    def visualize_result(self, fused_mask: np.ndarray, original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        可视化融合结果
        
        Args:
            fused_mask: 融合后的trainId掩码
            original_image: 原始图像（可选，用于创建叠加图）
            
        Returns:
            彩色可视化结果
        """
        height, width = fused_mask.shape
        color_result = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 为每个trainId应用对应的颜色（为分割头一重新分配颜色以提高可视化效果）
        for label in self.fused_labels:
            mask = fused_mask == label.trainId
            
            # 为Apollo类别（trainId 1-15）重新分配更鲜明的颜色
            if 1 <= label.trainId <= 15:
                # 使用与原颜色相似但更鲜明的颜色
                original_color = np.array(label.color)
                # 增加颜色饱和度和亮度
                new_color = original_color.copy()
                new_color = np.minimum(new_color * 1.2, 255).astype(np.uint8)
                color_result[mask] = new_color
            else:
                # 保持其他类别颜色不变
                color_result[mask] = label.color
        
        # 如果提供了原始图像，创建叠加图
        if original_image is not None:
            # 确保原始图像尺寸匹配
            if original_image.shape[:2] != (height, width):
                original_image = cv2.resize(original_image, (width, height))
            
            # 创建叠加图，增强可视化效果
            overlay = cv2.addWeighted(original_image, 0.7, color_result, 0.3, 0)
            return overlay
        
        return color_result
    
    def get_fusion_statistics(self, fused_mask: np.ndarray) -> Dict:
        """获取融合结果的统计信息"""
        stats = {}
        total_pixels = fused_mask.size
        
        # 统计每个类别的像素数量
        for label in self.fused_labels:
            pixel_count = np.sum(fused_mask == label.trainId)
            percentage = (pixel_count / total_pixels) * 100
            
            stats[label.name] = {
                'trainId': label.trainId,
                'pixel_count': pixel_count,
                'percentage': percentage,
                'color': label.color
            }
        
        return stats
    
    def save_fused_labels_info(self, output_path: str):
        """保存融合标签信息到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("融合标签定义\n")
            f.write("=" * 50 + "\n")
            
            for label in self.fused_labels:
                f.write(f"名称: {label.name}\n")
                f.write(f"trainId: {label.trainId}\n")
                f.write(f"类别: {label.category}\n")
                f.write(f"颜色(RGB): {label.color}\n")
                f.write("-" * 30 + "\n")
    
    def process_pipeline(self, head1_seg: np.ndarray, head2_seg: np.ndarray, 
                         original_image: Optional[np.ndarray] = None, 
                         skip_morphological: bool = False) -> Dict:
        """
        完整的后处理流程
        
        Args:
            head1_seg: 分割头一的分割结果
            head2_seg: 分割头二的分割结果
            original_image: 原始图像（可选）
            skip_morphological: 是否跳过形态学处理
            
        Returns:
            处理结果字典
        """
        print("开始后处理流程...")
        
        # 1. 映射到trainId
        print("步骤1: 映射分割结果到trainId")
        head1_trainId = self.map_to_trainId(head1_seg, is_head1=True)
        head2_trainId = self.map_to_trainId(head2_seg, is_head1=False)
        
        # 2. 形态学处理（可选）
        if skip_morphological:
            print("步骤2: 跳过形态学处理")
            head1_processed = head1_trainId
            head2_processed = head2_trainId
        else:
            print("步骤2: 形态学处理")
            head1_processed = self.morphological_processing(head1_trainId)
            head2_processed = self.morphological_processing(head2_trainId)
        
        # 3. 融合分割结果
        print("步骤3: 融合分割结果")
        fused_mask = self.fuse_segmentations(head1_processed, head2_processed, original_image)
        
        # 4. 可视化
        print("步骤4: 可视化结果")
        color_result = self.visualize_result(fused_mask, original_image)
        
        # 5. 统计信息
        print("步骤5: 生成统计信息")
        stats = self.get_fusion_statistics(fused_mask)
        
        result = {
            'head1_trainId': head1_trainId,
            'head2_trainId': head2_trainId,
            'head1_processed': head1_processed,
            'head2_processed': head2_processed,
            'fused_mask': fused_mask,
            'color_result': color_result,
            'statistics': stats
        }
        
        # 6. 如果使用了实例化优化，添加优化结果
        if self.use_instance_optimization and hasattr(self, 'optimization_result'):
            result['optimization_result'] = self.optimization_result
            print(f"车道线实例化优化完成，找到 {self.optimization_result['instance_count']} 个实例")
        
        print("后处理流程完成!")
        return result


def get_image_files(path):
    """获取目录中的所有图像文件"""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        files = []
        for filename in os.listdir(path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                files.append(os.path.join(path, filename))
        return sorted(files)
    else:
        return []

def find_matching_files(files1, files2):
    """找到匹配的文件对"""
    # 基于文件名（不含路径）进行匹配
    file_pairs = []
    
    # 创建文件名到完整路径的映射
    files1_map = {os.path.basename(f): f for f in files1}
    files2_map = {os.path.basename(f): f for f in files2}
    
    # 找到共同的文件名
    common_filenames = set(files1_map.keys()) & set(files2_map.keys())
    
    for filename in sorted(common_filenames):
        file_pairs.append((files1_map[filename], files2_map[filename]))
    
    return file_pairs

def process_single_pair(post_processor, head1_path, head2_path, original_path, output_dir, pair_index, total_pairs, skip_morphological=False):
    """处理单个文件对"""
    print(f"\n处理文件对 {pair_index}/{total_pairs}:")
    print(f"  Head1: {os.path.basename(head1_path)}")
    print(f"  Head2: {os.path.basename(head2_path)}")
    print(f"  跳过形态学处理: {'是' if skip_morphological else '否'}")
    
    # 创建子目录用于保存当前文件对的结果
    filename_base = os.path.splitext(os.path.basename(head1_path))[0]
    pair_output_dir = os.path.join(output_dir, filename_base)
    os.makedirs(pair_output_dir, exist_ok=True)
    
    # 读取输入图像
    head1_seg = cv2.imread(head1_path)
    head2_seg = cv2.imread(head2_path)
    
    if head1_seg is None or head2_seg is None:
        print(f"错误: 无法读取分割结果图像")
        return
    
    original_image = None
    if original_path and os.path.exists(original_path):
        if os.path.isfile(original_path):
            original_image = cv2.imread(original_path)
        else:
            # 在目录中查找匹配的原始图像
            original_files = get_image_files(original_path)
            original_filename = os.path.basename(head1_path).replace('_seg.png', '.png')
            original_filename = original_filename.replace('_seg.jpg', '.jpg')
            
            for orig_file in original_files:
                if os.path.basename(orig_file) == original_filename:
                    original_image = cv2.imread(orig_file)
                    break
    
    # 执行后处理流程
    results = post_processor.process_pipeline(head1_seg, head2_seg, original_image, skip_morphological)
    
    # 保存基础结果
    cv2.imwrite(os.path.join(pair_output_dir, 'fused_mask.png'), results['fused_mask'])
    cv2.imwrite(os.path.join(pair_output_dir, 'color_result.png'), results['color_result'])
    
    # 保存统计信息
    stats_path = os.path.join(pair_output_dir, 'fusion_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("融合结果统计信息\n")
        f.write("=" * 50 + "\n\n")
        
        for label_name, stat in results['statistics'].items():
            f.write(f"{label_name}: {stat['pixel_count']} 像素 ({stat['percentage']:.2f}%)\n")
    
    # 保存融合标签信息
    labels_path = os.path.join(pair_output_dir, 'fused_labels_info.txt')
    post_processor.save_fused_labels_info(labels_path)
    
    # 如果使用了实例化优化，保存优化结果
    if hasattr(post_processor, 'use_instance_optimization') and post_processor.use_instance_optimization and 'optimization_result' in results:
        optimization_dir = os.path.join(pair_output_dir, 'lane_optimization')
        os.makedirs(optimization_dir, exist_ok=True)
        
        # 保存优化结果可视化
        optimizations = post_processor.lane_optimizer.visualize_optimization(
            results['optimization_result'], original_image
        )
        
        for name, img in optimizations.items():
            if len(img.shape) == 2:  # 灰度图
                cv2.imwrite(os.path.join(optimization_dir, f'{name}.png'), img)
            else:  # 彩色图
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(optimization_dir, f'{name}.png'), img_bgr)
        
        # 保存实例信息
        instances_info = []
        for instance in results['optimization_result']['instances']:
            info = {
                'instance_id': instance.instance_id,
                'pixel_count': len(instance.pixels),
                'dominant_class': instance.dominant_class,
                'max_width': instance.max_width,
                'confidence': instance.confidence,
                'bbox': instance.bbox,
                'center': instance.center
            }
            instances_info.append(info)
        
        with open(os.path.join(optimization_dir, 'instances_info.txt'), 'w', encoding='utf-8') as f:
            f.write("车道线实例信息\n")
            f.write("=" * 50 + "\n\n")
            
            for info in instances_info:
                f.write(f"实例ID: {info['instance_id']}\n")
                f.write(f"像素数: {info['pixel_count']}\n")
                f.write(f"主要类别: {info['dominant_class']}\n")
                f.write(f"最大宽度: {info['max_width']:.2f}\n")
                f.write(f"置信度: {info['confidence']:.3f}\n")
                f.write(f"边界框: {info['bbox']}\n")
                f.write(f"中心点: {info['center']}\n")
                f.write("-" * 30 + "\n")
        
        print(f"  车道线优化结果已保存到: {optimization_dir}")
    
    print(f"  结果已保存到: {pair_output_dir}")
    return pair_output_dir

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='双分割头后处理融合')
    parser.add_argument('--head1_seg', type=str, required=True, default="D:\Ar\PEM\output\double_maskformer_apo_regularization\segmentation_head1", help='分割头一的分割结果路径（文件或目录）')
    parser.add_argument('--head2_seg', type=str, required=True, default="D:\Ar\PEM\output\double_maskformer_apo_regularization\segmentation_head2", help='分割头二的分割结果路径（文件或目录）')
    parser.add_argument('--original_image', type=str, default='D:\Ar\PEM\output\double_maskformer_apo_regularization\input_images', help='原始图像路径（文件或目录）')
    parser.add_argument('--output_dir', type=str, default='./fusion_results', help='输出目录')
    parser.add_argument('--use_instance_opt', action='store_true', help='使用实例化车道线优化')
    parser.add_argument('--skip_morphological', action='store_true', help='跳过形态学处理')
    parser.add_argument('--skip_instance_opt', action='store_true', help='跳过实例化车道线优化（仅基础融合）')
    
    args = parser.parse_args()
    
    # 参数冲突检查
    if args.use_instance_opt and args.skip_instance_opt:
        print("错误: 不能同时使用 --use_instance_opt 和 --skip_instance_opt")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定是否使用实例化优化
    use_instance_optimization = args.use_instance_opt and not args.skip_instance_opt
    
    # 初始化后处理类
    post_processor = PostProcessFusion(use_instance_optimization=use_instance_optimization)
    
    # 获取输入文件
    head1_files = get_image_files(args.head1_seg)
    head2_files = get_image_files(args.head2_seg)
    
    if not head1_files or not head2_files:
        print("错误: 无法找到有效的输入文件")
        return
    
    print(f"找到 {len(head1_files)} 个head1文件")
    print(f"找到 {len(head2_files)} 个head2文件")
    print(f"使用实例化优化: {'是' if use_instance_optimization else '否'}")
    print(f"跳过形态学处理: {'是' if args.skip_morphological else '否'}")
    
    # 找到匹配的文件对
    file_pairs = find_matching_files(head1_files, head2_files)
    
    if not file_pairs:
        print("错误: 没有找到匹配的文件对")
        return
    
    print(f"找到 {len(file_pairs)} 个匹配的文件对")
    
    # 处理每个文件对
    processed_dirs = []
    for i, (head1_path, head2_path) in enumerate(file_pairs, 1):
        output_dir = process_single_pair(
            post_processor, head1_path, head2_path, 
            args.original_image, args.output_dir, i, len(file_pairs),
            skip_morphological=args.skip_morphological
        )
        if output_dir:
            processed_dirs.append(output_dir)
    
    print(f"\n所有处理完成!")
    print(f"总处理文件对: {len(processed_dirs)}")
    print(f"结果保存在: {args.output_dir}")
    
    # 生成汇总报告
    summary_path = os.path.join(args.output_dir, 'processing_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("后处理融合汇总报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件对数量: {len(processed_dirs)}\n")
        f.write(f"使用实例化优化: {'是' if use_instance_optimization else '否'}\n")
        f.write(f"跳过形态学处理: {'是' if args.skip_morphological else '否'}\n")
        f.write(f"仅基础融合: {'是' if args.skip_instance_opt else '否'}\n\n")
        
        f.write("处理结果目录:\n")
        for dir_path in processed_dirs:
            f.write(f"  {os.path.basename(dir_path)}\n")
    
    print(f"汇总报告: {summary_path}")


if __name__ == "__main__":
    main()