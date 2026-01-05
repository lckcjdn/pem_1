#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车道线实例化优化脚本

功能：
1. 从分割头二中提取lane_mark类别
2. 通过骨架化和方向生长实现车道线实例化
3. 记录每个实例的最大宽度
4. 根据分割头一结果赋予细致的语义信息
5. 根据最大宽度进行膨胀操作，得到优化结果

作者：AI Assistant
日期：2024年
"""

import os
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import namedtuple
import sys

# 导入项目中的标签定义
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pem', 'data', 'datasets'))
from labels_double import apollo_labels


class LaneInstance:
    """车道线实例类"""
    
    def __init__(self, instance_id: int):
        self.instance_id = instance_id
        self.pixels = []  # 实例中的像素坐标
        self.skeleton_points = []  # 骨架点
        self.directions = []  # 方向向量
        self.max_width = 0  # 最大宽度
        self.dominant_class = 0  # 主要类别（来自分割头一）
        self.confidence = 0.0  # 置信度
    
    def add_pixel(self, x: int, y: int):
        """添加像素到实例"""
        self.pixels.append((x, y))
    
    def calculate_statistics(self):
        """计算实例的统计信息"""
        if not self.pixels:
            return
        
        # 计算边界框
        xs = [p[0] for p in self.pixels]
        ys = [p[1] for p in self.pixels]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))
        
        # 计算中心点
        self.center = (int(np.mean(xs)), int(np.mean(ys)))


class LaneInstanceOptimizer:
    """车道线实例化优化类"""
    
    def __init__(self, min_instance_size: int = 50, max_gap_length: int = 10):
        """
        初始化车道线实例化优化类
        
        Args:
            min_instance_size: 最小实例大小（像素数）
            max_gap_length: 最大间隙长度（用于连接断开的车道线）
        """
        self.min_instance_size = min_instance_size
        self.max_gap_length = max_gap_length
        
        # Apollo标签映射
        self.apollo_trainId2label = {label.trainId: label for label in apollo_labels}
        
        # 形态学处理核
        self.small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        print("车道线实例化优化类初始化完成")
    
    def extract_lane_mark_mask(self, head2_mask: np.ndarray) -> np.ndarray:
        """
        从分割头二结果中提取lane_mark掩码
        
        Args:
            head2_mask: 分割头二结果（trainId格式）
            
        Returns:
            lane_mark二值掩码
        """
        # Vestas中lane_mark的trainId为4
        lane_mark_mask = (head2_mask == 4).astype(np.uint8)
        
        # 形态学处理，去除噪声
        cleaned_mask = cv2.morphologyEx(lane_mark_mask, cv2.MORPH_OPEN, self.small_kernel)
        
        return cleaned_mask
    
    def skeletonize_lane_mark(self, lane_mark_mask: np.ndarray) -> np.ndarray:
        """
        对lane_mark进行骨架化
        
        Args:
            lane_mark_mask: lane_mark二值掩码
            
        Returns:
            骨架化结果
        """
        # 使用替代的骨架化方法（不使用ximgproc模块）
        skeleton = self._morphological_skeletonize(lane_mark_mask)
        
        return skeleton
    
    def _morphological_skeletonize(self, img: np.ndarray) -> np.ndarray:
        """
        使用形态学操作进行骨架化
        
        Args:
            img: 输入二值图像
            
        Returns:
            骨架化结果
        """
        skeleton = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        done = False
        temp_img = img.copy()
        
        while not done:
            eroded = cv2.erode(temp_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(temp_img, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            temp_img = eroded.copy()
            
            if cv2.countNonZero(temp_img) == 0:
                done = True
        
        return skeleton
    
    def _connect_nearby_instances(self, instances: List[LaneInstance]) -> List[LaneInstance]:
        """
        连接相邻的车道线实例，避免优化后断开
        
        Args:
            instances: 原始实例列表
            
        Returns:
            连接后的实例列表
        """
        if len(instances) <= 1:
            return instances
        
        # 计算实例之间的距离
        connected_instances = []
        merged = set()
        
        for i in range(len(instances)):
            if i in merged:
                continue
            
            current_instance = instances[i]
            
            # 查找与当前实例相邻的实例
            for j in range(i + 1, len(instances)):
                if j in merged:
                    continue
                
                other_instance = instances[j]
                
                # 计算两个实例边界框之间的距离
                x1_min, y1_min, x1_max, y1_max = current_instance.bbox
                x2_min, y2_min, x2_max, y2_max = other_instance.bbox
                
                # 计算水平距离和垂直距离
                dx = max(0, max(x1_min, x2_min) - min(x1_max, x2_max))
                dy = max(0, max(y1_min, y2_min) - min(y1_max, y2_max))
                
                # 如果实例之间的距离小于最大间隙长度，则合并
                if dx <= self.max_gap_length and dy <= self.max_gap_length:
                    # 合并实例
                    for pixel in other_instance.pixels:
                        current_instance.add_pixel(pixel[0], pixel[1])
                    
                    # 重新计算统计信息
                    current_instance.calculate_statistics()
                    
                    merged.add(j)
                    print(f"合并实例 {i+1} 和实例 {j+1}")
            
            connected_instances.append(current_instance)
        
        print(f"实例连接完成: {len(instances)} -> {len(connected_instances)} 个实例")
        return connected_instances
    
    def find_lane_instances(self, skeleton: np.ndarray) -> List[LaneInstance]:
        """
        从骨架中查找车道线实例，并连接相邻的实例
        
        Args:
            skeleton: 骨架化结果
            
        Returns:
            车道线实例列表
        """
        # 使用连通组件分析找到骨架分支
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            skeleton, connectivity=8
        )
        
        instances = []
        
        for i in range(1, num_labels):  # 跳过背景（标签0）
            instance_mask = (labels == i).astype(np.uint8)
            
            # 检查实例大小
            if np.sum(instance_mask) < self.min_instance_size:
                continue
            
            # 创建车道线实例
            instance = LaneInstance(i)
            
            # 添加像素坐标
            y_coords, x_coords = np.where(instance_mask)
            for y, x in zip(y_coords, x_coords):
                instance.add_pixel(x, y)
            
            # 计算统计信息
            instance.calculate_statistics()
            
            instances.append(instance)
        
        # 连接相邻的实例，避免优化后断开
        instances = self._connect_nearby_instances(instances)
        
        return instances
    
    def calculate_directions(self, instances: List[LaneInstance], 
                           lane_mark_mask: np.ndarray) -> List[LaneInstance]:
        """
        计算每个实例的方向
        
        Args:
            instances: 车道线实例列表
            lane_mark_mask: 原始lane_mark掩码
            
        Returns:
            更新后的实例列表
        """
        for instance in instances:
            if len(instance.pixels) < 2:
                continue
            
            # 使用主成分分析（PCA）计算主要方向
            points = np.array(instance.pixels)
            
            # 计算协方差矩阵
            cov_matrix = np.cov(points.T)
            
            # 计算特征值和特征向量
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # 主要方向是最大特征值对应的特征向量
            main_direction = eigenvectors[:, np.argmax(eigenvalues)]
            instance.directions = [main_direction]
            
            # 计算最大宽度
            instance.max_width = self._calculate_max_width(instance, lane_mark_mask)
        
        return instances
    
    def _calculate_max_width(self, instance: LaneInstance, lane_mark_mask: np.ndarray) -> float:
        """计算实例的最大宽度"""
        if not instance.pixels:
            return 0
        
        # 创建实例的边界框掩码
        x_min, y_min, x_max, y_max = instance.bbox
        
        # 检查边界框是否有效
        if x_max <= x_min or y_max <= y_min:
            return 0
        
        instance_region = lane_mark_mask[y_min:y_max+1, x_min:x_max+1]
        
        # 检查实例区域是否为空
        if instance_region.size == 0 or np.sum(instance_region) == 0:
            return 0
        
        # 计算距离变换
        distance_transform = cv2.distanceTransform(
            instance_region, cv2.DIST_L2, 5
        )
        
        # 获取最大距离值
        max_distance = float(np.max(distance_transform))
        
        # 添加合理的边界检查，避免异常值
        if max_distance <= 0 or max_distance > 1000:  # 最大宽度限制为1000像素
            return 0
        
        # 最大宽度是距离变换最大值的两倍
        max_width = max_distance * 2
        
        return max_width
    
    def assign_semantic_classes(self, instances: List[LaneInstance], 
                              head1_mask: np.ndarray) -> List[LaneInstance]:
        """
        根据分割头一结果赋予语义类别，并过滤斑马线和chevron类别
        使用整条车道线中最多的类别作为统一语义，避免多种语义时断开
        
        Args:
            instances: 车道线实例列表
            head1_mask: 分割头一结果（trainId格式）
            
        Returns:
            更新后的实例列表（已过滤斑马线和chevron类别）
        """
        filtered_instances = []
        
        # 统计所有实例的类别分布，用于全局语义统一
        global_class_counts = {}
        
        for instance in instances:
            # 统计实例区域内分割头一的类别分布
            class_counts = {}
            
            for x, y in instance.pixels:
                if 0 <= y < head1_mask.shape[0] and 0 <= x < head1_mask.shape[1]:
                    class_id = head1_mask[y, x]
                    if class_id != 0:  # 忽略背景
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        global_class_counts[class_id] = global_class_counts.get(class_id, 0) + 1
            
            if class_counts:
                # 选择出现次数最多的类别
                dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
                instance.dominant_class = dominant_class
                instance.confidence = class_counts[dominant_class] / len(instance.pixels)
                
                # 检查是否为斑马线（trainId=10）或chevron（trainId=8）
                if dominant_class == 10 or dominant_class == 8:  # zebra或chevron
                    print(f"实例 {instance.instance_id} 为斑马线或chevron类别，已删除")
                    continue  # 跳过这个实例，不添加到结果中
            else:
                # 如果没有匹配的类别，使用默认的车道线类别
                instance.dominant_class = 1  # 默认使用s_w_d
                instance.confidence = 0.0
            
            filtered_instances.append(instance)
        
        # 如果存在多个实例，检查是否需要统一语义类别
        if len(filtered_instances) > 1 and global_class_counts:
            # 找到全局出现次数最多的类别
            global_dominant_class = max(global_class_counts.items(), key=lambda x: x[1])[0]
            
            # 检查是否需要统一语义
            different_classes = set(instance.dominant_class for instance in filtered_instances)
            if len(different_classes) > 1:
                print(f"检测到多个语义类别: {different_classes}，统一使用全局最多类别: {global_dominant_class}")
                
                # 统一所有实例的语义类别
                for instance in filtered_instances:
                    instance.dominant_class = global_dominant_class
                    # 重新计算置信度
                    instance.confidence = global_class_counts.get(global_dominant_class, 0) / sum(global_class_counts.values())
        
        return filtered_instances
    
    def grow_lane_instances(self, instances: List[LaneInstance], 
                           lane_mark_mask: np.ndarray) -> np.ndarray:
        """
        根据实例信息直接加宽生长车道线，避免使用溶蚀等断开操作
        
        Args:
            instances: 车道线实例列表
            lane_mark_mask: 原始lane_mark掩码
            
        Returns:
            优化后的车道线掩码
        """
        optimized_mask = np.zeros_like(lane_mark_mask)
        
        # 直接对每个实例进行加宽生长
        for instance in instances:
            if instance.max_width < 1:
                continue
            
            # 根据最大宽度创建膨胀核
            try:
                max_width = float(instance.max_width)
                # 使用最大宽度作为kernel_size，确保加宽到最大宽度
                kernel_size = max(3, min(int(max_width), 100))
                
                # 调试信息
                print(f"实例 {instance.instance_id}: max_width={max_width}, kernel_size={kernel_size}")
                
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
            except (ValueError, TypeError) as e:
                print(f"警告: 实例 {instance.instance_id} 的max_width值无效: {instance.max_width}, 使用默认值3")
                kernel_size = 3
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                )
            
            # 创建实例掩码
            instance_mask = np.zeros_like(lane_mark_mask)
            for x, y in instance.pixels:
                instance_mask[y, x] = 1
            
            # 使用圆形膨胀将车道线直接加宽到最大宽度
            grown_instance = cv2.dilate(instance_mask, kernel)
            
            # 添加到优化掩码
            optimized_mask = np.logical_or(optimized_mask, grown_instance)
        
        return optimized_mask.astype(np.uint8)
    
    def _create_directional_kernel(self, size: int, angle: float) -> np.ndarray:
        """创建方向性膨胀核"""
        size = int(size)
        kernel = np.zeros((size, size), dtype=np.uint8)
        center = size // 2
        
        # 根据角度创建线性结构
        angle_rad = np.radians(angle)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        for i in range(size):
            x = int(center + (i - center) * dx)
            y = int(center + (i - center) * dy)
            
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1
        
        # 进行小范围膨胀以连接断点
        kernel = cv2.dilate(kernel, self.small_kernel)
        
        return kernel
    
    def create_optimized_lane_map(self, instances: List[LaneInstance], 
                                optimized_mask: np.ndarray) -> np.ndarray:
        """
        创建优化后的车道线地图（包含语义信息）
        
        Args:
            instances: 车道线实例列表
            optimized_mask: 优化后的车道线掩码
            
        Returns:
            包含语义信息的车道线地图（trainId格式）
        """
        lane_map = np.zeros_like(optimized_mask, dtype=np.uint8)
        
        for instance in instances:
            # 创建实例掩码
            instance_mask = np.zeros_like(optimized_mask)
            for x, y in instance.pixels:
                if 0 <= y < optimized_mask.shape[0] and 0 <= x < optimized_mask.shape[1]:
                    instance_mask[y, x] = 1
            
            # 根据最大宽度创建膨胀核
            kernel_size = max(3, min(int(instance.max_width), 100))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # 将实例掩码加宽到最大宽度
            grown_instance_mask = cv2.dilate(instance_mask, kernel)
            
            # 应用语义类别
            instance_class_mask = grown_instance_mask * instance.dominant_class
            
            # 合并到车道线地图
            lane_map = np.maximum(lane_map, instance_class_mask)
        
        return lane_map
    
    def optimize_lane_mark(self, head2_mask: np.ndarray, head1_mask: np.ndarray) -> Dict:
        """
        完整的车道线实例化优化流程
        
        Args:
            head2_mask: 分割头二结果（trainId格式）
            head1_mask: 分割头一结果（trainId格式）
            
        Returns:
            优化结果字典
        """
        print("开始车道线实例化优化...")
        
        # 1. 提取lane_mark掩码
        print("步骤1: 提取lane_mark掩码")
        lane_mark_mask = self.extract_lane_mark_mask(head2_mask)
        
        # 2. 骨架化
        print("步骤2: 骨架化处理")
        skeleton = self.skeletonize_lane_mark(lane_mark_mask)
        
        # 3. 查找车道线实例
        print("步骤3: 查找车道线实例")
        instances = self.find_lane_instances(skeleton)
        print(f"找到 {len(instances)} 个车道线实例")
        
        # 4. 计算方向和宽度
        print("步骤4: 计算方向和宽度")
        instances = self.calculate_directions(instances, lane_mark_mask)
        
        # 5. 赋予语义类别
        print("步骤5: 赋予语义类别")
        instances = self.assign_semantic_classes(instances, head1_mask)
        
        # 6. 生长车道线实例
        print("步骤6: 生长车道线实例")
        optimized_mask = self.grow_lane_instances(instances, lane_mark_mask)
        
        # 7. 创建优化后的车道线地图
        print("步骤7: 创建优化车道线地图")
        lane_map = self.create_optimized_lane_map(instances, optimized_mask)
        
        result = {
            'lane_mark_mask': lane_mark_mask,
            'skeleton': skeleton,
            'instances': instances,
            'optimized_mask': optimized_mask,
            'lane_map': lane_map,
            'instance_count': len(instances)
        }
        
        print("车道线实例化优化完成!")
        return result
    
    def visualize_optimization(self, result: Dict, original_image: Optional[np.ndarray] = None) -> Dict:
        """
        可视化优化结果
        
        Args:
            result: 优化结果字典
            original_image: 原始图像（可选）
            
        Returns:
            可视化结果字典
        """
        visualizations = {}
        
        # 1. 原始lane_mark掩码可视化
        lane_mark_vis = result['lane_mark_mask'] * 255
        visualizations['lane_mark'] = lane_mark_vis
        
        # 2. 骨架可视化
        skeleton_vis = result['skeleton'] * 255
        visualizations['skeleton'] = skeleton_vis
        
        # 3. 优化后的掩码可视化
        optimized_vis = result['optimized_mask'] * 255
        visualizations['optimized'] = optimized_vis
        
        # 4. 车道线地图可视化（使用Apollo颜色）
        lane_map_vis = np.zeros((*result['lane_map'].shape, 3), dtype=np.uint8)
        for trainId, label in self.apollo_trainId2label.items():
            if trainId == 0:  # 背景
                continue
            mask = result['lane_map'] == trainId
            lane_map_vis[mask] = label.color
        visualizations['lane_map'] = lane_map_vis
        
        # 5. 实例可视化（不同颜色表示不同实例）
        if result['instances']:
            instance_vis = np.zeros((*result['lane_mark_mask'].shape, 3), dtype=np.uint8)
            
            # 为每个实例分配不同颜色
            colors = self._generate_instance_colors(len(result['instances']))
            
            for i, instance in enumerate(result['instances']):
                color = colors[i]
                for x, y in instance.pixels:
                    if 0 <= y < instance_vis.shape[0] and 0 <= x < instance_vis.shape[1]:
                        instance_vis[y, x] = color
            
            visualizations['instances'] = instance_vis
        
        # 6. 叠加可视化（如果提供了原始图像）
        if original_image is not None:
            # 调整图像尺寸
            if original_image.shape[:2] != result['lane_map'].shape:
                original_image = cv2.resize(original_image, 
                                         (result['lane_map'].shape[1], result['lane_map'].shape[0]))
            
            # 车道线地图叠加
            lane_map_overlay = cv2.addWeighted(original_image, 0.7, 
                                             visualizations['lane_map'], 0.3, 0)
            visualizations['lane_map_overlay'] = lane_map_overlay
        
        return visualizations
    
    def _generate_instance_colors(self, num_instances: int) -> List[Tuple[int, int, int]]:
        """生成实例可视化颜色"""
        import colorsys
        
        colors = []
        for i in range(num_instances):
            # 使用HSV颜色空间生成不同颜色
            hue = i / max(1, num_instances)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
            color = tuple(int(c * 255) for c in rgb)
            colors.append(color)
        
        return colors
    
    def save_optimization_results(self, result: Dict, output_dir: str):
        """保存优化结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存可视化结果
        visualizations = self.visualize_optimization(result)
        
        for name, img in visualizations.items():
            if len(img.shape) == 2:  # 灰度图
                cv2.imwrite(os.path.join(output_dir, f'{name}.png'), img)
            else:  # 彩色图
                # 转换BGR格式
                if img.shape[2] == 3:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, f'{name}.png'), img_bgr)
        
        # 保存实例信息
        instances_info = []
        for instance in result['instances']:
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
        
        # 保存到文本文件
        with open(os.path.join(output_dir, 'instances_info.txt'), 'w', encoding='utf-8') as f:
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


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='车道线实例化优化')
    parser.add_argument('--head2_seg', type=str, required=True, help='分割头二的分割结果路径')
    parser.add_argument('--head1_seg', type=str, required=True, help='分割头一的分割结果路径')
    parser.add_argument('--original_image', type=str, help='原始图像路径')
    parser.add_argument('--output_dir', type=str, default='./lane_optimization_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 初始化优化器
    optimizer = LaneInstanceOptimizer()
    
    # 读取输入图像
    head2_mask = cv2.imread(args.head2_seg, cv2.IMREAD_GRAYSCALE)
    head1_mask = cv2.imread(args.head1_seg, cv2.IMREAD_GRAYSCALE)
    
    if head2_mask is None or head1_mask is None:
        print("错误: 无法读取分割结果图像")
        return
    
    original_image = None
    if args.original_image:
        original_image = cv2.imread(args.original_image)
    
    # 执行优化流程
    result = optimizer.optimize_lane_mark(head2_mask, head1_mask)
    
    # 保存结果
    optimizer.save_optimization_results(result, args.output_dir)
    
    print(f"优化结果已保存到: {args.output_dir}")
    print(f"找到 {result['instance_count']} 个车道线实例")


if __name__ == "__main__":
    main()