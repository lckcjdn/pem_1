#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
车道线单实线填充算法（灰度图版）
将单实线车道线两条边缘之间的缝隙填满相同的标注
适用于灰度像素图，其中像素值代表标签类别
仅处理单实线车道线（单实其他线、单白实线、单黄实线）
"""

import os
import cv2
import numpy as np
import time
from tqdm import tqdm

# 配置路径
INPUT_DIR = r"D:\Ar\PEM\output\lane_filled"
OUTPUT_DIR = r"D:\Ar\PEM\output\lane_filled"
MAX_IMAGES = None  # 只处理文件夹中的图片

class LaneEdgeFiller:
    """车道线边缘填充器（灰度图版）"""
    
    def __init__(self, use_visualization=False):
        # 根据BDD100k_0_labels标签定义，设置单实线车道线的像素值（id）
        self.single_solid_lane_ids = {
            11: "single_other_solid",       # 单实其他线
            13: "single_white_solid",      # 单白实线
            14: "single_yellow_solid"       # 单黄实线
        }
        
        # 添加单虚线车道线的像素值（id）
        self.single_dashed_lane_ids = {
            12: "single_other_dashed",       # 单虚其他线
            15: "single_white_dashed",       # 单白虚线
            16: "single_yellow_dashed"       # 单黄虚线
        }
        
        # 合并所有车道线类型（包括单实线和单虚线）
        self.all_lane_ids = {**self.single_solid_lane_ids, **self.single_dashed_lane_ids}
        
        # 霍夫变换参数
        self.hough_params = {
            'rho': 1,            # 极坐标半径精度
            'theta': np.pi / 180,  # 极坐标角度精度
            'threshold': 50,     # 累加器阈值
            'min_line_length': 40, # 线段最小长度
            'max_line_gap': 20,   # 线段最大间隔
            'angle_tolerance': np.pi / 18,  # 角度容差（10度）
            'distance_tolerance': 20       # 距离容差
        }
        
        # 可视化开关
        self.use_visualization = use_visualization
    
    def load_image(self, image_path):
        """加载灰度图像"""
        try:
            # 使用灰度模式加载图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")
            return image
        except Exception as e:
            print(f"加载图像时出错: {e}")
            return None
    
    def get_unique_pixel_values(self, image):
        """获取图像中所有唯一的像素值（标签类别）"""
        # 对于灰度图，直接获取唯一像素值
        unique_values = np.unique(image)
        return unique_values
    
    def extract_edge_mask(self, image, pixel_value):
        """提取指定像素值的边缘掩码"""
        # 创建像素值掩码
        value_mask = (image == pixel_value).astype(np.uint8) * 255
        # 检查掩码是否为空
        if np.sum(value_mask) == 0:
            return None
        return value_mask
    
    def detect_parallel_lines_hough(self, edge_mask):
        """使用霍夫变换检测边缘中的平行线"""
        # 应用Canny边缘检测来增强边缘
        edges = cv2.Canny(edge_mask, 50, 150)
        
        # 使用霍夫变换检测直线
        lines = cv2.HoughLinesP(
            edges, 
            rho=self.hough_params['rho'],
            theta=self.hough_params['theta'],
            threshold=self.hough_params['threshold'],
            minLineLength=self.hough_params['min_line_length'],
            maxLineGap=self.hough_params['max_line_gap']
        )
        
        if lines is None:
            return []
        
        # 计算每条直线的角度和参数
        line_info = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 计算直线的角度
            angle = np.arctan2(y2 - y1, x2 - x1)
            
            # 转换为标准形式：ax + by + c = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            
            # 标准化
            norm = np.sqrt(a**2 + b**2)
            a, b, c = a / norm, b / norm, c / norm
            
            line_info.append({
                'points': [(x1, y1), (x2, y2)],
                'angle': angle,
                'params': (a, b, c)
            })
        
        # 按角度分组，找出平行线对
        parallel_line_pairs = []
        processed_indices = set()
        
        for i in range(len(line_info)):
            if i in processed_indices:
                continue
                
            line1 = line_info[i]
            current_group = [line1]
            processed_indices.add(i)
            
            for j in range(i + 1, len(line_info)):
                if j in processed_indices:
                    continue
                    
                line2 = line_info[j]
                # 检查角度是否相近（平行）
                angle_diff = abs(line1['angle'] - line2['angle'])
                # 处理角度接近0或180度的情况
                angle_diff = min(angle_diff, np.pi - angle_diff)
                
                if angle_diff < self.hough_params['angle_tolerance']:
                    # 计算两条平行线之间的距离
                    # 使用标准直线方程的c值差作为距离度量
                    distance = abs(line1['params'][2] - line2['params'][2])
                    
                    # 确保距离在合理范围内
                    if 0 < distance < self.hough_params['distance_tolerance']:
                        current_group.append(line2)
                        processed_indices.add(j)
            
            # 对于每组平行线，生成所有可能的线对
            for i in range(len(current_group)):
                for j in range(i + 1, len(current_group)):
                    parallel_line_pairs.append((current_group[i], current_group[j]))
        
        return parallel_line_pairs
    
    def fill_parallel_lines_area(self, image, parallel_pairs, pixel_value, original_mask=None):
        """填充平行线对之间的区域"""
        filled_image = image.copy()
        non_zero_mask = (image > 0).astype(np.uint8) * 255
        
        # 创建填充掩码
        fill_mask = np.zeros_like(image, dtype=np.uint8)
        
        # 处理每对平行线
        for line1, line2 in parallel_pairs:
            # 获取直线参数
            a1, b1, c1 = line1['params']
            a2, b2, c2 = line2['params']
            
            # 确定哪条线在上方/左侧
            # 使用c值来判断相对位置
            if abs(c1) < abs(c2):
                left_line, right_line = line1, line2
            else:
                left_line, right_line = line2, line1
            
            # 为每条线创建掩码
            a_left, b_left, c_left = left_line['params']
            a_right, b_right, c_right = right_line['params']
            
            # 获取直线上的点，用于限制填充区域
            line1_points = line1['points']
            line2_points = line2['points']
            
            # 计算直线的边界框，限制填充区域
            all_x = [p[0] for p in line1_points + line2_points]
            all_y = [p[1] for p in line1_points + line2_points]
            min_x = max(0, int(min(all_x) - 20))
            max_x = min(image.shape[1] - 1, int(max(all_x) + 20))
            min_y = max(0, int(min(all_y) - 20))
            max_y = min(image.shape[0] - 1, int(max(all_y) + 20))
            
            # 优化：使用numpy向量化操作替代嵌套循环，提高效率
            # 创建边界框内的坐标网格
            y_coords, x_coords = np.mgrid[min_y:max_y+1, min_x:max_x+1]
            
            # 计算所有点到两条线的距离
            dist_left = a_left * x_coords + b_left * y_coords + c_left
            dist_right = a_right * x_coords + b_right * y_coords + c_right
            
            # 确定哪些点在两条线之间
            between_lines = ((dist_left <= 0) & (dist_right >= 0)) | ((dist_left >= 0) & (dist_right <= 0))
            
            # 检查哪些点已经有内容
            non_zero_in_bbox = non_zero_mask[min_y:max_y+1, min_x:max_x+1] > 0
            
            # 合并条件：在两条线之间且没有内容
            fill_cond = between_lines & ~non_zero_in_bbox
            
            # 如果提供了原始掩码，还需要检查周围是否有原始车道线像素
            if original_mask is not None:
                # 优化：使用numpy卷积操作替代嵌套循环检查周围区域
                kernel_size = 10
                # 创建卷积核
                kernel = np.ones((kernel_size*2+1, kernel_size*2+1), dtype=np.uint8)
                # 计算原始掩码的膨胀，得到周围有原始车道线像素的区域
                dilated_mask = cv2.dilate(original_mask, kernel)
                # 提取边界框内的膨胀掩码
                dilated_in_bbox = dilated_mask[min_y:max_y+1, min_x:max_x+1] > 0
                # 更新填充条件
                fill_cond = fill_cond & dilated_in_bbox
            
            # 将符合条件的点设置为填充掩码
            fill_mask[min_y:max_y+1, min_x:max_x+1][fill_cond] = 255
        
        # 应用形态学操作来增强填充效果
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fill_mask = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, kernel)
        
        # 将填充应用到图像上
        filled_pixels = np.where(fill_mask > 0)
        filled_image[filled_pixels] = pixel_value
        
        return filled_image
    
    def is_solid_lane_color(self, color):
        """判断是否为实线车道线的像素值
        根据BDD100k标签定义的实线车道线类别进行判断
        """
        # 判断像素值是否在单实线车道线id集合中
        return color in self.single_solid_lane_ids
        
    def is_dashed_lane_color(self, color):
        """判断是否为虚线车道线的像素值
        根据BDD100k标签定义的虚线车道线类别进行判断"""
        # 判断像素值是否在单虚线车道线id集合中
        return color in self.single_dashed_lane_ids

    def is_lane_color(self, color):
        """判断是否为车道线的像素值
        根据BDD100k标签定义的所有车道线类别进行判断
        """
        # 判断像素值是否在所有车道线id集合中
        return color in self.all_lane_ids
    
    def visualize_results(self, original_image, filled_image, save_path):
        """可视化填充结果，生成原始图像和处理后图像的对比图"""
        # 获取图像的高度和宽度
        height, width = original_image.shape
        
        # 对灰度图像进行对比度增强，使车道线更加明显
        def enhance_contrast(gray_image):
            # 计算灰度图像的最小值和最大值
            min_val = np.min(gray_image)
            max_val = np.max(gray_image)
            
            # 如果最小值等于最大值，说明图像是单色的，直接返回
            if min_val == max_val:
                return gray_image
            
            # 对比度增强
            enhanced = (gray_image - min_val) * (255.0 / (max_val - min_val))
            return enhanced.astype(np.uint8)
        
        # 增强原始图像和处理后图像的对比度
        original_enhanced = enhance_contrast(original_image)
        filled_enhanced = enhance_contrast(filled_image)
        
        # 将灰度图像转换为彩色图像，使用灰度值到彩色的映射
        def gray_to_color(gray_image):
            # 创建彩色图像
            color_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 使用灰度值到彩色的映射，使不同灰度值显示为不同颜色
            # 低灰度值（背景）显示为黑色，高灰度值（车道线）显示为彩色
            for y in range(height):
                for x in range(width):
                    val = gray_image[y, x]
                    if val == 0:
                        # 背景显示为黑色
                        color = (0, 0, 0)
                    elif val == 7:
                        # 单虚其他线 - 显示为浅绿色
                        color = (0, 200, 100)
                    elif val == 8:
                        # 单实其他线 - 显示为深绿色
                        color = (0, 100, 0)
                    elif val == 9:
                        # 单白虚线 - 显示为浅灰色
                        color = (200, 200, 200)
                    elif val == 10:
                        # 单白实线 - 显示为白色
                        color = (255, 255, 255)
                    elif val == 11:
                        # 单黄虚线 - 显示为浅黄色
                        color = (0, 200, 200)
                    elif val == 12:
                        # 单黄实线 - 显示为亮黄色
                        color = (0, 255, 255)
                    else:
                        # 其他灰度值 - 显示为红色
                        color = (0, 0, 255)
                    
                    color_image[y, x] = color
            
            return color_image
        
        # 将原始图像转换为彩色图像
        original_color = gray_to_color(original_enhanced)
        
        # 将处理后图像转换为彩色图像
        filled_color = gray_to_color(filled_enhanced)
        
        # 创建一个新的图像，用于并排显示原始图像和处理后图像
        viz_height = height
        viz_width = width * 2 + 10  # 10像素的间隔
        viz_image = np.ones((viz_height, viz_width, 3), dtype=np.uint8) * 255  # 白色背景
        
        # 将原始图像和处理后图像复制到可视化图像中
        viz_image[:, :width, :] = original_color
        viz_image[:, width+10:, :] = filled_color
        
        # 添加文字说明
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 0, 0)  # 黑色文字
        thickness = 1
        
        # 添加原始图像说明
        cv2.putText(viz_image, "Original Image", (10, 20), font, font_scale, font_color, thickness)
        
        # 添加处理后图像说明
        cv2.putText(viz_image, "Filled Image", (width + 20, 20), font, font_scale, font_color, thickness)
        
        # 保存可视化结果
        cv2.imwrite(save_path, viz_image)
    
    def process_image(self, image_path):
        """处理单张灰度图像，仅处理单实线车道线"""
        # 加载图像
        image = self.load_image(image_path)
        if image is None:
            return None
        
        # 获取唯一像素值（标签）
        unique_values = self.get_unique_pixel_values(image)
        
        # 处理每个像素值
        filled_image = image.copy()
        
        # 筛选出所有车道线的像素值（包括单实线和单虚线）
        lane_values = [val for val in unique_values if self.is_lane_color(val)]
        
        for value in lane_values:
            # 提取该像素值的边缘掩码
            edge_mask = self.extract_edge_mask(image, value)
            if edge_mask is None:
                continue
            
            # 使用霍夫变换检测平行线，填充车道线
            parallel_pairs = self.detect_parallel_lines_hough(edge_mask)
            if parallel_pairs:
                # 使用霍夫变换结果填充车道线，并传递原始掩码
                filled_image = self.fill_parallel_lines_area(filled_image, parallel_pairs, value, edge_mask)
        
        return filled_image


def main(use_visualization=True):
    """主函数，处理单实线车道线图像"""
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 获取输入目录中的所有图像文件（只处理PNG文件）
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    image_files = image_files[:MAX_IMAGES]  # 只处理指定数量的图片
    
    print(f"找到 {len(image_files)} 张图像，将处理前 {MAX_IMAGES} 张")
    
    # 创建填充器实例，传入可视化开关参数
    filler = LaneEdgeFiller(use_visualization=use_visualization)
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 处理每张图像，添加进度条
    for i, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="处理图像"):
        image_path = os.path.join(INPUT_DIR, image_file)
        
        # 记录单张图像开始时间
        img_start_time = time.time()
        
        # 加载原始图像用于可视化比较
        original_image = filler.load_image(image_path)
        if original_image is None:
            continue
        
        # 处理图像
        filled_image = filler.process_image(image_path)
        if filled_image is None:
            continue
        
        # 保存处理后的图像
        output_path = os.path.join(OUTPUT_DIR, f"filled_{image_file}")
        cv2.imwrite(output_path, filled_image)
        
        # 如果启用了可视化，生成并保存可视化结果
        if use_visualization:
            viz_path = os.path.join(OUTPUT_DIR, f"viz_{image_file.replace('.png', '.jpg')}")
            filler.visualize_results(original_image, filled_image, viz_path)
        
        # 记录单张图像结束时间，计算耗时
        img_end_time = time.time()
        img_duration = img_end_time - img_start_time
        print(f"\n处理图像 {image_file} 耗时: {img_duration:.2f} 秒")
    
    # 记录总结束时间，计算总耗时
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print(f"\n处理完成！")
    print(f"总耗时: {total_duration:.2f} 秒")
    print(f"平均每张图像耗时: {total_duration / len(image_files):.2f} 秒")


if __name__ == "__main__":
    main()