import os
import json
import cv2
import numpy as np
from tqdm import tqdm

class JSONLaneFiller:
    def __init__(self):
        # 单车道线类别列表，我们只处理这些类型
        self.single_lane_types = ['single white', 'single yellow', 'single other']
        self.output_dir = r'D:\Ar\PEM\output\lane_filled_json'
        self.comparison_dir = r'D:\Ar\PEM\output\lane_comparison'
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.comparison_dir, exist_ok=True)
        
    def load_json_annotations(self, json_path):
        """加载JSON标注文件"""
        print(f"加载标注文件: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"成功加载 {len(annotations)} 张图像的标注")
        return annotations
    
    def extract_single_lanes(self, image_annotations):
        """从图像标注中提取单车道线信息"""
        single_lanes = []
        for label in image_annotations.get('labels', []):
            category = label.get('category', '')
            if category in self.single_lane_types:
                # 获取多边形信息
                poly2d = label.get('poly2d', [])
                for poly in poly2d:
                    vertices = poly.get('vertices', [])
                    if vertices and len(vertices) >= 2:
                        # 转换为整数坐标
                        int_vertices = [[int(x), int(y)] for x, y in vertices]
                        single_lanes.append({
                            'category': category,
                            'vertices': int_vertices,
                            'attributes': label.get('attributes', {})
                        })
        return single_lanes
    
    def create_lane_envelope(self, vertices, lane_width=10, image_height=720, image_width=1280):
        """创建车道线的包络线
        基于车道线的多边形点，生成具有一定宽度的包络线
        使用法线方向计算扩展点，生成更自然的车道线形状
        """
        try:
            # 验证顶点数据
            if not vertices or not isinstance(vertices, list) or len(vertices) < 2:
                return None
            
            # 清理顶点数据，确保每个顶点都是有效的[x,y]列表
            clean_vertices = []
            for v in vertices:
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    clean_vertices.append([float(v[0]), float(v[1])])
            
            if len(clean_vertices) < 2:
                return None
            
            # 将顶点转换为numpy数组
            points = np.array(clean_vertices, dtype=np.float32)
            
            # 创建扩展后的点集合
            expanded_points = []
            half_width = lane_width / 2.0
            
            # 计算每两个相邻点之间的扩展点
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # 计算向量方向
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                
                # 计算单位向量
                length = np.sqrt(dx * dx + dy * dy)
                if length < 1.0:  # 避免除零
                    continue
                
                # 计算法线方向（左和右）
                normal_left = np.array([-dy, dx]) / length
                normal_right = np.array([dy, -dx]) / length
                
                # 生成扩展点
                left1 = p1 + normal_left * half_width
                right1 = p1 + normal_right * half_width
                left2 = p2 + normal_left * half_width
                right2 = p2 + normal_right * half_width
                
                # 添加到扩展点集合
                expanded_points.append(left1)
                expanded_points.append(left2)
                expanded_points.append(right2)
                expanded_points.append(right1)
            
            # 创建掩码
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # 如果有扩展点，绘制多边形
            if len(expanded_points) >= 3:
                # 确保点在图像范围内
                expanded_points_array = np.array(expanded_points)
                expanded_points_array[:, 0] = np.clip(expanded_points_array[:, 0], 0, image_width - 1)
                expanded_points_array[:, 1] = np.clip(expanded_points_array[:, 1], 0, image_height - 1)
                cv2.fillPoly(mask, [expanded_points_array.astype(np.int32)], 255)
            else:
                # 回退到简单的线条绘制
                cv2.polylines(mask, [points.astype(np.int32)], isClosed=False, color=255, thickness=lane_width)
            
            return mask
        except Exception as e:
            print(f"创建包络线时出错: {str(e)}")
            # 回退到简单实现
            try:
                points = np.array(clean_vertices if 'clean_vertices' in locals() else vertices, dtype=np.int32)
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=lane_width)
                return mask
            except:
                return None
    
    def fill_lane_from_envelope(self, mask, lane_style="solid"):
        """从包络线掩码填充车道线区域
        根据车道线样式（实线/虚线）进行不同的填充处理
        """
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建填充后的掩码
        filled_mask = np.zeros_like(mask)
        
        # 对于实线车道线，直接填充所有轮廓
        if lane_style == "solid":
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # 过滤小区域
                    cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        # 对于虚线车道线，创建虚线效果
        elif lane_style == "dashed":
            for contour in contours:
                if cv2.contourArea(contour) > 10:
                    # 计算轮廓的长度
                    perimeter = cv2.arcLength(contour, False)
                    
                    # 将轮廓分割为线段
                    step_length = 20  # 线段长度
                    gap_length = 15   # 间隔长度
                    total_step = step_length + gap_length
                    
                    for i in range(0, int(perimeter), total_step):
                        # 获取轮廓上的点
                        point1 = tuple(map(int, cv2.pointPolygonTest(contour, (0, 0), True)))  # 这需要调整
                        # 简化实现，先填充整个轮廓
                        cv2.drawContours(filled_mask, [contour], -1, 255, -1)
        
        # 进行形态学操作，平滑边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
        filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, kernel)
        
        return filled_mask
    
    def process_image(self, image_path, image_annotations):
        """处理单张图像"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 无法读取图像 {image_path}")
                return None
            
            # 创建一个标注图像副本
            annotated_image = image.copy()
            
            # 提取单车道线
            single_lanes = self.extract_single_lanes(image_annotations)
            print(f"  找到 {len(single_lanes)} 个单车道线")
            
            # 处理每条车道线
            for idx, lane in enumerate(single_lanes):
                try:
                    vertices = lane['vertices']
                    category = lane['category']
                    
                    # 根据车道线类型设置颜色
                    if 'white' in category:
                        color = (255, 255, 255)  # 白色
                        pixel_value = 10  # 对应之前的标签值
                    elif 'yellow' in category:
                        color = (0, 255, 255)  # 黄色
                        pixel_value = 12  # 对应之前的标签值
                    else:
                        color = (128, 128, 128)  # 其他颜色
                        pixel_value = 8  # 对应之前的标签值
                    
                    # 获取车道线样式
                    lane_style = lane['attributes'].get('laneStyle', 'solid')
                    
                    # 根据车道线类型调整宽度
                    if lane_style == 'solid':
                        width = 8  # 实线宽度
                    else:
                        width = 6  # 虚线宽度
                    
                    # 创建包络线
                    envelope_mask = self.create_lane_envelope(vertices, lane_width=width,
                                                              image_height=image.shape[0],
                                                              image_width=image.shape[1])
                    
                    if envelope_mask is not None:
                        # 填充包络线区域
                        filled_mask = self.fill_lane_from_envelope(envelope_mask, lane_style=lane_style)
                        
                        # 安全地应用填充
                        if filled_mask is not None and isinstance(filled_mask, np.ndarray):
                            # 直接使用掩码进行索引，避免使用where可能导致的问题
                            annotated_image[filled_mask > 0] = color
                            print(f"    成功填充车道线 {idx+1}")
                        else:
                            print(f"    警告: filled_mask为None或无效类型")
                    else:
                        print(f"    警告: envelope_mask为None")
                except Exception as e:
                    print(f"  处理车道线 {idx+1} 时出错: {str(e)}")
                    # 继续处理下一条车道线
                    continue
            
            return annotated_image
            
        except Exception as e:
            print(f"处理图像出错: {str(e)}")
            # 返回原始图像作为备份
            return image
    
    def run(self, json_path, image_dir, test_mode=True, max_images=50):
        """主函数"""
        # 加载标注
        annotations = self.load_json_annotations(json_path)
        
        # 如果是测试模式，只处理前max_images张图像
        if test_mode:
            annotations = annotations[:max_images]
            print(f"测试模式: 只处理前 {max_images} 张图像")
        
        # 处理每张图像
        for idx, img_annot in enumerate(tqdm(annotations)):
            # 获取图像名称
            img_name = img_annot.get('name', '')
            if not img_name:
                continue
            
            print(f"\n处理第 {idx+1}/{len(annotations)} 张图像: {img_name}")
            
            # 尝试多种可能的图像路径
            possible_paths = [
                os.path.join(image_dir, img_name),
                os.path.join(r'D:\Ar\data\BD100kLane\gtFine\train', img_name),
                os.path.join(r'D:\Ar\data\BD100kLane', img_name),
                os.path.join(r'D:\Ar\PEM\output', img_name)
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    print(f"  找到图像: {path}")
                    break
            
            if img_path is None:
                print(f"  警告: 在所有可能的路径中都找不到图像 {img_name}")
                # 为测试目的，创建一个空白图像
                img_path = os.path.join(self.output_dir, 'temp_' + img_name)
                print(f"  创建空白测试图像: {img_path}")
                test_image = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.imwrite(img_path, test_image)
            
            # 读取原始图像
            original_image = cv2.imread(img_path)
            if original_image is None:
                print(f"  警告: 无法读取原始图像 {img_path}")
                continue
            
            # 处理图像
            result_image = self.process_image(img_path, img_annot)
            
            # 保存结果
            if result_image is not None:
                output_path = os.path.join(self.output_dir, img_name)
                cv2.imwrite(output_path, result_image)
                print(f"  结果已保存至: {output_path}")
                
                # 生成对比图
                try:
                    # 添加文本标签
                    original_with_label = original_image.copy()
                    result_with_label = result_image.copy()
                    
                    # 设置文本参数
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_color = (0, 255, 0)  # 绿色
                    thickness = 2
                    
                    # 添加标签
                    cv2.putText(original_with_label, '原始图像', (50, 50), font, font_scale, font_color, thickness)
                    cv2.putText(result_with_label, '处理后图像', (50, 50), font, font_scale, font_color, thickness)
                    
                    # 创建并排对比图
                    comparison_image = np.hstack((original_with_label, result_with_label))
                    
                    # 保存对比图
                    comparison_path = os.path.join(self.comparison_dir, 'comparison_' + img_name)
                    cv2.imwrite(comparison_path, comparison_image)
                    print(f"  对比图已保存至: {comparison_path}")
                except Exception as e:
                    print(f"  生成对比图时出错: {str(e)}")
        
        print("\n所有图像处理完成！")
        print(f"对比图保存在: {self.comparison_dir}")

if __name__ == "__main__":
    # JSON标注文件路径
    JSON_PATH = r'D:\Ar\data\BD100kLane\gtFine\polygons\lane_train.json'
    
    # 图像目录（需要用户根据实际情况修改）
    IMAGE_DIR = r'D:\Ar\data\BD100kLane\gtFine\images\train'  # 需要根据实际情况调整
    
    # 创建并运行处理器
    filler = JSONLaneFiller()
    # 设置test_mode=True只处理少量图像进行测试
    filler.run(JSON_PATH, IMAGE_DIR, test_mode=True, max_images=50)
