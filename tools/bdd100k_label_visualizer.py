#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
BDD100k 标签可视化工具
用于可视化 bdd100k_0_labels 标签的标注图像
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置中文字体（如果需要显示中文标签）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except Exception as e:
    print(f"警告: 设置中文字体失败: {e}")

class BDD100KLabelVisualizer:
    """
    BDD100k 标签可视化器
    用于根据 bdd100k_0_labels 将灰度标注图像转换为彩色可视化图像
    """
    
    def __init__(self):
        """
        初始化可视化器
        """
        self.labels = None
        self.id_to_color = None
        self.id_to_name = None
        self.load_bdd100k_labels()
    
    def load_bdd100k_labels(self):
        """
        从 labels_bdd100k.py 加载 bdd100k_0_labels 标签信息
        """
        try:
            # 尝试导入 bdd100k_0_labels
            from pem.data.datasets.labels_bdd100k import bdd100k_0_labels
            self.labels = bdd100k_0_labels
            
            # 创建 ID 到颜色和名称的映射
            self.id_to_color = {label.id: label.color for label in self.labels}
            self.id_to_name = {label.id: label.name for label in self.labels}
            
            print(f"成功加载 bdd100k_0_labels，共 {len(self.labels)} 个标签")
            
            # 打印标签信息用于调试
            print("标签 ID 映射表:")
            for label in self.labels:
                print(f"  ID: {label.id}, 名称: {label.name}, 颜色: {label.color}")
            
        except ImportError as e:
            print(f"错误: 无法导入 labels_bdd100k.py: {e}")
            print(f"当前 Python 路径: {sys.path}")
            # 尝试直接导入文件
            labels_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'pem', 'data', 'datasets', 'labels_bdd100k.py')
            print(f"尝试直接从 {labels_path} 导入...")
            
            if os.path.exists(labels_path):
                try:
                    # 动态导入文件
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("labels_bdd100k", labels_path)
                    labels_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(labels_module)
                    self.labels = labels_module.bdd100k_0_labels
                    
                    # 创建映射
                    self.id_to_color = {label.id: label.color for label in self.labels}
                    self.id_to_name = {label.id: label.name for label in self.labels}
                    
                    print(f"成功从文件直接导入 bdd100k_0_labels")
                except Exception as inner_e:
                    print(f"错误: 直接导入文件失败: {inner_e}")
                    sys.exit(1)
            else:
                print(f"错误: 找不到 labels_bdd100k.py 文件: {labels_path}")
                sys.exit(1)
        except AttributeError as e:
            print(f"错误: 找不到 bdd100k_0_labels: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"错误: 加载标签时发生未知错误: {e}")
            sys.exit(1)
    
    def visualize_label_image(self, label_path, output_path=None):
        """
        可视化单个标签图像
        
        Args:
            label_path (str): 标签图像路径
            output_path (str, optional): 输出图像保存路径
        
        Returns:
            np.ndarray: 彩色可视化图像
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(label_path):
                print(f"错误: 文件不存在: {label_path}")
                return None
            
            # 读取灰度标签图像
            label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label_image is None:
                print(f"错误: 无法读取图像: {label_path}")
                return None
            
            print(f"成功读取图像: {label_path}, 形状: {label_image.shape}")
            print(f"图像中包含的唯一标签ID: {np.unique(label_image)}")
            
            # 创建彩色输出图像
            height, width = label_image.shape
            color_image = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 映射每个像素到对应的颜色
            for label_id, color in self.id_to_color.items():
                # 找到所有匹配当前标签ID的像素
                mask = label_image == label_id
                # 设置颜色（注意OpenCV使用BGR格式）
                color_image[mask] = (color[2], color[1], color[0])  # RGB -> BGR
            
            # 处理未映射的像素值
            unlabeled_mask = np.ones((height, width), dtype=bool)
            for label_id in self.id_to_color.keys():
                unlabeled_mask &= (label_image != label_id)
            
            if np.any(unlabeled_mask):
                print(f"警告: 图像中存在未映射的像素值: {np.unique(label_image[unlabeled_mask])}")
                # 为未映射的像素设置默认颜色（灰色）
                color_image[unlabeled_mask] = (128, 128, 128)  # 灰色
            
            # 保存结果
            if output_path:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 使用OpenCV保存图像
                success = cv2.imwrite(output_path, color_image)
                if success:
                    print(f"成功保存可视化图像到: {output_path}")
                else:
                    print(f"错误: 无法保存图像到: {output_path}")
            
            return color_image
            
        except Exception as e:
            print(f"错误: 可视化图像时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_legend(self):
        """
        创建标签图例
        
        Returns:
            list: 图例补丁列表
        """
        legend_patches = []
        
        try:
            # 为每个标签创建一个补丁
            for label in self.labels:
                # 将RGB颜色转换为Matplotlib使用的0-1范围
                color_normalized = tuple(c / 255.0 for c in label.color)
                
                # 创建补丁
                patch = Patch(color=color_normalized, label=f"{label.id}: {label.name}")
                legend_patches.append(patch)
            
            print(f"成功创建图例，包含 {len(legend_patches)} 个标签")
            
        except Exception as e:
            print(f"错误: 创建图例时发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        return legend_patches
    
    def visualize_multiple_images(self, image_dir, num_images=5, output_dir=None):
        """
        可视化多个标签图像
        
        Args:
            image_dir (str): 图像目录路径
            num_images (int): 要可视化的图像数量
            output_dir (str, optional): 输出目录路径
        """
        try:
            # 检查输入目录是否存在
            if not os.path.exists(image_dir):
                print(f"错误: 输入目录不存在: {image_dir}")
                return
            
            print(f"开始处理目录: {image_dir}")
            print(f"计划处理 {num_images} 张图像")
            
            # 获取目录中的所有图像文件
            supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
            image_files = []
            
            # 遍历目录获取图像文件
            for filename in os.listdir(image_dir):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in supported_extensions:
                    image_files.append(os.path.join(image_dir, filename))
            
            # 按文件名排序，确保处理顺序一致
            image_files.sort()
            
            print(f"找到 {len(image_files)} 张图像文件")
            
            # 限制处理的图像数量
            image_files = image_files[:num_images]
            print(f"将处理前 {len(image_files)} 张图像")
            
            # 处理每张图像
            processed_count = 0
            for idx, image_path in enumerate(image_files):
                print(f"\n处理图像 {idx + 1}/{len(image_files)}: {os.path.basename(image_path)}")
                
                # 生成输出路径
                if output_dir:
                    filename = os.path.basename(image_path)
                    name_without_ext, ext = os.path.splitext(filename)
                    output_path = os.path.join(output_dir, f"{name_without_ext}_visualized.png")
                else:
                    output_path = None
                
                # 可视化单个图像
                color_image = self.visualize_label_image(image_path, output_path)
                
                if color_image is not None:
                    processed_count += 1
                    print(f"图像 {idx + 1} 处理成功")
                else:
                    print(f"图像 {idx + 1} 处理失败")
            
            # 创建综合可视化结果（可选）
            if output_dir and processed_count > 0:
                try:
                    self._create_comprehensive_visualization(image_files, output_dir)
                except Exception as e:
                    print(f"警告: 创建综合可视化结果时出错: {e}")
            
            print(f"\n处理完成: 成功处理 {processed_count}/{len(image_files)} 张图像")
            
        except Exception as e:
            print(f"错误: 批量处理图像时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_comprehensive_visualization(self, image_files, output_dir):
        """
        创建综合可视化结果，包含所有处理的图像和图例
        
        Args:
            image_files (list): 处理的图像文件列表
            output_dir (str): 输出目录路径
        """
        try:
            # 限制最多显示4张图像，避免图表过大
            display_files = image_files[:4]
            n = len(display_files)
            
            if n == 0:
                return
            
            # 计算图表布局
            if n <= 2:
                rows, cols = 1, n
            else:
                rows, cols = (n + 1) // 2, 2
            
            # 创建图表
            fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
            if n == 1:
                axes = [axes]  # 确保axes是列表
            else:
                axes = axes.flatten()
            
            # 处理每张图像
            for i, image_path in enumerate(display_files):
                ax = axes[i]
                
                # 读取彩色图像
                output_filename = os.path.splitext(os.path.basename(image_path))[0] + "_visualized.png"
                output_path = os.path.join(output_dir, output_filename)
                
                if os.path.exists(output_path):
                    # 使用OpenCV读取并转换为RGB
                    color_image = cv2.imread(output_path)
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                    
                    # 显示图像
                    ax.imshow(color_image)
                    ax.set_title(f"图像 {i + 1}: {os.path.basename(image_path)}")
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, "图像不可用", ha='center', va='center')
                    ax.set_title(f"图像 {i + 1}: 处理失败")
                    ax.axis('off')
            
            # 隐藏多余的子图
            for i in range(n, len(axes)):
                axes[i].axis('off')
            
            # 添加图例
            legend_patches = self.create_legend()
            if legend_patches:
                # 为图例创建单独的子图
                legend_fig = plt.figure(figsize=(10, 8))
                legend_fig.legend(handles=legend_patches, loc='center', frameon=False,
                                 ncol=2, fontsize='small')
                legend_fig.suptitle("BDD100K 0 Labels 图例", fontsize=16)
                legend_fig.tight_layout()
                
                # 保存图例
                legend_path = os.path.join(output_dir, "bdd100k_0_labels_legend.png")
                legend_fig.savefig(legend_path, bbox_inches='tight', dpi=300)
                plt.close(legend_fig)
                print(f"成功保存图例到: {legend_path}")
            
            # 保存综合可视化结果
            fig.suptitle("BDD100K 0 Labels 标签可视化", fontsize=20)
            fig.tight_layout()
            
            综合_output_path = os.path.join(output_dir, "综合可视化结果.png")
            fig.savefig(综合_output_path, bbox_inches='tight', dpi=300)
            plt.close(fig)
            
            print(f"成功保存综合可视化结果到: {综合_output_path}")
            
        except Exception as e:
            print(f"错误: 创建综合可视化时发生错误: {e}")
            import traceback
            traceback.print_exc()

def parse_arguments():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 解析后的参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='BDD100K 标签可视化工具')
    
    # 输入目录参数
    parser.add_argument('--input-dir', 
                      default=r"D:\Ar\PEM\datasets\cityscapes\gtFine\train\bdd100k_O",
                      help='输入图像目录路径')
    
    # 输出目录参数
    parser.add_argument('--output-dir', 
                      default=r"D:\Ar\PEM\output\label_visualization",
                      help='输出结果目录路径')
    
    # 处理图像数量参数
    parser.add_argument('--num-images', 
                      type=int, 
                      default=10,
                      help='要处理的图像数量')
    
    # 是否显示详细日志
    parser.add_argument('--verbose', 
                      action='store_true',
                      help='显示详细日志信息')
    
    # 是否只生成图例
    parser.add_argument('--legend-only', 
                      action='store_true',
                      help='仅生成标签图例')
    
    return parser.parse_args()

def setup_logging(verbose=False):
    """
    设置日志配置
    
    Args:
        verbose (bool): 是否启用详细日志
    """
    import logging
    
    # 配置日志级别
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # 配置基本日志格式
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger(__name__)

def create_output_directory(output_dir):
    """
    创建输出目录并确保权限
    
    Args:
        output_dir (str): 输出目录路径
    
    Returns:
        bool: 创建是否成功
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试写入权限
        test_file = os.path.join(output_dir, "_test_write.txt")
        with open(test_file, 'w') as f:
            f.write("Test write permission")
        os.remove(test_file)
        
        print(f"成功创建输出目录并验证写入权限: {output_dir}")
        return True
        
    except PermissionError:
        print(f"错误: 没有写入权限到输出目录: {output_dir}")
        return False
    except Exception as e:
        print(f"错误: 创建输出目录时发生错误: {e}")
        return False

def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志
        logger = setup_logging(args.verbose)
        
        print("=== BDD100K 标签可视化工具 ===")
        print(f"输入目录: {args.input_dir}")
        print(f"输出目录: {args.output_dir}")
        print(f"处理图像数量: {args.num_images}")
        print(f"详细日志: {'启用' if args.verbose else '禁用'}")
        
        # 创建输出目录
        if not create_output_directory(args.output_dir):
            print("错误: 无法创建输出目录，程序退出")
            sys.exit(1)
        
        # 创建可视化器实例
        print("\n初始化可视化器...")
        visualizer = BDD100KLabelVisualizer()
        
        # 如果只需要生成图例
        if args.legend_only:
            print("\n仅生成标签图例...")
            legend_patches = visualizer.create_legend()
            if legend_patches:
                # 为图例创建单独的子图
                import matplotlib.pyplot as plt
                legend_fig = plt.figure(figsize=(10, 8))
                legend_fig.legend(handles=legend_patches, loc='center', frameon=False,
                                 ncol=2, fontsize='small')
                legend_fig.suptitle("BDD100K 0 Labels 图例", fontsize=16)
                legend_fig.tight_layout()
                
                # 保存图例
                legend_path = os.path.join(args.output_dir, "bdd100k_0_labels_legend.png")
                legend_fig.savefig(legend_path, bbox_inches='tight', dpi=300)
                plt.close(legend_fig)
                print(f"成功保存图例到: {legend_path}")
            print("图例生成完成")
            return
        
        # 验证输入目录
        if not os.path.exists(args.input_dir):
            print(f"错误: 输入目录不存在: {args.input_dir}")
            sys.exit(1)
        
        if not os.path.isdir(args.input_dir):
            print(f"错误: 输入路径不是一个目录: {args.input_dir}")
            sys.exit(1)
        
        # 验证图像数量参数
        if args.num_images <= 0:
            print("错误: 图像数量必须大于0")
            sys.exit(1)
        
        # 可视化图像
        print("\n开始处理图像...")
        visualizer.visualize_multiple_images(args.input_dir, args.num_images, args.output_dir)
        
        print("\n=== 处理完成 ===")
        print(f"所有结果已保存到: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: 程序执行时发生未捕获的异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()