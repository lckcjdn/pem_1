#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
标签文件匹配清理工具

此脚本用于对比两个文件夹中的文件（通常是图像文件夹和标签文件夹），
并删除没有对应匹配图像文件的标签文件。

使用方法:
python match_and_clean_labels.py --image-dir /path/to/images --label-dir /path/to/labels [--dry-run] [--pattern "gtFine"]
"""

import os
import argparse
import re
from typing import Set, List, Tuple


def get_file_basenames(directory: str, pattern: str = None) -> Set[str]:
    """
    获取目录中所有文件的基础名称集合（不包含扩展名）
    
    Args:
        directory: 要扫描的目录路径
        pattern: 可选的文件名过滤模式（正则表达式）
    
    Returns:
        文件名基础名称集合
    """
    basenames = set()
    
    if not os.path.exists(directory):
        print(f"警告: 目录 {directory} 不存在")
        return basenames
    
    for filename in os.listdir(directory):
        # 应用过滤模式（如果提供）
        if pattern and not re.search(pattern, filename):
            continue
            
        # 获取基础名称（不包含扩展名）
        name, _ = os.path.splitext(filename)
        basenames.add(name)
    
    return basenames


def find_unmatched_files(
    image_dir: str, 
    label_dir: str, 
    image_pattern: str = None,
    label_pattern: str = None,
    debug: bool = False
) -> List[str]:
    """
    查找标签目录中没有对应图像文件的标签文件
    
    Args:
        image_dir: 图像文件目录
        label_dir: 标签文件目录
        image_pattern: 图像文件过滤模式
        label_pattern: 标签文件过滤模式
        debug: 是否显示调试信息
    
    Returns:
        不匹配的标签文件路径列表
    """
    # 获取目录中所有文件的总数（用于显示信息）
    total_images = len([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])
    total_labels = len([f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))])
    
    # 获取图像文件的基础名称
    image_basenames = get_file_basenames(image_dir, image_pattern)
    
    # 获取标签文件的信息
    unmatched_labels = []
    label_basenames = set()
    
    if not os.path.exists(label_dir):
        print(f"警告: 标签目录 {label_dir} 不存在")
        return unmatched_labels
    
    for filename in os.listdir(label_dir):
        # 应用过滤模式（如果提供）
        if label_pattern and not re.search(label_pattern, filename):
            continue
            
        # 获取基础名称（不包含扩展名）
        name, _ = os.path.splitext(filename)
        label_basenames.add(name)
        
        # 检查是否有匹配的图像文件
        if name not in image_basenames:
            unmatched_labels.append(os.path.join(label_dir, filename))
    
    print(f"在 {image_dir} 中找到 {total_images} 个文件，经过过滤后匹配到 {len(image_basenames)} 个图像文件")
    print(f"在 {label_dir} 中找到 {total_labels} 个文件，经过过滤后匹配到 {len(label_basenames)} 个标签文件")
    
    # 调试信息
    if debug:
        print("\n=== 调试信息 ===")
        # 显示部分图像文件名
        print(f"前10个图像文件基础名称:")
        for i, name in enumerate(list(image_basenames)[:10]):
            print(f"  {i+1}. {name}")
        
        # 显示部分标签文件名
        print(f"\n前10个标签文件基础名称:")
        for i, name in enumerate(list(label_basenames)[:10]):
            print(f"  {i+1}. {name}")
        
        # 检查是否有标签数量大于图像数量但没有不匹配的情况
        if len(label_basenames) > len(image_basenames) and len(unmatched_labels) == 0:
            print(f"\n注意：标签文件数量({len(label_basenames)})大于图像文件数量({len(image_basenames)})，但没有找到不匹配的标签。")
            print("这可能是因为:")
            print("1. 一个图像对应多个标签文件")
            print("2. 文件名匹配逻辑需要调整")
            print("3. 过滤模式设置导致某些文件被跳过")
            
            # 检查是否有部分匹配的情况
            print("\n检查前几个标签文件名与图像文件名的相似性:")
            for label_name in list(label_basenames)[:5]:
                # 查找相似的图像文件名
                similar_images = [img for img in image_basenames if label_name in img or img in label_name]
                if similar_images:
                    print(f"  标签 '{label_name}' 可能与图像匹配: {similar_images[:3]}")
                else:
                    print(f"  标签 '{label_name}' 没有找到明显相似的图像文件")
    
    return unmatched_labels


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='清理不匹配的标签文件')
    parser.add_argument('--image-dir', required=True, help='图像文件所在目录')
    parser.add_argument('--label-dir', required=True, help='标签文件所在目录')
    parser.add_argument('--image-pattern', default=None, help='图像文件名过滤模式（正则表达式）')
    parser.add_argument('--label-pattern', default=None, help='标签文件名过滤模式（正则表达式）')
    parser.add_argument('--dry-run', action='store_true', help='仅显示将被删除的文件，不执行删除操作')
    parser.add_argument('--recursive', action='store_true', help='递归处理子目录')
    parser.add_argument('--debug', action='store_true', help='显示详细调试信息，包括文件名匹配情况')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    for dir_path in [args.image_dir, args.label_dir]:
        if not os.path.isdir(dir_path):
            print(f"错误: 目录 {dir_path} 不是有效目录")
            return
    
    total_unmatched = 0
    total_deleted = 0
    
    if args.recursive:
        # 递归处理子目录
        for root, dirs, files in os.walk(args.label_dir):
            # 计算对应的图像子目录
            rel_path = os.path.relpath(root, args.label_dir)
            if rel_path == '.':
                image_subdir = args.image_dir
            else:
                image_subdir = os.path.join(args.image_dir, rel_path)
            
            if os.path.exists(image_subdir):
                print(f"处理子目录: {rel_path}")
                unmatched = find_unmatched_files(
                    image_subdir, root, args.image_pattern, args.label_pattern, args.debug
                )
                
                total_unmatched += len(unmatched)
                
                # 处理不匹配的文件
                for file_path in unmatched:
                    if args.dry_run:
                        print(f"将删除: {file_path}")
                    else:
                        try:
                            os.remove(file_path)
                            print(f"已删除: {file_path}")
                            total_deleted += 1
                        except Exception as e:
                            print(f"删除失败 {file_path}: {e}")
            else:
                print(f"警告: 未找到对应的图像子目录 {image_subdir}")
    else:
        # 直接处理指定目录
        unmatched = find_unmatched_files(
            args.image_dir, args.label_dir, args.image_pattern, args.label_pattern, args.debug
        )
        
        total_unmatched = len(unmatched)
        
        # 处理不匹配的文件
        for file_path in unmatched:
            if args.dry_run:
                print(f"将删除: {file_path}")
            else:
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                    total_deleted += 1
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
    
    # 打印统计信息
    print(f"\n统计信息:")
    print(f"找到 {total_unmatched} 个不匹配的标签文件")
    if args.dry_run:
        print(f"（模拟）将删除 {total_deleted} 个文件")
    else:
        print(f"实际删除 {total_deleted} 个文件")
    
    if total_unmatched == 0:
        print("所有标签文件都有对应的图像文件，无需清理！")
        print("提示: 如果您怀疑存在匹配问题，请使用 --debug 参数查看详细的匹配信息")
    else:
        print(f"清理完成！移除了多余的标签文件。")


if __name__ == "__main__":
    main()