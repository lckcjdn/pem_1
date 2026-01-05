#!/usr/bin/env python
"""
调试融合逻辑，跳过实例化车道线分割部分
"""

import os
import sys
import numpy as np
import cv2

# 添加当前目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pem', 'data', 'datasets'))
from labels_double import vestas_labels, apollo_labels
from post_process_fusion import PostProcessFusion

def create_debug_masks():
    """创建调试用的分割掩码"""
    # 创建512x512的测试图像
    height, width = 512, 512
    
    # 分割头一（Apollo）的测试掩码 - 只包含背景和少量road
    head1_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 在头一中添加一些Apollo类别
    # road (trainId=1)
    head1_mask[100:200, 100:200] = apollo_labels[1].color  # road
    
    # 分割头二（Vestas）的测试掩码 - 包含多个类别
    head2_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 在头二中添加Vestas类别
    # car (trainId=1)
    head2_mask[50:150, 300:400] = vestas_labels[1].color  # car
    # human (trainId=2)
    head2_mask[200:300, 300:400] = vestas_labels[2].color  # human
    # road (trainId=3)
    head2_mask[350:450, 300:400] = vestas_labels[3].color  # road
    # lane_mark (trainId=4)
    head2_mask[100:200, 50:150] = vestas_labels[4].color  # lane_mark
    # curb (trainId=5)
    head2_mask[250:350, 50:150] = vestas_labels[5].color  # curb
    # road_mark (trainId=6)
    head2_mask[400:500, 50:150] = vestas_labels[6].color  # road_mark
    # guard_rail (trainId=7)
    head2_mask[50:150, 450:500] = vestas_labels[7].color  # guard_rail
    # traffic_sign (trainId=8)
    head2_mask[200:300, 450:500] = vestas_labels[8].color  # traffic_sign
    
    return head1_mask, head2_mask

def debug_fusion_process():
    """调试融合处理过程"""
    print("=" * 80)
    print("调试融合处理过程")
    print("=" * 80)
    
    # 创建测试数据
    head1_mask, head2_mask = create_debug_masks()
    
    # 初始化后处理类（不使用实例化优化）
    post_processor = PostProcessFusion(use_instance_optimization=False)
    
    print("\n1. 原始分割结果检查:")
    print(f"  分割头一形状: {head1_mask.shape}")
    print(f"  分割头二形状: {head2_mask.shape}")
    
    # 步骤1: 映射到trainId
    print("\n2. 映射分割结果到trainId:")
    head1_trainId = post_processor.map_to_trainId(head1_mask, is_head1=True)
    head2_trainId = post_processor.map_to_trainId(head2_mask, is_head1=False)
    
    print(f"  分割头一trainId唯一值: {np.unique(head1_trainId)}")
    print(f"  分割头二trainId唯一值: {np.unique(head2_trainId)}")
    
    # 检查Vestas类别在head2_trainId中的存在情况
    vestas_trainIds = [label.trainId for label in vestas_labels if label.trainId != 0]
    print(f"  Vestas有效trainId: {vestas_trainIds}")
    
    for trainId in vestas_trainIds:
        pixel_count = np.sum(head2_trainId == trainId)
        label_name = vestas_labels[trainId].name if trainId in vestas_labels else "unknown"
        print(f"    {label_name} (trainId={trainId}): {pixel_count} 像素")
    
    # 步骤2: 跳过形态学处理
    print("\n3. 跳过形态学处理:")
    head1_processed = head1_trainId
    head2_processed = head2_trainId
    
    print(f"  处理后分割头一trainId唯一值: {np.unique(head1_processed)}")
    print(f"  处理后分割头二trainId唯一值: {np.unique(head2_processed)}")
    
    # 步骤3: 融合分割结果
    print("\n4. 融合分割结果:")
    
    # 手动执行融合逻辑进行调试
    fused_mask = head1_processed.copy()
    
    print(f"  融合前基础mask唯一值: {np.unique(fused_mask)}")
    
    # 检查融合标签定义
    print("\n5. 融合标签定义检查:")
    for label in post_processor.fused_labels:
        print(f"  {label.name}: trainId={label.trainId}")
    
    # 检查_get_fused_trainId方法
    print("\n6. _get_fused_trainId方法检查:")
    vestas_category_names = [label.name for label in vestas_labels if label.trainId != 0 and label.name != 'lane_mark']
    for category in vestas_category_names:
        fused_trainId = post_processor._get_fused_trainId(category)
        print(f"  {category} -> 融合trainId: {fused_trainId}")
    
    # 执行基础融合策略
    print("\n7. 执行基础融合策略:")
    
    # Vestas特有类别
    vestas_special_categories = {'curb', 'guard_rail'}
    # 与Apollo共有的类别
    vestas_common_categories = {'car', 'human', 'road', 'road_mark', 'traffic_sign'}
    
    for label in vestas_labels:
        if label.trainId == 0:  # 背景
            continue
        elif label.name == 'lane_mark':  # lane_mark已经被替换
            continue
        
        fused_trainId = post_processor._get_fused_trainId(label.name)
        if fused_trainId == 0:
            print(f"  ⚠️ {label.name}: 未找到对应的融合trainId")
            continue
            
        vestas_class_mask = (head2_processed == label.trainId)
        pixel_count = np.sum(vestas_class_mask)
        
        print(f"  {label.name} (Vestas trainId={label.trainId} -> 融合trainId={fused_trainId}): {pixel_count} 像素")
        
        if label.name in vestas_special_categories:
            # Vestas特有类别：只在Apollo背景区域使用
            apollo_background_mask = (head1_processed == 0)
            valid_mask = vestas_class_mask & apollo_background_mask
            
            if valid_mask.any():
                fused_mask[valid_mask] = fused_trainId
                print(f"    -> 在Apollo背景区域添加 {np.sum(valid_mask)} 像素")
            else:
                print(f"    -> 没有有效的Apollo背景区域")
                
        elif label.name in vestas_common_categories:
            # 与Apollo共有的类别：直接使用Vestas的结果
            if vestas_class_mask.any():
                fused_mask[vestas_class_mask] = fused_trainId
                print(f"    -> 直接添加 {pixel_count} 像素")
            else:
                print(f"    -> 没有像素需要添加")
    
    print(f"\n8. 融合后结果检查:")
    print(f"  融合结果唯一trainId: {np.unique(fused_mask)}")
    
    # 统计每个类别的像素数
    print("\n9. 融合结果统计:")
    for label in post_processor.fused_labels:
        pixel_count = np.sum(fused_mask == label.trainId)
        if pixel_count > 0:
            print(f"  {label.name} (trainId={label.trainId}): {pixel_count} 像素")
    
    # 检查Vestas类别是否被正确融合
    print("\n10. Vestas类别融合检查:")
    missing_categories = []
    for category in vestas_category_names:
        fused_trainId = post_processor._get_fused_trainId(category)
        if fused_trainId > 0:
            pixel_count = np.sum(fused_mask == fused_trainId)
            if pixel_count > 0:
                print(f"  ✓ {category}: {pixel_count} 像素")
            else:
                print(f"  ✗ {category}: 0 像素")
                missing_categories.append(category)
        else:
            print(f"  ? {category}: 未找到融合trainId")
            missing_categories.append(category)
    
    # 保存调试结果
    output_dir = "./test_results/debug_fusion"
    os.makedirs(output_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(output_dir, "head1_mask.png"), head1_mask)
    cv2.imwrite(os.path.join(output_dir, "head2_mask.png"), head2_mask)
    cv2.imwrite(os.path.join(output_dir, "fused_mask.png"), fused_mask * 50)  # 放大显示
    
    # 保存详细的调试信息
    debug_info_path = os.path.join(output_dir, "debug_info.txt")
    with open(debug_info_path, 'w', encoding='utf-8') as f:
        f.write("融合调试信息\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("分割头一trainId统计:\n")
        for trainId in np.unique(head1_processed):
            pixel_count = np.sum(head1_processed == trainId)
            f.write(f"  trainId={trainId}: {pixel_count} 像素\n")
        
        f.write("\n分割头二trainId统计:\n")
        for trainId in np.unique(head2_processed):
            pixel_count = np.sum(head2_processed == trainId)
            label_name = vestas_labels[trainId].name if trainId in vestas_labels else "unknown"
            f.write(f"  {label_name} (trainId={trainId}): {pixel_count} 像素\n")
        
        f.write("\n融合结果trainId统计:\n")
        for trainId in np.unique(fused_mask):
            pixel_count = np.sum(fused_mask == trainId)
            label_name = "unknown"
            for label in post_processor.fused_labels:
                if label.trainId == trainId:
                    label_name = label.name
                    break
            f.write(f"  {label_name} (trainId={trainId}): {pixel_count} 像素\n")
        
        f.write("\n缺失的Vestas类别:\n")
        for category in missing_categories:
            f.write(f"  {category}\n")
    
    print(f"\n调试结果已保存到: {output_dir}")
    
    if missing_categories:
        print(f"\n⚠️ 警告: 以下Vestas类别缺失: {missing_categories}")
        return False
    else:
        print(f"\n✓ 所有Vestas类别都已正确融合")
        return True

if __name__ == "__main__":
    print("开始调试融合处理过程...")
    
    success = debug_fusion_process()
    
    if success:
        print("\n🎉 调试完成，融合逻辑正常")
    else:
        print("\n❌ 调试发现问题，请检查融合逻辑")
    
    print("调试完成！")