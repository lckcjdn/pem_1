# 双分割头评估模式使用指南

本文档介绍了如何使用修改后的双分割头架构进行特殊评估，该评估模式允许图像同时输入两个分割头，然后根据置信度规则处理和融合结果。

## 功能概述

修改后的代码支持一种特殊的评估模式，该模式实现了以下功能：

1. 将同一图像同时输入到两个分割头，生成两套预测结果
2. 根据以下规则处理预测结果：
   - 如果两个分割头预测的标签不同且不重合，则保留置信度高的标签，将低置信度的标签归为背景
   - 如果两个分割头预测的标签不同但属于重合类别，则统一采用置信度高的类别
3. 处理后的结果分别输入各自的评估器进行评估
4. 提供合并的结果图用于测试展示

## 配置与使用

### 1. 数据集映射器配置

在评估脚本中，创建DoubleMaskFormerSemanticDatasetMapper时启用特殊评估模式：

```python
from pem.data.dataset_mappers import DoubleMaskFormerSemanticDatasetMapper

# 创建评估用的数据集映射器，启用特殊评估模式
mapper = DoubleMaskFormerSemanticDatasetMapper.from_config(
    cfg,
    is_train=False,
    dataset_id=0,  # 数据集ID (0或1)
    is_evaluation=True  # 启用特殊评估模式
)
```

### 2. 修改评估脚本

在评估过程中，需要：

1. 使用修改后的数据集映射器加载测试数据
2. 运行模型进行推理
3. 处理特殊格式的输出结果

```python
# 加载模型和数据
model = get_model(cfg)

# 获取处理后的输出
outputs = model(batched_inputs)

# 处理评估输出
for output in outputs:
    # 获取每个分割头的处理后结果
    sem_seg_head1 = output["sem_seg_head1"]
    sem_seg_head2 = output["sem_seg_head2"]
    
    # 获取合并后的结果
    merged_sem_seg = output["merged_sem_seg"]
    
    # 分别评估两个分割头的结果
    evaluate_head1(sem_seg_head1, ground_truth1)
    evaluate_head2(sem_seg_head2, ground_truth2)
    
    # 可以保存合并结果用于测试展示
    save_result(merged_sem_seg, output_path)
```

## 实现细节

### 双分割头前向传播

修改后的`forward`方法在非训练模式下会检查输入中是否包含`is_evaluation=True`标志。如果设置了此标志，将启用特殊评估模式，并调用`_process_evaluation_outputs`方法处理结果。

### 标签处理逻辑

`_process_evaluation_outputs`方法实现了以下核心逻辑：

1. 检测两个分割头预测中的有效标签（非背景）
2. 比较同一位置的预测结果
3. 当两个预测都有效但不同时，根据置信度决定保留哪个标签
4. 为两个分割头生成处理后的输出，同时创建合并结果

### 数据集映射器

`DoubleMaskFormerSemanticDatasetMapper`已更新以支持：
- `dataset_id`：区分不同数据集
- `is_evaluation`：启用特殊评估模式

## 注意事项

1. 特殊评估模式仅在推理时启用，不影响训练过程
2. 默认假设背景类别为0
3. 如果两个分割头的输出维度不同，代码会自动调整大小以确保兼容性
4. 处理后的结果将包含三个关键字段：`sem_seg_head1`、`sem_seg_head2`和`merged_sem_seg`

## 扩展建议

1. 根据实际数据集调整背景类别的定义
2. 考虑为评估器添加对特殊输出格式的支持
3. 如需更复杂的标签融合规则，可以修改`_process_evaluation_outputs`方法