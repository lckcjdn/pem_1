# 双分割头训练使用指南

本文档介绍如何使用修改后的双分割头架构进行训练，该架构允许在同一个骨干网络上同时训练两个分割头，分别处理不同的数据集。文档包含最新的学习率优化策略和训练恢复方法。

## 功能概述

双分割头架构的主要特点：

1. **共享骨干网络**：两个分割头共享同一个特征提取骨干网络
2. **独立监督信号**：每个分割头接收来自特定数据集的监督信号
3. **独立损失计算**：根据dataset_id字段将样本路由到对应的分割头进行损失计算
4. **协同优化**：通过联合训练，实现不同数据集知识的迁移和融合
5. **优化的学习率策略**：支持组件差异化学习率、学习率预热和梯度裁剪

## 文件结构

- `pem/Double_maskformer.py`：双分割头模型的主要实现
- `pem/data/dataset_mappers/double_mask_former_semantic_dataset_mapper.py`：支持dataset_id的数据集映射器
- `train_net_double_maskformer.py`：训练脚本（已添加学习率记录功能）
- `configs/double_maskformer/bdd100k_vestas_double_maskformer.yaml`：推荐使用的配置文件

## 使用方法

### 1. 数据集准备

#### 支持的数据集

该项目支持以下数据集：
- **BDD100K**：自动驾驶场景数据集
- **Mapillary Vistas**：全球多地区街道场景数据集
- **ApolloScape**：自动驾驶场景数据集

#### 数据集组织方式

请确保数据集按照以下结构组织：

```
datasets/
└── cityscapes/
    ├── gtFine/
    │   ├── train/
    │   │   ├── apo/         # Apollo数据集标注
    │   │   ├── bdd100k/     # BDD100K数据集标注
    │   │   ├── bdd100k_0/   # BDD100K_0数据集标注
    │   │   └── vestas/      # Vestas数据集标注
    │   └── val/
    │       ├── apo/         # Apollo数据集验证集标注
    │       ├── bdd100k/     # BDD100K数据集验证集标注
    │       ├── bdd100k_0/   # BDD100K_0数据集验证集标注
    │       └── vestas/      # Vestas数据集验证集标注
    └── leftImg8bit/
        ├── train/
        │   ├── apo/         # Apollo数据集图像
        │   ├── bdd100k/     # BDD100K数据集图像
        │   ├── bdd100k_0/   # BDD100K_0数据集图像
        │   └── vestas/      # Vestas数据集图像
        └── val/
            ├── apo/         # Apollo数据集验证集图像
            ├── bdd100k/     # BDD100K数据集验证集图像
            └── vestas/      # Vestas数据集验证集图像

```

#### 数据集预处理步骤
数据集的格式统一都为Cityscapes格式，预处理步骤如下：
1. **BDD100K数据集**
   - 下载BDD100K数据集（images和labels部分）
   ```
2. **Vestas数据集**
   - 下载Vestas数据集（images和labels部分）
   ```
3. **Apollo数据集**
   - 下载Apollo数据集（images和labels部分）



#### 标签文件位置与说明

项目中包含多个标签定义文件，用于不同数据集的类别映射：

1. **labels_double.py** (位于 `pem/data/datasets/`)
   - 双分割头专用标签定义
   - 支持两个分割头使用不同的类别集
   - 适用于BDD100K和Vestas数据集的联合训练场景

2. **labels_bdd100k.py** (位于 `pem/data/datasets/`)
   - BDD100K数据集的标签定义
   - 包含BDD100K和BDD100K_0数据集的类别映射

#### 标签映射规则

在双分割头架构中，两个分割头可以使用不同的类别定义：
- **分割头1**：通常用于Apollo类别
- **分割头2**：通常用于Vistas或自定义类别（如Vestas）

确保在配置文件中正确设置`MODEL.DoubleMaskFormer.NUM_CLASSES`参数，以匹配使用的类别数量。

### 数据集注册方式

数据集注册会在导入相关模块时自动完成，无需手动注册。已注册的数据集名称为：

- Apollo训练集: `cityscapes_apollo_sem_seg_train`
- Apollo验证集: `cityscapes_apollo_sem_seg_val`
- Vestas训练集: `cityscapes_vestas_sem_seg_train`
- Vestas验证集: `cityscapes_vestas_sem_seg_val`
- BDD100K训练集: `cityscapes_bdd100k_sem_seg_train`
- BDD100K验证集: `cityscapes_bdd100k_sem_seg_val`
- 混合数据集(BDD100K_0+Apollo): `cityscapes_bdd100k_0_apo_mixed_sem_seg_train` #暂时无用

数据集注册代码位于`pem/data/datasets/register_double_cityscapes_with_bdd100k.py`文件中，支持多种数据集类型的统一注册。当使用`train_net_double_maskformer.py`脚本时，这些数据集会自动被导入和注册。

### 配置文件说明

bdd100k_vestas_double_maskformer.yaml是推荐使用的配置文件，位于`configs/double_maskformer/bdd100k_vestas_double_maskformer.yaml`。主要参数包括：

#### 模型配置
- `MODEL.DoubleMaskFormer.NUM_CLASSES`: 设置分割头1和分割头2的类别数量
- `SOLVER.MAX_ITER`: 最大迭代次数
- `SOLVER.HEAD1_LR_MULTIPLIER`: 分割头1学习率乘数
- `SOLVER.HEAD2_LR_MULTIPLIER`: 分割头2学习率乘数
- `SOLVER.IMS_PER_BATCH`: 每批次图像数量
#### 数据集配置
- `DATASETS.TRAIN`: 训练数据集配置，可指定多个数据集，例如["bdd100k_sem_seg_train", "vestas_sem_seg_train"]
- `DATASETS.TEST`: 测试数据集配置，例如["bdd100k_sem_seg_val"]
- `DATALOADER.NUM_WORKERS`: 数据加载工作线程数，建议设置为CPU核心数的一半

#### 其他配置
- `OUTPUT_DIR`: 输出目录
- `SOLVER.CHECKPOINT_PERIOD`: 模型保存周期
- `TEST.EVAL_PERIOD`: 评估周期

### 训练与评估

#### 推荐训练命令

使用以下命令启动双分割头训练：

```bash
python train_net_double_maskformer.py --config-file configs\double_maskformer\bdd100k_vestas_double_maskformer.yaml
```
主要参数说明：
- `--config-file`: 指定配置文件路径
- `--num-gpus`: 指定使用的GPU数量（可选）

#### 从检查点继续训练
如果您需要从之前的检查点继续训练，使用`--resume`参数：
```bash
python train_net_double_maskformer.py \
    --config-file configs\double_maskformer\bdd100k_vestas_double_maskformer.yaml \
    --resume \
    OUTPUT_DIR output/double_bdd100k_vestas
```
#### 评估命令
训练完成后，重复评估时使用以下命令评估模型性能：
```bash
python train_net_double_maskformer.py \
    --config-file configs\double_maskformer\bdd100k_vestas_double_maskformer.yaml \
    --eval-only \
    MODEL.WEIGHTS output/double_bdd100k_vestas/model_final.pth
```
对于双分割头模型，评估结果将分别显示两个分割头的性能指标。

### 推理

#### 推理命令及参数设置

使用以下命令进行图像推理并保存分割结果：

```bash
python train_net_double_maskformer.py \
    --config-file configs\double_maskformer\bdd100k_vestas_double_maskformer.yaml \
    --eval-only \
    --save-results \
    MODEL.WEIGHTS output/double_bdd100k_vestas/model_final.pth \
    OUTPUT_DIR output/inference_results
```

#### 推理参数说明

- `--config-file`: 指定配置文件路径
- `--eval-only`: 启用评估模式（必须使用此模式才能进行推理）
- `--save-results`: 启用推理结果保存功能
- `MODEL.WEIGHTS`: 指定模型权重文件路径
- `OUTPUT_DIR`: 指定输出目录（可选，默认值从配置文件中获取）

#### 输入文件位置

在进行推理前，需要将待处理的图像文件放在以下目录：
```
{OUTPUT_DIR}/input_images/
```

支持的图像格式包括：
- .jpg
- .png
- .jpeg

#### 默认输出位置

推理结果将默认保存在以下目录：

```
{OUTPUT_DIR}/
├── segmentation_head1/  # 分割头1的输出结果
│   ├── *seg.png        # 分割结果图（纯分割）
│   └── *overlay.png    # 分割结果叠加图（原图+分割）
└── segmentation_head2/  # 分割头2的输出结果
    ├── *seg.png        # 分割结果图（纯分割）
    └── *overlay.png    # 分割结果叠加图（原图+分割）
```

- **分割头1**：使用Apollo/BDD100K标签集进行分割
- **分割头2**：使用Vestas标签集进行分割

每个输入图像会生成两张输出图：
- `{图像名}_seg.png`：纯分割结果图
- `{图像名}_overlay.png`：分割结果与原图的叠加图