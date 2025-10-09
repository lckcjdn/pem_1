# 🚗 PEM

> 基于 PEM (Prototype-based Efficient MaskFormer) 的停车场与道路元素映射系统
> 
> 专注于道路场景语义分割，支持车道线、道路标记、交通标志等多类别识别

## 📋 目录

- [环境配置](#-环境配置)
- [训练说明](#️-训练说明)
- [数据集组织](#️-数据集组织结构)
- [性能记录](#-历史版本性能记录)
- [致谢](#-致谢)

## 🧩 环境配置

### 安装步骤

1. **激活环境**
   ```bash
   conda activate pem_ss
   ```

2. **依赖检查**
   确保已正确安装以下依赖：
   - PyTorch
   - Detectron2
   - CityscapesScripts

## 🏋️‍♂️ 训练说明

### 1️⃣ 启动训练

**Neusoft版本训练：**
```bash
python train_net_neusoft.py --num-gpus 1 \
  --config-file configs/cityscapes/semantic-segmentation/pem_R50_bs32_90k.yaml \
  MODEL.WEIGHTS output/model
```

**Mark版本训练：**
```bash
python train_net_mark.py --num-gpus 1 --config-file [CONFIG_PATH]
```

### 2️⃣ 训练配置修改

| 模块 | 文件路径 | 修改内容 | 说明 |
|------|----------|----------|------|
| **类别数** | `configs/cityscapes/semantic-segmentation/pem_R50_bs32_90k.yaml` | 修改 `NUM_CLASSES` | 若类别数量变化，请同步修改此参数 |
| **训练步长** | `configs/Base-Cityscapes-SemanticSegmentation.yaml` | 修改 `MAX_ITER` | 控制训练总步数，默认约 90,000 步 |
| **标签定义** | `cityscapesscripts/helpers/label.py` | 替换为自定义版本 | 已将修改后的 label.py 放置于当前目录中，请直接覆盖 |

## 🗂️ 数据集组织结构

PEM 使用与 Cityscapes 相同的数据结构格式：

```
datasets/cityscapes/
├── gtFine/             # 存放灰度标签图（Ground Truth Masks, grayscale）
│   ├── train/
│   ├── val/
│   └── test/
└── leftImg8bit/        # 存放原始RGB图像
    ├── train/
    ├── val/
    └── test/
```


## 📊 历史版本性能记录

以下为各版本在验证集上的 IoU 指标表现：

### 🧱 V1 — 区分车道线与道路标记，增加路沿与护栏

| 类别 | IoU | 语义 |
|------|-----|------|
| car | **0.945** | 车辆 |
| human | 0.834 | 行人 |
| road | **0.925** | 道路 |
| lane_mark | 0.439 | 车道线 |
| curb | 0.727 | 路沿 |
| road_mark | 0.575 | 道路标记 |
| guard_rail | 0.723 | 护栏 |
| **平均分** | **0.738** | |

### 🧭 V2 — 增加隔离带（Separator）类别 + 车道线实例化

<details>
<summary>点击查看详细结果</summary>

| 类别 | IoU |
|------|-----|
| car | 0.942 |
| human | 0.822 |
| road | 0.921 |
| lane_mark | 0.420 |
| curb | 0.709 |
| road_mark | 0.548 |
| guard_rail | 0.712 |
| **平均分** | **0.725** |

</details>

### 🚧 V3 — 增加交通标志牌 + 黄白实虚线区分

<details>
<summary>点击查看详细结果</summary>

| 类别 | IoU |
|------|-----|
| car | 0.943 |
| human | 0.832 |
| road | 0.908 |
| lane_mark | 0.395 |
| curb | 0.716 |
| road_mark | 0.551 |
| guard_rail | 0.757 |
| traffic_sign | **0.805** |
| **平均分** | **0.738** |

</details>

### 🛣️ V3.1 — 精修标注与优化训练

<details>
<summary>点击查看详细结果</summary>

| 类别 | IoU |
|------|-----|
| car | 0.941 |
| human | 0.829 |
| road | 0.917 |
| lane_mark | 0.407 |
| curb | 0.697 |
| road_mark | 0.557 |
| guard_rail | 0.755 |
| traffic_sign | 0.808 |
| **平均分** | **0.739** |

</details>

### 🛑 V4 — 道路标线细分语义（25 类）

<details>
<summary>点击查看详细结果</summary>

| 类别 | IoU | 类别 | IoU |
|------|-----|------|-----|
| box_junction | **0.903** | channelizing_line | 0.846 |
| crosswalk | 0.858 | motor_prohibited | 0.855 |
| stop_line | 0.697 | slow | 0.834 |
| solid_single_white | 0.475 | motor_priority_lane | 0.675 |
| solid_single_yellow | 0.760 | motor_waiting_zone | 0.739 |
| solid_single_red | 0.613 | left_turn_box | 0.539 |
| solid_double_white | 0.820 | motor_icon | 0.542 |
| solid_double_yellow | **0.868** | bike_icon | 0.556 |
| dashed_single_white | 0.701 | parking_lot | 0.650 |
| dashed_single_yellow | 0.699 | | |
| left_arrow | 0.517 | | |
| straight_arrow | 0.594 | | |
| right_arrow | 0.193 | | |
| left_straight_arrow | 0.542 | | |
| right_straight_arrow | 0.552 | | |

**平均分：0.668**

</details>

### ⚙️ V5 — 道路标线语义合并版本

<details>
<summary>点击查看详细结果</summary>

| 类别 | IoU |
|------|-----|
| crosswalk | 0.866 |
| stop_line | 0.745 |
| solid_single_white | 0.767 |
| solid_single_yellow | 0.358 |
| solid_double_white | 0.827 |
| solid_double_yellow | **0.872** |
| dashed_single_white | 0.712 |
| dashed_single_yellow | 0.744 |
| arrow | **0.819** |
| **平均分** | **0.746** |

</details>

### 🛰️ V6 — Apollo 格式道路标线语义

<details>
<summary>点击查看详细结果</summary>

| 类别 | IoU | 类别 | IoU |
|------|-----|------|-----|
| background | **0.996** | a_w_tl | 0.740 |
| s_w_d | 0.763 | a_w_tr | 0.766 |
| s_y_d | 0.821 | a_w_l | 0.783 |
| ds_y_dn | 0.571 | a_w_r | 0.700 |
| sb_w_do | 0.324 | a_n_lu | 0.000 |
| sb_y_do | 0.299 | b_n_sr | 0.393 |
| b_w_g | 0.677 | d_wy_za | 0.000 |
| s_w_s | 0.645 | r_wy_np | 0.000 |
| s_w_c | 0.790 | vom_wy_n | 0.618 |
| s_y_c | 0.788 | om_n_n | 0.361 |
| s_w_p | 0.595 | | |
| c_wy_z | 0.767 | | |
| a_w_u | 0.736 | | |
| a_w_t | 0.790 | | |

**平均分：0.580**

</details>

## 🧠 致谢

本项目基于以下开源项目开发：

- **[Detectron2](https://github.com/facebookresearch/detectron2)** - Meta AI 的目标检测框架
- **[PEM](https://github.com/NiccoloCavagnero/PEM)** - Prototype-based Efficient MaskFormer
- **[Cityscapes](https://www.cityscapes-dataset.com/)** - 城市场景数据集
- **[ApolloScape](http://apolloscape.auto/)** - Apollo 自动驾驶数据集

### 数据来源
- Vistas 数据集
- ApolloScape 数据集  
---

## 🚀 后续计划

- [ ] **继续训练** - 在现有模型基础上继续训练，提升模型性能
- [ ] **类别合并** - 优化语义类别定义，合并相似类别以提高训练效率
- [ ] **跨域标注** - 使用Vistas训练的模型对Apollo数据进行自动标注，构建混合数据集进行全类别训练
- [ ] **伪标签监督** - 引入伪标签技术，利用无标注数据提升模型泛化能力
- [ ] **双分割头架构** - 实现共享编码器+双分割头设计，独立处理两个数据集的监督信号

---

<div align="center">
  <sub>Built with ❤️ for autonomous driving research</sub>
</div>
