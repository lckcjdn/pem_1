# 🚗 PEM: Parking & Road Element Mapper
---

## 🧩 环境配置

在使用前请确保环境已正确配置：

```bash
conda activate pem_ss
如需在其他环境中运行，请确保已正确安装依赖（PyTorch、Detectron2、CityscapesScripts 等）。

🏋️‍♂️ 训练说明
1️⃣ 启动训练
使用以下命令开始训练：


python train_net_neusoft.py --num-gpus 1 \
--config-file configs/cityscapes/semantic-segmentation/pem_R50_bs32_90k.yaml \
MODEL.WEIGHTS output/model

python train_net_mark.py --num-gpus 1 --config-file ...
2️⃣ 训练配置修改
模块	文件路径	修改内容	说明
类别数	configs/cityscapes/semantic-segmentation/pem_R50_bs32_90k.yaml	修改 NUM_CLASSES	若类别数量变化，请同步修改此参数
训练步长	configs/Base-Cityscapes-SemanticSegmentation.yaml	修改 MAX_ITER	控制训练总步数，默认约 90,000 步
标签定义	cityscapesscripts/helpers/label.py	替换为自定义版本	已将修改后的 label.py 放置于当前目录中，请直接覆盖

🗂️ 数据集组织结构
PEM 使用与 Cityscapes 相同的数据结构格式：

datasets/cityscapes/
├── gtFine/             # 存放灰度标签图（Ground Truth Masks, grayscale）
│   ├── train/
│   ├── val/
│   └── test/
└── leftImg8bit/        # 存放原始RGB图像
    ├── train/
    ├── val/
    └── test/
gtFine 为灰度标签图，与 leftImg8bit 中原图一一对应。

若原始标签为彩色图，可通过以下脚本转换为灰度：

python data_preprocess.py /path/to/color_labels/ \
--gray_output datasets/cityscapes/gtFine/val \
--color_output datasets/cityscapes/gtFine/val_vis
📊 历史版本性能记录
以下为各版本在验证集上的 IoU 指标表现：

🧱 V1 — 区分车道线与道路标记，增加路沿与护栏
类别	IoU	语义
car	0.945	车辆
human	0.834	行人
road	0.925	道路
lane_mark	0.439	车道线
curb	0.727	路沿
road_mark	0.575	道路标记
guard_rail	0.723	护栏
平均分	0.738	

🧭 V2 — 增加隔离带（Separator）类别 + 车道线实例化
类别	IoU
car	0.942
human	0.822
road	0.921
lane_mark	0.420
curb	0.709
road_mark	0.548
guard_rail	0.712
平均分	0.725

🚧 V3 — 增加交通标志牌 + 黄白实虚线区分
类别	IoU
car	0.943
human	0.832
road	0.908
lane_mark	0.395
curb	0.716
road_mark	0.551
guard_rail	0.757
traffic_sign	0.805
平均分	0.738

🛣️ V3.1 — 精修标注与优化训练
类别	IoU
car	0.941
human	0.829
road	0.917
lane_mark	0.407
curb	0.697
road_mark	0.557
guard_rail	0.755
traffic_sign	0.808
平均分	0.739

🛑 V4 — 道路标线细分语义（25 类）
类别	IoU
box_junction	0.903
crosswalk	0.858
stop_line	0.697
solid_single_white	0.475
solid_single_yellow	0.760
solid_single_red	0.613
solid_double_white	0.820
solid_double_yellow	0.868
dashed_single_white	0.701
dashed_single_yellow	0.699
left_arrow	0.517
straight_arrow	0.594
right_arrow	0.193
left_straight_arrow	0.542
right_straight_arrow	0.552
channelizing_line	0.846
motor_prohibited	0.855
slow	0.834
motor_priority_lane	0.675
motor_waiting_zone	0.739
left_turn_box	0.539
motor_icon	0.542
bike_icon	0.556
parking_lot	0.650
平均分	0.668

⚙️ V5 — 道路标线语义合并版本
类别	IoU
crosswalk	0.866
stop_line	0.745
solid_single_white	0.767
solid_single_yellow	0.358
solid_double_white	0.827
solid_double_yellow	0.872
dashed_single_white	0.712
dashed_single_yellow	0.744
arrow	0.819
平均分	0.746

🛰️ V6 — Apollo 格式道路标线语义
类别	IoU
background	0.996
s_w_d	0.763
s_y_d	0.821
ds_y_dn	0.571
sb_w_do	0.324
sb_y_do	0.299
b_w_g	0.677
s_w_s	0.645
s_w_c	0.790
s_y_c	0.788
s_w_p	0.595
c_wy_z	0.767
a_w_u	0.736
a_w_t	0.790
a_w_tl	0.740
a_w_tr	0.766
a_w_l	0.783
a_w_r	0.700
a_n_lu	0.000
b_n_sr	0.393
d_wy_za	0.000
r_wy_np	0.000
vom_wy_n	0.618
om_n_n	0.361
平均分	0.580

🧠 致谢
本项目基于 Detectron2 框架开发与扩展。
数据来源包括 Cityscapes、ApolloScape 以及部分自建标注数据。