# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# 导入双数据集标签映射功能
from .labels_double import get_dataset_meta

# 定义cityscapes数据集的类别信息
CITYSCAPES_SEM_SEG_CATEGORIES = [
    # name, id, trainId, category, catId, hasInstances, ignoreInEval, color
    {"name": "unlabeled", "id": 0, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [0, 0, 0]},
    {"name": "ego vehicle", "id": 1, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [0, 0, 0]},
    {"name": "rectification border", "id": 2, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [0, 0, 0]},
    {"name": "out of roi", "id": 3, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [0, 0, 0]},
    {"name": "static", "id": 4, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [0, 0, 0]},
    {"name": "dynamic", "id": 5, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [111, 74, 0]},
    {"name": "ground", "id": 6, "trainId": 255, "category": "void", "catId": 0, "hasInstances": False, "ignoreInEval": True, "color": [81, 0, 81]},
    {"name": "road", "id": 7, "trainId": 0, "category": "flat", "catId": 1, "hasInstances": False, "ignoreInEval": False, "color": [128, 64, 128]},
    {"name": "sidewalk", "id": 8, "trainId": 1, "category": "flat", "catId": 1, "hasInstances": False, "ignoreInEval": False, "color": [244, 35, 232]},
    {"name": "parking", "id": 9, "trainId": 255, "category": "flat", "catId": 1, "hasInstances": False, "ignoreInEval": True, "color": [250, 170, 160]},
    {"name": "rail track", "id": 10, "trainId": 255, "category": "flat", "catId": 1, "hasInstances": False, "ignoreInEval": True, "color": [230, 150, 140]},
    {"name": "building", "id": 11, "trainId": 2, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": False, "color": [70, 70, 70]},
    {"name": "wall", "id": 12, "trainId": 3, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": False, "color": [102, 102, 156]},
    {"name": "fence", "id": 13, "trainId": 4, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": False, "color": [190, 153, 153]},
    {"name": "guard rail", "id": 14, "trainId": 255, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": True, "color": [180, 165, 180]},
    {"name": "bridge", "id": 15, "trainId": 255, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": True, "color": [150, 100, 100]},
    {"name": "tunnel", "id": 16, "trainId": 255, "category": "construction", "catId": 2, "hasInstances": False, "ignoreInEval": True, "color": [150, 120, 90]},
    {"name": "pole", "id": 17, "trainId": 5, "category": "object", "catId": 3, "hasInstances": False, "ignoreInEval": False, "color": [153, 153, 153]},
    {"name": "polegroup", "id": 18, "trainId": 255, "category": "object", "catId": 3, "hasInstances": False, "ignoreInEval": True, "color": [153, 153, 153]},
    {"name": "traffic light", "id": 19, "trainId": 6, "category": "object", "catId": 3, "hasInstances": False, "ignoreInEval": False, "color": [250, 170, 30]},
    {"name": "traffic sign", "id": 20, "trainId": 7, "category": "object", "catId": 3, "hasInstances": False, "ignoreInEval": False, "color": [220, 220, 0]},
    {"name": "vegetation", "id": 21, "trainId": 8, "category": "nature", "catId": 4, "hasInstances": False, "ignoreInEval": False, "color": [107, 142, 35]},
    {"name": "terrain", "id": 22, "trainId": 9, "category": "nature", "catId": 4, "hasInstances": False, "ignoreInEval": False, "color": [152, 251, 152]},
    {"name": "sky", "id": 23, "trainId": 10, "category": "sky", "catId": 5, "hasInstances": False, "ignoreInEval": False, "color": [70, 130, 180]},
    {"name": "person", "id": 24, "trainId": 11, "category": "human", "catId": 6, "hasInstances": True, "ignoreInEval": False, "color": [220, 20, 60]},
    {"name": "rider", "id": 25, "trainId": 12, "category": "human", "catId": 6, "hasInstances": True, "ignoreInEval": False, "color": [255, 0, 0]},
    {"name": "car", "id": 26, "trainId": 13, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": False, "color": [0, 0, 142]},
    {"name": "truck", "id": 27, "trainId": 14, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": False, "color": [0, 0, 70]},
    {"name": "bus", "id": 28, "trainId": 15, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": False, "color": [0, 60, 100]},
    {"name": "caravan", "id": 29, "trainId": 255, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": True, "color": [0, 0, 90]},
    {"name": "trailer", "id": 30, "trainId": 255, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": True, "color": [0, 0, 110]},
    {"name": "train", "id": 31, "trainId": 16, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": False, "color": [0, 80, 100]},
    {"name": "motorcycle", "id": 32, "trainId": 17, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": False, "color": [0, 0, 230]},
    {"name": "bicycle", "id": 33, "trainId": 18, "category": "vehicle", "catId": 7, "hasInstances": True, "ignoreInEval": False, "color": [119, 11, 32]},
    {"name": "license plate", "id": -1, "trainId": -1, "category": "vehicle", "catId": 7, "hasInstances": False, "ignoreInEval": True, "color": [0, 0, 142]},
]

# 加载cityscapes格式的语义分割数据集
def load_cityscapes_semantic(image_dir, gt_dir, dataset_type=None):
    """
    Args:
        image_dir (str): 图像目录路径，例如 "~/cityscapes/leftImg8bit/train".
        gt_dir (str): 标注目录路径，例如 "~/cityscapes/gtFine/train".
        dataset_type (str): 数据集类型，'apo'或'vestas'，用于确定子目录

    Returns:
        list[dict]: Detectron2标准格式的数据集字典列表
    """
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"标注目录不存在: {gt_dir}")
    
    # 根据dataset_type确定子目录
    if dataset_type == "apo":
        subdir = "apo"
        dataset_id = 0  # Apollo数据集ID
    elif dataset_type == "vestas":
        subdir = "vestas"
        dataset_id = 1  # Vestas数据集ID
    else:
        # 如果未指定dataset_type，尝试从image_dir推断
        if "vestas" in image_dir.lower():
            subdir = "vestas"
            dataset_id = 1
        else:
            subdir = "apo"
            dataset_id = 0
    
    # 构建完整的图像和标注目录路径
    full_image_dir = os.path.join(image_dir, subdir)
    full_gt_dir = os.path.join(gt_dir, subdir)
    
    # 检查子目录是否存在
    if not os.path.exists(full_image_dir):
        raise FileNotFoundError(f"图像子目录不存在: {full_image_dir}")
    if not os.path.exists(full_gt_dir):
        raise FileNotFoundError(f"标注子目录不存在: {full_gt_dir}")
    
    # 获取所有JPG和PNG图像文件
    image_files = glob.glob(os.path.join(full_image_dir, "*.jpg")) + glob.glob(os.path.join(full_image_dir, "*.png"))
    
    if not image_files:
        raise RuntimeError(f"在目录 {full_image_dir} 中未找到任何JPG或PNG图像文件")
    
    dataset_dicts = []
    
    for image_path in image_files:
        # 获取图像文件名（不包含扩展名）
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 构建对应的标注文件路径（使用相同的文件名但扩展名为.png）
        anno_path = os.path.join(full_gt_dir, img_basename + ".png")
        
        # 检查标注文件是否存在
        if os.path.exists(anno_path):
            # 获取图像尺寸
            try:
                with open(image_path, 'rb') as f:
                    img = Image.open(f)
                    height, width = img.height, img.width
                
                dataset_dicts.append({
                    "file_name": image_path,
                    "sem_seg_file_name": anno_path,
                    "height": height,
                    "width": width,
                    "image_id": img_basename,
                    "dataset_id": dataset_id,  # 添加数据集标识 (0: Apollo, 1: Vestas)
                    "dataset_type": subdir,  # 添加数据集类型信息
                })
            except Exception as e:
                print(f"处理图像 {image_path} 时出错: {e}")
                continue  # 跳过无法处理的图像
        else:
            print(f"警告: 未找到图像 {image_path} 对应的标注文件 {anno_path}")
    
    if len(dataset_dicts) == 0:
        raise RuntimeError(f"没有找到任何匹配的图像-标注对！请检查图像和标注文件的路径和命名。")
    
    print(f"成功加载 {len(dataset_dicts)} 个数据样本")
    return dataset_dicts

# 获取cityscapes元数据 - 根据数据集类型返回不同的元数据
def _get_cityscapes_meta(dataset_name="cityscapes_apollo"):
    """
    获取指定数据集的元数据
    
    Args:
        dataset_name (str): 数据集名称，用于确定使用哪种标签映射
    
    Returns:
        dict: 数据集元数据
    """
    return get_dataset_meta(dataset_name)

# 注册第一个cityscapes数据集（Apollo数据集）
def register_cityscapes_apollo(root):
    root = os.path.join(root, "cityscapes")
    dataset_name = "cityscapes_apollo"
    meta = _get_cityscapes_meta(dataset_name)
    
    # 打印注册信息
    print(f"\n注册Apollo数据集:")
    print(f"数据集根目录: {root}")
    
    # 注册训练集 - 使用apo子目录
    image_dir = os.path.join(root, "leftImg8bit", "train")
    gt_dir = os.path.join(root, "gtFine", "train")
    name = "cityscapes_apollo_sem_seg_train"
    
    print(f"注册训练集: {name}")
    print(f"  图像目录: {image_dir}/apo")
    print(f"  标注目录: {gt_dir}/apo")
    
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y, dataset_type="apo")
    )
    
    MetadataCatalog.get(name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="apollo",  # 添加数据集类型标识
        **meta,
    )
    
    # 注册验证集
    image_dir_val = os.path.join(root, "leftImg8bit", "val")
    gt_dir_val = os.path.join(root, "gtFine", "val")
    name_val = "cityscapes_apollo_sem_seg_val"
    
    print(f"注册验证集: {name_val}")
    print(f"  图像目录: {image_dir_val}/apo")
    print(f"  标注目录: {gt_dir_val}/apo")
    
    DatasetCatalog.register(
        name_val, lambda x=image_dir_val, y=gt_dir_val: load_cityscapes_semantic(x, y, dataset_type="apo")
    )
    
    MetadataCatalog.get(name_val).set(
        image_root=image_dir_val,
        sem_seg_root=gt_dir_val,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="apollo",  # 添加数据集类型标识
        **meta,
    )

# 注册第二个cityscapes数据集（Vestas数据集）
def register_cityscapes_vestas(root):
    root = os.path.join(root, "cityscapes")
    dataset_name = "cityscapes_vestas"
    meta = _get_cityscapes_meta(dataset_name)
    
    # 打印注册信息
    print(f"\n注册Vestas数据集:")
    print(f"数据集根目录: {root}")
    
    # 确保vestas子目录存在
    os.makedirs(os.path.join(root, "leftImg8bit", "train", "vestas"), exist_ok=True)
    os.makedirs(os.path.join(root, "gtFine", "train", "vestas"), exist_ok=True)
    os.makedirs(os.path.join(root, "leftImg8bit", "val", "vestas"), exist_ok=True)
    os.makedirs(os.path.join(root, "gtFine", "val", "vestas"), exist_ok=True)
    
    # 注册训练集 - 使用vestas子目录
    image_dir = os.path.join(root, "leftImg8bit", "train")
    gt_dir = os.path.join(root, "gtFine", "train")
    name = "cityscapes_vestas_sem_seg_train"
    
    print(f"注册训练集: {name}")
    print(f"  图像目录: {image_dir}/vestas")
    print(f"  标注目录: {gt_dir}/vestas")
    
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y, dataset_type="vestas")
    )
    
    MetadataCatalog.get(name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        gt_dir=gt_dir,  # 添加gt_dir属性，确保与代码中期望的格式兼容
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="vestas",  # 添加数据集类型标识
        **meta,
    )
    
    # 注册验证集
    image_dir_val = os.path.join(root, "leftImg8bit", "val")
    gt_dir_val = os.path.join(root, "gtFine", "val")
    name_val = "cityscapes_vestas_sem_seg_val"
    
    print(f"注册验证集: {name_val}")
    print(f"  图像目录: {image_dir_val}/vestas")
    print(f"  标注目录: {gt_dir_val}/vestas")
    
    DatasetCatalog.register(
        name_val, lambda x=image_dir_val, y=gt_dir_val: load_cityscapes_semantic(x, y, dataset_type="vestas")
    )
    
    MetadataCatalog.get(name_val).set(
        image_root=image_dir_val,
        sem_seg_root=gt_dir_val,
        gt_dir=gt_dir_val,  
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="vestas",  # 添加数据集类型标识
        **meta,
    )

# 注册所有数据集
def register_all_double_cityscapes(root):
    # 先检查是否已经注册
    from detectron2.data import DatasetCatalog
    
    # 定义要注册的数据集名称
    apollo_datasets = [
        "cityscapes_apollo_sem_seg_train",
        "cityscapes_apollo_sem_seg_val",
        "cityscapes_apollo_sem_seg_test"
    ]
    
    vestas_datasets = [
        "cityscapes_vestas_sem_seg_train",
        "cityscapes_vestas_sem_seg_val"
    ]
    
    # 检查Apollo数据集是否都已注册
    apollo_registered = all(dataset in DatasetCatalog for dataset in apollo_datasets)
    # 检查Vestas数据集是否都已注册
    vestas_registered = all(dataset in DatasetCatalog for dataset in vestas_datasets)
    
    # 如果Apollo数据集未完全注册，则注册
    if not apollo_registered:
        register_cityscapes_apollo(root)
    
    # 如果Vestas数据集未完全注册，则注册
    if not vestas_registered:
        register_cityscapes_vestas(root)

# 注意：不再默认注册数据集，而是由train_net_double_maskformer.py中的setup函数统一调用
# 这样可以避免重复注册数据集的问题
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_double_cityscapes(_root)