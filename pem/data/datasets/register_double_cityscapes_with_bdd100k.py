# Copyright (c) Facebook, Inc. and its affiliates.
import os
import glob
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

# 导入双数据集标签映射功能和BDD100k标签定义
from .labels_double import get_dataset_meta
from .labels_bdd100k import get_dataset_meta as get_bdd100k_dataset_meta

# 加载cityscapes格式的语义分割数据集
# 增强版支持多种数据集类型，包括apollo、vestas和bdd100k
def load_cityscapes_semantic(image_dir, gt_dir, dataset_type=None):
    """
    Args:
        image_dir (str): 图像目录路径，例如 "~/cityscapes/leftImg8bit/train".
        gt_dir (str): 标注目录路径，例如 "~/cityscapes/gtFine/train".
        dataset_type (str): 数据集类型，'apo'、'vestas'、'bdd100k'或'bdd100k_0'，用于确定子目录

    Returns:
        list[dict]: Detectron2标准格式的数据集字典列表
    """
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"图像目录不存在: {image_dir}")
    if not os.path.exists(gt_dir):
        raise FileNotFoundError(f"标注目录不存在: {gt_dir}")
    
    # 根据dataset_type确定子目录和数据集ID
    if dataset_type == "apo":
        subdir = "apo"
        dataset_id = 0  # Apollo数据集ID
    elif dataset_type == "vestas":
        subdir = "vestas"
        dataset_id = 1  # Vestas数据集ID
    elif dataset_type == "bdd100k":
        subdir = "bdd100k"
        dataset_id = 2  # BDD100k数据集ID
    elif dataset_type == "bdd100k_0":
        subdir = "bdd100k_0"
        dataset_id = 3  # BDD100k_0数据集ID
    else:
        # 如果未指定dataset_type，尝试从image_dir推断
        image_dir_lower = image_dir.lower()
        if "vestas" in image_dir_lower:
            subdir = "vestas"
            dataset_id = 1
        elif "bdd100k_0" in image_dir_lower:
            subdir = "bdd100k_0"
            dataset_id = 3
        elif "bdd100k" in image_dir_lower:
            subdir = "bdd100k"
            dataset_id = 2
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
    
    # 获取所有图像文件 (支持JPG和PNG格式)
    image_files = glob.glob(os.path.join(full_image_dir, "*.jpg")) + glob.glob(os.path.join(full_image_dir, "*.png"))
    
    if not image_files:
        raise RuntimeError(f"在目录 {full_image_dir} 中未找到任何JPG或PNG图像文件")
    
    dataset_dicts = []
    
    for image_path in image_files:
        # 获取图像文件名（不包含扩展名）
        img_basename = os.path.splitext(os.path.basename(image_path))[0]
        
        # 构建对应的标注文件路径
        # 尝试原始文件名+.png
        anno_path = os.path.join(full_gt_dir, img_basename + ".png")
        
        # 如果原始命名不存在，尝试_cityscapes_gtFine_labelIds.png命名格式
        if not os.path.exists(anno_path):
            anno_path_alt = os.path.join(full_gt_dir, img_basename + "_gtFine_labelIds.png")
            if os.path.exists(anno_path_alt):
                anno_path = anno_path_alt
        
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
                    "dataset_id": dataset_id,  # 添加数据集标识
                    "dataset_type": subdir,  # 添加数据集类型信息
                })
            except Exception as e:
                # 静默跳过无法处理的图像，不打印警告
                continue  # 跳过无法处理的图像
        # 移除警告信息，静默跳过没有标注文件的图像
    
    if len(dataset_dicts) == 0:
        raise RuntimeError(f"没有找到任何匹配的图像-标注对！请检查图像和标注文件的路径和命名。")
    
    print(f"成功加载 {len(dataset_dicts)} 个数据样本 (数据集类型: {dataset_type})")
    return dataset_dicts

# 获取数据集元数据 - 根据数据集类型返回不同的元数据
def _get_dataset_meta(dataset_name):
    """
    获取指定数据集的元数据
    
    Args:
        dataset_name (str): 数据集名称，用于确定使用哪种标签映射
    
    Returns:
        dict: 数据集元数据
    """
    if "bdd100k_0" in dataset_name:
        # 对于bdd100k_0数据集，使用bdd100k_0的标签映射
        return get_bdd100k_dataset_meta("bdd100k_0")
    elif "bdd100k" in dataset_name:
        # 对于bdd100k数据集，使用bdd100k的标签映射
        return get_bdd100k_dataset_meta("bdd100k")
    else:
        # 对于其他数据集，使用原始的标签映射
        return get_dataset_meta(dataset_name)

# 注册第一个cityscapes数据集（Apollo数据集）
def register_cityscapes_apollo(root):
    root = os.path.join(root, "cityscapes")
    dataset_name = "cityscapes_apollo"
    meta = _get_dataset_meta(dataset_name)
    
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
    meta = _get_dataset_meta(dataset_name)
    
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
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="vestas",  # 添加数据集类型标识
        **meta,
    )

# 注册BDD100k数据集
def register_cityscapes_bdd100k(root):
    root = os.path.join(root, "cityscapes")
    dataset_name = "cityscapes_bdd100k"
    meta = _get_dataset_meta(dataset_name)
    
    # 打印注册信息
    print(f"\n注册BDD100k数据集:")
    print(f"数据集根目录: {root}")
    
    # 确保bdd100k子目录存在
    os.makedirs(os.path.join(root, "leftImg8bit", "train", "bdd100k"), exist_ok=True)
    os.makedirs(os.path.join(root, "gtFine", "train", "bdd100k"), exist_ok=True)
    os.makedirs(os.path.join(root, "leftImg8bit", "val", "bdd100k"), exist_ok=True)
    os.makedirs(os.path.join(root, "gtFine", "val", "bdd100k"), exist_ok=True)
    
    # 注册训练集 - 使用bdd100k子目录
    image_dir = os.path.join(root, "leftImg8bit", "train")
    gt_dir = os.path.join(root, "gtFine", "train")
    name = "cityscapes_bdd100k_sem_seg_train"
    
    print(f"注册训练集: {name}")
    print(f"  图像目录: {image_dir}/bdd100k")
    print(f"  标注目录: {gt_dir}/bdd100k")
    
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y, dataset_type="bdd100k")
    )
    
    MetadataCatalog.get(name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="bdd100k",  # 添加数据集类型标识
        **meta,
    )
    
    # 注册验证集
    image_dir_val = os.path.join(root, "leftImg8bit", "val")
    gt_dir_val = os.path.join(root, "gtFine", "val")
    name_val = "cityscapes_bdd100k_sem_seg_val"
    
    print(f"注册验证集: {name_val}")
    print(f"  图像目录: {image_dir_val}/bdd100k")
    print(f"  标注目录: {gt_dir_val}/bdd100k")
    
    DatasetCatalog.register(
        name_val, lambda x=image_dir_val, y=gt_dir_val: load_cityscapes_semantic(x, y, dataset_type="bdd100k")
    )
    
    MetadataCatalog.get(name_val).set(
        image_root=image_dir_val,
        sem_seg_root=gt_dir_val,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="bdd100k",  # 添加数据集类型标识
        **meta,
    )

# 注册BDD100k_0数据集
def register_cityscapes_bdd100k_0(root):
    root = os.path.join(root, "cityscapes")
    dataset_name = "cityscapes_bdd100k_0"
    meta = _get_dataset_meta(dataset_name)
    
    # 打印注册信息
    print(f"\n注册BDD100k_0数据集:")
    print(f"数据集根目录: {root}")
    
    # 确保bdd100k_0子目录存在
    os.makedirs(os.path.join(root, "leftImg8bit", "train", "bdd100k_0"), exist_ok=True)
    os.makedirs(os.path.join(root, "gtFine", "train", "bdd100k_0"), exist_ok=True)
    os.makedirs(os.path.join(root, "leftImg8bit", "val", "bdd100k_0"), exist_ok=True)
    os.makedirs(os.path.join(root, "gtFine", "val", "bdd100k_0"), exist_ok=True)
    
    # 注册训练集 - 使用bdd100k_0子目录
    image_dir = os.path.join(root, "leftImg8bit", "train")
    gt_dir = os.path.join(root, "gtFine", "train")
    name = "cityscapes_bdd100k_0_sem_seg_train"
    
    print(f"注册训练集: {name}")
    print(f"  图像目录: {image_dir}/bdd100k_0")
    print(f"  标注目录: {gt_dir}/bdd100k_0")
    
    DatasetCatalog.register(
        name, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y, dataset_type="bdd100k_0")
    )
    
    MetadataCatalog.get(name).set(
        image_root=image_dir,
        sem_seg_root=gt_dir,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="bdd100k_0",  # 添加数据集类型标识
        **meta,
    )
    
    # 注册验证集
    image_dir_val = os.path.join(root, "leftImg8bit", "val")
    gt_dir_val = os.path.join(root, "gtFine", "val")
    name_val = "cityscapes_bdd100k_0_sem_seg_val"
    
    print(f"注册验证集: {name_val}")
    print(f"  图像目录: {image_dir_val}/bdd100k_0")
    print(f"  标注目录: {gt_dir_val}/bdd100k_0")
    
    DatasetCatalog.register(
        name_val, lambda x=image_dir_val, y=gt_dir_val: load_cityscapes_semantic(x, y, dataset_type="bdd100k_0")
    )
    
    MetadataCatalog.get(name_val).set(
        image_root=image_dir_val,
        sem_seg_root=gt_dir_val,
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="bdd100k_0",  # 添加数据集类型标识
        **meta,
    )

# 注册混合数据集 (bdd100k_0 + apo)
def register_cityscapes_bdd100k_0_apo_mixed(root):
    root = os.path.join(root, "cityscapes")
    
    # 打印注册信息
    print(f"\n注册混合数据集 (BDD100k_0 + Apollo):")
    print(f"数据集根目录: {root}")
    
    # 定义混合数据集加载函数
    def load_mixed_dataset():
        # 加载bdd100k_0数据集
        bdd100k_0_image_dir = os.path.join(root, "leftImg8bit", "train")
        bdd100k_0_gt_dir = os.path.join(root, "gtFine", "train")
        bdd100k_0_dicts = load_cityscapes_semantic(bdd100k_0_image_dir, bdd100k_0_gt_dir, dataset_type="bdd100k_0")
        
        # 加载apollo数据集
        apo_image_dir = os.path.join(root, "leftImg8bit", "train")
        apo_gt_dir = os.path.join(root, "gtFine", "train")
        apo_dicts = load_cityscapes_semantic(apo_image_dir, apo_gt_dir, dataset_type="apo")
        
        # 合并数据集并添加混合标识
        mixed_dicts = bdd100k_0_dicts + apo_dicts
        print(f"成功合并BDD100k_0 ({len(bdd100k_0_dicts)}) 和 Apollo ({len(apo_dicts)}) 数据集，总计 {len(mixed_dicts)} 个样本")
        
        return mixed_dicts
    
    # 注册训练集
    name = "cityscapes_bdd100k_0_apo_sem_seg_train"
    print(f"注册训练集: {name}")
    
    DatasetCatalog.register(name, load_mixed_dataset)
    
    # 使用bdd100k_0的元数据，因为我们主要关注这个数据集的标签
    meta = _get_dataset_meta("cityscapes_bdd100k_0")
    
    MetadataCatalog.get(name).set(
        image_root=os.path.join(root, "leftImg8bit", "train"),
        sem_seg_root=os.path.join(root, "gtFine", "train"),
        evaluator_type="cityscapes_sem_seg",
        ignore_label=255,
        dataset_type="mixed",  # 添加混合数据集标识
        **meta,
    )

# 注册所有数据集（包括原始数据集和新增的BDD100k数据集）
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
    
    bdd100k_datasets = [
        "cityscapes_bdd100k_sem_seg_train",
        "cityscapes_bdd100k_sem_seg_val"
    ]
    
    bdd100k_0_datasets = [
        "cityscapes_bdd100k_0_sem_seg_train",
        "cityscapes_bdd100k_0_sem_seg_val"
    ]
    
    mixed_datasets = [
        "cityscapes_bdd100k_0_apo_sem_seg_train"
    ]
    
    # 检查Apollo数据集是否都已注册
    apollo_registered = all(dataset in DatasetCatalog for dataset in apollo_datasets)
    # 检查Vestas数据集是否都已注册
    vestas_registered = all(dataset in DatasetCatalog for dataset in vestas_datasets)
    # 检查BDD100k数据集是否都已注册
    bdd100k_registered = all(dataset in DatasetCatalog for dataset in bdd100k_datasets)
    # 检查BDD100k_0数据集是否都已注册
    bdd100k_0_registered = all(dataset in DatasetCatalog for dataset in bdd100k_0_datasets)
    # 检查混合数据集是否都已注册
    mixed_registered = all(dataset in DatasetCatalog for dataset in mixed_datasets)
    
    # 如果Apollo数据集未完全注册，则注册
    if not apollo_registered:
        register_cityscapes_apollo(root)
    
    # 如果Vestas数据集未完全注册，则注册
    if not vestas_registered:
        register_cityscapes_vestas(root)
    
    # 如果BDD100k数据集未完全注册，则注册
    if not bdd100k_registered:
        register_cityscapes_bdd100k(root)
    
    # 如果BDD100k_0数据集未完全注册，则注册
    if not bdd100k_0_registered:
        register_cityscapes_bdd100k_0(root)
    
    # 如果混合数据集未完全注册，则注册
    if not mixed_registered:
        register_cityscapes_bdd100k_0_apo_mixed(root)

# 注意：不再默认注册数据集，而是由train_net_double_maskformer.py中的setup函数统一调用
# 这样可以避免重复注册数据集的问题
# _root = os.getenv("DETECTRON2_DATASETS", "datasets")
# register_all_double_cityscapes(_root)