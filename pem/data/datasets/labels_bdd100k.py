#!/usr/bin/python
#
# BDD100k Dataset Labels
#

from __future__ import print_function, absolute_import, division
from collections import namedtuple

#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).

    'trainId'     , # Feel free to modify these IDs as suitable for your method.
                    # For trainIds, multiple labels might have the same ID.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

#--------------------------------------------------------------------------------
# BDD100k Dataset Labels
#--------------------------------------------------------------------------------
bdd100k_labels = [
    #           name                     id  trainId      category      catId hasInstances ignoreInEval      color
    Label(  'background'            ,   0 ,     0 ,        'void' ,       0 ,      False ,      True , (  0,   0,   0) ),
    Label(  'crosswalk_dashed'      ,   1 ,     1 ,   'crosswalk' ,       1 ,      False ,      False , (255, 255, 255) ),
    Label(  'crosswalk_solid'       ,   2 ,     2 ,   'crosswalk' ,       1 ,      False ,      False , (200, 200, 200) ),
    Label(  'double_other_solid'    ,   3 ,     3 ,    'dividing' ,       2 ,      False ,      False , (150, 150, 150) ),
    Label(  'double_other_dashed'   ,   4 ,     4 ,    'dividing' ,       2 ,      False ,      False , (120, 120, 120) ),
    Label(  'double_white_dashed'   ,   5 ,     5 ,    'dividing' ,       2 ,      False ,      False , (220, 220,   0) ),
    Label(  'double_white_solid'    ,   6 ,     6 ,    'dividing' ,       2 ,      False ,      False , (255, 255,   0) ),
    Label(  'double_yellow_dashed'  ,   7 ,     7 ,    'dividing' ,       2 ,      False ,      False , (255, 128,   0) ),
    Label(  'double_yellow_solid'   ,   8 ,     8 ,    'dividing' ,       2 ,      False ,      False , (255,   0,   0) ),
    Label(  'road_curb_dashed'      ,   9 ,     9 ,  'road_curb' ,       3 ,      False ,      False , (100, 100, 100) ),
    Label(  'road_curb_solid'       ,  10 ,    10 ,  'road_curb' ,       3 ,      False ,      False , ( 50,  50,  50) ),
    Label(  'single_other_dashed'   ,  11 ,    11 ,    'dividing' ,       2 ,      False ,      False , (100, 100, 150) ),
    Label(  'single_other_solid'    ,  12 ,    12 ,    'dividing' ,       2 ,      False ,      False , ( 50,  50, 100) ),
    Label(  'single_white_dashed'   ,  13 ,    13 ,    'dividing' ,       2 ,      False ,      False , (150, 200, 150) ),
    Label(  'single_white_solid'    ,  14 ,    14 ,    'dividing' ,       2 ,      False ,      False , (100, 255, 100) ),
    Label(  'single_yellow_dashed'  ,  15 ,    15 ,    'dividing' ,       2 ,      False ,      False , (255, 150, 100) ),
    Label(  'single_yellow_solid'   ,  16 ,    16 ,    'dividing' ,       2 ,      False ,      False , (255, 100,  50) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

def create_label_mappings(labels_list):
    """为给定的标签列表创建映射字典"""
    # name to label object
    name2label = { label.name: label for label in labels_list }
    # id to label object
    id2label = { label.id: label for label in labels_list }
    # trainId to label object
    trainId2label = { label.trainId: label for label in reversed(labels_list) }
    # category to list of label objects
    category2labels = {}
    for label in labels_list:
        category = label.category
        if category in category2labels:
            category2labels[category].append(label)
        else:
            category2labels[category] = [label]
    
    return name2label, id2label, trainId2label, category2labels

# BDD100k标签映射
bdd100k_name2label, bdd100k_id2label, bdd100k_trainId2label, bdd100k_category2labels = create_label_mappings(bdd100k_labels)

# 获取数据集对应的标签映射
def get_label_mappings(dataset_name):
    """根据数据集名称获取对应的标签映射"""
    if "bdd100k" in dataset_name.lower():
        return {
            'labels': bdd100k_labels,
            'name2label': bdd100k_name2label,
            'id2label': bdd100k_id2label,
            'trainId2label': bdd100k_trainId2label,
            'category2labels': bdd100k_category2labels
        }
    else:  # 默认使用BDD100k标签
        return {
            'labels': bdd100k_labels,
            'name2label': bdd100k_name2label,
            'id2label': bdd100k_id2label,
            'trainId2label': bdd100k_trainId2label,
            'category2labels': bdd100k_category2labels
        }

# 获取数据集元数据
def get_dataset_meta(dataset_name):
    """获取数据集的元数据信息"""
    mappings = get_label_mappings(dataset_name)
    labels = mappings['labels']
    
    # 计算类别信息
    thing_classes = [l.name for l in labels if l.hasInstances]
    stuff_classes = [l.name for l in labels if not l.hasInstances]
    
    # 计算id到trainId的映射
    id_to_train_id = {label.id: label.trainId for label in labels}
    train_id_to_color = {label.trainId: label.color for label in labels}
    
    return {
        "thing_classes": thing_classes,
        "stuff_classes": stuff_classes,
        "id_to_train_id": id_to_train_id,
        "train_id_to_color": train_id_to_color,
        "thing_dataset_id_to_contiguous_id": {l.id: idx for idx, l in enumerate(labels) if l.hasInstances},
        "stuff_dataset_id_to_contiguous_id": {l.id: idx for idx, l in enumerate(labels) if not l.hasInstances}
    }

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

def assureSingleInstanceName(name, dataset_name="bdd100k"):
    """返回描述单个实例的标签名称"""
    mappings = get_label_mappings(dataset_name)
    name2label = mappings['name2label']
    
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name