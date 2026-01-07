#!/usr/bin/python
#
# Double Cityscapes labels (Apollo and Vestas)
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
# Apollo Dataset Labels
#--------------------------------------------------------------------------------
apollo_labels = [
    #           name          id  trainId      category     catId hasInstances ignoreInEval      color
    Label( 'background' ,     0 ,     0 ,       'void' ,      0 ,      False ,      True , (  0,   0,   0) ),
    Label(     's_w_d' ,      1 ,     1 ,   'dividing' ,      1 ,      False ,      False , ( 70, 130, 180) ),
    Label(     's_y_d' ,      2 ,     2 ,   'dividing' ,      1 ,      False ,      False , (220,  20,  60) ),
    Label(   'ds_y_dn' ,      3 ,     3 ,   'dividing' ,      1 ,      False ,      False , (255,   0,   0) ),
    Label(   'sb_w_do' ,      4 ,     4 ,   'dividing' ,      1 ,      False ,      False , (  0,   0,  60) ),
    Label(   'sb_y_do' ,      5 ,     5 ,   'dividing' ,      1 ,      False ,      False , (  0,  60, 100) ),
    Label(      'b_w_g' ,     6 ,     6 ,    'guiding' ,      2 ,      False ,      False , (  0,   0, 142) ),
    Label(      's_w_s' ,     7 ,     7 ,   'stopping' ,      3 ,      False ,      False , (220, 220,   0) ),
    Label(      's_*_c' ,     8 ,     8 ,    'chevron' ,      4 ,      False ,      False , (102, 102, 156) ),
    Label(      's_w_p' ,     9 ,     9 ,    'parking' ,      5 ,      False ,      False , (128,  64, 128) ),
    Label(      'c_wy_z' ,   10 ,    10 ,      'zebra' ,      6 ,      False ,      False , (190, 153, 153) ),
    Label(       'a_w_*',    11 ,    11 ,  'thru/turn' ,      7 ,      False ,      False , (  0,   0, 230) ),
    Label(      'b_n_sr',    12 ,    12 ,  'reduction' ,      8 ,      False ,      False , (255, 128,   0) ),
    Label(     'd_wy_za',    13 ,    13 ,  'attention' ,      9 ,      False ,      False , (  0, 255, 255) ),
    Label(      'r_wy_np',   14 ,    14 , 'no parking' ,     10 ,      False ,      False , (178, 132, 190) ),
    Label(     'vom_*_n',   15 ,    15 ,     'others' ,     11 ,      False ,      False , (128, 128,  64) )
]

#--------------------------------------------------------------------------------
# Vestas Dataset Labels
#--------------------------------------------------------------------------------
# 这里假设Vestas数据集的标签结构与Apollo类似但可能有不同的类别映射
# 实际使用时请根据Vestas数据集的真实标签定义进行修改
vestas_labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'background'           ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'car'                  ,  1 ,        1 , 'car'             , 1       , False        , False        , (  0, 60, 100) ),
    Label(  'human'                ,  2 ,        2 , 'human'           , 2       , False        , False        , (220, 20, 60) ),
    Label(  'road'                 ,  3 ,        3 , 'road'            , 3       , False        , False        , (128, 64, 128) ),
    Label(  'lane_mark'            ,  4 ,        4 , 'lane_mark'       , 4       , False        , False        , (255, 255, 255) ),
    Label(  'curb'                 ,  5 ,        5 , 'curb'            , 5       , False        , False        , (196, 196, 196) ),
    Label(  'road_mark'            ,  6 ,        6 , 'road_mark'       , 6       , False        , False        , (250, 170, 11) ),
    Label(  'guard_rail'           ,  7 ,        7 , 'guard_rail'      , 7       , False        , False        , (51, 0, 255) ),
    Label(  'traffic_sign'         ,  8 ,        8 , 'traffic_sign'    , 8       , False        , False        , (220, 220, 0) ),
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

# Apollo标签映射
apollo_name2label, apollo_id2label, apollo_trainId2label, apollo_category2labels = create_label_mappings(apollo_labels)

# Vestas标签映射
vestas_name2label, vestas_id2label, vestas_trainId2label, vestas_category2labels = create_label_mappings(vestas_labels)

# 获取数据集对应的标签映射
def get_label_mappings(dataset_name):
    """根据数据集名称获取对应的标签映射"""
    if "vestas" in dataset_name.lower():
        return {
            'labels': vestas_labels,
            'name2label': vestas_name2label,
            'id2label': vestas_id2label,
            'trainId2label': vestas_trainId2label,
            'category2labels': vestas_category2labels
        }
    else:  # 默认使用Apollo标签
        return {
            'labels': apollo_labels,
            'name2label': apollo_name2label,
            'id2label': apollo_id2label,
            'trainId2label': apollo_trainId2label,
            'category2labels': apollo_category2labels
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

def assureSingleInstanceName(name, dataset_name="apollo"):
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