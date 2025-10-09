#!/usr/bin/python
#
# Cityscapes labels
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
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
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
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!


# 东软道路感知
# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'background'           ,  0 ,        0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'car'                  ,  1 ,        1 , 'car'             , 1       , False        , False        , (  0, 60, 100) ),
#     Label(  'human'                ,  2 ,        2 , 'human'           , 2       , False        , False        , (220, 20, 60) ),
#     Label(  'road'                 ,  3 ,        3 , 'road'            , 3       , False        , False        , (128, 64, 128) ),
#     Label(  'lane_mark'            ,  4 ,        4 , 'lane_mark'       , 4       , False        , False        , (255, 255, 255) ),
#     Label(  'curb'                 ,  5 ,        5 , 'curb'            , 5       , False        , False        , (196, 196, 196) ),
#     Label(  'road_mark'            ,  6 ,        6 , 'road_mark'       , 6       , False        , False        , (250, 170, 11) ),
#     Label(  'guard_rail'           ,  7 ,        7 , 'guard_rail'      , 7       , False        , False        , (51, 0, 255) ),
#     Label(  'traffic_sign'         ,  8 ,        8 , 'traffic_sign'    , 8       , False        , False        , (220, 220, 0) ),
# ]

labels = [
    #           name          id  trainId      category     catId hasInstances ignoreInEval      color
    Label( 'background' ,     0 ,     0 ,       'void' ,      0 ,      False ,      False , (  0,   0,   0) ),
    Label(     's_w_d' ,      1 ,     1 ,   'dividing' ,      1 ,      False ,      False , ( 70, 130, 180) ),
    Label(     's_y_d' ,      2 ,     2 ,   'dividing' ,      1 ,      False ,      False , (220,  20,  60) ),
    Label(   'ds_y_dn' ,      3 ,     3 ,   'dividing' ,      1 ,      False ,      False , (255,   0,   0) ),
    Label(   'sb_w_do' ,      4 ,     4 ,   'dividing' ,      1 ,      False ,      False , (  0,   0,  60) ),
    Label(   'sb_y_do' ,      5 ,     5 ,   'dividing' ,      1 ,      False ,      False , (  0,  60, 100) ),
    Label(      'b_w_g' ,     6 ,     6 ,    'guiding' ,      2 ,      False ,      False , (  0,   0, 142) ),
    Label(      's_w_s' ,     7 ,     7 ,   'stopping' ,      3 ,      False ,      False , (220, 220,   0) ),
    Label(      's_w_c' ,     8 ,     8 ,    'chevron' ,      4 ,      False ,      False , (102, 102, 156) ),
    Label(      's_y_c' ,     9 ,     9 ,    'chevron' ,      4 ,      False ,      False , (128,   0,   0) ),
    Label(      's_w_p' ,    10 ,    10 ,    'parking' ,      5 ,      False ,      False , (128,  64, 128) ),
    Label(      'c_wy_z' ,   11 ,    11 ,      'zebra' ,      6 ,      False ,      False , (190, 153, 153) ),
    Label(       'a_w_u',    12 ,    12 ,  'thru/turn' ,      7 ,      False ,      False , (  0,   0, 230) ),
    Label(       'a_w_t',    13 ,    13 ,  'thru/turn' ,      7 ,      False ,      False , (128, 128,   0) ),
    Label(      'a_w_tl',    14 ,    14 ,  'thru/turn' ,      7 ,      False ,      False , (128,  78, 160) ),
    Label(      'a_w_tr',    15 ,    15 ,  'thru/turn' ,      7 ,      False ,      False , (150, 100, 100) ),
    Label(       'a_w_l',    16 ,    16 ,  'thru/turn' ,      7 ,      False ,      False , (180, 165, 180) ),
    Label(       'a_w_r',    17 ,    17 ,  'thru/turn' ,      7 ,      False ,      False , (107, 142,  35) ),
    Label(      'a_n_lu',    18 ,    18 ,  'thru/turn' ,      7 ,      False ,      False , (  0, 191, 255) ),
    Label(      'b_n_sr',    19 ,    19 ,  'reduction' ,      8 ,      False ,      False , (255, 128,   0) ),
    Label(     'd_wy_za',    20 ,    20 ,  'attention' ,      9 ,      False ,      False , (  0, 255, 255) ),
    Label(      'r_wy_np',   21 ,    21 , 'no parking' ,     10 ,      False ,      False , (178, 132, 190) ),
    Label(     'vom_wy_n',   22 ,    22 ,     'others' ,     11 ,      False ,      False , (128, 128,  64) ),
    Label(       'om_n_n',   23 ,    23 ,     'others' ,     11 ,      False ,      False , (102,   0, 204) ),
]


# 道路标记细分
# labels = [
#     # name,                   id, trainId, category,            catId, hasInstances, ignoreInEval, color (R,G,B)
#     Label('background',            0,   0, 'background',            0, False, True,  (  0,   0,   0)),
#     Label('crosswalk',             1,   1, 'crosswalk',             1, False, False, ( 34, 117,  76)),
#     Label('stop_line',             2,   2, 'stop_line',             2, False, False, ( 61,  72, 204)),
#     Label('solid_single_white',    3,   3, 'solid_single_white',    3, False, False, (210, 210, 210)),
#     Label('solid_single_yellow',   4,   4, 'solid_single_yellow',   4, False, False, (220, 220,   0)),
#     Label('solid_double_white',    5,   5, 'solid_double_white',    5, False, False, (160, 160, 160)),
#     Label('solid_double_yellow',   6,   6, 'solid_double_yellow',   6, False, False, (180, 180,   0)),
#     Label('dashed_single_white',   7,   7, 'dashed_single_white',   7, False, False, (255, 255, 255)),
#     Label('dashed_single_yellow',  8,   8, 'dashed_single_yellow',  8, False, False, (255, 255,   0)),
#     Label('left_arrow',            9,   9, 'left_arrow',            9, False, False, (0,   255, 255)),
# ]

# labels = [
#     # name,                   id, trainId, category,            catId, hasInstances, ignoreInEval, color (R,G,B)
#     Label('background',            0,   0, 'background',            0, False, True,  (  0,   0,   0)),
#     Label('box_junction',          1,   1, 'box_junction',          1, False, False, (255, 242,   0)),
#     Label('crosswalk',             2,   2, 'crosswalk',             2, False, False, ( 34, 117,  76)),
#     Label('stop_line',             3,   3, 'stop_line',             3, False, False, ( 61,  72, 204)),
#     Label('solid_single_white',    4,   4, 'solid_single_white',    4, False, False, (237,  28,  36)),
#     Label('solid_single_yellow',   5,   5, 'solid_single_yellow',   5, False, False, (163,  73, 164)),
#     Label('solid_single_red',      6,   6, 'solid_single_red',      6, False, False, (185, 122,  87)),
#     Label('solid_double_white',    7,   7, 'solid_double_white',    7, False, False, (136,   0,  21)),
#     Label('solid_double_yellow',   8,   8, 'solid_double_yellow',   8, False, False, (112, 146, 190)),
#     Label('dashed_single_white',   9,   9, 'dashed_single_white',   9, False, False, (181, 230,  29)),
#     Label('dashed_single_yellow', 10,  10, 'dashed_single_yellow', 10, False, False, (153, 217, 234)),
#     Label('left_arrow',           11,  11, 'left_arrow',           11, False, False, (158, 159,  76)),
#     Label('straight_arrow',       12,  12, 'straight_arrow',       12, False, False, (121, 138, 134)),
#     Label('right_arrow',          13,  13, 'right_arrow',          13, False, False, ( 41,  64,  96)),
#     Label('left_straight_arrow',  14,  14, 'left_straight_arrow',  14, False, False, (  7, 102, 146)),
#     Label('right_straight_arrow', 15,  15, 'right_straight_arrow', 15, False, False, (247, 153, 255)),
#     Label('channelizing_line',    16,  16, 'channelizing_line',    16, False, False, (255, 204, 153)),
#     Label('motor_prohibited',     17,  17, 'motor_prohibited',     17, False, False, (155, 255, 153)),
#     Label('slow',                 18,  18, 'slow',                 18, False, False, (255, 153, 173)),
#     Label('motor_priority_lane',  19,  19, 'motor_priority_lane',  19, False, False, (230, 224, 147)),
#     Label('motor_waiting_zone',   20,  20, 'motor_waiting_zone',   20, False, False, ( 35,  27,  87)),
#     Label('left_turn_box',        21,  21, 'left_turn_box',        21, False, False, (193, 158, 155)),
#     Label('motor_icon',           22,  22, 'motor_icon',           22, False, False, (109,  29,  78)),
#     Label('bike_icon',            23,  23, 'bike_icon',            23, False, False, (  3, 164, 204)),
#     Label('parking_lot',          24,  24, 'parking_lot',          24, False, False, (175, 157, 185)),
# ]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
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

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' ))
    print("    " + ('-' * 98))
    for label in labels:
        print("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval ))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print("ID of label '{name}': {id}".format( name=name, id=id ))

    # Map from ID to label
    category = id2label[id].category
    print("Category of label with ID '{id}': {category}".format( id=id, category=category ))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print("Name of label with trainID '{id}': {name}".format( id=trainId, name=name ))
