# remap_apollo_labels.py
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse

LABELS = [
    ('void',      0,  (  0,   0,   0)),
    ('s_w_d',   200,  ( 70, 130, 180)),
    ('s_y_d',   204,  (220,  20,  60)),
    ('ds_w_dn', 213,  (128,   0, 128)),
    ('ds_y_dn', 209,  (255,   0,   0)),
    ('sb_w_do', 206,  (  0,   0,  60)),
    ('sb_y_do', 207,  (  0,  60, 100)),
    ('b_w_g',   201,  (  0,   0, 142)),
    ('b_y_g',   203,  (119,  11,  32)),
    ('db_w_g',  211,  (244,  35, 232)),
    ('db_y_g',  208,  (  0,   0, 160)),
    ('db_w_s',  216,  (153, 153, 153)),
    ('s_w_s',   217,  (220, 220,   0)),
    ('ds_w_s',  215,  (250, 170,  30)),
    ('s_w_c',   218,  (102, 102, 156)),
    ('s_y_c',   219,  (128,   0,   0)),
    ('s_w_p',   210,  (128,  64, 128)),
    ('s_n_p',   232,  (238, 232, 170)),
    ('c_wy_z',  214,  (190, 153, 153)),
    ('a_w_u',   202,  (  0,   0, 230)),
    ('a_w_t',   220,  (128, 128,   0)),
    ('a_w_tl',  221,  (128,  78, 160)),
    ('a_w_tr',  222,  (150, 100, 100)),
    ('a_w_tlr', 231,  (255, 165,   0)),
    ('a_w_l',   224,  (180, 165, 180)),
    ('a_w_r',   225,  (107, 142,  35)),
    ('a_w_lr',  226,  (201, 255, 229)),
    ('a_n_lu',  230,  (  0, 191, 255)),
    ('a_w_tu',  228,  ( 51, 255,  51)),
    ('a_w_m',   229,  (250, 128, 114)),
    ('a_y_t',   233,  (127, 255,   0)),
    ('b_n_sr',  205,  (255, 128,   0)),
    ('d_wy_za', 212,  (  0, 255, 255)),
    ('r_wy_np', 227,  (178, 132, 190)),
    ('vom_wy_n',223,  (128, 128,  64)),
    ('om_n_n',  250,  (102,   0, 204)),
    ('noise',   249,  (  0, 153, 153)),
    ('ignored', 255,  (255, 255, 255)),
]

# ====== 并入背景 0 的类别======
MERGE_TO_BG = {
    'db_w_g', 'db_w_s', 'ds_w_s', 's_n_p', 'a_w_tlr', 'a_w_tu',
    'a_w_m', 'a_y_t', 'noise', 'ignored', 'void'
}

BG_COLOR = (0, 0, 0)

def build_remap_tables():
    """
    返回：
      - color_to_newid: {(R,G,B)->new_id}
      - color_to_color: {(R,G,B)->(R,G,B)}  # 输出彩色图用
      - name_to_newid:  {name->new_id}
    规则：MERGE_TO_BG → 0，其余类别按原列表顺序，依次编号 1,2,3,...
    彩色图对非背景类别使用“自带颜色”，背景用黑色。
    """
    name_to_newid = {}
    next_id = 1
    for name, _, _ in LABELS:
        if name in MERGE_TO_BG:
            name_to_newid[name] = 0
        else:
            name_to_newid[name] = next_id
            next_id += 1

    color_to_newid = {}
    color_to_color = {}
    for name, _, color in LABELS:
        r, g, b = color
        if name_to_newid[name] == 0:
            color_to_newid[(r, g, b)] = 0
            color_to_color[(r, g, b)] = BG_COLOR
        else:
            color_to_newid[(r, g, b)] = name_to_newid[name]
            color_to_color[(r, g, b)] = (r, g, b)

    return color_to_newid, color_to_color, name_to_newid

def process_one_image(args):
    in_path, gray_out_path, color_out_path, color_to_newid, color_to_color = args
    img_bgr = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] 无法读取：{in_path}")
        return False
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    gray_img = np.zeros((h, w), dtype=np.uint8)
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    uniq = np.unique(img.reshape(-1, 3), axis=0)
    for col in uniq:
        r, g, b = map(int, col.tolist())
        new_id = color_to_newid.get((r, g, b), 0)
        out_col = color_to_color.get((r, g, b), BG_COLOR)
        mask = np.all(img == col, axis=2)
        gray_img[mask] = new_id
        color_img[mask] = out_col

    gray_out_path.parent.mkdir(parents=True, exist_ok=True)
    color_out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(gray_out_path), gray_img)
    cv2.imwrite(str(color_out_path), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Apollo 车道线标签新映射：将指定类别并入背景 0，其余从 1 起重新编号；输出灰度ID图与彩色标注图，目录结构与 Label 相同。"
    )
    parser.add_argument("input_root", help="输入根目录，包含 road02/road03 等，每个内有 Label/... 彩色标注图")
    parser.add_argument("--gray_out_root", default="LabelIds", help="灰度ID图输出根目录（镜像组织）")
    parser.add_argument("--color_out_root", default="LabelColor", help="彩色标注图输出根目录（镜像组织）")
    args = parser.parse_args()

    color_to_newid, color_to_color, name_to_newid = build_remap_tables()

    input_root = Path(args.input_root)
    gray_root = Path(args.gray_out_root)
    color_root = Path(args.color_out_root)

    # 遍历所有 road*/Label/** 下的图片
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    candidates = []
    for road_dir in input_root.glob("road*"):
        label_dir = road_dir / "Label"
        if not label_dir.is_dir():
            continue
        for p in label_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in image_exts:
                # 构造相对路径：保持 roadXX/Label/RecordXXX/Camera Y/xxx.png
                rel = p.relative_to(input_root)
                try:
                    idx = list(rel.parts).index("Label")
                    tail = Path(*rel.parts[idx+1:])  # RecordXXX/Camera Y/xxx.png
                except ValueError:
                    tail = rel  

                gray_out = gray_root / rel.parts[0] / tail  # roadXX/...
                color_out = color_root / rel.parts[0] / tail
                candidates.append((p, gray_out, color_out))

    print(f"发现待处理标注图片：{len(candidates)}")

    num_workers = max(1, multiprocessing.cpu_count())
    tasks = [
        (in_path, gray_out.with_suffix(".png"), color_out.with_suffix(".png"),
         color_to_newid, color_to_color)
        for (in_path, gray_out, color_out) in candidates
    ]
    ok = 0
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        for res in tqdm(ex.map(process_one_image, tasks), total=len(tasks), desc="重映射 & 渲染"):
            ok += int(bool(res))
    print(f"完成：成功处理 {ok}/{len(tasks)} 张。")

    print("新ID分配（并入背景=0，其余从1递增）：")
    for name, _, _ in LABELS:
        print(f"{name:10s} -> {name_to_newid[name]}")

if __name__ == "__main__":
    main()
    
# labels = [
#     #           name          id  trainId      category     catId hasInstances ignoreInEval               color
#     Label( 'background' ,     0 ,     0 ,        'void' ,      0 ,      False ,      False , (  0,   0,   0) ),

#     Label(     's_w_d' ,      1 ,     1 ,   'dividing' ,      1 ,      False ,      False , ( 70, 130, 180) ),
#     Label(     's_y_d' ,      2 ,     2 ,   'dividing' ,      1 ,      False ,      False , (220,  20,  60) ),
#     Label(   'ds_w_dn' ,      3 ,     3 ,   'dividing' ,      1 ,      False ,      False , (128,   0, 128) ),
#     Label(   'ds_y_dn' ,      4 ,     4 ,   'dividing' ,      1 ,      False ,      False , (255,   0,   0) ),
#     Label(   'sb_w_do' ,      5 ,     5 ,   'dividing' ,      1 ,      False ,      False , (  0,   0,  60) ),
#     Label(   'sb_y_do' ,      6 ,     6 ,   'dividing' ,      1 ,      False ,      False , (  0,  60, 100) ),

#     Label(      'b_w_g' ,     7 ,     7 ,    'guiding' ,      2 ,      False ,      False , (  0,   0, 142) ),
#     Label(      'b_y_g' ,     8 ,     8 ,    'guiding' ,      2 ,      False ,      False , (119,  11,  32) ),
#     Label(     'db_y_g' ,     9 ,     9 ,    'guiding' ,      2 ,      False ,      False , (  0,   0, 160) ),

#     Label(      's_w_s' ,    10 ,    10 ,   'stopping' ,      3 ,      False ,      False , (220, 220,   0) ),

#     Label(      's_w_c' ,    11 ,    11 ,    'chevron' ,      4 ,      False ,      False , (102, 102, 156) ),
#     Label(      's_y_c' ,    12 ,    12 ,    'chevron' ,      4 ,      False ,      False , (128,   0,   0) ),

#     Label(      's_w_p' ,    13 ,    13 ,    'parking' ,      5 ,      False ,      False , (128,  64, 128) ),
#     Label(      'c_wy_z' ,   14 ,    14 ,      'zebra' ,      6 ,      False ,      False , (190, 153, 153) ),

#     Label(       'a_w_u',    15 ,    15 ,  'thru/turn' ,      7 ,      False ,      False , (  0,   0, 230) ),
#     Label(       'a_w_t',    16 ,    16 ,  'thru/turn' ,      7 ,      False ,      False , (128, 128,   0) ),
#     Label(      'a_w_tl',    17 ,    17 ,  'thru/turn' ,      7 ,      False ,      False , (128,  78, 160) ),
#     Label(      'a_w_tr',    18 ,    18 ,  'thru/turn' ,      7 ,      False ,      False , (150, 100, 100) ),
#     Label(       'a_w_l',    19 ,    19 ,  'thru/turn' ,      7 ,      False ,      False , (180, 165, 180) ),
#     Label(       'a_w_r',    20 ,    20 ,  'thru/turn' ,      7 ,      False ,      False , (107, 142,  35) ),
#     Label(      'a_w_lr',    21 ,    21 ,  'thru/turn' ,      7 ,      False ,      False , (201, 255, 229) ),
#     Label(      'a_n_lu',    22 ,    22 ,  'thru/turn' ,      7 ,      False ,      False , (  0, 191, 255) ),

#     Label(      'b_n_sr',    23 ,    23 ,  'reduction' ,      8 ,      False ,      False , (255, 128,   0) ),
#     Label(     'd_wy_za',    24 ,    24 ,  'attention' ,      9 ,      False ,      False , (  0, 255, 255) ),
#     Label(      'r_wy_np',   25 ,    25 , 'no parking' ,     10 ,      False ,      False , (178, 132, 190) ),
#     Label(     'vom_wy_n',   26 ,    26 ,     'others' ,     11 ,      False ,      False , (128, 128,  64) ),
#     Label(       'om_n_n',   27 ,    27 ,     'others' ,     11 ,      False ,      False , (102,   0, 204) ),
# ]

