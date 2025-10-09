import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# =============== 明确映射表 ===============
# 假设图像里只有这些 id (0~27, 但没有 3,8,9,21)
# 直接列出对应关系：旧id -> 新id
MAPPING = {
     0: 0,
     1: 1,
     2: 2,
     4: 3,
     5: 4,
     6: 5,
     7: 6,
    10: 7,
    11: 8,
    12: 9,
    13: 10,
    14: 11,
    15: 12,
    16: 13,
    17: 14,
    18: 15,
    19: 16,
    20: 17,
    22: 18,
    23: 19,
    24: 20,
    25: 21,
    26: 22,
    27: 23,
}
# ==========================================

def build_lut(mapping, max_val=255):
    lut = np.arange(max_val + 1, dtype=np.uint8)
    for k, v in mapping.items():
        lut[k] = v
    return lut

def process_one(in_path: Path, out_path: Path, lut: np.ndarray):
    try:
        img = Image.open(in_path).convert("L")
        arr = np.array(img, dtype=np.uint8)
        remapped = lut[arr]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(remapped, mode="L").save(out_path)
        return True, in_path
    except Exception as e:
        return False, (in_path, str(e))

def main():
    parser = argparse.ArgumentParser(
        description="根据映射表(MAPPING)修改灰度标注图"
    )
    parser.add_argument("input_dir", help="输入灰度标注图文件夹")
    parser.add_argument("--out_dir", default=None, help="输出目录（默认：*_remapped）")
    parser.add_argument("--workers", type=int, default=820, help="并行线程数")
    parser.add_argument("--overwrite", action="store_true", help="覆盖原图")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在：{in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(str(in_dir) + "_remapped")
    if args.overwrite:
        out_dir = in_dir

    lut = build_lut(MAPPING)

    img_paths = []
    for root, _, files in os.walk(in_dir):
        for fn in files:
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                img_paths.append(Path(root) / fn)
    if not img_paths:
        raise SystemExit("未找到灰度图片。")

    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for p in img_paths:
            rel = p.relative_to(in_dir)
            out_p = out_dir / rel
            futures.append(ex.submit(process_one, p, out_p, lut))

        ok = 0
        errors = []
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Remapping"):
            success, info = fut.result()
            if success:
                ok += 1
            else:
                errors.append(info)

    print("完成：", ok, "/", len(img_paths))
    if errors:
        print("失败的文件：")
        for e in errors:
            print(" -", e)

if __name__ == "__main__":
    main()
