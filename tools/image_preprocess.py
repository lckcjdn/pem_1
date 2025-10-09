# collect_left_images.py
import os
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main():
    parser = argparse.ArgumentParser(description="收集 ColorImage/**/Camera 5 下的所有图像到一个扁平目录 leftImg8bit/")
    parser.add_argument("input_root", help="输入根目录，包含 road02, road03 等")
    parser.add_argument("--camera", default="Camera 5", help="要收集的相机文件夹名，默认 Camera 5")
    parser.add_argument("--out_dir", default="leftImg8bit", help="输出目录（单层扁平化）")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 形如：roadXX/ColorImage/Record***/Camera 5/**/*.png
    patterns = [
        input_root.glob(f"road*/ColorImage/*/{args.camera}/**/*"),
        input_root.glob(f"road*/ColorImage/*/{args.camera}/*"),
    ]

    all_imgs = []
    for g in patterns:
        for p in g:
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                all_imgs.append(p)

    # 为避免重名，构造安全的新文件名：roadXX_RecordYYY_Camera5_原文件名
    # 例：road02/ColorImage/Record001/Camera 5/xxx.png -> road02_Record001_Camera5_xxx.png
    pbar = tqdm(all_imgs, desc="拷贝图片到 leftImg8bit")
    for src in pbar:
        parts = src.parts
        # 期望结构: .../roadXX/ColorImage/Record***/Camera 5/filename
        try:
            road = next(x for x in parts if x.startswith("road"))
            rec_idx = parts.index("ColorImage") + 1
            record = parts[rec_idx] if rec_idx < len(parts) else "Record"
        except Exception:
            road, record = "roadXX", "RecordX"

        new_name = f"{road}_{record}_{args.camera.replace(' ', '')}_{src.name}"
        dst = out_dir / new_name
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"完成：共收集 {len(all_imgs)} 张图片到 {out_dir}")

if __name__ == "__main__":
    main()
