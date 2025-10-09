#!/usr/bin/env bash
# 把下面这行里的路径换成你的图片目录；留空表示当前目录
IMG_DIR="/home/guitu/Data/ytj/data/rmld/labels/val/"

cd "$IMG_DIR" || exit 1            # 进入目录，失败就退出
shopt -s nullglob                   # 若没有匹配文件，不报错

# 按常见扩展名循环；需要其他格式自行添加
for f in *.jpg *.jpeg *.png *.bmp *.tif *.tiff; do
    [ -e "$f" ] || continue        # 若目录里没有此类文件，跳过
    basename="${f%.*}"             # 去掉扩展名
    ext="${f##*.}"                 # 取扩展名
    len=${#basename}
    if (( len < 6 )); then
        # %06s 会在左侧补空格，再用 tr 换成 0
        newname="$(printf '%06s' "$basename" | tr ' ' '0').$ext"
        # 若目标文件已存在，可在这里加处理逻辑，如附加下划线；示例简单覆盖
        mv -i -- "$f" "$newname"
        echo "重命名: $f -> $newname"
    fi
done

