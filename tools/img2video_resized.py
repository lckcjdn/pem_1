import os
import cv2
import argparse
import re
from tqdm import tqdm

def natural_sort_key(s):
    """按照自然排序方式对字符串进行排序（如：1, 2, 10 而不是 1, 10, 2）"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def images_to_video(input_dir, output_path, fps=30, codec='XVID', extension=None, prefix=None):
    """
    将图像序列转换为视频文件
    
    参数:
        input_dir (str): 输入图像所在的目录
        output_path (str): 输出视频文件的路径
        fps (int): 视频的帧率
        codec (str): 视频编码器
        extension (str): 只处理指定扩展名的图像文件
        prefix (str): 只处理指定前缀的图像文件
    """
    # 获取目录中的所有图像文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    if extension:
        if not extension.startswith('.'):
            extension = '.' + extension
        valid_extensions = (extension.lower(),)
    
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(valid_extensions) and 
                  (prefix is None or f.startswith(prefix))]
    
    if not image_files:
        print(f"错误: 在目录 {input_dir} 中没有找到符合条件的图像文件")
        return
    
    # 按照自然排序对文件名进行排序
    image_files.sort(key=natural_sort_key)
    # image_files = image_files[::4]
    
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 读取第一张图像以获取尺寸信息
    first_image_path = os.path.join(input_dir, image_files[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"错误: 无法读取图像 {first_image_path}")
        return
    
    # height, width = first_image.shape[:2]
    target_width = 2560
    target_height = 1440
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
    
    if not video_writer.isOpened():
        print(f"错误: 无法创建视频文件 {output_path}")
        return
    
    # 逐帧写入视频
    with tqdm(total=len(image_files), desc="生成视频") as pbar:
        for image_file in image_files:
            image_path = os.path.join(input_dir, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"警告: 无法读取图像 {image_path}，跳过")
                continue
            
            image = cv2.resize(image, (target_width, target_height))
            
            video_writer.write(image)
            pbar.update(1)
    
    # 释放资源
    video_writer.release()
    
    print(f"完成! 视频已保存到 {output_path}")
    print(f"视频信息: {target_height}, {fps} fps, 编码器: {codec}")

def main():
    parser = argparse.ArgumentParser(description='将图像序列转换为视频')
    parser.add_argument('input_dir', type=str, help='输入图像所在的目录')
    parser.add_argument('output_path', type=str, help='输出视频文件的路径')
    parser.add_argument('--fps', type=int, default=10, help='视频的帧率，默认为30')
    parser.add_argument('--codec', type=str, default='mp4v', help='视频编码器，默认为XVID')
    parser.add_argument('--extension', type=str, default=None, help='只处理指定扩展名的图像文件')
    parser.add_argument('--prefix', type=str, default=None, help='只处理指定前缀的图像文件')
    
    args = parser.parse_args()
    
    images_to_video(
        args.input_dir,
        args.output_path,
        args.fps,
        args.codec,
        args.extension,
        args.prefix
    )

if __name__ == '__main__':
    main()