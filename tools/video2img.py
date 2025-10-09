import os
import cv2
import argparse
from tqdm import tqdm

def video_to_frames(video_path, output_dir='AR', prefix='frame', extension='jpg'):
    #    将视频文件转换为图像序列
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / 2)  # 计算帧间隔
    
    print(f"视频信息:")
    print(f"- 总帧数: {total_frames}")
    print(f"- 帧率: {video_fps} fps")
    print(f"- 每秒提取2帧")
    
    frame_count = 0
    saved_count = 0
    
    with tqdm(total=total_frames, desc="提取帧") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            # 每隔frame_interval帧保存一次
            if frame_count % frame_interval == 0:
                frame_filename = f"{prefix}_{saved_count:06d}.{extension}"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    # 释放资源
    video.release()
    
    print(f"完成! 共提取了 {frame_count} 帧图像，保存在 {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='将视频转换为图像序列')
    parser.add_argument('video_path', type=str, help='输入视频文件的路径')
    parser.add_argument('--output_dir', type=str, default='AR', help='输出图像的目录，默认为AR')
    parser.add_argument('--prefix', type=str, default='frame', help='输出图像文件名前缀')
    parser.add_argument('--extension', type=str, default='png', help='输出图像的文件扩展名')
    
    args = parser.parse_args()
    
    video_to_frames(
        args.video_path,
        args.output_dir,
        args.prefix,
        args.extension
    )

if __name__ == '__main__':
    main()