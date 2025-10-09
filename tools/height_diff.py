import laspy
import numpy as np
from scipy.spatial import cKDTree
import multiprocessing
from tqdm import tqdm
import os

def process_point(args):
    point_idx, points, tree, radius, height_threshold = args
    point = points[point_idx]
    
    # 查找半径内的邻域点
    neighbor_indices = tree.query_ball_point(point, radius)
    if len(neighbor_indices) < 3:  # 至少需要3个点才能计算高差
        return None
    
    neighbor_points = points[neighbor_indices]
    
    # 计算高差
    min_z = np.min(neighbor_points[:, 2])
    max_z = np.max(neighbor_points[:, 2])
    height_diff = max_z - min_z
    
    # 检查高差是否超过阈值
    if height_diff > height_threshold:
        return point_idx
    
    return None

def extract_high_diff_points(input_las_path, output_las_path, 
                            radius=0.5, height_threshold=0.2,
                            num_processes=None):
    """
    提取高差大于阈值的点
    
    参数:
    input_las_path: 输入LAS文件路径
    output_las_path: 输出LAS文件路径
    radius: 邻域半径(米)，默认0.5米
    height_threshold: 高差阈值(米)，默认0.2米
    num_processes: 使用的进程数，默认None(自动确定)
    """
    print(f"正在处理文件: {input_las_path}")
    
    # 读取LAS文件
    las = laspy.read(input_las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"读取了 {len(points)} 个点")
    
    # 创建KDTree用于快速邻域搜索
    tree = cKDTree(points)
    
    # 设置进程数
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 1)
    
    print(f"使用 {num_processes} 个进程进行并行计算")
    
    # 准备参数
    args = [(i, points, tree, radius, height_threshold) for i in range(len(points))]
    
    # 查找高差点
    high_diff_indices = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_point, args),
            total=len(points),
            desc="分析点云进度"
        ))
    
    # 收集结果
    high_diff_indices = [idx for idx in results if idx is not None]
    print(f"找到 {len(high_diff_indices)} 个高差大于 {height_threshold} 米的点")
    
    # 保存结果到新LAS文件
    if high_diff_indices:
        # 创建新LAS文件
        header = laspy.LasHeader(point_format=las.header.point_format)
        header.scales = las.header.scales
        header.offsets = las.header.offsets
        output_las = laspy.LasData(header)
        
        # 复制所有属性
        for dim in las.point_format.dimensions:
            setattr(output_las, dim.name, getattr(las, dim.name)[high_diff_indices])
        
        # 写入文件
        output_las.write(output_las_path)
        print(f"成功保存高差点到 {output_las_path}")
        return len(high_diff_indices)
    else:
        print("未找到符合条件的高差点")
        return 0

# 使用示例
if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    
    input_las = "/home/guitu/Data/ytj/guitu/csf_with_smooth.las"  # 输入点云文件
    output_las = "/home/guitu/Data/ytj/guitu/high_diff_points.las"  # 输出高差点文件
    
    # 提取高差点
    count = extract_high_diff_points(
        input_las, 
        output_las,
        radius=0.5,  # 50cm邻域半径
        height_threshold=0.2,  # 20cm高差阈值
        num_processes=os.cpu_count() - 1  # 使用CPU核心数-1的进程数
    )
    
    print(f"处理完成，共找到 {count} 个高差点")