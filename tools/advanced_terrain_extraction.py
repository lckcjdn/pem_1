import laspy
import numpy as np
from sklearn.decomposition import PCA
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import morphology, measure
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
import os

# 定义全局分析格网函数，用于多进程处理
def analyze_grid(args):
    grid_item, params = args
    grid_key, grid_points_list = grid_item
    r, c = grid_key
    
    height_threshold = params['height_threshold']
    planarity_threshold = params['planarity_threshold']
    linearity_threshold = params['linearity_threshold']
    slope_min_angle = params['slope_min_angle']
    slope_max_angle = params['slope_max_angle']
    road_planarity_threshold = params['road_planarity_threshold']
    road_max_angle = params['road_max_angle']
    min_points_per_grid = params['min_points_per_grid']
    
    if len(grid_points_list) < min_points_per_grid:  # 点数不足
        return r, c, None, None, None, 0
    
    grid_points = np.array(grid_points_list)
    
    # 计算高差
    height_diff = np.max(grid_points[:, 2]) - np.min(grid_points[:, 2])
    
    # 进行PCA分析
    pca = PCA(n_components=3)
    pca.fit(grid_points)
    
    # 获取特征值和特征向量
    eigenvalues = pca.explained_variance_
    eigenvectors = pca.components_
    
    # 计算平面性指标 (λ3 / (λ1 + λ2 + λ3))
    planarity = eigenvalues[2] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
    
    # 计算线性特征指标 (λ1 - λ2) / λ1
    linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0

    
    # 计算法向量 (第三个特征向量)
    normal_vector = eigenvectors[2]
    
    # 计算法向量与z轴的夹角 (弧度)
    z_axis = np.array([0, 0, 1])
    angle_with_z = np.arccos(np.abs(np.dot(normal_vector, z_axis)))
    angle_degrees = np.degrees(angle_with_z)
    
    # 判断是否为沟渠/田埂/斜坡
    is_slope_feature = (height_diff > height_threshold and
                   planarity < planarity_threshold and linearity < linearity_threshold and
                   slope_min_angle < angle_degrees < slope_max_angle)

    # 判断是否为道路
    is_road = (height_diff < height_threshold and planarity < road_planarity_threshold and
            linearity < linearity_threshold and  # 道路应具有低线性度
            angle_degrees < road_max_angle)
    
    # 确定特征类型
    feature_type = 0
    if is_slope_feature:
        feature_type = 1  # 1表示沟渠/田埂/斜坡
    elif is_road:
        feature_type = 2  # 2表示道路
        
    return r, c, height_diff, planarity, angle_degrees, feature_type

def extract_advanced_terrain_features(input_las_path, output_las_path, 
                                     resolution=0.2, 
                                     height_threshold=0.1,
                                     planarity_threshold=0.1,
                                     linearity_threshold=0.7,
                                     slope_min_angle=10,
                                     slope_max_angle=80,
                                     road_planarity_threshold=0.05,
                                     road_max_angle=10,
                                     min_points_per_grid=3,
                                     visualize=False,
                                     num_processes=None):
    """
    从LAS地面点云中提取高级地形特征
    
    参数:
    input_las_path: 输入LAS文件路径
    output_las_path: 输出LAS文件路径
    resolution: 格网分辨率，默认20cm
    height_threshold: 高差阈值，默认10cm
    planarity_threshold: 平面性阈值，默认0.1
    slope_min_angle: 斜坡最小角度，默认10度
    slope_max_angle: 斜坡最大角度，默认80度
    road_planarity_threshold: 道路平面性阈值，默认0.05
    road_max_angle: 道路最大角度，默认10度
    min_points_per_grid: 每个格网最少点数，默认3
    visualize: 是否可视化结果，默认False
    
    返回:
    feature_points: 提取的特征点数组
    """
    print(f"正在处理文件: {input_las_path}")
    
    # 读取LAS文件
    las = laspy.read(input_las_path)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"读取了 {len(points)} 个点")
    
    # 坐标中心化
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    print(f"坐标中心化完成，中心点: {centroid}")
    
    # 获取点云范围
    x_min, y_min = np.min(centered_points[:, :2], axis=0)
    x_max, y_max = np.max(centered_points[:, :2], axis=0)
    
    # 设置格网分辨率
    cols = int((x_max - x_min) / resolution) + 1
    rows = int((y_max - y_min) / resolution) + 1
    print(f"创建 {rows}x{cols} 格网，分辨率: {resolution}m")
    
    # 创建格网索引
    x_idx = ((centered_points[:, 0] - x_min) / resolution).astype(int)
    y_idx = ((centered_points[:, 1] - y_min) / resolution).astype(int)
    
    # 创建格网字典，存储每个格网内的点
    grid_dict = {}
    for i in range(len(centered_points)):
        r, c = y_idx[i], x_idx[i]
        if 0 <= r < rows and 0 <= c < cols:  # 确保索引在范围内
            grid_key = (r, c)
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(centered_points[i])
    
    print(f"有效格网数量: {len(grid_dict)}")
    
    # 创建特征图
    feature_map = np.zeros((rows, cols), dtype=np.int8)
    planarity_map = np.full((rows, cols), np.nan)
    height_diff_map = np.full((rows, cols), np.nan)
    angle_map = np.full((rows, cols), np.nan)
    
    # 分析每个格网（并行处理）
    print("开始并行分析格网...")
    
    # 设置进程数
    if num_processes is None:
        num_processes = max(1, os.cpu_count() - 1)  # 默认使用CPU核心数-1的进程数
    
    print(f"使用 {num_processes} 个进程进行并行计算")
    
    # 准备参数字典
    params = {
        'height_threshold': height_threshold,
        'planarity_threshold': planarity_threshold,
        'linearity_threshold': linearity_threshold,
        'slope_min_angle': slope_min_angle,
        'slope_max_angle': slope_max_angle,
        'road_planarity_threshold': road_planarity_threshold,
        'road_max_angle': road_max_angle,
        'min_points_per_grid': min_points_per_grid
    }
    
    # 准备输入数据，为每个格网添加参数
    grid_items_with_params = [(item, params) for item in grid_dict.items()]
    
    # 创建进程池
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(analyze_grid, grid_items_with_params),
            total=len(grid_dict),
            desc="分析格网进度"
        ))
    
    # 处理结果
    print("合并并行计算结果...")
    for result in results:
        if result is None or len(result) < 6:
            continue
            
        r, c, height_diff, planarity, angle_degrees, feature_type = result
        
        if height_diff is not None:  # 确保有有效结果
            height_diff_map[r, c] = height_diff
            planarity_map[r, c] = planarity
            angle_map[r, c] = angle_degrees
            feature_map[r, c] = feature_type
    
    # 应用形态学操作清理特征图
    print("应用形态学操作清理特征图...")
    # 移除小区域
    for feature_type in [1, 2]:
        type_mask = feature_map == feature_type
        cleaned = morphology.remove_small_objects(type_mask, min_size=5)
        feature_map[type_mask] = 0  # 清除原始标记
        feature_map[cleaned] = feature_type  # 设置清理后的标记
    
    # 区域标记
    labeled_features = measure.label(feature_map > 0)
    regions = measure.regionprops(labeled_features)
    print(f"检测到 {len(regions)} 个特征区域")
    
    # 提取特征点
    feature_points = []
    feature_types = []
    colors = []
    
    for region in regions:
        # 获取区域坐标
        coords = region.coords  # (row, col)格式
        region_type = np.max(feature_map[coords[:, 0], coords[:, 1]])  # 获取区域类型
        
        # 为每个区域提取代表点
        for coord in coords:
            r, c = coord
            grid_key = (r, c)
            if grid_key in grid_dict and len(grid_dict[grid_key]) >= min_points_per_grid:
                # 计算格网中心点
                grid_center = np.mean(np.array(grid_dict[grid_key]), axis=0)
                # 转换回原始坐标
                original_center = grid_center + centroid
                
                feature_points.append(original_center)
                feature_types.append(region_type)
                
                if region_type == 1:  # 沟渠/田埂/斜坡
                    colors.append((65535, 0, 0))  # 红色
                else:  # 道路
                    colors.append((0, 65535, 0))  # 绿色
    
    # 可视化
    if visualize:
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(feature_map, cmap='viridis')
        plt.title('特征类型')
        plt.colorbar()
        
        plt.subplot(2, 2, 2)
        plt.imshow(height_diff_map, cmap='jet')
        plt.title('高差 (m)')
        plt.colorbar()
        
        plt.subplot(2, 2, 3)
        plt.imshow(planarity_map, cmap='jet')
        plt.title('平面性指标')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.imshow(angle_map, cmap='jet')
        plt.title('与Z轴夹角 (度)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(output_las_path.replace('.las', '_analysis.png'))
        plt.close()
    
    # 保存到LAS文件
    if feature_points:
        feature_points = np.array(feature_points)
        print(f"提取了 {len(feature_points)} 个特征点")
        
        # 创建LAS头
        header = laspy.LasHeader(point_format=3)  # 使用点格式3，支持RGB
        header.scales = las.header.scales
        header.offsets = las.header.offsets
        output_las = laspy.LasData(header)
        
        # 设置点坐标
        output_las.x = feature_points[:, 0]
        output_las.y = feature_points[:, 1]
        output_las.z = feature_points[:, 2]
        
        # 设置分类和颜色
        output_las.classification = feature_types
        output_las.red = [c[0] for c in colors]
        output_las.green = [c[1] for c in colors]
        output_las.blue = [c[2] for c in colors]
        
        # 写入文件
        output_las.write(output_las_path)
        print(f"成功保存特征点到 {output_las_path}")
        return feature_points
    else:
        print("未检测到符合条件的地形特征")
        return None

# 使用示例
if __name__ == "__main__":
    # 设置多进程启动方法（在Windows上使用spawn，在Linux/Mac上使用fork）
    multiprocessing.set_start_method('spawn', force=True)
    
    input_las = "/home/guitu/Data/ytj/csf_with_smooth.las"  # 输入地面点云文件
    output_las = "/home/guitu/Data/ytj/advanced_terrain_features.las"  # 输出特征点文件
    
    # 提取地形特征
    feature_points = extract_advanced_terrain_features(
        input_las, 
        output_las,
        resolution=0.5,  # 20cm格网
        height_threshold=0.1,  # 10cm高差阈值
        planarity_threshold=0.1,  # 平面性阈值
        linearity_threshold=0.3,
        slope_min_angle=10,  # 最小斜坡角度
        slope_max_angle=80,  # 最大斜坡角度
        road_planarity_threshold=0.05,  # 道路平面性阈值
        road_max_angle=5, # 道路最大角度
        visualize=False,  # 生成可视化结果
        num_processes=os.cpu_count() - 1  # 使用CPU核心数-1的进程数
    )