import laspy
import numpy as np
from scipy import ndimage
from skimage import morphology, measure

def process_ground_points(las_file, output_file):
    # 读取地面点云
    las = laspy.read(las_file)
    points = np.vstack((las.x, las.y, las.z)).T
    
    # 创建DEM网格
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    # 设置网格分辨率 (0.2米分辨率可检测1m宽特征)
    resolution = 0.5
    cols = int((x_max - x_min) / resolution) + 1
    rows = int((y_max - y_min) / resolution) + 1
    
    # 创建DEM (使用最小高程值填充网格)
    dem = np.full((rows, cols), np.nan)
    x_idx = ((points[:, 0] - x_min) / resolution).astype(int)
    y_idx = ((points[:, 1] - y_min) / resolution).astype(int)
    
    # 使用最低点填充DEM (避免植被残留)
    for i in range(len(points)):
        r, c = y_idx[i], x_idx[i]
        if 0 <= r < rows and 0 <= c < cols:  # 确保索引在范围内
            if np.isnan(dem[r, c]) or points[i, 2] < dem[r, c]:
                dem[r, c] = points[i, 2]
    
    # 填充小空隙
    dem_filled = ndimage.generic_filter(dem, np.nanmin, size=3)
    
    # 计算坡度
    sobel_x = ndimage.sobel(dem_filled, axis=1)
    sobel_y = ndimage.sobel(dem_filled, axis=0)
    slope = np.degrees(np.arctan(np.sqrt(sobel_x**2 + sobel_y**2)))
    
    # 1. 田埂/沟渠检测
    def detect_ridges(dem, slope_threshold=8, width_limit=3.5):
        """检测窄田埂/沟渠 (宽度约1米)"""
        flat_mask = slope < slope_threshold
        local_max = ndimage.maximum_filter(dem, size=3)
        local_min = ndimage.minimum_filter(dem, size=3)
        elevation_diff = local_max - local_min
        ridge_mask = (elevation_diff > 0.2) & (elevation_diff < 1.0) & flat_mask
        cleaned = morphology.remove_small_objects(ridge_mask, 20)
        cleaned = morphology.remove_small_holes(cleaned, 10)
        skeleton = morphology.skeletonize(cleaned)
        dist_transform = ndimage.distance_transform_edt(cleaned)
        skeleton[dist_transform > width_limit] = False
        return skeleton
    
    ridge_skeleton = detect_ridges(dem_filled)
    
    # 2. 斜坡区域处理
    def detect_slopes(slope_map, min_slope=15):
        """检测斜坡并提取坡顶坡底"""
        slope_mask = slope_map > min_slope
        slope_mask = morphology.binary_closing(slope_mask, morphology.disk(2))
        slope_mask = morphology.binary_opening(slope_mask, morphology.disk(1))
        labeled = measure.label(slope_mask)
        regions = measure.regionprops(labeled)
        
        top_lines = []
        bottom_lines = []
        
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            region_slope = slope[minr:maxr, minc:maxc]
            region_dem = dem_filled[minr:maxr, minc:maxc]
            
            row_variation = np.mean(np.abs(np.diff(region_slope, axis=0)))
            col_variation = np.mean(np.abs(np.diff(region_slope, axis=1)))
            
            if row_variation > col_variation:
                for col in range(minc, maxc):
                    col_slice = region_dem[:, col - minc]
                    if np.any(np.isfinite(col_slice)):
                        valid_idx = np.where(np.isfinite(col_slice))[0]
                        if len(valid_idx) > 2:
                            y_idx = valid_idx + minr
                            x_idx = np.full_like(y_idx, col)
                            line = np.column_stack((x_idx, y_idx))
                            top_idx = np.argmax(col_slice[valid_idx])
                            bottom_idx = np.argmin(col_slice[valid_idx])
                            top_lines.append(line[top_idx])
                            bottom_lines.append(line[bottom_idx])
            else:
                for row in range(minr, maxr):
                    row_slice = region_dem[row - minr, :]
                    if np.any(np.isfinite(row_slice)):
                        valid_idx = np.where(np.isfinite(row_slice))[0]
                        if len(valid_idx) > 2:
                            x_idx = valid_idx + minc
                            y_idx = np.full_like(x_idx, row)
                            line = np.column_stack((x_idx, y_idx))
                            top_idx = np.argmax(row_slice[valid_idx])
                            bottom_idx = np.argmin(row_slice[valid_idx])
                            top_lines.append(line[top_idx])
                            bottom_lines.append(line[bottom_idx])
        
        return np.array(top_lines), np.array(bottom_lines)
    
    slope_top, slope_bottom = detect_slopes(slope)
    
    # 3. 道路检测
    # def detect_roads(dem, slope_map, max_slope=3, min_length=20):
    #     """检测硬化道路"""
    #     road_mask = slope_map < max_slope
    #     road_mask = morphology.binary_closing(road_mask, morphology.rectangle(5, 15))
    #     road_mask = morphology.binary_opening(road_mask, morphology.rectangle(3, 3))
    #     skeleton = morphology.skeletonize(road_mask)
    #     labeled = measure.label(skeleton)
    #     regions = measure.regionprops(labeled)
        
    #     road_lines = []
    #     for region in regions:
    #         if region.area > min_length:
    #             coords = region.coords
    #             if len(coords) > 10:
    #                 road_lines.append(coords)
        
    #     return road_lines
    
    # road_centerlines = detect_roads(dem_filled, slope)
    
    # 坐标转换回真实坐标
    def grid_to_world(grid_coords):
        x = grid_coords[:, 0] * resolution + x_min
        y = grid_coords[:, 1] * resolution + y_min
        return x, y
    
    # 收集所有特征点并分配分类和颜色
    all_points = []
    classifications = []
    colors = []
    
    # 田埂/沟渠点 (分类1，红色)
    ridge_coords = np.argwhere(ridge_skeleton)
    if len(ridge_coords) > 0:
        x, y = grid_to_world(ridge_coords)
        z = [dem_filled[int(r), int(c)] for c, r in ridge_coords if 0 <= int(r) < dem_filled.shape[0] and 0 <= int(c) < dem_filled.shape[1]]
        all_points.extend(zip(x, y, z))
        classifications.extend([1] * len(z))
        colors.extend([(65535, 0, 0)] * len(z))  # 红色 (RGB: 255,0,0)
    
    # 斜坡顶部点 (分类2，绿色)
    if len(slope_top) > 0:
        x, y = grid_to_world(slope_top)
        z = [dem_filled[int(r), int(c)] for c, r in slope_top if 0 <= int(r) < dem_filled.shape[0] and 0 <= int(c) < dem_filled.shape[1]]
        all_points.extend(zip(x, y, z))
        classifications.extend([2] * len(z))
        colors.extend([(0, 65535, 0)] * len(z))  # 绿色 (RGB: 0,255,0)
    
    # 斜坡底部点 (分类3，蓝色)
    if len(slope_bottom) > 0:
        x, y = grid_to_world(slope_bottom)
        z = [dem_filled[int(r), int(c)] for c, r in slope_bottom if 0 <= int(r) < dem_filled.shape[0] and 0 <= int(c) < dem_filled.shape[1]]
        all_points.extend(zip(x, y, z))
        classifications.extend([3] * len(z))
        colors.extend([(0, 0, 65535)] * len(z))  # 蓝色 (RGB: 0,0,255)
    
    # # 道路点 (分类4，黄色)
    # for line in road_centerlines:
    #     if len(line) > 0:
    #         x, y = grid_to_world(line)
    #         z = [dem_filled[int(r), int(c)] for c, r in line if 0 <= int(r) < dem_filled.shape[0] and 0 <= int(c) < dem_filled.shape[1]]
    #         all_points.extend(zip(x, y, z))
    #         classifications.extend([4] * len(z))
    #         colors.extend([(65535, 65535, 0)] * len(z))  # 黄色 (RGB: 255,255,0)
    
    # 保存到LAS文件
    if all_points:
        points_array = np.array(all_points)
        header = laspy.LasHeader(point_format=3)  # 使用点格式3，支持RGB
        header.scales = las.header.scales
        header.offsets = las.header.offsets
        output_las = laspy.LasData(header)
        
        output_las.x = points_array[:, 0]
        output_las.y = points_array[:, 1]
        output_las.z = points_array[:, 2]
        output_las.classification = classifications
        output_las.red = [c[0] for c in colors]
        output_las.green = [c[1] for c in colors]
        output_las.blue = [c[2] for c in colors]
        
        output_las.write(output_file)
        print(f"成功保存 {len(points_array)} 个地形特征点到 {output_file}")
        return points_array
    else:
        print("未检测到符合条件的地形特征")
        return None

# 使用示例
if __name__ == "__main__":
    input_las = "/home/guitu/Data/ytj/csf_with_smooth.las"  # 输入地面点云文件
    output_las = "/home/guitu/Data/ytj/feature_points.las"  # 输出高程点文件
    
    # 处理点云并提取特征
    feature_points = process_ground_points(input_las, output_las)