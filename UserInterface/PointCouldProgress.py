import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

def custom_down_sample(points, downsample_rate=5):
    """
    自定义点云下采样函数
    通过对x坐标分组并按y坐标排序后进行均匀采样，保持点云形状特征

    参数：
        points (numpy.ndarray): 输入点云数据，形状为(N, 3)的二维数组，表示点的xyz坐标
        downsample_rate (int): 下采样率，每隔多少个点取一个，默认值为5

    返回：
        numpy.ndarray: 下采样后的点云数据
    """
    # 转换输入为NumPy数组以确保兼容性
    points = np.asarray(points)
    # 获取所有不重复的x坐标值
    unique_x = np.unique(points[:, 0])
    downsampled_points = []

    # 对每个x坐标值进行分组处理
    for x in unique_x:
        # 选取当前x坐标对应的所有点
        group = points[points[:, 0] == x]
        # 按y坐标排序以保持形状特征
        sorted_group = group[np.argsort(group[:, 1])]
        # 均匀采样
        downsampled_points.append(sorted_group[::downsample_rate])

    # 合并所有下采样后的点
    return np.vstack(downsampled_points)

def remove_statistical_outliers(pcd, nb_neighbors=100, std_ratio=0.5):
    """
    统计滤波去除点云中的异常点（噪声）

    参数:
        pcd (open3d.geometry.PointCloud): 输入的点云对象
        nb_neighbors (int): 计算统计值时考虑的临近点数量，默认100
        std_ratio (float): 标准差乘数，用于确定是否为异常点的阈值，默认0.5

    返回:
        open3d.geometry.PointCloud: 去除异常点后的点云对象
    """
    # 如果点云没有法线信息，计算法线
    if not hasattr(pcd, "normals"):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
    # 执行统计滤波
    cleaned_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, 
        std_ratio=std_ratio
    )
    return cleaned_pcd

def cut_point_cloud(pcd, x_min, x_max, y_min, y_max, z_min, z_max):
    """
    使用轴对齐包围盒裁剪点云

    参数:
        pcd (open3d.geometry.PointCloud): 输入的点云对象
        x_min, x_max (float): x轴方向的裁剪范围
        y_min, y_max (float): y轴方向的裁剪范围
        z_min, z_max (float): z轴方向的裁剪范围

    返回:
        open3d.geometry.PointCloud: 裁剪后的点云对象
    """
    # 创建轴对齐包围盒
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(x_min, y_min, z_min),
        max_bound=(x_max, y_max, z_max)
    )
    # 使用包围盒裁剪点云
    return pcd.crop(bbox)

def segmentPointCloud(points, x_range=None, y_range=None, z_range=None, 
                     x_mode='keep', y_mode='keep', z_mode='keep'):
    """
    基于坐标范围对点云进行分割

    参数:
        points (numpy.ndarray): 输入点云数据，形状为(N, 3)的数组
        x_range (list of tuple): x轴的分割范围列表，每个元素为(min, max)
        y_range (list of tuple): y轴的分割范围列表，每个元素为(min, max)
        z_range (list of tuple): z轴的分割范围列表，每个元素为(min, max)
        x_mode (str): x轴分割模式，'keep'表示保留范围内的点，'remove'表示去除范围内的点
        y_mode (str): y轴分割模式，'keep'表示保留范围内的点，'remove'表示去除范围内的点
        z_mode (str): z轴分割模式，'keep'表示保留范围内的点，'remove'表示去除范围内的点

    返回:
        numpy.ndarray: 分割后的点云数据
    """
    filtered_points = np.asarray(points)
    remove_diff_num = sum(mode == 'remove' for mode in [x_mode, y_mode, z_mode])
    remove_diff_range_num = sum(range is not None and range != [] for range in [x_range, y_range, z_range])
    if_filter_special = False
    filter_mask_special_use=[]
    first_range_num=0
    second_range_num=1

    if(remove_diff_num>1 and remove_diff_range_num==2):
        
       if_filter_special = True
        # 特殊处理
       use_ranges = [range_ for range_ in [x_range, y_range, z_range] if range_ is not None and range_ != []]
       if x_range==[]:
           first_range_num=1
           second_range_num=2
       if y_range==[]:
           first_range_num=0
           second_range_num=2
       if z_range==[]:
           first_range_num=0
           second_range_num=1
       # 一定有两个，之前处理过
       use_ranges_len =[len(range_) for range_ in use_ranges]
       min_len = min(use_ranges_len)
       # 开始解析filter_mask_special_use 两个use_ranges区间夹一个区间
       for i in range(min_len):
              x_min, x_max = use_ranges[first_range_num][i]
              y_min, y_max = use_ranges[second_range_num][i]
              filter_mask_special_use.append((x_min, x_max , y_min, y_max))
           

    if if_filter_special:
        # 特殊处理
        combined_mask = np.zeros(len(filtered_points), dtype=bool)
        for (x_min, x_max , y_min, y_max) in filter_mask_special_use:
            print(x_min, x_max , y_min, y_max)
            mask = (filtered_points[:, 0] >= x_min) & (filtered_points[:, 0] <= x_max) & (filtered_points[:, 1] >= y_min) & (filtered_points[:, 1] <= y_max)
            combined_mask = combined_mask | mask
        # 一次性应用所有mask
        filtered_points = filtered_points[~combined_mask]
    else:
        if x_range is not None and x_range != []:
            # 对于keep模式，我们需要保留在任意指定区域内的点
            if x_mode == 'keep':
                combined_mask = np.zeros(len(filtered_points), dtype=bool)
                for x_min, x_max in x_range:
                    mask = (filtered_points[:, 0] >= x_min) & (filtered_points[:, 0] <= x_max)
                    combined_mask = combined_mask | mask
                    print(f"X轴分割范围 [{x_min}, {x_max}]")
                filtered_points = filtered_points[combined_mask]
            # 对于remove模式，我们需要移除所有指定区域内的点
            else:
                for x_min, x_max in x_range:
                    mask = (filtered_points[:, 0] >= x_min) & (filtered_points[:, 0] <= x_max)
                    filtered_points = filtered_points[~mask]
                    print(f"X轴分割范围 [{x_min}, {x_max}]")
            print(f"X轴分割后点数: {len(filtered_points)}")

        # Y轴分割 并且y_range不为[](无元素)
        if y_range is not None and y_range != []:
            if y_mode == 'keep':
                combined_mask = np.zeros(len(filtered_points), dtype=bool)
                for y_min, y_max in y_range:
                    mask = (filtered_points[:, 1] >= y_min) & (filtered_points[:, 1] <= y_max)
                    combined_mask = combined_mask | mask
                    print(f"Y轴分割范围 [{y_min}, {y_max}]")
                filtered_points = filtered_points[combined_mask]
            else:
                for y_min, y_max in y_range:
                    mask = (filtered_points[:, 1] >= y_min) & (filtered_points[:, 1] <= y_max)
                    filtered_points = filtered_points[~mask]
                    print(f"Y轴分割范围 [{y_min}, {y_max}]")
            print(f"Y轴分割后点数: {len(filtered_points)}")

        # Z轴分割
        if z_range is not None and z_range != []:
            if z_mode == 'keep':
                combined_mask = np.zeros(len(filtered_points), dtype=bool)
                for z_min, z_max in z_range:
                    mask = (filtered_points[:, 2] >= z_min) & (filtered_points[:, 2] <= z_max)
                    combined_mask = combined_mask | mask
                    print(f"Z轴分割范围 [{z_min}, {z_max}]")
                filtered_points = filtered_points[combined_mask]
            else:
                for z_min, z_max in z_range:
                    mask = (filtered_points[:, 2] >= z_min) & (filtered_points[:, 2] <= z_max)
                    filtered_points = filtered_points[~mask]
                    print(f"Z轴分割范围 [{z_min}, {z_max}]")
            print(f"Z轴分割后点数: {len(filtered_points)}")


    print(f"分割后点云数量: {len(filtered_points)}")
    return filtered_points
