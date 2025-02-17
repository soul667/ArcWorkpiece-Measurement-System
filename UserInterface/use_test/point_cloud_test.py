import numpy as np
import open3d as o3d
import sys
import os

# Add parent directory to Python path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PointCouldProgress import segmentPointCloud, custom_down_sample

def generate_test_point_cloud(num_points=1000):
    """
    生成用于测试的点云数据
    生成一个立方体形状的点云，包含随机噪声
    """
    # 生成均匀分布的点云
    x = np.random.uniform(-5, 5, num_points)
    y = np.random.uniform(-5, 5, num_points)
    z = np.random.uniform(-5, 5, num_points)
    
    # 添加一些噪声点
    noise = np.random.normal(0, 0.2, (num_points, 3))
    points = np.column_stack((x, y, z)) + noise
    
    return points

def visualize_point_cloud(points, window_name="Point Cloud Viewer"):
    """
    使用Open3D可视化点云
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def test_segmentation():
    """
    测试点云分割功能
    """
    # 生成测试点云
    points = generate_test_point_cloud(2000)
    print("生成测试点云，点数：", len(points))
    
    # 可视化原始点云
    visualize_point_cloud(points, "原始点云")
    
    # 测试1：保留中心区域的点（x_mode=1）
    filtered_points1 = segmentPointCloud(
        points,
        x_range=[(-2, 2)],
        x_mode=1  # 保留-2到2之间的点
    )
    visualize_point_cloud(filtered_points1, "X轴分割-保留中心区域")
    
    # 测试2：去除中心区域的点（x_mode=0）
    filtered_points2 = segmentPointCloud(
        points,
        x_range=[(-2, 2)],
        x_mode=0  # 去除-2到2之间的点
    )
    visualize_point_cloud(filtered_points2, "X轴分割-去除中心区域")
    
    # 测试3：多轴组合分割
    filtered_points3 = segmentPointCloud(
        points,
        x_range=[(-2, 2)],
        y_range=[(-2, 2)],
        z_range=[(-2, 2)],
        x_mode=1,  # 保留X轴范围内的点
        y_mode=1,  # 保留Y轴范围内的点
        z_mode=0   # 去除Z轴范围内的点
    )
    visualize_point_cloud(filtered_points3, "多轴组合分割")
    
    # 测试4：下采样测试
    downsampled_points = custom_down_sample(points, downsample_rate=3)
    visualize_point_cloud(downsampled_points, "下采样后的点云")

if __name__ == "__main__":
    test_segmentation()
