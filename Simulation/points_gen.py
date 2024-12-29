import numpy as np
import open3d as o3d
# 获取当前文件的目录
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将相对路径转换为绝对路径
relative_path = "../"  # 相对路径，假设 "../test" 是你想添加的路径
absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
# 添加到 sys.path
sys.path.append(absolute_path)
from PointCloud.transformation import align_to_x_axis,align_to_axis
# 生成指定角度范围的圆柱面上的点云
def generate_partial_cylinder_points(center, axis, radius=1.0, height=2.0, angle_range=60, resolution=500, noise_stddev=0.01):
    # 归一化轴向量
    axis = axis / np.linalg.norm(axis)

    
    # 创建指定角度范围的圆柱面上的点云
    theta = np.linspace(0, np.deg2rad(angle_range), resolution)  # 角度范围从0到指定角度（以度为单位）
    z = np.linspace(-height / 2, height / 2, resolution)  # 高度范围从-高度/2到高度/2

    # 使用网格化坐标生成圆柱面上的点
    theta_grid, x_grid = np.meshgrid(theta, z)
    y_grid = radius * np.cos(theta_grid)
    z_grid = radius * np.sin(theta_grid)

    # 将生成的点展平成一维
    points = np.vstack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten())).T

    # 将点从局部坐标系转换到全局坐标系
    # points = align_to_axis(points,axis)+ center

    # points = align_to_axis(points,np.array([1,1,0]))
    # points[:,1]*=0.6
    # points =align_to_x_axis(points,np.array([1,1,0]))
    points =align_to_axis(points,axis)+ center
    # points =align_to_x_axis(points,axis)

    # points = align_to_axis(points,axis)+ center


    # 添加高斯噪声
    noise = np.random.normal(scale=noise_stddev, size=points.shape)
    noisy_points = points + noise

    return noisy_points

# 可视化点云
def visualize_point_cloud(points):
    # 创建Open3D点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 可视化
    o3d.visualization.draw_geometries([point_cloud])
axis_source= None
