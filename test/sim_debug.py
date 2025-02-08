# 读取 ../data/save/ans.ply
# 读取 ../data/save/ans1.ply

# 两个点云不同 颜色show

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import yaml  # 用于加载 YAML 文件
# 获取当前文件的目录


current_dir = os.path.dirname(os.path.abspath(__file__))

# 将相对路径转换为绝对路径
relative_path = "../"  # 相对路径，假设 "../test" 是你想添加的路径
absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
# 添加到 sys.path
sys.path.append(absolute_path)


def project_points_to_plane(points, normal, d):
    """
    将多个三维点投影到由法向量和偏移量定义的平面上。
    
    参数:
    points : array-like, 形状为(n,3)，三维点的坐标
    normal : array-like, 形状为(3,), 平面的法向量
    d : float，平面方程中的常数项
    
    返回:
    numpy数组，形状为(n,3)，投影点的坐标
    """
    # 确保输入是numpy数组
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    normal = np.asarray(normal, dtype=np.float64)
    
    # 计算点积
    dot_product = np.dot(points, normal) + d
    
    # 计算分母
    denominator = np.dot(normal, normal)
    
    # 计算投影向量
    projection_vector = (dot_product / denominator)[:, np.newaxis] * normal
    
    # 计算投影点
    P_prime = points - projection_vector
    
    return P_prime


from PointCloud.base import PointCloudBase

from Simulation.PcaTest import generate_partial_cylinder_points,visualize_point_cloud
center = np.array([0, 0, 0])  # 圆柱的中心点
axis = np.array([0.99959731,-0.00212741,-0.02829423])  # 圆柱的轴向量
# axis=np.array([0.98,0.02,0])
# axis=np.array([0,1,0])
axis_source=axis/np.linalg.norm(axis)
radius = 8.0  # 半径
height = 200  # 高度
angle_range = 60  # 圆柱面部分的角度范围，单位为度
resolution = 200  # 点云的分辨率，越大点云越密集
noise_stddev = 0.05 # 高斯噪声的标准差

# 生成圆柱面上的部分点云
points = generate_partial_cylinder_points(center, axis, radius, height, angle_range, resolution, noise_stddev)

# axis=np.array([0.99959731,-0.00212741,-0.02829423])
# axis=np.array([1,0,0])

# axis=np.array([0,1,0])

# 直接进行处理
# x_len=0.2
# 将0.2的touyi
from algorithm.pca.PcaAxis import *
PcaAxis_=PcaAxis()
# points = PcaAxis_.align_to_x_axis(points,axis)
a=points[:,0].T
b=points[:,1].T
c=points[:,2].T
print("c/a=",c/a)
# plt show points[:,1] points[:,2]
plt.figure(figsize=(10, 10))
plt.scatter(b, c, s=0.1)
plt.xlabel("x")
plt.ylabel("y")
# 设置xy比例相同
plt.axis('equal')
plt.show()

from algorithm.pca.PcaAxis import *
PcaAxis_=PcaAxis()

pca_model=PcaAxis_.fit(points,PCAMethod.ROBUST_PCA,alpha=0.25,final_MCD_step=True)
axis_list=pca_model.components_
# o3d.visualization.draw_geometries([pcd2])
# o3d.visualization.draw_geometries([pcd1])
# print(axis_list)
for axis_ in axis_list:
    source_points=points
    points=project_points_to_plane(source_points, axis_, 0)
    # 将 points 转换为 Open3D 点云
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    # # 半径滤波，设置半径和最低点数
    # radius = 0.02  # 搜索半径
    # min_neighbors = 100  # 半径内的最少点数
    # filtered_cloud, indices = point_cloud.remove_radius_outlier(nb_points=min_neighbors, radius=radius)

    # 对points进行半径滤波使用open3d
    #show points in plt 
    points_=np.array(points)
    plt.scatter(points_[:,1], points_[:,2], s=0.1)
    # set x y equal
    plt.axis('equal')
    plt.show()
    break