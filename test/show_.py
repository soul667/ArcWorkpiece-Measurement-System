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

current_dir = os.path.dirname(os.path.abspath(__file__))

# 将相对路径转换为绝对路径
relative_path = "../"  # 相对路径，假设 "../test" 是你想添加的路径
absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
# 添加到 sys.path
sys.path.append(absolute_path)

from PointCloud.base import PointCloudBase

# Load the point clouds
# pcd1 = o3d.io.read_point_cloud("data/example/1.ply")
pcd2 = o3d.io.read_point_cloud("../data/save/ans2.ply")
voxel_size = 0.05  # 设置体素大小，值越大，降采样程度越大
pcd2_down = pcd2.voxel_down_sample(voxel_size=voxel_size)
# o3d.visualization.draw_geometries([pcd2_down], window_name="Voxel Downsampled Point Cloud")

# Make sure they have the same number of points for comparison
# if len(pcd1.points) != len(pcd2.points):
#     print("Point clouds have different number of points.")
# else:
    # Convert point cloud data to numpy arrays
points1 = np.asarray(pcd2.points)
points2 = np.asarray(pcd2.points)

from algorithm.pca.PcaAxis import *
PcaAxis_=PcaAxis()

pca_model=PcaAxis_.fit(points2,PCAMethod.ROBUST_PCA,alpha=0.25,final_MCD_step=False)
# pca_model=PcaAxis_.fit(points2,PCAMethod.ORDINARY_PCA,alpha=0.75,final_MCD_step=False)

axis_list=pca_model.components_
# o3d.visualization.draw_geometries([pcd2])
# o3d.visualization.draw_geometries([pcd1])
# print(axis_list)
for axis_ in axis_list:
    source_points=points2
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

# -0.  0.  0.
# axis=np.array([-0.99959946,0.00153588,0.0282603])
# 0.99941534  0.00464058 -0.03387692
# -0.99959981    0
# axis=np.array([ 0.99976951,0.00566948,0.02070953])
# axis=np.array([0,1,0])

# 直接进行处理
# x_len=0.2
# # 将0.2的touyi
# from algorithm.pca.PcaAxis import *
# PcaAxis_=PcaAxis()
# ax1=PcaAxis_.get_axis(points2, model=PCAMethod.ORDINARY_PCA)

# points = PcaAxis_.align_to_x_axis(points2,axis)
# source_points=points2
# points=project_points_to_plane(points2, axis, 0)
# # 将 points 转换为 Open3D 点云
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)
# # 半径滤波，设置半径和最低点数
# radius = 0.02  # 搜索半径
# min_neighbors = 100  # 半径内的最少点数
# filtered_cloud, indices = point_cloud.remove_radius_outlier(nb_points=min_neighbors, radius=radius)

# # 对points进行半径滤波使用open3d
# #show points in plt 
# points_=np.array(filtered_cloud.points)
# plt.scatter(points_[:,1], points_[:,2], s=0.1)
# # set x y equal
# plt.axis('equal')
# plt.show()

# points[:,0]=points2[:,0]
# # points=points2
# from algorithm.PreciseArcFitting import PreciseArcFitting
# axis= np.array([ 0.99976659,0.00624024,0.02068647])
# # axis= np.array([ 1,0,0])

# arc_fitting=PreciseArcFitting(np.array(points2),axis)
# data=arc_fitting.fit()
# print(data[0],data[1])
# print(arc_fitting.fit_basic())
# xmin =np.min (points[:,0])
# xmax=np.max (points[:,0])
# ranges_=xmax-xmin
# PointCloudCalibrate.x_minax = [np.min(PointCloudCalibrate.points[:,0]),np.max(PointCloudCalibrate.points[:,0])]
# PointCloudCalibrate.y_minax = [np.min(PointCloudCalibrate.points[:,1]),np.max(PointCloudCalibrate.points[:,1])]
# PointCloudCalibrate.z_minax = [np.min(PointCloudCalibrate.points[:,2]),np.max(PointCloudCalibrate.points[:,2])]
# PointCloudCalibrate.seg_x_self_scale(model=[1,1,1],ranges=[[0.0,0.05]])
# # PointCloudCalibrate.show_points_2d(PointCloudCalibrate.points,1,2)
# points_use=np.stack([PointCloudCalibrate.points[:,1],PointCloudCalibrate.points[:,2]],axis=1)
# arc.fit_circle_arc(points_use)
# plt.scatter(points_use[:,0], points_use[:,1], s=0.1)
# # 设置xy比例相同
# plt.axis('equal')

# center = arc.first_param[0:2]
# radius = arc.first_param[2]

# # 绘制拟合的圆
# circle = plt.Circle((center[0], center[1]), radius, color='red', fill=False, label="拟合的圆形")
# plt.gca().add_patch(circle)

# # 绘制半径
# # 选择圆上的一点，例如角度0的方向
# x_Radius = center[0] + radius
# y_Radius = center[1]
# plt.plot([center[0], x_Radius], [center[1], y_Radius], 'r-', label="半径")

# # 标注半径的长度
# plt.text(center[0] + radius/2, center[1], f'r = {radius:.2f}', ha='center', va='center')

# # 显示图例
# plt.legend()

# plt.show()



# PointCloudCalibrate.save("use_see.ply")

# #save PointCloudCalibrate.points to 
# print(arc.first_param)