from matplotlib import pyplot as plt
import numpy as np
import open3d as o3d

import numpy as np
from scipy.optimize import minimize
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
def rotate_point(original_point,original_axis,target_vector):
    """
    将原来轴在(1, 0, 0)的圆柱的轴旋转到目标向量target_vector的方向，并旋转原始点。
    
    参数:
    original_point: list或numpy数组，原始点的坐标。
    target_vector: list或numpy数组，目标方向向量。
    
    返回:
    rotated_point: list，旋转后的点的坐标。
    """
    # 将输入转换为numpy数组并确保为浮点型
    original_point = np.array(original_point, dtype=np.float64)
    target_vector = np.array(target_vector, dtype=np.float64)
    
    # 归一化目标向量
    target_vector = target_vector / np.linalg.norm(target_vector)
    
    # 原轴向量
    # original_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    
    # 计算旋转轴
    rotation_axis = np.cross(original_axis, target_vector)
    
    # 如果旋转轴长度接近0，说明原轴和目标向量平行
    if np.linalg.norm(rotation_axis) < 1e-6:
        if np.dot(original_axis, target_vector) > 0:
            # 方向相同，无需旋转
            rotation_matrix = np.eye(3, dtype=np.float64)
        else:
            # 方向相反，旋转180度
            rotation_matrix = -np.eye(3, dtype=np.float64)
    else:
        # 归一化旋转轴
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # 计算旋转角度的余弦值
        cos_theta = np.dot(original_axis, target_vector)
        
        # 计算旋转角度
        theta = np.arccos(cos_theta)
        
        # 计算旋转矩阵的组成部分
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ], dtype=np.float64)
        
        # 罗德里格斯旋转公式
        rotation_matrix = (
            np.eye(3, dtype=np.float64) + 
            np.sin(theta) * K + 
            (1 - np.cos(theta)) * np.dot(K, K)
        )
    
    # 旋转原始点
    # rotated_point = np.dot(rotation_matrix, original_point)
    rotated_points = np.dot(rotation_matrix, original_point.T).T
    
    return rotated_points


# 生成指定角度范围的圆柱面上的点云
def generate_partial_cylinder_points(center, axis, radius=1.0, height=2.0, angle_range=60, resolution=500, noise_stddev=0.01,beta=1):
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
    original_axis=np.array([1,0,0])
    if beta==1:
        points = rotate_point(points,original_axis,axis)+ center
    else: 
        points=rotate_point(points,original_axis,axis)+ center
        points[:,1]=points[:,1]*beta
        # points=rotate_point(np.array(points),axis,original_axis)
        # print(points)


    noise = np.random.normal(scale=noise_stddev, size=points.shape)
    noisy_points = points + noise

    return noisy_points
def visualize_point_cloud(points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.visualization.draw_geometries([pcd])
# 调用函数生成点云并可视化
if __name__ == "__main__":
    # 可调参数
    center = np.array([0, 0, 0])  # 圆柱的中心点
    axis = np.array([0.99, 0.5, 0.05])  # 圆柱的轴向量
    axis_source=axis/np.linalg.norm(axis)
    radius = 9.0  # 半径
    height = 20.0  # 高度
    angle_range = 30  # 圆柱面部分的角度范围，单位为度
    resolution = 200  # 点云的分辨率，越大点云越密集
    noise_stddev = 0.01  # 高斯噪声的标准差

    # 生成圆柱面上的部分点云
    points = generate_partial_cylinder_points(center, axis, radius, height, angle_range, resolution, noise_stddev,0.9)
    visualize_point_cloud(points)
    # show in open3d 


    a=points[:,0].T
    b=points[:,1].T
    c=points[:,2].T

    
#     print("c/a=",c/a)
# plt show points[:,1] points[:,2]
    plt.figure(figsize=(10, 10))
    plt.scatter(b, c, s=0.1)
    plt.xlabel("x")
    plt.ylabel("y")
    # 设置xy比例相同
    plt.axis('equal')
    plt.show()

#     # 转换为open3d点云
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # 去噪和降采样
#     pcd = pcd.voxel_down_sample(voxel_size=0.1)
#     #去噪
#     pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
#     points = np.asarray(pcd.points)
    # def error_function(params):
    #     alpha, beta, x0, y0, z0, r = params
    #     total_error = 0
    #     for x, y, z in points:
    #         y_scaled = y / beta
    #         predicted_r2 = (x - x0)**2 + (y_scaled - y0)**2 + (z - z0)**2 - (x - x0)*alpha - (y_scaled - y0)*beta
    #         total_error += (predicted_r2 - r**2)**2
    #     return total_error
    #     # 初始猜测值 [alpha, beta, x0, y0, z0, r]
    # def callback(params):
    #     optimization_steps.append(params)
    #     print(f"Current parameters: {params}")
    # initial_guess = [0.7, 0.7, 0, 0, 0, 40]

    # # 优化求解
    # result = minimize(error_function, initial_guess, method='BFGS')
    # alpha_opt, beta_opt, x0_opt, y0_opt, z0_opt, r_opt = result.x

    # print(f"Optimal alpha: {alpha_opt}")
    # print(f"Optimal beta: {beta_opt}")
    # print(f"Optimal center: ({x0_opt}, {y0_opt}, {z0_opt})")
    # print(f"Optimal radius: {r_opt}")
    # show
    # o3d.visualization.draw_geometries([pcd])
    from algorithm.pca.PcaAxis import *
    PcaAxis_=PcaAxis()
    ax1=PcaAxis_.get_axis(points, model=PCAMethod.ORDINARY_PCA)
    # ax1=PcaAxis_.get_axis(points, model=PCAMethod.ORDINARY_PCA)

    print(ax1, ax1 / np.linalg.norm(axis))
    print(axis,axis / np.linalg.norm(axis))
    # ax2=PcaAxis_.get_axis(points, model=PCAMethod.ROBUST_PCA)
    # # 可视化点云
    # visualize_point_cloud(points)
# print(points)
# points=read ../points1.txt
