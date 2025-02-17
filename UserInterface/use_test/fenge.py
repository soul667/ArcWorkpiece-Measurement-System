import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
def align_to_x_axis(points, target_vector):
    """
    将点云变换，使给定的向量方向对齐到 x 轴。

    参数:
        points (np.ndarray): 点云数据，形状为 n x 3。
        target_vector (np.ndarray): 要对齐到 x 轴的向量，形状为 3。

    返回:
        aligned_points (np.ndarray): 对齐后的点云数据，形状为 n x 3。
    """
    # 定义目标 x 轴方向
    target_x = np.array([1, 0, 0])
    
    # 归一化目标向量
    target_vector = target_vector / np.linalg.norm(target_vector)
    
    # 计算旋转轴（叉积）
    rotation_axis = np.cross(target_vector, target_x)
    axis_norm = np.linalg.norm(rotation_axis)
    
    # 如果旋转轴接近零，说明向量已经对齐
    if axis_norm < 1e-6:
        return points
    
    rotation_axis /= axis_norm  # 归一化旋转轴
    
    # 计算旋转角度（点积）
    angle = np.arccos(np.clip(np.dot(target_vector, target_x), -1.0, 1.0))
    
    # 构建旋转矩阵
    R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
    
    # 应用旋转
    aligned_points = np.dot(points, R.T)
    
    return aligned_points

import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

# 从 ./ans1/1.txt 读取数据
points = np.loadtxt('./za/ans1/337.txt')
target_vector = -np.array([-0.999317,0.0206028,0.0306875])  # 新的目标向量

points = align_to_x_axis(points, target_vector)
# print(points)
points=points[:, 1:3]
point_spurce=points
# points = np.loadtxt('./ans1/100.txt')
# points=np.stack((x,y),axis=-1)

# 数据截断到小数点后三位
# data = np.round(data, 3)

def smooth_points(points, smoothing_factor=0.1):
    """
    使用 B 样条拟合平滑一组二维点。

    参数:
    points: numpy.ndarray
        一个 Nx2 的 (x, y) 坐标数组。
    smoothing_factor: float
        控制平滑程度的非负值。较大的值会产生更平滑的曲线（默认值为 0，表示插值点）。

    返回:
    smoothed_points: numpy.ndarray
        一个 Mx2 的平滑 (x, y) 坐标数组。
    """
    points = np.asarray(points)
    if points.shape[1] != 2:
        raise ValueError("输入点必须是一个 Nx2 的 (x, y) 坐标数组。")

    # 提取 x 和 y 坐标
    x, y = points[:, 0], points[:, 1]

    # 使用平滑拟合 B 样条
    tck, u = splprep([x, y], s=smoothing_factor)

    # 生成平滑点
    u_fine = np.linspace(0, 1, len(points) * 10)  # 将分辨率提高 10 倍
    x_smooth, y_smooth = splev(u_fine, tck)

    smoothed_points = np.column_stack((x_smooth, y_smooth))
    return smoothed_points

# 平滑处理点
points = smooth_points(points, smoothing_factor=2)

# 绘制散点图
plt.scatter(points[:, 0], points[:, 1], s=0.1)
plt.xlabel("x")
plt.ylabel("y")
# xy轴等比例
plt.axis('equal')
plt.show()

# 保存图像到当前目录 ./ans1+当前时间
# plt.savefig(f"./ans1/{time.time()}.png")


def fit_circle_arc(points): 
    """
    Fit a circle arc to given points using eigenvalue decomposition.
    
    points: a Nx2 numpy array of (x, y) coordinates
    """
    # Construct the design matrix
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack((x**2 + y**2, x, y, np.ones_like(x)))
    
    # Construct the Q matrix
    Q = np.array([[0, 0, 0, -2],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [-2, 0, 0, 0]])
    
    # Solve the generalized eigenvalue problem A.T A P = η Q P
    M = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Q) @ M)
    
    # Find the eigenvector corresponding to the smallest non-negative eigenvalue
    min_eigenvalue_index = np.argmin(np.abs(eigenvalues))
    P = eigenvectors[:, min_eigenvalue_index]
    
    # Extract the parameters
    A, B, C, D = P
    center_x = -B / (2 * A)
    center_y = -C / (2 * A)
    radius = np.sqrt((B**2 + C**2 - 4 * A * D) / (4 * A**2))
    
    return center_x, center_y, radius

qulv_x=[]
qulv_y=[]
def calculate_slope(points):
    """
    Calculate the slope at each point in a set of 2D points using matrix operations.

    Parameters:
    points: numpy.ndarray
        A Nx2 array of (x, y) coordinates.

    Returns:
    slopes: numpy.ndarray
        An array of slopes corresponding to each point.
    """
    points = np.asarray(points)
    if points.shape[1] != 2:
        raise ValueError("Input points must be a Nx2 array of (x, y) coordinates.")

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Calculate differences between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)

    # Compute slopes using central differences for interior points
    slopes = np.zeros_like(x)
    slopes[1:-1] = (dy[:-1] + dy[1:]) / (dx[:-1] + dx[1:])

    # Forward difference for the first point
    slopes[0] = dy[0] / dx[0]

    # Backward difference for the last point
    slopes[-1] = dy[-1] / dx[-1]

    return slopes
for i in range(20, len(points) - 20):
    # center_x, center_y, radius = fit_circle_arc(points[i-10:i+10])
    # if(radius>100 or radius<2):
    #     continue
    qulv_x.append(points[i][0])
    data_use=1/((1/calculate_slope(points[i-10:i+10])[0])**2+1)
    if(data_use==0 or abs(data_use)>100):
        continue
    qulv_y.append((data_use))  # Ensure qulv_y is a list of floats, not arrays
    # print(f"Center: ({center_x:.3f}, {center_y:.3f}), Radius: {radius:.3f}")

# Ensure qulv_x and qulv_y have the same length
min_length = min(len(qulv_x), len(qulv_y))
qulv_x = qulv_x[:min_length]
qulv_y = qulv_y[:min_length]

# plt show qulv
# points_=smooth_points(qulv_x, smoothing_factor=0.5)
plt.scatter(qulv_x, qulv_y, s=0.1)
plt.xlabel("x") 
plt.ylabel("qulv")
# 自动调整坐标轴范围
plt.axis('auto')
plt.show()

use_data=np.stack((qulv_x,qulv_y),axis=-1)
use_data=smooth_points(use_data, smoothing_factor=1)

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

points11=np.stack((np.ones(len(qulv_x)),qulv_x,qulv_y),axis=-1)
from RegionSelector import CoordinateSelector
CoordinateSelector(points11)