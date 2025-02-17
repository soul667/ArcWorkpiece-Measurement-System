import numpy as np
# import open3d as o3d

# import numpy as np
# from scipy.optimize import minimize

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
    # points = project_points_to_plane(points,axis,0)+ center
    noise = np.random.normal(scale=noise_stddev, size=points.shape)
    noisy_points = points + noise

    return noisy_points

# # 可视化点云
# def visualize_point_cloud(points):
#     # 创建Open3D点云对象
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points)

#     # 可视化
#     o3d.visualization.draw_geometries([point_cloud])
# axis_source= None

# # 调用函数生成点云并可视化
# if __name__ == "__main__":
#     # 可调参数
from py_cylinder_fitting import BestFitCylinder
from skspatial.objects import Points
center = np.array([0, 0, 0])  # 圆柱的中心点
axis = np.array([0, 0.3, 1])  # 圆柱的轴向量
axis_source=axis/np.linalg.norm(axis)
radius = 40.0  # 半径
height = 20.0  # 高度
angle_range = 360  # 圆柱面部分的角度范围，单位为度
resolution = 20  # 点云的分辨率，越大点云越密集
noise_stddev = 0.01  # 高斯噪声的标准差
# 生成圆柱面上的部分点云
points = generate_partial_cylinder_points(center, axis, radius, height, angle_range, resolution, noise_stddev)
# visualize_point_cloud(points)
# points = [[2, 0, 0], [0, 2, 0], [0, -2, 0], [2, 0, 4], [0, 2, 4], [0, -2, 4]]

best_fit_cylinder = BestFitCylinder(Points(points))

# 输出结果
print("Base center:", best_fit_cylinder.point)
print("Axis vector:", best_fit_cylinder.vector)
print("Radius:", best_fit_cylinder.radius)
print("Fitting error:", best_fit_cylinder.error)

