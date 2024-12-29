import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import time

from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from RegionSelector import CoordinateSelector

from PcaAxis import *

# def PointCloudBase:

#     def segmentPointCloudInside(points, x_range=None, y_range=None, z_range=None):
#         """
#         Filters the point cloud based on specified ranges for x, y, and z coordinates.

#         Parameters:
#             points (numpy.ndarray): Input point cloud as a 2D array of shape (N, 3).
#             x_range (list of tuple): List of (min, max) ranges for x-coordinate filtering.
#             y_range (list of tuple): List of (min, max) ranges for y-coordinate filtering.
#             z_range (list of tuple): List of (min, max) ranges for z-coordinate filtering.

#         Returns:
#             numpy.ndarray: Filtered point cloud.
#         """
#         filtered_points = np.asarray(points)
#         print(f"Initial number of points: {len(filtered_points)}")
#         # filtered_points = filtered_points[
#         #     ~(np.all(filtered_points == 0, axis=1))
#         # ]
#         filtered_points = filtered_points[
#         ~(np.all(filtered_points == 0, axis=1))
#         ]
#         if x_range is not None:
#             for x_range_ in x_range:
#                 print("xranges",x_range_)
#                 filtered_points = filtered_points[
#                     (filtered_points[:, 0] >= x_range_[0]) & (filtered_points[:, 0] <= x_range_[1])
#                 ]

#         if y_range is not None:
#             for y_range_ in y_range:
#                 print("yranges",y_range_)
#                 filtered_points = filtered_points[
#                     (filtered_points[:, 1] >= y_range_[0]) & (filtered_points[:, 1] <= y_range_[1])
#                 ]

#         if z_range is not None:
#             for z_range_ in z_range:
#                 print("zranges",z_range_)

#                 filtered_points = filtered_points[
#                     (filtered_points[:, 2] >= z_range_[0]) & (filtered_points[:, 2] <= z_range_[1])
#                 ]

#         # filtered_points = filtered_points[
#         #             ~(filtered_points[:, 0]==0 & filtered_points[:, 1]==0 & filtered_points[:, 2]==0)
#         #         ]
#         print(f"Number of points after filtering: {len(filtered_points)}")
#         return filtered_points


# def segmentPointCloud(points, x_range=None, y_range=None, z_range=None):
#     """
#     Filters the point cloud based on specified ranges for x, y, and z coordinates.

#     Parameters:
#         points (numpy.ndarray): Input point cloud as a 2D array of shape (N, 3).
#         x_range (list of tuple): List of (min, max) ranges for x-coordinate filtering.
#         y_range (list of tuple): List of (min, max) ranges for y-coordinate filtering.
#         z_range (list of tuple): List of (min, max) ranges for z-coordinate filtering.

#     Returns:
#         numpy.ndarray: Filtered point cloud.
#     """
#     filtered_points = np.asarray(points)
#     print(f"Initial number of points: {len(filtered_points)}")
#     # filtered_points = filtered_points[
#     #     ~(np.all(filtered_points == 0, axis=1))
#     # ]
#     filtered_points = filtered_points[
#     ~(np.all(filtered_points == 0, axis=1))
#     ]
#     if x_range is not None:
#         for x_range_ in x_range:
#             print("xranges",x_range_)
#             filtered_points = filtered_points[
#                 (filtered_points[:, 0] <= x_range_[0]) | (filtered_points[:, 0] >= x_range_[1])
#             ]

#     if y_range is not None:
#         for y_range_ in y_range:
#             print("yranges",y_range_)
#             filtered_points = filtered_points[
#                 (filtered_points[:, 1] <= y_range_[0]) | (filtered_points[:, 1] >= y_range_[1])
#             ]

#     if z_range is not None:
#         for z_range_ in z_range:
#             print("zranges",z_range_)

#             filtered_points = filtered_points[
#                 (filtered_points[:, 2] >= z_range_[0]) & (filtered_points[:, 2] <= z_range_[1])
#             ]

#     # filtered_points = filtered_points[
#     #             ~(filtered_points[:, 0]==0 & filtered_points[:, 1]==0 & filtered_points[:, 2]==0)
#     #         ]
#     print(f"Number of points after filtering: {len(filtered_points)}")
#     return filtered_points

class PointCloudBase:
    def __init__(self, path):
        self.pcd = o3d.io.read_point_cloud(path)
        self.points = np.asarray(self.pcd.points)

    def segmentAndDenoisePointCloud(self,y_range_=None,z_range_=None):
        points = self.points  # 点云数据
        filtered_points=segmentPointCloud(points,None, y_range_, z_range_)
        print(f"Initial number of points: {len(filtered_points)}")
        filtered_points[:,1]/=0.8
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        #对点云进行下采样
        cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
        # o3d.io.write_point_cloud("cropped_pcd1.ply", cl,write_ascii=True)
        o3d.visualization.draw_geometries([cl])
        self.pcd=cl
        self.points = np.asarray(cl.points)


        # o3d.visualization.draw_geometries([cl])
        
        # o3d.io.write_point_cloud("cl.ply", voxel_down_pcd_for_axis1,write_ascii=True)
    
    # 使用RANSC找出点云轴向(函数)
    def ExtractAxis(self):
        return (0.99,0.001,0.001)
    
    # 使用粗分割找出分割区域(函数)
    def ExtractSegment(self):
        return (1,(0,1))
    
    # 使用细分割找出分割区域(函数)  对于每一根线都使用ransc进行筛选特定半径的点
    def custom_down_sample_getx(self,pcd):
        points = np.asarray(pcd.points)  # 转换为 NumPy 数组
        
        # 使用 NumPy 按 x 值分组
        unique_x = np.unique(points[:, 0])  # 获取唯一的 x 值
        # 根据 x 坐标分组
        lines = {x: points[points[:, 0] == x] for x in unique_x}
        Allx = [line for line in lines.values()]
        downsampled_points = []
        Allx_new=[]
        for group in Allx:
            # 按 y 坐标排序
            sorted_group = group[np.argsort(group[:, 1])]  # 按 y 排序
            Allx_new.append(sorted_group[::1])  # 每隔 2 个点取一个

        
        # 转换为点云格式
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(np.vstack(Allx_new))  # 合并为单个数组
        self.Allx = Allx_new
        self.unique_x = unique_x
    
        # plt.figure(figsize=(10, 10))
        # plt.scatter(points[:, 1], points[:, 2], s=0.1)
        # plt.xlabel("x")
        # plt.ylabel("y")
        # # plt.show()
        # # save the figure in the current directory ./ans1+nowtime
        # plt.savefig(f"./ans1/{time.time()}.png")
        return downsampled_pcd

    # 对于每一根线进行RANSC去噪
    def PerformRansacOnLine(self,points,ii=0):
            # imshow points on the x-y plane
        x=points[:,1]
        y=points[:,2]
        # 将x y 保存成np.array
        num_points = len(points)
        points = np.array(points)
        # save points to a file(just x anf y)
        np.savetxt(f'./za/ans1/{ii}.txt',points)
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 1], points[:, 2], s=0.1)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.equal_aspect_ratio()
        # x y 相同比例
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        # save the figure in the current directory ./ans1+nowtime
        # plt.savefig(f"./za/ans1/{time.time()}.png")
        plt.savefig(f"./za/ans1/{ii}.png")


    def show_touying_and_choose(self):
        # 将点云投影到0xy平面上并且imshow并且转化为720*1280的图片，编写回调函数，点击鼠标左键选择点并输出点的坐标
        # imshow points on the x-y plane
        x=self.points[:,0]
        y=self.points[:,1]


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

# PointCloud_ = PointCloud(r"./50-20-1.ply")
import os

# 指定目录路径
import os

# import os

directory_path = r'C:\Users\guaoxiang\Desktop\data\angle15'
path1 = os.path.join(directory_path, r"1.ply")
# PointCloud_ = PointCloud(path1)

PointCloud_ = PointCloud(path1)

# PointCloud_.show_touying_and_choose()
#选择去除的中间高光区域
# o3d.io.write_point_cloud("cl.ply", custom_down_sample(PointCloud_.pcd),write_ascii=True)
DelSelector = CoordinateSelector(PointCloud_.points,1,2)
print(len(np.array(PointCloud_.points)))
DelSelector.show_touying_and_choose()
# print(DelSelector.x_regions)
PointCloud_.segmentAndDenoisePointCloud(y_range_=DelSelector.x_regions,z_range_=DelSelector.y_regions)
# 进行点云体素降采样
# voxel_down_pcd_for_axis1 =PointCloud_.pcd.voxel_down_sample(voxel_size=0.1)
voxel_down_pcd_for_axis1 =PointCloud_.pcd.voxel_down_sample(voxel_size=0.2)

# 继续对点云进行其他方法的平滑去噪

print(f"Number of points after voxel downsampling: {len(voxel_down_pcd_for_axis1.points)}")
# show the voxel_down_pcd_for_axis1
# # 选取PointCloud_.points沿x  (0.4, 0.6)
# # 计算点云范围的实际坐标


DelSelector_down = CoordinateSelector(np.array(voxel_down_pcd_for_axis1.points),0,1)
DelSelector_down.show_touying_and_choose()
voxel_down_pcd_for_axis1.points = o3d.utility.Vector3dVector(segmentPointCloudInside(np.array(voxel_down_pcd_for_axis1.points), DelSelector_down.x_regions, DelSelector_down.y_regions,None))
o3d.visualization.draw_geometries([voxel_down_pcd_for_axis1])

# voxel_down_pcd_for_axis1.points = o3d.utility.Vector3dVector(selected_points)
points2=np.array(voxel_down_pcd_for_axis1.points)

# points2 保留两位小数
# 将 points2 中的值保留两位小数
# points2 = np.round(points2, 2)
PcaAxis_=PcaAxis()
# ax1=PcaAxis_.get_axis(points2, model=PcaAxis.PCAMethod.ORDINARY_PCA)
# ax1=PcaAxis_.get_axis(points2, model=PCAMethod.ROBUST_PCA)
ax1=PcaAxis_.get_axis(points2, model=PCAMethod.ORDINARY_PCA)

# # ax2=PcaAxis_.get_axis(points, model=PCAMethod.ROBUST_PCA)
print(ax1)
# save np.array(voxel_down_pcd_for_axis1.points)
# np.savetxt(f'./points1.txt',np.array(voxel_down_pcd_for_axis1.points))

# savepoints1 to ply
# points1_pcd = o3d.geometry.PointCloud()
# points1_pcd.points = o3d.utility.Vector3dVector(points1)
#save points1
# o3d.io.write_point_cloud("points1.ply", points1_pcd,write_ascii=True)
#RANSC标定轴向
# print(len(np.array(PointCloud_.points)))


# # 选取测量区域
# # 选取PointCloud_.points沿x  (0.4, 0.6)
# # 计算点云范围的实际坐标
# points = PointCloud_.points
# x_min, x_max = points[:, 0].min(), points[:, 0].max()

# # 根据比例计算 x 轴的实际范围
# x_range = (x_min + 0.4 * (x_max - x_min), x_min + 0.6 * (x_max - x_min))

# # 筛选点云数据
# selected_points = points[(points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])]
# MeasureSelector = CoordinateSelector(selected_points,1,2)

# MeasureSelector.show_touying_and_choose()
# celiang=y_range_=MeasureSelector.x_regions[0]
# 对于ALLx应该对每一根线进行RANSC去噪
i=0
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


# for group in PointCloud_.Allx:
#     i=i+1
# #   PointCloud_.PerformRansacOnLine(group[group[:,1]>celiang[0] & group[:,1]<celiang[1]] ,ii=i)
#     celiang = np.array(celiang, dtype=group.dtype)  # 确保类型一致

#     # PointCloud_.PerformRansacOnLine(
#     #     group[(group[:, 1] > celiang[0]) & (group[:, 1] < celiang[1])],
#     #     ii=i
#     # )
    
#     points_new=group[(group[:, 1] > celiang[0]) & (group[:, 1] < celiang[1])]
#     # target_vector = -np.array([-0.999317,0.0206028,0.0306875])  # 新的目标向量
#     # points = align_to_x_axis(points_new, target_vector)
#     points =np.stack((points_new[:, 1], points_new[:, 2]),axis=-1)
#     # points = np.stack([points_new[:, 1], points_new[:, 2]]).T
#     center_x, center_y, radius = fit_circle_arc(points)
#     if(radius>15 or radius<4):
#         continue
#     # plt show the points and the radius
#     plt.figure(figsize=(10, 10))
#     plt.scatter(points_new[:, 1], points_new[:, 2], s=0.1)
#     circle = plt.Circle((center_x, center_y), radius, color="r", fill=False)
#     plt.gca().add_artist(circle)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     # 设置xy比例相同
#     plt.axis('equal')
#     # plt.show()
#     plt.savefig(f"./za/ans2/{i}{circle}.png")
#     # print(f"圆心: ({center_x:.2f}, {center_y:.2f}), 半径: {radius:.2f}")
#     # print(i)
#     plt.cla()
# #   break


# [ 0.99944034 -0.00298757 -0.03331778]
# [ 0.9992184   0.01726855 -0.03555815]
# [ 0.99900102 -0.00423695 -0.04448603]

# [ 0.99936528  0.01181418 -0.03360745]