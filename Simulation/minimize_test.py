import numpy as np
import open3d as o3d

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize

def align_to_x_axis(points, target_vector):
    # 定义目标 x 轴方向
    target_x = np.array([1, 0, 0])
    
    # 归一化目标向量
    target_vector = target_vector / np.linalg.norm(target_vector)
    a1=target_vector
    # 计算旋转轴（叉积）
    a2 = np.cross(target_vector, target_x)
    a3=np.cross(a2, target_vector)
    A=np.vstack([a1,a2,a3])
    # axis_norm = np.linalg.norm(rotation_axis)
    # print(a1,a2,a3)
    # print(A)
    return (A@points.T).T
    # return aligned_points
def align_to_axis(points, target_vector):
    # 定义目标 x 轴方向
    target_x = np.array([1, 0, 0])
    
    # 归一化目标向量
    target_vector = target_vector / np.linalg.norm(target_vector)
    a1=target_vector
    # 计算旋转轴（叉积）
    a2 = np.cross(target_vector, target_x)
    a3=np.cross(a2, target_vector)
    A=np.vstack([a1,a2,a3])
    # axis_norm = np.linalg.norm(rotation_axis)
    # print(a1,a2,a3)
    # print(A)
    return (np.linalg.inv(A)@points.T).T
    # return aligned_points

import numpy as np
import open3d as o3d

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
    points = align_to_axis(points,axis)+ center

    # points = align_to_axis(points,np.array([1,1,0]))
    # # points[:,1]*=0.6
    # points =align_to_x_axis(points,np.array([1,1,0]))
    # points =align_to_axis(points,axis)+ center
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


import numpy as np
from sklearn.decomposition import PCA
from enum import Enum
# from r_pca import R_pca
from scipy.spatial import ConvexHull
import cv2
# import numpy as np
# import cv2
import matplotlib.pyplot as plt

# from new_robopca import ROBPCA
import matplotlib.pyplot as plt
# 创建ROBPCA模型

class PCAMethod(Enum):
            ORDINARY_PCA = "Ordinary PCA"
            ROBUST_PCA = "Robust PCA"
class PcaAxis:
    # 将点集的轴对齐到 x 轴
    class PCAMethod(Enum):
            ORDINARY_PCA = "Ordinary PCA"
            ROBUST_PCA = "Robust PCA"
    def __init__(self):
        # self.model=PCAMethod.ORDINARY_PCA  # 原始PCA方法
        pass
    
    def align_to_x_axis(self, points, target_vector):
        # 定义目标 x 轴方向
        target_x = np.array([1, 0, 0])
        
        # 归一化目标向量
        target_vector = target_vector / np.linalg.norm(target_vector)
        a1 = target_vector
        
        # 计算旋转轴（叉积）
        a2 = np.cross(target_vector, target_x)
        a3 = np.cross(a2, target_vector)
        A = np.vstack([a1, a2, a3])
        
        # 对点云进行旋转
        return (A @ points.T).T
    
    def fit(self,points, model=PCAMethod.ORDINARY_PCA):
        """
        计算给定点集的主要轴

        :param points: 三维点集，形状为 (n_samples, 3)
        :return: 主轴的方向向量
        :model: PCA方法
        """
        center = np.mean(points, axis=0)
        centered_points = points - center

        if model==PCAMethod.ORDINARY_PCA:
        # 中心化数据
            pca = PCA(n_components=3)
            pca.fit(centered_points)
            # 返回PCA的特征向量
            return pca.components_    
        elif model==PCAMethod.ROBUST_PCA:
              # 拟合数据
            # rpca = R_pca(centered_points)
            # L, S = rpca.fit()
            # # 获取主成分
            # U, S, Vt = rpca.get_principal_components()
            # return Vt
            # pca_model = ROBPCA(n_components=3, k_min_var_explained=0.8, alpha=0.75)
            # pca_model = ROBPCA(n_components=3,alpha=0.8,final_MCD_step=False)
            pca_model = ROBPCA(n_components=3,alpha=0.5,final_MCD_step=True)
            # print(pca_model.components_)
            pca_model.fit(centered_points)
            pca_model.plot_outlier_map(centered_points)
            print(pca_model.components_)

            # 获取投影数据
            return pca_model.components_


    def map_points_to_image_and_find_contours(self,points,show=True):
        """
        Maps a set of points to a 1280x720 image, finds the contours, and displays them.

        Parameters:
            points (list of tuples): List of (x, y) coordinates representing points.

        Returns:
            contours (list of numpy arrays): Contours found in the image.
        """
        # Extract x and y coordinates
        x = points[:, 0]
        y = points[:, 1]

        # Normalize points to fit within 1280x720 resolution
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        x_img = (x_norm * 1279).astype(np.int32)
        y_img = (y_norm * 719).astype(np.int32)

        # Create a 720x1280 image and draw points
        image = np.zeros((720, 1280), dtype=np.uint8)
        image[y_img, x_img] = 255  # White points

        num_=0
        for i in range(1280):
             for j in range(720):
                if image[j,i]==255:
                    num_+=1

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(show):
        # Find contours
            # Draw contours on a copy of the image for visualization
            contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

            # Display the image with contours
            plt.figure(figsize=(12, 6))
            plt.imshow(contour_image)
            plt.title("Contours")
            plt.axis("off")
            plt.show()

        # 返回轮廓面积
        # print((contours))
        # return cv2.contourArea(contours)
        return num_


    # 从三个备选轴中筛选是圆柱轴的轴
    def get_axis(self, points, model=PCAMethod.ORDINARY_PCA):
        # 选择最长的轴作为圆柱轴
        axis_list= self.fit(points, model=model)
        min_counters_num=1000000000

        ans_axis=None
        for ax in axis_list:
            # print(ax)
            points_touying_3d=self.align_to_x_axis(points, ax)
            points_touying_2d=np.stack([points_touying_3d[:,1],points_touying_3d[:,2]],axis=1)
            counters_num=self.map_points_to_image_and_find_contours(points_touying_2d,show=False)
            if counters_num<min_counters_num:
                min_counters_num=counters_num
                ans_axis=ax
        return ans_axis
# 定义目标函数
# 定义目标函数
def cylinder_loss(params, points):
    """
    计算圆柱方程的损失值
    :param params: 参数数组 [px, py, pz, vx, vy, vz, r]
    :param points: 给定的点集 (N, 3)
    :return: 损失值（均方误差）
    """
    px, py, pz, vx, vy, vz, r = params
    p = np.array([px, py, pz])
    v = np.array([vx, vy, vz])
    v_norm = np.linalg.norm(v)

    # 防止方向向量为零
    if v_norm < 1e-8:
        return np.inf
    v = v / v_norm  # 归一化方向向量

    # 计算每个点到圆柱轴线的距离平方
    dists_squared = []
    for point in points:
        w = point - p
        proj = np.dot(w, v) * v  # 投影到轴线方向
        dist_vec = w - proj
        dist_squared = np.dot(dist_vec, dist_vec)
        dists_squared.append((dist_squared - r**2)**2)

    return np.mean(dists_squared)  # 返回均方误差

# 初始猜测值 [alpha, beta, x0, y0, z0]
# initial_guess = [0.7, 0.7, 0, 0, 0]

# 调用函数生成点云并可视化
if __name__ == "__main__":
    # 可调参数
    center = np.array([0, 0, 0])  # 圆柱的中心点
    axis = np.array([0, 0.3, 1])  # 圆柱的轴向量
    axis_source=axis/np.linalg.norm(axis)
    radius = 40.0  # 半径
    height = 2.0  # 高度
    angle_range = 200  # 圆柱面部分的角度范围，单位为度
    resolution = 50  # 点云的分辨率，越大点云越密集
    noise_stddev = 0.01  # 高斯噪声的标准差

    # 生成圆柱面上的部分点云
    points = generate_partial_cylinder_points(center, axis, radius, height, angle_range, resolution, noise_stddev)
    points = [[2, 0, 0], [0, 2, 0], [0, -2, 0], [2, 0, 4], [0, 2, 4], [0, -2, 4]]
    from py_cylinder_fitting import BestFitCylinder
    from skspatial.objects import Points
    best_fit_cylinder = BestFitCylinder(Points(points))
    print(best_fit_cylinder.point)
#     # 初始参数 [px, py, pz, vx, vy, vz, r]
#     init_params = np.array([0.1, -0.2, 0.1, 0, 0.2, 1.0, 39])

#     # 优化过程
#     result = minimize(
#         cylinder_loss,
#         init_params,
#         args=(points,),
#         method='BFGS',
#         options={'disp': True}
#     )

#     # 提取结果
#     optimized_params = result.x
#     px, py, pz, vx, vy, vz, r = optimized_params
#     print(f"优化后的圆柱参数:")
#     print(f"圆柱轴线上一点: ({px:.4f}, {py:.4f}, {pz:.4f})")
#     print(f"圆柱方向向量: ({vx:.4f}, {vy:.4f}, {vz:.4f})")
#     print(f"圆柱半径: {r:.4f}")
# #     a=points[:,0].T
#     b=points[:,1].T
#     c=points[:,2].T

    
#     print("c/a=",c/a)
# # plt show points[:,1] points[:,2]
#     plt.figure(figsize=(10, 10))
#     plt.scatter(b, c, s=0.1)
#     plt.xlabel("x")
#     plt.ylabel("y")
#     # 设置xy比例相同
#     plt.axis('equal')
#     plt.show()

#     # 转换为open3d点云
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)

#     # 去噪和降采样
#     pcd = pcd.voxel_down_sample(voxel_size=0.1)
#     #去噪
#     pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
#     points = np.asarray(pcd.points)
#     def error_function(params):
#         alpha, beta, x0, y0, z0, r = params
#         total_error = 0
#         for x, y, z in points:
#             y_scaled = y / beta
#             predicted_r2 = (x - x0)**2 + (y_scaled - y0)**2 + (z - z0)**2 - (x - x0)*alpha - (y_scaled - y0)*beta
#             total_error += (predicted_r2 - r**2)**2
#         return total_error
#         # 初始猜测值 [alpha, beta, x0, y0, z0, r]
#     def callback(params):
#         optimization_steps.append(params)
#         print(f"Current parameters: {params}")
#     initial_guess = [0.7, 0.7, 0, 0, 0, 40]

#     # 优化求解
#     result = minimize(error_function, initial_guess, method='BFGS')
#     alpha_opt, beta_opt, x0_opt, y0_opt, z0_opt, r_opt = result.x

#     print(f"Optimal alpha: {alpha_opt}")
#     print(f"Optimal beta: {beta_opt}")
#     print(f"Optimal center: ({x0_opt}, {y0_opt}, {z0_opt})")
#     print(f"Optimal radius: {r_opt}")
#     # show
#     # o3d.visualization.draw_geometries([pcd])

#     # PcaAxis_=PcaAxis()
#     # ax1=PcaAxis_.get_axis(points, model=PCAMethod.ORDINARY_PCA)
#     # ax1=PcaAxis_.get_axis(points, model=PCAMethod.ORDINARY_PCA)

#     # print(ax1, axis / np.linalg.norm(axis))
#     # ax2=PcaAxis_.get_axis(points, model=PCAMethod.ROBUST_PCA)
#     # 可视化点云
#     visualize_point_cloud(points)
# print(points)
# points=read ../points1.txt
