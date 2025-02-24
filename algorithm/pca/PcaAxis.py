import numpy as np
from sklearn.decomposition import PCA
from enum import Enum
from robpy.pca.robpca import ROBPCA
import cv2
import matplotlib.pyplot as plt
import pypcl_algorithms as pcl_algo

class PCAMethod(Enum):
    """PCA分析方法的枚举类"""
    ORDINARY_PCA = "普通PCA"  # 传统PCA方法
    ROBUST_PCA = "稳健PCA"    # 稳健PCA方法，对异常点不敏感
    FAXIAN_PCA= "使用法线的普通PCA"

class PcaAxis:
    """
    点云主轴分析类
    用于分析三维点云数据的主要方向，支持普通PCA和稳健PCA两种方法
    """
    
    def __init__(self):
        """初始化PcaAxis类"""
        pass
    
    def align_to_x_axis(self, points, target_vector):
        """
        将点集沿目标向量对齐到x轴方向
        
        参数:
            points: ndarray, 形状为(n, 3)的点云数据
            target_vector: ndarray, 目标方向向量
            
        返回:
            ndarray: 旋转变换后的点云数据，形状为(n, 3)
        """
        # 定义x轴方向向量
        target_x = np.array([1, 0, 0])
        
        # 归一化目标向量
        target_vector = target_vector / np.linalg.norm(target_vector)
        a1 = target_vector
        
        # 构建旋转矩阵的正交基
        a2 = np.cross(target_vector, target_x)
        if np.allclose(a2, 0):  # 处理与x轴平行的情况
            a2 = np.array([0, 1, 0])
        a2 = a2 / np.linalg.norm(a2)
        a3 = np.cross(a2, target_vector)
        
        # 构建旋转矩阵并进行变换
        rotation_matrix = np.vstack([a1, a2, a3])
        return (rotation_matrix @ points.T).T
    
    def fit(self, points, model=PCAMethod.ORDINARY_PCA, alpha=0.5, final_MCD_step=False):
        """
        计算给定点集的主要轴
        
        参数:
            points: ndarray, 形状为(n, 3)的点云数据
            model: PCAMethod, PCA计算方法
            alpha: float, ROBUST_PCA的参数，表示期望保留的数据比例
            final_MCD_step: bool, 是否在ROBUST_PCA中使用最小协方差行列式步骤
            
        返回:
            普通PCA: 返回主成分方向向量
            稳健PCA: 返回ROBPCA模型对象
        """
        # 数据中心化


        if model == PCAMethod.ORDINARY_PCA:
            center = np.mean(points, axis=0)
            centered_points = points - center
            # 使用普通PCA方法
            pca = PCA(n_components=3)
            pca.fit(centered_points)
            return pca.components_
            
        elif model == PCAMethod.ROBUST_PCA:
            # 使用稳健PCA方法
            center = np.mean(points, axis=0)
            centered_points = points - center
            pca_model = ROBPCA(n_components=3, alpha=alpha, final_MCD_step=final_MCD_step)
            pca_model.fit(centered_points)
            return pca_model

        elif model == PCAMethod.FAXIAN_PCA:
            centered_points = pcl_algo.compute_normals(points,k_neighbors=30)
            # 使用法线的普通PCA方法
            pca = PCA(n_components=3)
            print(centered_points)
            pca.fit(centered_points)
            return pca.components_
    def map_points_to_image_and_find_contours(self, points, show=True):
        """
        将点集映射到图像并计算轮廓点数
        
        参数:
            points: ndarray, 形状为(n, 2)的二维点集
            show: bool, 是否显示轮廓图像
            
        返回:
            int: 轮廓中的点数
        """
        # 提取坐标
        x = points[:, 0]
        y = points[:, 1]

        # 归一化坐标到1280x720分辨率
        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())
        x_img = (x_norm * 1279).astype(np.int32)
        y_img = (y_norm * 719).astype(np.int32)

        # 创建图像并绘制点
        image = np.zeros((720, 1280), dtype=np.uint8)
        image[y_img, x_img] = 255  # 白色点

        # 计算点数
        point_count = np.sum(image == 255)

        if show:
            # 寻找并绘制轮廓
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

            # 显示结果
            plt.figure(figsize=(12, 6))
            plt.imshow(contour_image)
            plt.title("点云投影轮廓")
            plt.axis("off")
            plt.show()

        return point_count

    def get_axis(self, points, model=PCAMethod.ORDINARY_PCA):
        """
        从三个候选轴中选择最佳的圆柱轴
        
        参数:
            points: ndarray, 形状为(n, 3)的点云数据
            model: PCAMethod, PCA计算方法
            
        返回:
            ndarray: 最佳圆柱轴的方向向量
        """
        # 获取主成分轴
        axis_list = self.fit(points, model=model)
        min_points_count = float('inf')
        best_axis = None
        # if model == PCAMethod.FAXIAN_PCA:
        #     return axis_list[1]
        # 遍历每个主轴
        for axis in axis_list:
            # 归一化轴方向
            axis = axis / np.linalg.norm(axis)
            
            # 将点云对齐到当前轴
            aligned_points = self.align_to_x_axis(points, axis)
            
            # 投影到YZ平面
            projected_points = np.stack([aligned_points[:, 1], aligned_points[:, 2]], axis=1)
            
            # 计算投影后的点数
            points_count = self.map_points_to_image_and_find_contours(projected_points, show=False)
            
            # 更新最佳轴（选择投影后点数最少的轴作为圆柱轴）
            if points_count < min_points_count:
                min_points_count = points_count
                best_axis = axis

        return best_axis
