import numpy as np
from sklearn.decomposition import PCA
from enum import Enum
# from r_pca import R_pca
from robpy.pca.robpca import ROBPCA
from robpy.preprocessing import RobustScaler
from scipy.spatial import ConvexHull
import cv2
# import numpy as np
# import cv2
import matplotlib.pyplot as plt
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
    
    def fit(self,points, model=PCAMethod.ORDINARY_PCA,alpha=0.5,final_MCD_step=False):
        """
        计算给定点集的主要轴

        :param points: 三维点集，形状为 (n_samples, 3)
        :return: 主轴的方向向量
        :model: PCA方法
        """
        # print(points)
        center = np.mean(points, axis=0)
        centered_points = points - center

        if model==PCAMethod.ORDINARY_PCA:
        # 中心化数据
            pca = PCA(n_components=3)
            pca.fit(centered_points)
            # 返回PCA的特征向量
            return pca    
        elif model==PCAMethod.ROBUST_PCA:
              # 拟合数据
        #    scaled_data = RobustScaler(with_centering=True).fit_transform( centered_points)
           pca_model = ROBPCA(n_components=3,alpha=alpha,final_MCD_step=final_MCD_step)
           pca_model.fit(centered_points)
           print(pca_model.components_)
        #    pca_model.plot_outlier_map(centered_points)
            # print(pca_model.components_)
           return pca_model


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
            print(ax/np.linalg.norm(ax))
            points_touying_3d=self.align_to_x_axis(points, ax)
            points_touying_2d=np.stack([points_touying_3d[:,1],points_touying_3d[:,2]],axis=1)
            counters_num=self.map_points_to_image_and_find_contours(points_touying_2d,show=True)
            if counters_num<min_counters_num:
                min_counters_num=counters_num
                ans_axis=ax
        return ans_axis