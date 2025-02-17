import open3d as o3d
import numpy as np 

from enum import Enum
class SacModel(Enum):
    SACMODEL_PLANE = 0  # 平面模型
    SACMODEL_LINE = 1  # 线模型
    SACMODEL_CIRCLE2D = 2  # 二维圆形模型
    SACMODEL_CIRCLE3D = 3  # 三维圆形模型
    SACMODEL_SPHERE = 4  # 球体模型
    SACMODEL_CYLINDER = 5  # 圆柱体模型
    SACMODEL_CONE = 6  # 圆锥模型
    SACMODEL_TORUS = 7  # 圆环（环面）模型
    SACMODEL_PARALLEL_LINE = 8  # 平行线模型
    SACMODEL_PERPENDICULAR_PLANE = 9  # 垂直平面模型
    SACMODEL_PARALLEL_LINES = 10  # 多条平行线模型
    SACMODEL_NORMAL_PLANE = 11  # 具有法向量约束的平面模型
    SACMODEL_NORMAL_SPHERE = 12  # 具有法向量约束的球体模型
    SACMODEL_REGISTRATION = 13  # 点云配准模型
    SACMODEL_REGISTRATION_2D = 14  # 二维点云配准模型
    SACMODEL_PARALLEL_PLANE = 15  # 平行平面模型
    SACMODEL_NORMAL_PARALLEL_PLANE = 16  # 具有法向量约束的平行平面模型
    SACMODEL_STICK = 17  # 棍状物体模型
    SACMODEL_ELLIPSE3D = 18  # 三维椭圆模型

class SacMethod(Enum):
    SAC_RANSAC = 0  # 随机采样一致性（RANSAC）
    SAC_LMEDS = 1   # 最小中值估计（LMEDS）
    SAC_MSAC = 2    # 修正的随机采样一致性（MSAC）
    SAC_RRANSAC = 3 # 重采样随机采样一致性（RRANSAC）
    SAC_RMSAC = 4   # 修正的重采样随机采样一致性（RMSAC）
    SAC_MLESAC = 5  # 最大似然估计随机采样一致性（MLESAC）
    SAC_PROSAC = 6  # 基于优先排序的随机采样一致性（PROSAC）

class SACSegmentation:
    def __init__(self, random: bool = False):
        self.model_ = None
        self.sac_ = None
        self.model_type_ = -1
        self.method_type_ = 0
        self.threshold_ = 0.0
        self.optimize_coefficients_ = True
        self.radius_min_ = -np.inf
        self.radius_max_ = np.inf
        self.samples_radius_ = 0.0
        self.samples_radius_search_ = None
        self.eps_angle_ = 0.0
        self.axis_ = np.zeros(3)
        self.max_iterations_ = 50
        self.threads_ = -1
        self.probability_ = 0.99
        self.random_ = random

    def set_model_type(self, model_type: int):
        self.model_type_ = model_type

    def get_model_type(self) -> int:
        return self.model_type_

    def get_method(self):
        return self.sac_

    def get_model(self):
        return self.model_

    def set_method_type(self, method_type: int):
        self.method_type_ = method_type

    def get_method_type(self) -> int:
        return self.method_type_

    def set_distance_threshold(self, threshold: float):
        self.threshold_ = threshold

    def get_distance_threshold(self) -> float:
        return self.threshold_

    def set_max_iterations(self, max_iterations: int):
        self.max_iterations_ = max_iterations

    def get_max_iterations(self) -> int:
        return self.max_iterations_

    def set_probability(self, probability: float):
        self.probability_ = probability

    def get_probability(self) -> float:
        return self.probability_

    def set_number_of_threads(self, nr_threads: int = -1):
        self.threads_ = nr_threads

    def set_optimize_coefficients(self, optimize: bool):
        self.optimize_coefficients_ = optimize

    def get_optimize_coefficients(self) -> bool:
        return self.optimize_coefficients_

    def set_radius_limits(self, min_radius: float, max_radius: float):
        self.radius_min_ = min_radius
        self.radius_max_ = max_radius

    def get_radius_limits(self) -> tuple:
        return self.radius_min_, self.radius_max_

    def set_samples_max_dist(self, radius: float, search):
        self.samples_radius_ = radius
        self.samples_radius_search_ = search

    def get_samples_max_dist(self) -> float:
        return self.samples_radius_

    def set_axis(self, ax: np.ndarray):
        self.axis_ = ax

    def get_axis(self) -> np.ndarray:
        return self.axis_

    def set_eps_angle(self, ea: float):
        self.eps_angle_ = ea

    def get_eps_angle(self) -> float:
        return self.eps_angle_

    def segment(self, inliers: list, model_coefficients: list):
        raise NotImplementedError("This is an abstract base class.")

    def init_sac_model(self, model_type: int) -> bool:
        raise NotImplementedError("This is an abstract method.")

    def init_sac(self, method_type: int):
        raise NotImplementedError("This is an abstract method.")

    def get_class_name(self) -> str:
        return "SACSegmentation"

class SACSegmentationFromNormals(SACSegmentation):
    def __init__(self, random: bool = False):
        super().__init__(random)
        self.normals_ = None
        self.distance_weight_ = 0.1
        self.distance_from_origin_ = 0.0
        self.min_angle_ = 0.0
        self.max_angle_ = np.pi / 2

    def set_input_normals(self, normals: o3d.geometry.PointCloud):
        self.normals_ = normals

    def get_input_normals(self):
        return self.normals_

    def set_normal_distance_weight(self, distance_weight: float):
        self.distance_weight_ = distance_weight

    def get_normal_distance_weight(self) -> float:
        return self.distance_weight_

    def set_min_max_opening_angle(self, min_angle: float, max_angle: float):
        self.min_angle_ = min_angle
        self.max_angle_ = max_angle

    def get_min_max_opening_angle(self) -> tuple:
        return self.min_angle_, self.max_angle_

    def set_distance_from_origin(self, d: float):
        self.distance_from_origin_ = d

    def get_distance_from_origin(self) -> float:
        return self.distance_from_origin_

    def init_sac_model(self, model_type: int) -> bool:
        raise NotImplementedError("This is an abstract method.")

    def get_class_name(self) -> str:
        return "SACSegmentationFromNormals"
    
    def segment(self, inliers: list, model_coefficients: list):
        raise NotImplementedError("This is an abstract base class.")
    
# 直接继承类 SACSegmentationFromNormals
class RanscAxis(SACSegmentationFromNormals):
    def __init__(cloud,self):
        self.cloud = cloud
        # 计算点云法线
        self.normals = self.compute_normals(cloud) 
    # 使用 Open3D 计算点云的法线
    def compute_normals(cloud):    
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        return np.asarray(pcd.normals)


if __name__ == "__main__":
    num_points = 1000
    points = np.random.rand(num_points, 3) * 100
    ransac_axis = RanscAxis(points)
    ransac_axis.set_normal_distance_weight(0.1)
    # 拟合圆柱体
    ransac_axis.set_model_type(SacModel.SACMODEL_PLANE)
    ransac_axis.set_method_type(SacMethod.SAC_RANSAC)
    # 于设置分割算法是否在计算模型后优化模型系数。 true 参数 表示启用模型系数优化。
    ransac_axis.set_optimize_coefficients(True)
    # 设置最大迭代次数
    ransac_axis.set_max_iterations(1000)
    # 设置距离阈值
    ransac_axis.set_distance_threshold(0.2)
    # 设置半径范围
    ransac_axis.set_radius_limits(10, 50)