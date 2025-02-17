import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import time

from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin

from RegionSelector import CoordinateSelector
class CircleModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        """拟合圆模型，计算圆心和半径"""
        data = np.column_stack((X, y))
        A = np.c_[2 * data[:, 0], 2 * data[:, 1], np.ones(data.shape[0])]
        b = data[:, 0]**2 + data[:, 1]**2
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        self.center_ = np.array([params[0], params[1]])
        self.radius_ = np.sqrt(params[2] + self.center_[0]**2 + self.center_[1]**2)
        return self

    def predict(self, X):
        """预测点到拟合圆的距离"""
        distances = np.linalg.norm(X - self.center_, axis=1)
        return distances

def custom_down_sample(pcd):
    points = np.asarray(pcd.points)  # 转换为 NumPy 数组
    
    # 使用 NumPy 按 x 值分组
    unique_x = np.unique(points[:, 0])  # 获取唯一的 x 值
    downsampled_points = []
    
    for x in unique_x:
        # 提取当前 x 对应的点并按 y 排序
        group = points[points[:, 0] == x]
        sorted_group = group[np.argsort(group[:, 1])]  # 按 y 排序
        downsampled_points.append(sorted_group[::1])  # 每隔 10 个点取一个
    
    # 转换为点云格式
    downsampled_pcd = o3d.geometry.PointCloud()
    downsampled_pcd.points = o3d.utility.Vector3dVector(np.vstack(downsampled_points))  # 合并为单个数组
    
    return downsampled_pcd

class PointCloud:
    def __init__(self, path):
        self.debug_cl=True

        self.start_time = time.time()
        self.pcd = o3d.io.read_point_cloud(path)
        self.points = np.asarray(self.pcd.points)
        # self.SourcePoints = np.asarray(self.pcd.points)
        # self.segmentAndDenoisePointCloud()
        # print(f"预处理点云消耗时间: {self.end_time1 - self.start_time:.2f} seconds")

    def segmentAndDenoisePointCloud(self,y_range_=None,z_range_=None):
        points = self.points  # 点云数据
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()

        x_ratio = (0.1, 0.9)  # 在 x 方向上保留 20%-80%
        # y_ratio = (0, 1)  # 在 y 方向上保留 10%-90%
        y_ratio = (0.2, 0.4)  # 在 y 方向上保留 10%-90%  # 这个地方可以改成手动输入
        z_ratio = (0, 0.53)  # 在 z 方向上保留 30%-70%

        x_range = (x_min + x_ratio[0] * (x_max - x_min), x_min + x_ratio[1] * (x_max - x_min))
        if(y_range_!=None ):
            y_range = y_range_
        else:
             y_range = (y_min + y_ratio[0] * (y_max - y_min), y_min + y_ratio[1] * (y_max - y_min))
        if(z_range_!=None ):
            z_range = z_range_
        else:
            z_range = (z_min + z_ratio[0] * (z_max - z_min), z_min + z_ratio[1] * (z_max - z_min))


        # # 过滤点云
        filtered_points = points[
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        ]
        for y_range_ in y_range:
            filtered_points = filtered_points[
            (filtered_points[:, 1] <= y_range_[0]) | (filtered_points[:, 1] >= y_range_[1])
        ]
        for z_range_ in z_range:
            filtered_points = filtered_points[
            (filtered_points[:, 2] <= z_range_[0])
        ]
        #         filtered_points = points[
        #     (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        #     ((points[:, 1] <= y_range[0]) | (points[:, 1] >= y_range[1])) &
        #     (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        # ]
        #将点云中反光部分直接切除
        filtered_points[:,1]/=0.8
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        voxel_down_pcd_for_axis1 = (cropped_pcd)
        # voxel_down_pcd_for_axis1 = custom_down_sample(cropped_pcd)


        self.start_time = time.time()
        cl, ind = voxel_down_pcd_for_axis1.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
        self.end_time1 = time.time()
        self.points = np.asarray(cl.points)
        cl=self.custom_down_sample_getx(cl)

        o3d.visualization.draw_geometries([cl])
        
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

# PointCloud_ = PointCloud(r"./11.ply")
PointCloud_ = PointCloud(r"../assets/temp/temp.ply")

# PointCloud_.show_touying_and_choose()
#选择去除的中间高光区域
# o3d.io.write_point_cloud("cl.ply", custom_down_sample(PointCloud_.pcd),write_ascii=True)
DelSelector = CoordinateSelector(PointCloud_.points,1,2)
print(len(np.array(PointCloud_.points)))
DelSelector.show_touying_and_choose()
# print(DelSelector.x_regions)
PointCloud_.segmentAndDenoisePointCloud(y_range_=DelSelector.x_regions,z_range_=DelSelector.y_regions)
#RANSC标定轴向
print(len(np.array(PointCloud_.points)))


# 选取测量区域
# 选取PointCloud_.points沿x  (0.4, 0.6)
# 计算点云范围的实际坐标
points = PointCloud_.points
x_min, x_max = points[:, 0].min(), points[:, 0].max()

# 根据比例计算 x 轴的实际范围
x_range = (x_min + 0.4 * (x_max - x_min), x_min + 0.6 * (x_max - x_min))

# 筛选点云数据
selected_points = points[(points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])]
MeasureSelector = CoordinateSelector(selected_points,1,2)

MeasureSelector.show_touying_and_choose()
celiang=y_range_=MeasureSelector.x_regions[0]
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


for group in PointCloud_.Allx:
    i=i+1
#   PointCloud_.PerformRansacOnLine(group[group[:,1]>celiang[0] & group[:,1]<celiang[1]] ,ii=i)
    celiang = np.array(celiang, dtype=group.dtype)  # 确保类型一致

    # PointCloud_.PerformRansacOnLine(
    #     group[(group[:, 1] > celiang[0]) & (group[:, 1] < celiang[1])],
    #     ii=i
    # )
    
    points_new=group[(group[:, 1] > celiang[0]) & (group[:, 1] < celiang[1])]
    # target_vector = -np.array([-0.999317,0.0206028,0.0306875])  # 新的目标向量
    # points = align_to_x_axis(points_new, target_vector)
    points =np.stack((points_new[:, 1], points_new[:, 2]),axis=-1)
    # points = np.stack([points_new[:, 1], points_new[:, 2]]).T
    center_x, center_y, radius = fit_circle_arc(points)
    if(radius>15 or radius<4):
        continue
    # plt show the points and the radius
    plt.figure(figsize=(10, 10))
    plt.scatter(points_new[:, 1], points_new[:, 2], s=0.1)
    circle = plt.Circle((center_x, center_y), radius, color="r", fill=False)
    plt.gca().add_artist(circle)
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.show()
    plt.savefig(f"./za/ans2/{i}{circle}.png")
    # print(f"圆心: ({center_x:.2f}, {center_y:.2f}), 半径: {radius:.2f}")
    # print(i)
    plt.cla()
#   break