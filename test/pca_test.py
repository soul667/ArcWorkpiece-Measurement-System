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

import open3d as o3d
import numpy as np

from PointCloud.base import PointCloudBase
from UserInterface.RegionSelector import CoordinateSelector
from PointCloud.segment import  PointsSegment

PointCloudCalibrate = PointCloudBase()
# PointCloudCalibrate.read(PointCloudCalibrate.info['example_points_path'])
PointCloudCalibrate.read("../data/example/1.ply")

# print(PointCloudCalibrate.z_minax[1]-PointCloudCalibrate.z_minax[0])


# PointCloudCalibrate.show_points(PointCloudCalibrate.points)

PointCloudCalibrate.seg_yz_self(model=[1,1,1])
PointCloudCalibrate.denoise(nb_neighbors=120,std_ratio=0.5)
# o3d.draw_geometries([PointCloudCalibrate.points])
PointCloudCalibrate.show_points(PointCloudCalibrate.points)

# # PointCloudCalibrate.show_points_2d(PointCloudCalibrate.points,1,2)

# PointCloudCalibrate.seg_xy_self(model=[1,1,1])

# PointCloudCalibrate.seg_yz_self(model=[1,1,1])
# PointCloudCalibrate.seg_xy_self(model=[1,1,1])


# # # PCA
# # PointCloudCalibrate.down_sample(voxel_size=0.1)
# # PointCloudCalibrate.show_points(PointCloudCalibrate.down_sample_points)

# from algorithm.pca.PcaAxis import *
# PcaAxis_=PcaAxis()
# # class PCAMethod(Enum):
# #             ORDINARY_PCA = "Ordinary PCA"
# #             ROBUST_PCA = "Robust PCA"
# # ax1=PcaAxis_.get_axis(PointCloudCalibrate.down_sample_points, model=PCAMethod.ORDINARY_PCA)
# # ax1=np.array([0.99933844,0.0085073,-0.03535967])
# # 将点集的轴对齐到 x 轴
# # points = PcaAxis_.align_to_x_axis(PointCloudCalibrate.points, ax1)
# points =PointCloudCalibrate.points
# PointCloudCalibrate.show_points_2d(points,1,2)

# PointCloudCalibrate.points=points
# # save points to ply
# PointCloudCalibrate.save("ans.ply")