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
PointCloudCalibrate.read("data/save/ans.ply")

from algorithm.pca.PcaAxis import *
PcaAxis_=PcaAxis()
# class PCAMethod(Enum):
#             ORDINARY_PCA = "Ordinary PCA"
#             ROBUST_PCA = "Robust PCA"
# ax1=PcaAxis_.get_axis(PointCloudCalibrate.down_sample_points, model=PCAMethod.ORDINARY_PCA)
ax1=np.array([-0.9996030926704407, 0.0020672173704952, 0.028095927089452744])
# 将点集的轴对齐到 x 轴
points = PcaAxis_.align_to_x_axis(PointCloudCalibrate.points, ax1)
# points =PointCloudCalibrate.points
PointCloudCalibrate.show_points_2d(points,1,2)

# PointCloudCalibrate.points=points
# save points to ply
# PointCloudCalibrate.save("ans.ply")