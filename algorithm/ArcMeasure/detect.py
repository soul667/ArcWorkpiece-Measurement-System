import sys
import os
import yaml  # 用于加载 YAML 文件
# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将相对路径转换为绝对路径
relative_path = "../../"  # 相对路径，假设 "../test" 是你想添加的路径
absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
# 添加到 sys.path
sys.path.append(absolute_path)

import open3d as o3d
import numpy as np

from PointCloud.base import PointCloudBase
from UserInterface.RegionSelector import CoordinateSelector
from PointCloud.segment import  PointsSegment


PointCloudCalibrate = PointCloudBase()
PointCloudCalibrate.read(PointCloudCalibrate.info['example_points_path'])

PointCloudCalibrate.seg_yz_self(model=[1,0,1])
PointCloudCalibrate.seg_xy_self(model=[1,1,1])
PointCloudCalibrate.seg_yz_self(model=[1,1,1])

PointCloudCalibrate.denoise(std_ratio=0.2)
# PointCloudCalibrate.show_points(PointCloudCalibrate.points)
PointCloudCalibrate.down_sample(voxel_size=0.1)

# 对降采样点云进行PCA 寻找主轴
PointCloudCalibrate.show_points(PointCloudCalibrate.down_sample_points)