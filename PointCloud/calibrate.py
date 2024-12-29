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

PointCloudCalibrate = PointCloudBase()
PointCloudCalibrate.read(PointCloudCalibrate.info['calibrate_points_path'])
PointCloudCalibrate.seg_yz_self(model=[1,1,1])
PointCloudCalibrate.seg_xy_self(model=[1,1,1])

# # 去噪
# PointCloudCalibrate.remove_noise()
# # 下采样
# PointCloudCalibrate.down_sample(voxel_size=0.1)

# show
PointCloudCalibrate.show_points(PointCloudCalibrate.points)
