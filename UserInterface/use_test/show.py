import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

pcd = o3d.io.read_point_cloud("./1.ply")
# 去噪
# cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
o3d.visualization.draw_geometries([pcd])
