from lms_minimize_test import generate_partial_cylinder_points
import numpy as np
from PcaTest import visualize_point_cloud
if __name__ == "__main__":
    center = np.array([1, 40, 5])  # 圆柱的中心点
    axis = np.array([0, 0.3, 1])  # 圆柱的轴向量
    axis_source=axis/np.linalg.norm(axis)
    radius = 2.0  # 半径
    height = 3.0  # 高度
    angle_range = 50  # 圆柱面部分的角度范围，单位为度
    resolution = 40  # 点云的分辨率，越大点云越密集
    resolution_z=30
    resolution_theta=250
    noise_stddev = 0.001  # 高斯噪声的标准差


    # 生成圆柱面上的部分点云
    points = generate_partial_cylinder_points(center, axis, radius, height, angle_range, resolution_theta=resolution_theta,resolution_z=resolution_z, noise_stddev=noise_stddev)
    # 输出点的数量
    print(f"生成的点云数量: {points.shape[0]}")
    visualize_point_cloud(points)

# 使用ransc方法