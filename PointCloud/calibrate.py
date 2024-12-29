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
PointCloudCalibrate.read(PointCloudCalibrate.info['calibrate_points_path'])
PointCloudCalibrate.seg_yz_self(model=[1,1,1])
# PointCloudCalibrate.seg_xy_self(model=[1,1,1])
selector = CoordinateSelector(PointCloudCalibrate.points,0,1)
selector.show_touying_and_choose()
        # print ("Axis_____",selector.axis)
PointsSegment_=PointsSegment(points=PointCloudCalibrate.points,x_range=selector.x_regions,z_range=selector.y_regions,use_axis=selector.axis,model=[1,1,1])
PointCloudCalibrate.points=PointsSegment_.get_points(show=True)
PointCloudCalibrate.pcd.points = o3d.utility.Vector3dVector(PointCloudCalibrate.points)

# # 去噪
# PointCloudCalibrate.remove_noise()
# # 下采样
# PointCloudCalibrate.down_sample(voxel_size=0.1)

# show
# PointCloudCalibrate.show_points(PointCloudCalibrate.points)
PointCloudCalibrate.denoise(std_ratio=0.2)
# PointCloudCalibrate.show_points(PointCloudCalibrate.points)
PointCloudCalibrate.down_sample(voxel_size=0.1)
# PointCloudCalibrate.show_points(PointCloudCalibrate.down_sample_points)

# from robpy.pca.robpca import ROBPCA
from sklearn.decomposition import PCA

pca_model = PCA(n_components=3)

pca_model.fit(PointCloudCalibrate.points)
# pca_model.plot_outlier_map(centered_points)
print(pca_model.components_)

import matplotlib.pyplot as plt
# 保留第二和第三主成分，忽略 X 轴
oyz_projection = PointCloudCalibrate.down_sample_points[:, [0, 1]]
x = PointCloudCalibrate.down_sample_points[:, 0]
y = PointCloudCalibrate.down_sample_points[:, 1]
# # 绘制投影到 OYZ 平面的点云
# plt.figure(figsize=(8, 6))
# plt.scatter(x,y, s=1, c='blue', alpha=0.6)
# plt.scatter(0, 0, color='black', label="Origin")
# plt.show()

import cv2
x_norm = (x - x.min()) / (x.max() - x.min())
y_norm = (y - y.min()) / (y.max() - y.min())
x_img = (x_norm * 1279).astype(np.int32)
y_img = (y_norm * 719).astype(np.int32)

image = np.zeros((720, 1280), dtype=np.uint8)
image[y_img, x_img] = 255  # White points

num_=0
for i in range(1280):
        for j in range(720):
            if image[j,i]==255:
                num_+=1
# 查找所有轮廓，包括内部轮廓

# save img 
cv2.imwrite("test.jpg", image)
# 对image闭运算操作
kernel = np.ones((5,5),np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

# Display the image with contours
plt.figure(figsize=(12, 6))
plt.imshow(contour_image)
plt.title("Contours")
plt.axis("off")
plt.show()

# 将轮廓按照面积大小进行排序
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# # 画出最大的轮廓
# contour_image = image.copy()
# cv2.drawContours(contour_image, [contours[2]], -1, (0, 255, 0), 2)
# plt.figure(figsize=(12, 6))
# plt.imshow(contour_image)
# plt.title("Largest Contour")
            
line_points = np.squeeze(np.array(contours[0]))  # [[x, y], ...] -> [x, y]

    # 按 x 坐标排序并选取 [10%-90%] 的点
line_points = line_points[np.argsort(line_points[:, 0])]
line_points = line_points[int(len(line_points) * 0.1):int(len(line_points) * 0.9)]

    # 显示点
    # 确保原始图像是彩色的
if len(image.shape) == 2:  # 如果是灰度图像
        line_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
else:
        line_image = image.copy()

    # 在图像上绘制选定的点
for point in line_points:
        cv2.circle(line_image, tuple(point), 3, (0, 255, 0), -1)

# 显示图像
# plt.figure(figsize=(12, 6))
# plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
# plt.title("Line Points")
# plt.axis("off")
# plt.show()

# 将点恢复到真实点

# 求解line_points的斜率
x = line_points[:, 0]
y = line_points[:, 1]
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]
# 恢复到原始的斜率
xbeilv= 1280/(x.max() - x.min())
ybeilv= 720/(y.max() - y.min())

slope = slope / (ybeilv / xbeilv)

print(slope)
vect=np.array([np.sqrt(1 - slope ** 2),slope,0])
print(vect)
print(np.arctan(vect[1]/vect[0])/np.pi*180)

from PointCloud.transformation import align_to_x_axis,align_to_axis
PointCloudCalibrate.points /= vect[0]
points_=align_to_x_axis(PointCloudCalibrate.points,vect)
# PointCloudCalibrate.show_points(points_)
# plt show points_[:,0] and points_[:,1]
plt.figure(figsize=(8, 6))
plt.scatter(points_[:,0],points_[:,1], s=1, c='blue', alpha=0.6)
plt.axis('equal')
plt.show()
# x y轴相等比例