from sklearn.decomposition import PCA
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# 加载点云
plane_pcd = o3d.io.read_point_cloud("./test_plane.ply")
points = np.asarray(plane_pcd.points)
use_pcd = o3d.io.read_point_cloud("./use.ply")
use_points = np.asarray(use_pcd.points)

# PCA
center = points.mean(axis=0)
centered = points - center

pca = PCA(n_components=3).fit(centered)
n = pca.components_[2]

min_z_p=-400.425
use_center = points.mean(axis=0)
# use_centered = use_points - use_center
use_points = [
    use_point 
    for use_point in use_points 
    if np.sqrt(use_point.dot(n)) > min_z_p
]
# print(use_points)
use_centered=use_points
use_centered=np.array(use_centered)
# 构建 u, v
tmp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
u = tmp - np.dot(tmp, n) * n
u /= np.linalg.norm(u)
v = np.cross(n, u)
v /= np.linalg.norm(v)

# use_centered=[ for]
# 投影
x2d = use_centered.dot(u)
y2d = use_centered.dot(v)

# 绘制
plt.figure(figsize=(6,6))
# print()

# plt.scatter(x2d, y2d, s=2)
# plt.scatter(use_centered[0], use_centered[2], s=2)
# plt.gca().set_aspect('equal', 'box')
# plt.savefig("./touying1.png")

points_2d = np.column_stack((x2d, y2d))
# sorted_indices = np.argsort(points_2d[:, 1])

step=1000
y_max=np.max(y2d)
y_min=np.min(y2d)
range_step=(y_max-y_min)/step
# print(y_min,y_max,range)

use_range = np.arange(y_min, y_max,range_step )
bin_indices = np.searchsorted(use_range, y2d, side='right')
# np.savetxt("bin_indices.txt", bin_indices, fmt='%d')
# 分组（index -> 点集）
from collections import defaultdict
groups = defaultdict(list)
y_use=defaultdict(list)
#z_use=defaultdict(list)
#z_list=[]

index=0
for idx, x, y in zip(bin_indices,x2d,y2d):
    groups[idx].append(x)
    y_use[idx].append(y)
    z_=np.sqrt(use_points[index].dot(n))
    #z_use[idx].append(z_)
    #z_list.append(z_)
    # print(z_,index)
    index=index+1
    # z_=np.sqrt(use_.dot(n))

#z_min=np.min(z_list)
#z_max=np.max(z_list)


points_right_2d=[]
for i in range(1,step-1):
    # print(groups[i],y_use[i])
    x_=np.max(groups[i])
    y_=np.mean(y_use[i])
    # print(x_,y_)
    points_right_2d.append((x_,y_))
points_right_2d=np.array(points_right_2d)
plt.scatter(points_right_2d[:,0], points_right_2d[:,1], s=2)
plt.gca().set_aspect('equal', 'box')
plt.savefig("./touying_right.png")

# for groups_,y_use_ in zip(groups,y_use):
#     print(groups_)
# print(bin_indices)
import os
import sys
# 直接使用相对路径
project_root = "../"  # 相对于notebook的路径
# 获取绝对路径
absolute_path = os.path.abspath(project_root)
# 添加到 sys.path
if absolute_path not in sys.path:
    sys.path.append(absolute_path)
# 直接求解圆心和半径
from algorithm.circle_arc import CircleArc
arc = CircleArc()

arc.hyper_circle_fit(points_right_2d)
center = arc.first_param[0:2]
radius = arc.first_param[2]

print(center,radius)

r=np.sqrt((x2d-center[0])**2+(y2d-center[1])**2)
z=[(use_.dot(n)) for use_ in use_points]
# print(use_centered[1])
# print(r)
plt.figure(figsize=(6,6))
plt.scatter(r, z, s=2)
# plt.scatter(use_centered[0], use_centered[2], s=2)
plt.gca().set_aspect('equal', 'box')
plt.savefig("./2.png")
# use_points=np.array(use_points)
# poins_final=np.array(np.column_stack((r,z,use_points[:,1],use_points[:,0],use_points[:,2])))
# # poins_final=np.array(np.column_stack((r,z,use_points[:,1],use_points[:,0],use_points[:,2])))

# poins_final = np.array([
#     poins_final_
#     for poins_final_ in poins_final
#     if poins_final_[0] < 52.47
# ])

# plt.scatter(poins_final[:,0], poins_final[:,1], s=2)
# plt.gca().set_aspect('equal', 'box')
# plt.savefig("./3.png")

# # 将poins_final 按照后面的x分组  
# fenzu_step=1000
# x_values = poins_final[:, 2]  # 获取最后一列作为分组依据
# x_min, x_max = x_values.min(), x_values.max()
# step_value=(x_max-x_min)/fenzu_step
# np.savetxt("1.out",x_values)
# print(x_min,x_max,step_value)
# # 计算分组边界
# use_x_range = np.arange(x_min, x_max,step_value )
# np.savetxt("use_x_range.out",use_x_range)
# bin_indices_x = np.searchsorted(use_x_range, poins_final[:,2], side='right')
# np.savetxt("bin_indices_x.out",bin_indices_x)

# groups_x = defaultdict(list)
# #z_use=defaultdict(list)
# #z_list=[]

# index=0
# for idx, poins_final_ in zip(bin_indices_x,poins_final):
#     groups_x[idx].append((poins_final_[3],poins_final_[4]))

# np.savetxt("groups1.out",groups_x[1])

# x_vals = [pt[0] for pt in groups_x[5]]
# y_vals = [pt[1] for pt in groups_x[5]]

# plt.cla()
# plt.scatter(x_vals, y_vals, s=2)
# plt.gca().set_aspect('equal', 'box')
# plt.savefig("./4.png")

# arc.hyper_circle_fit(poins_final)
# center_small = arc.first_param[0:2]
# radius_small = arc.first_param[2]

# print(center_small,radius_small)


# 按照y分组
# use_points = np.column_stack((poins_final[:, 2], poins_final[:, 3], poins_final[:, 4]))
# use_points=np.array(use_points)
from UserInterface.arc_fitting_processor import ArcFittingProcessor
from UserInterface.point_cloud_grouper import PointCloudGrouper
groups=PointCloudGrouper.get_all_groups(use_points,'y')
index=0
radius_list=[]
radius_list_xyz=[]
for line_idx, line_points in enumerate(groups):
    index+=1
    # plt.cla()
    # plt.scatter(line_points[:,0], line_points[:,2], s=2)
    # plt.gca().set_aspect('equal', 'box')
    # plt.savefig(f"./img/{index}.png")
    # 计算Z和R
    x2d_ = line_points.dot(u)
    y2d_ = line_points.dot(v)
    # print(center)
    r=np.sqrt((x2d_-center[0])**2+(y2d_-center[1])**2)
    z=[(use_.dot(n)) for use_ in line_points]
    # z=line_points[:,2]
    arc.hyper_circle_fit(np.array(np.column_stack((r,z))))
    # center = arc.first_param[0:2]
    radius = arc.first_param[2]
    radius_list.append(radius)

    arc.hyper_circle_fit(np.array(np.column_stack((line_points[:,0],line_points[:,2]))))
    # center = arc.first_param[0:2]
    radius = arc.first_param[2]
    radius_list_xyz.append(radius)
    # plt.cla()
    # plt.scatter(r,z, s=2)
    # plt.gca().set_aspect('equal', 'box')
    # plt.savefig(f"./img/zr{index}.png")
    # np.savetxt(f"./txt/{index}.txt",line_points)
    # if index==100:
    #     break
    np.savetxt(f"./txt/r.txt",np.array(np.column_stack((radius_list,radius_list_xyz))))
print(np.median(np.array(radius_list_xyz)),np.median(np.array(radius_list)))
    