import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
import time

from sklearn.linear_model import RANSACRegressor
from sklearn.base import BaseEstimator, RegressorMixin


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
from UserInterface.RegionSelector import CoordinateSelector
from PointCloud.segment import  PointsSegment

class PointCloudBase:
    def __init__(self, path=None):

        # load data/info.yml
        info_path = os.path.join(absolute_path, "data/info.yml")
            
        try:
            # 尝试加载 YAML 文件
            with open(info_path, 'r', encoding='utf-8') as f:
                self.info = yaml.safe_load(f)
                print("成功加载 info.yml 文件")
        except FileNotFoundError:
            # 文件不存在的异常处理
            print(f"错误：文件 {info_path} 不存在，请检查路径是否正确。")
            self.info = None  # 设置为默认值或空值
        # 转换固有坐标倍率系数
        if(path is not None):
            self.read(path)
            # self.pcd = o3d.io.read_point_cloud(path)
            # self.points = np.asarray(self.pcd.points)
            # self.points[:,1]/=self.info["collection_speed"]/self.info["collection_speed"]
            # self.pcd.points = o3d.utility.Vector3dVector(self.points)
    def read(self,path):
        print("read",path)
        self.pcd = o3d.io.read_point_cloud(path)
        self.points = np.asarray(self.pcd.points)
        self.points[:,1]/=self.info["collection_speed"]/self.info["true_speed"]
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        # 给出 xyz min max
        self.x_minax = [np.min(self.points[:,0]),np.max(self.points[:,0])]
        self.y_minax = [np.min(self.points[:,1]),np.max(self.points[:,1])]
        self.z_minax = [np.min(self.points[:,2]),np.max(self.points[:,2])]
    def show_points(self,points):
                # 创建Open3D点云对象
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # 可视化
        o3d.visualization.draw_geometries([point_cloud])
    
    def show_points_2d(self,points,ii,jj):
        plt.scatter(points[:,ii],points[:,jj],s=0.1)
        # 设置xy比例相同
        plt.axis('equal')
        plt.show()
    @ staticmethod
    def seg_xy(points,model=[1,1,1]):
        selector = CoordinateSelector(points,0,1)
        selector.show_touying_and_choose()
        PointsSegment_=PointsSegment(points=points,x_range=selector.x_regions,y_range=selector.y_regions,use_axis=selector.axis,model=model)
        return PointsSegment_.get_points(show=False)
    @ staticmethod
    def seg_xz(points,model=[1,1,1]):
        selector = CoordinateSelector(points,0,2)
        selector.show_touying_and_choose()
        PointsSegment_=PointsSegment(points=points,x_range=selector.x_regions,z_range=selector.y_regions,use_axis=selector.axis,model=model)
        return PointsSegment_.get_points(show=False)
    @ staticmethod
    def seg_yz(points,model=[1,1,1]):
        selector = CoordinateSelector(points,1,2)
        selector.show_touying_and_choose()
        PointsSegment_=PointsSegment(points=points,x_range=selector.x_regions,z_range=selector.y_regions,use_axis=selector.axis,model=model)
        return PointsSegment_.get_points(show=False)

    def seg_xy_self(self,model=[1,1,1]):
        selector = CoordinateSelector(self.points,0,1)
        selector.show_touying_and_choose()
        # print ("Axis_____",selector.axis)
        PointsSegment_=PointsSegment(points=self.points,x_range=selector.x_regions,y_range=selector.y_regions,use_axis=selector.axis,model=model)
        points_use=PointsSegment_.get_points(show=True)
        self.points=points_use
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        return (selector.x_regions,selector.y_regions,selector.axis),points_use

    def seg_xz_self(self,model=[1,1,1]):
        selector = CoordinateSelector(self.points,0,2)
        selector.show_touying_and_choose()
        PointsSegment_=PointsSegment(points=self.points,x_range=selector.x_regions,z_range=selector.y_regions,use_axis=selector.axis,model=model)
        points_use=PointsSegment_.get_points(show=True)
        self.points=points_use
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        return (selector.x_regions,selector.y_regions,selector.axis),points_use

    def seg_yz_self(self,model=[1,1,1]):
        selector = CoordinateSelector(self.points,1,2)
        selector.show_touying_and_choose()
        PointsSegment_=PointsSegment(points=self.points,y_range=selector.x_regions,z_range=selector.y_regions,use_axis=selector.axis,model=model)
        points_use=PointsSegment_.get_points(show=True)
        self.points=points_use
        self.pcd.points = o3d.utility.Vector3dVector(self.points)
        return (selector.x_regions,selector.y_regions,selector.axis),points_use
    
    def seg_z_self_scale(self,model=[1,1,1],ranges=None): 
        for i in range(len(ranges)):
            # 确保 ranges[i] 是一个范围列表
            # range_ =self.z_minax
            if isinstance(ranges[i], (list, tuple)) and len(ranges[i]) == 2:
                # print(i,ranges[i],self.z_minax,self.z_minax[1] - self.z_minax[0])
                # ranges=float(ranges)
                # 0 [0, 0.5] [-65.580002, 0.0] 65.580002
                ranges[i][0] = ranges[i][0] * (self.z_minax[1] - self.z_minax[0]) + self.z_minax[0]
                ranges[i][1] = ranges[i][1] * (self.z_minax[1] - self.z_minax[0]) + self.z_minax[0]
            else:
                raise TypeError(f"ranges[{i}] must be a list or tuple of two elements")

        PointsSegment_=PointsSegment(points=self.points,z_range=ranges,use_axis=None,model=model)
        points_use=PointsSegment_.get_points(show=True)
        self.points=points_use
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

    def seg_y_self_scale(self,model=[1,1,1],ranges=None): 
        for i in range(len(ranges)):
            # 确保 ranges[i] 是一个范围列表
            # range_ =self.z_minax
            if isinstance(ranges[i], (list, tuple)) and len(ranges[i]) == 2:
                ranges[i][0] = ranges[i][0] * (self.y_minax[1] - self.y_minax[0]) + self.y_minax[0]
                ranges[i][1] = ranges[i][1] * (self.y_minax[1] - self.y_minax[0]) + self.y_minax[0]
            else:
                raise TypeError(f"ranges[{i}] must be a list or tuple of two elements")

        PointsSegment_=PointsSegment(points=self.points,y_range=ranges,use_axis=None,model=model)
        points_use=PointsSegment_.get_points(show=True)
        self.points=points_use
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

    def seg_x_self_scale(self,model=[1,1,1],ranges=None): 
        for i in range(len(ranges)):
            # 确保 ranges[i] 是一个范围列表
            # range_ =self.z_minax
            if isinstance(ranges[i], (list, tuple)) and len(ranges[i]) == 2:
                ranges[i][0] = ranges[i][0] * (self.x_minax[1] - self.x_minax[0]) + self.x_minax[0]
                ranges[i][1] = ranges[i][1] * (self.x_minax[1] - self.x_minax[0]) + self.x_minax[0]
            else:
                raise TypeError(f"ranges[{i}] must be a list or tuple of two elements")

        PointsSegment_=PointsSegment(points=self.points,x_range=ranges,use_axis=None,model=model)
        points_use=PointsSegment_.get_points(show=True)
        self.points=points_use
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

    def denoise(self,nb_neighbors=100,std_ratio=0.5):
        # 50 20 的参数
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        self.pcd=cl
        self.points = np.asarray(cl.points)

    def down_sample(self,voxel_size=0.1):
        self.down_sample_pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        self.down_sample_points = np.asarray(self.down_sample_pcd.points)

    def save(self,path):
        path1=os.path.join(self.info['save_path'], path)
        o3d.io.write_point_cloud(path1,self.pcd,write_ascii=True)

# test
if __name__ == "__main__":
    PointCloud=PointCloudBase("data/example/pian2.ply")
    PointCloud.seg_yz_self(model=[1,0,1])
    PointCloud.show_points(PointCloud.points)
    # PointCloud.denoise()
    # PointCloud.down_sample()