import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# 将相对路径转换为绝对路径
relative_path = "../"  # 相对路径，假设 "../test" 是你想添加的路径
absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
# 添加到 sys.path
sys.path.append(absolute_path)
from algorithm.circle_arc import CircleArc
# from algorithm.Ransc_circle import CircleFitter



class PreciseArcFitting():
    def __init__(self,points,axis,denoise=True):
        points=points[points[:,0].argsort()]
        self.points=points # 已经排序过的点
        self.source_points=points
        self.axis=axis
        self.arc = CircleArc()
        # self.fitter=CircleFitter()
        self.project_points_to_plane()
        if denoise:
            self.denoise()
        # self.points[:,0]=self.source_points[:,0] #转换x轴
        

    def project_points_to_plane(self):

        if self.axis is None:
            return self.points
        
        normal=self.axis
        points=self.points
        # 确保输入是numpy数组
        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        normal = np.asarray(normal, dtype=np.float64)
        
        # 计算点积
        dot_product = np.dot(points, normal)
        
        # 计算分母
        denominator = np.dot(normal, normal)
        
        # 计算投影向量
        projection_vector = (dot_product / denominator)[:, np.newaxis] * normal
        
        # 计算投影点
        P_prime = points - projection_vector
        self.points=P_prime
        return P_prime
    def denoise(self,nb_neighbors=100,std_ratio=0.02):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(self.points)
        # 半径滤波，设置半径和最低点数
        radius = std_ratio  # 搜索半径
        min_neighbors = nb_neighbors  # 半径内的最少点数
        filtered_cloud, indices = point_cloud.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
        # cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        self.points=np.array(filtered_cloud.points)
        self.points[:, 0] = self.source_points[indices, 0]
        # return points
        return self.points
    def fit_basic(self):
        # poins=
        self.arc.fit_circle_arc(self.points[:,1:3])
        center = self.arc.first_param[0:2]
        radius = self.arc.first_param[2]
        return (center,radius)
    
    # 每根线进行拟合 并且剔除掉异常值
    # True 的时候会显示一个箱式图
    def save_pic(self,points_use,i):
        plt.scatter(points_use[:, 0], points_use[:, 1], s=0.5)
        # 保存图像
        plt.axis('equal')
        plt.savefig(f'../data/temp/fenge_use_radius_{i}.png')

        # set xy equal
        # set x y equal
        plt.clf()  # 清除当前图形，以便在下一次迭代中重新绘制

    def ransc(self):
    
        pass

    def fit(self,show=True):
        now_x=-128910
        points_temp=[]
        len_temp_use=1
        radius_list=[]
        img_points_use=[]
        i=0
        for point in self.points:
            if point[0]!=now_x:
                if(len_temp_use>1):
                    # print("points_temp",points_temp)
                    points_temp=np.array(points_temp)
                    # 先使用RANSC进行分割去噪
                    # fitted_params, points_inner = self.fitter.fit_circle_ransac(points_temp, inlier_ratio=0.9, residual_threshold=0.01)
                    # print(len(points_inner),i)
                    if 1:
                    # if fitted_params:
                    #     if(show):
                    #             self.fitter.plot_circle(fitted_params[:2], fitted_params[2], points_temp, points_inner)
                    # else:
                    #     continue
                        self.arc.fit_circle_arc(points_temp)
                        center =  self.arc.first_param[0:2]
                        radius =  self.arc.first_param[2]
                        radius_list.append(radius)
                        if(show):
                            self.save_pic(points_temp,radius)
                    i=i+1
                    # points_temp[:,0]+=i*3
                    # img_points_use.append(points_temp)
                    # print("center",center)
                    # print("radius",radius)
                now_x=point[0]
                points_temp=[]
                points_temp.append(point[1:3])

            else:
                points_temp.append(point[1:3])
                len_temp_use+=1

        # if(show):
        #     for i in img_points_use:
        #         filtered_points = i[i[:, 0] < 0.7]
        #         plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=0.5)
        #     # 保存图像
            # plt.savefig(f'../data/temp/use.png')
        radii_median = np.median(radius_list)
        threshold = 0.4
        filtered_data = np.array(radius_list)[np.abs(np.array(radius_list) - radii_median) <= threshold]
        # print(np.mean(filtered_data),np.median(filtered_data))
        if(show):
            plt.boxplot(filtered_data, vert=False, patch_artist=True)
            plt.xlabel('$R$')
            plt.title('Boxplot of R Measurement')
            plt.show()
        
        # 返回平均值 中位数 和过滤后的数据
        return np.mean(filtered_data),np.median(filtered_data),filtered_data
    


if __name__ == "__main__":
    pcd1 = o3d.io.read_point_cloud("../data/save/ans2.ply")
    cl, ind = pcd1.remove_statistical_outlier(nb_neighbors=200, std_ratio=0.01)
    points=np.array(cl.points)
    # points = np.asarray(pcd1.points)
    # show use open3d
    o3d.visualization.draw_geometries([cl])

    # axis=np.array([0.99959731,-0.00212741,-0.02829423])
    axis= np.array([ 0.99977154,0.00326363,0.02112524])


    arc_fitting=PreciseArcFitting(points,axis)
    data=arc_fitting.fit()
    print(data[0],data[1])