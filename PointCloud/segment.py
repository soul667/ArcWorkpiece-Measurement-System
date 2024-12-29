import numpy as np
from .transformation import align_to_x_axis,align_to_axis
# 输入一根轴和 x_range y_range z_range
class PointsSegment():
    def __init__(self, points, x_range=None, y_range=None, z_range=None,use_axis=None,model=None):
        self.points = np.array(points)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.use_axis = use_axis
        self.model = model if model is not None else [0,0,0]
        # print("model",model)
    # model like
    # [0,0,0] [1,1,1] 0 is < | >  1 is >= & <= 默认000

    def get_points(self,show=False):
        if self.use_axis is not None:
            # 先对齐到(1,0,0)
            self.points = align_to_x_axis(self.points, self.use_axis)
            if show:
                print("min,max x",np.min(self.points[:,0]),np.max(self.points[:,0]))    
                print("min,max y",np.min(self.points[:,1]),np.max(self.points[:,1]))    
                print("min,max z",np.min(self.points[:,2]),np.max(self.points[:,2]))    
        # 去掉(0,0,0)无用点
        filtered_points = self.points[
            ~(np.all(self.points == 0, axis=1))
        ]
        if show:
            print(f"Initial number of points: {len(self.points)}")

        if self.x_range is not None:
            for x_range_ in self.x_range:
                temp1=min(x_range_[0],x_range_[1])
                temp2=max(x_range_[0],x_range_[1])
                x_range_=[temp1,temp2] # 保证x_range_是从小到大的
                if show:
                    print("xranges",x_range_)
                filtered_points = filtered_points[
                    (filtered_points[:, 0] < x_range_[0]) | (filtered_points[:, 0] > x_range_[1])
                ] if self.model[0]==0 else filtered_points[
                    (filtered_points[:, 0] >= x_range_[0]) & (filtered_points[:, 0] <= x_range_[1])
                ]
        if show:
            print(f"number of x_else points: {len(filtered_points)}")
        if self.y_range is not None:
            for y_range_ in self.y_range:
                temp1=min(y_range_[0],y_range_[1])
                temp2=max(y_range_[0],y_range_[1])
                y_range_=[temp1,temp2] # 保证x_range_是从小到大的
                if show:
                    print("yranges",y_range_)
                    print("y_mode",self.model[1])

                filtered_points = filtered_points[
                    (filtered_points[:, 1] < y_range_[0]) | (filtered_points[:, 1] > y_range_[1])
                ] if self.model[1]==0 else filtered_points[
                    (filtered_points[:, 1] >= y_range_[0]) & (filtered_points[:, 1] <= y_range_[1])
                ]
        if show:
            print(f"number of y_else points: {len(filtered_points)}")
        if self.z_range is not None:
            for z_range_ in self.z_range:
                temp1=min(z_range_[0],z_range_[1])
                temp2=max(z_range_[0],z_range_[1])
                z_range_=[temp1,temp2] # 保证x_ra
                if show:
                    print("zranges",z_range_)
                filtered_points = filtered_points[
                    (filtered_points[:, 2] < z_range_[0]) | (filtered_points[:, 2] > z_range_[1])
                ] if self.model[2]==0 else filtered_points[
                    (filtered_points[:, 2] >= z_range_[0]) & (filtered_points[:, 2] <= z_range_[1])
                ]

        if self.use_axis is not None:
            # 先对齐到(1,0,0)
            filtered_points = align_to_axis(filtered_points, self.use_axis)
        
        return filtered_points
    
