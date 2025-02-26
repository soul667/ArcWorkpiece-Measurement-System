import numpy as np

class AxisProjector:
    """将三维点云投影到与给定轴向量垂直的平面上"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def normalize(v):
        """标准化向量"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    
    @staticmethod
    def get_projection_basis(v):
        """在投影平面上构建正交基向量u,w，其法向量为v
        
        参数:
            v: 轴向量（将被标准化）
            
        返回:
            元组 (u,w)，包含两个正交单位基向量
        """
        n = AxisProjector.normalize(v)
        
        # Check if vector aligned with y-axis
        if n[0] == 0 and n[2] == 0:
            u = np.array([1.0, 0.0, 0.0]) 
            w = np.array([0.0, 0.0, -1.0])
        else:
            # Construct basis using cross products
            u = np.array([n[2], 0, -n[0]])
            u = AxisProjector.normalize(u)
            w = np.cross(n, u)
            
        return u, w
            
    def project_points(self, points, axis_vector, axis_point):
        """将三维点投影到垂直于轴向量的平面上
        
        参数:
            points: Nx3数组，包含N个三维点
            axis_vector: 定义投影方向的三维向量
            axis_point: 轴线经过的三维点
            
        返回:
            元组，包含:
            - projected_points: Nx3数组，投影后的三维点
            - planar_coords: Nx2数组，投影平面上的二维坐标
        """
        # Ensure inputs are numpy arrays
        points = np.asarray(points)
        v = np.asarray(axis_vector)
        x = np.asarray(axis_point)
        
        # 获取单位方向向量
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            raise ValueError("轴向量不能为零向量")
        
        # 将每个点投影到平面上
        # Q = P + t*v where t = ((x-P)·v)/(v·v)
        dots = np.dot(x - points, v)
        t = dots / np.dot(v,v)
        
        # 获取投影点
        projected = points + np.outer(t, v)
        
        # 获取投影平面中的基向量
        u, w = self.get_projection_basis(v)
        
        # 获取相对于轴点的位移向量
        d = projected - x
        
        # 通过投影到基向量上获得二维坐标
        planar_coords = np.column_stack([
            np.dot(d, u),
            np.dot(d, w)
        ])
        
        return projected, planar_coords

def test_axis_projector():
    """测试函数演示使用方法"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 创建样本点（例如：围绕圆柱的点）
    theta = np.linspace(0, 2*np.pi, 50)
    z = np.linspace(0, 5, 50)
    theta, z = np.meshgrid(theta, z)
    
    r = 2.0  # radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    # 定义任意轴
    axis_vector = np.array([1.0, 1.0, 1.0])  # 45度角
    axis_point = np.array([0.0, 0.0, 0.0])   # 经过原点
    
    # 创建投影器并获取结果
    projector = AxisProjector()
    projected_points, planar_coords = projector.project_points(points, axis_vector, axis_point)
    
    # 绘制结果
    fig = plt.figure(figsize=(12,5))
    
    # 3D图显示原始点和投影点
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='.', label='Original')
    ax1.scatter(projected_points[:,0], projected_points[:,1], projected_points[:,2], 
                c='r', marker='.', label='Projected')
    
    # 绘制轴线
    t = np.linspace(-5, 5, 2)
    axis_line = np.outer(t, axis_vector) + axis_point
    ax1.plot(axis_line[:,0], axis_line[:,1], axis_line[:,2], 'g-', label='Axis')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # 2D投影图
    ax2 = fig.add_subplot(122)
    ax2.scatter(planar_coords[:,0], planar_coords[:,1], c='r', marker='.')
    ax2.set_xlabel('u')
    ax2.set_ylabel('w')
    ax2.axis('equal')
    ax2.grid(True)
    ax2.set_title('2D Projection')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_axis_projector()
