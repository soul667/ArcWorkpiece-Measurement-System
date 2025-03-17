import numpy as np
from typing import List

def generate_cylinder_points(
    point_count: int = 1000, 
    radius: float = 0.5, 
    height: float = 2.0, 
    noise_std: float = 0.01, 
    arc_angle: float = 360.0,
    axis_direction: List[float] = [0, 0, 1]
) -> np.ndarray:
    """生成圆柱体点云数据
    
    Args:
        point_count: 点云数量
        radius: 圆柱体半径
        height: 圆柱体高度 
        noise_std: 噪声标准差
        arc_angle: 圆心角(度)
        axis_direction: 圆柱体轴向方向
        
    Returns:
        points: 生成的圆柱体点云，numpy数组(N,3) 
    """
    # 归一化轴向向量
    axis = np.array(axis_direction)
    axis = axis / np.linalg.norm(axis)
    
    # 创建旋转矩阵，将[0,0,1]对齐到目标轴向
    if np.allclose(axis, [0, 0, 1]):
        R = np.eye(3)
    else:
        # 计算旋转轴
        rot_axis = np.cross([0, 0, 1], axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        
        # 计算旋转角度
        cos_angle = np.dot([0, 0, 1], axis)
        angle = np.arccos(cos_angle)
        
        # Rodriguez旋转公式
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                     [rot_axis[2], 0, -rot_axis[0]],
                     [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * K.dot(K)
    
    # 计算弧度
    arc_rad = np.deg2rad(arc_angle)
    
    # 生成点云
    thetas = np.random.uniform(0, arc_rad, point_count)
    heights = np.random.uniform(-height/2, height/2, point_count)
    
    # 生成圆柱面上的点
    x = radius * np.cos(thetas)
    y = radius * np.sin(thetas)
    z = heights
    
    # 合并为点云数组
    points = np.column_stack([x, y, z])
    
    # 旋转点云以对齐目标轴向
    points = points.dot(R.T)
    
    # 添加随机噪声
    noise = np.random.normal(0, noise_std, (point_count, 3))
    points += noise
    
    return points
