import numpy as np
# 从target_vector对齐到x轴
def align_to_x_axis(points, target_vector):
        target_x = np.array([1, 0, 0])
        
        # 归一化目标向量
        target_vector = target_vector / np.linalg.norm(target_vector)
        a1 = target_vector
        
        # 计算旋转轴（叉积）
        a2 = np.cross(target_vector, target_x)
        a3 = np.cross(a2, target_vector)
        A = np.vstack([a1, a2, a3])
        print(a1, a2, a3,A)
        # 对点云进行旋转
        return (A @ points.T).T

#从(1,0,0)对齐到target_vector
def align_to_axis(points, target_vector):
    # 定义目标 x 轴方向
    target_x = np.array([1, 0, 0])
    
    # 归一化目标向量
    target_vector = target_vector / np.linalg.norm(target_vector)
    a1=target_vector
    # 计算旋转轴（叉积）
    a2 = np.cross(target_vector, target_x)
    a3=np.cross(a2, target_vector)
    A=np.vstack([a1,a2,a3])
    # axis_norm = np.linalg.norm(rotation_axis)
    print(a1,a2,a3)
    print(A)
    return (np.linalg.inv(A)@points.T).T