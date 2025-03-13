import numpy as np
import open3d as o3d

def generate_cylinder_points(radius=10, height=50, angle=60, n_points_per_circle=100, n_layers=500):
    """
    生成圆柱体点云
    
    参数:
        radius: 圆柱体半径
        height: 圆柱体高度
        angle: 圆心角（度）
        n_points_per_circle: 每个圆周上的点数
        n_layers: 圆柱体的层数
    """
    # 将角度转换为弧度
    angle_rad = np.deg2rad(angle)
    
    # 生成角度序列
    theta = np.linspace(0, angle_rad, n_points_per_circle)
    
    # 生成高度序列
    z = np.linspace(0, height, n_layers)
    
    # 创建网格点
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # 计算x,y,z坐标
    y = radius * np.cos(theta_grid)
    z = radius * np.sin(theta_grid)
    x = z_grid
    
    # 将坐标组合成点云
    points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd

if __name__ == "__main__":
    # 生成点云
    pcd = generate_cylinder_points()
    
    # 设置点云颜色（可选）
    pcd.paint_uniform_color([0.5, 0.5, 1.0])  # 蓝色
    
    # 显示点云
    o3d.visualization.draw_geometries([pcd])
    
    # 保存点云
    o3d.io.write_point_cloud("test_cylinder.ply", pcd)
    print("点云已保存为 'test_cylinder.ply'")
