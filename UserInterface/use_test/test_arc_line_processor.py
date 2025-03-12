import numpy as np
import open3d as o3d
import os
from arc_line_processor import ArcLineProcessor

def generate_arc_points(x_offset, radius, center_y, center_z, start_angle, end_angle, num_points):
    """生成圆弧上的点
    
    Args:
        x_offset (float): x轴偏移量
        radius (float): 圆弧半径
        center_y (float): 圆心y坐标
        center_z (float): 圆心z坐标
        start_angle (float): 起始角度
        end_angle (float): 结束角度
        num_points (int): 点的数量
    
    Returns:
        tuple: (x, y, z) 坐标数组
    """
    t = np.linspace(start_angle, end_angle, num_points)
    x = np.full_like(t, x_offset)
    y = center_y + radius * np.cos(t)
    z = center_z + radius * np.sin(t)
    return x, y, z

def generate_test_point_cloud():
    """生成测试用的点云数据
    包含3条线，每条线由2-3个圆弧段组成，
    确保每条线段之间的连接点相同
    """
    points = []
    num_points = 2000  # 每段弧线上的点数
    
    # 第一条线: 两段圆弧
    x_offset = 0.0
    
    # 第一段: 半径1的圆弧
    x1, y1, z1 = generate_arc_points(x_offset, 1, 0, 0, 0, np.pi/2, num_points)
    points.extend(np.column_stack([x1, y1, z1]))
    
    # 获取第一段的终点作为第二段的起点
    end_y1, end_z1 = y1[-1], z1[-1]
    
    # 第二段: 半径2的圆弧，从第一段终点开始
    x2, y2, z2 = generate_arc_points(x_offset, 2, 1, 0, 0, np.pi/2, num_points)
    # 调整第二段使其起点与第一段终点重合
    y2 = y2 - (y2[0] - end_y1)
    z2 = z2 - (z2[0] - end_z1)
    points.extend(np.column_stack([x2, y2, z2]))
    
    # 第二条线: 三段圆弧
    x_offset = 5.0
    radius = 1.5
    t_range = (-np.pi/3, np.pi/3)
    
    # 第一段
    x3, y3, z3 = generate_arc_points(x_offset, radius, 0, 0, t_range[0], t_range[1], num_points)
    points.extend(np.column_stack([x3, y3, z3]))
    
    # 第二段，从第一段终点开始
    x4, y4, z4 = generate_arc_points(x_offset, radius, 2, 0, t_range[0], t_range[1], num_points)
    # 确保与前一段连续
    y4 = y4 - (y4[0] - y3[-1])
    z4 = z4 - (z4[0] - z3[-1])
    points.extend(np.column_stack([x4, y4, z4]))
    
    # 第三段，从第二段终点开始
    x5, y5, z5 = generate_arc_points(x_offset, radius, 4, 0, t_range[0], t_range[1], num_points)
    # 确保与前一段连续
    y5 = y5 - (y5[0] - y4[-1])
    z5 = z5 - (z5[0] - z4[-1])
    points.extend(np.column_stack([x5, y5, z5]))
    
    # 第三条线: 两段圆弧
    x_offset = 10.0
    
    # 第一段：半径3的圆弧
    x6, y6, z6 = generate_arc_points(x_offset, 3, 0, 0, 0, np.pi, num_points)
    points.extend(np.column_stack([x6, y6, z6]))
    
    # 第二段：半径1.5的圆弧，从第一段终点开始
    x7, y7, z7 = generate_arc_points(x_offset, 1.5, 0, 4, 0, np.pi, num_points)
    # 确保与前一段连续
    y7 = y7 - (y7[0] - y6[-1])
    z7 = z7 - (z7[0] - z6[-1])
    points.extend(np.column_stack([x7, y7, z7]))
    
    # 添加随机噪声
    points = np.array(points)
    noise = np.random.normal(0, 0.02, points.shape)
    # points += noise
    
    return points

def save_point_cloud(points, path):
    """保存点云到PLY文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)
    # show
    o3d.visualization.draw_geometries([pcd])

def main():
    # 生成测试点云
    points = generate_test_point_cloud()
    
    # 确保temp目录存在
    temp_dir = os.path.join("UserInterface/assets", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # 保存点云
    temp_ply_path = os.path.join(temp_dir, 'temp.ply')
    save_point_cloud(points, temp_ply_path)
    print(f"点云已保存到: {temp_ply_path}")
    
    # 使用处理器处理点云
    processor = ArcLineProcessor()
    processor.set_debug(True)  # 启用调试模式查看可视化结果
    
    if processor.load_point_cloud(ply_path=temp_ply_path):
        results = processor.process()
        
        print("\n处理完成!")
        print(f"共检测到{len(results)}条线")
        for i, line in enumerate(results):
            print(f"\n线{i}:")
            print(f"点数: {len(line['points'])}")
            print(f"圆弧段数: {len(line['segments'])}")
            for j, segment in enumerate(line['segments']):
                print(f"\n  圆弧{j}:")
                print(f"    圆心: ({segment['center'][0]:.3f}, {segment['center'][1]:.3f})")
                print(f"    半径: {segment['radius']:.3f}")
                print(f"    拟合误差: {segment['error']:.6f}")
    else:
        print("加载点云失败")

if __name__ == "__main__":
    main()
