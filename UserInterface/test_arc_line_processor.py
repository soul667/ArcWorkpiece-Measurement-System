import numpy as np
import open3d as o3d
import os
from arc_line_processor import ArcLineProcessor

def generate_test_point_cloud():
    """生成测试用的点云数据
    包含3条线，每条线由2-3个圆弧段组成
    """
    points = []
    
    # 增加点密度
    t = np.linspace(0, np.pi/2, 200)  # 增加到200个点
    
    # 第一条线: 两段圆弧
    # 第一段: 半径1的圆弧
    x1 = np.full_like(t, 0.0)  # 固定x坐标，使用full_like更明确
    y1 = np.cos(t)
    z1 = np.sin(t)
    points.extend(np.column_stack([x1, y1, z1]))
    
    # 第二段: 半径2的圆弧
    t2 = np.linspace(0, np.pi/2, 200)
    x2 = np.full_like(t2, 0.0)
    y2 = -2 * np.cos(t2) + 1
    z2 = 2 * np.sin(t2)
    points.extend(np.column_stack([x2, y2, z2]))
    
    # 第二条线: 三段圆弧
    t = np.linspace(-np.pi/3, np.pi/3, 200)
    
    # 为第二条线增加x坐标间隔
    x_offset = 5.0  # 增大间隔
    
    # 第一段
    x3 = np.full_like(t, x_offset)
    y3 = 1.5 * np.cos(t)
    z3 = 1.5 * np.sin(t)
    points.extend(np.column_stack([x3, y3, z3]))
    
    # 第二段
    x4 = np.full_like(t, x_offset)
    y4 = 1.5 * np.cos(t) + 2
    z4 = 1.5 * np.sin(t)
    points.extend(np.column_stack([x4, y4, z4]))
    
    # 第三段
    x5 = np.full_like(t, x_offset)
    y5 = 1.5 * np.cos(t) + 4
    z5 = 1.5 * np.sin(t)
    points.extend(np.column_stack([x5, y5, z5]))
    
    # 第三条线: 两段圆弧
    t = np.linspace(0, np.pi, 200)
    x_offset = 10.0  # 进一步增大间隔
    
    # 第一段
    x6 = np.full_like(t, x_offset)
    y6 = 3 * np.cos(t)
    z6 = 3 * np.sin(t)
    points.extend(np.column_stack([x6, y6, z6]))
    
    # 第二段
    x7 = np.full_like(t, x_offset)
    y7 = 1.5 * np.cos(t)
    z7 = 1.5 * np.sin(t) + 4
    points.extend(np.column_stack([x7, y7, z7]))
    
    # 添加随机噪声
    points = np.array(points)
    noise = np.random.normal(0, 0.02, points.shape)
    points += noise
    
    return points

def save_point_cloud(points, path):
    """保存点云到PLY文件"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(path, pcd)

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
