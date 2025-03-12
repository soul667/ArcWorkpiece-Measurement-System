import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import pypcl_algorithms as pcl_algo

# Configure matplotlib display settings
matplotlib.rcParams['font.family'] = "DejaVu Sans Mono"

def generate_test_points(point_count=1000, radius=0.5, height=2.0, noise_std=0.01, 
                        axis_direction=np.array([0, 0, 1])):
    """生成圆柱体点云数据
    
    Args:
        point_count: 点云数量
        radius: 圆柱体半径
        height: 圆柱体高度
        noise_std: 噪声标准差
        axis_direction: 圆柱体轴向方向
        
    Returns:
        points: 生成的圆柱体点云，numpy数组(N,3)
    """
    # 归一化轴向向量
    axis = axis_direction / np.linalg.norm(axis_direction)
    
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
    
    # 首先生成Z轴方向的圆柱体点云
    thetas = np.random.uniform(0, 2*np.pi, point_count)
    heights = np.random.uniform(-height/2, height/2, point_count)
    
    # 在圆柱体表面生成点
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

def visualize_projection(points, projected_2d, v, x, filename):
    """可视化原始三维点和投影结果"""
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制三维点云和投影方向
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='gray', alpha=0.5, s=1, label='Original Points')
    
    # 绘制投影向量
    center = np.mean(points, axis=0)
    line_points = np.array([center - v, center + v])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'r--', 
            linewidth=2, label='Projection Direction')
    
    # 绘制投影平面上的参考点
    ax.scatter([x[0]], [x[1]], [x[2]], c='r', s=100, marker='*', label='Reference Point')
    
    ax.legend()
    
    # 绘制投影结果
    ax = fig.add_subplot(122)
    ax.scatter(projected_2d[:,0], projected_2d[:,1], c='blue', alpha=0.5, s=1)
    ax.set_aspect('equal')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_o3d_visualization(points, v, x, filename, width=1920, height=1080):
    """使用Open3D保存点云可视化结果"""
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.8, 0.8, 0.8])  # 浅灰色
    
    # 创建投影方向线
    center = np.mean(points, axis=0)
    line_points = np.vstack([
        center - v,
        center + v
    ])
    lines = [[0, 1]]  # 连接首尾点
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])  # 红色
    
    # 创建坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    
    # 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    
    # 添加几何体
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    vis.add_geometry(coord_frame)
    
    # 优化视图
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([1, 1, 1])
    
    # 自动对齐相机
    vis.get_view_control().set_zoom(0.8)
    vis.update_renderer()
    
    # 捕获并保存图像
    image = vis.capture_screen_float_buffer(True)
    plt.imsave(filename, np.asarray(image))
    
    vis.destroy_window()

def visualize_axis_comparison(points, true_axis, ransac_result, svd_axis, title, filename):
    """可视化和比较不同方法检测的轴线"""
    point_on_axis, axis_ransac, radius = ransac_result
    
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制点云和轴线
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='gray', alpha=0.5, s=1)
    
    # 绘制真实轴线
    center = np.mean(points, axis=0)
    line_points = np.array([center - true_axis * 2, center + true_axis * 2])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'b-', 
            linewidth=2, label='True Axis')
    
    # 绘制RANSAC结果
    line_points = np.array([
        point_on_axis - axis_ransac * 2,
        point_on_axis + axis_ransac * 2
    ])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'r--', 
            linewidth=2, label='RANSAC')
    
    # 绘制SVD结果
    line_points = np.array([center - svd_axis * 2, center + svd_axis * 2])
    ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], 'g--', 
            linewidth=2, label='SVD')
    
    ax.legend()
    
    # 绘制角度误差
    ax = fig.add_subplot(122)
    ransac_angle = np.arccos(np.abs(np.dot(axis_ransac, true_axis))) * 180 / np.pi
    svd_angle = np.arccos(np.abs(np.dot(svd_axis, true_axis))) * 180 / np.pi
    
    bars = ax.bar(['RANSAC', 'SVD'], [ransac_angle, svd_angle])
    ax.set_ylabel('Angle Error (degrees)')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}°',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def project_points_to_plane(points, v, x):
    """
    将三维点投影到由向量v和点x定义的平面上。
    
    Args:
        points: 三维点集，可以是numpy数组(N,3)或open3d点云
        v: 投影方向向量，numpy数组(3,)
        x: 平面上的一点，numpy数组(3,)
        
    Returns:
        projected_2d: 投影后的二维点集，numpy数组(N,2)
    """
    # 将输入转换为numpy数组
    if isinstance(points, o3d.geometry.PointCloud):
        points_np = np.asarray(points.points)
    else:
        points_np = np.asarray(points)
    
    v = np.asarray(v)
    x = np.asarray(x)
    
    # 确保输入维度正确
    assert points_np.shape[1] == 3, "Points must be 3D"
    assert v.shape == (3,), "Vector v must be 3D"
    assert x.shape == (3,), "Point x must be 3D"
    
    # 1. 归一化投影方向向量
    n = v / np.linalg.norm(v)
    
    # 2. 计算投影点
    # 计算每个点到平面的距离
    t = np.dot(x - points_np, n) / np.dot(n, n)
    # 计算投影点
    Q = points_np + np.outer(t, v)
    
    # 3. 构造局部坐标系
    if abs(n[0]) > 1e-6 or abs(n[2]) > 1e-6:  # x或z分量非零
        u = np.array([n[2], 0, -n[0]])
        u = u / np.linalg.norm(u)
        w = np.cross(n, u)
    else:  # n沿y轴方向
        u = np.array([1, 0, 0])
        w = np.array([0, 0, -1])
    
    # 4. 转换到二维坐标
    d = Q - x  # 相对于平面上点x的位移
    projected_2d = np.column_stack([
        np.dot(d, u),  # u方向的坐标
        np.dot(d, w)   # w方向的坐标
    ])
    
    return projected_2d

def test_projection():
    """测试投影功能"""
    print("\n" + "="*50)
    print("Starting Point Cloud Projection Tests")
    print("="*50)
    
    test_cases = [
        ("Z-axis Projection", np.array([0, 0, 1]), np.array([0, 0, 0])),
        ("X-axis Projection", np.array([1, 0, 0]), np.array([0, 0, 0])),
        ("Diagonal Projection", np.array([1, 1, 1])/np.sqrt(3), np.array([0.5, 0.5, 0.5])),
    ]
    
    for i, (case_name, v, x) in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {case_name}")
        print("-"*30)
        
        # 生成测试点云
        print("\n1. Generating test cylinder point cloud...")
        points = generate_test_points(
            point_count=10000,
            radius=0.5,
            height=2.0,
            noise_std=0.01,
            axis_direction=v
        )
        print(f"   - Number of points: {len(points)}")
        print(f"   - Point cloud bounds:")
        print(f"     X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
        print(f"     Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
        print(f"     Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
        
        # 使用pypcl_algorithms检测轴线
        print("\n2. Computing normals...")
        normals = pcl_algo.compute_normals(points, k_neighbors=30)
        
        print("\n3. Detecting cylinder axis...")
        print("   3.1 Using RANSAC method...")
        ransac_result = pcl_algo.fit_cylinder_ransac(
            points,
            distance_threshold=0.01,
            max_iterations=1000,
            k_neighbors=30,
            normal_distance_weight=0.1,
            min_radius=0.1,
            max_radius=1.0
        )
        
        print("   3.2 Using SVD method...")
        svd_axis = pcl_algo.find_cylinder_axis_svd(normals)
        
        # 可视化轴线检测结果
        print("\n4. Visualizing axis detection results...")
        print(f"   - True axis: {v}")
        print(f"   - RANSAC axis: {ransac_result[1]}")
        print(f"   - SVD axis: {svd_axis}")
        
        axis_filename = f'axis_detection_{case_name.lower().replace(" ", "_")}.png'
        print(f"   - Saving axis comparison to: {axis_filename}")
        visualize_axis_comparison(points, v, ransac_result, svd_axis, case_name, axis_filename)
        
        # 创建Open3D点云对象并显示
        print("\n5. Creating and displaying Open3D point cloud...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        print(f"   Visualization window will open. Close it to continue...")
        o3d.visualization.draw_geometries([pcd])
        
        # 测试投影
        print("\n6. Testing projection...")
        print("\n6.1. With numpy array input...")
        print(f"   - Projection direction: {v}")
        print(f"   - Reference point: {x}")
        projected_np = project_points_to_plane(points, v, x)
        print(f"   - Projected points shape: {projected_np.shape}")
        print(f"   - 2D bounds:")
        print(f"     X: [{projected_np[:,0].min():.3f}, {projected_np[:,0].max():.3f}]")
        print(f"     Y: [{projected_np[:,1].min():.3f}, {projected_np[:,1].max():.3f}]")
        
        print("\n6.2. With Open3D point cloud input...")
        projected_o3d = project_points_to_plane(pcd, v, x)
        
        # 验证结果
        print("\n7. Validating results...")
        if np.allclose(projected_np, projected_o3d, atol=1e-6):
            print("   ✓ Results match between numpy and Open3D inputs")
        else:
            print("   ✗ Results differ between numpy and Open3D inputs")
            max_diff = np.max(np.abs(projected_np - projected_o3d))
            print(f"   - Maximum difference: {max_diff:.6f}")
            assert False, "Results validation failed!"
        
        # 保存投影可视化结果
        print("\n8. Saving projection visualization results...")
        filename_base = case_name.lower().replace(' ', '_')
        
        # Matplotlib visualization
        vis_filename = f'projection_{filename_base}.png'
        print(f"   - Saving matplotlib visualization to: {vis_filename}")
        visualize_projection(points, projected_np, v, x, vis_filename)
        
        # Open3D visualization
        o3d_filename = f'projection_o3d_{filename_base}.png'
        print(f"   - Saving Open3D visualization to: {o3d_filename}")
        save_o3d_visualization(points, v, x, o3d_filename)
        
        print(f"\nTest case {i} completed successfully!")
    
    print("\n" + "="*50)
    print("All tests completed successfully!")
    print("="*50)

if __name__ == "__main__":
    test_projection()
