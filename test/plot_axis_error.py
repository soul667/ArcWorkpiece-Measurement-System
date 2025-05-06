import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import open3d as o3d
import pypcl_algorithms as pcl_algo

import os
import sys

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录路径（假设 test/ 和 algorithm/ 在同一级目录下）
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# 将项目根目录添加到 sys.path 中
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithm.pca.PcaAxis import *

# 设置中文字体
font_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'algorithm', 'SimSun.ttf')
fm.fontManager.addfont(font_path)
font_props = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = [font_props.get_name(), 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def generate_test_points(point_count=1000, radius=0.5, height=2.0, noise_std=0.01, 
                        axis_direction=np.array([0, 0, 1]), angle_range=(0, 360)):
    """生成圆柱体点云数据"""
    # 归一化轴向向量
    axis = axis_direction / np.linalg.norm(axis_direction)
    
    # 创建旋转矩阵，将[0,0,1]对齐到目标轴向
    if np.allclose(axis, [0, 0, 1]):
        R = np.eye(3)
    else:
        rot_axis = np.cross([0, 0, 1], axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        cos_angle = np.dot([0, 0, 1], axis)
        angle = np.arccos(cos_angle)
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                     [rot_axis[2], 0, -rot_axis[0]],
                     [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * K.dot(K)
    
    # 角度范围转换为弧度
    start_angle, end_angle = np.deg2rad(angle_range[0]), np.deg2rad(angle_range[1])
    
    # 生成点
    thetas = np.random.uniform(start_angle, end_angle, point_count)
    heights = np.random.uniform(-height/2, height/2, point_count)
    radius_noise = np.random.normal(0, noise_std, point_count)
    r = radius + radius_noise
    
    x = r * np.cos(thetas)
    y = r * np.sin(thetas)
    z = heights
    
    points = np.column_stack([x, y, z])
    points = points.dot(R.T)
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, (point_count, 3))
    points += noise
    
    return points

def generate_random_test_cases(n):
    """生成随机测试用例"""
    test_cases = []
    for i in range(n):
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        point = np.random.rand(3)
        name = f"Random Test {i+1}"
        test_cases.append((name, axis, point))
    return test_cases

def process_error(error, angle):
    """处理误差值
    - 过滤大于10度的误差
    - 80度以上使用90-误差
    """
    # if error > 10:
    #     return None
    if error > 80 and error < 90:
        return 90 - error
    return error

def main():
    # 测试角度范围
    angles = np.linspace(5, 15, 8)  # 更密集的采样点
    n_simulations = 100  # 每个角度仿真30次
    
    # 记录所有测试结果
    results = {angle: {
        'RANSAC': [],
        '最小二乘': [],
        'PCA': []
    } for angle in angles}
    
    # 设置横轴显示位置
    x_positions = {'RANSAC': 1, '最小二乘': 2, 'PCA': 3}
    
    # 生成基准测试轴向
    true_axis = np.array([0, 0.3, 1])  # 使用固定的轴向进行测试
    true_axis= true_axis / np.linalg.norm(true_axis)
    for angle in angles:
        print(f"Processing angle: {angle}°")
        for sim in range(n_simulations):
            # 生成测试点云
            points = generate_test_points(
                point_count=20000,
                radius=8,
                height=5,
                noise_std=0.001,
                axis_direction=true_axis,
                angle_range=(95+0, 95+angle)
            )
            
            # 计算法向量
            normals = pcl_algo.compute_normals(points, k_neighbors=15)
            
            # 各种方法检测轴线
            ransac_result = pcl_algo.fit_cylinder_ransac(
                points,
                distance_threshold=0.01,
                max_iterations=1000,
                k_neighbors=30,
                normal_distance_weight=0.7,
                min_radius=0,
                max_radius=40
            )
            
            svd_axis = pcl_algo.find_cylinder_axis_svd(normals)
            
            PcaAxis_ = PcaAxis()
            normal_pca_axis = PcaAxis_.get_axis(points, model=PCAMethod.ORDINARY_PCA)
            
            # 计算误差
            errors = {
                'RANSAC': np.arccos(np.abs(np.dot(ransac_result[1], true_axis))) * 180 / np.pi,
                '最小二乘': np.arccos(np.abs(np.dot(svd_axis, true_axis))) * 180 / np.pi,
                'PCA': np.arccos(np.abs(np.dot(normal_pca_axis, true_axis))) * 180 / np.pi
            }
            
            # 处理每个方法的误差并存储
            for method, error in errors.items():
                processed_error = process_error(error, angle)
                if processed_error is not None:
                    results[angle][method].append(processed_error)

        # 为当前角度生成箱型图
        plt.figure(figsize=(10, 6))
        methods = ['RANSAC', '最小二乘', 'PCA']
        data = [results[angle][method] for method in methods]
        positions = [x_positions[method] for method in methods]
        
        colors = ['#1f77b4', '#d62728', '#2ca02c']
        box = plt.boxplot(data, 
                         positions=positions,
                         tick_labels=methods,
                         patch_artist=True)
        
        # 设置箱型图样式
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(box[element], color='black')
        
        # 设置填充颜色
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # 设置异常值点的样式
        plt.setp(box['fliers'], marker='o', markersize=4)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel('拟合误差 (度)', fontsize=12)
        plt.title(f'圆心角 {angle:.1f}° 的轴向拟合误差比较 (N={n_simulations})', fontsize=14)
        
        # 保存当前角度的图片
        plt.savefig(f'axis_error_comparison_{angle:.1f}.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()
