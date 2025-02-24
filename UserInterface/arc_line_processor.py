import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from typing import List, Dict, Any, Optional
import open3d as o3d
import os
from matplotlib import font_manager

# 设置字体文件路径
font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'

# 创建 FontProperties 对象，指定字体路径
prop = font_manager.FontProperties(fname=font_path)

# 设置 matplotlib 使用该字体
plt.rcParams['font.sans-serif'] = [prop.get_name()]  # 使用通过 FontProperties 获取的字体名称
plt.rcParams['axes.unicode_minus'] = False 

class ArcLineProcessor:
    """圆弧线段处理器"""
    
    def __init__(self):
        """初始化处理器"""
        self.points = None
        self.lines = []
        self.debug = False
    
    def load_point_cloud(self, points: np.ndarray = None, ply_path: str = None) -> bool:
        """加载点云数据
        Args:
            points: 直接传入的点云数据, shape (N, 3)
            ply_path: PLY文件路径
        Returns:
            bool: 加载是否成功
        """
        if points is not None:
            self.points = points
        elif ply_path is not None and os.path.exists(ply_path):
            point_cloud = o3d.io.read_point_cloud(ply_path)
            self.points = np.asarray(point_cloud.points)
        else:
            return False
        return True

    def set_debug(self, debug: bool):
        """设置是否显示调试信息"""
        self.debug = debug
        
    def calculate_curvature(self, points: np.ndarray, window_size: int = None) -> np.ndarray:
        """计算点序列的曲率
        Args:
            points: shape (N, 2) 的点序列
            window_size: 窗口大小(None则自动计算)
        Returns:
            曲率数组
        """
        n = len(points)
        
        # 处理点数过少的情况
        if n < 3:
            return np.zeros(n)  # 点数太少，返回全零曲率
            
        # 自适应窗口大小
        if window_size is None:
            # 建议窗口大小为点数的5%-10%，且为奇数
            # 确保窗口大小不超过点数
            window_size = max(3, min(n // 20, min(51, n)))
            if window_size % 2 == 0:
                window_size = min(window_size + 1, n)
                
        if self.debug:
            print(f"点数: {n}, 使用窗口大小: {window_size}")
            
            # 创建可视化窗口
            plt.figure(figsize=(15, 10))
            
            # 绘制原始点
            plt.subplot(221)
            plt.title("原始点云")
            plt.scatter(points[:, 0], points[:, 1], c='blue', s=1)
            plt.axis('equal')
            
        curvatures = np.zeros(n)
        half_window = window_size // 2
        
        # 如果点数太少，就返回全零曲率
        if n <= window_size:
            if self.debug:
                print(f"警告: 点数({n})小于等于窗口大小({window_size}), 返回全零曲率")
            return curvatures
            
        for i in range(half_window, n - half_window):
            window = points[i-half_window:i+half_window+1]
            
            if self.debug and i % (n//10) == 0:  # 显示部分窗口示例
                plt.subplot(221)
                plt.scatter(window[:, 0], window[:, 1], c='red', s=3)
            
            dx = np.gradient(window[:, 0])
            dy = np.gradient(window[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            # 避免除零错误
            denominator = (dx * dx + dy * dy) ** 1.5
            mask = denominator > 1e-10  # 忽略接近零的值
            if np.any(mask):
                curvature = np.abs(dx * ddy - dy * ddx) / denominator
                curvatures[i] = np.mean(curvature[mask])
        
        # 处理边界点
        curvatures[:half_window] = curvatures[half_window]
        curvatures[-half_window:] = curvatures[-half_window-1]
        
        if self.debug:
            # 绘制曲率分布
            plt.subplot(222)
            plt.title("曲率分布")
            plt.plot(curvatures)
            plt.xlabel("点索引")
            plt.ylabel("曲率")
            
            # 绘制曲率热图
            plt.subplot(223)
            plt.title("曲率热图")
            scatter = plt.scatter(points[:, 0], points[:, 1], 
                                c=curvatures, cmap='viridis', s=2)
            plt.colorbar(scatter)
            plt.axis('equal')
            
            # 曲率直方图
            plt.subplot(224)
            plt.title("曲率直方图")
            plt.hist(curvatures, bins=50)
            plt.xlabel("曲率值")
            plt.ylabel("频数")
            
            plt.tight_layout()
            plt.show()
            
            # 打印统计信息
            print(f"曲率统计:")
            print(f"最小值: {np.min(curvatures):.6f}")
            print(f"最大值: {np.max(curvatures):.6f}")
            print(f"均值: {np.mean(curvatures):.6f}")
            print(f"标准差: {np.std(curvatures):.6f}")
            
        return curvatures

    def segment_by_meanshift(self, points: np.ndarray, curvatures: np.ndarray, 
                           bandwidth: float = 0.5) -> np.ndarray:
        """使用Mean Shift进行分段
        Args:
            points: 原始点
            curvatures: 曲率值
            bandwidth: 带宽参数(None则自动估计)
        Returns:
            标签数组
        """
        # 处理点数过少的情况
        if len(points) < 3:
            return np.zeros(len(points), dtype=int)
            
        features = np.column_stack([points, curvatures.reshape(-1, 1)])
        
        # 确保数据有效
        if not np.all(np.isfinite(features)):
            if self.debug:
                print("警告: 特征数据包含无效值，将被替换为0")
            features = np.nan_to_num(features, 0)
            
        if self.debug:
            print(f"使用带宽: {bandwidth}")
        
        clustering = MeanShift(bandwidth=bandwidth).fit(features)
        labels = clustering.labels_
        
        if self.debug:
            # 可视化聚类结果
            plt.figure(figsize=(15, 5))
            
            # 空间分布
            plt.subplot(131)
            plt.title(f"聚类结果 (共{len(np.unique(labels))}类)")
            scatter = plt.scatter(points[:, 0], points[:, 1], 
                                c=labels, cmap='tab20', s=2)
            plt.colorbar(scatter)
            plt.axis('equal')
            
            # 曲率与聚类关系
            plt.subplot(132)
            plt.title("各类曲率分布")
            for label in np.unique(labels):
                mask = labels == label
                plt.plot(curvatures[mask], 'o', label=f'类别{label}', 
                        markersize=1, alpha=0.5)
            plt.xlabel("点索引")
            plt.ylabel("曲率")
            plt.legend()
            
            # 聚类统计
            plt.subplot(133)
            counts = np.bincount(labels)
            plt.title("各类点数统计")
            plt.bar(range(len(counts)), counts)
            plt.xlabel("类别")
            plt.ylabel("点数")
            
            plt.tight_layout()
            plt.show()
            
            # 打印统计信息
            print("\n聚类统计:")
            print(f"类别数量: {len(np.unique(labels))}")
            for label in np.unique(labels):
                mask = labels == label
                print(f"\n类别 {label}:")
                print(f"点数: {np.sum(mask)}")
                print(f"平均曲率: {np.mean(curvatures[mask]):.6f}")
                print(f"曲率标准差: {np.std(curvatures[mask]):.6f}")
        
        return labels

    def group_by_x(self, points: np.ndarray, tolerance: float = 0.1) -> List[np.ndarray]:
        """按X坐标对点进行分组
        Args:
            points: 点云数据
            tolerance: X坐标容差
        Returns:
            分组后的点列表
        """
        if len(points) == 0:
            return []
            
        # 先按X坐标排序
        sorted_idx = np.argsort(points[:, 0])
        sorted_points = points[sorted_idx]
        
        # 找出不连续的点
        x_diff = np.diff(sorted_points[:, 0])
        split_indices = np.where(x_diff > tolerance)[0] + 1
        
        # 根据分割点划分组
        groups = np.split(sorted_points, split_indices)
        
        # 过滤掉点数太少的组
        valid_groups = [group for group in groups if len(group) >= 3]
        
        if self.debug:
            print(f"共分成{len(valid_groups)}条线")
            for i, group in enumerate(valid_groups):
                print(f"线{i}: {len(group)}个点")
                
            if valid_groups:  # 只在有分组时显示可视化
                plt.figure(figsize=(10, 6))
                plt.title("线分组结果")
                for i, group in enumerate(valid_groups):
                    plt.scatter(group[:, 0], group[:, 1], label=f'线{i}', s=1)
                plt.legend()
                plt.axis('equal')
                plt.show()
            
        return valid_groups

    def fit_arc(self, points: np.ndarray) -> Dict[str, Any]:
        """拟合圆弧
        Args:
            points: 点云数据
        Returns:
            圆弧参数字典
        """
        if len(points) < 3:
            return None
            
        try:
            # 使用最小二乘法拟合圆
            x = points[:, 0]
            y = points[:, 1]
            
            # 构建方程组
            A = np.column_stack([
                2 * x,
                2 * y,
                np.ones(len(x))
            ])
            b = x**2 + y**2
            
            # 求解最小二乘
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            
            # 提取圆心和半径
            center_x, center_y = params[0], params[1]
            radius = np.sqrt(params[2] + center_x**2 + center_y**2)
            
            # 计算起始和结束角度
            angles = np.arctan2(y - center_y, x - center_x)
            start_angle = np.min(angles)
            end_angle = np.max(angles)
            
            # 计算拟合误差
            distances = np.abs(np.sqrt((x - center_x)**2 + (y - center_y)**2) - radius)
            error = np.mean(distances)
            
            return {
                'center': (center_x, center_y),
                'radius': radius,
                'start_angle': start_angle,
                'end_angle': end_angle,
                'error': error,
                'points': points
            }
        except:
            if self.debug:
                print("警告: 圆弧拟合失败")
            return None

    def process_line(self, points: np.ndarray) -> List[Dict[str, Any]]:
        """处理单条线
        Args:
            points: 线上的点
        Returns:
            圆弧段列表
        """
        # 处理点数过少的情况
        if len(points) < 3:
            if self.debug:
                print(f"警告: 点数过少({len(points)}个点), 无法进行圆弧拟合")
            return []
            
        # 计算曲率
        curvatures = self.calculate_curvature(points[:, :2])
        
        # 使用Mean Shift进行分段
        labels = self.segment_by_meanshift(points[:, :2], curvatures)
        
        # 对每个分段拟合圆弧
        segments = []
        for label in np.unique(labels):
            mask = labels == label
            segment_points = points[mask]
            
            # 只处理点数足够的段
            if len(segment_points) >= 3:
                arc_params = self.fit_arc(segment_points)
                if arc_params is not None:
                    segments.append(arc_params)
            elif self.debug:
                print(f"警告: 类别{label}点数过少({len(segment_points)}个点), 已跳过")
            
        if self.debug and segments:  # 只在有成功拟合的段时显示可视化
            # 可视化拟合结果
            plt.figure(figsize=(10, 6))
            plt.title("圆弧拟合结果")
            
            # 绘制原始点
            plt.scatter(points[:, 0], points[:, 1], c='blue', 
                        s=1, alpha=0.5, label='原始点')
            
            # 绘制拟合的圆弧
            for i, segment in enumerate(segments):
                center = segment['center']
                radius = segment['radius']
                start_angle = segment['start_angle']
                end_angle = segment['end_angle']
                
                # 生成圆弧点
                theta = np.linspace(start_angle, end_angle, 100)
                x = center[0] + radius * np.cos(theta)
                y = center[1] + radius * np.sin(theta)
                plt.plot(x, y, '-', label=f'圆弧{i}')
                plt.plot(center[0], center[1], 'rx')  # 标记圆心
                
            plt.legend()
            plt.axis('equal')
            plt.show()
            
        return segments

    def process(self, tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """处理点云数据
        Args:
            tolerance: X坐标容差
        Returns:
            处理结果
        """
        if self.points is None:
            raise ValueError("未加载点云数据")
            
        # 按X坐标分组
        lines = self.group_by_x(self.points, tolerance)
        
        # 处理每条线
        results = []
        for i, line_points in enumerate(lines):
            if self.debug:
                print(f"\n处理第{i}条线...")
            segments = self.process_line(line_points)
            if segments:  # 只添加包含有效段的线
                results.append({
                    'points': line_points,
                    'segments': segments
                })
            
        self.lines = results
        return results

if __name__ == "__main__":
    # 测试代码
    processor = ArcLineProcessor()
    processor.set_debug(True)  # 启用调试模式
    
    # 从文件加载点云
    temp_dir = os.path.join("UserInterface/assets", "temp")
    temp_ply_path = os.path.join(temp_dir, 'temp.ply')
    
    if os.path.exists(temp_ply_path):
        processor.load_point_cloud(ply_path=temp_ply_path)
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
