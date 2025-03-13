import numpy as np
from typing import List, Dict, Any, Tuple
from algorithm.circle_arc import CircleArc
from UserInterface.point_cloud_grouper import PointCloudGrouper
from algorithm.axis_projection import AxisProjector
import random

class ArcFittingProcessor:
    def __init__(self):
        self.circle_fitter = CircleArc()
        self.projector = AxisProjector()
        self.grouper = PointCloudGrouper()

    def process_all_lines(self, points: np.ndarray, settings: Dict) -> Dict[str, Any]:
        """
        处理所有线的圆拟合

        Args:
            points (np.ndarray): 点云数据
            settings (Dict): 拟合参数，包含：
                - arcNormalNeighbors: 邻近点数量
                - fitIterations: 拟合迭代次数n
                - samplePercentage: 采样百分比m
                - axis_direction: 轴方向向量
                - point_on_axis: 轴上一点

        Returns:
            Dict: 包含所有线的拟合结果和统计信息
        """
        # 获取所有分组
        groups = self.grouper.get_all_groups(points, axis='x')
        all_lines_stats = []
        all_valid_radii = []

        # 处理每一组（每一根线）
        for line_idx, line_points in enumerate(groups):
            # 获取每个点的邻近点
            neighbors, _ = self.grouper.find_neighbors_kdtree(
                line_points, 
                settings['arcNormalNeighbors']
            )
            
            # 去除重复点
            unique_neighbors = self.grouper.remove_duplicate_neighbors(neighbors)
            
            # 投影到垂直于轴的平面上
            line_stats = self.process_single_line(
                unique_neighbors,
                settings['axis_direction'],
                settings['point_on_axis'],
                settings['fitIterations'],
                settings['samplePercentage']
            )
            
            line_stats['lineIndex'] = line_idx
            all_lines_stats.append(line_stats)
            all_valid_radii.extend([r['radius'] for r in line_stats['radiusData'] if r['isValid']])

        # 计算总体统计信息（仅使用有效值）
        overall_stats = {
            'overallMedian': float(np.median(all_valid_radii)),
            'overallStd': float(np.std(all_valid_radii)),
            'minValidRadius': float(np.min(all_valid_radii)),
            'maxValidRadius': float(np.max(all_valid_radii)),
            'totalValidCount': len(all_valid_radii)
        }

        return {
            'lineStats': all_lines_stats,
            'overallStats': overall_stats
        }

    def process_single_line(
        self,
        neighbors: List[np.ndarray],
        axis_direction: np.ndarray,
        point_on_axis: np.ndarray,
        n_iterations: int,
        sample_percentage: int
    ) -> Dict[str, Any]:
        """
        处理单根线的圆拟合

        Args:
            neighbors: 每个点的邻近点列表
            axis_direction: 轴方向向量
            point_on_axis: 轴上一点
            n_iterations: 迭代次数n
            sample_percentage: 采样百分比m

        Returns:
            Dict: 包含该线的拟合结果和统计信息
        """
        radii = []
        
        # 将所有邻近点合并为一个数组
        all_points = np.vstack(neighbors)
        
        # 获取投影点
        projected_points, planar_coords = self.projector.project_points(
            all_points, 
            axis_direction, 
            point_on_axis
        )

        # 迭代拟合
        for _ in range(n_iterations):
            # 随机采样m%的点
            n_samples = int(len(planar_coords) * sample_percentage / 100)
            sample_indices = random.sample(range(len(planar_coords)), n_samples)
            sampled_points = planar_coords[sample_indices]
            
            # 使用HyperFit方法拟合圆
            _, _, radius = self.circle_fitter.fit_circle_arc(sampled_points)
            radii.append(radius)

        # 异常值筛除
        q1 = np.percentile(radii, 25)
        q3 = np.percentile(radii, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        valid_radii = [r for r in radii if lower_bound <= r <= upper_bound]
        
        # 计算统计信息（使用有效值）
        median_radius = float(np.median(valid_radii))
        std_dev = float(np.std(valid_radii))
        
        # 为K线图准备数据
        radius_data = [
            {
                'iteration': i + 1,
                'radius': radius,
                'isValid': int(lower_bound <= radius <= upper_bound)
            } for i, radius in enumerate(radii)
        ]

        return {
            'radiusData': radius_data,  # K线图数据
            'medianRadius': median_radius,  # 中位数
            'stdDev': std_dev,
            'outlierStats': {  # 异常值统计
                'totalCount': len(radii),
                'validCount': len(valid_radii),
                'outlierCount': len(radii) - len(valid_radii),
                'bounds': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound)
                }
            }
        }
