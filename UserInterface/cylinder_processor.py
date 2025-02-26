import numpy as np
import open3d as o3d
import logging
import pypcl_algorithms as pcl_algo
from algorithm.axis_projection import AxisProjector
from typing import Dict, Tuple, Optional, Union, List

class CylinderProcessor:
    """圆柱体拟合处理器类"""
    
    def __init__(self):
        """初始化处理器"""
        self.logger = logging.getLogger(__name__)
        self.projector = AxisProjector()

    def _validate_input(
        self,
        points: np.ndarray,
        cylinder_method: str,
        normal_neighbors: int,
        ransac_threshold: float,
        min_radius: float,
        max_radius: float
    ) -> None:
        """验证输入参数

        Args:
            points: 点云数据
            cylinder_method: 拟合方法
            normal_neighbors: 法向量计算的邻居点数
            ransac_threshold: RANSAC距离阈值
            min_radius: 最小半径
            max_radius: 最大半径

        Raises:
            ValueError: 当参数无效时
        """
        if points.size == 0:
            raise ValueError("点云数据为空")
        if cylinder_method not in ['RANSAC', 'SVD']:
            raise ValueError(f"无效的圆柱体拟合方法: {cylinder_method}")
        if normal_neighbors <= 0:
            raise ValueError(f"法向量邻居点数必须大于0: {normal_neighbors}")
        if ransac_threshold <= 0:
            raise ValueError(f"RANSAC阈值必须大于0: {ransac_threshold}")
        if min_radius <= 0 or max_radius <= 0 or min_radius >= max_radius:
            raise ValueError(f"无效的半径范围: {min_radius} - {max_radius}")

    def _fit_cylinder_ransac(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        ransac_threshold: float,
        min_radius: float,
        max_radius: float
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """使用RANSAC方法拟合圆柱体

        Args:
            points: 点云数据
            normals: 法向量数据
            ransac_threshold: RANSAC距离阈值
            min_radius: 最小半径
            max_radius: 最大半径

        Returns:
            轴点、轴向量和半径的元组
        """
        try:
            point_on_axis, axis_direction, radius = pcl_algo.fit_cylinder_ransac(
                points,
                distance_threshold=ransac_threshold,
                k_neighbors=len(normals),
                min_radius=min_radius,
                max_radius=max_radius
            )
            self.logger.info(f"RANSAC拟合成功: 轴点={point_on_axis}, 方向={axis_direction}, 半径={radius}")
            return point_on_axis, axis_direction, radius
        except Exception as e:
            self.logger.warning(f"RANSAC拟合失败: {str(e)}，切换到SVD方法")
            return self._fit_cylinder_svd(points, normals)

    def _fit_cylinder_svd(
        self,
        points: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """使用SVD方法拟合圆柱体

        Args:
            points: 点云数据
            normals: 法向量数据

        Returns:
            轴点、轴向量和半径(None)的元组
        """
        axis_direction = pcl_algo.find_cylinder_axis_svd(normals)
        point_on_axis = np.mean(points, axis=0)
        self.logger.info(f"SVD拟合成功: 轴点={point_on_axis}, 方向={axis_direction}")
        return point_on_axis, axis_direction, None

    def process_cylinder_fitting(
        self,
        points: np.ndarray,
        cylinder_method: str = 'RANSAC',
        normal_neighbors: int = 30,
        ransac_threshold: float = 0.1,
        min_radius: float = 6,
        max_radius: float = 11
    ) -> Dict[str, Union[str, Dict[str, List[float]]]]:
        """执行圆柱体拟合处理

        Args:
            points: 点云数据
            cylinder_method: 拟合方法，'RANSAC'或'SVD'
            normal_neighbors: 法向量计算的邻居点数
            ransac_threshold: RANSAC距离阈值
            min_radius: 最小半径(mm)
            max_radius: 最大半径(mm)

        Returns:
            包含状态和轴信息的字典

        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当处理过程发生错误时
        """
        try:
            # 验证输入参数
            self._validate_input(
                points, cylinder_method, normal_neighbors,
                ransac_threshold, min_radius, max_radius
            )

            # 计算法向量
            normals = pcl_algo.compute_normals(points, k_neighbors=normal_neighbors)
            
            # 根据方法选择拟合算法
            if cylinder_method == 'RANSAC':
                point_on_axis, axis_direction, radius = self._fit_cylinder_ransac(
                    points, normals, ransac_threshold, min_radius, max_radius
                )
            else:
                point_on_axis, axis_direction, radius = self._fit_cylinder_svd(points, normals)

            # 投影点云到平面
            projected_points, planar_coords = self.projector.project_points(
                points, axis_direction, point_on_axis
            )

            # 构造返回结果
            result = {
                "status": "success",
                "axis": {
                    "point": point_on_axis.tolist(),
                    "direction": axis_direction.tolist()
                }
            }
            if radius is not None:
                result["radius"] = radius

            return result

        except ValueError as ve:
            self.logger.error(f"输入参数错误: {str(ve)}")
            raise
        except Exception as e:
            self.logger.error(f"处理过程出错: {str(e)}")
            raise RuntimeError(f"圆柱体拟合失败: {str(e)}")
