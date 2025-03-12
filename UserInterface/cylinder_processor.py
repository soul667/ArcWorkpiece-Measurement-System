import numpy as np
import open3d as o3d
import logging
import os
import time
import pypcl_algorithms as pcl_algo
import matplotlib
from matplotlib import font_manager
from algorithm.axis_projection import AxisProjector
from typing import Dict, Tuple, Optional, Union, List

logger = logging.getLogger(__name__)

def setup_matplotlib():
    """设置Matplotlib环境"""
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt

    # 设置中文字体
    font_path = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
    try:
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.sans-serif'] = [prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        logger.info(f"成功加载中文字体: {font_path}")
        return plt, prop
    except Exception as e:
        logger.warning(f"加载中文字体失败: {str(e)}，将使用默认字体")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        return plt, None

class CylinderProcessor:
    """圆柱体拟合处理器类"""
    
    VALID_METHODS = ['NormalRANSAC', 'NormalLeastSquares', 'NormalPCA', 'PCA', 'RobPCA']
    
    def __init__(self):
        """初始化处理器"""
        self.logger = logging.getLogger(__name__)
        self.projector = AxisProjector()
        self.ransac_filtered_points = None  # 存储RANSAC过滤后的点
        self.plt, self.prop = setup_matplotlib()
        self.dpi = 150  # 设置图像分辨率

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
        if cylinder_method not in self.VALID_METHODS:
            raise ValueError(f"无效的圆柱体拟合方法: {cylinder_method}，可选方法: {', '.join(self.VALID_METHODS)}")
        if normal_neighbors <= 0:
            raise ValueError(f"法向量邻居点数必须大于0: {normal_neighbors}")
        if ransac_threshold <= 0:
            raise ValueError(f"RANSAC阈值必须大于0: {ransac_threshold}")
        if min_radius <= 0 or max_radius <= 0 or min_radius >= max_radius:
            raise ValueError(f"无效的半径范围: {min_radius} - {max_radius}")

    def _fit_cylinder_normal_ransac(
        self,
        points: np.ndarray,
        normal_neighbors: int,
        ransac_threshold: float,
        min_radius: float,
        max_radius: float,
        normal_distance_weight: float,
        max_iterations: int
    ) -> Tuple[np.ndarray, np.ndarray, float,np.ndarray]:
        """使用基于法向量的RANSAC方法拟合圆柱体"""
        try:
            point_on_axis, axis_direction, radius, iterations, filtered_points = pcl_algo.fit_cylinder_ransac(
                points,
                distance_threshold=ransac_threshold,
                k_neighbors=normal_neighbors,
                min_radius=min_radius,
                max_radius=max_radius,
                normal_distance_weight=normal_distance_weight,
                max_iterations=max_iterations
            )
            self.logger.info(f"RANSAC迭代次数: {iterations}")
            # 更新全局filtered_points变量
            self.ransac_filtered_points = filtered_points
            self.logger.info(f"NormalRANSAC拟合成功: 轴点={point_on_axis}, 方向={axis_direction}, 半径={radius}")
            return point_on_axis, axis_direction, radius ,filtered_points
        except Exception as e:
            self.logger.warning(f"NormalRANSAC拟合失败: {str(e)}")
            raise

    def _fit_cylinder_normal_least_squares(
        self,
        points: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """使用基于法向量的最小二乘法拟合圆柱体"""
        try:
            axis_direction = pcl_algo.find_cylinder_axis_svd(normals)
            point_on_axis = np.mean(points, axis=0)
            self.logger.info(f"NormalLeastSquares拟合成功: 轴点={point_on_axis}, 方向={axis_direction}")
            return point_on_axis, axis_direction, None
        except Exception as e:
            self.logger.warning(f"NormalLeastSquares拟合失败: {str(e)}")
            raise

    def _fit_cylinder_normal_pca(
        self,
        points: np.ndarray,
        normals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """使用基于法向量的PCA方法拟合圆柱体"""
        try:
            axis_direction = pcl_algo.find_cylinder_axis_svd(normals)  # 暂时用SVD代替
            point_on_axis = np.mean(points, axis=0)
            self.logger.info(f"NormalPCA拟合成功: 轴点={point_on_axis}, 方向={axis_direction}")
            return point_on_axis, axis_direction, None
        except Exception as e:
            self.logger.warning(f"NormalPCA拟合失败: {str(e)}")
            raise

    def _fit_cylinder_pca(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """使用PCA方法拟合圆柱体（未实现）"""
        raise NotImplementedError("PCA方法尚未实现")

    def _fit_cylinder_robust_pca(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, None]:
        """使用鲁棒PCA方法拟合圆柱体（未实现）"""
        raise NotImplementedError("RobPCA方法尚未实现")

    def _visualize_projection(self, planar_coords: np.ndarray) -> str:
        """可视化投影结果并保存为图片
        
        Args:
            planar_coords: 投影后的2D坐标
        
        Returns:
            str: 保存的图片路径
        """
        fig = None
        try:
            # 创建高DPI的图像以获得更好的质量
            fig = self.plt.figure(figsize=(10, 10), dpi=self.dpi)
            self.plt.scatter(planar_coords[:, 0], planar_coords[:, 1], c='b', s=1, alpha=0.5)

            # 使用字体属性添加标题和标签
            font_props = self.prop if self.prop else None
            self.plt.axis('equal')
            self.plt.grid(True)
            
            # 保存图片到临时文件夹
            temp_dir = os.path.join("UserInterface/assets", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            image_path = os.path.join(temp_dir, 'projection.png')
            self.plt.savefig(
                image_path,
                bbox_inches='tight',
                pad_inches=0.1,
                dpi=self.dpi,
                facecolor='white',
                edgecolor='none'
            )
            
            self.logger.info(f"成功保存投影图像到: {image_path}")
            return image_path
        except Exception as e:
            self.logger.error(f"生成投影图像失败: {str(e)}")
            raise RuntimeError(f"生成投影视图失败: {str(e)}")
        finally:
            # 确保始终清理图形资源
            try:
                if fig is not None:
                    self.plt.close(fig)
            except Exception as e:
                self.logger.warning(f"清理图形资源时出错: {str(e)}")

    def process_cylinder_fitting(
        self,
        points: np.ndarray,
        cylinder_method: str = 'NormalRANSAC',
        normal_neighbors: int = 30,
        ransac_threshold: float = 0.1,
        min_radius: float = 6,
        max_radius: float = 11,
        max_iterations: int = 1000,
        normal_distance_weight: float = 0.8
    ) -> Dict[str, Union[str, Dict[str, List[float]]]]:
        """执行圆柱体拟合处理

        Args:
            points: 点云数据
            cylinder_method: 拟合方法
                - NormalRANSAC: 基于法向量的RANSAC方法
                - NormalLeastSquares: 基于法向量的最小二乘法
                - NormalPCA: 基于法向量的PCA方法
                - PCA: PCA方法（未实现）
                - RobPCA: 鲁棒PCA方法（未实现）
            normal_neighbors: 法向量计算的邻居点数
            ransac_threshold: RANSAC距离阈值
            min_radius: 最小半径(mm)
            max_radius: 最大半径(mm)

        Returns:
            包含状态和轴信息的字典

        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当处理过程发生错误时
            NotImplementedError: 当选择未实现的方法时
        """
        try:
            # 验证输入参数
            self._validate_input(
                points, cylinder_method, normal_neighbors,
                ransac_threshold, min_radius, max_radius
            )

            self.logger.info(f"开始处理点云数据: {points.shape[0]} 个点")
            start = time.time()
            
            # 对于需要法向量的方法，先计算法向量
            # 计算点云法线
            normals = None
            if cylinder_method.startswith('Normal'):
                try:
                    self.logger.info("计算点云法线...")
                    normals = pcl_algo.compute_normals(points, k_neighbors=normal_neighbors)
                    self.logger.info(f"法线计算完成，耗时: {time.time()-start:.2f}s")
                except Exception as e:
                    self.logger.error(f"法线计算失败: {str(e)}")
                    raise RuntimeError(f"计算点云法线失败: {str(e)}")
            if cylinder_method == 'NormalRANSAC':
                point_on_axis, axis_direction, radius,filtered_points = self._fit_cylinder_normal_ransac(
                    points, normal_neighbors, ransac_threshold, min_radius, max_radius, normal_distance_weight, max_iterations
                )
            elif cylinder_method == 'NormalLeastSquares':
                point_on_axis, axis_direction, radius = self._fit_cylinder_normal_least_squares(
                    points, normals
                )
                self.logger.debug(f"最小二乘法拟合结果: 轴点={point_on_axis}, 方向={axis_direction}")
            elif cylinder_method == 'NormalPCA':
                point_on_axis, axis_direction, radius = self._fit_cylinder_normal_pca(
                    points, normals
                )
            elif cylinder_method == 'PCA':
                point_on_axis, axis_direction, radius = self._fit_cylinder_pca(points)
            elif cylinder_method == 'RobPCA':
                point_on_axis, axis_direction, radius = self._fit_cylinder_robust_pca(points)

            if cylinder_method == 'NormalRANSAC':
                points = filtered_points
            # 投影点云到平面
            projected_points, planar_coords = self.projector.project_points(
                points, axis_direction, point_on_axis
            )
            
            # 保存投影结果为图片
            projection_image = self._visualize_projection(planar_coords)

            # 构造返回结果
            result = {
                "status": "success",
                "method": cylinder_method,
                "axis": {
                    "point": point_on_axis.tolist(),
                    "direction": axis_direction.tolist()
                },
                "projectionImage": "/assets/temp/projection.png"  # 添加投影图像路径
            }
            if radius is not None:
                result["radius"] = radius

            return result

        except ValueError as ve:
            self.logger.error(f"输入参数错误: {str(ve)}")
            raise
        except NotImplementedError as ne:
            self.logger.error(f"方法未实现: {str(ne)}")
            raise
        except Exception as e:
            self.logger.error(f"处理过程出错: {str(e)}")
            raise RuntimeError(f"圆柱体拟合失败: {str(e)}")
