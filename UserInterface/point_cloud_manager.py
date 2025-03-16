import os
import numpy as np
import open3d as o3d
import cv2
import logging
from typing import Dict, Tuple, Optional, Union, List
from io import BytesIO

class PointCloudManager:
    """点云管理类：处理点云文件的上传、可视化和信息管理"""
    
    def __init__(self, temp_dir: str = "UserInterface/assets/temp"):
        """初始化点云管理器
        
        Args:
            temp_dir: 临时文件存储目录
        """
        self.temp_dir = temp_dir
        self.logger = logging.getLogger(__name__)
        self._ensure_temp_dir()
        
    def _ensure_temp_dir(self) -> None:
        """确保临时目录存在"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
    def normalize_and_map(
        self,
        x: np.ndarray,
        y: np.ndarray,
        image_width: int = 1280,
        image_height: int = 720
    ) -> np.ndarray:
        """归一化并映射坐标到图像空间
        
        Args:
            x: x坐标数组
            y: y坐标数组
            image_width: 输出图像宽度
            image_height: 输出图像高度
            
        Returns:
            生成的图像数组
            
        Raises:
            ValueError: 当输入参数无效时
        """
        if x.size == 0 or y.size == 0:
            return np.zeros((image_height, image_width, 3), dtype=np.uint8)

        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"无效的图像尺寸: width={image_width}, height={image_height}")

        x_norm = (x - x.min()) / (x.max() - x.min())
        y_norm = (y - y.min()) / (y.max() - y.min())

        x_img = (x_norm * (image_width - 1)).astype(np.int32)
        y_img = (y_norm * (image_height - 1)).astype(np.int32)

        image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        image[y_img, x_img] = (255, 255, 255)
        return image
        
    def adjust_points_by_speed(self, points: np.ndarray, actual_speed: float, acquisition_speed: float) -> np.ndarray:
        """根据实际速度和采集速度的比例调整点云的y坐标
        
        Args:
            points: 点云坐标数组
            actual_speed: 实际运动速度
            acquisition_speed: 采集时的速度
            
        Returns:
            调整后的点云坐标数组
        """
        speed_ratio = actual_speed / acquisition_speed
        points[:, 1] = points[:, 1] * speed_ratio  # y轴坐标除以速度比
        return points

    def upload_point_cloud(self, file_content: bytes, actual_speed: float = 100, acquisition_speed: float = 100) -> Tuple[o3d.geometry.PointCloud, float]:
        """上传并处理点云文件
        
        Args:
            file_content: 上传的文件内容
            actual_speed: 实际运动速度
            acquisition_speed: 采集时的速度
            
        Returns:
            点云对象和文件大小的元组
            
        Raises:
            RuntimeError: 当文件处理失败时
        """
        try:
            # 保存上传的文件
            temp_ply_path = os.path.join(self.temp_dir, 'temp.ply')
            with open(temp_ply_path, "wb") as buffer:
                buffer.write(file_content)
            
            # 读取点云
            point_cloud = o3d.io.read_point_cloud(temp_ply_path)
            file_size_mb = len(file_content) / (1024 * 1024)
            
            self.logger.info(f"成功读取点云文件，共 {len(point_cloud.points)} 个点，"
                           f"文件大小为 {file_size_mb:.2f} MB。")
            
            # 读取并调整点云
            points = np.array(point_cloud.points)
            adjusted_points = self.adjust_points_by_speed(points, actual_speed, acquisition_speed)
            point_cloud.points = o3d.utility.Vector3dVector(adjusted_points)
            
            # 生成可视化和更新信息
            points = adjusted_points
            self.generate_views(points)
            self.update_cloud_info(points)
            
            return point_cloud, file_size_mb
            
        except Exception as e:
            self.logger.error(f"点云文件处理失败: {str(e)}")
            raise RuntimeError(f"点云文件处理失败: {str(e)}")

    def generate_views(self, points: np.ndarray) -> None:
        """生成点云三视图
        
        Args:
            points: 点云数据数组
        """
        try:
            img_xy = self.normalize_and_map(points[:, 0], points[:, 1])
            img_yz = self.normalize_and_map(points[:, 1], points[:, 2])
            img_xz = self.normalize_and_map(points[:, 0], points[:, 2])
            
            cv2.imwrite(os.path.join(self.temp_dir, 'xy.jpg'), img_xy)
            cv2.imwrite(os.path.join(self.temp_dir, 'yz.jpg'), img_yz)
            cv2.imwrite(os.path.join(self.temp_dir, 'xz.jpg'), img_xz)
            
        except Exception as e:
            self.logger.error(f"生成三视图失败: {str(e)}")
            raise RuntimeError(f"生成三视图失败: {str(e)}")
            
    def update_cloud_info(self, points: np.ndarray) -> None:
        """更新点云信息到yml文件
        
        Args:
            points: 点云数据数组
        """
        try:
            x_min = np.min(points[:, 0])
            x_max = np.max(points[:, 0])
            y_min = np.min(points[:, 1])
            y_max = np.max(points[:, 1])
            z_min = np.min(points[:, 2])
            z_max = np.max(points[:, 2])

            with open(os.path.join(self.temp_dir, 'info.yml'), 'w') as f:
                f.write(f"x_min: {x_min}\n")
                f.write(f"x_max: {x_max}\n")
                f.write(f"y_min: {y_min}\n")
                f.write(f"y_max: {y_max}\n")
                f.write(f"z_min: {z_min}\n")
                f.write(f"z_max: {z_max}\n")
                
        except Exception as e:
            self.logger.error(f"更新点云信息失败: {str(e)}")
            raise RuntimeError(f"更新点云信息失败: {str(e)}")
