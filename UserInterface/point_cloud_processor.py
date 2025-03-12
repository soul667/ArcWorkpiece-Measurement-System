import os
import numpy as np
import open3d as o3d
import logging
from typing import Dict, Tuple, Optional, Union, List
from UserInterface.PointCouldProgress import segmentPointCloud

class PointCloudProcessor:
    """点云处理类：处理点云裁剪和去噪等操作"""
    
    def __init__(self, temp_dir: str = "UserInterface/assets/temp"):
        """初始化点云处理器
        
        Args:
            temp_dir: 临时文件存储目录
        """
        self.temp_dir = temp_dir
        self.logger = logging.getLogger(__name__)
        
    def crop_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        regions: dict,
        modes: dict
    ) -> o3d.geometry.PointCloud:
        """裁剪点云
        
        Args:
            point_cloud: 输入点云
            regions: 裁剪区域 {x_regions, y_regions, z_regions}
            modes: 裁剪模式 {x_mode, y_mode, z_mode}
            
        Returns:
            裁剪后的点云
            
        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当裁剪操作失败时
        """
        try:
            if not isinstance(point_cloud, o3d.geometry.PointCloud):
                raise ValueError("无效的点云对象")
                
            if not regions or not modes:
                raise ValueError("未提供裁剪区域或模式")
                
            # 执行裁剪
            points = np.array(point_cloud.points)
            filtered_points = segmentPointCloud(
                points,
                regions.get('x_regions', None),
                regions.get('y_regions', None),
                regions.get('z_regions', None),
                modes.get('x_mode', 'keep'),
                modes.get('y_mode', 'keep'),
                modes.get('z_mode', 'keep')
            )
            
            # 创建新的点云对象
            cropped_pcd = o3d.geometry.PointCloud()
            cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
            
            self.logger.info(f"裁剪成功，剩余点数: {len(cropped_pcd.points)}")
            return cropped_pcd
            
        except Exception as e:
            self.logger.error(f"点云裁剪失败: {str(e)}")
            raise RuntimeError(f"点云裁剪失败: {str(e)}")
            
    def denoise_point_cloud(
        self,
        point_cloud: o3d.geometry.PointCloud,
        nb_neighbors: int = 100,
        std_ratio: float = 0.5
    ) -> o3d.geometry.PointCloud:
        """点云去噪
        
        Args:
            point_cloud: 输入点云
            nb_neighbors: 邻居点数量
            std_ratio: 标准差比率
            
        Returns:
            去噪后的点云
            
        Raises:
            ValueError: 当输入参数无效时
            RuntimeError: 当去噪操作失败时
        """
        try:
            if not isinstance(point_cloud, o3d.geometry.PointCloud):
                raise ValueError("无效的点云对象")
                
            if nb_neighbors <= 0:
                raise ValueError(f"邻居点数必须大于0: {nb_neighbors}")
                
            if std_ratio <= 0:
                raise ValueError(f"标准差比率必须大于0: {std_ratio}")
                
            # 执行统计滤波去噪
            denoised_pcd, ind = point_cloud.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            self.logger.info(f"去噪成功，剩余点数: {len(denoised_pcd.points)}")
            return denoised_pcd
            
        except Exception as e:
            self.logger.error(f"点云去噪失败: {str(e)}")
            raise RuntimeError(f"点云去噪失败: {str(e)}")
