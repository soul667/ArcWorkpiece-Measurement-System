import open3d as o3d
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PointCloudDenoiser:
    def __init__(self):
        """Initialize the point cloud denoiser."""
        pass
    
    def denoise_point_cloud(self, point_cloud: o3d.geometry.PointCloud, nb_neighbors: int = 100, std_ratio: float = 0.5) -> o3d.geometry.PointCloud:
        """
        Remove noise from point cloud using statistical outlier removal.
        
        Args:
            point_cloud (o3d.geometry.PointCloud): Input point cloud to denoise
            nb_neighbors (int): Number of neighbors to use for mean distance calculation
            std_ratio (float): Standard deviation ratio threshold
            
        Returns:
            o3d.geometry.PointCloud: Denoised point cloud
        """
        try:
            # Apply statistical outlier removal
            denoised_cloud, _ = point_cloud.remove_statistical_outlier(
                nb_neighbors=nb_neighbors,
                std_ratio=std_ratio
            )
            
            return denoised_cloud
            
        except Exception as e:
            logger.error(f"Point cloud denoising failed: {str(e)}")
            raise
