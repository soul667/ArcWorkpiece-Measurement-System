import os
import logging
from datetime import datetime
import open3d as o3d
import numpy as np
from .auth.db import Database

logger = logging.getLogger(__name__)

class TempCloudManager:
    def __init__(self):
        self.base_dir = os.path.join("UserInterface/assets", "temp")
        
    def store_cloud(self, point_cloud, user_id):
        """存储点云及其视图到临时目录"""
        try:
            # 创建时间戳目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cloud_dir = os.path.join(self.base_dir, timestamp)
            os.makedirs(cloud_dir, exist_ok=True)
            
            # 保存点云文件
            cloud_path = os.path.join(cloud_dir, "cloud.ply")
            o3d.io.write_point_cloud(cloud_path, point_cloud)
            
            # 生成三视图
            points = np.asarray(point_cloud.points)
            self.generate_views(points, cloud_dir)
            
            # 记录到数据库
            db = Database()
            query = """
                INSERT INTO temp_clouds 
                (user_id, timestamp) 
                VALUES (%s, %s)
            """
            cloud_id = db.execute_query(query, (user_id, timestamp))
            
            return {
                "id": cloud_id,
                "timestamp": timestamp,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"存储点云失败: {str(e)}")
            raise
            
    def list_clouds(self, user_id):
        """获取用户的暂存点云列表"""
        try:
            db = Database()
            query = """
                SELECT id, timestamp, created_at
                FROM temp_clouds
                WHERE user_id = %s
                ORDER BY created_at DESC
            """
            clouds = db.execute_query(query, (user_id,))
            
            result = []
            for cloud in clouds:
                result.append({
                    "id": cloud["id"],
                    "timestamp": cloud["timestamp"],
                    "createdAt": cloud["created_at"].isoformat(),
                    "views": {
                        "xy": f"{cloud['timestamp']}/xy_view.jpg",
                        "yz": f"{cloud['timestamp']}/yz_view.jpg",
                        "xz": f"{cloud['timestamp']}/xz_view.jpg"
                    }
                })
            
            return result
            
        except Exception as e:
            logger.error(f"获取点云列表失败: {str(e)}")
            raise
            
    def load_cloud(self, timestamp, user_id):
        """加载指定的点云"""
        try:
            # 验证点云所有权
            db = Database()
            query = """
                SELECT id
                FROM temp_clouds
                WHERE timestamp = %s AND user_id = %s
            """
            results = db.execute_query(query, (timestamp, user_id))
            
            if not results:
                raise ValueError("未找到指定点云或无权访问")
            
            # 加载点云文件
            cloud_path = os.path.join(self.base_dir, timestamp, "cloud.ply")
            if not os.path.exists(cloud_path):
                raise ValueError("点云文件不存在")
                
            point_cloud = o3d.io.read_point_cloud(cloud_path)
            return point_cloud
            
        except Exception as e:
            logger.error(f"加载点云失败: {str(e)}")
            raise
            
    def generate_views(self, points: np.ndarray, output_dir: str):
        """生成点云的三视图"""
        try:
            # 获取点云的边界框
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            
            # 创建图像大小
            image_size = 500
            margin = 50
            
            # 生成三个视图
            # XY视图（俯视图）
            xy_img = self._generate_projection(
                points, 0, 1, image_size, margin,
                min_bound, max_bound
            )
            
            # YZ视图（侧视图）
            yz_img = self._generate_projection(
                points, 1, 2, image_size, margin,
                min_bound, max_bound
            )
            
            # XZ视图（正视图）
            xz_img = self._generate_projection(
                points, 0, 2, image_size, margin,
                min_bound, max_bound
            )
            
            # 保存图像
            import cv2
            cv2.imwrite(os.path.join(output_dir, "xy_view.jpg"), xy_img)
            cv2.imwrite(os.path.join(output_dir, "yz_view.jpg"), yz_img)
            cv2.imwrite(os.path.join(output_dir, "xz_view.jpg"), xz_img)
            
        except Exception as e:
            logger.error(f"生成视图失败: {str(e)}")
            raise
            
    def _generate_projection(self, points, axis1, axis2, size, margin, min_bound, max_bound):
        """生成指定轴平面的投影图"""
        import numpy as np
        import cv2
        
        # 创建空白图像
        img = np.zeros((size, size), dtype=np.uint8)
        
        # 计算缩放因子
        scale = (size - 2 * margin) / max(
            max_bound[axis1] - min_bound[axis1],
            max_bound[axis2] - min_bound[axis2]
        )
        
        # 投影点并绘制
        projected_points = points[:, [axis1, axis2]]
        
        # 归一化到图像坐标
        projected_points = (projected_points - min_bound[[axis1, axis2]]) * scale + margin
        projected_points = projected_points.astype(np.int32)
        
        # 确保点在图像范围内
        mask = (
            (projected_points[:, 0] >= 0) &
            (projected_points[:, 0] < size) &
            (projected_points[:, 1] >= 0) &
            (projected_points[:, 1] < size)
        )
        projected_points = projected_points[mask]
        
        # 在图像上标记点
        img[projected_points[:, 1], projected_points[:, 0]] = 255
        
        # 应用高斯模糊使点更明显
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        return img
