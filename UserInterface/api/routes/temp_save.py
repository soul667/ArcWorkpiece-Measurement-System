import os
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List

from UserInterface.auth.service import get_current_user
from UserInterface.auth.db import Database
from UserInterface.point_cloud_manager import PointCloudManager
from ..models.point_cloud import TempCloudResponse, TempCloudListResponse, StoreCloudResponse
# from UserInterface.point_cloud_manager import generate_views
# from 
# generate_views(points)
logger = logging.getLogger(__name__)
router = APIRouter()

class TempCloudStorage:
    def __init__(self):
        self.base_dir = os.path.join("UserInterface/assets", "temp")
        self.cloud_manager = PointCloudManager()

    def ensure_temp_dir(self, timestamp: str) -> str:
        """确保临时目录存在并返回路径"""
        dir_path = os.path.join(self.base_dir, timestamp)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

temp_storage = TempCloudStorage()

@router.post("/store")
async def store_cloud(current_user: dict = Depends(get_current_user)) -> StoreCloudResponse:
    """暂存当前点云"""
    try:
        # 获取当前点云
        point_cloud,success = temp_storage.cloud_manager.get_current_cloud()
        if point_cloud is None:
            raise HTTPException(status_code=400, detail="无可用点云")

        # 创建时间戳目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cloud_dir = temp_storage.ensure_temp_dir(timestamp)

        # 保存点云文件
        cloud_path = os.path.join(cloud_dir, "cloud.ply")
        temp_storage.cloud_manager.save_point_cloud(point_cloud, cloud_path)

        # 生成三视图
        points = temp_storage.cloud_manager.get_points(point_cloud)
        temp_storage.cloud_manager.generate_views(points, 
            prefix=os.path.join(cloud_dir, "view"))

        # 保存到数据库
        db = Database()
        query = """
            INSERT INTO temp_clouds (user_id, timestamp)
            VALUES (%s, %s)
        """
        db.execute_query(query, (current_user['id'], timestamp))

        return StoreCloudResponse(
            status="success",
            message="点云已暂存",
            timestamp=timestamp
        )

    except Exception as e:
        logger.error(f"点云暂存失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_clouds(current_user: dict = Depends(get_current_user)) -> TempCloudListResponse:
    """获取暂存的点云列表"""
    try:
        db = Database()
        query = """
            SELECT id, timestamp, created_at
            FROM temp_clouds
            WHERE user_id = %s
            ORDER BY created_at DESC
        """
        results = db.execute_query(query, (current_user['id'],))
        
        clouds = [TempCloudResponse(
            id=item['id'],
            timestamp=item['timestamp'],
            views={
                'xy': f"{os.path.join(item['timestamp'], 'view')}_xy.jpg",
                'yz': f"{os.path.join(item['timestamp'], 'view')}_yz.jpg",
                'xz': f"{os.path.join(item['timestamp'], 'view')}_xz.jpg"
            },
            created_at=item['created_at'].isoformat()
        ) for item in results]
        
        return TempCloudListResponse(
            status="success",
            data=clouds
        )
        
    except Exception as e:
        logger.error(f"获取点云列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{timestamp}/load")
async def load_cloud(
    timestamp: str,
    current_user: dict = Depends(get_current_user)
):
    """加载指定的点云"""
    try:
        # 验证点云所有权
        db = Database()
        query = """
            SELECT id FROM temp_clouds
            WHERE timestamp = %s AND user_id = %s
        """
        results = db.execute_query(query, (timestamp, current_user['id']))
        if not results:
            raise HTTPException(status_code=404, detail="未找到指定点云")

        # 加载点云文件
        cloud_path = os.path.join(temp_storage.base_dir, timestamp, "cloud.ply")
        if not os.path.exists(cloud_path):
            raise HTTPException(status_code=404, detail="点云文件不存在")

        # 加载到系统
        temp_storage.cloud_manager.load_point_cloud(cloud_path)
        # 更新三视图
        # PointCloudManager.generate_views(PointCloudManager.current_cloud.points)
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "点云已加载"}
        )

    except Exception as e:
        logger.error(f"加载点云失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{timestamp}")
async def delete_cloud(
    timestamp: str,
    current_user: dict = Depends(get_current_user)
):
    """删除指定的点云"""
    try:
        # 验证点云所有权
        db = Database()
        query = """
            DELETE FROM temp_clouds
            WHERE timestamp = %s AND user_id = %s
        """
        db.execute_query(query, (timestamp, current_user['id']))

        # 删除文件目录
        cloud_dir = os.path.join(temp_storage.base_dir, timestamp)
        if os.path.exists(cloud_dir):
            import shutil
            shutil.rmtree(cloud_dir)

        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "点云已删除"}
        )

    except Exception as e:
        logger.error(f"删除点云失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
