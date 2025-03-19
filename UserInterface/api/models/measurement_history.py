from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class MeasurementRecord(BaseModel):
    """测量记录模型"""
    id: Optional[int] = None
    cloud_name: str
    timestamp: str
    created_at: Optional[str] = None
    
    # 测量结果
    radius: float
    axis_vector_x: float
    axis_vector_y: float
    axis_vector_z: float
    axis_point_x: float
    axis_point_y: float
    axis_point_z: float
    
    # 投影图路径
    original_projection: Optional[str] = None
    axis_projection: Optional[str] = None

class MeasurementRecordList(BaseModel):
    """测量记录列表响应"""
    status: str
    data: List[MeasurementRecord]

class MeasurementStatistics(BaseModel):
    """统计数据模型"""
    average_radius: float
    radius_std_dev: float
    average_axis_vector: dict
    axis_vector_std_dev: dict
    sample_count: int

class ExportRequest(BaseModel):
    """导出报告请求"""
    record_ids: List[int]
