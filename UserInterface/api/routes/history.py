from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse, Response
from typing import List, Optional
import logging
import os
from datetime import datetime

from UserInterface.auth.service import get_current_user
from UserInterface.auth.db import Database
from UserInterface.api.models.measurement_history import (
    MeasurementRecord, 
    MeasurementRecordList,
    MeasurementStatistics,
    ExportRequest
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/list")
async def list_records(current_user: dict = Depends(get_current_user)):
    """获取历史记录列表"""
    try:
        db = Database()
        query = """
            SELECT id, cloud_name, timestamp, created_at,
                   radius, 
                   axis_vector_x, axis_vector_y, axis_vector_z,
                   axis_point_x, axis_point_y, axis_point_z,
                   original_projection, axis_projection
            FROM measurement_history
            WHERE user_id = %s
            ORDER BY created_at DESC
        """
        results = db.execute_query(query, (current_user['id'],))
        # print(results)
        if not results:
            return {"status": "success", "data": []}
        records = [MeasurementRecord(
            id=item['id'],
            cloud_name=item['cloud_name'],
            timestamp=item['timestamp'],
            created_at=item['created_at'].isoformat(),
            radius=item['radius'],
            axis_vector_x=item['axis_vector_x'],
            axis_vector_y=item['axis_vector_y'],
            axis_vector_z=item['axis_vector_z'],
            axis_point_x=item['axis_point_x'],
            axis_point_y=item['axis_point_y'],
            axis_point_z=item['axis_point_z'],
            original_projection=item['original_projection'],
            axis_projection=item['axis_projection']
        ) for item in results]

        return {"status": "success", "data": records}

    except Exception as e:
        logger.error(f"获取历史记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/store")
async def store_record(
    record: MeasurementRecord,
    current_user: dict = Depends(get_current_user)
):
    """存储新的测量记录"""
    try:
        db = Database()
        query = """
            INSERT INTO measurement_history (
                user_id, cloud_name, timestamp, 
                radius,
                axis_vector_x, axis_vector_y, axis_vector_z,
                axis_point_x, axis_point_y, axis_point_z,
                original_projection, axis_projection
            ) VALUES (
                %s, %s, %s, %s, 
                %s, %s, %s, 
                %s, %s, %s,
                %s, %s
            )
        """
        values = (
            current_user['id'],
            record.cloud_name,
            record.timestamp,
            record.radius,
            record.axis_vector_x,
            record.axis_vector_y,
            record.axis_vector_z,
            record.axis_point_x,
            record.axis_point_y,
            record.axis_point_z,
            record.original_projection,
            record.axis_projection
        )
        
        db.execute_query(query, values)
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "记录已保存"}
        )

    except Exception as e:
        logger.error(f"保存测量记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{timestamp}")
async def delete_record(
    timestamp: str,
    current_user: dict = Depends(get_current_user)
):
    """删除指定的测量记录"""
    try:
        db = Database()
        
        # 获取记录信息
        query = """
            SELECT original_projection, axis_projection
            FROM measurement_history
            WHERE timestamp = %s AND user_id = %s
        """
        results = db.execute_query(query, (timestamp, current_user['id']))
        if not results:
            raise HTTPException(status_code=404, detail="记录不存在")
        
        # 删除关联的投影图文件
        for image_path in [results[0]['original_projection'], results[0]['axis_projection']]:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        
        # 删除数据库记录
        query = """
            DELETE FROM measurement_history
            WHERE timestamp = %s AND user_id = %s
        """
        db.execute_query(query, (timestamp, current_user['id']))
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "记录已删除"}
        )

    except Exception as e:
        logger.error(f"删除测量记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{timestamp}")
async def update_record(timestamp: str, data: dict, current_user: dict = Depends(get_current_user)):
    """更新指定记录的名称"""
    try:
        db = Database()
        query = """
            UPDATE measurement_history
            SET cloud_name = %s
            WHERE timestamp = %s AND user_id = %s
        """
        db.execute_query(query, (data['cloud_name'], timestamp, current_user['id']))
        return {"status": "success", "message": "名称更新成功"}
    except Exception as e:
        logger.error(f"更新测量记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_report(
    request: ExportRequest,
    current_user: dict = Depends(get_current_user)
):
    """导出选中记录的测量报告"""
    try:
        if not request.record_ids:
            raise HTTPException(status_code=400, detail="请选择要导出的记录")

        db = Database()
        # 获取记录数据,确保只获取属于当前用户的记录
        query = """
            SELECT *
            FROM measurement_history
            WHERE id IN %s AND user_id = %s
            ORDER BY created_at DESC
        """
        results = db.execute_query(query, (tuple(request.record_ids), current_user['id']))
        
        if not results:
            raise HTTPException(status_code=404, detail="未找到记录")

        # TODO: 生成PDF报告
        # 这里需要实现报告生成逻辑
        
        # 临时返回一个空PDF作为示例
        pdf_content = b"%PDF-1.7\n1 0 obj\n<</Type/Catalog>>\nendobj\ntrailer\n<</Root 1 0 R>>\n%%EOF"
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=measurement_report_{current_user['id']}.pdf"
            }
        )

    except HTTPException as he:
        logger.warning(f"导出报告请求错误: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"导出报告失败: {str(e)}")
        raise HTTPException(status_code=500, detail="导出报告失败，请稍后重试")
