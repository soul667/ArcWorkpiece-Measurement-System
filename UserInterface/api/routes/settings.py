import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, List
from datetime import datetime

from UserInterface.auth.service import get_current_user
from UserInterface.auth.db import Database
from UserInterface.api.models.settings import (
    SaveSettingsRequest,
    LineSettings,
    SettingsResponse
)
from UserInterface.api.config import logger

router = APIRouter()

@router.post("/save")
async def save_settings(data: dict, current_user: dict = Depends(get_current_user)):
    """保存参数设置到数据库"""
    try:
        # 打印接收到的完整数据
        logger.info("Received settings save request:")
        logger.info(f"Full data: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        name = data.get('name')
        cylinder_settings = data.get('cylinderSettings')
        arc_settings = data.get('arcSettings')
        
        # 打印每个部分的数据
        logger.info(f"Name: {name}")
        logger.info(f"Cylinder settings: {json.dumps(cylinder_settings, indent=2, ensure_ascii=False)}")
        logger.info(f"Arc settings: {json.dumps(arc_settings, indent=2, ensure_ascii=False)}")
        
        if not all([name, cylinder_settings, arc_settings]):
            missing = []
            if not name: missing.append('name')
            if not cylinder_settings: missing.append('cylinderSettings')
            if not arc_settings: missing.append('arcSettings')
            error_msg = f"缺少必要参数: {', '.join(missing)}"
            logger.warning(error_msg)
            return JSONResponse(
                status_code=400, 
                content={"error": error_msg}
            )
        
        query = """
            INSERT INTO parameter_settings 
            (user_id, name, cylinder_settings, arc_settings) 
            VALUES (%s, %s, %s, %s)
        """
        
        db = Database()
        params = (
            current_user['id'],
            name,
            json.dumps(cylinder_settings),
            json.dumps(arc_settings)
        )
        db.execute_query(query, params)
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "设置保存成功"}
        )
    except Exception as e:
        logger.error(f"保存设置失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"保存设置失败: {str(e)}"}
        )


@router.get("/list")
async def list_settings(current_user: dict = Depends(get_current_user)) -> Dict:
    """获取当前用户的所有参数设置"""
    try:
        query = """
            SELECT id, name, cylinder_settings, arc_settings, created_at 
            FROM parameter_settings 
            WHERE user_id = %s 
            ORDER BY created_at DESC
        """
        
        db = Database()
        results = db.execute_query(query, (current_user['id'],))
        
        settings_list = [{
            'id': item['id'],
            'name': item['name'],
            'cylinderSettings': json.loads(item['cylinder_settings']),
            'arcSettings': json.loads(item['arc_settings']),
            'createdAt': item['created_at'].isoformat()
        } for item in results]
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "data": settings_list}
        )
    except Exception as e:
        logger.error(f"获取设置列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取设置列表失败: {str(e)}")

# @app.get("/api/settings/latest")
@router.get("/latest")
async def get_latest_setting(current_user: dict = Depends(get_current_user)):
    """获取当前用户最新的参数设置"""
    try:
        query = """
            SELECT cylinder_settings, arc_settings 
            FROM parameter_settings 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 1
        """
        
        db = Database()
        results = db.execute_query(query, (current_user['id'],))
        
        if not results:
            return JSONResponse(
                status_code=200,
                content={"status": "success", "data": {}}
            )
            
        item = results[0]
        print(json.loads(item['cylinder_settings']), json.loads(item['arc_settings']))
        setting = {
            'cylinderSettings': json.loads(item['cylinder_settings']),
            'arcSettings': json.loads(item['arc_settings'])
        }
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "data": setting}
        )
    except Exception as e:
        logger.error(f"获取最新设置失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"获取设置失败: {str(e)}"}
        )


@router.delete("/deleteAll")
async def delete_all_settings(current_user: dict = Depends(get_current_user)):
    """删除当前用户的所有设置"""
    try:
        query = """
            DELETE FROM parameter_settings 
            WHERE user_id = %s
        """
        db = Database()
        db.execute_query(query, (current_user['id'],))
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "所有设置已删除"}
        )
    except Exception as e:
        logger.error(f"删除所有设置失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"删除失败: {str(e)}"}
        )

# 这是由于 FastAPI 的路由匹配顺序导致的。在你的路由中，包含了一个带有路径参数的路由：
@router.get("/{setting_id}")
async def get_setting(
    setting_id: int, 
    current_user: dict = Depends(get_current_user)
) -> Dict:
    """获取特定的参数设置"""
    try:
        query = """
            SELECT id, name, cylinder_settings, arc_settings, created_at 
            FROM parameter_settings 
            WHERE id = %s AND user_id = %s
        """
        
        db = Database()
        results = db.execute_query(query, (setting_id, current_user['id']))
        
        if not results:
            raise HTTPException(status_code=404, detail="未找到指定的设置")
            
        item = results[0]
        setting = {
            'id': item['id'],
            'name': item['name'],
            'cylinderSettings': json.loads(item['cylinder_settings']),
            'arcSettings': json.loads(item['arc_settings']),
            'createdAt': item['created_at'].isoformat()
        }
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "data": setting}
        )
    except Exception as e:
        logger.error(f"获取设置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取设置失败: {str(e)}")

@router.put("/{setting_id}")
async def update_setting(
    setting_id: int, 
    data: Dict,
    current_user: dict = Depends(get_current_user)
) -> Dict:
    """更新设置名称"""
    try:
        query = """
            UPDATE parameter_settings 
            SET name = %s
            WHERE id = %s AND user_id = %s
        """
        db = Database()
        db.execute_query(query, (data['name'], setting_id, current_user['id']))
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "更新成功"}
        )
    except Exception as e:
        logger.error(f"更新设置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")

@router.delete("/{setting_id}")
async def delete_setting(
    setting_id: int,
    current_user: dict = Depends(get_current_user)
) -> Dict:
    """删除指定的设置"""
    try:
        query = """
            DELETE FROM parameter_settings 
            WHERE id = %s AND user_id = %s
        """
        db = Database()
        db.execute_query(query, (setting_id, current_user['id']))
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "删除成功"}
        )
    except Exception as e:
        logger.error(f"删除设置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

