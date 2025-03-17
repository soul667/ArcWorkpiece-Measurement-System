import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from UserInterface.api.config import logger, TEMP_DIR

router = APIRouter()

@router.get("/img/{img_name}")
async def get_image(img_name: str, v: Optional[str] = None):
    """提供图片文件服务
    
    Args:
        img_name: 图片名称（不含扩展名）
        v: 版本号（用于缓存控制）
    """
    img_path = os.path.join(TEMP_DIR, f"{img_name}.jpg")
    logger.info(f"请求图片: {img_name}, 版本: {v}")
    
    if os.path.exists(img_path):
        file_version = str(os.path.getmtime(img_path))
        response = FileResponse(img_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="未找到图片")

@router.get("/yml/{yml_name}")
async def get_yml(yml_name: str, v: Optional[str] = None):
    """提供YAML文件服务
    
    Args:
        yml_name: YAML文件名称（不含扩展名）
        v: 版本号（用于缓存控制）
    """
    yml_path = os.path.join(TEMP_DIR, f"{yml_name}.yml")
    logger.info(f"请求YAML文件: {yml_name}")
    
    if os.path.exists(yml_path):
        file_version = str(os.path.getmtime(yml_path))
        response = FileResponse(yml_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="未找到YAML文件")

@router.get("/model/{model_name}")
async def get_model(model_name: str):
    """提供模型文件服务
    
    Args:
        model_name: 模型文件名称（不含扩展名）
    """
    model_path = os.path.join(TEMP_DIR, f"{model_name}.onnx")
    
    if os.path.exists(model_path):
        return FileResponse(
            model_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={model_name}.onnx"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="未找到模型文件")

@router.get("/cloud/{cloud_name}")
async def get_cloud(cloud_name: str):
    """提供点云文件服务
    
    Args:
        cloud_name: 点云文件名称（不含扩展名）
    """
    cloud_path = os.path.join(TEMP_DIR, f"{cloud_name}.ply")
    
    if os.path.exists(cloud_path):
        return FileResponse(
            cloud_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={cloud_name}.ply"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="未找到点云文件")
