import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Optional
from UserInterface.api.config import logger, TEMP_DIR

router = APIRouter()

@router.get("/img/{path:path}")
async def get_image(path: str, v: Optional[str] = None):
    """提供图片文件服务
    
    Args:
        path: 图片路径（相对于temp目录）
        v: 版本号（用于缓存控制）
    """
    img_path = os.path.join(TEMP_DIR, path)
    logger.info(f"请求图片: {path}, 版本: {v}")
    
    # 安全检查，防止目录遍历
    if not os.path.abspath(img_path).startswith(os.path.abspath(TEMP_DIR)):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    if os.path.exists(img_path):
        file_version = str(os.path.getmtime(img_path))
        response = FileResponse(img_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="未找到图片")

@router.get("/yml/{path:path}")
async def get_yml(path: str, v: Optional[str] = None):
    """提供YAML文件服务
    
    Args:
        path: YAML文件路径（相对于temp目录）
        v: 版本号（用于缓存控制）
    """
    yml_path = os.path.join(TEMP_DIR, path)
    logger.info(f"请求YAML文件: {path}")
    
    # 安全检查，防止目录遍历
    if not os.path.abspath(yml_path).startswith(os.path.abspath(TEMP_DIR)):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    if os.path.exists(yml_path):
        file_version = str(os.path.getmtime(yml_path))
        response = FileResponse(yml_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="未找到YAML文件")

@router.get("/model/{path:path}")
async def get_model(path: str):
    """提供模型文件服务
    
    Args:
        path: 模型文件路径（相对于temp目录）
    """
    model_path = os.path.join(TEMP_DIR, path)
    
    # 安全检查，防止目录遍历
    if not os.path.abspath(model_path).startswith(os.path.abspath(TEMP_DIR)):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    if os.path.exists(model_path):
        return FileResponse(
            model_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(path)}"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="未找到模型文件")

@router.get("/cloud/{path:path}")
async def get_cloud(path: str):
    """提供点云文件服务
    
    Args:
        path: 点云文件路径（相对于temp目录）
    """
    cloud_path = os.path.join(TEMP_DIR, path)
    
    # 安全检查，防止目录遍历
    if not os.path.abspath(cloud_path).startswith(os.path.abspath(TEMP_DIR)):
        raise HTTPException(status_code=403, detail="访问被拒绝")
    
    if os.path.exists(cloud_path):
        return FileResponse(
            cloud_path,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={os.path.basename(path)}"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="未找到点云文件")
