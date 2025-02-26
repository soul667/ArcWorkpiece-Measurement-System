from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import open3d as o3d
import cv2
import numpy as np
from UserInterface.PointCouldProgress import *
from UserInterface.cylinder_processor import CylinderProcessor
from UserInterface.point_cloud_manager import PointCloudManager
from UserInterface.point_cloud_processor import PointCloudProcessor
from typing import Optional
from io import BytesIO
import logging

log_file_path = "./UserInterface/fastapi.log"
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

from algorithm.axis_projection import AxisProjector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory="UserInterface/assets"), name="assets")
templates = Jinja2Templates(directory="templates")

# 创建处理器实例
cloud_manager = PointCloudManager()
cloud_processor = PointCloudProcessor()
cylinder_processor = CylinderProcessor()

# 全局变量
global_source_point_cloud = None
settings = {
    'show': True  
}

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})

@app.get("/img/{img_name}")
async def get_image(img_name: str, v: Optional[str] = None):
    img_path = os.path.join("UserInterface/assets", "temp", f"{img_name}.jpg")
    logger.info(f"Requested image: {img_name} with version: {v}")
    if os.path.exists(img_path):
        # 获取文件最后修改时间作为版本号
        file_version = str(os.path.getmtime(img_path))
        response = FileResponse(img_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"  # 1年缓存
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/yml/{yml_name}")
async def get_yml(yml_name: str, v: Optional[str] = None):
    yml_path = os.path.join("UserInterface/assets", "temp", f"{yml_name}.yml")
    logger.info(f"Requested YAML file: {yml_name}")
    logger.info(f"Full path to YAML file: {yml_path}")
    
    if os.path.exists(yml_path):
        logger.info(f"YAML file found: {yml_path}")
        # 获取文件最后修改时间作为版本号
        file_version = str(os.path.getmtime(yml_path))
        response = FileResponse(yml_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"  # 1年缓存
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        logger.error(f"YAML file not found: {yml_path}")
        raise HTTPException(status_code=404, detail="YAML file not found")

@app.get("/get_ply/{ply_name}")
async def get_ply(ply_name: str, v: Optional[str] = None):
    ply_path = os.path.join("UserInterface/assets", "temp", f"{ply_name}.ply")
    if os.path.exists(ply_path):
        # 获取文件最后修改时间作为版本号
        file_version = str(os.path.getmtime(ply_path))
        response = FileResponse(ply_path, media_type='application/octet-stream')
        response.headers["Cache-Control"] = "public, max-age=31536000"  # 1年缓存
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="PLY file not found")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """处理点云文件上传"""
    if not file:
        return JSONResponse(status_code=400, content={"error": "未提供文件"})

    try:
        # 读取并处理文件
        file_content = await file.read()
        global global_source_point_cloud
        global_source_point_cloud, file_size_mb = cloud_manager.upload_point_cloud(file_content)
        
        return JSONResponse(
            status_code=200, 
            content={"message": f"文件上传成功，大小: {file_size_mb:.2f} MB"}
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/crop")
async def submit(data: dict):
    """处理点云裁剪请求"""
    if not data:
        return JSONResponse(status_code=400, content={"error": "未接收到数据"})
    
    logger.info(f"接收到裁剪请求数据: {data}")
    
    global global_source_point_cloud
    if global_source_point_cloud is None:
        # 尝试从文件中读取点云数据
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        if not os.path.exists(temp_ply_path):
            return JSONResponse(status_code=400, content={"error": "无可用的点云数据"})
        global_source_point_cloud = o3d.io.read_point_cloud(temp_ply_path)

    try:
        # 执行裁剪
        data_region = data.get('regions', None)
        data_mode = data.get('modes', None)
        cropped_pcd = cloud_processor.crop_point_cloud(
            global_source_point_cloud,
            data_region,
            data_mode
        )

        # 显示3D视图
        if data.get('settings', {}).get('show', False):
            o3d.visualization.draw_geometries([cropped_pcd])

        # 更新点云数据
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, cropped_pcd)
        global_source_point_cloud = cropped_pcd

        # 更新可视化
        points = np.array(cropped_pcd.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)

        return JSONResponse(status_code=200, content={"status": "success", "received": data})

    except Exception as e:
        logger.error(f"点云裁剪失败: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/denoise")
async def denoise(data: dict):
    """处理点云去噪请求"""
    if not data:
        return JSONResponse(status_code=400, content={"error": "未接收到数据"})
    
    logger.info(f"接收到去噪请求数据: {data}")
    
    global global_source_point_cloud
    if global_source_point_cloud is None:
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        if not os.path.exists(temp_ply_path):
            return JSONResponse(status_code=400, content={"error": "无可用的点云数据"})
        global_source_point_cloud = o3d.io.read_point_cloud(temp_ply_path)

    try:
        # 执行去噪
        denoised_pcd = cloud_processor.denoise_point_cloud(
            global_source_point_cloud,
            nb_neighbors=data.get('nb_neighbors', 100),
            std_ratio=data.get('std_ratio', 0.5)
        )

        # 显示3D视图
        if data.get('settings', {}).get('show', False):
            o3d.visualization.draw_geometries([denoised_pcd])

        # 更新点云数据
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, denoised_pcd)
        global_source_point_cloud = denoised_pcd

        # 更新可视化
        points = np.array(denoised_pcd.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)

        return JSONResponse(status_code=200, content={"status": "success", "received": data})

    except Exception as e:
        logger.error(f"点云去噪失败: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process")
async def process(data: dict):
    """处理点云数据拟合圆柱体"""
    if not data:
        return JSONResponse(status_code=400, content={"error": "未接收到数据"})
    logger.info(f"接收到处理请求数据: {data}")
    
    # 获取点云数据
    global global_source_point_cloud
    if global_source_point_cloud is None:
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        if not os.path.exists(temp_ply_path):
            return JSONResponse(status_code=400, content={"error": "无可用的点云数据"})
        global_source_point_cloud = o3d.io.read_point_cloud(temp_ply_path)

    # 获取请求参数并处理
    try:
        points = np.asarray(global_source_point_cloud.points)
        result = cylinder_processor.process_cylinder_fitting(
            points=points,
            cylinder_method=data.get('cylinderMethod', 'RANSAC'),
            normal_neighbors=data.get('normalNeighbors', 30),
            ransac_threshold=data.get('ransacThreshold', 0.1),
            min_radius=data.get('minRadius', 6),
            max_radius=data.get('maxRadius', 11)
        )
        return JSONResponse(status_code=200, content=result)
        
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logger.error(f"处理点云数据时出错: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"点云处理失败: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9304)
