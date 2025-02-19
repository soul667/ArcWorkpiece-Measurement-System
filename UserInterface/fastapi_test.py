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

# 全局变量，用于存储点云对象
global_source_point_cloud = None
global_source_point_cloud_down = None
settings = {
    'show': True  
}
def normalize_and_map(x, y, image_width=1280, image_height=720):
    """归一化并映射 x 和 y 坐标到指定图像尺寸的坐标,并返回 OpenCV 图片。

    参数:
        x (numpy.ndarray): 输入的 x 坐标数组
        y (numpy.ndarray): 输入的 y 坐标数组
        image_width (int): 图像宽度,默认为 1280
        image_height (int): 图像高度,默认为 720

    返回:
        numpy.ndarray: 归一化并映射后的 OpenCV 图片
    """
    if x.size == 0 or y.size == 0:
        return np.zeros((image_height, image_width, 3), dtype=np.uint8)

    if image_width <= 0 or image_height <= 0:
        raise ValueError("无效的图像尺寸: width={}, height={}".format(image_width, image_height))

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    x_img = (x_norm * (image_width - 1)).astype(np.int32)
    y_img = (y_norm * (image_height - 1)).astype(np.int32)

    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    image[y_img, x_img] = (255, 255, 255)
    return image

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index1.html", {"request": request})

@app.get("/img/{img_name}")
async def get_image(img_name: str):
    img_path = os.path.join("UserInterface/assets", "temp", f"{img_name}.jpg")
    logger.info(f"Requested image: {img_name}")
    if os.path.exists(img_path):
        return FileResponse(img_path)
    else:
        raise HTTPException(status_code=404, detail="Image not found")

@app.get("/yml/{yml_name}")
async def get_yml(yml_name: str):
    yml_path = os.path.join("UserInterface/assets", "temp", f"{yml_name}.yml")
    # if os.path.exists(yml_path):
    #     return FileResponse(yml_path)
    # else:
    #     raise HTTPException(status_code=404, detail="YAML file not found")
    logger.info(f"Requested YAML file: {yml_name}")
    logger.info(f"Full path to YAML file: {yml_path}")
    
    if os.path.exists(yml_path):
        logger.info(f"YAML file found: {yml_path}")
        return FileResponse(yml_path)
    else:
        logger.error(f"YAML file not found: {yml_path}")
        raise HTTPException(status_code=404, detail="YAML file not found")

@app.get("/get_ply/{ply_name}")
async def get_ply(ply_name):
    ply_path = os.path.join("UserInterface/assets", "temp", f"{ply_name}.ply")
    return FileResponse(ply_path, media_type='application/octet-stream')

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"error": "No file part"})


    temp_dir = os.path.join("UserInterface/assets", "temp")
    temp_ply_path = os.path.join(temp_dir, 'temp.ply')
    source_temp_pcd_path = os.path.join(temp_dir, 'source.pcd')

    
    #temp_pcd_path = os.path.join(temp_dir, 'temp.pcd')
    # content = await file.read()
    
    # 使用 BytesIO 将上传的字节流转换为文件对象
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    try:
        # Read file content
        file_content = await file.read()
        
        ply_data = BytesIO(file_content)
        # Try to read point cloud directly from bytes
        
        # Save to temp file for later use
        with open(temp_ply_path, "wb") as buffer:
            buffer.write(file_content)
        
        point_cloud = o3d.io.read_point_cloud(temp_ply_path)
        global global_source_point_cloud
        global_source_point_cloud = point_cloud
        # print(f"成功读取点云文件，共 {len(point_cloud.points)} 个点。")
        file_size_mb = len(file_content) / (1024 * 1024)
        logger.info(f"成功读取点云文件，共 {len(point_cloud.points)} 个点，文件大小为 {file_size_mb:.2f} MB。")
        # logger.info(f"成功读取点云文件，共 {len(point_cloud.points)} 个点。")
        points = np.array(point_cloud.points)
        img_xy = normalize_and_map(points[:, 0], points[:, 1])
        img_yz = normalize_and_map(points[:, 1], points[:, 2])
        img_xz = normalize_and_map(points[:, 0], points[:, 2])
        cv2.imwrite(os.path.join(temp_dir, 'xy.jpg'), img_xy)
        cv2.imwrite(os.path.join(temp_dir, 'yz.jpg'), img_yz)
        cv2.imwrite(os.path.join(temp_dir, 'xz.jpg'), img_xz)
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])

        with open(os.path.join(temp_dir, 'info.yml'), 'w') as f:
            f.write(f"x_min: {x_min}\n")
            f.write(f"x_max: {x_max}\n")
            f.write(f"y_min: {y_min}\n")
            f.write(f"y_max: {y_max}\n")
            f.write(f"z_min: {z_min}\n")
            f.write(f"z_max: {z_max}\n")

        return JSONResponse(status_code=200, content={"message": "File uploaded and saved as both PLY and PCD successfully"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/crop")
async def submit(data: dict):
    if not data:
        return JSONResponse(status_code=400, content={"error": "No data received"})
    # print(data)
    logger.info(f"Crop Received data: {data}")
    #global global_source_point_cloud
    global global_source_point_cloud
    if global_source_point_cloud is None:
        # 尝试从文件中读取点云数据
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        if not os.path.exists(temp_ply_path):
            return JSONResponse(status_code=400, content={"error": "No point cloud data available"})
        point_cloud = o3d.io.read_point_cloud(temp_ply_path)
        global_source_point_cloud = point_cloud
    data_region = data.get('regions', None)
    data_mode = data.get('modes', None)

    filtered_points = segmentPointCloud(
        global_source_point_cloud.points,
        data_region.get('x_regions', None),
        data_region.get('y_regions', None),
        data_region.get('z_regions', None),
        data_mode.get('x_mode', 'keep'),
        data_mode.get('y_mode', 'keep'),
        data_mode.get('z_mode', 'keep')
    )

    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    # show 3d
    if(data.get('settings', None)):
        settings = data.get('settings')
        if settings.get('show', False):
            o3d.visualization.draw_geometries([cropped_pcd])
    # o3d.visualization.draw_geometries([cropped_pcd])
    # cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)

    temp_dir = os.path.join("UserInterface/assets", "temp")
    temp_ply_path = os.path.join(temp_dir, 'temp.ply')
    o3d.io.write_point_cloud(temp_ply_path, cropped_pcd)
    global_source_point_cloud = cropped_pcd
    logger.info(f"Cropped success and point cloud saved to {temp_ply_path}")
    # 重置三视图和信息
    points = np.array(global_source_point_cloud.points)
    img_xy = normalize_and_map(points[:, 0], points[:, 1])
    img_yz = normalize_and_map(points[:, 1], points[:, 2])
    img_xz = normalize_and_map(points[:, 0], points[:, 2])
    cv2.imwrite(os.path.join(temp_dir, 'xy.jpg'), img_xy)
    cv2.imwrite(os.path.join(temp_dir, 'yz.jpg'), img_yz)
    cv2.imwrite(os.path.join(temp_dir, 'xz.jpg'), img_xz)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    with open(os.path.join(temp_dir, 'info.yml'), 'w') as f:
        f.write(f"x_min: {x_min}\n")
        f.write(f"x_max: {x_max}\n")
        f.write(f"y_min: {y_min}\n")
        f.write(f"y_max: {y_max}\n")
        f.write(f"z_min: {z_min}\n")
        f.write(f"z_max: {z_max}\n")
    return JSONResponse(status_code=200, content={"status": "success", "received": data})

@app.post("/model_data_write")
async def model_data_write(data: dict):
# 将数据写到本地 然后可以开始手动标注
    sort_by = data.get('sort_by', None)
    if sort_by is None:
        return JSONResponse(status_code=400, content={"error": "No sort_by data received"})
    
    temp_dir = os.path.join("UserInterface/assets", "temp")
    points = np.array(global_source_point_cloud.points)
    pass



@app.post("/denoise")
async def denoise(data: dict):
    if not data:
        return JSONResponse(status_code=400, content={"error": "No data received"})
    # print(data)
    logger.info(f"denoise Received data: {data}")
    global global_source_point_cloud
    if global_source_point_cloud is None:
        # 尝试从文件中读取点云数据
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        if not os.path.exists(temp_ply_path):
            return JSONResponse(status_code=400, content={"error": "No point cloud data available"})
        point_cloud = o3d.io.read_point_cloud(temp_ply_path)
        global_source_point_cloud = point_cloud

    nb_neighbors_= data.get('nb_neighbors', 100)
    std_ratio_= data.get('std_ratio', 0.5)
    denoise_pcd, ind= global_source_point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors_, std_ratio=std_ratio_)

    # 输出denoise_pcd 点数
    logger.info(f"denoise success, points count: {len(denoise_pcd.points)}")
    # # show 3d
    if(data.get('settings', None)):
        settings = data.get('settings')
        if settings.get('show', False):
            o3d.visualization.draw_geometries([denoise_pcd])
    # o3d.visualization.draw_geometries([cropped_pcd])
    # cl, ind = cropped_pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)

    temp_dir = os.path.join("UserInterface/assets", "temp")
    temp_ply_path = os.path.join(temp_dir, 'temp.ply')
    o3d.io.write_point_cloud(temp_ply_path, denoise_pcd)
    global_source_point_cloud = denoise_pcd
    logger.info(f"Denoise success and point cloud saved to {temp_ply_path}")
    # 重置三视图和信息
    points = np.array(global_source_point_cloud.points)
    img_xy = normalize_and_map(points[:, 0], points[:, 1])
    img_yz = normalize_and_map(points[:, 1], points[:, 2])
    img_xz = normalize_and_map(points[:, 0], points[:, 2])
    cv2.imwrite(os.path.join(temp_dir, 'xy.jpg'), img_xy)
    cv2.imwrite(os.path.join(temp_dir, 'yz.jpg'), img_yz)
    cv2.imwrite(os.path.join(temp_dir, 'xz.jpg'), img_xz)
    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])
    z_min = np.min(points[:, 2])
    z_max = np.max(points[:, 2])

    with open(os.path.join(temp_dir, 'info.yml'), 'w') as f:
        f.write(f"x_min: {x_min}\n")
        f.write(f"x_max: {x_max}\n")
        f.write(f"y_min: {y_min}\n")
        f.write(f"y_max: {y_max}\n")
        f.write(f"z_min: {z_min}\n")
        f.write(f"z_max: {z_max}\n")
    return JSONResponse(status_code=200, content={"status": "success", "received": data})

# @app.post("/settings")
# async def settings(data: dict):
#     # change global settings
#     global settings
#     settings = data
#     return JSONResponse(status_code=200, content={"status": "success", "settings": settings})

# @app.post("/settings")
# async def model_data_write(data: dict):
# # 将数据写到本地 然后可以开始手动标注
#     sort_by = data.get('sort_by', None)
#     if sort_by is None:
#         return JSONResponse(status_code=400, content={"error": "No sort_by data received"})
    
#     temp_dir = os.path.join("UserInterface/assets", "temp")
#     points = np.array(global_source_point_cloud.points)
#     pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9304)
