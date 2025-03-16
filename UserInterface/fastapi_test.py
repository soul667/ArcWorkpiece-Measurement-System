# Python 标准库
import os
import json
import logging
from typing import Optional, Tuple, List
from io import BytesIO

# 第三方库
import cv2
import numpy as np
import open3d as o3d
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
import onnxruntime as ort
from scipy.interpolate import interp1d
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 本地模块
from UserInterface.PointCouldProgress import (
    remove_statistical_outliers,
    segmentPointCloud
)
from UserInterface.arc_fitting_processor import ArcFittingProcessor
from UserInterface.point_cloud_grouper import PointCloudGrouper
from UserInterface.auth.routes import router as auth_router
from UserInterface.auth.service import AuthService, get_current_user
from UserInterface.auth.db import Database
from UserInterface.cylinder_processor import CylinderProcessor
from UserInterface.point_cloud_denoiser import PointCloudDenoiser
from UserInterface.point_cloud_manager import PointCloudManager
from UserInterface.point_cloud_processor import PointCloudProcessor
from algorithm.axis_projection import AxisProjector

def generate_cylinder_points(point_count: int = 1000, radius: float = 0.5, height: float = 2.0, 
                         noise_std: float = 0.01, arc_angle: float = 360.0,
                         axis_direction: List[float] = [0, 0, 1]) -> np.ndarray:
    """生成圆柱体点云数据
    
    Args:
        point_count: 点云数量
        radius: 圆柱体半径
        height: 圆柱体高度 
        noise_std: 噪声标准差
        arc_angle: 圆心角(度)
        axis_direction: 圆柱体轴向方向
        
    Returns:
        points: 生成的圆柱体点云，numpy数组(N,3) 
    """
    # 归一化轴向向量
    axis = np.array(axis_direction)
    axis = axis / np.linalg.norm(axis)
    
    # 创建旋转矩阵，将[0,0,1]对齐到目标轴向
    if np.allclose(axis, [0, 0, 1]):
        R = np.eye(3)
    else:
        # 计算旋转轴
        rot_axis = np.cross([0, 0, 1], axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        
        # 计算旋转角度
        cos_angle = np.dot([0, 0, 1], axis)
        angle = np.arccos(cos_angle)
        
        # Rodriguez旋转公式
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                     [rot_axis[2], 0, -rot_axis[0]],
                     [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * K.dot(K)
    
    # 计算弧度
    arc_rad = np.deg2rad(arc_angle)
    
    # 生成点云
    thetas = np.random.uniform(0, arc_rad, point_count)
    heights = np.random.uniform(-height/2, height/2, point_count)
    
    # 生成圆柱面上的点
    x = radius * np.cos(thetas)
    y = radius * np.sin(thetas)
    z = heights
    
    # 合并为点云数组
    points = np.column_stack([x, y, z])
    
    # 旋转点云以对齐目标轴向
    points = points.dot(R.T)
    
    # 添加随机噪声
    noise = np.random.normal(0, noise_std, (point_count, 3))
    points += noise
    
    return points

# 配置日志
log_file_path = "./UserInterface/fastapi.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# 服务器配置
reload_dirs = ["."]
reload_excludes = ["node_modules", ".git", "__pycache__", ".pytest_cache"]

# 创建FastAPI应用
app = FastAPI(title="圆弧工件测量系统")

@app.post("/api/remove-defect-lines")
async def remove_defect_lines(data: dict):
    """
    从点云中删除标记为缺陷的线条并重新生成预处理文件
    """
    try:
        defect_indices = data.get('defect_indices', [])
        if not defect_indices:
            return {"status": "success", "message": "没有需要删除的线条"}
            
        # 获取当前点云数据
        point_cloud, success = get_point_cloud()
        if not success:
            raise HTTPException(status_code=400, detail="无可用的点云数据")
            
        points = np.asarray(point_cloud.points)
        
        # 使用 PointCloudGrouper 删除缺陷线条
        grouper = PointCloudGrouper()
        try:
            remaining_points = grouper.remove_groups(points, defect_indices)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # 创建新的点云对象
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(remaining_points)
        
        # 保存新的点云文件
        temp_dir = os.path.join("UserInterface/assets", "temp")
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, new_cloud)
        
        # 更新全局点云
        global global_source_point_cloud
        global_source_point_cloud = new_cloud
        
        # 使用点云管理器重新生成投影图和yml
        points = np.array(new_cloud.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)
        
        return {
            "status": "success",
            "message": "缺陷线条已删除，预处理文件已更新",
            "removed_count": len(defect_indices)
        }
        
    except Exception as e:
        logger.error(f"删除缺陷线条失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def normalize_sequence(points, target_length=500):
    """将点序列插值到指定长度"""
    current_length = len(points)
    if current_length == 0:
        raise ValueError("Empty point sequence")
        
    if current_length == target_length:
        return points
        
    # 创建原始序列的索引
    x_original = np.linspace(0, 1, current_length)
    x_target = np.linspace(0, 1, target_length)
    
    # 使用线性插值
    interpolator = interp1d(x_original, points, kind='linear')
    interpolated = interpolator(x_target)
    
    return interpolated

def normalize_input(x):
    """标准化输入数据"""
    return (x - np.mean(x)) / np.std(x)

@app.post("/api/model/predict")
async def predict_quality(data: dict):
    """
    使用ONNX模型预测线条质量
    
    参数:
    - points: 要预测的点序列（z坐标值）
    
    返回:
    - label: 0表示正常，1表示有缺陷
    - probability: 预测为缺陷的概率
    """
    try:
        points = np.array(data['points'])
        
        # 插值到500点
        points = normalize_sequence(points)
        
        # 标准化
        points = normalize_input(points)
        
        # 准备模型输入
        input_data = points.reshape(1, 500, 1).astype(np.float32)
        
        # 加载模型
        model_path = os.path.join("UserInterface/assets/temp", "arc_quality_model.onnx")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="模型文件不存在")
            
        session = ort.InferenceSession(model_path)
        
        # 获取预测
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run([output_name], {input_name: input_data})[0]
        
        probability = float(prediction[0][0])
        label = 1 if probability > 0.5 else 0
        
        return {
            "status": "success",
            "label": label,
            "probability": probability
        }
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/upload")
async def upload_model(model: UploadFile = File(...)):
    """
    上传新的ONNX模型文件
    """
    try:
        if not model.filename.endswith('.onnx'):
            raise HTTPException(status_code=400, detail="只支持.onnx格式的模型文件")
            
        content = await model.read()
        model_path = os.path.join("UserInterface/assets/temp", "arc_quality_model.onnx")
        
        # 验证模型文件
        try:
            # 保存到临时文件进行验证
            temp_path = model_path + '.temp'
            with open(temp_path, "wb") as f:
                f.write(content)
            
            # 尝试加载模型
            session = ort.InferenceSession(temp_path)
            inputs = session.get_inputs()
            
            # 验证输入形状
            if len(inputs) != 1 or inputs[0].shape != [1, 500, 1]:
                raise ValueError("模型输入形状不正确，需要 [1, 500, 1]")
            
            # 验证通过，替换原有模型
            os.replace(temp_path, model_path)
            
        except Exception as e:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise ValueError(f"模型验证失败: {str(e)}")
            
        return {
            "status": "success", 
            "message": "模型上传成功"
        }
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"模型上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 添加认证路由
app.include_router(auth_router, prefix="/auth", tags=["authentication"])

# 配置OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
auth_service = AuthService()

@app.post("/api/generate-point-cloud")
async def generate_point_cloud(data: dict):
    """生成点云数据并提供下载
    
    请求体:
    {
        "noise_std": float,      // 噪声大小
        "arc_angle": float,      // 圆心角(度)
        "axis_direction": [x,y,z],// 轴线方向
        "axis_density": int,     // 沿轴线密度
        "arc_density": int       // 圆弧密度
    }
    """
    try:
        # 验证和提取参数
        noise_std = float(data.get("noise_std", 0.01))
        arc_angle = float(data.get("arc_angle", 360.0))
        axis_direction = data.get("axis_direction", [0, 0, 1])
        axis_density = int(data.get("axis_density", 500))
        arc_density = int(data.get("arc_density", 100))
        
        # 参数验证
        if not isinstance(axis_direction, list) or len(axis_direction) != 3:
            raise ValueError("axis_direction must be a list of 3 numbers")
        
        # 生成点云
        points = generate_cylinder_points(
            point_count=axis_density * arc_density,
            radius=10.0,  # 固定半径
            height=50.0,  # 固定高度
            noise_std=noise_std,
            arc_angle=arc_angle,
            axis_direction=axis_direction
        )
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 保存点云文件
        temp_dir = os.path.join("UserInterface/assets", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_ply_path = os.path.join(temp_dir, 'generated_cloud.ply')
        o3d.io.write_point_cloud(temp_ply_path, pcd)
        
        # 更新全局点云
        global global_source_point_cloud
        global_source_point_cloud = pcd
        
        # 返回文件下载响应
        return FileResponse(
            path=temp_ply_path,
            filename="generated_cloud.ply",
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=generated_cloud.ply"
            }
        )
        
    except Exception as e:
        logger.error(f"生成点云失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 配置CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置静态文件和模板
app.mount("/assets", StaticFiles(directory="UserInterface/assets"), name="assets")
templates = Jinja2Templates(directory="templates")

# 创建处理器实例
cloud_manager = PointCloudManager()
cloud_processor = PointCloudProcessor()
cylinder_processor = CylinderProcessor()
cloud_denoiser = PointCloudDenoiser()
point_cloud_grouper = PointCloudGrouper()
arc_fitting_processor = ArcFittingProcessor()

# 全局变量
global_source_point_cloud = None
global_axis_direction = None  # 存储轴线方向向量
settings = {
    'show': True  
}

def get_point_cloud() -> Tuple[o3d.geometry.PointCloud, bool]:
    """获取当前点云数据
    
    如果全局点云数据为空，尝试从文件加载。
    
    Returns:
        Tuple[o3d.geometry.PointCloud, bool]: 点云对象和是否成功的标志
    """
    global global_source_point_cloud
    if global_source_point_cloud is not None:
        return global_source_point_cloud, True

    temp_dir = os.path.join("UserInterface/assets", "temp")
    temp_ply_path = os.path.join(temp_dir, 'temp.ply')
    if not os.path.exists(temp_ply_path):
        return None, False

    try:
        global_source_point_cloud = o3d.io.read_point_cloud(temp_ply_path)
        return global_source_point_cloud, True
    except Exception as e:
        logger.error(f"读取点云文件失败: {str(e)}")
        return None, False

# 需要认证的路由
@app.get("/api/settings/latest")
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

@app.get("/")
async def home(request: Request, current_user: dict = Depends(get_current_user)):
    return templates.TemplateResponse("index1.html", {"request": request})

@app.get("/img/{img_name}")
async def get_image(img_name: str, v: Optional[str] = None):
    img_path = os.path.join("UserInterface/assets", "temp", f"{img_name}.jpg")
    logger.info(f"请求图片: {img_name}, 版本: {v}")
    if os.path.exists(img_path):
        file_version = str(os.path.getmtime(img_path))
        response = FileResponse(img_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="未找到图片")

@app.get("/yml/{yml_name}")
async def get_yml(yml_name: str, v: Optional[str] = None):
    yml_path = os.path.join("UserInterface/assets", "temp", f"{yml_name}.yml")
    logger.info(f"请求YAML文件: {yml_name}")
    if os.path.exists(yml_path):
        file_version = str(os.path.getmtime(yml_path))
        response = FileResponse(yml_path)
        response.headers["Cache-Control"] = "public, max-age=31536000"
        response.headers["ETag"] = f'W/"{file_version}"'
        return response
    else:
        raise HTTPException(status_code=404, detail="未找到YAML文件")

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    actual_speed: float = Form(100.0),
    acquisition_speed: float = Form(100.0)
):
    if not file:
        return JSONResponse(status_code=400, content={"error": "未提供文件"})

    try:
        file_content = await file.read()
        global global_source_point_cloud
        global_source_point_cloud, file_size_mb = cloud_manager.upload_point_cloud(
            file_content,
            actual_speed=actual_speed,
            acquisition_speed=acquisition_speed
        )
        
        return JSONResponse(
            status_code=200, 
            content={"message": f"文件上传成功，大小: {file_size_mb:.2f} MB"}
        )
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})

# 参数设置相关端点
@app.post("/api/line-settings/save")
async def save_line_settings(data: dict, current_user: dict = Depends(get_current_user)):
    """保存线条显示设置和缺陷标记"""
    try:
        query = """
            INSERT INTO line_settings 
            (user_id, point_size, defect_lines, created_at) 
            VALUES (%s, %s, %s, NOW())
        """
        
        db = Database()
        params = (
            current_user['id'],
            data.get('point_size', 3),
            json.dumps(data.get('defect_lines', []))
        )
        db.execute_query(query, params)
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "设置保存成功"}
        )
    except Exception as e:
        logger.error(f"保存线条设置失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"保存失败: {str(e)}"}
        )

@app.get("/api/line-settings/latest")
async def get_latest_line_settings(current_user: dict = Depends(get_current_user)):
    """获取最新的线条设置"""
    try:
        query = """
            SELECT point_size, defect_lines
            FROM line_settings
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """
        
        db = Database()
        results = db.execute_query(query, (current_user['id'],))
        print("result=",results)
        if not results:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success", 
                    "data": {"point_size": 3, "defect_lines": []}
                }
            )
            
        item = results[0]
        settings = {
            'point_size': item['point_size'],
            'defect_lines': json.loads(item['defect_lines'])
        }
        
        return JSONResponse(
            status_code=200,
            content={"status": "success", "data": settings}
        )
    except Exception as e:
        logger.error(f"获取线条设置失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"获取设置失败: {str(e)}"}
        )

@app.post("/api/settings/save")
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

@app.put("/api/settings/{setting_id}")
async def update_setting(
    setting_id: int, 
    data: dict,
    current_user: dict = Depends(get_current_user)
):
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
        return JSONResponse(
            status_code=500,
            content={"error": f"更新失败: {str(e)}"}
        )
# 从出现422错误
@app.delete("/api/settings/deleteAll")
async def delete_all_settings(
    current_user: dict = Depends(get_current_user)):
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
    
@app.delete("/api/settings/{setting_id}")
async def delete_setting(
    setting_id: int,
    current_user: dict = Depends(get_current_user)
):
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
        return JSONResponse(
            status_code=500,
            content={"error": f"删除失败: {str(e)}"}
        )

@app.get("/api/settings/list")
async def list_settings(current_user: dict = Depends(get_current_user)):
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
        
        # 格式化返回数据
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
        return JSONResponse(
            status_code=500,
            content={"error": f"获取设置列表失败: {str(e)}"}
        )

@app.get("/api/settings/{setting_id}")
async def get_setting(setting_id: int, current_user: dict = Depends(get_current_user)):
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
            return JSONResponse(
                status_code=404,
                content={"error": "未找到指定的设置"}
            )
            
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
        return JSONResponse(
            status_code=500,
            content={"error": f"获取设置失败: {str(e)}"}
        )

@app.post("/crop")
async def submit(data: dict):
    if not data:
        return JSONResponse(status_code=400, content={"error": "未接收到数据"})
    
    logger.info(f"接收到裁剪请求数据: {data}")
    
    point_cloud, success = get_point_cloud()
    if not success:
        return JSONResponse(status_code=400, content={"error": "无可用的点云数据"})

    try:
        data_region = data.get('regions', None)
        data_mode = data.get('modes', None)
        cropped_pcd = cloud_processor.crop_point_cloud(
            point_cloud,
            data_region,
            data_mode
        )

        if data.get('settings', {}).get('show', False):
            o3d.visualization.draw_geometries([cropped_pcd])

        temp_dir = os.path.join("UserInterface/assets", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, cropped_pcd)
        global global_source_point_cloud
        global_source_point_cloud = cropped_pcd

        points = np.array(cropped_pcd.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)

        return JSONResponse(status_code=200, content={"status": "success", "received": data})

    except Exception as e:
        logger.error(f"点云裁剪失败: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/denoise")
async def denoise_point_cloud(data: dict):
    """
    对当前点云数据进行去噪处理
    
    请求体:
    {
        "nb_neighbors": 100,        # 计算统计值时考虑的邻居点数
        "std_ratio": 0.5,          # 标准差比例
        "settings": {
            "show": true           # 是否显示结果
        }
    }
    """
    try:
        point_cloud, success = get_point_cloud()
        if not success:
            return JSONResponse(status_code=400, content={"error": "无可用的点云数据"})

        nb_neighbors = data.get("nb_neighbors", 100)
        std_ratio = data.get("std_ratio", 0.5)
        settings = data.get("settings", {"show": True})

        # 应用去噪处理
        denoised_cloud = cloud_denoiser.denoise_point_cloud(
            point_cloud,
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

        # 如果需要则显示点云
        if settings.get("show", False):
            o3d.visualization.draw_geometries([denoised_cloud])

        # 保存去噪后的点云
        temp_dir = os.path.join("UserInterface/assets", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_ply_path = os.path.join(temp_dir, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, denoised_cloud)

        # 更新全局点云
        global global_source_point_cloud
        global_source_point_cloud = denoised_cloud

        # 更新点云信息
        points = np.array(denoised_cloud.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "received": data
            }
        )

    except Exception as e:
        logger.error(f"点云去噪失败: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/api/group-points")
async def group_points(data: dict):
    """
    获取指定索引的线条数据
    请求体:
    {
        "axis": "x" | "y" | "z",  # 分组轴
        "index": 0                 # 线条索引
    }
    """
    try:
        # 获取点云数据
        point_cloud, success = get_point_cloud()
        if not success:
            return JSONResponse(status_code=400, content={"error": "无可用的点云数据"})
            
        # 获取参数
        axis = data.get('axis', 'x')
        print("axis=",axis)
        index = data.get('index', 0)
        
        # 参数验证
        if axis not in ['x', 'y', 'z']:
            return JSONResponse(status_code=400, content={"error": "无效的轴参数，必须为 x、y 或 z"})
        if not isinstance(index, int) or index < 0:
            return JSONResponse(status_code=400, content={"error": "无效的线条索引"})
            
        # 获取指定线条数据
        points = np.asarray(point_cloud.points)
        result = point_cloud_grouper.group_by_axis(
            points,
            axis=axis,
            index=index
        )
            
        return JSONResponse(status_code=200, content=result)
            
    except Exception as e:
        logger.error(f"点云分组失败: {str(e)}")
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
        # 获取基本参数
        cylinder_method = data.get('cylinder_method', 'NormalRANSAC')
        normal_neighbors = data.get('normal_neighbors', 30)
        min_radius = data.get('min_radius', 6)
        max_radius = data.get('max_radius', 11)

        # 获取RANSAC参数，支持两种结构
        # ransac_params = data.get('ransac_params', {})
        # ransac_threshold = data.get('ransac_threshold') or ransac_params.get('distance_threshold', 0.01)
        # max_iterations = data.get('max_iterations') or ransac_params.get('max_iterations', 1000)
        # normal_distance_weight = data.get('normal_distance_weight') or ransac_params.get('normal_distance_weight', 0.1)
        ransac_threshold = data.get('ransac_threshold',0.01)
        max_iterations = data.get('max_iterations',1000) 
        normal_distance_weight = data.get('normal_distance_weight',0.8) 

        result = cylinder_processor.process_cylinder_fitting(
            points=points,
            cylinder_method=cylinder_method,
            normal_neighbors=normal_neighbors,
            ransac_threshold=ransac_threshold,
            min_radius=min_radius,
            max_radius=max_radius,
            max_iterations=max_iterations,
            normal_distance_weight=normal_distance_weight
        )
        
        # 保存轴线方向向量
        global global_axis_direction
        global_axis_direction = result.get('axis', {}).get('direction')
        
        return JSONResponse(status_code=200, content=result)
        
    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        logger.error(f"处理点云数据时出错: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"点云处理失败: {str(e)}"})
def validate_json_data(data):
    """
    递归验证并转换数据，确保所有数据都是JSON可序列化的
    """
    if isinstance(data, dict):
        return {k: validate_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [validate_json_data(item) for item in data]
    elif isinstance(data, bool):
        return int(data)
    elif isinstance(data, (int, float, str)) or data is None:
        return data
    else:
        return str(data)

# 新增了传入的时候需要输入轴
@app.post("/api/arc-fitting-stats")
async def get_arc_fitting_stats(data: dict):
    """获取圆弧拟合统计信息"""
    try:
        if not global_axis_direction:
            raise HTTPException(status_code=400, detail="请先完成轴线拟合")
        
        point_cloud, success = get_point_cloud()
        if not success:
            raise HTTPException(status_code=400, detail="无可用的点云数据")
        # print("data",data)
        settings = {
            'arcNormalNeighbors': data.get('arcNormalNeighbors', 10),
            'fitIterations': data.get('fitIterations', 50),
            'samplePercentage': data.get('samplePercentage', 50),
            'axis_direction': global_axis_direction,
            'point_on_axis': np.array([0, 0, 0])  # 可以使用任意轴上点
        }
        axis_now = data.get('axis_now', 'x')  # Get axis_now from request data
        print('data', data)
        print("axis_now", axis_now)
        points = np.asarray(point_cloud.points)
        results = arc_fitting_processor.process_all_lines(points, settings, axis_now)
        
        # 验证数据确保可以JSON序列化
        validated_results = validate_json_data(results)
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "lineStats": validated_results['lineStats'],
                "overallStats": validated_results['overallStats']
            }
        )
        
    except Exception as e:
        logger.error(f"圆弧拟合失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"圆弧拟合失败: {str(e)}"}
        )



if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("UserInterface/assets/temp", exist_ok=True)

    # 启动服务器
    uvicorn.run(
        "fastapi_test:app",
        host="0.0.0.0",
        port=9304,
        reload=True,
        reload_dirs=reload_dirs,
        reload_excludes=reload_excludes
    )
