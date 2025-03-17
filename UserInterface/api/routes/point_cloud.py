import os
import json
import numpy as np
import open3d as o3d
import onnxruntime as ort
from datetime import datetime
from fastapi import APIRouter, File, Form, HTTPException, Depends, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, Optional

from UserInterface.auth.service import get_current_user
from UserInterface.auth.db import Database
from UserInterface.point_cloud_manager import PointCloudManager
from UserInterface.point_cloud_processor import PointCloudProcessor
from UserInterface.point_cloud_denoiser import PointCloudDenoiser
from UserInterface.point_cloud_grouper import PointCloudGrouper
from UserInterface.cylinder_processor import CylinderProcessor
from UserInterface.arc_fitting_processor import ArcFittingProcessor
from UserInterface.api.utils.point_cloud_generator import generate_cylinder_points
from UserInterface.api.utils.sequence_processor import normalize_sequence, normalize_input
from UserInterface.api.models.point_cloud import (
    PointCloudProcessRequest,
    DenoiseRequest,
    GroupPointsRequest,
    DefectLinesRequest,
    ModelPredictionRequest,
    ProcessingResponse,
    DefectLinesResponse,
    PredictionResponse,
    UploadResponse
)
from UserInterface.api.config import (
    logger,
    TEMP_DIR,
    DEFAULT_NORMAL_NEIGHBORS,
    DEFAULT_MIN_RADIUS,
    DEFAULT_MAX_RADIUS,
    DEFAULT_RANSAC_THRESHOLD,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_NORMAL_DISTANCE_WEIGHT
)

router = APIRouter()

# 创建处理器实例
cloud_manager = PointCloudManager()
cloud_processor = PointCloudProcessor()
cylinder_processor = CylinderProcessor()
cloud_denoiser = PointCloudDenoiser()
point_cloud_grouper = PointCloudGrouper()
arc_fitting_processor = ArcFittingProcessor()

# 全局变量
global_source_point_cloud = None
global_axis_direction = None

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

def get_point_cloud() -> tuple[o3d.geometry.PointCloud, bool]:
    """获取当前点云数据"""
    global global_source_point_cloud
    if global_source_point_cloud is not None:
        return global_source_point_cloud, True

    temp_ply_path = os.path.join(TEMP_DIR, 'temp.ply')
    if not os.path.exists(temp_ply_path):
        return None, False

    try:
        global_source_point_cloud = o3d.io.read_point_cloud(temp_ply_path)
        return global_source_point_cloud, True
    except Exception as e:
        logger.error(f"读取点云文件失败: {str(e)}")
        return None, False

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    actual_speed: float = Form(100.0),
    acquisition_speed: float = Form(100.0)
) -> UploadResponse:
    """上传点云文件"""
    if not file:
        raise HTTPException(status_code=400, detail="未提供文件")

    try:
        file_content = await file.read()
        global global_source_point_cloud
        global_source_point_cloud, file_size_mb = cloud_manager.upload_point_cloud(
            file_content,
            actual_speed=actual_speed,
            acquisition_speed=acquisition_speed
        )
        
        return UploadResponse(
            message=f"文件上传成功，大小: {file_size_mb:.2f} MB",
            file_size_mb=file_size_mb
        )
    except Exception as e:
        logger.error(f"文件上传失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate")
async def generate_point_cloud(data: Dict):
    """生成点云数据并提供下载"""
    try:
        # 提取和验证参数
        noise_std = float(data.get("noise_std", 0.01))
        arc_angle = float(data.get("arc_angle", 360.0))
        axis_direction = data.get("axis_direction", [0, 0, 1])
        axis_density = int(data.get("axis_density", 500))
        arc_density = int(data.get("arc_density", 100))
        
        if not isinstance(axis_direction, list) or len(axis_direction) != 3:
            raise ValueError("axis_direction must be a list of 3 numbers")
        
        # 生成点云
        points = generate_cylinder_points(
            point_count=axis_density * arc_density,
            radius=10.0,
            height=50.0,
            noise_std=noise_std,
            arc_angle=arc_angle,
            axis_direction=axis_direction
        )
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 保存点云文件
        temp_ply_path = os.path.join(TEMP_DIR, 'generated_cloud.ply')
        o3d.io.write_point_cloud(temp_ply_path, pcd)
        
        # 更新全局点云
        global global_source_point_cloud
        global_source_point_cloud = pcd
        
        return FileResponse(
            path=temp_ply_path,
            filename="generated_cloud.ply",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"生成点云失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process")
async def process_cylinder(data: Dict):
    """处理点云数据拟合圆柱体"""
    if not data:
        raise HTTPException(status_code=400, detail="未接收到数据")
    
    # 获取点云数据
    point_cloud, success = get_point_cloud()
    if not success:
        raise HTTPException(status_code=400, detail="无可用的点云数据")

    try:
        points = np.asarray(point_cloud.points)
        
        result = cylinder_processor.process_cylinder_fitting(
            points=points,
            cylinder_method=data.get('cylinder_method', 'NormalRANSAC'),
            normal_neighbors=data.get('normal_neighbors', DEFAULT_NORMAL_NEIGHBORS),
            ransac_threshold=data.get('ransac_threshold', DEFAULT_RANSAC_THRESHOLD),
            min_radius=data.get('min_radius', DEFAULT_MIN_RADIUS),
            max_radius=data.get('max_radius', DEFAULT_MAX_RADIUS),
            max_iterations=data.get('max_iterations', DEFAULT_MAX_ITERATIONS),
            normal_distance_weight=data.get('normal_distance_weight', DEFAULT_NORMAL_DISTANCE_WEIGHT)
        )
        
        # 保存轴线方向向量
        global global_axis_direction
        global_axis_direction = result.get('axis', {}).get('direction')
        
        return JSONResponse(status_code=200, content=result)
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"处理点云数据时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"点云处理失败: {str(e)}")

@router.post("/arc-fitting-stats")
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


@router.post("/group-points")
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

@router.post("/remove-defect-lines")
async def remove_defect_lines(data: DefectLinesRequest):
    """从点云中删除标记为缺陷的线条"""
    try:
        if not data.defect_indices:
            return DefectLinesResponse(
                status="success",
                message="没有需要删除的线条"
            )
            
        point_cloud, success = get_point_cloud()
        if not success:
            raise HTTPException(status_code=400, detail="无可用的点云数据")
            
        points = np.asarray(point_cloud.points)
        
        try:
            remaining_points = point_cloud_grouper.remove_groups(points, data.defect_indices)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        new_cloud = o3d.geometry.PointCloud()
        new_cloud.points = o3d.utility.Vector3dVector(remaining_points)
        
        temp_ply_path = os.path.join(TEMP_DIR, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, new_cloud)
        
        global global_source_point_cloud
        global_source_point_cloud = new_cloud
        
        points = np.array(new_cloud.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)
        
        return DefectLinesResponse(
            status="success",
            message="缺陷线条已删除，预处理文件已更新",
            removed_count=len(data.defect_indices)
        )
        
    except Exception as e:
        logger.error(f"删除缺陷线条失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/predict")
async def predict_quality(data: ModelPredictionRequest):
    """使用ONNX模型预测线条质量"""
    try:
        points = np.array(data.points)
        points = normalize_sequence(points)
        points = normalize_input(points)
        
        # 准备模型输入
        input_data = points.reshape(1, 500, 1).astype(np.float32)
        
        # 加载模型
        model_path = os.path.join(TEMP_DIR, "arc_quality_model.onnx")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail="模型文件不存在")
            
        session = ort.InferenceSession(model_path)
        
        # 获取预测
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        prediction = session.run([output_name], {input_name: input_data})[0]
        
        probability = float(prediction[0][0])
        label = 1 if probability > 0.5 else 0
        
        return PredictionResponse(
            status="success",
            label=label,
            probability=probability
        )
        
    except Exception as e:
        logger.error(f"预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/denoise")
async def denoise_point_cloud(data: DenoiseRequest):
    """对当前点云数据进行去噪处理"""
    try:
        point_cloud, success = get_point_cloud()
        if not success:
            raise HTTPException(status_code=400, detail="无可用的点云数据")

        # 应用去噪处理
        denoised_cloud = cloud_denoiser.denoise_point_cloud(
            point_cloud,
            nb_neighbors=data.nb_neighbors,
            std_ratio=data.std_ratio
        )

        # 保存去噪后的点云
        temp_ply_path = os.path.join(TEMP_DIR, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, denoised_cloud)

        # 更新全局点云
        global global_source_point_cloud
        global_source_point_cloud = denoised_cloud

        # 更新点云信息
        points = np.array(denoised_cloud.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)

        return ProcessingResponse(
            status="success",
            message="点云去噪完成",
            received=data.dict()
        )

    except Exception as e:
        logger.error(f"点云去噪失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/crop")
async def crop_point_cloud(data: dict):
    # logger.error("111111111111111111111111111111111111")

    """裁剪点云数据"""
    if not data:
        raise HTTPException(status_code=400, detail="未接收到数据")
    # out data
    print(data)

    point_cloud, success = get_point_cloud()
    if not success:
        raise HTTPException(status_code=400, detail="无可用的点云数据")

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

        temp_ply_path = os.path.join(TEMP_DIR, 'temp.ply')
        o3d.io.write_point_cloud(temp_ply_path, cropped_pcd)

        global global_source_point_cloud
        global_source_point_cloud = cropped_pcd

        points = np.array(cropped_pcd.points)
        cloud_manager.generate_views(points)
        cloud_manager.update_cloud_info(points)

        return ProcessingResponse(
            status="success",
            message="点云裁剪完成",
            received=data
        )

    except Exception as e:
        logger.error(f"点云裁剪失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
