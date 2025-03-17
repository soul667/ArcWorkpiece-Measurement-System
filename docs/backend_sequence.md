# 后端交互流程

## 点云上传流程

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant PointCloudManager
    participant FileSystem
    
    Client->>FastAPI: POST /upload
    FastAPI->>PointCloudManager: upload_point_cloud()
    PointCloudManager->>FileSystem: 保存PLY文件
    PointCloudManager->>FileSystem: 生成三视图
    PointCloudManager->>FileSystem: 保存YML信息
    PointCloudManager-->>FastAPI: 返回结果
    FastAPI-->>Client: 200 OK
```

## 点云裁剪流程

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant PointCloudProcessor
    participant PointCloudManager
    participant FileSystem
    
    Client->>FastAPI: POST /api/point-cloud/crop
    FastAPI->>PointCloudProcessor: crop_point_cloud()
    PointCloudProcessor->>PointCloudProcessor: 执行裁剪
    PointCloudProcessor-->>FastAPI: 返回裁剪结果
    FastAPI->>FileSystem: 保存裁剪后点云
    FastAPI->>PointCloudManager: 更新三视图
    FastAPI->>PointCloudManager: 更新信息文件
    FastAPI-->>Client: 200 OK
```

## 点云去噪流程

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant PointCloudProcessor
    participant PointCloudManager
    participant FileSystem
    
    Client->>FastAPI: POST /denoise
    FastAPI->>PointCloudProcessor: denoise_point_cloud()
    PointCloudProcessor->>PointCloudProcessor: 统计滤波去噪
    PointCloudProcessor-->>FastAPI: 返回去噪结果
    FastAPI->>FileSystem: 保存去噪后点云
    FastAPI->>PointCloudManager: 更新三视图
    FastAPI->>PointCloudManager: 更新信息文件
    FastAPI-->>Client: 200 OK
```

## 圆柱体拟合流程

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant CylinderProcessor
    
    Client->>FastAPI: POST /process
    FastAPI->>CylinderProcessor: process_cylinder_fitting()
    CylinderProcessor->>CylinderProcessor: 计算法向量
    opt RANSAC方法
        CylinderProcessor->>CylinderProcessor: RANSAC拟合
    end
    opt SVD方法
        CylinderProcessor->>CylinderProcessor: SVD拟合
    end
    CylinderProcessor-->>FastAPI: 返回拟合结果
    FastAPI-->>Client: 200 OK
```

## 错误处理流程

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant Processor
    participant Logger
    
    Client->>FastAPI: 任意请求
    FastAPI->>Processor: 处理请求
    
    alt 成功场景
        Processor-->>FastAPI: 返回结果
        FastAPI-->>Client: 200 OK
    else 验证错误
        Processor->>Logger: 记录错误
        Processor-->>FastAPI: 抛出ValueError
        FastAPI-->>Client: 400 Bad Request
    else 处理错误
        Processor->>Logger: 记录错误
        Processor-->>FastAPI: 抛出RuntimeError
        FastAPI-->>Client: 500 Server Error
    end
