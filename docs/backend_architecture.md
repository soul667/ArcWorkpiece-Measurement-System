# 后端架构文档

## 项目结构

```
UserInterface/
├── fastapi_test.py         # FastAPI主应用程序
├── point_cloud_manager.py  # 点云管理类
├── point_cloud_processor.py # 点云处理类
├── cylinder_processor.py   # 圆柱体拟合处理类
└── assets/                 # 静态资源目录
    └── temp/              # 临时文件存储
```

## 核心组件

### 1. 点云管理器 (PointCloudManager)

负责处理点云文件的上传、可视化和信息管理。

主要功能:
- 文件上传和保存
- 生成三视图
- 更新点云信息

关键方法:
```python
def upload_point_cloud(self, file_content: bytes) -> Tuple[o3d.geometry.PointCloud, float]:
    """上传并处理点云文件"""

def generate_views(self, points: np.ndarray) -> None:
    """生成点云三视图"""

def update_cloud_info(self, points: np.ndarray) -> None:
    """更新点云信息到yml文件"""
```

### 2. 点云处理器 (PointCloudProcessor)

处理点云的裁剪和去噪操作。

主要功能:
- 点云裁剪
- 点云去噪

关键方法:
```python
def crop_point_cloud(
    self,
    point_cloud: o3d.geometry.PointCloud,
    regions: dict,
    modes: dict
) -> o3d.geometry.PointCloud:
    """裁剪点云"""

def denoise_point_cloud(
    self,
    point_cloud: o3d.geometry.PointCloud,
    nb_neighbors: int = 100,
    std_ratio: float = 0.5
) -> o3d.geometry.PointCloud:
    """点云去噪"""
```

### 3. 圆柱体处理器 (CylinderProcessor)

处理圆柱体拟合相关操作。

主要功能:
- 参数验证
- RANSAC圆柱体拟合
- SVD圆柱体拟合

关键方法:
```python
def process_cylinder_fitting(
    self,
    points: np.ndarray,
    cylinder_method: str = 'RANSAC',
    normal_neighbors: int = 30,
    ransac_threshold: float = 0.1,
    min_radius: float = 6,
    max_radius: float = 11
) -> Dict[str, Union[str, Dict[str, List[float]]]]:
    """执行圆柱体拟合处理"""
```

## API 端点

### 文件操作

#### 1. 点云上传
```
POST /upload
Content-Type: multipart/form-data

参数:
- file: 点云文件 (.ply 格式)

返回:
{
    "message": "文件上传成功，大小: xxx MB"
}
```

#### 2. 获取点云文件
```
GET /get_ply/{ply_name}

返回:
- PLY 文件
```

#### 3. 获取图片
```
GET /img/{img_name}

返回:
- JPG 图片
```

#### 4. 获取信息文件
```
GET /yml/{yml_name}

返回:
- YML 文件
```

### 点云处理

#### 1. 裁剪点云
```
POST /crop

请求体:
{
    "regions": {
        "x_regions": [...],
        "y_regions": [...],
        "z_regions": [...]
    },
    "modes": {
        "x_mode": "keep",
        "y_mode": "keep",
        "z_mode": "keep"
    },
    "settings": {
        "show": true
    }
}

返回:
{
    "status": "success",
    "received": {...}
}
```

#### 2. 点云去噪
```
POST /denoise

请求体:
{
    "nb_neighbors": 100,
    "std_ratio": 0.5,
    "settings": {
        "show": true
    }
}

返回:
{
    "status": "success",
    "received": {...}
}
```

#### 3. 圆柱体拟合
```
POST /process

请求体:
{
    "cylinderMethod": "RANSAC",
    "normalNeighbors": 30,
    "ransacThreshold": 0.1,
    "minRadius": 6,
    "maxRadius": 11
}

返回:
{
    "status": "success",
    "axis": {
        "point": [x, y, z],
        "direction": [dx, dy, dz]
    },
    "radius": r  // 仅RANSAC方法返回
}
```

## 错误处理

所有端点都包含统一的错误处理机制:

- 400 Bad Request: 请求参数无效
- 404 Not Found: 请求的资源不存在
- 500 Internal Server Error: 服务器处理错误

错误响应格式:
```json
{
    "error": "错误描述信息"
}
```

## 全局配置

- 日志配置:
  - 日志文件: ./UserInterface/fastapi.log
  - 日志级别: INFO
  - 格式: 时间 - 名称 - 级别 - 消息

- CORS配置:
  - 允许所有来源
  - 允许所有方法
  - 允许所有头部

## 缓存策略

- 静态资源缓存:
  - Cache-Control: public, max-age=31536000 (1年)
  - ETag: 基于文件修改时间
