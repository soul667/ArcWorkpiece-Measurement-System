# 后端环境配置

## 依赖项

主要依赖库:

```plaintext
fastapi          # Web框架
uvicorn          # ASGI服务器
open3d           # 点云处理
numpy            # 数值计算
opencv-python    # 图像处理
```

完整依赖列表：
```bash
fastapi>=0.68.0
uvicorn>=0.15.0
open3d>=0.13.0
numpy>=1.21.0
opencv-python>=4.5.3
python-multipart  # 用于处理文件上传
python-jose[cryptography]  # 用于JWT
```

## 项目设置

1. 创建并激活虚拟环境:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 创建必要的目录:
```bash
mkdir -p UserInterface/assets/temp
```

## 运行服务器

1. 开发模式运行:
```bash
uvicorn UserInterface.fastapi_test:app --reload --port 9304
```

2. 生产模式运行:
```bash
uvicorn UserInterface.fastapi_test:app --host 0.0.0.0 --port 9304
```

## 配置说明

### 1. 环境变量

无需特殊环境变量配置。

### 2. 文件路径

主要路径配置:
```python
TEMP_DIR = "UserInterface/assets/temp"  # 临时文件目录
LOG_FILE = "./UserInterface/fastapi.log" # 日志文件
```

### 3. 日志配置

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./UserInterface/fastapi.log"),
        logging.StreamHandler()
    ]
)
```

## 开发指南

### 1. 代码风格

- 使用类型提示
- 添加中文文档注释
- 使用异常处理包装所有IO操作

示例:
```python
def process_data(data: np.ndarray) -> Dict[str, Any]:
    """处理数据
    
    Args:
        data: 输入数据数组
        
    Returns:
        处理结果字典
        
    Raises:
        ValueError: 当输入数据无效时
    """
    try:
        # 处理逻辑
        pass
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        raise RuntimeError(f"处理失败: {str(e)}")
```

### 2. 错误处理

- 使用适当的HTTP状态码
- 提供清晰的错误消息
- 记录详细的错误日志

### 3. 性能优化

- 使用异步IO操作
- 避免重复计算
- 合理使用缓存策略

## 测试

1. 运行单元测试:
```bash
pytest tests/
```

2. API测试:
可以使用FastAPI的交互式文档页面:
```
http://localhost:9304/docs
```

## 常见问题

### 1. 文件权限问题

确保temp目录有正确的读写权限:
```bash
chmod -R 755 UserInterface/assets/temp
```

### 2. 内存使用

对于大型点云文件，可能需要调整内存限制:
```bash
export MALLOC_ARENA_MAX=2
```

### 3. 并发处理

当前实现使用全局变量存储点云数据，在并发场景下需要注意:
- 考虑使用会话机制
- 实现数据隔离
- 添加并发锁

## 部署注意事项

1. 安全配置:
```python
# 生产环境配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["your-frontend-domain"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. 性能监控:
- 使用prometheus监控指标
- 设置合适的超时时间
- 实现健康检查端点

3. 备份策略:
- 定期清理临时文件
- 实现文件限额机制
