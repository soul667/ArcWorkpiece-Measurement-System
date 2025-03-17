"""API模块

此模块包含FastAPI应用程序的所有组件，包括：
- 路由处理器
- 数据模型
- 工具函数
- 配置
"""

from UserInterface.api.main import app
from UserInterface.api.config import logger, TEMP_DIR

__version__ = "1.0.0"
__all__ = ['app', 'logger', 'TEMP_DIR']
