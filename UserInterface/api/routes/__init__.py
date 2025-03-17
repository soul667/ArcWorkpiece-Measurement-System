"""API路由模块"""
# from UserInterface.api.routes.auth import router as auth_router
# from UserInterface.auth.routes import router as auth_router
from UserInterface.api.routes.settings import router as settings_router
from UserInterface.api.routes.files import router as files_router
from UserInterface.api.routes.point_cloud import router as point_cloud_router
from UserInterface.api.routes.temp_save import router as temp_cloud_router
# 为了方便导入，提供所有路由模块
__all__ = ['settings', 'files', 'point_cloud', 'temp_save']
