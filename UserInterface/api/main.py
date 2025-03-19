import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from UserInterface.api.routes import  settings, files, point_cloud, temp_save, history
from UserInterface.api.config import logger, RELOAD_DIRS, RELOAD_EXCLUDES, TEMP_DIR
from UserInterface.auth.routes import router as auth_router
from UserInterface.auth.init_db import init_database
from UserInterface.auth.db import Database

# 创建FastAPI应用
app = FastAPI(title="圆弧工件测量系统")

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行的操作"""
    logger.info("初始化系统...")
    
    # 创建必要的目录
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"已创建临时目录: {TEMP_DIR}")
    
    # 检查并初始化数据库
    logger.info("检查数据库表...")
    db = Database()
    if not db.check_table_exists('temp_clouds') or not db.check_table_exists('measurement_history'):
        logger.info("temp_clouds 表不存在，开始初始化数据库...")
        if init_database():
            logger.info("数据库初始化完成")
        else:
            logger.error("数据库初始化失败")
    else:
        logger.info("数据库表检查完成，temp_clouds 表已存在")

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

# 注册路由
app.include_router(auth_router, prefix="/auth", tags=["authentication"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(point_cloud.router, prefix="/api/point-cloud", tags=["point-cloud"])
app.include_router(temp_save.router, prefix="/api/temp-save", tags=["temp-save"])
app.include_router(history.router, prefix="/api/history", tags=["history"])

@app.get("/")
async def home(request: Request):
    """主页路由"""
    return templates.TemplateResponse("index1.html", {"request": request})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器"""
    logger.error(f"Request {request.url} failed: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    
    # 创建必要的目录
    os.makedirs(TEMP_DIR, exist_ok=True)

    # 启动服务器
    uvicorn.run(
        "UserInterface.api.main:app",
        host="0.0.0.0",
        port=9304,
        reload=True,
        reload_dirs=RELOAD_DIRS,
        reload_excludes=RELOAD_EXCLUDES
    )
