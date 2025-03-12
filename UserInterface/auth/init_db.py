from .db import Database
from .service import AuthService
import logging

logger = logging.getLogger(__name__)

def init_database():
    """初始化数据库和默认用户"""
    try:
        # 初始化数据库连接和表
        db = Database()
        db.initialize_tables()
        logger.info("数据库表初始化成功")

        # 创建默认管理员用户
        auth_service = AuthService()
        
        # 检查admin用户是否已存在
        query = "SELECT * FROM users WHERE username = %s"
        users = db.execute_query(query, ("admin",))
        # print(users)
        logger.info("默认管理员账号创建成功")

        if not users:
            success = auth_service.create_user("admin", "admin123")
            if success:
                logger.info("默认管理员账号创建成功")
            else:
                logger.error("创建默认管理员账号失败")
        else:
            logger.info("默认管理员账号已存在")

        return True
    except Exception as e:
        logger.error(f"数据库初始化失败: {str(e)}")
        return False

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('auth_init.log')
        ]
    )

    # 初始化数据库
    if init_database():
        print("""
=== 认证系统初始化成功 ===

默认管理员账号:
用户名: admin
密码: admin123

请及时修改默认密码！
""")
    else:
        print("认证系统初始化失败，请检查日志文件。")
