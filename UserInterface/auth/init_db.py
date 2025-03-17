from .db import Database
from .service import auth_service  # Use the global instance
import logging

logger = logging.getLogger(__name__)

def init_database():
    """Initialize database and default user"""
    try:
        logger.info("开始初始化数据库...")
        
        # Initialize database tables using the singleton Database instance
        db = Database()
        
        # 检查 SQL 文件
        sql_file = os.path.join(os.path.dirname(__file__), 'init_db_update_temp_clouds.sql')
        if not os.path.exists(sql_file):
            logger.error(f"SQL 文件不存在: {sql_file}")
            return False
        logger.info(f"找到 SQL 文件: {sql_file}")
        
        # 检查数据库连接
        try:
            with db.get_connection() as conn:
                logger.info("数据库连接测试成功")
        except Exception as e:
            logger.error(f"数据库连接失败: {str(e)}")
            return False
            
        # 初始化表
        if not db.initialize_tables():
            logger.error("表初始化失败")
            return False
        logger.info("数据库表初始化成功")

        # Check if admin user exists using the global auth_service instance
        query = "SELECT * FROM users WHERE username = %s"
        users = db.execute_query(query, ("admin",))

        if not users:
            # Create default admin user if it doesn't exist
            if auth_service.create_user("admin", "admin123"):
                logger.info("Default admin account created successfully")
            else:
                logger.error("Failed to create default admin account")
                return False
        else:
            logger.info("Default admin account already exists")

        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('auth_init.log')
        ]
    )

    # Initialize database
    if init_database():
        print("""
=== Authentication System Initialized Successfully ===

Default admin account:
Username: admin
Password: admin123

Please change the default password!
""")
    else:
        print("Authentication system initialization failed. Please check the log file.")
