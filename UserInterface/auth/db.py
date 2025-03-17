import mysql.connector
from mysql.connector import Error
import os
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.config = settings.db_config
        self.connection = None
        
    def connect(self):
        """Connect to the MySQL database"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.config)
                if self.connection.is_connected():
                    logger.info('Successfully connected to MySQL database')
                    return True
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return False
            
    def execute_sql_file(self, file_path):
        """执行SQL文件中的所有语句"""
        try:
            with open(file_path, 'r') as f:
                sql_content = f.read()
                statements = sql_content.split(';')
                
                cursor = self.connection.cursor()
                for statement in statements:
                    if statement.strip():
                        cursor.execute(statement)
                
                self.connection.commit()
                logger.info(f"SQL file executed successfully: {file_path}")
                return True
        except Error as e:
            logger.error(f"执行SQL文件失败: {str(e)}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
                
    def _create_base_tables(self):
        """创建基础表(users和parameter_settings)"""
        try:
            cursor = self.connection.cursor()
            
            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    username VARCHAR(50) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create parameter_settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_settings (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    user_id INT NOT NULL,
                    name VARCHAR(100) NOT NULL,
                    cylinder_settings JSON NOT NULL,
                    arc_settings JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
            
            self.connection.commit()
            logger.info("Base tables initialized successfully")
            return True
        except Error as e:
            logger.error(f"创建基础表失败: {str(e)}")
            return False
        finally:
            cursor.close()
            
    def initialize_tables(self):
        """Initialize database tables"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
            
        # 执行基础表创建
        base_tables_result = self._create_base_tables()
        
        # 执行临时点云表创建
        temp_clouds_sql = os.path.join(os.path.dirname(__file__), 'init_db_update_temp_clouds.sql')
        clouds_table_result = self.execute_sql_file(temp_clouds_sql)
        
        return base_tables_result and clouds_table_result
            
    def execute_query(self, query, params=None):
        """Execute a database query

        Args:
            query (str): SQL query to execute
            params (tuple, optional): Query parameters. Defaults to None.

        Returns:
            list: Query results for SELECT queries
            int: Last row id for INSERT queries
        """
        try:
            if not self.connection or not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                return result
            else:
                self.connection.commit()
                return cursor.lastrowid
        except Error as e:
            logger.error(f"Error executing query: {e}")
            raise
        finally:
            cursor.close()

    def close(self):
        """Close the database connection"""
        try:
            if self.connection and self.connection.is_connected():
                self.connection.close()
                logger.info('Database connection closed')
        except Error as e:
            logger.error(f"Error closing database connection: {e}")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
