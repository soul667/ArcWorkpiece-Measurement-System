import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool
import os
import logging
from ..config import settings

logger = logging.getLogger(__name__)

class Database:
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if cls._pool is None:
                pool_config = {
                    **settings.db_config,
                    'pool_name': 'mypool',
                    'pool_size': 10,
                    'pool_reset_session': True,
                    'connect_timeout': 10
                }
                try:
                    cls._pool = MySQLConnectionPool(**pool_config)
                    logger.info('Successfully initialized MySQL connection pool')
                except Error as e:
                    logger.error(f"Failed to initialize connection pool: {e}")
                    raise
        return cls._instance

    def __init__(self):
        pass

    def get_connection(self):
        """Get a connection from the pool"""
        try:
            return self._pool.get_connection()
        except Error as e:
            logger.error(f"Error getting connection from pool: {e}")
            raise

    def execute_query(self, query, params=None):
        """Execute a database query with proper connection handling"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self.get_connection() as connection:
                    with connection.cursor(dictionary=True) as cursor:
                        cursor.execute(query, params or ())
                        
                        if query.strip().upper().startswith('SELECT'):
                            result = cursor.fetchall()
                            return result
                        else:
                            connection.commit()
                            return cursor.lastrowid
                            
            except Error as e:
                retry_count += 1
                logger.error(f"Error executing query (attempt {retry_count}): {e}")
                if retry_count == max_retries:
                    raise
                continue

    def execute_sql_file(self, file_path):
        """Execute SQL file with proper connection handling"""
        try:
            with open(file_path, 'r') as f:
                sql_content = f.read()
                statements = sql_content.split(';')
                
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    for statement in statements:
                        if statement.strip():
                            cursor.execute(statement)
                    connection.commit()
                    logger.info(f"SQL file executed successfully: {file_path}")
                    return True
        except Error as e:
            logger.error(f"Failed to execute SQL file: {str(e)}")
            return False
                
    def _create_base_tables(self):
        """Create base tables with proper connection handling"""
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
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
                    
                    connection.commit()
                    logger.info("Base tables initialized successfully")
                    return True
        except Error as e:
            logger.error(f"Failed to create base tables: {str(e)}")
            return False
            
    def initialize_tables(self):
        """Initialize database tables"""
        # Execute base tables creation
        base_tables_result = self._create_base_tables()
        
        # Execute temporary clouds table creation
        # temp_clouds_sql = os.path.join(os.path.dirname(__file__), 'init_db_update_temp_clouds.sql')
        # clouds_table_result = self.execute_sql_file(temp_clouds_sql)
        temp_clouds_sql = os.path.join(os.path.dirname(__file__), 'init_db_update_temp_clouds.sql')
        logger.info(f"Loading SQL file from: {temp_clouds_sql}")
        clouds_table_result = self.execute_sql_file(temp_clouds_sql)
        logger.info(f"Temp clouds table creation result: {clouds_table_result}")
        return base_tables_result and clouds_table_result

    def __enter__(self):
        """Context manager entry - returns self since we're using connection pool"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - no need to close anything since connections return to pool"""
        pass
        
    def check_table_exists(self, table_name: str) -> bool:
        """检查表是否存在
        
        Args:
            table_name: 要检查的表名
            
        Returns:
            bool: 表是否存在
        """
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
                    return cursor.fetchone() is not None
        except Error as e:
            logger.error(f"Error checking table {table_name}: {str(e)}")
            return False
