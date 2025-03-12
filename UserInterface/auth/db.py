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
            
    def initialize_tables(self):
        """Initialize database tables"""
        if not self.connection or not self.connection.is_connected():
            self.connect()
            
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
            logger.info("Database tables initialized successfully")
            return True
        except Error as e:
            logger.error(f"Error initializing tables: {e}")
            return False
        finally:
            cursor.close()
            
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
