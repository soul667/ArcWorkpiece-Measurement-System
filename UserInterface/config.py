import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    # Database settings
    MYSQL_ROOT_PASSWORD = os.getenv('MYSQL_ROOT_PASSWORD', 'mysqlroot')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'arcworkpiece')
    MYSQL_USER = os.getenv('MYSQL_USER', 'dev')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'devpass')
    DB_HOST = os.getenv('DB_HOST', 'mysql')
    DB_PORT = int(os.getenv('DB_PORT', 3306))

    # JWT settings
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your-production-secret-key-here')
    JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', 30))

    # Server settings
    BACKEND_HOST = os.getenv('BACKEND_HOST', '0.0.0.0')
    BACKEND_PORT = int(os.getenv('BACKEND_PORT', 12345))
    FRONTEND_PORT = int(os.getenv('FRONTEND_PORT', 3000))

    # CORS settings
    ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:3000').split(',')
    CORS_ALLOW_CREDENTIALS = os.getenv('CORS_ALLOW_CREDENTIALS', 'true').lower() == 'true'
    CORS_ALLOW_METHODS = os.getenv('CORS_ALLOW_METHODS', '*').split(',')
    CORS_ALLOW_HEADERS = os.getenv('CORS_ALLOW_HEADERS', '*').split(',')

    # Development settings
    DEBUG = os.getenv('DEBUG', 'true').lower() == 'true'
    RELOAD = os.getenv('RELOAD', 'true').lower() == 'true'

    # Storage settings
    TEMP_STORAGE_PATH = os.getenv('TEMP_STORAGE_PATH', 'UserInterface/assets/temp')

    # Logging settings
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'UserInterface/fastapi.log')

    # Ensure required directories exist
    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.TEMP_STORAGE_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(cls.LOG_FILE), exist_ok=True)

    # Database configuration dictionary
    @property
    def db_config(self):
        return {
            'host': self.DB_HOST,
            'user': self.MYSQL_USER,
            'password': self.MYSQL_PASSWORD,
            'database': self.MYSQL_DATABASE,
            'port': self.DB_PORT
        }

    # CORS configuration dictionary
    @property
    def cors_config(self):
        return {
            'allow_origins': self.ALLOWED_ORIGINS,
            'allow_credentials': self.CORS_ALLOW_CREDENTIALS,
            'allow_methods': self.CORS_ALLOW_METHODS,
            'allow_headers': self.CORS_ALLOW_HEADERS,
        }

# Create a global settings instance
settings = Settings()

# Setup required directories
settings.setup_directories()
