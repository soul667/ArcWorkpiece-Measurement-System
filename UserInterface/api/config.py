# Python standard library
import os
import logging

# Logging configuration
log_file_path = "./UserInterface/fastapi.log"
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Server configuration
RELOAD_DIRS = ["."]
RELOAD_EXCLUDES = ["node_modules", ".git", "__pycache__", ".pytest_cache"]

# Asset paths
TEMP_DIR = os.path.join("UserInterface/assets", "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Default parameters
DEFAULT_NORMAL_NEIGHBORS = 30
DEFAULT_MIN_RADIUS = 6
DEFAULT_MAX_RADIUS = 11
DEFAULT_RANSAC_THRESHOLD = 0.01
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_NORMAL_DISTANCE_WEIGHT = 0.8
