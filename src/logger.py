import logging
import os
from datetime import datetime


def setup_logger():
    """Set up the logger configuration"""
    # Create log filename
    LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"

    # Create logs directory
    logs_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Create full path to log file
    LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

    # Configure logging
    logging.basicConfig(
        filename=LOG_FILE_PATH,
        format='%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        force=True  # This ensures the configuration is applied even if logging was previously configured
    )

    return LOG_FILE_PATH


def get_logger(name):
    """Get a logger instance"""
    return logging.getLogger(name)
