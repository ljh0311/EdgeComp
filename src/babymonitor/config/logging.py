"""
Logging Configuration
-------------------
Handles logging configuration for the Baby Monitor System.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logging(config):
    """
    Set up logging configuration for the application
    
    Args:
        config (dict): Logging configuration dictionary containing:
            - level: Logging level (str)
            - format: Log message format (str)
            - file: Log file path (str)
            - max_size: Maximum size of log file before rotation (int)
            - backup_count: Number of backup files to keep (int)
            - console_output: Whether to output logs to console (bool)
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(config.get('file', 'logs/babymonitor.log')).parent
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(config.get('level', 'INFO'))
    
    # Create formatter
    formatter = logging.Formatter(
        config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Set up file handler with rotation
    file_handler = RotatingFileHandler(
        config.get('file', 'logs/babymonitor.log'),
        maxBytes=config.get('max_size', 10 * 1024 * 1024),  # Default 10MB
        backupCount=config.get('backup_count', 5)
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Set up console handler if enabled
    if config.get('console_output', True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Create logger for this module
    logger = logging.getLogger(__name__)
    logger.info('Logging system initialized')
    
    return logger

def get_logger(name):
    """
    Get a logger instance with the specified name
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name) 