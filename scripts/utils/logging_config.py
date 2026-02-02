# ABOUTME: Logging configuration for GNSS-IR processing pipeline
# ABOUTME: Sets up main and day-specific loggers with console and file handlers

import logging
from pathlib import Path

def setup_main_logger(log_file, log_level=logging.INFO):
    """
    Setup the main logger for both console and file output.

    Args:
        log_file (str or Path): Path to the main log file
        log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
                                  Defaults to logging.INFO.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Ensure log directory exists
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Silence noisy third-party loggers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)
    
    # File handler (DEBUG level to capture all details)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)
    
    logging.info(f"Main logger initialized with log file: {log_file}")
    logging.info(f"Console logging level: {logging.getLevelName(log_level)}")
    
    return root_logger

def setup_day_logger(log_file, log_level=logging.DEBUG):
    """
    Setup a logger for a specific DOY or processing task.
    
    Args:
        log_file (str or Path): Path to the log file for this specific task
        log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG).
                                  Defaults to logging.DEBUG.
                                  
    Returns:
        logging.Logger: Configured logger instance
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    # Ensure parent directory exists
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a new logger for this day/task
    logger_name = str(log_file)
    day_logger = logging.getLogger(logger_name)
    day_logger.setLevel(log_level)
    
    # Remove any existing handlers
    if day_logger.handlers:
        for handler in day_logger.handlers:
            day_logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    day_logger.addHandler(file_handler)
    
    # Keep propagation to root logger so console output works too
    day_logger.propagate = True
    
    return day_logger
