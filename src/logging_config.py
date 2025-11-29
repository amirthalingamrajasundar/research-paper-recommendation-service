"""
Centralized logging configuration for the project.
"""
import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging with consistent formatting.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        name: Logger name (optional, uses root if None)
    
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default project logger
logger = setup_logging(name="scholar_stream")
