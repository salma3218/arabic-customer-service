"""
Logging utility using loguru
"""

from loguru import logger
import sys
from pathlib import Path

# Import settings
try:
    from config.settings import LOG_LEVEL, LOG_FORMAT
except ImportError:
    # Fallback if settings not available
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Add custom handler for console output
logger.add(
    sys.stdout,
    format=LOG_FORMAT,
    level=LOG_LEVEL,
    colorize=True
    # ✅ REMOVED: encoding parameter (not supported by loguru)
)

# Add file handler for errors
logger.add(
    "logs/errors.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="10 MB"
    # ✅ REMOVED: encoding parameter (not supported by loguru)
)

# Add file handler for all logs (optional, useful for debugging)
logger.add(
    "logs/application.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="50 MB",
    retention="7 days"
    # ✅ REMOVED: encoding parameter (not supported by loguru)
)


def get_logger(name: str):
    """
    Get logger instance with specific name
    
    Args:
        name: Logger name (usually __name__ from calling module)
        
    Returns:
        Logger instance bound to the given name
        
    Example:
        >>> from utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process...")
    """
    return logger.bind(name=name)


# Import text formatters (must be after logger setup to avoid circular import)
try:
    from utils.text_formatter import (
        format_query_preview,
        format_log_safe,
        format_arabic_for_terminal
    )
    
    __all__ = [
        'get_logger',
        'format_query_preview',
        'format_log_safe',
        'format_arabic_for_terminal'
    ]
except ImportError:
    # If text_formatter not available, provide dummy functions
    def format_query_preview(query: str, max_length: int = 50) -> str:
        """Fallback query formatter"""
        if len(query) > max_length:
            return query[:max_length - 3] + "..."
        return query
    
    def format_log_safe(text: str, max_length: int = 100) -> str:
        """Fallback safe formatter"""
        if len(text) > max_length:
            return text[:max_length - 3] + "..."
        return text
    
    def format_arabic_for_terminal(text: str, max_length: int = 50) -> str:
        """Fallback Arabic formatter"""
        return format_query_preview(text, max_length)
    
    __all__ = [
        'get_logger',
        'format_query_preview',
        'format_log_safe',
        'format_arabic_for_terminal'
    ]