"""Comprehensive logging configuration for the AI Science Platform"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class SecurityFilter(logging.Filter):
    """Filter to remove sensitive information from logs"""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'api_key',
        'auth', 'credential', 'private', 'confidential'
    ]
    
    def filter(self, record):
        """Remove sensitive information from log records"""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            msg_lower = record.msg.lower()
            for pattern in self.SENSITIVE_PATTERNS:
                if pattern in msg_lower:
                    record.msg = record.msg.replace(
                        record.msg[record.msg.lower().find(pattern):],
                        '[REDACTED]'
                    )
        return True


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    enable_json: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> Dict[str, Any]:
    """Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: ./logs)
        enable_console: Enable console logging
        enable_file: Enable file logging
        enable_json: Use JSON format for logs
        max_file_size: Maximum file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Dictionary with logging configuration details
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Formatter selection
    if enable_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Security filter
    security_filter = SecurityFilter()
    
    handlers_info = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.addFilter(security_filter)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(console_handler)
        handlers_info.append("console")
    
    # File handlers
    if enable_file:
        # Main log file (rotating)
        main_log_file = log_dir / "ai_science_platform.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        main_handler.setFormatter(formatter)
        main_handler.addFilter(security_filter)
        main_handler.setLevel(getattr(logging, log_level.upper()))
        root_logger.addHandler(main_handler)
        handlers_info.append(f"file: {main_log_file}")
        
        # Error log file (errors and critical only)
        error_log_file = log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.addFilter(security_filter)
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        handlers_info.append(f"error_file: {error_log_file}")
    
    # Performance logger (separate logger for performance metrics)
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.handlers.RotatingFileHandler(
        log_dir / "performance.log",
        maxBytes=max_file_size,
        backupCount=backup_count
    )
    perf_handler.setFormatter(formatter)
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    config_info = {
        "log_level": log_level,
        "log_directory": str(log_dir),
        "handlers": handlers_info,
        "json_format": enable_json,
        "max_file_size_mb": max_file_size / (1024 * 1024),
        "backup_count": backup_count
    }
    
    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: {config_info}")
    
    return config_info


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return logging.getLogger(name)


def log_performance(operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    perf_logger = logging.getLogger("performance")
    perf_data = {
        "operation": operation,
        "duration_seconds": duration,
        "timestamp": datetime.now().isoformat(),
        **kwargs
    }
    perf_logger.info(f"PERFORMANCE: {json.dumps(perf_data)}")


# Configure logging on import
setup_logging()