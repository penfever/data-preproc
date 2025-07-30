"""Simple logging configuration for data preprocessing."""

import logging
import os
import sys


def get_logger(
    name: str, use_environ: bool = False, log_level: str = "INFO"
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name.
        use_environ: Whether to use LOG_LEVEL from environment.
        log_level: Default log level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    if use_environ:
        log_level = os.environ.get("LOG_LEVEL", log_level)
    
    # Configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, log_level.upper()))
    
    return logger