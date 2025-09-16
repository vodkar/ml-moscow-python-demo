"""Logging utilities for structured and consistent logging across the application."""

import logging
import sys
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    name: str = "titanic_ml",
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO",
    log_file: Path | None = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Set up a logger with optional file output and rich formatting.
    
    Args:
        name: Name of the logger.
        level: Logging level.
        log_file: Optional path to log file. If None, only console logging.
        use_rich: Whether to use Rich for enhanced console output.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set the logging level
    log_level = getattr(logging, level.upper())
    logger.setLevel(log_level)
    
    # Create formatter
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Console handler with Rich if requested
    if use_rich:
        console = Console(stderr=True)
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
    else:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    logger.info(f"Logger '{name}' initialized with level {level}")
    return logger