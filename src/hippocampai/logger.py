"""Logging configuration and utilities for HippocampAI."""

import logging
import logging.config
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_file: Optional[str] = None,
    log_level: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> None:
    """
    Set up logging configuration.

    Args:
        config_file: Path to logging_config.yaml (optional)
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Override log directory path

    Example:
        setup_logging(log_level="DEBUG", log_dir="custom_logs")
    """
    # Determine project root
    project_root = Path(__file__).resolve().parents[2]

    # Load logging config file
    if config_file is None:
        config_file = project_root / "config" / "logging_config.yaml"
    else:
        config_file = Path(config_file)

    # Create logs directory
    if log_dir:
        logs_path = Path(log_dir)
    else:
        logs_path = project_root / "logs"

    logs_path.mkdir(parents=True, exist_ok=True)

    # Try to load YAML config
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            # Update log file paths to use log_dir
            for handler_name, handler_config in config.get("handlers", {}).items():
                if "filename" in handler_config:
                    filename = Path(handler_config["filename"]).name
                    handler_config["filename"] = str(logs_path / filename)

            # Apply log level override
            if log_level:
                config["root"]["level"] = log_level.upper()
                for logger_config in config.get("loggers", {}).values():
                    logger_config["level"] = log_level.upper()

            # Apply configuration
            logging.config.dictConfig(config)

            logging.info(f"Logging configured from {config_file}")
            return

        except Exception as e:
            print(f"Warning: Failed to load logging config: {e}", file=sys.stderr)
            # Fall through to basic config

    # Fallback to basic configuration
    _setup_basic_logging(log_level, logs_path)


def _setup_basic_logging(log_level: Optional[str] = None, logs_path: Path = None) -> None:
    """Set up basic logging configuration as fallback."""
    if logs_path is None:
        logs_path = Path("logs")
        logs_path.mkdir(parents=True, exist_ok=True)

    # Determine log level
    level = getattr(logging, log_level.upper()) if log_level else logging.INFO

    # Create formatters
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)

    # File handler with rotation
    log_file = logs_path / "hippocampai.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Error file handler
    error_file = logs_path / "hippocampai_error.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logging.info(f"Basic logging configured (level={logging.getLevelName(level)})")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    Mixin class that provides logging functionality.

    Usage:
        class MyClass(LoggerMixin):
            def my_method(self):
                self.logger.info("Message")
    """

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)
