"""Structured JSON logging utilities.

This module provides structured logging with request IDs, context tracking,
and JSON formatting for production observability.
"""

import contextvars
import logging
import sys
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pythonjsonlogger import jsonlogger

# Context variable for request/trace ID
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")


class StructuredLogger(logging.Logger):
    """Logger that automatically includes structured context."""

    def _log(
        self,
        level: int,
        msg: Any,
        args: Any,
        exc_info: Any = None,
        extra: Optional[Mapping[str, object]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
    ) -> None:
        """Override _log to inject request_id into all log records."""
        # Create mutable dict for extra fields
        extra_dict: dict = dict(extra) if extra else {}

        # Add request ID if available
        request_id = request_id_var.get("")
        if request_id:
            extra_dict["request_id"] = request_id

        # Add timestamp
        extra_dict["timestamp"] = datetime.now(timezone.utc).isoformat()

        super()._log(level, msg, args, exc_info, extra_dict, stack_info, stacklevel + 1)


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields."""

    def add_fields(
        self, log_record: dict[str, Any], record: logging.LogRecord, message_dict: dict[str, Any]
    ) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)

        # Add standard fields
        log_record["timestamp"] = datetime.now(timezone.utc).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno

        # Add process/thread info
        log_record["process_id"] = record.process
        log_record["thread_id"] = record.thread

        # Add request_id if available
        if hasattr(record, "request_id"):
            log_record["request_id"] = getattr(record, "request_id", None)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
            }


def setup_structured_logging(
    level: int = logging.INFO,
    format_json: bool = True,
    include_timestamp: bool = True,
) -> None:
    """Setup structured logging for the application.

    Args:
        level: Logging level (default: INFO)
        format_json: Whether to format logs as JSON (default: True)
        include_timestamp: Whether to include timestamp in logs (default: True)

    Example:
        setup_structured_logging(level=logging.DEBUG, format_json=True)
    """
    # Replace default logger class
    logging.setLoggerClass(StructuredLogger)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if format_json:
        # Use JSON formatter
        formatter: logging.Formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s",
            rename_fields={
                "levelname": "level",
                "name": "logger",
                "pathname": "file_path",
            },
        )
    else:
        # Use standard formatter
        if include_timestamp:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        else:
            fmt = "%(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set level for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Operation completed", extra={"user_id": "123", "count": 5})
    """
    return logging.getLogger(name)


def generate_request_id() -> str:
    """Generate a unique request ID.

    Returns:
        UUID-based request ID
    """
    return str(uuid.uuid4())


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set request ID for current context.

    Args:
        request_id: Request ID to set (generates new one if not provided)

    Returns:
        The request ID that was set

    Example:
        request_id = set_request_id()
        logger.info("Starting request")  # Will include request_id
    """
    if request_id is None:
        request_id = generate_request_id()

    request_id_var.set(request_id)
    return request_id


def get_request_id() -> str:
    """Get current request ID from context.

    Returns:
        Current request ID or empty string if not set
    """
    return request_id_var.get("")


def clear_request_id() -> None:
    """Clear request ID from current context."""
    request_id_var.set("")


class RequestContext:
    """Context manager for request/operation tracking.

    Automatically generates and sets request ID for the duration of the context.

    Example:
        with RequestContext() as ctx:
            logger.info("Processing request")  # Includes ctx.request_id
            # ... do work ...
        # request_id automatically cleared
    """

    def __init__(self, request_id: Optional[str] = None, operation: Optional[str] = None):
        """Initialize request context.

        Args:
            request_id: Optional request ID (generates new one if not provided)
            operation: Optional operation name for logging
        """
        self.request_id = request_id or generate_request_id()
        self.operation = operation
        self.start_time: Optional[datetime] = None
        self.logger = get_logger(__name__)

    def __enter__(self) -> "RequestContext":
        """Enter context - set request ID."""
        set_request_id(self.request_id)
        self.start_time = datetime.now(timezone.utc)

        if self.operation:
            self.logger.info(
                f"Starting operation: {self.operation}",
                extra={"operation": self.operation, "request_id": self.request_id},
            )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Exit context - clear request ID and log completion."""
        duration = (
            (datetime.now(timezone.utc) - self.start_time).total_seconds()
            if self.start_time
            else 0.0
        )

        if exc_type is None:
            if self.operation:
                self.logger.info(
                    f"Completed operation: {self.operation}",
                    extra={
                        "operation": self.operation,
                        "request_id": self.request_id,
                        "duration_seconds": duration,
                        "status": "success",
                    },
                )
        else:
            if self.operation:
                self.logger.error(
                    f"Failed operation: {self.operation}",
                    extra={
                        "operation": self.operation,
                        "request_id": self.request_id,
                        "duration_seconds": duration,
                        "status": "error",
                        "error_type": exc_type.__name__,
                        "error_message": str(exc_val),
                    },
                    exc_info=True,
                )

        clear_request_id()
        return False  # Don't suppress exceptions


def log_with_context(logger: logging.Logger, level: int, message: str, **kwargs: Any) -> None:
    """Log message with additional context.

    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **kwargs: Additional context to include in log

    Example:
        log_with_context(
            logger,
            logging.INFO,
            "User logged in",
            user_id="123",
            ip_address="192.168.1.1"
        )
    """
    extra = kwargs.copy()
    logger.log(level, message, extra=extra)
