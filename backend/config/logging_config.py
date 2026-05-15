"""
logging_config.py
=================
JSON structured logging for the NBA ML Predictor.
Replaces all print() statements with proper logging.
"""

import logging
import sys
import os
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON for structured logging."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0]:
            log_data["exception"] = self.formatException(record.exc_info)
        return str(log_data)


def get_logger(name: str) -> logging.Logger:
    """
    Creates a configured logger instance.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger with JSON formatting

    Usage:
        from config.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Predicción exitosa")
        logger.error("Error en predicción", exc_info=True)
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Console handler (JSON structured) ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Use JSON in production, readable format in development
    if os.environ.get("PORT"):
        # Production: JSON format
        console_handler.setFormatter(JSONFormatter())
    else:
        # Development: Human-readable format
        console_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)-20s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger.addHandler(console_handler)

    # ── File handler (for debugging, only in development) ──
    if not os.environ.get("PORT"):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "app.log"), encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    return logger
