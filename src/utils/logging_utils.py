"""
Logging utilities for training and inference.
Provides consistent formatting and optional file logging.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import json


class JsonFormatter(logging.Formatter):
    """Simple JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def setup_logger(
    log_path: Optional[Path] = None,
    level: int = logging.INFO,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure root logger with console and optional file handler.

    Args:
        log_path: Optional log file path.
        level: Logging level.
        json_format: Emit JSON logs if True.
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = JsonFormatter() if json_format else logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, float],
    prefix: str,
    step: Optional[int] = None,
) -> None:
    """Log a set of metrics with a consistent prefix."""
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            if step is not None:
                logger.info(f"{prefix}/{key}={value:.6f} step={step}")
            else:
                logger.info(f"{prefix}/{key}={value:.6f}")