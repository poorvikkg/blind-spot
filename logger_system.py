"""
=============================================================================
  BLIND SPOT DETECTION SYSTEM — Logging Module
  File: logger_system.py
=============================================================================
  Provides structured, rotating file logging plus console output.
  Every detection event, alert, and system error is recorded with a
  timestamp and severity level.
=============================================================================
"""

import logging
import os
from logging.handlers import RotatingFileHandler

import config


def get_logger(name: str = "BlindSpot") -> logging.Logger:
    """
    Build and return a configured logger instance.

    The logger writes to both:
      • Console  (INFO and above)
      • Rotating log file (DEBUG and above, configurable in config.py)

    Args:
        name: Logger name (appears in every log line prefix).

    Returns:
        logging.Logger: Configured logger ready for use.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ─────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler (rotating) ──────────────────────────────────────────────
    if config.ENABLE_FILE_LOGGING:
        log_dir = os.path.dirname(config.LOG_FILE_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=config.LOG_FILE_PATH,
            maxBytes=config.LOG_MAX_BYTES,
            backupCount=config.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ── Module-level default logger ──────────────────────────────────────────────
log = get_logger()
