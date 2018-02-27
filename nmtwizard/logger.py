"""Logging utilities."""

import os
import logging

logging.basicConfig()


def get_logger(name=None):
    """Returns a logger with configured level."""
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logger
