"""Logging utilities."""

import os
import logging

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d000 UTC [%(module)s@%(processName)s] %(levelname)s %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
    level=os.getenv("LOG_LEVEL", "INFO"),
)


def get_logger(name=None):
    """Returns a logger with configured level."""
    logger = logging.getLogger(name)
    return logger
