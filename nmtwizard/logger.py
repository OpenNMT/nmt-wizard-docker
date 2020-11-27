"""Logging utilities."""

import os
import logging

logging.basicConfig(format='%(asctime)s.%(msecs)06d [%(module)s@%(processName)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%SZ',
                    level=logging.INFO)

def get_logger(name=None):
    """Returns a logger with configured level."""
    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    return logger
