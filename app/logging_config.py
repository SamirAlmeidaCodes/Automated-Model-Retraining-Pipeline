"""Logging setup shared by the API, the CLI, and the scheduler."""

import logging
import sys


def setup_logging(level: str) -> None:
    """Configure the root logger without removing handlers another framework installed.

    Uvicorn installs its own handlers before the application lifespan runs.
    Adjust the level and formatter in place when those handlers already exist.
    """
    log_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setLevel(log_level)
            handler.setFormatter(formatter)
