"""Shared logger factory so every module logs consistently to file + console."""

from __future__ import annotations

import logging
import sys

from config.settings import LOG_FILE, LOG_LEVEL

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        root = logging.getLogger()
        root.setLevel(LOG_LEVEL)

        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)

        root.addHandler(file_handler)
        root.addHandler(console_handler)
        _CONFIGURED = True

    return logger
