"""Unified logger setup for the benchmark harness."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from .paths import LOGS_DIR


def get_logger(name: str, logfile: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if logfile is None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logfile = LOGS_DIR / f"{name.replace('.', '_')}.log"
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger
