"""Loads config/logging.yaml and configures the root logger."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGGING_PATH = PROJECT_ROOT / "config" / "logging.yaml"

_configured = False


def setup_logging(path: Path | str = DEFAULT_LOGGING_PATH) -> None:
    global _configured
    if _configured:
        return
    config = yaml.safe_load(Path(path).read_text())
    logging.config.dictConfig(config)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
