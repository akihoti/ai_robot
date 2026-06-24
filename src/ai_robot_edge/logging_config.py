from __future__ import annotations

import logging
import os
from pathlib import Path


def configure_logging(level: str) -> None:
    log_format = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    log_file = os.environ.get("AI_ROBOT_LOG_FILE", "").strip()
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,
    )
