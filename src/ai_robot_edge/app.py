from __future__ import annotations

import asyncio
import logging

from .config import EdgeConfig

LOGGER = logging.getLogger(__name__)


class EdgeApp:
    """Top-level application placeholder for the edge service."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

    async def run(self) -> None:
        LOGGER.info("starting edge service for device %s", self.config.device_id)
        if self.config.runtime.mode == "simulated":
            LOGGER.info("running in simulated mode")
        await asyncio.Event().wait()
