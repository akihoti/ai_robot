from __future__ import annotations

import asyncio
import logging

from .config import EdgeConfig
from .devices.factory import build_camera
from .vision import CameraWorker, build_person_detector

LOGGER = logging.getLogger(__name__)


class EdgeApp:
    """Top-level application placeholder for the edge service."""

    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

    async def run(self) -> None:
        LOGGER.info("starting edge service for device %s", self.config.device_id)
        if self.config.runtime.mode == "simulated":
            LOGGER.info("running in simulated mode")
        vision_queue: asyncio.Queue = asyncio.Queue(maxsize=32)
        tasks: list[asyncio.Task] = []
        if self.config.camera.enabled:
            camera = build_camera(self.config)
            detector = build_person_detector(self.config)
            worker = CameraWorker(
                camera=camera,
                detector=detector,
                output_queue=vision_queue,
                frame_skip=self.config.camera.frame_skip,
            )
            tasks.append(asyncio.create_task(worker.run(), name="camera-worker"))

        tasks.append(asyncio.create_task(self._log_vision_events(vision_queue)))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()

    async def _log_vision_events(self, queue: asyncio.Queue) -> None:
        while True:
            event = await queue.get()
            LOGGER.info("vision event: %s", event)
