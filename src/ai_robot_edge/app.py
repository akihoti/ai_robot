from __future__ import annotations

import asyncio
import logging

from .config import EdgeConfig
from .actions import ActionDispatcher
from .devices.factory import build_camera, build_servo_controller
from .events import ActionIntent, VisionEvent
from .interaction import InteractionCoordinator
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
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue(maxsize=32)
        action_queue: asyncio.Queue[ActionIntent] = asyncio.Queue(maxsize=32)
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

        coordinator = InteractionCoordinator(
            vision_config=self.config.vision,
            vision_queue=vision_queue,
            action_queue=action_queue,
        )
        dispatcher = ActionDispatcher(
            action_queue=action_queue,
            servo_controller=build_servo_controller(self.config),
        )
        tasks.append(asyncio.create_task(coordinator.run(), name="interaction"))
        tasks.append(asyncio.create_task(dispatcher.run(), name="actions"))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
