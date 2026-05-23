from __future__ import annotations

import asyncio
import logging

from .devices.base import ServoController
from .events import ActionIntent

LOGGER = logging.getLogger(__name__)


class ActionDispatcher:
    def __init__(
        self,
        action_queue: asyncio.Queue[ActionIntent],
        servo_controller: ServoController,
    ) -> None:
        self.action_queue = action_queue
        self.servo_controller = servo_controller

    async def run(self) -> None:
        while True:
            intent = await self.action_queue.get()
            LOGGER.info("dispatching action intent: %s", intent)
            await self.servo_controller.execute(intent)
