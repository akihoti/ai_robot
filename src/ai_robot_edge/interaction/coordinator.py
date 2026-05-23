from __future__ import annotations

import asyncio
import logging
from time import monotonic

from ..config import VisionConfig
from ..events import ActionIntent, ActionName, VisionEvent, VisionEventType

LOGGER = logging.getLogger(__name__)


class InteractionCoordinator:
    def __init__(
        self,
        vision_config: VisionConfig,
        vision_queue: asyncio.Queue[VisionEvent],
        action_queue: asyncio.Queue[ActionIntent],
    ) -> None:
        self.vision_config = vision_config
        self.vision_queue = vision_queue
        self.action_queue = action_queue
        self._present_frames = 0
        self._last_welcome_at = 0.0

    async def run(self) -> None:
        while True:
            event = await self.vision_queue.get()
            if event.event_type == VisionEventType.PERSON_PRESENT:
                await self._handle_person_present(event)
            else:
                self._present_frames = 0
                LOGGER.debug("person absent; stable counter reset")

    async def _handle_person_present(self, event: VisionEvent) -> None:
        self._present_frames += 1
        if self._present_frames < self.vision_config.stable_frames:
            LOGGER.debug(
                "person candidate %s/%s",
                self._present_frames,
                self.vision_config.stable_frames,
            )
            return

        now = monotonic()
        elapsed = now - self._last_welcome_at
        if elapsed < self.vision_config.welcome_cooldown_seconds:
            LOGGER.debug("welcome suppressed by cooldown: %.1fs", elapsed)
            return

        self._last_welcome_at = now
        welcome_event = VisionEvent(
            event_type=VisionEventType.WELCOME_TRIGGERED,
            confidence=event.confidence,
            source=event.source,
            cooldown_active=False,
        )
        LOGGER.info("welcome triggered: %s", welcome_event)
        await self.action_queue.put(ActionIntent(name=ActionName.WELCOME_MOTION))
