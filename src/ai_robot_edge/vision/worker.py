from __future__ import annotations

import asyncio
import logging

from ..devices.base import Camera
from ..events import VisionEvent, VisionEventType
from .detector import PersonDetector

LOGGER = logging.getLogger(__name__)


class CameraWorker:
    def __init__(
        self,
        camera: Camera,
        detector: PersonDetector,
        output_queue: asyncio.Queue[VisionEvent],
        frame_skip: int = 1,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.output_queue = output_queue
        self.frame_skip = max(1, frame_skip)

    async def run(self) -> None:
        async for frame in self.camera.frames():
            if frame.sequence % self.frame_skip != 0:
                continue
            result = await self.detector.detect(frame)
            event_type = (
                VisionEventType.PERSON_PRESENT
                if result.person_present
                else VisionEventType.PERSON_ABSENT
            )
            event = VisionEvent(
                event_type=event_type,
                confidence=result.confidence,
                source=f"camera:{result.backend}",
            )
            await self.output_queue.put(event)
            LOGGER.debug("vision event queued: %s", event)
