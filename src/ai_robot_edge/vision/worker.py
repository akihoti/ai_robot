from __future__ import annotations

import asyncio
import logging

from ..devices.base import Camera
from ..events import VisionEvent, VisionEventType
from .detector import PersonDetector
from .tracking import PanTiltTracker, TrackingTarget

LOGGER = logging.getLogger(__name__)


class CameraWorker:
    def __init__(
        self,
        camera: Camera,
        detector: PersonDetector,
        output_queue: asyncio.Queue[VisionEvent],
        frame_skip: int = 1,
        tracker: PanTiltTracker | None = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.output_queue = output_queue
        self.frame_skip = max(1, frame_skip)
        self.tracker = tracker

    async def run(self) -> None:
        async for frame in self.camera.frames():
            if frame.sequence % self.frame_skip != 0:
                continue
            detect_faces = getattr(self.detector, "detect_faces", None)
            if self.tracker is not None and callable(detect_faces):
                # ACL contexts are thread-affine. Keep model initialization,
                # inference, and release on this worker's event-loop thread.
                faces = detect_faces(frame.data)
                confidence = max((face.confidence for face in faces), default=0.0)
                event_type = (
                    VisionEventType.PERSON_PRESENT
                    if faces
                    else VisionEventType.PERSON_ABSENT
                )
                if faces:
                    height, width = frame.data.shape[:2]
                    decision = await self.tracker.update(
                        [
                            TrackingTarget(
                                x=face.x1,
                                y=face.y1,
                                width=face.width,
                                height=face.height,
                                confidence=face.confidence,
                            )
                            for face in faces
                        ],
                        frame_width=width,
                        frame_height=height,
                    )
                    LOGGER.debug("face tracking decision: %s", decision)
                else:
                    await self.tracker.target_lost()
                backend = "yolov5-face-om"
            else:
                result = await self.detector.detect(frame)
                confidence = result.confidence
                backend = result.backend
                event_type = (
                    VisionEventType.PERSON_PRESENT
                    if result.person_present
                    else VisionEventType.PERSON_ABSENT
                )
            event = VisionEvent(
                event_type=event_type,
                confidence=confidence,
                source=f"camera:{backend}",
            )
            await self.output_queue.put(event)
            LOGGER.debug("vision event queued: %s", event)
