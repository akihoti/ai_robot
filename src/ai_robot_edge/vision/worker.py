from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..admin.runtime_state import runtime_state
from ..devices.base import Camera
from ..events import VisionEvent, VisionEventType
from ..session import EdgeSessionState
from ..interaction import tracking_policy_for_state
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
        session_state_provider: Callable[[], EdgeSessionState] | None = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.output_queue = output_queue
        self.frame_skip = max(1, frame_skip)
        self.tracker = tracker
        self.session_state_provider = session_state_provider
        self._tracker_degraded = False
        self._last_tracking_mode: str | None = None

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
                    tracking_targets = [
                        TrackingTarget(
                            x=face.x1,
                            y=face.y1,
                            width=face.width,
                            height=face.height,
                            confidence=face.confidence,
                        )
                        for face in faces
                    ]
                    if self._should_track(frame.sequence):
                        try:
                            decision = await self.tracker.update(
                                tracking_targets,
                                frame_width=width,
                                frame_height=height,
                            )
                            LOGGER.debug("face tracking decision: %s", decision)
                        except Exception:
                            self._degrade_tracker()
                else:
                    try:
                        await self.tracker.target_lost()
                    except Exception:
                        self._degrade_tracker()
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
            runtime_state.record_vision_event(
                event_type=event.event_type.value,
                confidence=event.confidence,
                source=event.source,
            )
            await self.output_queue.put(event)
            LOGGER.debug("vision event queued: %s", event)

    def _degrade_tracker(self) -> None:
        if not self._tracker_degraded:
            LOGGER.exception(
                "pan-tilt tracker failed; disabling tracking and continuing with vision events"
            )
        self._tracker_degraded = True
        runtime_state.set_component_state("tracker_degraded", True)
        self.tracker = None

    def _should_track(self, sequence: int) -> bool:
        state = (
            self.session_state_provider()
            if self.session_state_provider is not None
            else EdgeSessionState.TRACKING
        )
        policy = tracking_policy_for_state(state)
        self._log_tracking_mode(policy.mode)
        if not policy.should_track:
            return False
        return sequence % max(1, policy.cadence_divisor) == 0

    def _log_tracking_mode(self, mode: str) -> None:
        if mode == self._last_tracking_mode:
            return
        self._last_tracking_mode = mode
        LOGGER.info("tracking mode switched to %s", mode)
