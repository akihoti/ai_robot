from __future__ import annotations

import asyncio
import logging

from .audio import AudioWorker, EnergyVadSegmenter, build_wake_word_detector
from .config import EdgeConfig
from .actions import ActionDispatcher
from .devices.factory import build_camera, build_microphone, build_servo_controller
from .events import ActionIntent, Utterance, VisionEvent
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
        utterance_queue: asyncio.Queue[Utterance] = asyncio.Queue(maxsize=8)
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
        if self.config.microphone.enabled and self.config.wake_word.enabled:
            audio_worker = AudioWorker(
                microphone=build_microphone(self.config),
                wake_word=build_wake_word_detector(self.config.wake_word),
                vad=EnergyVadSegmenter(self.config.vad),
                utterance_queue=utterance_queue,
            )
            tasks.append(asyncio.create_task(audio_worker.run(), name="audio-worker"))

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
        tasks.append(asyncio.create_task(self._log_utterances(utterance_queue)))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()

    async def _log_utterances(self, queue: asyncio.Queue[Utterance]) -> None:
        while True:
            utterance = await queue.get()
            LOGGER.info(
                "utterance queued for server: request_id=%s bytes=%s",
                utterance.request_id,
                len(utterance.audio_bytes),
            )
