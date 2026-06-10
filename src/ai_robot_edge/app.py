from __future__ import annotations

import asyncio
import logging

from .audio.vad import EnergyVadSegmenter
from .audio.wake_word import build_wake_word_detector
from .audio.worker import AudioWorker
from .config import EdgeConfig
from .actions import ActionDispatcher
from .devices.factory import (
    build_camera,
    build_microphone,
    build_servo_controller,
    build_speaker,
)
from .events import ActionIntent, Utterance, VisionEvent
from .interaction import InteractionCoordinator
from .playback import PlaybackWorker, TtsChunk
from .server import ConversationClient, ConversationWorker, ManagementClient
from .vision import CameraWorker, build_person_detector
from .devices.gimbal import PanTiltGimbal
from .vision import PanTiltTracker

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
        tts_queue: asyncio.Queue[TtsChunk] = asyncio.Queue(maxsize=32)
        tasks: list[asyncio.Task] = []
        servo_controller = build_servo_controller(self.config)
        detector = None
        if self.config.camera.enabled:
            camera = build_camera(self.config)
            detector = build_person_detector(self.config)
            tracker = (
                PanTiltTracker(servo_controller, self.config.tracking)
                if self.config.tracking.enabled
                and isinstance(servo_controller, PanTiltGimbal)
                else None
            )
            worker = CameraWorker(
                camera=camera,
                detector=detector,
                output_queue=vision_queue,
                frame_skip=self.config.camera.frame_skip,
                tracker=tracker,
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
            servo_controller=servo_controller,
        )
        tasks.append(asyncio.create_task(coordinator.run(), name="interaction"))
        tasks.append(asyncio.create_task(dispatcher.run(), name="actions"))
        conversation_client = ConversationClient(
            config=self.config,
            on_tts=lambda audio, sample_rate, channels: tts_queue.put(
                TtsChunk(audio=audio, sample_rate=sample_rate, channels=channels)
            ),
            on_action=action_queue.put,
        )
        conversation_worker = ConversationWorker(
            utterance_queue=utterance_queue,
            client=conversation_client,
        )
        tasks.append(asyncio.create_task(conversation_worker.run(), name="conversation"))
        if self.config.admin.enabled:
            tasks.append(
                asyncio.create_task(
                    ManagementClient(self.config).run(),
                    name="management-client",
                )
            )
        if self.config.speaker.enabled:
            playback_worker = PlaybackWorker(
                queue=tts_queue,
                speaker=build_speaker(self.config),
            )
            tasks.append(asyncio.create_task(playback_worker.run(), name="playback"))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            close_detector = getattr(detector, "close", None)
            if callable(close_detector):
                close_detector()
            if isinstance(servo_controller, PanTiltGimbal):
                await servo_controller.close()
