from __future__ import annotations

import asyncio
import logging
import wave
from io import BytesIO
from pathlib import Path

from .admin.runtime_state import runtime_state
from .audio.vad import build_vad_segmenter
from .audio.wake_word import build_wake_word_detector
from .audio.worker import AudioWorker
from .config import EdgeConfig
from .actions import ActionDispatcher
from .devices.factory import build_camera, build_microphone, build_servo_controller, build_speaker
from .events import (
    ActionIntent,
    ConversationEvent,
    ConversationEventType,
    Utterance,
    VisionEvent,
    VisionEventType,
)
from .interaction import (
    InteractionCoordinator,
    microphone_should_suppress_during_speaking,
)
from .playback import PlaybackWorker, TtsChunk
from .session import SessionController
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
        runtime_state.reset(
            device_id=self.config.device_id,
            runtime_mode=self.config.runtime.mode,
        )
        vision_queue: asyncio.Queue[VisionEvent] = asyncio.Queue(maxsize=32)
        action_queue: asyncio.Queue[ActionIntent] = asyncio.Queue(maxsize=32)
        utterance_queue: asyncio.Queue[Utterance] = asyncio.Queue(maxsize=8)
        conversation_queue: asyncio.Queue[ConversationEvent] = asyncio.Queue(maxsize=8)
        tts_queue: asyncio.Queue[TtsChunk] = asyncio.Queue(maxsize=32)
        playback_idle = asyncio.Event()
        playback_idle.set()
        playback_active = asyncio.Event()
        playback_worker: PlaybackWorker | None = None
        session_controller = SessionController(
            welcome_once_per_session=self.config.voice.welcome_once_per_session,
            on_update=runtime_state.record_session,
        )
        runtime_state.register_queue("vision", vision_queue.qsize)
        runtime_state.register_queue("actions", action_queue.qsize)
        runtime_state.register_queue("utterances", utterance_queue.qsize)
        runtime_state.register_queue("conversation", conversation_queue.qsize)
        runtime_state.register_queue("tts", tts_queue.qsize)
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
                session_state_provider=lambda: session_controller.state,
            )
            tasks.append(asyncio.create_task(worker.run(), name="camera-worker"))
        audio_worker: AudioWorker | None = None
        wake_word_detector = (
            build_wake_word_detector(self.config.wake_word)
            if self.config.wake_word.enabled
            else None
        )

        async def handle_wake_word_detected() -> None:
            if not await session_controller.try_start_welcome():
                return
            if self.config.voice.local_welcome_enabled:
                await queue_local_welcome(
                    VisionEvent(
                        event_type=VisionEventType.WELCOME_TRIGGERED,
                        confidence=1.0,
                        source="wake_word",
                    )
                )
                return
            LOGGER.info("wake-word-triggered welcome queued")
            await conversation_queue.put(
                ConversationEvent(event_type=ConversationEventType.WELCOME)
            )

        if self.config.microphone.enabled:
            audio_worker = AudioWorker(
                microphone=build_microphone(self.config),
                vad=build_vad_segmenter(self.config.vad),
                utterance_queue=utterance_queue,
                listen_timeout_ms=self.config.voice.visual_listen_timeout_ms,
                suppress_event=(
                    playback_active
                    if microphone_should_suppress_during_speaking(
                        suppress_mic_while_speaking=self.config.voice.suppress_mic_while_speaking,
                        speech_interrupt_enabled=self.config.voice.speech_interrupt_enabled,
                    )
                    else None
                ),
                wake_word_detector=wake_word_detector,
                on_wake_word_detected=handle_wake_word_detected,
                on_utterance_ready=session_controller.note_utterance_ready,
                on_listening_expired=(
                    lambda: session_controller.recover_to_tracking("listen_timeout")
                ),
            )
            tasks.append(asyncio.create_task(audio_worker.run(), name="audio-worker"))

        if self.config.speaker.enabled:
            playback_worker = PlaybackWorker(
                queue=tts_queue,
                speaker=build_speaker(self.config),
                idle_event=playback_idle,
                active_event=playback_active,
                on_playback_started=session_controller.note_playback_started,
            )
            tasks.append(asyncio.create_task(playback_worker.run(), name="playback"))

        async def queue_local_welcome(_event: VisionEvent) -> None:
            if playback_worker is None:
                LOGGER.warning("local welcome skipped because speaker is disabled")
                await session_controller.note_welcome_playback_finished()
                if audio_worker is not None:
                    audio_worker.arm_presence_listening()
                return
            try:
                chunk = _load_local_welcome_chunk(
                    self.config.voice.welcome_audio_path,
                    fallback_sample_rate=self.config.speaker.sample_rate,
                )
            except OSError:
                LOGGER.exception(
                    "local welcome audio unavailable: %s",
                    self.config.voice.welcome_audio_path,
                )
                await session_controller.note_welcome_playback_finished()
                if audio_worker is not None:
                    audio_worker.arm_presence_listening()
                return

            await tts_queue.put(chunk)
            await _wait_for_playback_cycle(playback_active, playback_idle)
            await session_controller.note_welcome_playback_finished()
            if audio_worker is not None:
                audio_worker.arm_presence_listening()

        coordinator = InteractionCoordinator(
            vision_config=self.config.vision,
            vision_queue=vision_queue,
            action_queue=action_queue,
            conversation_queue=conversation_queue,
            session_controller=session_controller,
            disarm_listening=audio_worker.disarm if audio_worker is not None else lambda: None,
            idle_return_to_center_seconds=(
                self.config.voice.face_absence_stop_listening_seconds
                if self.config.voice.presence_listening_enabled
                else self.config.tracking.idle_return_to_center_seconds
            ),
            arm_presence_listening=(
                audio_worker.arm_presence_listening
                if audio_worker is not None
                else lambda: None
            ),
            queue_local_welcome=(
                queue_local_welcome
                if self.config.voice.local_welcome_enabled
                else None
            ),
            local_welcome_enabled=self.config.voice.local_welcome_enabled,
        )
        dispatcher = ActionDispatcher(
            action_queue=action_queue,
            servo_controller=servo_controller,
        )
        tasks.append(asyncio.create_task(coordinator.run(), name="interaction"))
        tasks.append(asyncio.create_task(dispatcher.run(), name="actions"))
        conversation_client = ConversationClient(
            config=self.config,
            on_tts=lambda audio, sample_rate, channels, media_type: tts_queue.put(
                TtsChunk(
                    audio=audio,
                    sample_rate=sample_rate,
                    channels=channels,
                    media_type=media_type,
                )
            ),
            on_action=action_queue.put,
        )
        arm_after_welcome = (
            audio_worker.arm_presence_listening
            if audio_worker is not None
            and self.config.voice.presence_listening_enabled
            else (
                (
                    lambda: audio_worker.arm_listening_window(
                        self.config.voice.visual_listen_timeout_ms
                    )
                )
                if audio_worker is not None
                else lambda: None
            )
        )
        arm_after_response = (
            audio_worker.arm_presence_listening
            if audio_worker is not None
            and self.config.voice.presence_listening_enabled
            else (
                (
                    lambda: audio_worker.arm_listening_window(
                        self.config.voice.followup_listen_timeout_ms
                    )
                )
                if audio_worker is not None
                else lambda: None
            )
        )
        conversation_worker = ConversationWorker(
            utterance_queue=utterance_queue,
            conversation_queue=conversation_queue,
            client=conversation_client,
            playback_idle=playback_idle,
            playback_active=playback_active if playback_worker is not None else None,
            auto_listen_after_welcome=self.config.voice.auto_listen_after_welcome,
            arm_welcome_listening_window=arm_after_welcome,
            arm_followup_listening_window=arm_after_response,
            session_controller=session_controller,
            interrupt_playback=(
                playback_worker.interrupt if playback_worker is not None else None
            ),
            speech_interrupt_enabled=self.config.voice.speech_interrupt_enabled,
        )
        tasks.append(asyncio.create_task(conversation_worker.run(), name="conversation"))
        if self.config.admin.enabled:
            tasks.append(
                asyncio.create_task(
                    ManagementClient(self.config).run(),
                    name="management-client",
                )
            )
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


async def _wait_for_playback_cycle(
    playback_active: asyncio.Event,
    playback_idle: asyncio.Event,
) -> None:
    try:
        await asyncio.wait_for(playback_active.wait(), timeout=1)
    except asyncio.TimeoutError:
        await playback_idle.wait()
        return
    await playback_idle.wait()


def _load_local_welcome_chunk(
    path: str,
    *,
    fallback_sample_rate: int,
) -> TtsChunk:
    audio_path = Path(path)
    data = audio_path.read_bytes()
    sample_rate = fallback_sample_rate
    channels = 1
    with wave.open(BytesIO(data), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
    return TtsChunk(
        audio=data,
        sample_rate=sample_rate,
        channels=channels,
        media_type="audio/wav",
    )
