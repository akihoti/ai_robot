from __future__ import annotations

import asyncio
import logging
from time import monotonic
from collections.abc import Awaitable, Callable

from ..admin.runtime_state import runtime_state
from ..devices.base import Microphone
from ..events import Utterance
from .wake_word import WakeWordDetector
from .vad import EnergyVadSegmenter

LOGGER = logging.getLogger(__name__)


class AudioWorker:
    def __init__(
        self,
        microphone: Microphone,
        vad: EnergyVadSegmenter,
        utterance_queue: asyncio.Queue[Utterance],
        listen_timeout_ms: int,
        suppress_event: asyncio.Event | None = None,
        wake_word_detector: WakeWordDetector | None = None,
        on_wake_word_detected: Callable[[], Awaitable[None]] | None = None,
        on_utterance_ready: Callable[[], Awaitable[None]] | None = None,
        on_listening_expired: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.microphone = microphone
        self.vad = vad
        self.utterance_queue = utterance_queue
        self.listen_timeout_ms = listen_timeout_ms
        self.suppress_event = suppress_event
        self.wake_word_detector = wake_word_detector
        self.on_wake_word_detected = on_wake_word_detected
        self.on_utterance_ready = on_utterance_ready
        self.on_listening_expired = on_listening_expired
        self._listening_deadline: float | None = None

    def arm_listening_window(self, timeout_ms: int | None = None) -> None:
        self.vad.reset()
        effective_timeout_ms = timeout_ms or self.listen_timeout_ms
        self._listening_deadline = monotonic() + effective_timeout_ms / 1000
        runtime_state.record_listening(armed=True, timeout_ms=effective_timeout_ms)
        LOGGER.info(
            "listening window armed for %.1fs",
            effective_timeout_ms / 1000,
        )

    def disarm(self) -> None:
        self.vad.reset()
        self._listening_deadline = None
        runtime_state.record_listening(armed=False)

    async def _is_listening(self) -> bool:
        if self._listening_deadline is None:
            return False
        if monotonic() >= self._listening_deadline:
            LOGGER.info("listening window expired")
            self.disarm()
            if self.on_listening_expired is not None:
                await self.on_listening_expired()
            return False
        return True

    async def run(self) -> None:
        async for frame in self.microphone.frames():
            if self.suppress_event is not None and self.suppress_event.is_set():
                if self._listening_deadline is not None:
                    LOGGER.debug("listening suppressed while speaking")
                    self.disarm()
                continue
            if not await self._is_listening():
                if self.wake_word_detector is not None:
                    detected = await self.wake_word_detector.detect(frame)
                    if detected:
                        LOGGER.info("wake word detected")
                        runtime_state.record_wake_word()
                        if self.on_wake_word_detected is not None:
                            await self.on_wake_word_detected()
                continue

            utterance = self.vad.accept(frame)
            if utterance is not None:
                runtime_state.record_utterance(
                    request_id=utterance.request_id,
                    duration_ms=utterance.duration_ms,
                    reason=utterance.reason,
                )
                LOGGER.info(
                    "utterance ready: request_id=%s duration=%sms reason=%s",
                    utterance.request_id,
                    utterance.duration_ms,
                    utterance.reason,
                )
                if self.on_utterance_ready is not None:
                    await self.on_utterance_ready()
                await self.utterance_queue.put(utterance)
                self.disarm()
