from __future__ import annotations

import asyncio
import logging
from time import monotonic

from ..devices.base import Microphone
from ..events import Utterance
from .vad import EnergyVadSegmenter

LOGGER = logging.getLogger(__name__)


class AudioWorker:
    def __init__(
        self,
        microphone: Microphone,
        vad: EnergyVadSegmenter,
        utterance_queue: asyncio.Queue[Utterance],
        listen_timeout_ms: int,
    ) -> None:
        self.microphone = microphone
        self.vad = vad
        self.utterance_queue = utterance_queue
        self.listen_timeout_ms = listen_timeout_ms
        self._listening_deadline: float | None = None

    def arm_listening_window(self) -> None:
        self.vad.reset()
        self._listening_deadline = monotonic() + self.listen_timeout_ms / 1000
        LOGGER.info(
            "listening window armed for %.1fs",
            self.listen_timeout_ms / 1000,
        )

    def disarm(self) -> None:
        self.vad.reset()
        self._listening_deadline = None

    def _is_listening(self) -> bool:
        if self._listening_deadline is None:
            return False
        if monotonic() >= self._listening_deadline:
            LOGGER.info("listening window expired")
            self.disarm()
            return False
        return True

    async def run(self) -> None:
        async for frame in self.microphone.frames():
            if not self._is_listening():
                continue

            utterance = self.vad.accept(frame)
            if utterance is not None:
                LOGGER.info(
                    "utterance ready: request_id=%s duration=%sms reason=%s",
                    utterance.request_id,
                    utterance.duration_ms,
                    utterance.reason,
                )
                await self.utterance_queue.put(utterance)
                self.disarm()
