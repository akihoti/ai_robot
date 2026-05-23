from __future__ import annotations

import asyncio
import logging

from ..devices.base import Microphone
from ..events import Utterance
from .vad import EnergyVadSegmenter
from .wake_word import WakeWordDetector

LOGGER = logging.getLogger(__name__)


class AudioWorker:
    def __init__(
        self,
        microphone: Microphone,
        wake_word: WakeWordDetector,
        vad: EnergyVadSegmenter,
        utterance_queue: asyncio.Queue[Utterance],
    ) -> None:
        self.microphone = microphone
        self.wake_word = wake_word
        self.vad = vad
        self.utterance_queue = utterance_queue
        self._listening = False

    async def run(self) -> None:
        async for frame in self.microphone.frames():
            if not self._listening:
                if await self.wake_word.detect(frame):
                    LOGGER.info("wake word detected; starting VAD")
                    self._listening = True
                    self.vad.reset()
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
                self._listening = False
