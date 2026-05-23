from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .devices.base import Speaker


@dataclass(frozen=True)
class TtsChunk:
    audio: bytes
    sample_rate: int
    channels: int = 1


class PlaybackWorker:
    def __init__(
        self,
        queue: asyncio.Queue[TtsChunk],
        speaker: Speaker,
    ) -> None:
        self.queue = queue
        self.speaker = speaker

    async def run(self) -> None:
        while True:
            chunk = await self.queue.get()
            await self.speaker.play(chunk.audio, chunk.sample_rate, chunk.channels)
