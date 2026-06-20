from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .devices.base import Speaker


@dataclass(frozen=True)
class TtsChunk:
    audio: bytes
    sample_rate: int
    channels: int = 1
    media_type: str = "audio/pcm"


class PlaybackWorker:
    def __init__(
        self,
        queue: asyncio.Queue[TtsChunk],
        speaker: Speaker,
        idle_event: asyncio.Event | None = None,
    ) -> None:
        self.queue = queue
        self.speaker = speaker
        self.idle_event = idle_event
        if self.idle_event is not None:
            self.idle_event.set()

    async def run(self) -> None:
        while True:
            chunk = await self.queue.get()
            if self.idle_event is not None:
                self.idle_event.clear()
            await self.speaker.play(
                chunk.audio,
                chunk.sample_rate,
                chunk.channels,
                chunk.media_type,
            )
            if self.idle_event is not None and self.queue.empty():
                self.idle_event.set()
