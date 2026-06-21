from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from collections.abc import Awaitable, Callable

from .admin.runtime_state import runtime_state
from .devices.base import Speaker

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TtsChunk:
    audio: bytes
    sample_rate: int
    channels: int = 1
    media_type: str = "audio/pcm"


class PlaybackWorker:
    BUFFER_WINDOW_SECONDS = 0.12

    def __init__(
        self,
        queue: asyncio.Queue[TtsChunk],
        speaker: Speaker,
        idle_event: asyncio.Event | None = None,
        active_event: asyncio.Event | None = None,
        on_playback_started: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.queue = queue
        self.speaker = speaker
        self.idle_event = idle_event
        self.active_event = active_event
        self.on_playback_started = on_playback_started
        self._interrupt_requested = False
        if self.idle_event is not None:
            self.idle_event.set()
        if self.active_event is not None:
            self.active_event.clear()

    async def run(self) -> None:
        while True:
            chunk = await self.queue.get()
            chunk, merged_count = await self._coalesce_chunks(chunk)
            was_idle = self.idle_event is None or self.idle_event.is_set()
            if self.idle_event is not None:
                self.idle_event.clear()
            if self.active_event is not None:
                self.active_event.set()
            runtime_state.set_playback_active(True)
            if was_idle and self.on_playback_started is not None:
                await self.on_playback_started()
            if self._interrupt_requested:
                self._interrupt_requested = False
                continue
            try:
                runtime_state.record_playback_chunk(
                    bytes_count=len(chunk.audio),
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    media_type=chunk.media_type,
                    merged_chunks=merged_count,
                )
                await self.speaker.play(
                    chunk.audio,
                    chunk.sample_rate,
                    chunk.channels,
                    chunk.media_type,
                )
            except Exception:
                LOGGER.exception("speaker playback failed; dropping current TTS chunk")
            finally:
                if self.queue.empty():
                    if self.idle_event is not None:
                        self.idle_event.set()
                    if self.active_event is not None:
                        self.active_event.clear()
                    runtime_state.set_playback_active(False)

    async def _coalesce_chunks(self, first: TtsChunk) -> tuple[TtsChunk, int]:
        if "wav" in first.media_type.lower():
            return first, 1

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.BUFFER_WINDOW_SECONDS
        audio_parts = [first.audio]
        merged_count = 1
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                candidate = await asyncio.wait_for(self.queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if not self._is_compatible(first, candidate):
                self.queue.put_nowait(candidate)
                break
            audio_parts.append(candidate.audio)
            merged_count += 1
        if merged_count > 1:
            LOGGER.info("coalesced %s TTS chunks into one playback burst", merged_count)
        return (
            TtsChunk(
                audio=b"".join(audio_parts),
                sample_rate=first.sample_rate,
                channels=first.channels,
                media_type=first.media_type,
            ),
            merged_count,
        )

    def _is_compatible(self, left: TtsChunk, right: TtsChunk) -> bool:
        return (
            "wav" not in right.media_type.lower()
            and left.sample_rate == right.sample_rate
            and left.channels == right.channels
            and left.media_type == right.media_type
        )

    async def interrupt(self) -> None:
        self._interrupt_requested = True
        dropped = 0
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                dropped += 1
        LOGGER.info("interrupting playback; dropped %s queued chunks", dropped)
        runtime_state.record_playback_interrupt()
        await self.speaker.stop()
