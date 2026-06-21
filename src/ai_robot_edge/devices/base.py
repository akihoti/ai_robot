from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..events import ActionIntent, AudioFrame, CameraFrame


class Camera(ABC):
    @abstractmethod
    async def frames(self) -> AsyncIterator[CameraFrame]:
        """Yield camera frames until cancelled."""


class Microphone(ABC):
    @abstractmethod
    async def frames(self) -> AsyncIterator[AudioFrame]:
        """Yield microphone frames until cancelled."""


class Speaker(ABC):
    @abstractmethod
    async def play(
        self,
        audio: bytes,
        sample_rate: int,
        channels: int = 1,
        media_type: str = "audio/pcm",
    ) -> None:
        """Play one audio chunk."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop current audio playback as soon as possible."""


class ServoController(ABC):
    @abstractmethod
    async def execute(self, intent: ActionIntent) -> None:
        """Execute a high-level motion intent."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the current motion as soon as possible."""
