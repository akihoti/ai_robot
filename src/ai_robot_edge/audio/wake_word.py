from __future__ import annotations

from typing import Protocol

from ..config import WakeWordConfig
from ..events import AudioFrame
from .vad import frame_energy


class WakeWordDetector(Protocol):
    async def detect(self, frame: AudioFrame) -> bool:
        """Return true when the configured wake word is detected."""


class SimulatedWakeWordDetector:
    def __init__(self, keyword_id: str, trigger_after_voiced_frames: int = 6) -> None:
        self.keyword_id = keyword_id
        self.trigger_after_voiced_frames = trigger_after_voiced_frames
        self._voiced_frames = 0

    async def detect(self, frame: AudioFrame) -> bool:
        if frame_energy(frame) > 0.01:
            self._voiced_frames += 1
        else:
            self._voiced_frames = 0
        if self._voiced_frames >= self.trigger_after_voiced_frames:
            self._voiced_frames = 0
            return True
        return False


def build_wake_word_detector(config: WakeWordConfig) -> WakeWordDetector:
    if config.engine != "simulated":
        raise NotImplementedError(
            f"wake-word engine {config.engine!r} is not implemented yet"
        )
    return SimulatedWakeWordDetector(config.keyword_id)
