from __future__ import annotations

import asyncio
import logging
import math

import numpy as np

from ..events import ActionIntent, AudioFrame, CameraFrame
from .base import Camera, Microphone, ServoController, Speaker

LOGGER = logging.getLogger(__name__)


class SimulatedCamera(Camera):
    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self._sequence = 0

    async def frames(self):
        interval = 1 / self.fps
        while True:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            if self._sequence % 20 >= 5:
                x0 = self.width // 3
                x1 = self.width * 2 // 3
                y0 = self.height // 4
                y1 = self.height * 3 // 4
                frame[y0:y1, x0:x1, :] = 180
            yield CameraFrame(data=frame, sequence=self._sequence)
            self._sequence += 1
            await asyncio.sleep(interval)


class SimulatedMicrophone(Microphone):
    def __init__(self, sample_rate: int, frame_ms: int) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self._sequence = 0

    async def frames(self):
        samples_per_frame = int(self.sample_rate * self.frame_ms / 1000)
        interval = self.frame_ms / 1000
        while True:
            phase = self._sequence % 90
            if 10 <= phase < 50:
                data = _tone_pcm16(samples_per_frame, self.sample_rate, self._sequence)
            else:
                data = bytes(samples_per_frame * 2)
            yield AudioFrame(
                data=data,
                sample_rate=self.sample_rate,
                channels=1,
                sequence=self._sequence,
            )
            self._sequence += 1
            await asyncio.sleep(interval)


class SimulatedSpeaker(Speaker):
    async def play(
        self,
        audio: bytes,
        sample_rate: int,
        channels: int = 1,
        media_type: str = "audio/pcm",
    ) -> None:
        LOGGER.info(
            "simulated speaker received %s bytes at %s Hz/%s ch (%s)",
            len(audio),
            sample_rate,
            channels,
            media_type,
        )

    async def stop(self) -> None:
        LOGGER.info("simulated speaker stop requested")


class NoopServoController(ServoController):
    async def execute(self, intent: ActionIntent) -> None:
        LOGGER.info("noop servo accepted action intent: %s", intent)

    async def stop(self) -> None:
        LOGGER.info("noop servo stop requested")


def _tone_pcm16(samples: int, sample_rate: int, offset: int) -> bytes:
    amplitude = 0.08
    frequency = 440
    values = []
    start = offset * samples
    for index in range(samples):
        sample = amplitude * math.sin(2 * math.pi * frequency * (start + index) / sample_rate)
        values.append(int(sample * 32767))
    return np.array(values, dtype=np.int16).tobytes()
