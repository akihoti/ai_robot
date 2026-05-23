from __future__ import annotations

from collections import deque
from array import array

from ..config import VadConfig
from ..events import AudioFrame, Utterance


def frame_energy(frame: AudioFrame) -> float:
    if not frame.data:
        return 0.0
    samples = array("h")
    samples.frombytes(frame.data)
    if not samples:
        return 0.0
    square_sum = sum((sample / 32768.0) ** 2 for sample in samples)
    return (square_sum / len(samples)) ** 0.5


class EnergyVadSegmenter:
    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._pre_roll: deque[AudioFrame] = deque()
        self._frames: list[AudioFrame] = []
        self._speech_started = False
        self._silence_ms = 0
        self._duration_ms = 0

    def accept(self, frame: AudioFrame) -> Utterance | None:
        self._append_pre_roll(frame)
        energy = frame_energy(frame)
        voiced = energy >= self.config.energy_threshold

        if not self._speech_started:
            if voiced:
                self._speech_started = True
                self._frames.extend(self._pre_roll)
                self._pre_roll.clear()
            else:
                return None

        self._frames.append(frame)
        self._duration_ms += frame.duration_ms
        self._silence_ms = 0 if voiced else self._silence_ms + frame.duration_ms

        if self._duration_ms >= self.config.max_utterance_ms:
            return self._finish("max_duration")
        if self._silence_ms >= self.config.silence_ms:
            return self._finish("vad_silence")
        return None

    def reset(self) -> None:
        self._pre_roll.clear()
        self._frames.clear()
        self._speech_started = False
        self._silence_ms = 0
        self._duration_ms = 0

    def _append_pre_roll(self, frame: AudioFrame) -> None:
        self._pre_roll.append(frame)
        max_frames = max(1, self.config.pre_roll_ms // max(1, frame.duration_ms))
        while len(self._pre_roll) > max_frames:
            self._pre_roll.popleft()

    def _finish(self, reason: str) -> Utterance:
        utterance = Utterance(frames=list(self._frames), reason=reason)
        self.reset()
        return utterance
