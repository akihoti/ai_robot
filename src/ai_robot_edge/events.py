from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

import numpy as np


def now_ms() -> int:
    return int(time() * 1000)


class VisionEventType(str, Enum):
    PERSON_PRESENT = "person_present"
    PERSON_ABSENT = "person_absent"
    WELCOME_TRIGGERED = "welcome_triggered"


class ActionName(str, Enum):
    LOOK_AT_USER = "look_at_user"
    NOD = "nod"
    IDLE = "idle"
    WELCOME_MOTION = "welcome_motion"


@dataclass(frozen=True)
class CameraFrame:
    data: np.ndarray
    timestamp_ms: int = field(default_factory=now_ms)
    sequence: int = 0


@dataclass(frozen=True)
class AudioFrame:
    data: bytes
    sample_rate: int
    channels: int = 1
    timestamp_ms: int = field(default_factory=now_ms)
    sequence: int = 0

    @property
    def duration_ms(self) -> int:
        bytes_per_sample = 2
        sample_count = len(self.data) // bytes_per_sample // self.channels
        return int(sample_count / self.sample_rate * 1000)


@dataclass(frozen=True)
class Utterance:
    frames: list[AudioFrame]
    request_id: str = field(default_factory=lambda: str(uuid4()))
    reason: str = "vad_silence"

    @property
    def audio_bytes(self) -> bytes:
        return b"".join(frame.data for frame in self.frames)

    @property
    def duration_ms(self) -> int:
        return sum(frame.duration_ms for frame in self.frames)


@dataclass(frozen=True)
class VisionEvent:
    event_type: VisionEventType
    confidence: float
    timestamp_ms: int = field(default_factory=now_ms)
    source: str = "camera"
    cooldown_active: bool = False


@dataclass(frozen=True)
class ActionIntent:
    name: ActionName
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
