from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import EdgeConfig
from ..events import CameraFrame

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectionResult:
    person_present: bool
    confidence: float
    backend: str


class PersonDetector(Protocol):
    async def detect(self, frame: CameraFrame) -> DetectionResult:
        """Detect whether a person is visible in the frame."""


class SimulatedPersonDetector:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    async def detect(self, frame: CameraFrame) -> DetectionResult:
        brightness = float(np.mean(frame.data)) / 255.0
        confidence = min(1.0, brightness * 4)
        return DetectionResult(
            person_present=confidence >= self.threshold,
            confidence=confidence,
            backend="simulated",
        )


class CpuPersonDetector:
    """CPU detector placeholder.

    The adapter keeps the public shape stable while the final model choice is
    still open. For now it uses the same lightweight heuristic as simulated
    mode so the pipeline is executable without OpenCV model files.
    """

    def __init__(self, threshold: float) -> None:
        self._fallback = SimulatedPersonDetector(threshold)

    async def detect(self, frame: CameraFrame) -> DetectionResult:
        result = await self._fallback.detect(frame)
        return DetectionResult(result.person_present, result.confidence, "cpu")


class RknnPersonDetector:
    """RK3588/NPU detector placeholder.

    Replace this class with an RKNN model runner once the model path and runtime
    package are chosen.
    """

    def __init__(self, threshold: float) -> None:
        self._fallback = SimulatedPersonDetector(threshold)
        LOGGER.warning("RKNN detector not configured; using heuristic fallback")

    async def detect(self, frame: CameraFrame) -> DetectionResult:
        result = await self._fallback.detect(frame)
        return DetectionResult(result.person_present, result.confidence, "rknn-fallback")


def build_person_detector(config: EdgeConfig) -> PersonDetector:
    detector = config.vision.detector.lower()
    if detector == "rknn" or (detector == "auto" and config.runtime.prefer_npu):
        return RknnPersonDetector(config.vision.person_threshold)
    if detector == "cpu" or detector == "auto":
        return CpuPersonDetector(config.vision.person_threshold)
    return SimulatedPersonDetector(config.vision.person_threshold)
