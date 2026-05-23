from __future__ import annotations

from ..config import EdgeConfig
from .base import Camera, Microphone, ServoController, Speaker
from .simulated import (
    NoopServoController,
    SimulatedCamera,
    SimulatedMicrophone,
    SimulatedSpeaker,
)


def build_camera(config: EdgeConfig) -> Camera:
    return SimulatedCamera(
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
    )


def build_microphone(config: EdgeConfig) -> Microphone:
    return SimulatedMicrophone(
        sample_rate=config.microphone.sample_rate,
        frame_ms=config.microphone.frame_ms,
    )


def build_speaker(config: EdgeConfig) -> Speaker:
    return SimulatedSpeaker()


def build_servo_controller(config: EdgeConfig) -> ServoController:
    return NoopServoController()
