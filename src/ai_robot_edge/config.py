from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ServerConfig:
    websocket_url: str
    bearer_token: str
    connect_timeout_seconds: float
    heartbeat_seconds: float
    reconnect_initial_delay_seconds: float
    reconnect_max_delay_seconds: float


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    log_level: str
    prefer_npu: bool


@dataclass(frozen=True)
class CameraConfig:
    enabled: bool
    source: int | str
    width: int
    height: int
    fps: int
    frame_skip: int


@dataclass(frozen=True)
class VisionConfig:
    detector: str
    person_threshold: float
    stable_frames: int
    welcome_cooldown_seconds: float


@dataclass(frozen=True)
class MicrophoneConfig:
    enabled: bool
    sample_rate: int
    channels: int
    frame_ms: int
    device: str | None


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool
    engine: str
    keyword_id: str
    model_path: str


@dataclass(frozen=True)
class VadConfig:
    energy_threshold: float
    silence_ms: int
    max_utterance_ms: int
    pre_roll_ms: int


@dataclass(frozen=True)
class SpeakerConfig:
    enabled: bool
    device: str | None
    sample_rate: int


@dataclass(frozen=True)
class ServoConfig:
    enabled: bool
    controller: str


@dataclass(frozen=True)
class EdgeConfig:
    device_id: str
    server: ServerConfig
    runtime: RuntimeConfig
    camera: CameraConfig
    vision: VisionConfig
    microphone: MicrophoneConfig
    wake_word: WakeWordConfig
    vad: VadConfig
    speaker: SpeakerConfig
    servo: ServoConfig


def load_config(path: str | Path) -> EdgeConfig:
    data = _load_yaml(path)
    config = _parse_config(data)
    _validate_config(config)
    return config


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _parse_config(data: dict[str, Any]) -> EdgeConfig:
    server = data.get("server", {})
    runtime = data.get("runtime", {})
    camera = data.get("camera", {})
    vision = data.get("vision", {})
    microphone = data.get("microphone", {})
    wake_word = data.get("wake_word", {})
    vad = data.get("vad", {})
    speaker = data.get("speaker", {})
    servo = data.get("servo", {})
    reconnect = server.get("reconnect", {})

    return EdgeConfig(
        device_id=str(data.get("device_id", "")),
        server=ServerConfig(
            websocket_url=str(server.get("websocket_url", "")),
            bearer_token=str(server.get("bearer_token", "")),
            connect_timeout_seconds=float(server.get("connect_timeout_seconds", 10)),
            heartbeat_seconds=float(server.get("heartbeat_seconds", 20)),
            reconnect_initial_delay_seconds=float(
                reconnect.get("initial_delay_seconds", 1)
            ),
            reconnect_max_delay_seconds=float(reconnect.get("max_delay_seconds", 30)),
        ),
        runtime=RuntimeConfig(
            mode=str(runtime.get("mode", "simulated")),
            log_level=str(runtime.get("log_level", "INFO")),
            prefer_npu=bool(runtime.get("prefer_npu", True)),
        ),
        camera=CameraConfig(
            enabled=bool(camera.get("enabled", True)),
            source=camera.get("source", 0),
            width=int(camera.get("width", 640)),
            height=int(camera.get("height", 480)),
            fps=int(camera.get("fps", 10)),
            frame_skip=int(camera.get("frame_skip", 1)),
        ),
        vision=VisionConfig(
            detector=str(vision.get("detector", "simulated")),
            person_threshold=float(vision.get("person_threshold", 0.55)),
            stable_frames=int(vision.get("stable_frames", 3)),
            welcome_cooldown_seconds=float(
                vision.get("welcome_cooldown_seconds", 30)
            ),
        ),
        microphone=MicrophoneConfig(
            enabled=bool(microphone.get("enabled", True)),
            sample_rate=int(microphone.get("sample_rate", 16000)),
            channels=int(microphone.get("channels", 1)),
            frame_ms=int(microphone.get("frame_ms", 30)),
            device=microphone.get("device"),
        ),
        wake_word=WakeWordConfig(
            enabled=bool(wake_word.get("enabled", True)),
            engine=str(wake_word.get("engine", "simulated")),
            keyword_id=str(wake_word.get("keyword_id", "")),
            model_path=str(wake_word.get("model_path", "")),
        ),
        vad=VadConfig(
            energy_threshold=float(vad.get("energy_threshold", 0.015)),
            silence_ms=int(vad.get("silence_ms", 800)),
            max_utterance_ms=int(vad.get("max_utterance_ms", 10000)),
            pre_roll_ms=int(vad.get("pre_roll_ms", 300)),
        ),
        speaker=SpeakerConfig(
            enabled=bool(speaker.get("enabled", True)),
            device=speaker.get("device"),
            sample_rate=int(speaker.get("sample_rate", 16000)),
        ),
        servo=ServoConfig(
            enabled=bool(servo.get("enabled", False)),
            controller=str(servo.get("controller", "noop")),
        ),
    )


def _validate_config(config: EdgeConfig) -> None:
    if not config.device_id:
        raise ValueError("device_id is required")
    if "{device_id}" not in config.server.websocket_url:
        raise ValueError("server.websocket_url must contain {device_id}")
    if not config.server.bearer_token:
        raise ValueError("server.bearer_token is required")
    if config.wake_word.enabled and not config.wake_word.keyword_id:
        raise ValueError("wake_word.keyword_id is required when wake word is enabled")
    if config.microphone.channels != 1:
        raise ValueError("only mono microphone capture is supported in v1")
