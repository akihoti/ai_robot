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
    face_model_path: str = "pretrain/yolov5s-face.om"
    face_input_size: int = 640
    face_iou_threshold: float = 0.45
    face_device_id: int = 0


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
class VoiceConfig:
    visual_listen_timeout_ms: int
    followup_listen_timeout_ms: int
    welcome_text: str
    auto_listen_after_welcome: bool
    speech_interrupt_enabled: bool
    suppress_mic_while_speaking: bool
    welcome_once_per_session: bool


@dataclass(frozen=True)
class SpeakerConfig:
    enabled: bool
    device: str | None
    sample_rate: int


@dataclass(frozen=True)
class ServoAxisConfig:
    servo_id: int
    min_angle: float
    max_angle: float
    neutral_angle: float
    inverted: bool


@dataclass(frozen=True)
class ServoConfig:
    enabled: bool
    controller: str
    port: str
    baudrate: int
    timeout_seconds: float
    startup_delay_seconds: float
    default_move_time_ms: int
    dry_run: bool
    pan: ServoAxisConfig
    tilt: ServoAxisConfig


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    tilt_enabled: bool
    dead_zone_x: float
    dead_zone_y: float
    pan_gain: float
    tilt_gain: float
    pan_ki: float
    pan_kd: float
    tilt_ki: float
    tilt_kd: float
    pan_integral_zone: float
    tilt_integral_zone: float
    max_step_degrees: float
    max_pan_step_degrees: float
    max_tilt_step_degrees: float
    max_delta_change_degrees: float
    move_time_ms: int
    min_update_interval_ms: int
    min_effective_pan_delta: float
    min_effective_tilt_delta: float
    derivative_filter_alpha: float
    pan_response_exponent: float
    tilt_response_exponent: float
    target_stickiness: float
    target_lock_timeout_ms: int
    prediction_lead_seconds: float = 0.08
    stale_detection_timeout_ms: int = 300
    pan_direction: float = -1
    tilt_direction: float = 1
    center_on_target_lost: bool = True
    target_lost_timeout_seconds: float = 3
    idle_return_to_center_seconds: float = 0.0


@dataclass(frozen=True)
class AdminConfig:
    enabled: bool
    host: str
    port: int
    auth_token: str
    allow_remote_ops: bool
    command_timeout_seconds: float
    log_path: str
    allowed_commands: tuple[str, ...]


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
    voice: VoiceConfig
    speaker: SpeakerConfig
    servo: ServoConfig
    tracking: TrackingConfig
    admin: AdminConfig


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
    voice = data.get("voice", {})
    speaker = data.get("speaker", {})
    servo = data.get("servo", {})
    tracking = data.get("tracking", {})
    admin = data.get("admin", {})
    pan = servo.get("pan", {})
    tilt = servo.get("tilt", {})
    reconnect = server.get("reconnect", {})
    allowed_commands = admin.get(
        "allowed_commands",
        [
            "logs",
            "restart_edge_service",
            "pull_update",
            "run_install",
            "test_camera",
            "test_microphone",
            "test_speaker",
            "test_server_connection",
        ],
    )
    if not isinstance(allowed_commands, list):
        raise ValueError("admin.allowed_commands must be a list")

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
            face_model_path=str(
                vision.get("face_model_path", "pretrain/yolov5s-face.om")
            ),
            face_input_size=int(vision.get("face_input_size", 640)),
            face_iou_threshold=float(vision.get("face_iou_threshold", 0.45)),
            face_device_id=int(vision.get("face_device_id", 0)),
        ),
        microphone=MicrophoneConfig(
            enabled=bool(microphone.get("enabled", True)),
            sample_rate=int(microphone.get("sample_rate", 16000)),
            channels=int(microphone.get("channels", 1)),
            frame_ms=int(microphone.get("frame_ms", 30)),
            device=microphone.get("device"),
        ),
        wake_word=WakeWordConfig(
            enabled=bool(wake_word.get("enabled", False)),
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
        voice=VoiceConfig(
            visual_listen_timeout_ms=int(
                voice.get("visual_listen_timeout_ms", 6000)
            ),
            followup_listen_timeout_ms=int(
                voice.get("followup_listen_timeout_ms", 8000)
            ),
            welcome_text=str(
                voice.get("welcome_text", "你好，我在这里。有什么可以帮你的吗？")
            ),
            auto_listen_after_welcome=bool(
                voice.get("auto_listen_after_welcome", True)
            ),
            speech_interrupt_enabled=bool(
                voice.get("speech_interrupt_enabled", False)
            ),
            suppress_mic_while_speaking=bool(
                voice.get("suppress_mic_while_speaking", True)
            ),
            welcome_once_per_session=bool(
                voice.get("welcome_once_per_session", True)
            ),
        ),
        speaker=SpeakerConfig(
            enabled=bool(speaker.get("enabled", True)),
            device=speaker.get("device"),
            sample_rate=int(speaker.get("sample_rate", 16000)),
        ),
        servo=ServoConfig(
            enabled=bool(servo.get("enabled", False)),
            controller=str(servo.get("controller", "noop")),
            port=str(
                servo.get(
                    "port",
                    "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
                )
            ),
            baudrate=int(servo.get("baudrate", 115200)),
            timeout_seconds=float(servo.get("timeout_seconds", 0.2)),
            startup_delay_seconds=float(servo.get("startup_delay_seconds", 2)),
            default_move_time_ms=int(servo.get("default_move_time_ms", 300)),
            dry_run=bool(servo.get("dry_run", True)),
            pan=_parse_servo_axis(
                pan,
                default_servo_id=0,
                default_min_angle=75,
                default_max_angle=195,
            ),
            tilt=_parse_servo_axis(
                tilt,
                default_servo_id=1,
                default_min_angle=105,
                default_max_angle=165,
            ),
        ),
        tracking=TrackingConfig(
            enabled=bool(tracking.get("enabled", False)),
            tilt_enabled=bool(tracking.get("tilt_enabled", False)),
            dead_zone_x=float(tracking.get("dead_zone_x", 0.05)),
            dead_zone_y=float(tracking.get("dead_zone_y", 0.05)),
            pan_gain=float(tracking.get("pan_gain", 4.2)),
            tilt_gain=float(tracking.get("tilt_gain", 3.4)),
            pan_ki=float(tracking.get("pan_ki", 0.08)),
            pan_kd=float(tracking.get("pan_kd", 0.35)),
            tilt_ki=float(tracking.get("tilt_ki", 0.05)),
            tilt_kd=float(tracking.get("tilt_kd", 0.28)),
            pan_integral_zone=float(tracking.get("pan_integral_zone", 0.22)),
            tilt_integral_zone=float(tracking.get("tilt_integral_zone", 0.18)),
            max_step_degrees=float(tracking.get("max_step_degrees", 2.4)),
            max_pan_step_degrees=float(
                tracking.get(
                    "max_pan_step_degrees",
                    tracking.get("max_step_degrees", 2.4),
                )
            ),
            max_tilt_step_degrees=float(
                tracking.get(
                    "max_tilt_step_degrees",
                    tracking.get("max_step_degrees", 2.4),
                )
            ),
            max_delta_change_degrees=float(
                tracking.get("max_delta_change_degrees", 0.75)
            ),
            move_time_ms=int(tracking.get("move_time_ms", 45)),
            min_update_interval_ms=int(tracking.get("min_update_interval_ms", 45)),
            min_effective_pan_delta=float(
                tracking.get("min_effective_pan_delta", 0.2)
            ),
            min_effective_tilt_delta=float(
                tracking.get("min_effective_tilt_delta", 0.15)
            ),
            derivative_filter_alpha=float(tracking.get("derivative_filter_alpha", 0.35)),
            pan_response_exponent=float(
                tracking.get("pan_response_exponent", 1.35)
            ),
            tilt_response_exponent=float(
                tracking.get("tilt_response_exponent", 1.2)
            ),
            target_stickiness=float(tracking.get("target_stickiness", 0.25)),
            target_lock_timeout_ms=int(
                tracking.get("target_lock_timeout_ms", 1200)
            ),
            prediction_lead_seconds=float(
                tracking.get("prediction_lead_seconds", 0.05)
            ),
            stale_detection_timeout_ms=int(
                tracking.get("stale_detection_timeout_ms", 180)
            ),
            pan_direction=float(tracking.get("pan_direction", -1)),
            tilt_direction=float(tracking.get("tilt_direction", 1)),
            center_on_target_lost=bool(
                tracking.get("center_on_target_lost", True)
            ),
            target_lost_timeout_seconds=float(
                tracking.get("target_lost_timeout_seconds", 3)
            ),
            idle_return_to_center_seconds=float(
                tracking.get("idle_return_to_center_seconds", 1.5)
            ),
        ),
        admin=AdminConfig(
            enabled=bool(admin.get("enabled", True)),
            host=str(admin.get("host", "0.0.0.0")),
            port=int(admin.get("port", 8090)),
            auth_token=str(admin.get("auth_token", "change-me")),
            allow_remote_ops=bool(admin.get("allow_remote_ops", True)),
            command_timeout_seconds=float(admin.get("command_timeout_seconds", 30)),
            log_path=str(admin.get("log_path", "/var/log/ai-robot-edge-admin.log")),
            allowed_commands=tuple(str(command) for command in allowed_commands),
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
    if config.voice.visual_listen_timeout_ms <= 0:
        raise ValueError("voice.visual_listen_timeout_ms must be positive")
    if config.voice.followup_listen_timeout_ms <= 0:
        raise ValueError("voice.followup_listen_timeout_ms must be positive")
    if config.tracking.target_lock_timeout_ms < 0:
        raise ValueError("tracking.target_lock_timeout_ms must be non-negative")
    if config.admin.enabled and not config.admin.auth_token:
        raise ValueError("admin.auth_token is required when admin is enabled")
    for name, axis in (("pan", config.servo.pan), ("tilt", config.servo.tilt)):
        if not 0 <= axis.min_angle < axis.max_angle <= 270:
            raise ValueError(f"servo.{name} angles must satisfy 0 <= min < max <= 270")
        if not axis.min_angle <= axis.neutral_angle <= axis.max_angle:
            raise ValueError(f"servo.{name}.neutral_angle must be within limits")
    if config.tracking.dead_zone_x < 0 or config.tracking.dead_zone_x >= 0.5:
        raise ValueError("tracking.dead_zone_x must be between 0 and 0.5")
    if config.tracking.dead_zone_y < 0 or config.tracking.dead_zone_y >= 0.5:
        raise ValueError("tracking.dead_zone_y must be between 0 and 0.5")
    if config.tracking.max_pan_step_degrees <= 0:
        raise ValueError("tracking.max_pan_step_degrees must be positive")
    if config.tracking.max_tilt_step_degrees <= 0:
        raise ValueError("tracking.max_tilt_step_degrees must be positive")
    if config.tracking.max_delta_change_degrees < 0:
        raise ValueError("tracking.max_delta_change_degrees must be non-negative")
    if config.tracking.pan_ki < 0:
        raise ValueError("tracking.pan_ki must be non-negative")
    if config.tracking.pan_kd < 0:
        raise ValueError("tracking.pan_kd must be non-negative")
    if config.tracking.tilt_ki < 0:
        raise ValueError("tracking.tilt_ki must be non-negative")
    if config.tracking.tilt_kd < 0:
        raise ValueError("tracking.tilt_kd must be non-negative")
    if not 0 <= config.tracking.pan_integral_zone <= 1:
        raise ValueError("tracking.pan_integral_zone must be between 0 and 1")
    if not 0 <= config.tracking.tilt_integral_zone <= 1:
        raise ValueError("tracking.tilt_integral_zone must be between 0 and 1")
    if config.tracking.min_effective_pan_delta < 0:
        raise ValueError("tracking.min_effective_pan_delta must be non-negative")
    if config.tracking.min_effective_tilt_delta < 0:
        raise ValueError("tracking.min_effective_tilt_delta must be non-negative")
    if not 0 < config.tracking.derivative_filter_alpha <= 1:
        raise ValueError("tracking.derivative_filter_alpha must be between 0 (exclusive) and 1")
    if config.tracking.pan_response_exponent <= 0:
        raise ValueError("tracking.pan_response_exponent must be positive")
    if config.tracking.tilt_response_exponent <= 0:
        raise ValueError("tracking.tilt_response_exponent must be positive")
    if not 0 <= config.tracking.target_stickiness <= 1:
        raise ValueError("tracking.target_stickiness must be between 0 and 1")
    if config.tracking.prediction_lead_seconds < 0:
        raise ValueError("tracking.prediction_lead_seconds must be non-negative")
    if config.tracking.stale_detection_timeout_ms < 0:
        raise ValueError("tracking.stale_detection_timeout_ms must be non-negative")
    if config.tracking.pan_direction not in (-1, 1):
        raise ValueError("tracking.pan_direction must be -1 or 1")
    if config.tracking.tilt_direction not in (-1, 1):
        raise ValueError("tracking.tilt_direction must be -1 or 1")
    if config.tracking.target_lost_timeout_seconds < 0:
        raise ValueError("tracking.target_lost_timeout_seconds must be non-negative")
    if config.tracking.idle_return_to_center_seconds < 0:
        raise ValueError("tracking.idle_return_to_center_seconds must be non-negative")
    if config.vision.face_input_size <= 0 or config.vision.face_input_size % 32:
        raise ValueError("vision.face_input_size must be a positive multiple of 32")
    if not 0 < config.vision.face_iou_threshold < 1:
        raise ValueError("vision.face_iou_threshold must be between 0 and 1")


def _parse_servo_axis(
    data: dict[str, Any],
    *,
    default_servo_id: int,
    default_min_angle: float,
    default_max_angle: float,
) -> ServoAxisConfig:
    min_angle = float(data.get("min_angle", default_min_angle))
    max_angle = float(data.get("max_angle", default_max_angle))
    return ServoAxisConfig(
        servo_id=int(data.get("servo_id", default_servo_id)),
        min_angle=min_angle,
        max_angle=max_angle,
        neutral_angle=float(data.get("neutral_angle", (min_angle + max_angle) / 2)),
        inverted=bool(data.get("inverted", False)),
    )
