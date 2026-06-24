from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

from .runtime_state import runtime_state
from ..config import EdgeConfig
from ..events import now_ms


def collect_edge_status(config: EdgeConfig) -> dict[str, Any]:
    return {
        "device_id": config.device_id,
        "timestamp_ms": now_ms(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "atlas": {
            "target": "Atlas 200I DK A2",
            "ascend_toolkit_home": os.environ.get("ASCEND_TOOLKIT_HOME", ""),
            "cann_env_present": bool(os.environ.get("ASCEND_TOOLKIT_HOME")),
            "npu_smi": shutil.which("npu-smi") or "",
            "acl_runtime_configured": config.vision.detector.lower() in {"acl", "auto"}
            and config.runtime.prefer_npu,
        },
        "server": {
            "websocket_url": config.server.websocket_url.format(
                device_id=config.device_id
            ),
            "heartbeat_seconds": config.server.heartbeat_seconds,
        },
        "runtime": {
            "mode": config.runtime.mode,
            "prefer_npu": config.runtime.prefer_npu,
            "log_level": config.runtime.log_level,
        },
        "devices": {
            "camera": {
                "enabled": config.camera.enabled,
                "source": config.camera.source,
                "width": config.camera.width,
                "height": config.camera.height,
                "fps": config.camera.fps,
            },
            "microphone": {
                "enabled": config.microphone.enabled,
                "sample_rate": config.microphone.sample_rate,
                "device": config.microphone.device,
            },
            "speaker": {
                "enabled": config.speaker.enabled,
                "sample_rate": config.speaker.sample_rate,
                "device": config.speaker.device,
                "normalize_loudness": config.speaker.normalize_loudness,
                "target_rms_dbfs": config.speaker.target_rms_dbfs,
                "peak_limit": config.speaker.peak_limit,
                "max_output_channels": config.speaker.max_output_channels,
            },
            "wake_word": {
                "enabled": config.wake_word.enabled,
                "engine": config.wake_word.engine,
                "keyword_id": config.wake_word.keyword_id,
            },
            "vad": {
                "energy_threshold": config.vad.energy_threshold,
                "silence_ms": config.vad.silence_ms,
                "max_utterance_ms": config.vad.max_utterance_ms,
            },
        },
        "admin": {
            "enabled": config.admin.enabled,
            "port": config.admin.port,
            "allow_remote_ops": config.admin.allow_remote_ops,
            "allowed_commands": list(config.admin.allowed_commands),
        },
        "monitoring": runtime_state.snapshot(),
    }


def read_recent_logs(log_path: str, max_lines: int = 120) -> list[str]:
    path = Path(log_path)
    if not path.exists():
        return [
            f"log file does not exist: {path}",
            "Use the logs command for systemd journal output on deployed devices.",
        ]
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]
