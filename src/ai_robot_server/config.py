from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_VOICE_SYSTEM_PROMPT = (
    "你是机器视觉工作室的语音讲解助手，只回答项目、方案、设备、流程相关问题。"
    "项目问题必须基于知识库；知识库无依据时只回答：知识库中未找到您要的答案，我可以换个问法继续帮你查。"
    "用中文口语化回答，每次1到2句，80到120字以内；禁止 Markdown、长列表、代码和推理过程。"
)


@dataclass(frozen=True)
class HttpConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    admin_token: str = "change-me"


@dataclass(frozen=True)
class ConnectorConfig:
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 15


@dataclass(frozen=True)
class EdgeAuthConfig:
    bearer_tokens: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class VoiceGatewayConfig:
    auth_token: str = ""
    asr_model: str = ""
    tts_model: str = ""
    tts_voice: str = ""
    ragflow_chat_id: str = ""
    tts_media_type: str = "audio/wav"
    welcome_text: str = "你好，我在这里。有什么可以帮你的吗？"
    system_prompt: str = DEFAULT_VOICE_SYSTEM_PROMPT


@dataclass(frozen=True)
class TtsConfig:
    provider: str = "xinference"
    base_url: str = ""
    api_key: str = ""
    timeout_seconds: float = 15
    voice: str = ""
    speed: float = 1.0
    fallback_provider: str = ""


@dataclass(frozen=True)
class ServerAppConfig:
    http: HttpConfig
    ragflow: ConnectorConfig
    xinference: ConnectorConfig
    tts: TtsConfig
    edge: EdgeAuthConfig
    voice_gateway: VoiceGatewayConfig


def load_server_config(path: str | Path) -> ServerAppConfig:
    data = _load_yaml(path)
    return parse_server_config(data)


def parse_server_config(data: dict[str, Any]) -> ServerAppConfig:
    http = data.get("http", {})
    ragflow = data.get("ragflow", {})
    xinference = data.get("xinference", {})
    tts = data.get("tts", {})
    edge = data.get("edge", {})
    voice_gateway = data.get("voice_gateway", {})
    bearer_tokens = edge.get("bearer_tokens", {})
    if bearer_tokens is None:
        bearer_tokens = {}
    if not isinstance(bearer_tokens, dict):
        raise ValueError("edge.bearer_tokens must be a mapping")

    return ServerAppConfig(
        http=HttpConfig(
            host=str(http.get("host", "0.0.0.0")),
            port=int(http.get("port", 8000)),
            admin_token=str(http.get("admin_token", "change-me")),
        ),
        ragflow=ConnectorConfig(
            base_url=str(ragflow.get("base_url", "")).rstrip("/"),
            api_key=str(ragflow.get("api_key", "")),
            timeout_seconds=float(ragflow.get("timeout_seconds", 15)),
        ),
        xinference=ConnectorConfig(
            base_url=str(xinference.get("base_url", "")).rstrip("/"),
            api_key=str(xinference.get("api_key", "")),
            timeout_seconds=float(xinference.get("timeout_seconds", 15)),
        ),
        tts=TtsConfig(
            provider=str(tts.get("provider", "xinference")).strip() or "xinference",
            base_url=str(tts.get("base_url", "")).rstrip("/"),
            api_key=str(tts.get("api_key", "")),
            timeout_seconds=float(tts.get("timeout_seconds", 15)),
            voice=str(tts.get("voice", "")),
            speed=float(tts.get("speed", 1.0)),
            fallback_provider=str(tts.get("fallback_provider", "")).strip(),
        ),
        edge=EdgeAuthConfig(
            bearer_tokens={str(k): str(v) for k, v in bearer_tokens.items()}
        ),
        voice_gateway=VoiceGatewayConfig(
            auth_token=str(voice_gateway.get("auth_token", "")),
            asr_model=str(voice_gateway.get("asr_model", "")),
            tts_model=str(voice_gateway.get("tts_model", "")),
            tts_voice=str(voice_gateway.get("tts_voice", "")),
            ragflow_chat_id=str(voice_gateway.get("ragflow_chat_id", "")),
            tts_media_type=str(voice_gateway.get("tts_media_type", "audio/wav")),
            welcome_text=str(
                voice_gateway.get(
                    "welcome_text",
                    "你好，我在这里。有什么可以帮你的吗？",
                )
            ),
            system_prompt=str(
                voice_gateway.get(
                    "system_prompt",
                    DEFAULT_VOICE_SYSTEM_PROMPT,
                )
            ),
        ),
    )


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"server config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("server config root must be a mapping")
    return data
