from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
class ServerAppConfig:
    http: HttpConfig
    ragflow: ConnectorConfig
    xinference: ConnectorConfig
    edge: EdgeAuthConfig


def load_server_config(path: str | Path) -> ServerAppConfig:
    data = _load_yaml(path)
    return parse_server_config(data)


def parse_server_config(data: dict[str, Any]) -> ServerAppConfig:
    http = data.get("http", {})
    ragflow = data.get("ragflow", {})
    xinference = data.get("xinference", {})
    edge = data.get("edge", {})
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
        edge=EdgeAuthConfig(
            bearer_tokens={str(k): str(v) for k, v in bearer_tokens.items()}
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
