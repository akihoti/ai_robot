from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import ConnectorConfig


@dataclass(frozen=True)
class ConnectorStatus:
    name: str
    configured: bool
    reachable: bool
    message: str


class BaseConnector:
    name = "connector"
    health_paths = ("/health",)

    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config

    @property
    def configured(self) -> bool:
        return bool(self.config.base_url)

    def _headers(self) -> dict[str, str]:
        if not self.config.api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.api_key}"}

    async def health(self) -> ConnectorStatus:
        if not self.configured:
            return ConnectorStatus(
                self.name,
                configured=False,
                reachable=False,
                message="base_url is not configured",
            )
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await self._first_reachable(client, self.health_paths)
            return ConnectorStatus(
                self.name,
                configured=True,
                reachable=response.status_code < 500,
                message=f"HTTP {response.status_code}",
            )
        except httpx.HTTPError as exc:
            return ConnectorStatus(
                self.name,
                configured=True,
                reachable=False,
                message=str(exc),
            )

    async def _first_reachable(
        self, client: httpx.AsyncClient, paths: tuple[str, ...]
    ) -> httpx.Response:
        last_response: httpx.Response | None = None
        for path in paths:
            response = await client.get(
                f"{self.config.base_url}{path}", headers=self._headers()
            )
            if response.status_code != 404:
                return response
            last_response = response
        assert last_response is not None
        return last_response


class XinferenceClient(BaseConnector):
    name = "xinference"
    health_paths = ("/docs", "/v1/models")

    async def list_models(self) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        paths = ["/v1/models", "/models"]
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for path in paths:
                try:
                    response = await client.get(
                        f"{self.config.base_url}{path}", headers=self._headers()
                    )
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    data = response.json()
                    return _as_items(data, "models")
                except httpx.HTTPError:
                    if path == paths[-1]:
                        raise
        return []

    async def model_action(self, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "message": "xinference is not configured"}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/models/actions",
                headers=self._headers(),
                json={"action": action, **payload},
            )
        if response.status_code == 404:
            return {
                "ok": False,
                "message": "model action endpoint is not available on upstream",
            }
        response.raise_for_status()
        return {"ok": True, "upstream": response.json()}

    async def transcribe_audio(
        self,
        *,
        model: str,
        filename: str,
        audio_bytes: bytes,
        content_type: str,
    ) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/audio/transcriptions",
                headers=self._headers(),
                data={"model": model},
                files={
                    "file": (
                        filename,
                        audio_bytes,
                        content_type or "application/octet-stream",
                    )
                },
            )
        response.raise_for_status()
        return response.json()

    async def synthesize_speech(
        self,
        *,
        model: str,
        text: str,
        voice: str,
    ) -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/audio/speech",
                headers=self._headers() | {"Content-Type": "application/json"},
                json={"model": model, "input": text, "voice": voice},
            )
        response.raise_for_status()
        media_type = response.headers.get("content-type", "audio/mpeg")
        return response.content, media_type


class RagflowClient(BaseConnector):
    name = "ragflow"
    health_paths = ("/", "/api/v1/datasets")

    async def list_knowledge_bases(self) -> list[dict[str, Any]]:
        if not self.configured:
            return []
        paths = ["/api/v1/datasets", "/api/v1/knowledge-bases"]
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for path in paths:
                try:
                    response = await client.get(
                        f"{self.config.base_url}{path}", headers=self._headers()
                    )
                    if response.status_code == 404:
                        continue
                    response.raise_for_status()
                    data = response.json()
                    return _as_items(data, "knowledge_bases")
                except httpx.HTTPError:
                    if path == paths[-1]:
                        raise
        return []

    async def sync_knowledge_base(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "message": "ragflow is not configured"}
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/datasets/sync",
                headers=self._headers(),
                json=payload,
            )
        if response.status_code == 404:
            return {
                "ok": False,
                "message": "knowledge-base sync endpoint is not available upstream",
            }
        response.raise_for_status()
        return {"ok": True, "upstream": response.json()}

    async def query(self, question: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.configured:
            return {
                "answer": "",
                "sources": [],
                "message": "ragflow is not configured",
            }
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/chats/completions",
                headers=self._headers(),
                json={"question": question, **payload},
            )
        response.raise_for_status()
        return response.json()

    async def query_chat(
        self, *, chat_id: str, question: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        system_prompt = str(payload.get("system_prompt", "")).strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/openai/{chat_id}/chat/completions",
                headers=self._headers() | {"Content-Type": "application/json"},
                json={
                    "model": payload.get("model", "model"),
                    "messages": messages,
                    "stream": False,
                    **{
                        k: v
                        for k, v in payload.items()
                        if k not in {"question", "system_prompt"}
                    },
                },
            )
        response.raise_for_status()
        return response.json()


def _as_items(data: Any, preferred_key: str) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in (preferred_key, "data", "items", "models"):
        value = data.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [dict(value)]
    return [data]
