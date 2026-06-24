from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import ConnectorConfig


LOGGER = logging.getLogger(__name__)


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
        preferred_media_type: str = "",
    ) -> tuple[bytes, str]:
        payload: dict[str, Any] = {"model": model, "input": text}
        if voice.strip():
            payload["voice"] = voice
        response_format = _response_format_for_media_type(preferred_media_type)
        if response_format is not None:
            payload["response_format"] = response_format
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/audio/speech",
                headers=self._headers() | {"Content-Type": "application/json"},
                json=payload,
            )
        response.raise_for_status()
        media_type = _normalize_tts_media_type(
            response.headers.get("content-type", "application/octet-stream"),
            response.content,
            preferred_media_type,
        )
        return response.content, media_type


class MeloTtsClient(BaseConnector):
    name = "melotts_sidecar"
    health_paths = ("/health",)

    async def synthesize_speech(
        self,
        *,
        text: str,
        voice: str = "",
        speed: float = 1.0,
        preferred_media_type: str = "",
    ) -> tuple[bytes, str]:
        payload: dict[str, Any] = {"input": text, "speed": speed}
        if voice.strip():
            payload["voice"] = voice
        response_format = _response_format_for_media_type(preferred_media_type)
        if response_format is not None:
            payload["response_format"] = response_format
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/v1/audio/speech",
                headers=self._headers() | {"Content-Type": "application/json"},
                json=payload,
            )
        response.raise_for_status()
        media_type = _normalize_tts_media_type(
            response.headers.get("content-type", "application/octet-stream"),
            response.content,
            preferred_media_type,
        )
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
        messages = _build_chat_messages(question, payload)
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/api/v1/openai/{chat_id}/chat/completions",
                headers=self._headers() | {"Content-Type": "application/json"},
                json={
                    "model": payload.get("model", "model"),
                    "messages": messages,
                    "stream": False,
                    **_chat_passthrough_payload(payload),
                },
            )
        response.raise_for_status()
        return response.json()

    async def stream_query_chat(
        self, *, chat_id: str, question: str, payload: dict[str, Any]
    ) -> AsyncIterator[str]:
        messages = _build_chat_messages(question, payload)
        dedupe = _RagflowStreamDedupe()
        timeout = httpx.Timeout(
            connect=self.config.timeout_seconds,
            read=120.0,
            write=self.config.timeout_seconds,
            pool=self.config.timeout_seconds,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.config.base_url}/api/v1/openai/{chat_id}/chat/completions",
                headers=self._headers() | {"Content-Type": "application/json"},
                json={
                    "model": payload.get("model", "model"),
                    "messages": messages,
                    "stream": True,
                    **_chat_passthrough_payload(payload),
                },
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    text = _extract_stream_delta_text(chunk)
                    output, action = dedupe.accept(text)
                    if action != "delta":
                        LOGGER.info(
                            "ragflow stream dedupe action=%s raw_chars=%s output_chars=%s emitted_chars=%s",
                            action,
                            len(text),
                            len(output),
                            len(dedupe.emitted_text),
                        )
                    if output:
                        yield output


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


def _response_format_for_media_type(media_type: str) -> str | None:
    normalized = media_type.split(";", 1)[0].strip().lower()
    if not normalized:
        return None
    if "wav" in normalized:
        return "wav"
    if "mpeg" in normalized or "mp3" in normalized:
        return "mp3"
    if "pcm" in normalized:
        return "pcm"
    if "flac" in normalized:
        return "flac"
    if "ogg" in normalized or "opus" in normalized:
        return "opus"
    return None


def _normalize_tts_media_type(
    reported_media_type: str,
    audio_bytes: bytes,
    preferred_media_type: str,
) -> str:
    normalized = reported_media_type.split(";", 1)[0].strip().lower()
    if normalized and normalized not in {
        "application/octet-stream",
        "binary/octet-stream",
        "application/octetstream",
    }:
        return normalized

    sniffed = _sniff_audio_media_type(audio_bytes)
    if sniffed is not None:
        return sniffed

    preferred = preferred_media_type.split(";", 1)[0].strip().lower()
    if preferred:
        return preferred
    return normalized or "application/octet-stream"


def _sniff_audio_media_type(audio_bytes: bytes) -> str | None:
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return "audio/wav"
    if audio_bytes[:3] == b"ID3":
        return "audio/mpeg"
    if len(audio_bytes) >= 2 and audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    if audio_bytes[:4] == b"OggS":
        return "audio/ogg"
    return None


def _build_chat_messages(question: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_prompt = str(payload.get("system_prompt", "")).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    history = _normalize_chat_history(payload.get("conversation_history", []))
    messages.extend(history)
    messages.append({"role": "user", "content": _question_with_history(question, history)})
    return messages


def _chat_passthrough_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in payload.items()
        if k not in {"question", "system_prompt", "conversation_history", "messages"}
    }


def _normalize_chat_history(history: Any) -> list[dict[str, str]]:
    if not isinstance(history, list):
        return []
    messages: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        if role not in {"user", "assistant"}:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages


def _question_with_history(question: str, history: list[dict[str, str]]) -> str:
    question = str(question).strip()
    if not history:
        return question
    lines = ["对话历史："]
    for message in history:
        label = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{label}：{message['content']}")
    lines.append("")
    lines.append(f"当前用户问题：{question}")
    return "\n".join(lines)


def _extract_stream_delta_text(chunk: dict[str, Any]) -> str:
    try:
        choice = chunk["choices"][0]
    except (KeyError, IndexError, TypeError):
        return ""
    for candidate in (
        choice.get("delta", {}).get("content"),
        choice.get("message", {}).get("content"),
    ):
        text = _normalize_stream_content(candidate)
        if text:
            return text
    return ""


class _RagflowStreamDedupe:
    """Normalize RAGFlow streams that mix delta chunks with full-answer chunks."""

    _MIN_EXACT_DUPLICATE_CHARS = 8

    def __init__(self) -> None:
        self.emitted_text = ""

    def accept(self, text: str) -> tuple[str, str]:
        if not text:
            return "", "empty"

        if self.emitted_text and text.startswith(self.emitted_text):
            suffix = text[len(self.emitted_text) :]
            if suffix:
                self.emitted_text = text
                return suffix, "cumulative_suffix"
            if self._looks_like_full_answer_duplicate(text):
                return "", "duplicate_full"

        self.emitted_text += text
        return text, "delta"

    def _looks_like_full_answer_duplicate(self, text: str) -> bool:
        return len(text) >= self._MIN_EXACT_DUPLICATE_CHARS and any(
            char in text for char in "。！？；!?"
        )


def _normalize_stream_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    return "".join(parts)
