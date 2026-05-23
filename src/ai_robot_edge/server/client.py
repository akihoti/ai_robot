from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import websockets
from websockets.exceptions import WebSocketException

from ..config import EdgeConfig
from ..events import ActionIntent, ActionName, Utterance, now_ms

LOGGER = logging.getLogger(__name__)

TtsCallback = Callable[[bytes, int, int], Awaitable[None]]
ActionCallback = Callable[[ActionIntent], Awaitable[None]]


class ConversationClient:
    def __init__(
        self,
        config: EdgeConfig,
        on_tts: TtsCallback,
        on_action: ActionCallback,
    ) -> None:
        self.config = config
        self.on_tts = on_tts
        self.on_action = on_action

    async def send_utterance(self, utterance: Utterance) -> None:
        url = self.config.server.websocket_url.format(device_id=self.config.device_id)
        headers = {"Authorization": f"Bearer {self.config.server.bearer_token}"}
        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=self.config.server.connect_timeout_seconds,
                ping_interval=self.config.server.heartbeat_seconds,
            ) as websocket:
                await self._send_session(websocket, utterance)
                await self._receive_until_done(websocket)
        except TypeError:
            async with websockets.connect(
                url,
                extra_headers=headers,
                open_timeout=self.config.server.connect_timeout_seconds,
                ping_interval=self.config.server.heartbeat_seconds,
            ) as websocket:
                await self._send_session(websocket, utterance)
                await self._receive_until_done(websocket)
        except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
            LOGGER.error("conversation websocket failed: %s", exc)

    async def _send_session(self, websocket: Any, utterance: Utterance) -> None:
        await websocket.send(
            _frame(
                "session.start",
                utterance.request_id,
                {
                    "device_id": self.config.device_id,
                    "audio": {
                        "encoding": "pcm_s16le",
                        "sample_rate": self.config.microphone.sample_rate,
                        "channels": self.config.microphone.channels,
                    },
                    "wake_word_id": self.config.wake_word.keyword_id,
                    "context": {},
                },
            )
        )
        for sequence, audio_frame in enumerate(utterance.frames, start=1):
            await websocket.send(
                _frame(
                    "audio.chunk",
                    utterance.request_id,
                    {
                        "sequence": sequence,
                        "duration_ms": audio_frame.duration_ms,
                        "encoding": "pcm_s16le",
                    },
                )
            )
            await websocket.send(audio_frame.data)
        await websocket.send(
            _frame(
                "audio.end",
                utterance.request_id,
                {
                    "total_chunks": len(utterance.frames),
                    "reason": utterance.reason,
                },
            )
        )

    async def _receive_until_done(self, websocket: Any) -> None:
        tts_meta: dict[str, Any] | None = None
        async for message in websocket:
            if isinstance(message, bytes):
                if tts_meta is None:
                    LOGGER.warning("received binary frame without tts metadata")
                    continue
                await self.on_tts(
                    message,
                    int(tts_meta.get("sample_rate", self.config.speaker.sample_rate)),
                    int(tts_meta.get("channels", 1)),
                )
                if bool(tts_meta.get("is_final", False)):
                    break
                tts_meta = None
                continue

            envelope = json.loads(message)
            frame_type = envelope.get("type")
            payload = envelope.get("payload", {})
            if frame_type == "tts.chunk":
                tts_meta = payload
            elif frame_type == "action.intent":
                await self._handle_action(payload)
            elif frame_type in {"asr.partial", "asr.final", "llm.partial", "llm.final"}:
                LOGGER.info("%s: %s", frame_type, payload.get("text", ""))
            elif frame_type == "error":
                LOGGER.error("server error: %s", payload)
                if not payload.get("retryable", False):
                    break

    async def _handle_action(self, payload: dict[str, Any]) -> None:
        name = payload.get("name")
        try:
            action_name = ActionName(name)
        except ValueError:
            LOGGER.warning("unknown action intent ignored: %s", payload)
            return
        await self.on_action(
            ActionIntent(
                name=action_name,
                parameters=dict(payload.get("parameters", {})),
            )
        )


class ConversationWorker:
    def __init__(
        self,
        utterance_queue: asyncio.Queue[Utterance],
        client: ConversationClient,
    ) -> None:
        self.utterance_queue = utterance_queue
        self.client = client

    async def run(self) -> None:
        while True:
            utterance = await self.utterance_queue.get()
            await self.client.send_utterance(utterance)


def _frame(frame_type: str, request_id: str, payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "type": frame_type,
            "request_id": request_id,
            "timestamp_ms": now_ms(),
            "payload": payload,
        },
        ensure_ascii=False,
    )
