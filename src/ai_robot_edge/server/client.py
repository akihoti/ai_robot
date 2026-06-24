from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import websockets
from websockets.exceptions import WebSocketException

from ..admin.runtime_state import runtime_state
from ..config import EdgeConfig
from ..events import (
    ActionIntent,
    ActionName,
    ConversationEvent,
    Utterance,
    now_ms,
)
from ..session import SessionController

LOGGER = logging.getLogger(__name__)

TtsCallback = Callable[[bytes, int, int, str], Awaitable[None]]
ActionCallback = Callable[[ActionIntent], Awaitable[None]]


@dataclass(frozen=True)
class ConversationResult:
    success: bool
    had_audio: bool = False
    error_message: str = ""


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

    async def send_utterance(
        self,
        utterance: Utterance,
        context: dict[str, Any] | None = None,
    ) -> ConversationResult:
        return await self._open_session_and_exchange(
            request_id=utterance.request_id,
            send_payload=lambda websocket: self._send_session(
                websocket, utterance, context or {}
            ),
        )

    async def send_welcome_event(self, event: ConversationEvent) -> ConversationResult:
        request_id = event.request_id or str(uuid4())
        payload = {
            "event": (
                event.vision_event.event_type.value
                if event.vision_event is not None
                else "welcome_triggered"
            ),
            "confidence": event.vision_event.confidence if event.vision_event else 1.0,
            "source": event.vision_event.source if event.vision_event else "camera",
            "cooldown_active": False,
            "welcome_text": self.config.voice.welcome_text,
        }
        return await self._open_session_and_exchange(
            request_id=request_id,
            send_payload=lambda websocket: websocket.send(
                _frame("event.vision", request_id, payload)
            ),
        )

    async def _open_session_and_exchange(
        self,
        *,
        request_id: str,
        send_payload: Callable[[Any], Awaitable[None]],
    ) -> ConversationResult:
        url = self.config.server.websocket_url.format(device_id=self.config.device_id)
        headers = {"Authorization": f"Bearer {self.config.server.bearer_token}"}
        connect_kwargs = {
            "open_timeout": self.config.server.connect_timeout_seconds,
            "ping_interval": self.config.server.heartbeat_seconds,
        }
        if "additional_headers" in inspect.signature(websockets.connect).parameters:
            connect_kwargs["additional_headers"] = headers
        else:
            connect_kwargs["extra_headers"] = headers
        try:
            async with websockets.connect(url, **connect_kwargs) as websocket:
                await send_payload(websocket)
                return await self._receive_until_done(websocket, request_id)
        except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
            LOGGER.error("conversation websocket failed: %s", exc)
            return ConversationResult(success=False, error_message=str(exc))

    async def _send_session(
        self,
        websocket: Any,
        utterance: Utterance,
        context: dict[str, Any],
    ) -> None:
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
                    "context": context,
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

    async def _receive_until_done(self, websocket: Any, request_id: str) -> ConversationResult:
        tts_meta: dict[str, Any] | None = None
        saw_audio = False
        error_message = ""
        async for message in websocket:
            if isinstance(message, bytes):
                if tts_meta is None:
                    LOGGER.warning("received binary frame without tts metadata")
                    continue
                saw_audio = True
                await self.on_tts(
                    message,
                    int(tts_meta.get("sample_rate", self.config.speaker.sample_rate)),
                    int(tts_meta.get("channels", 1)),
                    str(tts_meta.get("media_type", "audio/pcm")),
                )
                if bool(tts_meta.get("is_final", False)):
                    return ConversationResult(success=True, had_audio=saw_audio)
                tts_meta = None
                continue

            envelope = json.loads(message)
            if envelope.get("request_id") not in {"", request_id}:
                continue
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
                error_message = str(payload.get("message", payload))
                return ConversationResult(
                    success=False,
                    had_audio=saw_audio,
                    error_message=error_message,
                )
        return ConversationResult(
            success=not bool(error_message),
            had_audio=saw_audio,
            error_message=error_message,
        )

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
        conversation_queue: asyncio.Queue[ConversationEvent],
        client: ConversationClient,
        playback_idle: asyncio.Event,
        playback_active: asyncio.Event | None,
        auto_listen_after_welcome: bool,
        arm_welcome_listening_window: Callable[[], None],
        arm_followup_listening_window: Callable[[], None],
        session_controller: SessionController,
        interrupt_playback: Callable[[], Awaitable[None]] | None = None,
        speech_interrupt_enabled: bool = False,
    ) -> None:
        self.utterance_queue = utterance_queue
        self.conversation_queue = conversation_queue
        self.client = client
        self.playback_idle = playback_idle
        self.playback_active = playback_active
        self.auto_listen_after_welcome = auto_listen_after_welcome
        self.arm_welcome_listening_window = arm_welcome_listening_window
        self.arm_followup_listening_window = arm_followup_listening_window
        self.session_controller = session_controller
        self.interrupt_playback = interrupt_playback
        self.speech_interrupt_enabled = speech_interrupt_enabled

    async def run(self) -> None:
        while True:
            utterance_task = asyncio.create_task(self.utterance_queue.get())
            conversation_task = asyncio.create_task(self.conversation_queue.get())
            done, pending = await asyncio.wait(
                {utterance_task, conversation_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            event = await done.pop()
            if isinstance(event, Utterance):
                context = await self.session_controller.next_turn_context()
                LOGGER.info(
                    "starting conversation turn request_id=%s session_id=%s turn_index=%s duration_ms=%s",
                    event.request_id,
                    context.get("session_id", ""),
                    context.get("turn_index", ""),
                    event.duration_ms,
                )
                runtime_state.record_server_turn(
                    phase="started",
                    request_id=event.request_id,
                    session_id=str(context.get("session_id", "")) or None,
                    turn_index=int(context.get("turn_index", 0)),
                )
                result = await self.client.send_utterance(event, context=context)
                runtime_state.record_server_turn(
                    phase="completed",
                    request_id=event.request_id,
                    session_id=str(context.get("session_id", "")) or None,
                    turn_index=int(context.get("turn_index", 0)),
                    success=result.success,
                    error_message=result.error_message,
                )
                if result.success:
                    await self._wait_for_playback_with_interrupts(
                        expected_audio=result.had_audio
                    )
                    await self.session_controller.note_followup_listening()
                    self.arm_followup_listening_window()
                else:
                    LOGGER.warning(
                        "conversation turn failed, rearming followup listening: %s",
                        result.error_message,
                    )
                    await self.session_controller.note_followup_listening()
                    self.arm_followup_listening_window()
                continue

            LOGGER.info("sending welcome event request_id=%s", event.request_id)
            runtime_state.record_server_turn(
                phase="welcome_started",
                request_id=event.request_id,
            )
            result = await self.client.send_welcome_event(event)
            runtime_state.record_server_turn(
                phase="welcome_completed",
                request_id=event.request_id,
                success=result.success,
                error_message=result.error_message,
            )
            interrupted = False
            if result.success:
                interrupted = await self._wait_for_playback_with_interrupts(
                    expected_audio=result.had_audio
                )
            else:
                LOGGER.warning("welcome event failed: %s", result.error_message)
            if self.auto_listen_after_welcome:
                if interrupted:
                    await self.session_controller.note_followup_listening()
                    self.arm_followup_listening_window()
                else:
                    await self.session_controller.note_welcome_playback_finished()
                    self.arm_welcome_listening_window()

    async def _wait_for_playback_with_interrupts(self, *, expected_audio: bool) -> bool:
        interrupted = False
        if expected_audio:
            await self._await_playback_start()
        while True:
            if not self.speech_interrupt_enabled:
                await self.playback_idle.wait()
                return interrupted

            idle_task = asyncio.create_task(self.playback_idle.wait())
            utterance_task = asyncio.create_task(self.utterance_queue.get())
            done, pending = await asyncio.wait(
                {idle_task, utterance_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

            if idle_task in done and idle_task.result():
                return interrupted

            utterance = utterance_task.result()
            interrupted = True
            LOGGER.info(
                "speech interrupt triggered: request_id=%s duration_ms=%s",
                utterance.request_id,
                utterance.duration_ms,
            )
            runtime_state.record_playback_interrupt()
            if self.interrupt_playback is not None:
                await self.interrupt_playback()
            await self.session_controller.note_utterance_ready()
            context = await self.session_controller.next_turn_context()
            result = await self.client.send_utterance(utterance, context=context)
            if not result.success:
                LOGGER.warning(
                    "interrupted conversation turn failed: %s",
                    result.error_message,
                )

    async def _await_playback_start(self, timeout_seconds: float = 1.0) -> None:
        if self.playback_active is None:
            return
        if self.playback_active.is_set() or not self.playback_idle.is_set():
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        while loop.time() < deadline:
            if self.playback_active.is_set() or not self.playback_idle.is_set():
                return
            await asyncio.sleep(0.01)


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
