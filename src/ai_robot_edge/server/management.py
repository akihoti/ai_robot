from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import websockets
from websockets.exceptions import WebSocketException

from ..admin.commands import CommandRejected, run_command
from ..admin.status import collect_edge_status
from ..config import EdgeConfig
from ..events import now_ms

LOGGER = logging.getLogger(__name__)


class ManagementClient:
    def __init__(self, config: EdgeConfig) -> None:
        self.config = config

    async def run(self) -> None:
        delay = self.config.server.reconnect_initial_delay_seconds
        while True:
            try:
                await self._run_once()
                delay = self.config.server.reconnect_initial_delay_seconds
            except asyncio.CancelledError:
                raise
            except (OSError, WebSocketException, asyncio.TimeoutError) as exc:
                LOGGER.warning("management websocket failed: %s", exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.server.reconnect_max_delay_seconds)

    async def _run_once(self) -> None:
        url = self.config.server.websocket_url.format(device_id=self.config.device_id)
        headers = {"Authorization": f"Bearer {self.config.server.bearer_token}"}
        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=self.config.server.connect_timeout_seconds,
                ping_interval=self.config.server.heartbeat_seconds,
            ) as websocket:
                await self._serve(websocket)
        except TypeError:
            async with websockets.connect(
                url,
                extra_headers=headers,
                open_timeout=self.config.server.connect_timeout_seconds,
                ping_interval=self.config.server.heartbeat_seconds,
            ) as websocket:
                await self._serve(websocket)

    async def _serve(self, websocket: Any) -> None:
        sender = asyncio.create_task(self._send_status_loop(websocket))
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_text(websocket, message)
        finally:
            sender.cancel()

    async def _send_status_loop(self, websocket: Any) -> None:
        while True:
            await websocket.send(
                _frame("device.status", "", collect_edge_status(self.config))
            )
            await asyncio.sleep(self.config.server.heartbeat_seconds)

    async def _handle_text(self, websocket: Any, message: str) -> None:
        try:
            envelope = json.loads(message)
        except json.JSONDecodeError:
            return
        if envelope.get("type") != "command.request":
            return
        request_id = str(envelope.get("request_id", ""))
        payload = envelope.get("payload", {})
        command = str(payload.get("command", ""))
        await websocket.send(
            _frame(
                "command.progress",
                request_id,
                {"command": command, "status": "started"},
            )
        )
        try:
            result = await run_command(self.config, command)
        except CommandRejected as exc:
            result = {"ok": False, "command": command, "stderr": str(exc)}
        await websocket.send(_frame("command.result", request_id, result))


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
