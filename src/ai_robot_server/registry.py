from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EdgeDevice:
    device_id: str
    online: bool = False
    connection_count: int = 0
    last_seen_ms: int | None = None
    status: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)


class EdgeDeviceRegistry:
    def __init__(self) -> None:
        self._devices: dict[str, EdgeDevice] = {}
        self._command_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}

    def list_devices(self) -> list[dict[str, Any]]:
        return [asdict(device) for device in self._devices.values()]

    def get_device(self, device_id: str) -> EdgeDevice:
        if device_id not in self._devices:
            self._devices[device_id] = EdgeDevice(device_id=device_id)
        return self._devices[device_id]

    def mark_online(self, device_id: str) -> None:
        device = self.get_device(device_id)
        device.connection_count += 1
        device.online = True
        self._command_queues.setdefault(device_id, asyncio.Queue(maxsize=32))

    def mark_offline(self, device_id: str) -> None:
        device = self.get_device(device_id)
        if device.connection_count > 0:
            device.connection_count -= 1
        device.online = device.connection_count > 0

    def update_status(self, device_id: str, status: dict[str, Any]) -> None:
        device = self.get_device(device_id)
        device.online = True
        device.status = dict(status)
        if "timestamp_ms" in status:
            device.last_seen_ms = int(status["timestamp_ms"])

    def add_log(self, device_id: str, line: str) -> None:
        device = self.get_device(device_id)
        device.logs.append(line)
        device.logs = device.logs[-200:]

    async def enqueue_command(
        self, device_id: str, command: dict[str, Any]
    ) -> dict[str, Any]:
        device = self.get_device(device_id)
        if not device.online:
            return {"queued": False, "message": "device is offline"}
        queue = self._command_queues.setdefault(device_id, asyncio.Queue(maxsize=32))
        await queue.put(command)
        return {"queued": True, "message": "command queued"}

    async def next_command(self, device_id: str) -> dict[str, Any]:
        queue = self._command_queues.setdefault(device_id, asyncio.Queue(maxsize=32))
        return await queue.get()
