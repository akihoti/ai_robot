from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from .registry import EdgeDeviceRegistry


SERVER_ALLOWED_COMMANDS = {
    "logs",
    "restart_edge_service",
    "pull_update",
    "run_install",
    "test_camera",
    "test_microphone",
    "test_speaker",
    "test_server_connection",
}


@dataclass(frozen=True)
class RemoteCommand:
    command: str
    request_id: str
    parameters: dict[str, Any]


class RemoteCommandService:
    def __init__(
        self,
        registry: EdgeDeviceRegistry,
        allowed_commands: set[str] | None = None,
    ) -> None:
        self.registry = registry
        self.allowed_commands = allowed_commands or SERVER_ALLOWED_COMMANDS

    async def request(
        self, device_id: str, command: str, parameters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if command not in self.allowed_commands:
            return {
                "queued": False,
                "message": f"command is not allowed: {command}",
            }
        remote_command = RemoteCommand(
            command=command,
            request_id=str(uuid4()),
            parameters=parameters or {},
        )
        result = await self.registry.enqueue_command(
            device_id,
            {
                "type": "command.request",
                "request_id": remote_command.request_id,
                "payload": {
                    "command": remote_command.command,
                    "parameters": remote_command.parameters,
                },
            },
        )
        return {**result, "request_id": remote_command.request_id}
