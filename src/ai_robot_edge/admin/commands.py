from __future__ import annotations

import asyncio
from dataclasses import dataclass
import socket
from typing import Any
from urllib.parse import urlparse

from ..config import EdgeConfig


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    description: str
    mutating: bool = False


COMMAND_SPECS: dict[str, CommandSpec] = {
    "logs": CommandSpec(
        ("journalctl", "-u", "ai-robot-edge", "-n", "120", "--no-pager"),
        "Read recent edge service logs",
    ),
    "restart_edge_service": CommandSpec(
        ("systemctl", "restart", "ai-robot-edge"),
        "Restart the edge service",
        mutating=True,
    ),
    "pull_update": CommandSpec(
        ("git", "-C", "/opt/ai_robot", "pull", "--ff-only"),
        "Pull the latest repository update",
        mutating=True,
    ),
    "run_install": CommandSpec(
        ("/opt/ai_robot/scripts/install_edge.sh",),
        "Run the edge install script",
        mutating=True,
    ),
    "test_camera": CommandSpec(
        ("python3", "-m", "ai_robot_edge.admin.probes", "camera"),
        "Run a camera probe",
    ),
    "test_microphone": CommandSpec(
        ("python3", "-m", "ai_robot_edge.admin.probes", "microphone"),
        "Run a microphone probe",
    ),
    "test_speaker": CommandSpec(
        ("python3", "-m", "ai_robot_edge.admin.probes", "speaker"),
        "Run a speaker probe",
    ),
    "test_server_connection": CommandSpec(
        ("python3", "-m", "ai_robot_edge.admin.probes", "server"),
        "Run a server connectivity probe",
    ),
}


class CommandRejected(ValueError):
    pass


def validate_command(config: EdgeConfig, command_name: str) -> CommandSpec:
    if command_name not in COMMAND_SPECS:
        raise CommandRejected(f"unknown command: {command_name}")
    if command_name not in config.admin.allowed_commands:
        raise CommandRejected(f"command is not enabled: {command_name}")
    spec = COMMAND_SPECS[command_name]
    if spec.mutating and not config.admin.allow_remote_ops:
        raise CommandRejected("remote operations are disabled")
    return spec


async def run_command(config: EdgeConfig, command_name: str) -> dict[str, Any]:
    spec = validate_command(config, command_name)
    if command_name == "test_server_connection":
        return await _run_server_connection_probe(config, command_name, spec)
    process = await asyncio.create_subprocess_exec(
        *spec.argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=config.admin.command_timeout_seconds
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        return {
            "ok": False,
            "command": command_name,
            "returncode": None,
            "stdout": "",
            "stderr": "command timed out",
        }
    return {
        "ok": process.returncode == 0,
        "command": command_name,
        "description": spec.description,
        "returncode": process.returncode,
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
    }


async def _run_server_connection_probe(
    config: EdgeConfig,
    command_name: str,
    spec: CommandSpec,
) -> dict[str, Any]:
    target = _server_probe_target(config)

    def connect() -> dict[str, Any]:
        try:
            with socket.create_connection(
                (target["host"], target["port"]),
                timeout=config.server.connect_timeout_seconds,
            ):
                return {
                    "ok": True,
                    "stdout": (
                        f"connected to {target['host']}:{target['port']} "
                        f"from {target['url']}\n"
                    ),
                    "stderr": "",
                }
        except OSError as exc:
            return {"ok": False, "stdout": "", "stderr": f"{exc}\n"}

    result = await asyncio.to_thread(connect)
    return {
        "ok": result["ok"],
        "command": command_name,
        "description": spec.description,
        "returncode": 0 if result["ok"] else 1,
        "stdout": result["stdout"],
        "stderr": result["stderr"],
    }


def _server_probe_target(config: EdgeConfig) -> dict[str, Any]:
    url = config.server.websocket_url.format(device_id=config.device_id)
    parsed = urlparse(url)
    if not parsed.hostname:
        raise CommandRejected(f"server websocket URL has no host: {url}")
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "wss":
        port = 443
    else:
        port = 80
    return {"url": url, "host": parsed.hostname, "port": port}
