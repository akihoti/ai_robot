from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

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
