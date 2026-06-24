from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from .devices.gimbal import SongJiaProtocol

LOGGER = logging.getLogger(__name__)

WAVE_SERVO_ID = 2
WAVE_STEP_TIME_MS = 280
WAVE_ANGLES = (90, 135, 45, 135, 45, 90)

SleepFunc = Callable[[float], Awaitable[None]]


def welcome_wave_commands() -> list[str]:
    return [
        SongJiaProtocol.move_command(WAVE_SERVO_ID, angle, WAVE_STEP_TIME_MS)
        for angle in WAVE_ANGLES
    ]


async def execute_welcome_wave_action_group(
    servo_controller: object,
    *,
    sleep: SleepFunc = asyncio.sleep,
) -> bool:
    send = getattr(servo_controller, "_send", None)
    lock = getattr(servo_controller, "_lock", None)
    if not callable(send) or lock is None:
        LOGGER.info(
            "welcome wave action group skipped; servo controller does not expose gimbal transport"
        )
        return False

    commands = welcome_wave_commands()
    for index, command in enumerate(commands):
        async with lock:
            await send(command)
        if index < len(commands) - 1:
            await sleep(WAVE_STEP_TIME_MS / 1000)
    return True
