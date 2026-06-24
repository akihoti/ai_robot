from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from collections.abc import Awaitable, Callable

from .action_groups import execute_welcome_wave_action_group
from .admin.runtime_state import runtime_state
from .devices.base import ServoController
from .events import ActionIntent, ActionName
from .interaction import action_priority

LOGGER = logging.getLogger(__name__)
SleepFunc = Callable[[float], Awaitable[None]]


class ActionDispatcher:
    def __init__(
        self,
        action_queue: asyncio.Queue[ActionIntent],
        servo_controller: ServoController,
        *,
        wave_sleep: SleepFunc = asyncio.sleep,
    ) -> None:
        self.action_queue = action_queue
        self.servo_controller = servo_controller
        self.wave_sleep = wave_sleep
        self._degraded = False

    async def run(self) -> None:
        current_task: asyncio.Task[None] | None = None
        current_intent: ActionIntent | None = None
        while True:
            if current_task is None:
                intent = await self.action_queue.get()
                if not await self._start_intent(intent):
                    continue
                current_intent = intent
                current_task = asyncio.create_task(
                    self._execute_intent(intent),
                    name=f"action-{intent.name.value}",
                )
                continue

            next_intent_task = asyncio.create_task(self.action_queue.get())
            done, pending = await asyncio.wait(
                {current_task, next_intent_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if current_task in done:
                with suppress(asyncio.CancelledError):
                    await current_task
                current_task = None
                current_intent = None
            if next_intent_task in done:
                next_intent = next_intent_task.result()
                if current_task is None:
                    if await self._start_intent(next_intent):
                        current_intent = next_intent
                        current_task = asyncio.create_task(
                            self._execute_intent(next_intent),
                            name=f"action-{next_intent.name.value}",
                        )
                    continue
                if action_priority(next_intent) > action_priority(current_intent):
                    LOGGER.info(
                        "interrupting action %s with higher-priority action %s",
                        current_intent,
                        next_intent,
                    )
                    runtime_state.record_action(
                        name=next_intent.name.value,
                        interrupted=True,
                    )
                    current_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await current_task
                    await self.servo_controller.stop()
                    if await self._start_intent(next_intent):
                        current_intent = next_intent
                        current_task = asyncio.create_task(
                            self._execute_intent(next_intent),
                            name=f"action-{next_intent.name.value}",
                        )
                    else:
                        current_task = None
                        current_intent = None
                else:
                    LOGGER.info(
                        "dropping lower-priority action while %s is running: %s",
                        current_intent,
                        next_intent,
                    )
            for task in pending:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def _start_intent(self, intent: ActionIntent) -> bool:
        if self._degraded:
            LOGGER.warning("action dropped in degraded servo mode: %s", intent)
            return False
        LOGGER.info("dispatching action intent: %s", intent)
        runtime_state.record_action(name=intent.name.value)
        return True

    async def _execute_intent(self, intent: ActionIntent) -> None:
        try:
            if intent.name == ActionName.WELCOME_MOTION:
                handled = await execute_welcome_wave_action_group(
                    self.servo_controller,
                    sleep=self.wave_sleep,
                )
                if handled:
                    return
            await self.servo_controller.execute(intent)
        except Exception:
            self._degraded = True
            runtime_state.set_component_state("servo_degraded", True)
            LOGGER.exception(
                "servo controller failed; entering degraded mode and dropping future actions"
            )
