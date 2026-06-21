from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from time import monotonic

from ..config import VisionConfig
from ..events import (
    ActionIntent,
    ActionName,
    ConversationEvent,
    ConversationEventType,
    VisionEvent,
    VisionEventType,
)
from ..session import SessionController

LOGGER = logging.getLogger(__name__)


class InteractionCoordinator:
    def __init__(
        self,
        vision_config: VisionConfig,
        vision_queue: asyncio.Queue[VisionEvent],
        action_queue: asyncio.Queue[ActionIntent],
        conversation_queue: asyncio.Queue[ConversationEvent],
        session_controller: SessionController,
        disarm_listening: Callable[[], None],
        idle_return_to_center_seconds: float,
    ) -> None:
        self.vision_config = vision_config
        self.vision_queue = vision_queue
        self.action_queue = action_queue
        self.conversation_queue = conversation_queue
        self.session_controller = session_controller
        self.disarm_listening = disarm_listening
        self.idle_return_to_center_seconds = max(0.0, idle_return_to_center_seconds)
        self._present_frames = 0
        self._last_welcome_at = -vision_config.welcome_cooldown_seconds
        self._idle_return_task: asyncio.Task[None] | None = None

    async def run(self) -> None:
        while True:
            event = await self.vision_queue.get()
            if event.event_type == VisionEventType.PERSON_PRESENT:
                self._cancel_idle_return()
                await self.session_controller.note_person_present()
                await self._handle_person_present(event)
            else:
                self._present_frames = 0
                self.disarm_listening()
                await self.session_controller.note_person_absent()
                self._schedule_idle_return()
                LOGGER.debug("person absent; stable counter reset")

    async def _handle_person_present(self, event: VisionEvent) -> None:
        self._present_frames += 1
        if self._present_frames < self.vision_config.stable_frames:
            LOGGER.debug(
                "person candidate %s/%s",
                self._present_frames,
                self.vision_config.stable_frames,
            )
            return

        now = monotonic()
        elapsed = now - self._last_welcome_at
        if elapsed < self.vision_config.welcome_cooldown_seconds:
            LOGGER.debug("welcome suppressed by cooldown: %.1fs", elapsed)
            return

        if not await self.session_controller.try_start_welcome():
            LOGGER.debug("welcome ignored by session state")
            return

        self._last_welcome_at = now
        welcome_event = VisionEvent(
            event_type=VisionEventType.WELCOME_TRIGGERED,
            confidence=event.confidence,
            source=event.source,
            cooldown_active=False,
        )
        LOGGER.info("welcome triggered: %s", welcome_event)
        await self.action_queue.put(ActionIntent(name=ActionName.WELCOME_MOTION))
        await self.conversation_queue.put(
            ConversationEvent(
                event_type=ConversationEventType.WELCOME,
                vision_event=welcome_event,
            )
        )

    def _schedule_idle_return(self) -> None:
        self._cancel_idle_return()
        if self.idle_return_to_center_seconds == 0:
            self._idle_return_task = asyncio.create_task(self._send_idle_action())
            return
        self._idle_return_task = asyncio.create_task(self._delayed_idle_return())

    def _cancel_idle_return(self) -> None:
        if self._idle_return_task is not None:
            self._idle_return_task.cancel()
            self._idle_return_task = None

    async def _delayed_idle_return(self) -> None:
        try:
            await asyncio.sleep(self.idle_return_to_center_seconds)
            await self._send_idle_action()
        except asyncio.CancelledError:
            raise

    async def _send_idle_action(self) -> None:
        LOGGER.info("queueing idle return action after person absence")
        await self.action_queue.put(ActionIntent(name=ActionName.IDLE))
