from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
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
from ..session import EdgeSessionState, SessionController

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
        arm_presence_listening: Callable[[], None] | None = None,
        queue_local_welcome: Callable[[VisionEvent], Awaitable[None]] | None = None,
        local_welcome_enabled: bool = False,
    ) -> None:
        self.vision_config = vision_config
        self.vision_queue = vision_queue
        self.action_queue = action_queue
        self.conversation_queue = conversation_queue
        self.session_controller = session_controller
        self.disarm_listening = disarm_listening
        self.arm_presence_listening = arm_presence_listening or (lambda: None)
        self.queue_local_welcome = queue_local_welcome
        self.local_welcome_enabled = local_welcome_enabled
        self.idle_return_to_center_seconds = max(0.0, idle_return_to_center_seconds)
        self._present_frames = 0
        self._absent_frames = 0
        self._last_welcome_at = -vision_config.welcome_cooldown_seconds
        self._idle_return_task: asyncio.Task[None] | None = None
        self._idle_return_scheduled = False
        self._idle_return_sent_for_absence = False

    async def run(self) -> None:
        while True:
            event = await self.vision_queue.get()
            if event.event_type == VisionEventType.PERSON_PRESENT:
                self._absent_frames = 0
                self._cancel_idle_return()
                await self.session_controller.note_person_present()
                await self._handle_person_present(event)
            else:
                self._present_frames = 0
                self._absent_frames += 1
                if self._absent_frames < self.vision_config.stable_frames:
                    LOGGER.debug(
                        "person absent candidate %s/%s",
                        self._absent_frames,
                        self.vision_config.stable_frames,
                    )
                    continue
                self._schedule_idle_return()
                LOGGER.debug("person absent stable; idle return scheduled")

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
        if self.local_welcome_enabled:
            await self._handle_local_welcome(event, now=now, elapsed=elapsed)
            return

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

    async def _handle_local_welcome(
        self,
        event: VisionEvent,
        *,
        now: float,
        elapsed: float,
    ) -> None:
        if elapsed < self.vision_config.welcome_cooldown_seconds:
            LOGGER.debug("local welcome suppressed by cooldown: %.1fs", elapsed)
            self._arm_presence_if_session_allows()
            return

        if not await self.session_controller.try_start_welcome():
            LOGGER.debug("local welcome ignored by session state")
            self._arm_presence_if_session_allows()
            return

        self._last_welcome_at = now
        welcome_event = VisionEvent(
            event_type=VisionEventType.WELCOME_TRIGGERED,
            confidence=event.confidence,
            source=event.source,
            cooldown_active=False,
        )
        LOGGER.info("local welcome triggered: %s", welcome_event)
        await self.action_queue.put(ActionIntent(name=ActionName.WELCOME_MOTION))
        if self.queue_local_welcome is not None:
            await self.queue_local_welcome(welcome_event)
        else:
            await self.session_controller.note_welcome_playback_finished()
            self._arm_presence_if_session_allows()

    def _arm_presence_if_session_allows(self) -> None:
        if self.session_controller.state in {
            EdgeSessionState.TRACKING,
            EdgeSessionState.LISTENING,
            EdgeSessionState.FOLLOWUP_LISTENING,
        }:
            self.arm_presence_listening()

    def _schedule_idle_return(self) -> None:
        if self._idle_return_scheduled:
            return
        self._idle_return_scheduled = True
        self._idle_return_task = asyncio.create_task(self._confirm_absence_after_delay())

    def _cancel_idle_return(self) -> None:
        if self._idle_return_task is not None:
            self._idle_return_task.cancel()
            self._idle_return_task = None
        self._idle_return_scheduled = False
        self._idle_return_sent_for_absence = False

    async def _confirm_absence_after_delay(self) -> None:
        try:
            if self.idle_return_to_center_seconds > 0:
                await asyncio.sleep(self.idle_return_to_center_seconds)
            absent_accepted = await self.session_controller.note_person_absent(
                force=self.local_welcome_enabled
            )
            if not absent_accepted:
                self._idle_return_scheduled = False
                return
            self.disarm_listening()
            await self._send_idle_action()
            self._idle_return_sent_for_absence = True
        except asyncio.CancelledError:
            raise
        finally:
            self._idle_return_task = None
            self._idle_return_scheduled = False

    async def _send_idle_action(self) -> None:
        if (
            self._idle_return_sent_for_absence
            and self.session_controller.state == EdgeSessionState.DISENGAGED
        ):
            return
        LOGGER.info("queueing idle return action after person absence")
        await self.action_queue.put(ActionIntent(name=ActionName.IDLE))
