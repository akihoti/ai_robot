from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4
from collections.abc import Callable

LOGGER = logging.getLogger(__name__)


class EdgeSessionState(str, Enum):
    IDLE = "idle"
    TRACKING = "tracking"
    WELCOMING = "welcoming"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    FOLLOWUP_LISTENING = "followup_listening"
    DISENGAGED = "disengaged"


@dataclass(frozen=True)
class SessionSnapshot:
    state: EdgeSessionState
    reason: str
    session_id: str | None
    turn_index: int


class SessionController:
    def __init__(
        self,
        *,
        welcome_once_per_session: bool = True,
        on_update: Callable[[SessionSnapshot], None] | None = None,
    ) -> None:
        self._state = EdgeSessionState.IDLE
        self._reason = "startup"
        self._session_id: str | None = None
        self._turn_index = 0
        self._welcome_sent = False
        self._welcome_once_per_session = welcome_once_per_session
        self._on_update = on_update
        self._lock = asyncio.Lock()

    @property
    def state(self) -> EdgeSessionState:
        return self._state

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            state=self._state,
            reason=self._reason,
            session_id=self._session_id,
            turn_index=self._turn_index,
        )

    def _emit_update(self) -> None:
        if self._on_update is not None:
            self._on_update(self.snapshot())

    async def transition(self, state: EdgeSessionState, reason: str) -> None:
        async with self._lock:
            if self._state == state and self._reason == reason:
                return
            previous = self._state
            self._state = state
            self._reason = reason
            LOGGER.info("session state: %s -> %s (%s)", previous, state, reason)
            self._emit_update()

    async def note_person_present(self) -> None:
        if self._state in {EdgeSessionState.IDLE, EdgeSessionState.DISENGAGED}:
            await self.transition(EdgeSessionState.TRACKING, "person_present")

    async def try_start_welcome(self) -> bool:
        async with self._lock:
            if (
                self._welcome_once_per_session
                and self._session_id is not None
                and self._welcome_sent
            ):
                LOGGER.info("welcome suppressed for active session %s", self._session_id)
                return False
            if self._state not in {
                EdgeSessionState.IDLE,
                EdgeSessionState.TRACKING,
                EdgeSessionState.DISENGAGED,
            }:
                return False
            if self._session_id is None:
                self._session_id = str(uuid4())
                self._turn_index = 0
            self._welcome_sent = True
            previous = self._state
            self._state = EdgeSessionState.WELCOMING
            self._reason = "welcome_triggered"
            LOGGER.info(
                "session state: %s -> %s (%s)",
                previous,
                self._state,
                self._reason,
            )
            self._emit_update()
            return True

    async def note_welcome_playback_finished(self) -> None:
        await self.transition(EdgeSessionState.LISTENING, "welcome_playback_finished")

    async def note_followup_listening(self) -> None:
        await self.transition(
            EdgeSessionState.FOLLOWUP_LISTENING,
            "response_playback_finished",
        )

    async def note_utterance_ready(self) -> None:
        await self.transition(EdgeSessionState.THINKING, "utterance_ready")

    async def note_playback_started(self) -> None:
        await self.transition(EdgeSessionState.SPEAKING, "playback_started")

    async def note_person_absent(self) -> None:
        if self._state in {
            EdgeSessionState.IDLE,
            EdgeSessionState.TRACKING,
            EdgeSessionState.LISTENING,
            EdgeSessionState.FOLLOWUP_LISTENING,
            EdgeSessionState.DISENGAGED,
        }:
            self._session_id = None
            self._turn_index = 0
            self._welcome_sent = False
            await self.transition(EdgeSessionState.DISENGAGED, "person_absent")

    async def recover_to_tracking(self, reason: str) -> None:
        await self.transition(EdgeSessionState.TRACKING, reason)

    async def next_turn_context(self) -> dict[str, int | str]:
        async with self._lock:
            if self._session_id is None:
                self._session_id = str(uuid4())
                self._welcome_sent = False
                self._turn_index = 0
            self._turn_index += 1
            self._emit_update()
            return {
                "session_id": self._session_id,
                "turn_index": self._turn_index,
            }
