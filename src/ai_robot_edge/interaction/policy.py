from __future__ import annotations

from dataclasses import dataclass

from ..events import ActionIntent, ActionName
from ..session import EdgeSessionState


@dataclass(frozen=True)
class TrackingPolicy:
    mode: str
    should_track: bool
    cadence_divisor: int = 1


def action_priority(intent: ActionIntent | None) -> int:
    if intent is None:
        return -1
    if intent.name in {ActionName.IDLE, ActionName.LOOK_AT_USER}:
        return 3
    if intent.name == ActionName.WELCOME_MOTION:
        return 2
    return 1


def tracking_policy_for_state(state: EdgeSessionState) -> TrackingPolicy:
    return TrackingPolicy(mode=f"active-{state.value}", should_track=True, cadence_divisor=1)


def microphone_should_suppress_during_speaking(
    *,
    suppress_mic_while_speaking: bool,
    speech_interrupt_enabled: bool,
) -> bool:
    return suppress_mic_while_speaking and not speech_interrupt_enabled
