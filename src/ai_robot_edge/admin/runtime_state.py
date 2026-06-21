from __future__ import annotations

import os
import threading
from time import time
from typing import Any


def _now_ms() -> int:
    return int(time() * 1000)


class RuntimeState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue_providers: dict[str, Any] = {}
        self.reset(device_id="", runtime_mode="")

    def reset(self, *, device_id: str, runtime_mode: str) -> None:
        with self._lock:
            now = _now_ms()
            self._queue_providers = {}
            self._data: dict[str, Any] = {
                "process": {
                    "pid": os.getpid(),
                    "started_at_ms": now,
                    "device_id": device_id,
                    "runtime_mode": runtime_mode,
                },
                "session": {
                    "state": "idle",
                    "reason": "startup",
                    "session_id": None,
                    "turn_index": 0,
                    "updated_at_ms": now,
                },
                "listening": {
                    "armed": False,
                    "timeout_ms": None,
                    "updated_at_ms": now,
                },
                "playback": {
                    "active": False,
                    "updated_at_ms": now,
                },
                "queues": {},
                "components": {
                    "tracker_degraded": False,
                    "servo_degraded": False,
                },
                "last_events": {
                    "vision": None,
                    "wake_word": None,
                    "utterance": None,
                    "server_turn": None,
                    "playback_chunk": None,
                    "action": None,
                },
                "counters": {
                    "vision_present": 0,
                    "vision_absent": 0,
                    "wake_word_detected": 0,
                    "utterances": 0,
                    "turns_ok": 0,
                    "turns_failed": 0,
                    "playback_chunks": 0,
                    "playback_interrupts": 0,
                    "action_interrupts": 0,
                },
            }

    def register_queue(self, name: str, provider: Any) -> None:
        with self._lock:
            self._queue_providers[name] = provider

    def record_session(self, snapshot: Any) -> None:
        with self._lock:
            self._data["session"] = {
                "state": str(getattr(snapshot, "state", "unknown")),
                "reason": getattr(snapshot, "reason", ""),
                "session_id": getattr(snapshot, "session_id", None),
                "turn_index": int(getattr(snapshot, "turn_index", 0)),
                "updated_at_ms": _now_ms(),
            }

    def record_listening(self, *, armed: bool, timeout_ms: int | None = None) -> None:
        with self._lock:
            self._data["listening"] = {
                "armed": armed,
                "timeout_ms": timeout_ms,
                "updated_at_ms": _now_ms(),
            }

    def record_vision_event(
        self,
        *,
        event_type: str,
        confidence: float,
        source: str,
    ) -> None:
        with self._lock:
            self._data["last_events"]["vision"] = {
                "event_type": event_type,
                "confidence": confidence,
                "source": source,
                "timestamp_ms": _now_ms(),
            }
            counter_key = "vision_present" if "present" in event_type else "vision_absent"
            self._data["counters"][counter_key] += 1

    def record_wake_word(self) -> None:
        with self._lock:
            self._data["last_events"]["wake_word"] = {
                "timestamp_ms": _now_ms(),
            }
            self._data["counters"]["wake_word_detected"] += 1

    def record_utterance(self, *, request_id: str, duration_ms: int, reason: str) -> None:
        with self._lock:
            self._data["last_events"]["utterance"] = {
                "request_id": request_id,
                "duration_ms": duration_ms,
                "reason": reason,
                "timestamp_ms": _now_ms(),
            }
            self._data["counters"]["utterances"] += 1

    def record_server_turn(
        self,
        *,
        phase: str,
        request_id: str,
        session_id: str | None = None,
        turn_index: int | None = None,
        success: bool | None = None,
        error_message: str = "",
    ) -> None:
        with self._lock:
            self._data["last_events"]["server_turn"] = {
                "phase": phase,
                "request_id": request_id,
                "session_id": session_id,
                "turn_index": turn_index,
                "success": success,
                "error_message": error_message,
                "timestamp_ms": _now_ms(),
            }
            if success is True:
                self._data["counters"]["turns_ok"] += 1
            elif success is False:
                self._data["counters"]["turns_failed"] += 1

    def record_playback_chunk(
        self,
        *,
        bytes_count: int,
        sample_rate: int,
        channels: int,
        media_type: str,
        merged_chunks: int,
    ) -> None:
        with self._lock:
            self._data["last_events"]["playback_chunk"] = {
                "bytes": bytes_count,
                "sample_rate": sample_rate,
                "channels": channels,
                "media_type": media_type,
                "merged_chunks": merged_chunks,
                "timestamp_ms": _now_ms(),
            }
            self._data["counters"]["playback_chunks"] += merged_chunks

    def set_playback_active(self, active: bool) -> None:
        with self._lock:
            self._data["playback"] = {
                "active": active,
                "updated_at_ms": _now_ms(),
            }

    def record_playback_interrupt(self) -> None:
        with self._lock:
            self._data["counters"]["playback_interrupts"] += 1

    def record_action(self, *, name: str, interrupted: bool = False) -> None:
        with self._lock:
            self._data["last_events"]["action"] = {
                "name": name,
                "interrupted": interrupted,
                "timestamp_ms": _now_ms(),
            }
            if interrupted:
                self._data["counters"]["action_interrupts"] += 1

    def set_component_state(self, name: str, degraded: bool) -> None:
        with self._lock:
            self._data["components"][name] = degraded

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            data = {
                key: (value.copy() if isinstance(value, dict) else value)
                for key, value in self._data.items()
            }
            last_events = self._data["last_events"]
            data["last_events"] = {
                key: (value.copy() if isinstance(value, dict) else value)
                for key, value in last_events.items()
            }
            data["counters"] = dict(self._data["counters"])
            data["components"] = dict(self._data["components"])
            data["queues"] = {
                name: self._safe_queue_size(provider)
                for name, provider in self._queue_providers.items()
            }
            return data

    def _safe_queue_size(self, provider: Any) -> int | None:
        try:
            return int(provider())
        except Exception:
            return None


runtime_state = RuntimeState()
