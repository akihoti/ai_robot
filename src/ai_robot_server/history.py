from __future__ import annotations

from collections import OrderedDict
from threading import Lock

ChatMessage = dict[str, str]
SessionKey = tuple[str, str]


class ConversationHistoryStore:
    def __init__(self, *, max_turns: int = 6, max_sessions: int = 128) -> None:
        self.max_turns = max(1, max_turns)
        self.max_sessions = max(1, max_sessions)
        self._history: OrderedDict[SessionKey, list[ChatMessage]] = OrderedDict()
        self._lock = Lock()

    def for_context(self, *, device_id: str, session_id: str) -> list[ChatMessage]:
        key = self._make_key(device_id=device_id, session_id=session_id)
        if key is None:
            return []
        with self._lock:
            messages = self._history.get(key)
            if not messages:
                return []
            self._history.move_to_end(key)
            return [dict(message) for message in messages]

    def record_turn(
        self,
        *,
        device_id: str,
        session_id: str,
        user_text: str,
        assistant_text: str,
    ) -> None:
        key = self._make_key(device_id=device_id, session_id=session_id)
        if key is None:
            return
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text or not assistant_text:
            return

        with self._lock:
            messages = list(self._history.get(key, []))
            messages.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            max_messages = self.max_turns * 2
            self._history[key] = messages[-max_messages:]
            self._history.move_to_end(key)
            while len(self._history) > self.max_sessions:
                self._history.popitem(last=False)

    def clear_session(self, *, device_id: str, session_id: str) -> None:
        key = self._make_key(device_id=device_id, session_id=session_id)
        if key is None:
            return
        with self._lock:
            self._history.pop(key, None)

    @staticmethod
    def _make_key(*, device_id: str, session_id: str) -> SessionKey | None:
        device_id = str(device_id).strip()
        session_id = str(session_id).strip()
        if not device_id or not session_id:
            return None
        return device_id, session_id
