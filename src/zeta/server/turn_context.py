from __future__ import annotations

from typing import Any

from .protocol import ProtocolError

MAX_BYTES = 4_096


class PendingTurnContexts:
    """Own bounded, one-shot host context for server-started session turns."""

    def __init__(self) -> None:
        self._pending: dict[str, str] = {}

    def set(self, session_id: str | None, params: dict[str, Any]) -> dict[str, object]:
        if session_id is None:
            raise ProtocolError(-32003, "no active session")
        if "text" not in params or (
            params["text"] is not None and not isinstance(params["text"], str)
        ):
            raise ProtocolError(-32602, "text must be a string or null")
        text = params["text"]
        if text is not None and len(text.encode("utf-8")) > MAX_BYTES:
            raise ProtocolError(
                -32602,
                f"text must not exceed {MAX_BYTES} UTF-8 bytes",
            )
        if text is None:
            self._pending.pop(session_id, None)
        else:
            self._pending[session_id] = text
        return {
            "accepted": True,
            "session_id": session_id,
            "pending": text is not None,
        }

    def take(self, session_id: str) -> str | None:
        return self._pending.pop(session_id, None)
