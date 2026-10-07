from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypedDict

from .protocol import ProtocolError

MAX_BYTES = 4_096


class TurnContextDelivery(TypedDict, total=False):
    turn_context: str
    on_turn_context_persisted: Callable[[], None]


@dataclass(frozen=True)
class _PendingContext:
    version: int
    text: str
    claimed: bool = False


class PendingTurnContexts:
    """Own bounded, one-shot host context for server-started session turns."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingContext] = {}
        self._next_version = 1

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
            self._pending[session_id] = _PendingContext(self._next_version, text)
            self._next_version += 1
        return {
            "accepted": True,
            "session_id": session_id,
            "pending": text is not None,
        }

    @contextmanager
    def delivery(self, session_id: str) -> Iterator[TurnContextDelivery]:
        """Release an uncommitted claim when its notification turn ends."""

        claim = self.claim(session_id)
        if claim is None:
            yield {}
            return
        version, text = claim
        try:
            yield {
                "turn_context": text,
                "on_turn_context_persisted": lambda: self.commit(session_id, version),
            }
        finally:
            self.release(session_id, version)

    def claim(self, session_id: str) -> tuple[int, str] | None:
        """Reserve the current value until its durable message is committed."""

        pending = self._pending.get(session_id)
        if pending is None or pending.claimed:
            return None
        self._pending[session_id] = _PendingContext(
            pending.version, pending.text, claimed=True
        )
        return pending.version, pending.text

    def commit(self, session_id: str, version: int) -> None:
        """Consume a claim without disturbing a value set after it."""

        pending = self._pending.get(session_id)
        if pending is not None and pending.version == version and pending.claimed:
            del self._pending[session_id]

    def release(self, session_id: str, version: int) -> None:
        """Make a failed claim available unless a newer value replaced it."""

        pending = self._pending.get(session_id)
        if pending is not None and pending.version == version and pending.claimed:
            self._pending[session_id] = _PendingContext(version, pending.text)
