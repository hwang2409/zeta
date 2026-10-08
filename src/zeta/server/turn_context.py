from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from ..protocol.types import StreamEvent, StreamEventType
from .protocol import ProtocolError

if TYPE_CHECKING:
    from ..runtime.loop.agent import AgentLoop
    from .runtime import ServerRuntime

MAX_BYTES = 4_096


class TurnContextDelivery(TypedDict, total=False):
    turn_context: str
    on_turn_context_persisted: Callable[[], None]


class _EventSink(Protocol):
    def __call__(
        self, event: StreamEvent, *, session_id: str
    ) -> Awaitable[None]: ...


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

    def set_request(
        self,
        features: set[str],
        runtime: ServerRuntime,
        params: dict[str, Any],
    ) -> dict[str, object]:
        """Validate negotiation and set context for the active session."""
        if "turn_context" not in features:
            raise ProtocolError(
                -32601, "set_turn_context requires negotiated turn_context feature"
            )
        session_id = runtime.session_id if runtime.opened is not None else None
        return self.set(session_id, params)

    def add_request(self, requests: list[str], features: set[str]) -> None:
        """Advertise the request only when its feature was negotiated."""
        if "turn_context" in features:
            requests.append("set_turn_context")

    async def run_notification_turn(
        self,
        session_id: str,
        loop: AgentLoop,
        emit: _EventSink,
    ) -> tuple[bool, StreamEvent | None]:
        """Deliver claimed context and classify notification-turn events."""
        success = True
        agent_end = None
        with self.delivery(session_id) as context:
            async for event in loop.run_notification_turn(**context):
                if event.type is StreamEventType.ERROR:
                    success = False
                if event.type is StreamEventType.AGENT_END:
                    agent_end = event
                else:
                    await emit(event, session_id=session_id)
        return success, agent_end

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
