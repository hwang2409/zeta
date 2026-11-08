"""Connection-scoped atomic history and live event joins."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from .protocol import ErrorDescriptor, ProtocolError

REPLAY_MAX_RECORDS = 256
REPLAY_MAX_BYTES = 4 * 1024 * 1024
REPLAY_SCOPE_MAX_RECORDS = 64
REPLAY_SCOPE_MAX_BYTES = 1 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReplayScopeId:
    kind: str
    value: str

    @classmethod
    def top(cls, session_id: str) -> ReplayScopeId:
        return cls("top", session_id)

    @classmethod
    def child(cls, run_id: str) -> ReplayScopeId:
        return cls("child", run_id)


@dataclass(frozen=True, slots=True)
class PublishedSessionEvent:
    stream_id: str
    seq: int
    scope: ReplayScopeId
    size: int


@dataclass(slots=True)
class PendingReplayJoin:
    connection: Any
    scope: ReplayScopeId
    stream_id: str
    frames: list[tuple[Any, dict[str, object]]] = field(default_factory=list)
    stale: bool = False


@dataclass(slots=True)
class _Subscription:
    scope: ReplayScopeId
    stream_id: str


@dataclass(slots=True)
class _Scope:
    ordinal: int
    records: int = 0
    bytes: int = 0
    charges: list[int] = field(default_factory=list)


class SessionReplay:
    """Owns frontier, sequence, join, and bounded replay accounting."""

    def __init__(self) -> None:
        self._subscriptions: dict[Any, _Subscription] = {}
        self._pending: dict[Any, PendingReplayJoin] = {}
        self._next_sequences: dict[Any, int] = {}
        self._scopes: dict[ReplayScopeId, _Scope] = {}
        self._next_ordinal = 0
        self._records = 0
        self._bytes = 0
        self._invalidated: set[ReplayScopeId] = set()

    def _scope(self, scope: ReplayScopeId) -> _Scope:
        current = self._scopes.get(scope)
        if current is None:
            current = _Scope(self._next_ordinal)
            self._next_ordinal += 1
            self._scopes[scope] = current
        return current

    def _invalidate_scope(self, scope: ReplayScopeId) -> None:
        self._invalidated.add(scope)
        self._drop_scope(scope)

    def take_invalidated(self) -> set[ReplayScopeId]:
        invalidated, self._invalidated = self._invalidated, set()
        return invalidated

    def _drop_scope(self, scope: ReplayScopeId) -> None:
        current = self._scopes.pop(scope, None)
        if current is not None:
            self._records -= current.records
            self._bytes -= current.bytes
        for connection, pending in tuple(self._pending.items()):
            if pending.scope == scope:
                pending.stale = True
                self._pending.pop(connection, None)
        for connection, subscription in tuple(self._subscriptions.items()):
            if subscription.scope == scope:
                self._subscriptions.pop(connection, None)

    def _largest_scope(self) -> ReplayScopeId | None:
        if not self._scopes:
            return None
        return max(
            self._scopes,
            key=lambda scope: (
                self._scopes[scope].bytes,
                self._scopes[scope].records,
                -self._scopes[scope].ordinal,
            ),
        )

    def _reserve(self, scope: ReplayScopeId, size: int) -> bool:
        owner = self._scope(scope)
        if (
            owner.records + 1 > REPLAY_SCOPE_MAX_RECORDS
            or owner.bytes + size > REPLAY_SCOPE_MAX_BYTES
        ):
            self._invalidate_scope(scope)
            return False
        while (
            self._records + 1 > REPLAY_MAX_RECORDS
            or self._bytes + size > REPLAY_MAX_BYTES
        ):
            victim = self._largest_scope()
            if victim is None:
                return False
            self._invalidate_scope(victim)
            if victim == scope:
                return False
        owner = self._scope(scope)
        owner.records += 1
        owner.bytes += size
        owner.charges.append(size)
        self._records += 1
        self._bytes += size
        return True

    def stream_id(self, connection: Any) -> str | None:
        subscription = self._subscriptions.get(connection)
        return None if subscription is None else subscription.stream_id

    def begin_join(self, connection: Any, scope: ReplayScopeId) -> PendingReplayJoin:
        self.connection_closed(connection)
        pending = PendingReplayJoin(connection, scope, secrets.token_urlsafe(9))
        self._next_sequences[connection] = 1
        self._pending[connection] = pending
        return pending

    def finish_join(self, connection: Any, history_snapshot: dict[str, object]) -> dict[str, object]:
        pending = self._pending.get(connection)
        if pending is None or pending.stale:
            raise ProtocolError(ErrorDescriptor.STALE_EVENT_SEQUENCE, "replay continuity was invalidated")
        return {**history_snapshot, "replay": {"stream_id": pending.stream_id, "history_seq": 0}}

    def pending_tail(self, connection: Any) -> list[tuple[Any, dict[str, object]]]:
        pending = self._pending.get(connection)
        return [] if pending is None else list(pending.frames)

    def activate_join(self, connection: Any) -> list[tuple[Any, dict[str, object]]]:
        pending = self._pending.pop(connection, None)
        if pending is None or pending.stale:
            raise ProtocolError(ErrorDescriptor.STALE_EVENT_SEQUENCE, "replay continuity was invalidated")
        self._subscriptions[connection] = _Subscription(pending.scope, pending.stream_id)
        return pending.frames

    def publish(self, connection: Any, scope: ReplayScopeId, event: Any, fields: dict[str, object], frame_size: int) -> PublishedSessionEvent | None:
        subscription = self._subscriptions.get(connection)
        pending = self._pending.get(connection)
        target = subscription if subscription is not None else pending
        if target is None or target.scope != scope or not self._reserve(scope, frame_size):
            return None
        seq = self._next_sequences.get(connection, 1)
        self._next_sequences[connection] = seq + 1
        payload = {
            "stream_id": target.stream_id,
            "seq": seq,
            "_replay_scope": scope,
            "_replay_size": frame_size,
            **fields,
        }
        if pending is not None:
            pending.frames.append((event, payload))
        return PublishedSessionEvent(target.stream_id, seq, scope, frame_size)

    def release(self, scope: ReplayScopeId, size: int) -> None:
        owner = self._scopes.get(scope)
        if owner is None:
            return
        try:
            owner.charges.remove(size)
        except ValueError:
            return
        owner.records -= 1
        owner.bytes -= size
        self._records -= 1
        self._bytes -= size
        if owner.records <= 0:
            self._scopes.pop(scope, None)

    def connection_closed(self, connection: Any) -> None:
        pending = self._pending.pop(connection, None)
        subscription = self._subscriptions.pop(connection, None)
        self._next_sequences.pop(connection, None)
        for item in (pending, subscription):
            if item is None:
                continue
            for _, fields in item.frames if pending is item else item.held:
                self.release(
                    fields["_replay_scope"], fields["_replay_size"]
                )

    def session_replaced(self) -> None:
        for pending in self._pending.values():
            pending.stale = True
        self._pending.clear()
        self._subscriptions.clear()
        for scope in tuple(self._scopes):
            self._invalidate_scope(scope)
        self._scopes.clear()
        self._records = self._bytes = 0
