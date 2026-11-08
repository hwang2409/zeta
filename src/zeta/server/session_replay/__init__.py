"""Connection-scoped atomic history and live event joins."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from ..protocol import ErrorDescriptor, ProtocolError

REPLAY_MAX_RECORDS = 256
REPLAY_MAX_BYTES = 4 * 1024 * 1024
REPLAY_SCOPE_MAX_RECORDS = 64
REPLAY_SCOPE_MAX_BYTES = 1 * 1024 * 1024
REPLAY_IMPLIED_FEATURES = frozenset({"session_history_fidelity", "session_history_backward"})


def history_features(features: frozenset[str] | set[str]) -> frozenset[str]:
    """Return the features a connection's history reads and live filters honor."""
    if "session_replay_seq" in features:
        return frozenset(features) | REPLAY_IMPLIED_FEATURES
    return frozenset(features)


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
    size: int = 0


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
    """Own replay stream state and its bounded charge ledger."""

    def __init__(self) -> None:
        self._subscriptions: dict[Any, _Subscription] = {}
        self._pending: dict[Any, PendingReplayJoin] = {}
        self._next_sequences: dict[Any, int] = {}
        self._scopes: dict[ReplayScopeId, _Scope] = {}
        self._next_ordinal = 0
        self._records = 0
        self._bytes = 0

    def _scope(self, scope: ReplayScopeId) -> _Scope:
        current = self._scopes.get(scope)
        if current is None:
            current = _Scope(self._next_ordinal)
            self._next_ordinal += 1
            self._scopes[scope] = current
        return current

    def scopes(self) -> tuple[ReplayScopeId, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self._scopes,
                    *(pending.scope for pending in self._pending.values()),
                    *(subscription.scope for subscription in self._subscriptions.values()),
                )
            )
        )

    def scope_for(self, connection: Any) -> ReplayScopeId | None:
        state = self._subscriptions.get(connection) or self._pending.get(connection)
        return None if state is None else state.scope

    def is_pending(self, connection: Any) -> bool:
        return connection in self._pending

    def stream_id(self, connection: Any) -> str | None:
        subscription = self._subscriptions.get(connection)
        return None if subscription is None else subscription.stream_id

    def _drop_scope(self, scope: ReplayScopeId) -> list[tuple[Any, _Subscription]]:
        current = self._scopes.pop(scope, None)
        if current is not None:
            self._records -= current.records
            self._bytes -= current.bytes
        for connection, pending in tuple(self._pending.items()):
            if pending.scope == scope:
                pending.stale = True
                self._pending.pop(connection, None)
                self._next_sequences.pop(connection, None)
        ended = [
            (connection, subscription)
            for connection, subscription in self._subscriptions.items()
            if subscription.scope == scope
        ]
        for connection, _ in ended:
            self._subscriptions.pop(connection, None)
            self._next_sequences.pop(connection, None)
        return ended

    def invalidate_scope(self, scope: ReplayScopeId) -> list[tuple[Any, _Subscription]]:
        """End every stream of one scope; return the active streams that ended."""
        return self._drop_scope(scope)

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

    def reserve(self, scope: ReplayScopeId, size: int) -> ReplayScopeId | None:
        owner = self._scope(scope)
        if (
            owner.records + 1 > REPLAY_SCOPE_MAX_RECORDS
            or owner.bytes + size > REPLAY_SCOPE_MAX_BYTES
        ):
            return scope
        if (
            self._records + 1 > REPLAY_MAX_RECORDS
            or self._bytes + size > REPLAY_MAX_BYTES
        ):
            return self._largest_scope() or scope
        owner.records += 1
        owner.bytes += size
        owner.charges.append(size)
        self._records += 1
        self._bytes += size
        return None

    def begin_join(self, connection: Any, scope: ReplayScopeId) -> PendingReplayJoin:
        self.connection_closed(connection)
        pending = PendingReplayJoin(connection, scope, secrets.token_urlsafe(9))
        self._next_sequences[connection] = 1
        self._pending[connection] = pending
        return pending

    def finish_join(
        self, connection: Any, history_snapshot: dict[str, object]
    ) -> dict[str, object]:
        pending = self._pending.get(connection)
        if pending is None or pending.stale:
            raise ProtocolError(
                ErrorDescriptor.STALE_EVENT_SEQUENCE,
                "replay continuity was invalidated",
            )
        return {
            **history_snapshot,
            "replay": {"stream_id": pending.stream_id, "history_seq": 0},
        }

    def pending_tail(self, connection: Any) -> list[tuple[Any, dict[str, object]]]:
        pending = self._pending.get(connection)
        return [] if pending is None else list(pending.frames)

    def activate_join(self, connection: Any) -> None:
        pending = self._pending.pop(connection, None)
        if pending is None or pending.stale:
            raise ProtocolError(
                ErrorDescriptor.STALE_EVENT_SEQUENCE,
                "replay continuity was invalidated",
            )
        self._subscriptions[connection] = _Subscription(
            pending.scope, pending.stream_id
        )

    def prepare(
        self, connection: Any, scope: ReplayScopeId
    ) -> PublishedSessionEvent | None:
        target = self._subscriptions.get(connection) or self._pending.get(connection)
        if target is None or target.scope != scope:
            return None
        return PublishedSessionEvent(
            target.stream_id, self._next_sequences.get(connection, 1), scope
        )

    def commit(
        self,
        connection: Any,
        event: Any,
        fields: dict[str, object],
        published: PublishedSessionEvent,
        frame_size: int,
    ) -> PublishedSessionEvent | None:
        current = self.prepare(connection, published.scope)
        if current != published:
            return None
        self._next_sequences[connection] = published.seq + 1
        pending = self._pending.get(connection)
        payload = {
            "stream_id": published.stream_id,
            "seq": published.seq,
            "_replay_scope": published.scope,
            "_replay_size": frame_size,
            **fields,
        }
        if pending is not None:
            pending.frames.append((event, payload))
        return PublishedSessionEvent(
            published.stream_id, published.seq, published.scope, frame_size
        )

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

    def connection_closed(self, connection: Any) -> _Subscription | None:
        """End one connection's streams; return its active stream if one ended."""
        pending = self._pending.pop(connection, None)
        ended = self._subscriptions.pop(connection, None)
        self._next_sequences.pop(connection, None)
        if pending is not None:
            for _, fields in pending.frames:
                self.release(fields["_replay_scope"], fields["_replay_size"])
        return ended
