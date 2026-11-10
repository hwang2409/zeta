"""Connection-scoped atomic history and live event joins."""
from __future__ import annotations

import secrets
from dataclasses import dataclass, field, replace
from typing import Any

from ..protocol import ErrorDescriptor, ProtocolError
from .rows import Covers, OldRow

REPLAY_MAX_RECORDS = 256
REPLAY_MAX_BYTES = 4 * 1024 * 1024
REPLAY_SCOPE_MAX_RECORDS = 64
REPLAY_SCOPE_MAX_BYTES = 1 * 1024 * 1024
REPLAY_IMPLIED_FEATURES = frozenset(
    {"session_history_fidelity", "session_history_backward", "user_message_event"}
)


def history_features(features: frozenset[str] | set[str]) -> frozenset[str]:
    """Return the features a connection's history reads and live filters honor."""
    if "session_replay_seq" in features:
        return frozenset(features) | REPLAY_IMPLIED_FEATURES
    return frozenset(features)


async def join_history_and_subscribe(client: Any, params: dict[str, Any]) -> object:
    """Validate and atomically join a client's session history stream."""
    connections = client.server.connections
    try:
        if set(params) != {"session_id"} or not isinstance(
            params.get("session_id"), str
        ) or not params["session_id"]:
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "session_history_and_subscribe requires session_id",
            )
        session_id = params["session_id"]
        if client.server.runtime.session_id != session_id:
            raise ProtocolError(
                ErrorDescriptor.NO_ACTIVE_SESSION,
                "session is not active",
            )
        connections.begin_replay_join(client, ReplayScopeId.top(session_id))
        read = await connections.read_replay_history(client, session_id)
        return connections.finish_replay_join(client, *read)
    except BaseException:
        connections.invalidate_replay_connection(client, reason="rejoined")
        raise


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
    seq: int | None
    scope: ReplayScopeId
    size: int = 0


@dataclass(frozen=True, slots=True)
class Frontier:
    """The newest row seq and stable id included by the join's history read."""

    seq: int
    reset_generation: int
    row_id: str | None = None


@dataclass(slots=True)
class HeldFrame:
    """A frame buffered for a pending join; seq is assigned at the join's end."""

    event: Any
    fields: dict[str, object]
    covers: Covers
    size: int
    seq: int | None = None


@dataclass(slots=True)
class DeferredFrame:
    """A publish-first replay copy waiting for its history row to install."""

    event: Any
    session_id: str
    fields: dict[str, object]
    required_feature: str | None
    facts: Any
    key: Any
    size: int
    # False once a pressure eviction took the scope's charges; the record stays held
    charged: bool = True


@dataclass(slots=True)
class PendingReplayJoin:
    connection: Any
    scope: ReplayScopeId
    stream_id: str
    reset_generation: int
    frontier_seq: int
    frontier_row_id: str | None
    frames: list[HeldFrame] = field(default_factory=list)
    frontier: Frontier | None = None
    stale: bool = False


@dataclass(slots=True)
class _Subscription:
    scope: ReplayScopeId
    stream_id: str
    frontier: Frontier


class _Decision:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


# a replay frame whose row is in the stream's history, or that cannot be placed
DROP = _Decision("DROP")
EVICT = _Decision("EVICT")


def decide(covers: Covers, frontier: Frontier) -> _Decision | None:
    """Return DROP, EVICT, or None (send) for one frame on one stream."""
    if covers is None:
        return None
    if isinstance(covers, OldRow):
        return DROP if frontier.seq >= covers.bound else EVICT
    return DROP if covers <= frontier.seq else None


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
        # at most one record per publish-first event kind, so two per scope
        self._deferred: dict[ReplayScopeId, dict[Any, DeferredFrame]] = {}
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

    def _drop_scope(
        self, scope: ReplayScopeId, *, keep_deferred: bool
    ) -> list[tuple[Any, _Subscription]]:
        # the scope's charges leave with it; a kept record outlives its streams uncharged
        if keep_deferred:
            for record in self._deferred.get(scope, {}).values():
                record.charged = False
        else:
            self._deferred.pop(scope, None)
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

    def invalidate_scope(
        self, scope: ReplayScopeId, *, keep_deferred: bool
    ) -> list[tuple[Any, _Subscription]]:
        """End every stream of one scope; return the active streams that ended.

        With keep_deferred, held publish-first records stay: their rows still
        install, so a rejoin before the row must get the copy live.
        """
        return self._drop_scope(scope, keep_deferred=keep_deferred)

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

    def defer(self, scope: ReplayScopeId, frame: DeferredFrame) -> None:
        """Keep a charged record; the caller closed any record of the same event."""
        records = self._deferred.setdefault(scope, {})
        assert frame.event not in records
        records[frame.event] = frame

    def deferred(self, scope: ReplayScopeId) -> tuple[DeferredFrame, ...]:
        records = self._deferred.get(scope)
        return () if records is None else tuple(records.values())

    def take_deferred(self, scope: ReplayScopeId, frame: DeferredFrame) -> bool:
        """Remove one record and its charge; False if a scope drop took it first."""
        records = self._deferred.get(scope)
        if records is None or records.get(frame.event) is not frame:
            return False
        del records[frame.event]
        if not records:
            del self._deferred[scope]
        if frame.charged:
            self.release(scope, frame.size)
        return True

    def begin_join(
        self,
        connection: Any,
        scope: ReplayScopeId,
        reset_generation: int,
        frontier_seq: int,
        frontier_row_id: str | None,
    ) -> PendingReplayJoin:
        self.connection_closed(connection)
        pending = PendingReplayJoin(
            connection,
            scope,
            secrets.token_urlsafe(9),
            reset_generation,
            frontier_seq,
            frontier_row_id,
        )
        self._next_sequences[connection] = 1
        self._pending[connection] = pending
        return pending

    def finish_join(
        self,
        connection: Any,
        history_snapshot: dict[str, object],
        frontier_seq: int,
        frontier_row_id: str | None = None,
    ) -> dict[str, object]:
        """Fix the join's frontier; drop held frames its history already shows."""
        pending = self._pending.get(connection)
        if pending is None or pending.stale:
            raise ProtocolError(
                ErrorDescriptor.STALE_EVENT_SEQUENCE,
                "replay continuity was invalidated",
            )
        frontier = Frontier(frontier_seq, pending.reset_generation, frontier_row_id)
        kept: list[HeldFrame] = []
        for held in pending.frames:
            decision = decide(held.covers, frontier)
            if decision is EVICT:
                raise ProtocolError(
                    ErrorDescriptor.STALE_EVENT_SEQUENCE,
                    "replay continuity was invalidated",
                )
            if decision is DROP:
                self.release(pending.scope, held.size)
                continue
            held.seq = self._next_sequences[connection]
            self._next_sequences[connection] = held.seq + 1
            kept.append(held)
        pending.frames = kept
        pending.frontier = frontier
        return {
            **history_snapshot,
            "replay": {"stream_id": pending.stream_id, "history_seq": 0},
        }

    def pending_tail(self, connection: Any) -> list[HeldFrame]:
        pending = self._pending.get(connection)
        return [] if pending is None else list(pending.frames)

    def activate_join(self, connection: Any) -> None:
        pending = self._pending.pop(connection, None)
        if pending is None or pending.stale or pending.frontier is None:
            raise ProtocolError(
                ErrorDescriptor.STALE_EVENT_SEQUENCE,
                "replay continuity was invalidated",
            )
        self._subscriptions[connection] = _Subscription(
            pending.scope, pending.stream_id, pending.frontier
        )

    def frontier_identity(self, connection: Any) -> tuple[int, str | None] | None:
        pending = self._pending.get(connection)
        return None if pending is None else (pending.frontier_seq, pending.frontier_row_id)

    def frontier(self, connection: Any) -> Frontier | None:
        subscription = self._subscriptions.get(connection)
        if subscription is not None:
            return subscription.frontier
        pending = self._pending.get(connection)
        return None if pending is None else pending.frontier

    def refresh_reset_generation(self, connection: Any, reset_generation: int) -> None:
        subscription = self._subscriptions.get(connection)
        if subscription is not None:
            subscription.frontier = replace(
                subscription.frontier, reset_generation=reset_generation
            )
            return
        pending = self._pending.get(connection)
        if pending is not None:
            pending.reset_generation = reset_generation

    def reset_generation(self, connection: Any) -> int | None:
        """The store reset generation a connection's stream or join began at."""
        subscription = self._subscriptions.get(connection)
        if subscription is not None:
            return subscription.frontier.reset_generation
        pending = self._pending.get(connection)
        return None if pending is None else pending.reset_generation

    def pending_stream_id(self, connection: Any) -> str | None:
        pending = self._pending.get(connection)
        return None if pending is None else pending.stream_id

    def prepare(
        self, connection: Any, scope: ReplayScopeId, covers: Covers = None
    ) -> PublishedSessionEvent | _Decision | None:
        """Return the frame's stream position, DROP, EVICT, or None (no stream).

        A frame for a join still reading its history gets no seq yet.
        """
        target = self._subscriptions.get(connection) or self._pending.get(connection)
        if target is None or target.scope != scope:
            return None
        if target.frontier is None:
            return PublishedSessionEvent(target.stream_id, None, scope)
        decision = decide(covers, target.frontier)
        if decision is not None:
            return decision
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
        covers: Covers = None,
    ) -> PublishedSessionEvent | None:
        current = self.prepare(connection, published.scope, covers)
        if current != published:
            return None
        if published.seq is not None:
            self._next_sequences[connection] = published.seq + 1
        pending = self._pending.get(connection)
        if pending is not None:
            pending.frames.append(
                HeldFrame(event, fields, covers, frame_size, published.seq)
            )
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
            for held in pending.frames:
                self.release(pending.scope, held.size)
        return ended
