"""Replay output transitions of the one connection output owner."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from .. import ergonomics
from ..protocol import EventDescriptor, _Frame
from . import ReplayScopeId, _Subscription, history_features


@dataclass
class _OutputFrame:
    payload: _Frame
    future: asyncio.Future[bool] | None = None
    replay_stream_id: str | None = None
    replay_scope: ReplayScopeId | None = None
    replay_size: int = 0


class ReplayOutput:
    """ConnectionRegistry's replay transitions: reserve, evict, admit, discard, release."""

    def _publish_observer_replay(
        self,
        event: EventDescriptor,
        session_id: str | None,
        required_feature: str | None,
        fields: dict[str, object],
    ) -> None:
        """Send a controller event to observers that joined its replay scope."""
        for observer in tuple(self._observers):
            if not self._feature_rejected(observer, event, required_feature, fields):
                self._publish_replay(observer, event, session_id, fields)

    def begin_replay_join(self, connection: Any, scope: ReplayScopeId) -> None:
        self.invalidate_replay_connection(connection, reason="rejoined")
        self.replay.begin_join(connection, scope)

    def invalidate_replay_connection(
        self,
        connection: Any,
        stream_id: str | None = None,
        *,
        reason: str | None = None,
    ) -> None:
        """End one connection's replay stream; with stream_id, only that generation.

        Without a reason the connection is closing or failed, so it gets no end event.
        """
        if stream_id is not None and self.replay.stream_id(connection) != stream_id:
            return
        ended = self.replay.connection_closed(connection)
        self._discard_connection_replay(connection)
        if ended is not None and reason is not None:
            self._send_stream_ended(connection, ended, reason)

    def invalidate_replay_scope(self, scope: ReplayScopeId, reason: str) -> None:
        ended = self.replay.invalidate_scope(scope)
        self._discard_replay_scope(scope)
        for connection, subscription in ended:
            self._send_stream_ended(connection, subscription, reason)

    def _send_stream_ended(
        self, connection: Any, ended: _Subscription, reason: str
    ) -> None:
        """The one place an active stream's end is announced, once per stream_id.

        Callers discard the stream's queued frames first, so the event follows every
        frame already written or in flight and precedes any later frame.
        """
        state = self._connections.get(connection)
        if state is None:
            return
        frame = connection.codec.notification(
            EventDescriptor.REPLAY_STREAM_ENDED,
            ended.scope.value,
            stream_id=ended.stream_id,
            reason=reason,
        )
        if frame is not None:
            self._publish_observer(state, frame)

    def replay_session_replaced(self) -> None:
        for scope in self.replay.scopes():
            self.invalidate_replay_scope(scope, "session_replaced")

    def _discard_connection_replay(self, connection: Any) -> None:
        state = self._connections.get(connection)
        if state is None or state.output is None:
            return
        output = state.output
        kept: deque[_OutputFrame] = deque()
        while output.queue:
            item = output.queue.popleft()
            if item.replay_scope is not None:
                self._finish_frame_accounting(output, item)
            else:
                kept.append(item)
        output.queue = kept
        self._assert_retained_accounting(output)

    def _discard_replay_scope(self, scope: ReplayScopeId) -> None:
        for state in tuple(self._connections.values()):
            output = state.output
            if output is None:
                continue
            kept: deque[_OutputFrame] = deque()
            while output.queue:
                item = output.queue.popleft()
                if item.replay_scope == scope:
                    self._finish_frame_accounting(output, item)
                else:
                    kept.append(item)
            output.queue = kept
            self._assert_retained_accounting(output)

    def _publish_replay(
        self,
        state: Any,
        event: EventDescriptor,
        session_id: str | None,
        fields: dict[str, object],
    ) -> bool:
        if "session_replay_seq" not in state.features or session_id is None:
            return False
        scope = ReplayScopeId.top(session_id)
        prepared = self.replay.prepare(state.connection, scope)
        if prepared is None or state.output is None:
            return False
        payload = {"stream_id": prepared.stream_id, "seq": prepared.seq, **fields}
        frame = state.connection.codec.notification(event, session_id, **payload)
        if frame is None:
            return False
        while (victim := self.replay.reserve(scope, len(frame))) is not None:
            self.invalidate_replay_scope(victim, "evicted")
            prepared = self.replay.prepare(state.connection, scope)
            if prepared is None:
                return True
        published = self.replay.commit(
            state.connection,
            event,
            {**fields, "_session_id": session_id},
            prepared,
            len(frame),
        )
        if published is None:
            self.replay.release(scope, len(frame))
            return True
        if self.replay.is_pending(state.connection):
            return True
        if not self._admit(state, frame):
            self.replay.release(scope, published.size)
            self.invalidate_replay_connection(state.connection)
            return True
        self._enqueue(
            state.output,
            _OutputFrame(
                frame,
                replay_stream_id=published.stream_id,
                replay_scope=published.scope,
                replay_size=published.size,
            ),
        )
        return True

    async def admit_join_response(self, connection: Any, frame: _Frame) -> bool:
        state = self._connections.get(connection)
        if state is None or state.output is None:
            self.invalidate_replay_connection(connection)
            return False
        output = state.output
        if output.phase not in {"claimed_buffering", "flushing", "ready"}:
            self.invalidate_replay_connection(connection)
            return False
        held = self.replay.pending_tail(connection)
        items: list[_OutputFrame] = [_OutputFrame(frame)]
        for event, fields in held:
            public = {
                key: value
                for key, value in fields.items()
                if key not in {"_session_id", "_replay_scope", "_replay_size"}
            }
            payload = connection.codec.notification(
                event, fields["_session_id"], **public
            )
            if payload is None:
                self.invalidate_replay_connection(connection)
                return False
            items.append(
                _OutputFrame(
                    payload,
                    replay_stream_id=public["stream_id"],
                    replay_scope=fields["_replay_scope"],
                    replay_size=fields["_replay_size"],
                )
            )
        if not self._fits(
            output, sum(len(item.payload) for item in items), len(items)
        ):
            self.invalidate_replay_connection(connection)
            self._begin_overflow(state, enqueue_control=True)
            return False
        for item in items:
            self._enqueue(output, item, wake=False)
        self.replay.activate_join(connection)
        # the response is not a stream frame: invalidation keeps it, its failure ends the stream
        items[0].replay_stream_id = self.replay.stream_id(connection)
        output.wake.set()
        future = self.loop.create_future()
        items[0].future = future
        self._begin_caller(output)
        try:
            return await asyncio.shield(future)
        finally:
            self._finish_caller(output)

    def _fail_replay_frame(self, state: Any, item: _OutputFrame) -> None:
        stream_id = item.replay_stream_id
        if stream_id is None or self.replay.stream_id(state.connection) != stream_id:
            return
        # fail closed: a failed connection cannot reliably get an end event; it rejoins
        self.invalidate_replay_connection(state.connection)
        state.connection.writer.close()

    async def read_replay_history(
        self, connection: Any, session_id: str
    ) -> dict[str, object]:
        """Read the newest history page of the pinned active store off-loop."""
        store = ergonomics.active(connection.server.runtime, {"session_id": session_id})
        pinned = SimpleNamespace(opened=SimpleNamespace(store=store))
        features = history_features(self._connections[connection].features)
        result = await self.run_read(
            connection,
            lambda: ergonomics.history(
                pinned,
                {"session_id": session_id, "before": None},
                include_delivery_id={
                    "delivery_id", "user_message_event", "user_message_delivery_id"
                } <= features,
                backward_enabled="session_history_backward" in features,
                fidelity="session_history_fidelity" in features,
                content_read="content_read" in features,
                served_turn_end="served_turn_end" in features,
            ),
        )
        assert isinstance(result, dict)
        return result
