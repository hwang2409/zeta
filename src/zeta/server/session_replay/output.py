"""Replay output transitions of the one connection output owner."""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..handlers import validation
from ..handlers.history import replay_history
from ..protocol import EVENT_ACCESS, Access, EventDescriptor, FrameCodec, _Frame
from . import (
    DROP,
    EVICT,
    DeferredFrame,
    ReplayScopeId,
    _Subscription,
    history_features,
    rows,
)

logger = logging.getLogger(__name__)


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
        covers: rows.Covers,
    ) -> None:
        """Send an event to observers that joined its replay scope."""
        if EVENT_ACCESS.get(event, Access.READ) is not Access.READ:
            return
        for observer in tuple(self._observers):
            if not self._feature_rejected(observer, event, required_feature, fields):
                self._publish_replay(
                    observer,
                    event,
                    session_id,
                    self._fields_for_connection(observer, event, fields),
                    covers,
                )

    def _replay_covers(
        self,
        event: EventDescriptor,
        session_id: str | None,
        fields: dict[str, object],
        replay_row: rows.RowFacts | None,
        required_feature: str | None = None,
    ) -> rows.Covers | DeferredFrame:
        """Find the history row one publication presents, once for every stream.

        Run first for every replay admission: deferred copies whose rows have
        installed go out before this frame. A publish-first frame returns its
        deferred record instead: its replay copies wait for the row.
        """
        if session_id is None:
            return None
        self.release_deferred_replay(session_id)
        if rows.ROUTES.get((event, "")) is rows.LIVE:
            return None
        scope = ReplayScopeId.top(session_id)
        store = self._replay_store(scope)
        # a publish-first record is kept with no stream: a join can start before its row
        if store is None and event not in rows.PUBLISH_FIRST_EVENTS:
            return None
        facts = replay_row or rows.RowFacts()
        try:
            source = rows.route(event, fields, facts)
        except rows.UnroutedEvent as exc:
            logger.error("replay event has no route table entry: %s", exc)
            self.invalidate_replay_scope(scope, "evicted")
            return None
        if isinstance(source, rows.PublishFirst) and facts.seq is not None:
            return self._defer_replay(
                scope, DeferredFrame(
                    event, session_id, dict(fields), required_feature, facts,
                    source.key(facts), FrameCodec().serialized_size(fields),
                )
            )
        if store is None:
            return None
        return rows.covers(event, source, store, fields, facts)

    def _defer_replay(self, scope: ReplayScopeId, frame: DeferredFrame) -> DeferredFrame:
        """Hold one charged record per publish-first event; non-replay output is not held."""
        for old in self.replay.deferred(scope):
            # supersede: the release check just ran and found no row for it
            if old.event is frame.event and self.replay.take_deferred(scope, old):
                self._admit_deferred(old, None)
        while (victim := self.replay.reserve(scope, frame.size)) is not None:
            self.invalidate_replay_scope(victim, "evicted")
            if victim == scope:
                # its row still installs: hold it uncharged, like the scope's kept records
                frame.charged = False
                break
        self.replay.defer(scope, frame)
        return frame

    def close_deferred_replay_when_done(
        self, session_id: str, task: asyncio.Task[Any]
    ) -> None:
        """Close the records held when a turn task starts once that task ends.

        A task canceled before its first step runs no pump end, so its echo would
        wait for a later row and could match another send's row.
        """
        held = self.replay.deferred(ReplayScopeId.top(session_id))
        if held:
            task.add_done_callback(
                lambda _: self.release_deferred_replay(session_id, close=True, only=held)
            )

    def release_deferred_replay(
        self,
        session_id: str,
        *,
        close: bool = False,
        only: tuple[DeferredFrame, ...] | None = None,
    ) -> None:
        """Admit deferred copies whose rows installed, before any later frame.

        With close (a turn's end), a record with no row is admitted live-only:
        its row will never install, so it is provisional across a stream end.
        With only, consider just those records.
        """
        scope = ReplayScopeId.top(session_id)
        records = self.replay.deferred(scope)
        if only is not None:
            records = tuple(r for r in records if any(r is h for h in only))
        if not records:
            return
        store = self._replay_store(scope)
        ready: list[tuple[int | None, DeferredFrame]] = []
        for record in records:
            seq = None if store is None else store.recent_row_seq(record.key)
            if seq is None and store is not None and record.key.seq + 1 < store.recent_row_floor():
                # more rows than the lookup window since the mark: fail closed
                for held in records:
                    self.replay.take_deferred(scope, held)
                self.invalidate_replay_scope(scope, "evicted")
                return
            if seq is not None or close:
                ready.append((seq, record))
        # rows go out in install order; live-only copies after them
        for seq, record in sorted(ready, key=lambda item: (item[0] is None, item[0] or 0)):
            if self.replay.take_deferred(scope, record):
                self._admit_deferred(record, seq)

    def _admit_deferred(self, record: DeferredFrame, covers: rows.Covers) -> None:
        """Send one deferred copy to every replay stream of its scope."""
        for state in (self._controller, *self._observers):
            if state is None or (
                state is self._controller and self._controller_output_paused
            ):
                continue
            if not self._feature_rejected(
                state, record.event, record.required_feature, record.fields
            ):
                self._publish_replay(
                    state,
                    record.event,
                    record.session_id,
                    self._fields_for_connection(state, record.event, record.fields),
                    covers,
                )

    def _replay_store(self, scope: ReplayScopeId) -> Any:
        """The scope's store while a replay stream or join needs it, else None."""
        for connection in self._connections:
            if self.replay.scope_for(connection) == scope:
                runtime = connection.server.runtime
                if runtime.opened is not None and runtime.session_id == scope.value:
                    return runtime.opened.store
        return None

    def begin_replay_join(self, connection: Any, scope: ReplayScopeId) -> None:
        self.invalidate_replay_connection(connection, reason="rejoined")
        store = validation.active(connection.server.runtime, {"session_id": scope.value})
        self.replay.begin_join(connection, scope, store.reset_generation())

    def finish_replay_join(
        self,
        connection: Any,
        history: dict[str, object],
        frontier_seq: int,
        frontier_row_id: str | None,
    ) -> dict[str, object]:
        scope = self.replay.scope_for(connection)
        if scope is not None and not self._frontier_holds(connection, scope):
            self.replay.mark_pending_stale(connection)
        return self.replay.finish_join(connection, history, frontier_seq, frontier_row_id)

    def _frontier_holds(
        self,
        connection: Any,
        scope: ReplayScopeId,
    ) -> bool:
        """Check a generation change against the stream or pending join."""
        generation = self.replay.reset_generation(connection)
        store = self._replay_store(scope)
        if generation is None or store is None:
            return generation is None
        current = store.reset_generation()
        if current == generation:
            return True
        if self.replay.is_pending(connection):
            return False
        frontier = self.replay.frontier(connection)
        if frontier is None or frontier.row_id is None:
            return False
        if store.row_id_at(frontier.seq) != frontier.row_id:
            return False
        self.replay.refresh_reset_generation(connection, current)
        return True

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
        # a replaced session renumbers rows, so its held records' marks are void
        ended = self.replay.invalidate_scope(
            scope, keep_deferred=reason != "session_replaced"
        )
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
        covers: rows.Covers,
    ) -> bool:
        """Admit one frame to a replay stream; False when the caller sends it plainly."""
        if "session_replay_seq" not in state.features or session_id is None:
            return False
        scope = ReplayScopeId.top(session_id)
        connection = state.connection
        if covers is not None and not self._frontier_holds(connection, scope):
            if self.replay.is_pending(connection):
                self.replay.mark_pending_stale(connection)
            else:
                self.invalidate_replay_scope(scope, "session_replaced")
            return False
        prepared = self.replay.prepare(connection, scope, covers)
        if prepared is DROP:
            return True
        if prepared is EVICT:
            self.invalidate_replay_connection(connection, reason="evicted")
            return False
        if prepared is None or state.output is None:
            return False
        # a pending join's frames get their seq when the join ends
        position = {"stream_id": prepared.stream_id}
        if prepared.seq is not None:
            position["seq"] = prepared.seq
        frame = connection.codec.notification(event, session_id, **position, **fields)
        if frame is None:
            return False
        while (victim := self.replay.reserve(scope, len(frame))) is not None:
            self.invalidate_replay_scope(victim, "evicted")
            prepared = self.replay.prepare(connection, scope, covers)
            if prepared is None:
                return True
        published = self.replay.commit(
            connection,
            event,
            {**fields, "_session_id": session_id},
            prepared,
            len(frame),
            covers,
        )
        if published is None:
            self.replay.release(scope, len(frame))
            return True
        if self.replay.is_pending(connection):
            return True
        if not self._admit(state, frame):
            self.replay.release(scope, published.size)
            self.invalidate_replay_connection(connection)
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
        stream_id = self.replay.pending_stream_id(connection)
        items: list[_OutputFrame] = [_OutputFrame(frame)]
        for frame_state in held:
            public = {
                key: value
                for key, value in frame_state.fields.items()
                if key != "_session_id"
            }
            payload = connection.codec.notification(
                frame_state.event,
                frame_state.fields["_session_id"],
                stream_id=stream_id,
                seq=frame_state.seq,
                **public,
            )
            if payload is None:
                self.invalidate_replay_connection(connection)
                return False
            items.append(
                _OutputFrame(
                    payload,
                    replay_stream_id=stream_id,
                    replay_scope=self.replay.scope_for(connection),
                    replay_size=frame_state.size,
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
    ) -> tuple[dict[str, object], int, str | None]:
        """Read the newest history page and its stable frontier row off-loop."""
        store = validation.active(connection.server.runtime, {"session_id": session_id})
        pinned = SimpleNamespace(opened=SimpleNamespace(store=store))
        features = history_features(self._connections[connection].features)
        def read() -> tuple[dict[str, object], int, str | None]:
            history, frontier_seq, frontier_row_id = replay_history(
                pinned,
                include_delivery_id={
                    "delivery_id", "user_message_event", "user_message_delivery_id"
                } <= features,
                fidelity="session_history_fidelity" in features,
                content_read="content_read" in features,
                served_turn_end="served_turn_end" in features,
            )
            return history, frontier_seq, frontier_row_id

        result = await self.run_read(connection, read)
        assert isinstance(result, tuple)
        return result
