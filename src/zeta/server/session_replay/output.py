"""Replay output transitions of the one connection output owner."""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .. import event_contract
from ..handlers import lifecycle, validation
from ..handlers.history import replay_history
from ..protocol import (
    EVENT_ACCESS,
    Access,
    ErrorDescriptor,
    EventDescriptor,
    FrameCodec,
    ProtocolError,
    _Frame,
)
from . import (
    DROP,
    EVICT,
    DeferredFrame,
    ReplayScopeId,
    _Subscription,
    history_features,
    rows,
)

if TYPE_CHECKING:
    from ..session_controllers import SessionUse

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

    def _publish_replay_subscribers(
        self,
        event: EventDescriptor,
        session_id: str | None,
        required_feature: str | None,
        fields: dict[str, object],
        covers: rows.Covers,
        replay_row: rows.RowFacts | None = None,
        *,
        exclude: Any = None,
        scope: ReplayScopeId | None = None,
        legacy_observers: bool = False,
        feature_fields: event_contract.FeatureFields | None = None,
    ) -> None:
        replay_allowed = EVENT_ACCESS.get(event, Access.READ) is Access.READ
        if not replay_allowed and not legacy_observers:
            return
        for state in tuple(self._connections.values()):
            connection = state.connection
            if (connection is exclude or state.paused or state.role is None
                    or self._feature_rejected(state, event, required_feature, fields)):
                continue
            shaped = self._fields_for_connection(state, event, fields, feature_fields=feature_fields)
            if self.replay.scope_for(connection) is not None and self._publish_replay(
                state, event, session_id, shaped, covers, scope=scope, replay_row=replay_row,
            ):
                continue
            if legacy_observers and state.role == "observer" and state.output is not None:
                frame = connection.codec.notification(event, session_id, **shaped)
                if frame is not None and state.output.phase == "ready":
                    self._publish_local(state, state.output, frame, wait_for_write=False)

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
        self._publish_replay_subscribers(
            record.event, record.session_id, record.required_feature, record.fields, covers,
        )

    def _replay_store(self, scope: ReplayScopeId) -> Any:
        """The scope's store while a replay stream or join needs it, else None."""
        if scope.kind == "child":
            return self.child_runs.live_store(scope.value)
        for connection in self._connections:
            if self.replay.scope_for(connection) == scope:
                runtime = self.replay.controller_for(connection).runtime
                if runtime.opened is not None and runtime.session_id == scope.value:
                    return runtime.opened.store
        return None

    async def begin_replay_join(
        self, connection: Any, scope: ReplayScopeId, controller: SessionUse
    ) -> None:
        """Begin a join on ``controller``, the use the request selected; the
        stream keeps its record, not the use."""
        if self.frozen:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        if scope.kind == "child":
            store = self.child_runs.live_store(scope.value)
        else:
            store = validation.active(controller.runtime, {"session_id": scope.value})
        self.invalidate_replay_connection(connection, reason="rejoined")
        if connection.role == "observer" and connection.server.controllers.controller_mode == "multi":
            self.require_observer_room(controller.runtime.session_id)
        pending = self.replay.begin_join(
            connection,
            scope,
            None if store is None else store.reset_generation(),
            {},
            controller.record,
        )
        pending.status = await lifecycle.live_status_baseline(
            connection, controller
        )

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
        elif (
            scope is not None
            and scope.kind == "top"
            and "in_flight_replay" in self._connections[connection].features
        ):
            self._hold_in_flight(connection, scope)
        return self.replay.finish_join(connection, history, frontier_seq, frontier_row_id)

    def _hold_in_flight(self, connection: Any, scope: ReplayScopeId) -> None:
        """Hold the whole in-flight state in place of the live frames it carries.

        A pending join never holds those live frames (``_publish_replay``). This
        runs in the same loop step as the join's seq assignment, so each frame
        published before it is in these frames and each one after gets a later seq.
        """
        state = self._connections[connection]
        stream_id = self.replay.pending_stream_id(connection)
        for event, fields in self.replay.controller_for(connection).control.in_flight.frames(scope.value):
            shaped = self._fields_for_connection(state, event, fields)
            # the largest seq bounds the frame size the stream sends
            frame = connection.codec.notification(
                event, scope.value, stream_id=stream_id, seq=2**53, **shaped
            )
            if frame is None:
                self.replay.mark_pending_stale(connection)
                return
            while (victim := self.replay.reserve(scope, len(frame))) is not None:
                self.invalidate_replay_scope(victim, "evicted")
                if not self.replay.is_pending(connection):
                    return
            self.replay.hold(
                connection, event, scope.value, shaped, len(frame)
            )

    def _frontier_holds(
        self,
        connection: Any,
        scope: ReplayScopeId,
    ) -> bool:
        """Check a generation change against the stream or pending join."""
        generation = self.replay.reset_generation(connection)
        store = self._replay_store(scope)
        if generation is None:
            return True
        if store is None:
            # the child scope's store is gone only because the run ended; an ended
            # run's rows are frozen and cannot renumber, so the stream holds until it
            # ends. a replaced or evicted scope is invalidated directly, not here.
            return scope.kind == "child"
        current = store.reset_generation()
        if current == generation:
            return True
        if self.replay.is_pending(connection):
            return False
        frontier = self.replay.frontier(connection)
        if frontier is None:
            return False
        if frontier.seq == 0:
            self.replay.refresh_reset_generation(connection, current)
            return True
        if (
            frontier.row_id is None
            or store.row_id_at(frontier.seq) != frontier.row_id
        ):
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
        self.publish_connection(
            connection, EventDescriptor.REPLAY_STREAM_ENDED, ended.scope.value,
            stream_id=ended.stream_id, reason=reason,
        )

    def clear_replay_scopes(self) -> None:
        for scope in self.replay.scopes():
            self.replay.invalidate_scope(scope, keep_deferred=False)

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

    def _member_scope(
        self,
        connection: Any,
        session_id: str,
        fields: dict[str, object],
        facts: rows.RowFacts | None,
    ) -> ReplayScopeId | None:
        scope = self.replay.scope_for(connection)
        if scope is None:
            return None
        if scope.kind == "top":
            return scope if scope.value == session_id else None
        data = fields.get("data")
        agent = data.get("agent_instance_id") if isinstance(data, dict) else None
        agent = agent or fields.get("agent_instance_id") or (facts.agent if facts else None)
        if type(agent) is str and (agent == scope.value or agent.startswith(scope.value + ":")):
            return scope
        return None

    def _publish_replay(
        self,
        state: Any,
        event: EventDescriptor,
        session_id: str | None,
        fields: dict[str, object],
        covers: rows.Covers,
        *,
        scope: ReplayScopeId | None = None,
        replay_row: rows.RowFacts | None = None,
    ) -> bool:
        """Admit one frame to a replay stream; False when the caller sends it plainly."""
        if "session_replay_seq" not in state.features or session_id is None:
            return False
        connection = state.connection
        if scope is None:
            scope = self._member_scope(connection, session_id, fields, replay_row)
            if scope is None:
                return False
        if self.replay.scope_for(connection) != scope:
            return False
        if event is EventDescriptor.CHILD_PROGRESS and self.replay.is_pending(connection):
            return True
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
        if (
            prepared.seq is None
            and scope.kind == "top"
            and "in_flight_replay" in state.features
            and self.replay.controller_for(connection).control.in_flight.owns(event, fields)
        ):
            # the join's in-flight frames carry it, so holding it only costs room
            return True
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
            session_id,
            dict(fields),
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

    async def admit_join_response(
        self,
        connection: Any,
        request_id: str | int | None,
        result: object,
        baseline_builder: Any = None,
    ) -> bool:
        state = self._connections.get(connection)
        if state is None or state.output is None:
            self.invalidate_replay_connection(connection)
            return False
        output = state.output
        if output.phase not in {"claimed_buffering", "flushing", "ready"}:
            self.invalidate_replay_connection(connection)
            return False
        scope = self.replay.scope_for(connection)
        if scope is None or not self._frontier_holds(connection, scope):
            self.invalidate_replay_connection(connection)
            return False
        if baseline_builder is not None:
            result = baseline_builder(result)
        frame = connection.codec._result_response(request_id, result)
        if frame is None:
            # fail closed: the client never gets this stream id, so no frame may carry it
            self.invalidate_replay_connection(connection)
            return await self.send(connection, connection.codec.response(request_id, result))
        held = self.replay.pending_tail(connection)
        stream_id = self.replay.pending_stream_id(connection)
        items: list[_OutputFrame] = [_OutputFrame(frame)]
        for frame_state in held:
            payload = connection.codec.notification(
                frame_state.event,
                frame_state.session_id,
                stream_id=stream_id,
                seq=frame_state.seq,
                **frame_state.fields,
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
        self,
        connection: Any,
        controller: SessionUse,
        session_id: str,
        history_validator: object | None = None,
    ) -> tuple[dict[str, object], int, str | None]:
        """Read the newest history page and its stable frontier row off-loop."""
        store = validation.active(controller.runtime, {"session_id": session_id})
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
                child_runs="child_run_events" in features,
                **(
                    {"supplied_validator": history_validator}
                    if history_validator is not None
                    else {}
                ),
            )
            return history, frontier_seq, frontier_row_id

        result = await self.run_read(connection, read, controller, fence=False)
        assert isinstance(result, tuple)
        history, *rest = result
        shaped = self._strip_restart_reason(connection, history)
        if shaped is history:
            return result
        return (shaped, *rest)

    def _strip_restart_reason(self, connection: Any, history: object) -> object:
        """Strip ``server_restarted`` from projected approval rows for a legacy
        connection (no ``approval_restart_reason`` feature). One output seam for
        every history page; returns the same object when nothing is stripped."""
        from ...core.child_runs import child_run_for_features

        features = self._connections[connection].features
        if not isinstance(history, dict):
            return history
        rows = history.get("messages")
        if not isinstance(rows, list):
            return history
        shaped = event_contract.shape_approval_history_rows(
            rows, include_restart_reason="approval_restart_reason" in features,
        )
        if "child_recovery" not in features:
            projected = [child_run_for_features(row, child_recovery=False) if isinstance(row, dict) else row for row in shaped]
            if any(a is not b for a, b in zip(shaped, projected, strict=True)):
                shaped = projected
        if shaped is rows:
            return history
        return {**history, "messages": shaped}
