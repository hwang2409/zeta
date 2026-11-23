"""Own and resolve the session runtime/control records used by serve."""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from ..runtime.foreground_events import SessionEventSource
from .protocol import ErrorDescriptor, ProtocolError

if TYPE_CHECKING:
    from ..session_retention.activation import RetentionActivationOwner
    from .control import ServeControl
    from .runtime import ServerRuntime

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HelloAdmission:
    connection: Any
    mode: Literal["legacy", "multi"]
    generation: int


@dataclass(frozen=True, slots=True)
class SessionAcquisition:
    connection: Any
    session_id: str
    generation: int
    previous: SessionController | None
    target: SessionController
    created: bool


@dataclass(slots=True, eq=False)
class SessionController:
    """The persistent runtime/control pair for one resident session. Only
    ``SessionControllers`` changes ``uses`` and the close state."""

    control: ServeControl
    generation: int
    # false for a runtime the server borrowed (an observer host's): its owner
    # closes it, so the record's close never does
    owns_runtime: bool = True
    uses: set[SessionUse] = field(default_factory=set)
    # set by ``retire``: no resolution, wake, or claim from here on, but the
    # record stays resident (and counts) until ``close_task`` ends
    closing: bool = False
    close_task: asyncio.Task[None] | None = None
    # the session the record held when its close began
    closing_session_id: str | None = None

    @property
    def runtime(self) -> ServerRuntime:
        return self.control.runtime


class SessionUse:
    """One request's hold on a record, from its resolve until the request ends
    or the completion it was handed to settles. A close of the record waits
    until it holds no use."""

    __slots__ = ("_owner", "_state", "acquisition", "generation", "record")

    def __init__(
        self,
        owner: SessionControllers,
        record: SessionController,
        acquisition: SessionAcquisition | None = None,
    ) -> None:
        self._owner = owner
        self._state: Literal["held", "transferred", "released"] = "held"
        self.record = record
        self.acquisition = acquisition
        self.generation = record.generation

    @property
    def control(self) -> ServeControl:
        return self.record.control

    @property
    def runtime(self) -> ServerRuntime:
        return self.record.runtime

    def is_current_binding(self, connection: Any) -> bool:
        """Whether a new owned read would still use this record."""
        owner = self._owner
        return not self.record.closing and owner._by_connection.get(connection, owner._attached) is self.record

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def release(self) -> None:
        """End the request's hold; a no-op once transferred or released."""
        if self._state == "held":
            self._state = "released"
            self._owner._settle(self)

    def transfer_to(self, completion: asyncio.Future[Any]) -> None:
        """Hand this use to ``completion``: only its settling (result, error, or
        cancel) releases it, so cancelling the request no longer does."""
        if self._state != "held":
            raise RuntimeError(f"session use is already {self._state}")
        self._state = "transferred"
        completion.add_done_callback(lambda _done: self._owner._settle(self))

    async def run[T](self, work: Coroutine[Any, Any, T]) -> T:
        """Run ``work`` as the one completion that owns this use. A cancel of the
        request reaches ``work`` once and drops only the response: what ``work``
        does on cancel (abort before its commit, finish after it) still holds
        the record open."""
        task = asyncio.create_task(work)
        self.transfer_to(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            raise


class SessionControllers:
    """Own the attached record and the records available for selection."""

    def __init__(
        self, control: ServeControl, *, owns_runtime: bool = True,
        retention_owner: RetentionActivationOwner | None = None,
    ) -> None:
        self._retention_owner = retention_owner
        self._generations = itertools.count(1)
        self._records: dict[ServeControl, SessionController] = {
            control: SessionController(control, next(self._generations), owns_runtime)
        }
        self._attached = self._records[control]
        self._parked: dict[str, SessionController] = {}
        # These maps are the sole semantic ownership state.  A reservation is
        # installed before any target runtime operation can await.
        self._by_connection: dict[Any, SessionController] = {}
        self._by_session: dict[str, SessionAcquisition | SessionController] = {}
        self._session_generations = itertools.count(1)
        # hello admission is semantic state.  The generation makes rollback
        # safe when a failed connection races a later hello.
        self._controller_mode: Literal["empty", "legacy", "multi"] = "empty"
        self._controller_members: dict[Any, int] = {}
        self._hello_records: dict[Any, SessionController] = {}
        self._hello_generations = itertools.count(1)
        # a reconnect grace keeps the negotiated mode alive after its member
        # closes.  admission state follows one table: tokens (provisional or
        # committed) keep the mode; then exact grace does; otherwise it is empty.
        self._grace_session_id: str | None = None
        self._grace_generation: int | None = None
        # close waiters per record, woken when its last use settles
        self._drains: dict[SessionController, list[asyncio.Future[None]]] = {}

    @property
    def controller_mode(self) -> Literal["empty", "legacy", "multi"]:
        return self._controller_mode

    def admit_hello(
        self,
        connection: Any,
        *,
        negotiated_multi: bool,
        reconnect_session_id: str | None = None,
    ) -> HelloAdmission:
        """Reserve process controller mode before claiming the physical socket."""
        requested: Literal["legacy", "multi"] = "multi" if negotiated_multi else "legacy"
        if self._controller_mode == "empty":
            self._controller_mode = requested
        elif not self._controller_members:
            if self._controller_mode != requested:
                raise ProtocolError(
                    ErrorDescriptor.CONNECTION_CAPACITY,
                    "another client is already connected",
                )
            if requested == "multi" and (
                reconnect_session_id is None
                or reconnect_session_id != self._grace_session_id
            ):
                raise ProtocolError(
                    ErrorDescriptor.CONNECTION_CAPACITY,
                    "another client is already connected",
                )
        elif self._controller_mode != requested or (
            requested == "legacy" and self._controller_members
        ):
            raise ProtocolError(
                ErrorDescriptor.CONNECTION_CAPACITY,
                "another client is already connected",
            )
        generation = next(self._hello_generations)
        self._controller_members[connection] = generation
        return HelloAdmission(connection, requested, generation)

    def _refresh_controller_mode(self) -> None:
        """Apply the admission state table after every token or release change.

        ``tokens`` includes both provisional and committed hello reservations.
        An exact reconnect grace is the only state that survives without one.
        Resident and closing records are session state, not controller tokens;
        they do not keep an unattached mode alive.
        """
        if self._controller_members or self._grace_generation is not None:
            return
        self._controller_mode = "empty"

    def commit_hello(self, admission: HelloAdmission) -> None:
        if self._controller_members.get(admission.connection) != admission.generation:
            raise RuntimeError("hello admission is no longer current")

    def abort_hello(self, admission: HelloAdmission) -> None:
        if self._controller_members.get(admission.connection) != admission.generation:
            return
        del self._controller_members[admission.connection]
        self._refresh_controller_mode()

    def is_controller(self, connection: Any) -> bool:
        return connection in self._controller_members

    def controller_for(self, connection: Any) -> SessionController | None:
        """Return the exact committed or provisional record for ``connection``."""
        record = self._by_connection.get(connection)
        if record is not None:
            return record
        record = self._hello_records.get(connection)
        if record is not None:
            return record
        return next((
            owner.target for owner in self._by_session.values()
            if isinstance(owner, SessionAcquisition) and owner.connection is connection
        ), None)

    async def begin_controller_release(
        self, connection: Any, *, reconnect: bool, keep_turn: bool
    ) -> None:
        """Release only this connection's semantic controller state."""
        record = self.controller_for(connection)
        if record is not None:
            await record.control.begin_controller_release(
                reconnect=reconnect, keep_turn=keep_turn
            )

    def release_connection(self, connection: Any, reason: str = "eof") -> None:
        del reason
        previous = self._by_connection.get(connection)
        self._hello_records.pop(connection, None)
        generation = self._controller_members.pop(connection, None)
        if generation is None:
            self._refresh_controller_mode()
            return
        self._by_connection.pop(connection, None)
        if not self._controller_members:
            session_id = (
                previous.runtime.session_id
                if previous is not None and previous.runtime.opened is not None
                else None
            )
            if session_id is None:
                for record in self._records.values():
                    if record.runtime.opened is not None:
                        session_id = record.runtime.session_id
                        break
            self._grace_session_id = session_id
            # legacy has a reconnect grace without a session id; dormant multi
            # does not. the mode selected by the released token distinguishes them.
            self._grace_generation = (
                generation if self._controller_mode == "legacy" or session_id is not None else None
            )
        self._refresh_controller_mode()

    def records(self) -> tuple[SessionController, ...]:
        return tuple(self._records.values())

    def record_for(self, control: ServeControl) -> SessionController:
        return self._records[control]

    def event_record(self, source: SessionEventSource) -> SessionController | None:
        return next((record for record in self._records.values()
                     if record.generation == source.generation
                     and (record.runtime.opened.metadata.session_id if record.runtime.opened is not None else None) == source.session_id
                     and (record.close_task is None or not record.close_task.done())), None)

    def legacy_event_owner(self, record: SessionController) -> Any | None:
        # The sole legacy hello member controls this record, including during
        # reconnect setup before the hello response can be queued.
        if self._controller_mode == "legacy" and self._attached is record:
            return next(iter(self._controller_members), None)
        return None

    def event_owner(self, source: SessionEventSource) -> Any | None:
        record = self.event_record(source)
        if record is None:
            return None
        return self.legacy_event_owner(record) if self._controller_mode == "legacy" else next(
            (connection for connection, owner in self._by_connection.items() if owner is record),
            None,
        )

    def legacy_observers_of(self, source: SessionEventSource) -> bool:
        # Empty mode also covers legacy observers waiting without a controller.
        return self._controller_mode != "multi" and self.event_record(source) is self._attached

    def register(self, control: ServeControl) -> SessionUse:
        """A new generation for ``control``, reserved by the returned use until
        its setup ends."""
        record = SessionController(control, next(self._generations))
        self._records[control] = record
        return self._use(record)

    def attached(self) -> SessionController:
        return self._attached

    def owned(self, connection: Any, session_id: str | None = None) -> SessionUse:
        """A use of the persistent record controlled by ``connection``."""
        if session_id is not None:
            acquisition = self.reserve_existing(connection, session_id)
            return self._use(acquisition.target, acquisition)
        record = self._by_connection.get(connection, self._attached)
        if self._controller_mode == "multi" and connection not in self._by_connection:
            self._hello_records[connection] = record
        if record.closing:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        return self._use(record)

    def selected(
        self,
        connection: Any,
        session_id: str,
        *,
        unattached_error: ProtocolError | None = None,
    ) -> SessionUse:
        """A use of the persistent resident record selected by ``session_id``.

        ``unattached_error`` replaces the absence error while no session is
        attached, for an operation whose legacy wire text differs."""
        current = self._by_connection.get(connection, self._attached)
        attached = current.runtime
        if (
            not current.closing
            and attached.opened is not None
            and attached.session_id == session_id
        ):
            return self._use(current)
        record = self._by_session.get(session_id)
        if isinstance(record, SessionAcquisition):
            if record.connection is not connection:
                raise ProtocolError(
                    ErrorDescriptor.SESSION_CONTROLLED,
                    "session is controlled by another client",
                    {"session_id": session_id},
                )
            record = record.target
        if record is None:
            record = self._parked.get(session_id)
        if record is not None:
            return self._use(record)
        if attached.opened is None and unattached_error is not None:
            raise unattached_error
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def resident(self, session_id: str) -> SessionController | None:
        """Return the live record for ``session_id`` without changing ownership."""
        owner = self._by_session.get(session_id)
        if isinstance(owner, SessionAcquisition):
            return owner.target
        if owner is not None and not owner.closing:
            return owner
        parked = self._parked.get(session_id)
        if parked is not None and not parked.closing:
            return parked
        return next((
            record for record in self._records.values()
            if record.runtime.opened is not None
            and record.runtime.session_id == session_id
            and not record.closing
        ), None)

    def reserve_existing(self, connection: Any, session_id: str) -> SessionAcquisition:
        """Reserve an existing resident session for an exact reconnect."""
        record = self.resident(session_id)
        if record is None or record.closing:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        return self.reserve_session(connection, session_id, record)

    def reserve_session(
        self, connection: Any, session_id: str, candidate: SessionController
    ) -> SessionAcquisition:
        """Reserve a session synchronously, before opening its runtime."""
        existing = self._by_session.get(session_id)
        target = existing.target if isinstance(existing, SessionAcquisition) else existing
        holder = next((
            owner for owner, record in self._by_connection.items()
            if record is target and owner is not connection
        ), None)
        if isinstance(existing, SessionAcquisition) or holder is not None:
            raise ProtocolError(
                ErrorDescriptor.SESSION_CONTROLLED,
                "session is controlled by another client",
                {"session_id": session_id},
            )
        previous = self._by_connection.get(connection)
        acquisition = SessionAcquisition(
            connection, session_id, next(self._session_generations), previous,
            candidate, existing is None and candidate is not previous,
        )
        self._by_session[session_id] = acquisition
        return acquisition

    def use_acquisition(self, acquisition: SessionAcquisition) -> SessionUse:
        """Hold the exact target reserved by ``acquisition`` during setup."""
        if self._by_session.get(acquisition.session_id) is not acquisition:
            raise RuntimeError("session acquisition is no longer current")
        return self._use(acquisition.target)

    def commit_session(self, acquisition: SessionAcquisition) -> SessionUse:
        if self._by_session.get(acquisition.session_id) is not acquisition:
            raise RuntimeError("session acquisition is no longer current")
        self._by_session[acquisition.session_id] = acquisition.target
        self._by_connection[acquisition.connection] = acquisition.target
        self._hello_records.pop(acquisition.connection, None)
        self._parked.pop(acquisition.session_id, None)
        return self._use(acquisition.target)

    def abort_session(self, acquisition: SessionAcquisition) -> asyncio.Task[None] | None:
        if self._by_session.get(acquisition.session_id) is not acquisition:
            return None
        if acquisition.created:
            del self._by_session[acquisition.session_id]
            if self._records.get(acquisition.target.control) is acquisition.target:
                return self.retire(acquisition.target)
        else:
            self._by_session[acquisition.session_id] = acquisition.target
        return None

    def bind_retention_planner(self) -> None:
        # Bind at startup and legacy swaps, not when the planner publishes.
        if self._retention_owner is not None:
            self._retention_owner.set_publish(self._attached.runtime.retention_event_sink)

    def attach(self, record: SessionController) -> SessionController:
        assert not record.closing
        self._parked.pop(record.runtime.session_id, None)
        self._attached = record
        self.bind_retention_planner()
        return record

    def park(self, record: SessionController) -> SessionController:
        assert not record.closing
        self._parked[record.runtime.session_id] = record
        return record

    def unpark(self, record: SessionController) -> SessionController | None:
        session_id = record.runtime.session_id
        if self._parked.get(session_id) is not record:
            return None
        del self._parked[session_id]
        return record

    def parked_controls(self) -> tuple[ServeControl, ...]:
        return tuple(record.control for record in self._parked.values())

    def parked_record(self, session_id: str) -> SessionController | None:
        return self._parked.get(session_id)

    def closing_session_ids(self) -> list[str]:
        """The sessions of detached records still closing; each counts toward
        the resident limit."""
        return [
            record.closing_session_id
            for record in self._records.values()
            if record.closing and record is not self._attached and record.closing_session_id
        ]

    def is_closing(self, control: ServeControl) -> bool:
        """``control`` has begun its close (or ended it): no wake or new work."""
        record = self._records.get(control)
        return record is None or record.closing

    def retire(
        self, record: SessionController, settlement_timeout: float | None = None
    ) -> asyncio.Task[None]:
        """The one close of a resident record; a second call returns the same
        close. Resolution stops now; the control closes, then the runtime, only
        once no use holds the record; approval settlement is verified last;
        only then does the record leave ``records()``. A failed close is
        logged, since nobody need await it."""
        return self._begin_close(record, settlement_timeout, None)

    def _begin_close(
        self,
        record: SessionController,
        settlement_timeout: float | None,
        before_runtime_close: Callable[[], Awaitable[None]] | None,
        *,
        report: bool = True,
    ) -> asyncio.Task[None]:
        if record.close_task is not None:
            return record.close_task
        assert self._records.get(record.control) is record, "retire of a foreign record"
        runtime = record.runtime
        record.closing = True
        record.closing_session_id = runtime.session_id if runtime.opened is not None else None
        # by identity: a record whose setup failed has no session id
        self._parked = {key: item for key, item in self._parked.items() if item is not record}
        task = asyncio.create_task(
            self._close(record, settlement_timeout, before_runtime_close)
        )
        if report:
            task.add_done_callback(lambda done: _report_close(done, record.closing_session_id))
        record.close_task = task
        return task

    async def _close(
        self,
        record: SessionController,
        settlement_timeout: float | None,
        before_runtime_close: Callable[[], Awaitable[None]] | None,
    ) -> None:
        control = record.control
        try:
            try:
                await self._drained(record)
                await control.close()
            finally:
                try:
                    try:
                        if before_runtime_close is not None:
                            await before_runtime_close()
                    finally:
                        if record.owns_runtime:
                            await record.runtime.close()
                finally:
                    await control.wait_for_approval_settlement(settlement_timeout)
        finally:
            del self._records[control]
            for session_id, owner in tuple(self._by_session.items()):
                if owner is record:
                    del self._by_session[session_id]
            for connection, owner in tuple(self._by_connection.items()):
                if owner is record:
                    del self._by_connection[connection]

    async def wait_closed(self, session_id: str) -> None:
        """Wait until every closing record of ``session_id`` has closed."""
        for record in tuple(self._records.values()):
            if record.closing and record.closing_session_id == session_id:
                assert record.close_task is not None
                # a cancelled waiter must not cancel the close
                await asyncio.shield(record.close_task)

    async def close(
        self,
        settlement_timeout: float | None,
        before_attached_runtime_close: Callable[[], Awaitable[None]],
    ) -> None:
        """Server shutdown: close every resident record and wait for every close.
        ``before_attached_runtime_close`` runs just before the attached record's
        runtime would close (owned or not), and that close's error propagates;
        the others are logged by their close. A cancel of the shutdown does not
        cancel a close."""
        committed = {
            *self._parked.values(),
            *self._by_connection.values(),
            *(record for record in self._records.values() if not record.uses),
        }
        for record in committed:
            if record is not self._attached:
                self.retire(record, settlement_timeout)
        done = self._attached.close_task
        if done is not None and done.done():
            # a repeated shutdown: the first one already reported this close
            await asyncio.gather(
                *(record.close_task for record in self._records.values() if record.close_task),
                return_exceptions=True,
            )
            return
        # its close error reaches the caller, so it is not logged as well
        attached = self._begin_close(
            self._attached, settlement_timeout, before_attached_runtime_close, report=False
        )
        others = [
            record.close_task
            for record in self._records.values()
            if record.close_task is not None and record.close_task is not attached
        ]
        try:
            await asyncio.shield(attached)
        finally:
            await asyncio.gather(*others, return_exceptions=True)

    async def _drained(self, record: SessionController) -> None:
        """Wait until no use of ``record`` is held by a completion. A use its
        request still holds does not delay the close: that request may wait on
        the registry close that follows, and its finalizer work, if any, already
        owns a transferred use."""
        while self._transferred(record):
            waiter = asyncio.get_running_loop().create_future()
            self._drains.setdefault(record, []).append(waiter)
            await waiter

    def _use(
        self,
        record: SessionController,
        acquisition: SessionAcquisition | None = None,
    ) -> SessionUse:
        use = SessionUse(self, record, acquisition)
        record.uses.add(use)
        return use

    def _settle(self, use: SessionUse) -> None:
        record = use.record
        record.uses.discard(use)
        if not self._transferred(record):
            for waiter in self._drains.pop(record, ()):
                if not waiter.done():
                    waiter.set_result(None)

    @staticmethod
    def _transferred(record: SessionController) -> bool:
        return any(use._state == "transferred" for use in record.uses)

    def require_current(
        self, connection: Any, use: SessionUse, session_id: str | None
    ) -> None:
        """Fail unless the use's record is still resident, still at the use's
        generation, and still holds ``session_id``."""
        del connection
        record = use.record
        runtime = record.runtime
        if record.closing or record not in self._records.values() or record.generation != use.generation or (
            session_id is not None
            and (runtime.opened is None or runtime.session_id != session_id)
        ):
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def is_attached(self, control: ServeControl) -> bool:
        return self._attached.control is control


def _report_close(task: asyncio.Task[None], session_id: str | None) -> None:
    # a close that nobody awaits would otherwise fail without a trace
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.error("closing session %s failed", session_id, exc_info=exc)
