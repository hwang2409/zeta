"""Own and resolve the session runtime/control records used by serve."""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from .protocol import ErrorDescriptor, ProtocolError

if TYPE_CHECKING:
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

    __slots__ = ("_owner", "_state", "generation", "record")

    def __init__(self, owner: SessionControllers, record: SessionController) -> None:
        self._owner = owner
        self._state: Literal["held", "transferred", "released"] = "held"
        self.record = record
        self.generation = record.generation

    @property
    def control(self) -> ServeControl:
        return self.record.control

    @property
    def runtime(self) -> ServerRuntime:
        return self.record.runtime

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

    def __init__(self, control: ServeControl, *, owns_runtime: bool = True) -> None:
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

    def release_connection(self, connection: Any, reason: str = "eof") -> None:
        del reason
        generation = self._controller_members.pop(connection, None)
        if generation is None:
            self._refresh_controller_mode()
            return
        self._by_connection.pop(connection, None)
        if not self._controller_members:
            session_id = None
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

    def register(self, control: ServeControl) -> SessionUse:
        """A new generation for ``control``, reserved by the returned use until
        its setup ends."""
        record = SessionController(control, next(self._generations))
        self._records[control] = record
        return self._use(record)

    def attached(self) -> SessionController:
        return self._attached

    def owned(self, connection: Any) -> SessionUse:
        """A use of the persistent record controlled by ``connection``."""
        record = self._by_connection.get(connection, self._attached)
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
            record = record.target
        if record is None:
            record = self._parked.get(session_id)
        if record is not None:
            return self._use(record)
        if attached.opened is None and unattached_error is not None:
            raise unattached_error
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def claim_existing(self, connection: Any, session_id: str) -> None:
        """Claim an already resident session for a reconnecting controller."""
        record = self._by_session.get(session_id)
        if isinstance(record, SessionAcquisition):
            record = record.target
        if record is None:
            for candidate in self._records.values():
                if candidate.runtime.session_id == session_id and not candidate.closing:
                    record = candidate
                    break
        if record is None or record.closing:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        holder = next((owner for owner, item in self._by_connection.items() if item is record), None)
        if holder is not None and holder is not connection:
            raise ProtocolError(
                ErrorDescriptor.SESSION_CONTROLLED,
                "session is controlled by another client",
                {"session_id": session_id},
            )
        self._by_session[session_id] = record
        self._by_connection[connection] = record

    def reserve_session(
        self, connection: Any, session_id: str, candidate: SessionController
    ) -> SessionAcquisition:
        """Reserve a session synchronously, before opening its runtime."""
        existing = self._by_session.get(session_id)
        if existing is not None:
            target = existing.target if isinstance(existing, SessionAcquisition) else existing
            if self._by_connection.get(connection) is not target:
                raise ProtocolError(
                    ErrorDescriptor.SESSION_CONTROLLED,
                    "session is controlled by another client",
                    {"session_id": session_id},
                )
        previous = self._by_connection.get(connection)
        acquisition = SessionAcquisition(
            connection, session_id, next(self._session_generations), previous,
            candidate, candidate not in self._records.values() or existing is None,
        )
        self._by_session[session_id] = acquisition
        return acquisition

    def commit_session(self, acquisition: SessionAcquisition) -> SessionUse:
        if self._by_session.get(acquisition.session_id) is not acquisition:
            raise RuntimeError("session acquisition is no longer current")
        self._by_session[acquisition.session_id] = acquisition.target
        self._by_connection[acquisition.connection] = acquisition.target
        return self._use(acquisition.target)

    def acquire_created(self, connection: Any, record: SessionController) -> SessionUse:
        """Reserve and commit a newly created target after it gets its id."""
        session_id = record.runtime.session_id
        if session_id is None:
            raise RuntimeError("created session has no id")
        acquisition = self.reserve_session(connection, session_id, record)
        return self.commit_session(acquisition)

    def abort_session(self, acquisition: SessionAcquisition) -> asyncio.Task[None] | None:
        if self._by_session.get(acquisition.session_id) is acquisition:
            del self._by_session[acquisition.session_id]
        if acquisition.created and self._records.get(acquisition.target.control) is acquisition.target:
            return self.retire(acquisition.target)
        return None

    def attach(self, record: SessionController) -> SessionController:
        assert not record.closing
        self._parked.pop(record.runtime.session_id, None)
        self._attached = record
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
        for record in tuple(self._parked.values()):
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

    def _use(self, record: SessionController) -> SessionUse:
        use = SessionUse(self, record)
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
