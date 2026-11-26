"""Own and resolve the session runtime/control records used by serve."""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from ..runtime.foreground_events import SessionEventSource
from .protocol import ErrorDescriptor, ProtocolError

if TYPE_CHECKING:
    from ..session_retention.activation import RetentionActivationOwner
    from .control import ServeControl
    from .runtime import ServerRuntime

logger = logging.getLogger(__name__)

MAX_MULTI_CONTROLLERS = 16
MAX_PARKED_RECORDS = 3
# resident records: multi controllers, one sessionless record, and parked records.
MAX_RESIDENT_RECORDS = MAX_MULTI_CONTROLLERS + 1 + MAX_PARKED_RECORDS


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
    previous_index: SessionController | None


@dataclass(slots=True, eq=False)
class _MultiGrace:
    session_id: str
    release: asyncio.Future[None]
    idle_deadline: asyncio.Task[None] | None = None


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
    setup_by: object | None = None
    parked_by: object | None = None
    parked_scoped: bool = False
    parked_last_sent: dict[str, object] | None = None

    @property
    def runtime(self) -> ServerRuntime:
        return self.control.runtime


class SessionUse:
    """One request's hold on a record, from its resolve until the request ends
    or the completion it was handed to settles. A close of the record waits
    until it holds no use."""

    __slots__ = ("_owner", "_state", "acquisition", "generation", "is_view", "record", "setup")

    def __init__(
        self,
        owner: SessionControllers,
        record: SessionController,
        acquisition: SessionAcquisition | None = None,
        *, setup: bool = False, is_view: bool = False,
    ) -> None:
        self._owner = owner
        self._state: Literal["held", "transferred", "released"] = "held"
        self.record = record
        self.acquisition = acquisition
        self.setup = setup
        self.is_view = is_view
        self.generation = record.generation

    @property
    def control(self) -> ServeControl:
        return self.record.control

    @property
    def runtime(self) -> ServerRuntime:
        return self.record.runtime

    def is_current_binding(self, connection: Any) -> bool:
        """Whether a new read through the same binding would still use this record."""
        if self.record.closing:
            return False
        if self.is_view:
            return connection.connections.replay.controller_for(connection) is self.record
        return self._owner._request_record(connection) is self.record

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
        drain_events: Callable[[SessionController], Awaitable[None]] | None = None,
    ) -> None:
        self._frozen = False
        self._record_drain = asyncio.Event()
        self._drain_events = drain_events
        self._retention_owner = retention_owner
        self._generations = itertools.count(1)
        self._records: dict[ServeControl, SessionController] = {
            control: SessionController(control, next(self._generations), owns_runtime)
        }
        self._attached = self._records[control]
        # Unbound multi requests share an unopened record, never the legacy default.
        self._sessionless = self._attached
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
        self._multi_graces: dict[SessionController, _MultiGrace] = {}
        # close waiters per record, woken when its last use settles
        self._drains: dict[SessionController, list[asyncio.Future[None]]] = {}
        if self._retention_owner is not None:
            control.runtime.set_session_changed_sink(self._snapshot_retention_eligibility)
        self._snapshot_retention_eligibility()

    @property
    def controller_mode(self) -> Literal["empty", "legacy", "multi"]:
        # ServeControl can finish grace without a connection transition.
        self._refresh_controller_mode()
        return self._controller_mode

    def freeze(self) -> None:
        self._frozen = True

    def _require_open(self) -> None:
        if self._frozen:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def admit_hello(
        self,
        connection: Any,
        *,
        negotiated_multi: bool,
    ) -> HelloAdmission:
        """Reserve process controller mode before claiming the physical socket."""
        self._require_open()
        requested: Literal["legacy", "multi"] = "multi" if negotiated_multi else "legacy"
        if self.controller_mode == "empty":
            if requested == "multi" and (
                self._sessionless.closing or self._sessionless.runtime.opened is not None
            ):
                source = self._sessionless.control
                with self.register(lambda: source.server._make_sibling_control(source.runtime)) as use:
                    self._sessionless = use.record
            self._controller_mode = requested
            self._snapshot_retention_eligibility()
            self.bind_retention_planner()
        elif not self._controller_members:
            # Only grace or cleanup holds the mode. A fresh multi controller is
            # admitted; graced sessions stay reachable only by their exact id.
            if self._controller_mode != requested:
                raise ProtocolError(
                    ErrorDescriptor.CONNECTION_CAPACITY,
                    "another client is already connected",
                )
        elif self._controller_mode != requested or (
            requested == "legacy"
            or len(self._by_connection) >= MAX_MULTI_CONTROLLERS
        ):
            raise ProtocolError(
                ErrorDescriptor.CONNECTION_CAPACITY,
                "another client is already connected",
            )
        generation = next(self._hello_generations)
        self._controller_members[connection] = generation
        return HelloAdmission(connection, requested, generation)

    def _refresh_controller_mode(self) -> None:
        if self._controller_members or self._multi_graces:
            return
        if self._controller_mode == "legacy" and any(
            (work := record.control.reconnect_grace_task) is not None and not work.done()
            for record in self._records.values()
        ):
            return
        if self._controller_mode == "multi" and any(
            record.closing or record.parked_scoped
            for record in self._records.values()
        ):
            return
        if self._controller_mode != "empty":
            self._controller_mode = "empty"
            self._snapshot_retention_eligibility()

    def commit_hello(self, admission: HelloAdmission) -> None:
        self._require_open()
        if self._controller_members.get(admission.connection) != admission.generation:
            raise RuntimeError("hello admission is no longer current")

    def abort_hello(self, admission: HelloAdmission) -> None:
        if self._controller_members.get(admission.connection) != admission.generation:
            return
        del self._controller_members[admission.connection]
        self._hello_records.pop(admission.connection, None)
        self._refresh_controller_mode()

    def is_controller(self, connection: Any) -> bool:
        return connection in self._controller_members

    def controller_for(self, connection: Any) -> SessionController | None:
        return self._by_connection.get(connection, self._hello_records.get(connection))

    def records_charged_to(self, connection: object) -> tuple[SessionController, ...]:
        """Snapshot every attached, provisional, parked, or reserved record once."""
        return tuple(
            record for record in self._records.values()
            if any(holder is connection for holder in self._charges(record))
        )

    def _charges(self, record: SessionController) -> Iterator[object]:
        for bindings in (self._by_connection, self._hello_records):
            yield from (connection for connection, target in bindings.items() if target is record)
        for reservation in self._by_session.values():
            if isinstance(reservation, SessionAcquisition) and reservation.target is record:
                yield reservation.connection
        if record.parked_by is not None:
            yield record.parked_by
        if record.setup_by is not None:
            yield record.setup_by

    def owner_of(self, record: SessionController) -> object | _MultiGrace | None:
        """Return the live controller, reconnect grace, or None for a free record."""
        return next(
            (connection for connection in self._charges(record) if connection in self._controller_members),
            self._multi_graces.get(record),
        )

    def begin_controller_release(
        self, connection: object, *, reconnect: bool, keep_turn: bool
    ) -> asyncio.Task[None]:
        """Start all per-record releases before the physical slot is freed."""
        releases: list[asyncio.Future[None]] = []
        for record in self.records_charged_to(connection):
            if record.closing:
                continue
            try:
                release = record.control.begin_controller_release(
                    reconnect=reconnect and not self._frozen, keep_turn=keep_turn,
                    preserve_grace=True,
                )
            except BaseException as exc:  # noqa: BLE001 - settle the other records too
                release = asyncio.get_running_loop().create_future()
                release.set_exception(exc)
            releases.append(release)
            if not self._frozen and record.runtime.opened is not None:
                self._watch_multi_grace(record, release)

        async def settle() -> None:
            await asyncio.gather(*releases, return_exceptions=True)
            for release in releases:
                release.result()

        return asyncio.create_task(settle())

    def _watch_multi_grace(self, record: SessionController, release: asyncio.Future[None]) -> None:
        self._clear_grace(record)
        grace = _MultiGrace(record.runtime.session_id, release)
        self._multi_graces[record] = grace
        work: asyncio.Future[None] | None = record.control.reconnect_grace_task
        if work is None:
            work = release
            if record.control.server.approval_reconnect_grace != 0:
                async def idle_deadline() -> None:
                    await record.control._grace_deadline()
                    await asyncio.wait((release,))

                work = grace.idle_deadline = asyncio.create_task(idle_deadline())
        work.add_done_callback(lambda done: self._expire_multi_grace(record, grace, done))

    def _clear_grace(self, record: SessionController) -> None:
        grace = self._multi_graces.pop(record, None)
        if grace is not None:
            self._refresh_controller_mode()
            if grace.idle_deadline is not None:
                grace.idle_deadline.cancel()

    def _expire_multi_grace(
        self, record: SessionController, grace: _MultiGrace, work: asyncio.Future[None]
    ) -> None:
        if (work.cancelled() or self._frozen or record.closing
                or self._multi_graces.get(record) is not grace):
            return
        self.retire(record)

    def release_connection(self, connection: Any) -> None:
        self._hello_records.pop(connection, None)
        generation = self._controller_members.pop(connection, None)
        if generation is not None:
            self._by_connection.pop(connection, None)
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

    def register(
        self, make_control: Callable[[], ServeControl], *, connection: object | None = None
    ) -> SessionUse:
        """A new generation for ``control``, reserved by the returned use until
        its setup ends."""
        self._require_open()
        if self._controller_mode == "multi" and len(self._records) >= MAX_RESIDENT_RECORDS:
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "too many resident sessions",
                {"reason": "resident_limit"},
            )
        control = make_control()
        record = SessionController(control, next(self._generations), setup_by=connection)
        self._records[control] = record
        if self._retention_owner is not None:
            control.runtime.set_session_changed_sink(self._snapshot_retention_eligibility)
            self._snapshot_retention_eligibility()
        return self._use(record, setup=True)

    def attached(self) -> SessionController:
        return self._attached

    def _request_record(self, connection: Any) -> SessionController:
        if self._controller_mode == "multi":
            return self._by_connection.get(connection, self._sessionless)
        return self._attached

    def owned(self, connection: Any, session_id: str | None = None) -> SessionUse:
        """A use of the persistent record controlled by ``connection``."""
        self._require_open()
        if session_id is not None:
            acquisition = self.reserve_existing(connection, session_id)
            self._hello_records[connection] = acquisition.target
            return self._use(acquisition.target, acquisition, setup=True)
        record = self._request_record(connection)
        if self._controller_mode == "multi" and connection not in self._by_connection:
            self._hello_records[connection] = record
        if record.closing:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        return self._use(record)

    def observer_view(self, connection: Any) -> SessionUse:
        """A tracked use of the exact record selected by this observer's replay join."""
        self._require_open()
        record = connection.connections.replay.controller_for(connection)
        if (
            record is None or record.closing or record.runtime.opened is None
            or self._records.get(record.control) is not record
        ):
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        return self._use(record, is_view=True)

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
        self._require_open()
        current = self._request_record(connection)
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
                raise self._controlled(session_id)
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
        if self._frozen:
            return None
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

    @staticmethod
    def _controlled(session_id: str) -> ProtocolError:
        return ProtocolError(
            ErrorDescriptor.SESSION_CONTROLLED,
            "session is controlled by another client",
            {"session_id": session_id},
        )

    def acquisition_target(
        self, connection: object, session_id: str, *, reconnect: bool,
    ) -> SessionController | None:
        """Find a resident target, refusing another controller's charge before setup."""
        self._require_open()
        record = self.resident(session_id)
        if record is not None:
            self._require_acquirable(connection, record, session_id, reconnect=reconnect)
        return record

    def _require_acquirable(
        self, connection: object, record: SessionController, session_id: str, *, reconnect: bool,
    ) -> None:
        owner = self.owner_of(record)
        if isinstance(owner, _MultiGrace):
            if not reconnect:
                raise self._controlled(session_id)
        elif owner is not None and owner is not connection:
            raise self._controlled(session_id)

    def reserve_existing(self, connection: Any, session_id: str) -> SessionAcquisition:
        """Reserve an existing resident session for an exact reconnect."""
        return self._reserve_session(connection, session_id, None)

    def reserve_session(
        self, connection: Any, session_id: str, candidate: SessionController
    ) -> SessionAcquisition:
        return self._reserve_session(connection, session_id, candidate)

    def _reserve_session(
        self, connection: Any, session_id: str, candidate: SessionController | None
    ) -> SessionAcquisition:
        """Reserve a session synchronously, before any target setup can await."""
        resident = self.acquisition_target(connection, session_id, reconnect=candidate is None)
        existing = self._by_session.get(session_id)
        if isinstance(existing, SessionAcquisition):
            raise self._controlled(session_id)
        target = candidate if candidate is not None else resident
        if target is None or target.closing:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        previous = self._by_connection.get(connection)
        acquisition = SessionAcquisition(
            connection, session_id, next(self._session_generations), previous,
            target, target is not resident and target is not previous, existing,
        )
        self._by_session[session_id] = acquisition
        return acquisition

    def use_acquisition(self, acquisition: SessionAcquisition) -> SessionUse:
        """Hold the exact target reserved by ``acquisition`` during setup."""
        self._require_open()
        if self._by_session.get(acquisition.session_id) is not acquisition:
            raise RuntimeError("session acquisition is no longer current")
        return self._use(acquisition.target, setup=True)

    def commit_session(self, acquisition: SessionAcquisition) -> SessionUse:
        self._require_open()
        if self._by_session.get(acquisition.session_id) is not acquisition or acquisition.target.closing:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        self._by_session[acquisition.session_id] = acquisition.target
        self._by_connection[acquisition.connection] = acquisition.target
        if acquisition.target is self._attached:
            # Once multi owns this record, its release may retire it.
            self._attached = self._sessionless
        acquisition.target.setup_by = None
        self._clear_grace(acquisition.target)
        self._hello_records.pop(acquisition.connection, None)
        if acquisition.target.runtime.opened is not None:
            self.unpark(acquisition.target)
        return self._use(acquisition.target)

    def abort_session(self, acquisition: SessionAcquisition) -> asyncio.Task[None] | None:
        if self._by_session.get(acquisition.session_id) is not acquisition:
            return None
        if acquisition.previous_index is None:
            del self._by_session[acquisition.session_id]
        else:
            self._by_session[acquisition.session_id] = acquisition.previous_index
        if acquisition.created and self._records.get(acquisition.target.control) is acquisition.target:
            return self.retire(acquisition.target)
        return None

    def has_open_session(self) -> bool:
        return self._retention_eligible

    def _snapshot_retention_eligibility(self) -> None:
        # Only the event loop reads records; the retention worker reads this bool.
        if self._controller_mode == "multi":
            self._retention_eligible = any(
                not record.closing and record.runtime.opened is not None
                for record in self._records.values()
            )
        else:
            self._retention_eligible = self._attached.runtime.opened is not None

    def bind_retention_planner(self) -> None:
        # Bind at startup and legacy swaps, not when the planner publishes.
        if self._retention_owner is not None:
            self._retention_owner.set_publish(self._request_record(None).runtime.retention_event_sink)

    def attach(self, record: SessionController) -> SessionController:
        self._require_open()
        assert not record.closing
        self._parked.pop(record.runtime.session_id, None)
        self._attached = record
        self._snapshot_retention_eligibility()
        self.bind_retention_planner()
        return record

    @staticmethod
    def _require_parked_room(held: list[str]) -> None:
        if len(held) >= MAX_PARKED_RECORDS:
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY,
                "too many parked sessions are running",
                {"reason": "parked_limit", "parked": held},
            )

    def require_legacy_parked_room(self) -> None:
        self._require_parked_room([*self._parked, *self.closing_session_ids()])

    def session_of(self, connection: object) -> SessionController | None:
        """The connection's committed session record, or None."""
        return self._by_connection.get(connection)

    @contextmanager
    def reserve_park(
        self, connection: object, record: SessionController, *, target: SessionController | None
    ) -> Iterator[None]:
        """Hold parked room for ``record``, which this swap parks; ``target``,
        the session the swap attaches, leaves the parked list."""
        self._require_open()
        assert record.parked_by is None
        self._require_parked_room([
            held.closing_session_id if held.closing_session_id is not None else held.runtime.session_id
            for held in self._records.values() if held.parked_by is connection and held is not target
        ])
        record.parked_by = connection
        try:
            yield
        finally:
            if not record.closing and self._parked.get(record.runtime.session_id) is not record:
                record.parked_by = None

    def parked_records(self, connection: object) -> Iterator[SessionController]:
        return (
            record for record in self._parked.values()
            if (record.parked_scoped and record.parked_by is connection)
            or (not record.parked_scoped and self._controller_mode != "multi")
        )

    def park(self, record: SessionController, connection: object | None = None) -> SessionController:
        self._require_open()
        assert not record.closing
        record.parked_by = connection
        record.parked_scoped = self._controller_mode == "multi"
        self._parked[record.runtime.session_id] = record
        return record

    def unpark(self, record: SessionController) -> SessionController | None:
        self._require_open()
        session_id = record.runtime.session_id
        if self._parked.get(session_id) is not record:
            return None
        del self._parked[session_id]
        record.parked_by = None
        record.parked_scoped = False
        record.parked_last_sent = None
        return record

    def parked_controls(self) -> tuple[ServeControl, ...]:
        return tuple(record.control for record in self._parked.values())

    def parked_in_grace(self, control: ServeControl) -> bool:
        return self._records.get(control) in self._multi_graces

    def parked_record(self, session_id: str) -> SessionController | None:
        return self._parked.get(session_id)

    def closing_session_ids(self) -> list[str]:
        """The sessions of detached records still closing."""
        return [
            record.closing_session_id
            for record in self._records.values()
            if record.closing and record is not self._attached and record.closing_session_id
        ]

    def is_closing(self, control: ServeControl) -> bool:
        """``control`` has begun its close (or ended it): no wake or new work."""
        record = self._records.get(control)
        return record is None or record.closing

    def retire_for_reopen(self, record: SessionController) -> asyncio.Task[None]:
        """Retire a free resident record that a multi resume reopens from disk."""
        if record is self._attached:
            # as in commit_session: once multi takes it, it is not the legacy default
            self._attached = self._sessionless
        return self.retire(record)

    def retire(
        self, record: SessionController, settlement_timeout: float | None = None
    ) -> asyncio.Task[None]:
        """Reject resolution now and finish the record once its work settles."""
        if record.close_task is not None:
            return record.close_task
        assert self._records.get(record.control) is record, "retire of a foreign record"
        runtime = record.runtime
        record.closing = True
        self._snapshot_retention_eligibility()
        record.closing_session_id = runtime.session_id if runtime.opened is not None else None
        self._parked = {key: item for key, item in self._parked.items() if item is not record}
        task = asyncio.create_task(self._close(record, settlement_timeout))
        if not self._frozen:
            task.add_done_callback(lambda done: _report_close(done, record.closing_session_id))
        record.close_task = task
        return task

    async def _close(self, record: SessionController, settlement_timeout: float | None) -> None:
        first: BaseException | None = None

        async def attempt(operation: Awaitable[object]) -> None:
            nonlocal first
            try:
                await operation
            except BaseException as exc:  # noqa: BLE001
                if first is None:
                    first = exc

        control = record.control
        try:
            if self._frozen:
                await self._record_drain.wait()
            await self._drained(record)
            grace = self._multi_graces.get(record)
            if grace is not None:
                await asyncio.wait((grace.release,))
            await attempt(control.close())
            await attempt(control.wait_for_approval_settlement(settlement_timeout))
            if self._frozen:
                await self._record_drain.wait()
            if self._drain_events is not None and (self._frozen or self._controller_mode == "multi"):
                await attempt(self._drain_events(record))
            if grace is not None and grace.idle_deadline is not None:
                grace.idle_deadline.cancel()
                await attempt(asyncio.gather(grace.idle_deadline, return_exceptions=True))
            if record.owns_runtime:
                await attempt(record.runtime.close())
            if first is not None:
                raise first
        finally:
            del self._records[control]
            if self._retention_owner is not None:
                record.runtime.set_session_changed_sink(None)
                self._snapshot_retention_eligibility()
            self._clear_grace(record)
            record.parked_by = None
            record.setup_by = None
            for session_id, owner in tuple(self._by_session.items()):
                if owner is record or isinstance(owner, SessionAcquisition) and owner.target is record:
                    del self._by_session[session_id]
            for connection, owner in tuple(self._by_connection.items()):
                if owner is record:
                    del self._by_connection[connection]
            self._refresh_controller_mode()

    async def wait_closed(self, session_id: str) -> None:
        """Wait until every closing record of ``session_id`` has closed."""
        for record in tuple(self._records.values()):
            if record.closing and record.closing_session_id == session_id:
                assert record.close_task is not None
                # a cancelled waiter must not cancel the close
                await asyncio.shield(record.close_task)

    async def close(
        self, records: tuple[SessionController, ...], settlement_timeout: float | None
    ) -> None:
        """Drain the frozen generations in snapshot order, including active retires."""
        self._record_drain.set()
        results = await asyncio.gather(
            *(self.retire(record, settlement_timeout) for record in records),
            return_exceptions=True,
        )
        self._controller_members.clear()
        self._hello_records.clear()
        self._refresh_controller_mode()
        for record, result in zip(records, results, strict=True):
            if isinstance(result, BaseException):
                # gather fabricates a CancelledError; result() preserves the original.
                assert record.close_task is not None
                record.close_task.result()

    async def _drained(self, record: SessionController) -> None:
        # An ordinary request may be waiting on a close. Only setup and work
        # transferred to a completion can still use a record after cancellation.
        while self._holds_work(record):
            waiter = asyncio.get_running_loop().create_future()
            self._drains.setdefault(record, []).append(waiter)
            await waiter

    def _use(
        self,
        record: SessionController,
        acquisition: SessionAcquisition | None = None,
        *, setup: bool = False, is_view: bool = False,
    ) -> SessionUse:
        use = SessionUse(self, record, acquisition, setup=setup, is_view=is_view)
        record.uses.add(use)
        return use

    def _settle(self, use: SessionUse) -> None:
        record = use.record
        record.uses.discard(use)
        if not self._holds_work(record):
            for waiter in self._drains.pop(record, ()):
                if not waiter.done():
                    waiter.set_result(None)

    @staticmethod
    def _holds_work(record: SessionController) -> bool:
        return any(use.setup or use._state == "transferred" for use in record.uses)

    def require_current(
        self, connection: Any, use: SessionUse, session_id: str | None
    ) -> None:
        """Fail unless the use's record is still resident, still at the use's
        generation, and still holds ``session_id``."""
        record = use.record
        runtime = record.runtime
        if self._frozen or record.closing or record not in self._records.values() or record.generation != use.generation or (
            session_id is not None
            and (runtime.opened is None or runtime.session_id != session_id)
        ) or (use.is_view and not use.is_current_binding(connection)):
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def is_attached(self, control: ServeControl) -> bool:
        return self._controller_mode != "multi" and self._attached.control is control


def _report_close(task: asyncio.Task[None], session_id: str | None) -> None:
    # a close that nobody awaits would otherwise fail without a trace
    if not task.cancelled() and (exc := task.exception()) is not None:
        logger.error("closing session %s failed", session_id, exc_info=exc)
