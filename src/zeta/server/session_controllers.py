"""Own and resolve the session runtime/control records used by serve."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

from .protocol import ErrorDescriptor, ProtocolError

if TYPE_CHECKING:
    from .control import ServeControl
    from .runtime import ServerRuntime


@dataclass(slots=True, eq=False)
class SessionController:
    """The persistent runtime/control pair for one resident session. Only
    ``SessionControllers`` changes ``uses``."""

    control: ServeControl
    generation: int
    uses: set[SessionUse] = field(default_factory=set)

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


class SessionControllers:
    """Own the attached record and the records available for selection."""

    def __init__(self, control: ServeControl) -> None:
        self._generations = itertools.count(1)
        self._records: dict[ServeControl, SessionController] = {
            control: SessionController(control, next(self._generations))
        }
        self._attached = self._records[control]
        self._parked: dict[str, SessionController] = {}
        # close waiters per record, woken when its last use settles
        self._drains: dict[SessionController, list[asyncio.Future[None]]] = {}

    def records(self) -> tuple[SessionController, ...]:
        return tuple(self._records.values())

    def record_for(self, control: ServeControl) -> SessionController:
        return self._records[control]

    def register(self, control: ServeControl) -> SessionController:
        record = SessionController(control, next(self._generations))
        self._records[control] = record
        return record

    def attached(self) -> SessionController:
        return self._attached

    def owned(self, connection: Any) -> SessionUse:
        """A use of the persistent record controlled by ``connection``."""
        del connection
        return self._use(self._attached)

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
        del connection
        attached = self._attached.runtime
        if (
            self._attached in self._records.values()
            and attached.opened is not None
            and attached.session_id == session_id
        ):
            return self._use(self._attached)
        record = self._parked.get(session_id)
        if record is not None:
            return self._use(record)
        if attached.opened is None and unattached_error is not None:
            raise unattached_error
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def attach(self, record: SessionController) -> SessionController:
        self._parked.pop(record.runtime.session_id, None)
        self._attached = record
        return record

    def park(self, record: SessionController) -> SessionController:
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

    def retire(self, record: SessionController) -> SessionController:
        """Take ``record`` out of every index before its close starts: the one
        step every close of a resident control goes through. ``attached()``
        keeps naming a retired final record only for shutdown."""
        # by identity: a record whose setup failed has no session id
        self._parked = {key: item for key, item in self._parked.items() if item is not record}
        self._records.pop(record.control, None)
        return record

    async def drained(self, record: SessionController) -> None:
        """Wait until no use of ``record`` is held by a completion; a close runs
        only after this. A use its request still holds does not delay a close
        yet: that request may wait on the registry close that follows."""
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
        if record not in self._records.values() or record.generation != use.generation or (
            session_id is not None
            and (runtime.opened is None or runtime.session_id != session_id)
        ):
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")

    def is_attached(self, control: ServeControl) -> bool:
        return self._attached.control is control
