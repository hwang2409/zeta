"""Process-local one-shot wake scheduling and autonomous-run admission."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal

from ..protocol.types import Message, harness_notice

WakeSessionClass = Literal["top_level", "parent_owned"]
WakeState = Literal["scheduled", "due", "passive"]
WakePassiveReason = Literal[
    "top_level_chain_limit",
    "parent_lifetime_limit",
]

MIN_WAKE_DELAY_SECONDS = 10
MAX_WAKE_DELAY_SECONDS = 7 * 24 * 60 * 60
MAX_PENDING_WAKES = 8
MAX_WAKE_NOTE_CHARS = 1_000
MAX_TOP_LEVEL_ROOTS = 8
MAX_TOP_LEVEL_GENERATION = 3
MAX_PARENT_ADMISSIONS_PER_PROCESS = 96
MAX_WAKE_RUN_COMPLETIONS = 8
_WAKE_WAIT_SLICE_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class WakeRecord:
    wake_id: str
    note: str
    scheduled_at_utc: datetime
    due_at_utc: datetime
    session_class: WakeSessionClass
    chain_id: str | None
    generation: int | None
    state: WakeState
    passive_reason: WakePassiveReason | None = None


@dataclass(frozen=True, slots=True)
class WakeAdmission:
    admission_id: str
    admitted_at_utc: datetime
    records: tuple[WakeRecord, ...]


@dataclass(frozen=True, slots=True)
class WakeActivation:
    admission: WakeAdmission | None = None
    passive: tuple[WakeRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class WakeScheduleResult:
    record: WakeRecord
    autonomous: bool
    reason: WakePassiveReason | None


@dataclass(frozen=True, slots=True)
class WakeOwnedWork:
    pending_ids: tuple[str, ...]
    earliest_due_at_utc: datetime | None


class WakeError(ValueError):
    """A visible wake operation error."""


class WakeScheduler:
    """Own one session's bounded wake records for one process lifetime."""

    def __init__(
        self,
        session_class: WakeSessionClass,
        *,
        on_due: Callable[[], None],
        wall_clock: Callable[[], datetime] | None = None,
        assert_owned: Callable[[], None] | None = None,
        id_factory: Callable[[], str] | None = None,
        wait_for_change: Callable[[asyncio.Event, float], Awaitable[None]]
        | None = None,
    ) -> None:
        self.session_class = session_class
        self._on_due = on_due
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._assert_owned = assert_owned or (lambda: None)
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._wait_for_change = wait_for_change or _wait_for_change
        self._records: dict[str, WakeRecord] = {}
        self._admission_count = 0
        self._changed = asyncio.Event()
        self._timer_task: asyncio.Task[None] | None = None
        self._closed = False
        self._due_notified = False
        self._external_root_count = 0
        self._wake_chain_id: str | None = None
        self._wake_generation: int | None = None
        self._wake_successor_scheduled = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def admission_count(self) -> int:
        return self._admission_count

    def begin_external_run(self) -> None:
        """Start one user-origin run with a fresh top-level root allowance."""

        self._require_open()
        self._external_root_count = 0
        self._wake_chain_id = None
        self._wake_generation = None
        self._wake_successor_scheduled = False

    def begin_wake_run(self, admission: WakeAdmission) -> None:
        """Set the chain facts used by schedules made during an admitted run."""

        self._require_open()
        top_level = [
            record
            for record in admission.records
            if record.session_class == "top_level"
        ]
        if top_level:
            selected = min(top_level, key=lambda record: record.wake_id)
            self._wake_chain_id = selected.chain_id
            self._wake_generation = max(record.generation or 0 for record in top_level)
        else:
            self._wake_chain_id = None
            self._wake_generation = None
        self._wake_successor_scheduled = False

    def schedule(self, *, after_seconds: float, note: str) -> WakeScheduleResult:
        self._require_open()
        self._assert_owned()
        delay = _validate_delay(after_seconds)
        bounded_note = _validate_note(note)
        if len(self._records) >= MAX_PENDING_WAKES:
            raise WakeError(f"at most {MAX_PENDING_WAKES} wakes may be pending")
        now = _utc(self._wall_clock())
        chain_id: str | None = None
        generation: int | None = None
        reason: WakePassiveReason | None = None
        if self.session_class == "top_level":
            chain_id, generation, reason = self._top_level_schedule_facts()
        elif self._admission_count >= MAX_PARENT_ADMISSIONS_PER_PROCESS:
            reason = "parent_lifetime_limit"
        wake_id = "wake_" + self._id_factory()
        record = WakeRecord(
            wake_id=wake_id,
            note=bounded_note,
            scheduled_at_utc=now,
            due_at_utc=now + timedelta(seconds=delay),
            session_class=self.session_class,
            chain_id=chain_id,
            generation=generation,
            state="passive" if reason is not None else "scheduled",
            passive_reason=reason,
        )
        self._records[wake_id] = record
        self._signal_changed()
        return WakeScheduleResult(record, reason is None, reason)

    def list_pending(self) -> tuple[WakeRecord, ...]:
        self._require_open()
        self._assert_owned()
        return tuple(
            sorted(
                self._records.values(),
                key=lambda record: (record.due_at_utc, record.wake_id),
            )
        )

    def cancel(self, wake_id: str) -> WakeRecord:
        self._require_open()
        self._assert_owned()
        record = self._records.get(wake_id)
        if record is None:
            raise WakeError(f"wake {wake_id!r} is not pending")
        if record.state == "due":
            raise WakeError(f"wake {wake_id!r} is already due")
        del self._records[wake_id]
        self._due_notified = False
        self._signal_changed()
        return record

    def owned_work(self) -> WakeOwnedWork:
        records = tuple(
            record for record in self._records.values() if record.state != "passive"
        )
        return WakeOwnedWork(
            pending_ids=tuple(sorted(record.wake_id for record in records)),
            earliest_due_at_utc=min(
                (record.due_at_utc for record in records), default=None
            ),
        )

    def has_due(self) -> bool:
        """Return whether any wake presentation is ready without changing state."""

        self._require_open()
        now = _utc(self._wall_clock())
        return any(record.due_at_utc <= now for record in self._records.values())

    def poll_due(self) -> bool:
        """Mark elapsed records due and report whether idle activation is needed."""

        self._require_open()
        self._assert_owned()
        now = _utc(self._wall_clock())
        changed = False
        for wake_id, record in tuple(self._records.items()):
            if record.state == "scheduled" and record.due_at_utc <= now:
                self._records[wake_id] = replace(record, state="due")
                changed = True
        due_exists = any(
            record.due_at_utc <= now and record.state in {"due", "passive"}
            for record in self._records.values()
        )
        if changed:
            self._due_notified = False
        if due_exists and not self._due_notified:
            self._due_notified = True
            return True
        return False

    def admit_due(self) -> WakeActivation | None:
        """Close one due cohort at the idle admission seam."""

        self._require_open()
        self.poll_due()
        now = _utc(self._wall_clock())
        passive = tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.state == "passive" and record.due_at_utc <= now
                ),
                key=lambda record: (record.due_at_utc, record.wake_id),
            )
        )
        due = tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.state == "due" and record.due_at_utc <= now
                ),
                key=lambda record: (record.due_at_utc, record.wake_id),
            )
        )
        if due and self.session_class == "parent_owned":
            reason = self._parent_denial_reason()
            if reason is not None:
                denied = tuple(
                    replace(record, state="passive", passive_reason=reason)
                    for record in due
                )
                for record in denied:
                    self._records[record.wake_id] = record
                passive = (*passive, *denied)
                due = ()
        admission: WakeAdmission | None = None
        if due:
            admission = WakeAdmission(
                admission_id="wake_admission_" + self._id_factory(),
                admitted_at_utc=now,
                records=due,
            )
            for record in due:
                del self._records[record.wake_id]
            self._admission_count += 1
        for record in passive:
            self._records.pop(record.wake_id, None)
        self._due_notified = False
        self._signal_changed()
        if admission is None and not passive:
            return None
        return WakeActivation(admission=admission, passive=passive)

    def restore_activation(self, activation: WakeActivation) -> None:
        """Restore an admitted cohort when dispatch is canceled before it starts."""

        self._require_open()
        self._assert_owned()
        if activation.admission is not None:
            for record in activation.admission.records:
                self._records[record.wake_id] = replace(record, state="due")
            self._admission_count -= 1
        for record in activation.passive:
            self._records[record.wake_id] = record
        self._due_notified = False
        self._signal_changed()

    def consume_due_passive(self) -> tuple[WakeRecord, ...]:
        self._require_open()
        now = _utc(self._wall_clock())
        records = tuple(
            sorted(
                (
                    record
                    for record in self._records.values()
                    if record.state == "passive" and record.due_at_utc <= now
                ),
                key=lambda record: (record.due_at_utc, record.wake_id),
            )
        )
        for record in records:
            del self._records[record.wake_id]
        if records:
            self._signal_changed()
        return records

    async def close(self) -> int:
        """Discard all process-local records and drain the timer task."""

        if self._closed:
            return 0
        self._closed = True
        lost = len(self._records)
        self._records.clear()
        self._changed.set()
        task = self._timer_task
        self._timer_task = None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return lost

    def _top_level_schedule_facts(
        self,
    ) -> tuple[str | None, int | None, WakePassiveReason | None]:
        if self._wake_generation is None:
            if self._external_root_count >= MAX_TOP_LEVEL_ROOTS:
                return None, None, "top_level_chain_limit"
            self._external_root_count += 1
            return self._id_factory(), 1, None
        if (
            self._wake_successor_scheduled
            or self._wake_generation >= MAX_TOP_LEVEL_GENERATION
        ):
            return (
                self._wake_chain_id,
                self._wake_generation + 1,
                "top_level_chain_limit",
            )
        self._wake_successor_scheduled = True
        return self._wake_chain_id, self._wake_generation + 1, None

    def _parent_denial_reason(self) -> WakePassiveReason | None:
        if self._admission_count >= MAX_PARENT_ADMISSIONS_PER_PROCESS:
            return "parent_lifetime_limit"
        return None

    def _signal_changed(self) -> None:
        self._changed.set()
        if self._timer_task is None or self._timer_task.done():
            self._timer_task = asyncio.create_task(
                self._run_timer(), name="zeta wake scheduler"
            )

    async def _run_timer(self) -> None:
        try:
            while not self._closed:
                next_due = min(
                    (record.due_at_utc for record in self._records.values()),
                    default=None,
                )
                if next_due is None:
                    return
                delta = (next_due - _utc(self._wall_clock())).total_seconds()
                if delta <= 0:
                    if self.poll_due():
                        self._on_due()
                    self._changed.clear()
                    await self._wait_for_change(self._changed, _WAKE_WAIT_SLICE_SECONDS)
                    continue
                self._changed.clear()
                await self._wait_for_change(
                    self._changed, min(delta, _WAKE_WAIT_SLICE_SECONDS)
                )
        finally:
            if self._timer_task is asyncio.current_task():
                self._timer_task = None

    def _require_open(self) -> None:
        if self._closed:
            raise WakeError("wake scheduler is closed")


async def _wait_for_change(event: asyncio.Event, timeout: float) -> None:
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
    except TimeoutError:
        pass


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("wake clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def wake_notice(activation: WakeActivation) -> Message:
    """Build the production harness notice for one closed wake cohort."""

    admitted = activation.admission.records if activation.admission is not None else ()
    records = admitted + activation.passive
    lines = [
        f"- {record.note} (wake_id: {record.wake_id}"
        f"{f', passive reason: {record.passive_reason}' if record.passive_reason else ''})"
        for record in records
    ]
    return harness_notice(
        "wake reminder",
        "scheduled reminders are due:\n" + "\n".join(lines),
        event="wake",
        metadata={
            "admission_id": (
                activation.admission.admission_id
                if activation.admission is not None
                else None
            ),
            "wake_ids": [record.wake_id for record in records],
            "scheduled_at": [record.scheduled_at_utc.isoformat() for record in records],
            "due_at": [record.due_at_utc.isoformat() for record in records],
            "admission_reason": "scheduled_due",
        },
    )


def _validate_delay(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WakeError("after_seconds must be a number")
    delay = float(value)
    if not MIN_WAKE_DELAY_SECONDS <= delay <= MAX_WAKE_DELAY_SECONDS:
        raise WakeError(
            f"after_seconds must be between {MIN_WAKE_DELAY_SECONDS} and "
            f"{MAX_WAKE_DELAY_SECONDS}"
        )
    return delay


def _validate_note(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise WakeError("note must be nonempty text")
    if len(value) > MAX_WAKE_NOTE_CHARS:
        raise WakeError(f"note must be at most {MAX_WAKE_NOTE_CHARS} characters")
    return value
