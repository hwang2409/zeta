"""Pure scheduling decisions; the caller owns time and transactional claims."""

from datetime import UTC, datetime, timedelta
from typing import Protocol

from .models import DueOccurrence, JobState, timestamp
from .trigger import Poll, cron_matches

CATCH_UP = timedelta(hours=2)


class ScheduleStore(Protocol):
    def jobs(self) -> tuple[JobState, ...]: ...


def tick(store: ScheduleStore, now: datetime) -> tuple[DueOccurrence, ...]:
    timestamp(now)
    now = now.astimezone(UTC)
    result = []
    for state in store.jobs():
        if not state.enabled or state.approved_at is None or state.last_run is None:
            continue
        trigger = state.job.trigger
        if isinstance(trigger, Poll):
            previous = state.last_check or state.approved_at
            if now < previous + timedelta(seconds=trigger.interval_seconds):
                continue
            due = now
        else:
            due = now.replace(second=0, microsecond=0)
            lower = max(now - CATCH_UP, state.approved_at)
            while due >= lower:
                if state.last_due is not None and due <= state.last_due:
                    break
                if cron_matches(trigger, due):
                    break
                due -= timedelta(minutes=1)
            if due < lower or (state.last_due is not None and due <= state.last_due):
                continue
        result.append(
            DueOccurrence(state.job.name, state.revision, due, now, state.last_run)
        )
    return tuple(sorted(result, key=lambda item: (item.due_at, item.name)))
