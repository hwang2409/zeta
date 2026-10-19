"""Bounded retry state for notification-driven turns."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from ..protocol.types import ErrorInfo
from ..providers.retry_policy import retryable_provider_error

RetryKind = Literal["wake", "history"]


@dataclass(frozen=True, slots=True)
class NotificationFailure:
    """A failed notification turn with enough provider detail to classify it."""

    error: BaseException | ErrorInfo
    event_data: Mapping[str, object] | None = None

    @property
    def message(self) -> str:
        if isinstance(self.error, ErrorInfo):
            return self.error.message
        try:
            return str(self.error).strip() or type(self.error).__name__
        except Exception:  # noqa: BLE001 - malformed exception text is recoverable
            return type(self.error).__name__


NotificationOutcome = NotificationFailure | None


class NotificationRetry:
    """Own notification attempts, failure classification, and one backoff timer."""

    def __init__(
        self,
        *,
        wake: Callable[[], None],
        has_notification: Callable[[], bool],
        schedule_history: Callable[[], None],
        rollback: Callable[[], None],
        report_failure: Callable[[str], None],
        report_stopped: Callable[[str], None],
    ) -> None:
        self._wake = wake
        self._has_notification = has_notification
        self._schedule_history = schedule_history
        self._rollback = rollback
        self._report_failure = report_failure
        self._report_stopped = report_stopped
        self._call_later = lambda delay, timer_callback: (
            asyncio.get_running_loop().call_later(delay, timer_callback)
        )
        self._kind: RetryKind | None = None
        self._ready = False
        self._handle: asyncio.TimerHandle | None = None
        self._delay = 1.0
        self._failure_reported = False

    @property
    def ready_kind(self) -> RetryKind | None:
        return self._kind if self._ready else None

    @property
    def pending(self) -> bool:
        return self._kind is not None

    @property
    def failure_reported(self) -> bool:
        return self._failure_reported

    def request_wake(self) -> None:
        if self._kind is None:
            self._kind = "wake"
            self._ready = True

    def attempt(self, start_provider: Callable[[], None]) -> None:
        kind = self.ready_kind
        if kind is None:
            return
        try:
            if kind == "wake" and not self._has_notification():
                self.recovered()
                return
            if kind == "history":
                self._schedule_history()
            self._ready = False
            start_provider()
        except Exception as exc:  # noqa: BLE001
            self._rollback()
            self._handle_failure(kind, NotificationFailure(exc))

    def provider_finished(self, task: asyncio.Task[NotificationOutcome]) -> None:
        if task.cancelled():
            self.recovered()
            return
        try:
            failure = task.result()
        except Exception as exc:  # noqa: BLE001
            failure = NotificationFailure(exc)
        if failure is None:
            self.recovered()
            return
        if not self._handle_failure("history", failure):
            return
        try:
            self._schedule_history()
        except Exception:  # noqa: BLE001
            # The timer retries this same preparation without re-reporting it.
            return

    def recovered(self) -> None:
        self._kind = None
        self._ready = False
        self._delay = 1.0
        self._failure_reported = False
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def close(self) -> None:
        self.recovered()

    def _handle_failure(self, kind: RetryKind, failure: NotificationFailure) -> bool:
        if not self._is_transient(failure):
            self.recovered()
            self._report_stopped(failure.message)
            return False
        if self._failed(kind):
            self._report_failure(failure.message)
        return True

    @staticmethod
    def _is_transient(failure: NotificationFailure) -> bool:
        error = failure.error
        return (
            isinstance(error, OSError)
            or isinstance(error, ErrorInfo)
            and error.code == "os_error"
            or retryable_provider_error(error, failure.event_data)
        )

    def _failed(self, kind: RetryKind) -> bool:
        """Schedule one retry and report whether this starts a failure episode."""

        self._kind = kind
        self._ready = False
        first_failure = not self._failure_reported
        self._failure_reported = True
        if self._handle is None:
            self._handle = self._call_later(self._delay, self._fire)
            self._delay = min(self._delay * 2, 30.0)
        return first_failure

    def _fire(self) -> None:
        self._handle = None
        self._ready = True
        self._wake()
