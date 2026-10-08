"""Shared provider retry classification, timing, and per-turn budget."""

from __future__ import annotations

import random
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import httpx

MAX_PROVIDER_ATTEMPTS = 5
MAX_PROVIDER_RETRY_SECONDS = 60.0
MAX_PROVIDER_RETRY_DELAY_SECONDS = 30.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
_TERMINAL_ERROR_CODES = {
    "auth_error",
    "authentication_error",
    "context_length_exceeded",
    "permission_denied",
    "permission_error",
}


@dataclass(frozen=True, slots=True)
class RetryPlan:
    """One scheduled retry within a shared turn budget."""

    attempt: int
    reason: str
    delay: float
    owner: str
    is_stall: bool = False

    def metadata(self) -> dict[str, object]:
        return {
            "attempt": self.attempt,
            "reason": self.reason,
            "delay": self.delay,
            "decision": "retried",
        }


@dataclass(slots=True)
class ProviderRetryBudget:
    """Bound all provider requests and retry waits made by one agent turn."""

    started_at: float = field(default_factory=lambda: time.monotonic())
    attempts: int = 0
    retry_wait_seconds: float = 0.0
    original_error: Any | None = None
    exhausted: bool = False
    max_stall_retries: int = 2
    stall_retries: int = 0
    excluded_stall_seconds: float = 0.0
    records: list[dict[str, object]] = field(default_factory=list)
    attempt_records: list[dict[str, object]] = field(default_factory=list)

    def _effective_elapsed(self) -> float:
        return time.monotonic() - self.started_at - self.excluded_stall_seconds

    def start_attempt(self, owner: str, *, is_stall: bool = False) -> bool:
        if self.attempts >= MAX_PROVIDER_ATTEMPTS:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return False
        if not is_stall and self._effective_elapsed() >= MAX_PROVIDER_RETRY_SECONDS:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return False
        self.attempts += 1
        self.attempt_records.append({"attempt": self.attempts, "owner": owner})
        return True

    def plan(
        self,
        source: BaseException | object,
        *,
        owner: str,
        event_data: Mapping[str, Any] | None = None,
    ) -> RetryPlan | None:
        if self.original_error is None:
            self.original_error = source
        if self.exhausted:
            return None
        if not retryable_provider_error(source, event_data):
            return None
        if self.attempts >= MAX_PROVIDER_ATTEMPTS:
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return None
        delay = retry_wait_seconds(source, self.attempts, event_data)
        is_stall = bool(getattr(source, "is_stall", False))
        stall_seconds = getattr(source, "stall_seconds", 0.0)
        has_stall_duration = isinstance(stall_seconds, (int, float)) and stall_seconds > 0
        if is_stall:
            if has_stall_duration:
                self.excluded_stall_seconds += float(stall_seconds)
            if self.stall_retries >= self.max_stall_retries:
                self.exhausted = True
                self.records.append({"decision": "budget-exhausted"})
                return None
            self.stall_retries += 1
        if (
            (not is_stall or has_stall_duration)
            and self._effective_elapsed() + delay > MAX_PROVIDER_RETRY_SECONDS
        ):
            self.exhausted = True
            self.records.append({"decision": "budget-exhausted"})
            return None
        return RetryPlan(
            attempt=self.attempts + 1,
            reason=retry_reason(source),
            delay=delay,
            owner=owner,
            is_stall=bool(getattr(source, "is_stall", False)),
        )

    def record_retry(self, plan: RetryPlan) -> None:
        self.retry_wait_seconds += plan.delay
        self.records.append(plan.metadata())


_CURRENT_BUDGET: ContextVar[ProviderRetryBudget | None] = ContextVar(
    "provider_retry_budget", default=None
)


def current_retry_budget() -> ProviderRetryBudget | None:
    return _CURRENT_BUDGET.get()


@contextmanager
def use_retry_budget(budget: ProviderRetryBudget) -> Iterator[None]:
    token = _CURRENT_BUDGET.set(budget)
    try:
        yield
    finally:
        _CURRENT_BUDGET.reset(token)


async def apply_retry_budget[T](
    source: AsyncIterator[T], budget: ProviderRetryBudget
) -> AsyncIterator[T]:
    """Run one provider stream inside its turn-level retry context."""

    with use_retry_budget(budget):
        try:
            async for item in source:
                yield item
        finally:
            close = getattr(source, "aclose", None)
            if close is not None:
                await close()


def transient_network_error(error: BaseException | object) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError, httpx.TransportError)):
        return True
    cause = error.__cause__ if isinstance(error, BaseException) else None
    while cause is not None:
        if isinstance(cause, (ConnectionError, TimeoutError, httpx.TransportError)):
            return True
        cause = cause.__cause__
    return False


def retryable_provider_error(
    error: BaseException | object,
    event_data: Mapping[str, Any] | None = None,
) -> bool:
    status_code = getattr(error, "status_code", None)
    code = getattr(error, "code", None)
    if code in _TERMINAL_ERROR_CODES:
        return False
    if type(status_code) is int and 400 <= status_code < 500 and status_code != 429:
        return False
    if status_code in _RETRYABLE_STATUS_CODES:
        return True
    if (
        getattr(error, "retryable", False)
        or getattr(error, "is_stall", False)
        or transient_network_error(error)
    ):
        return True
    message = getattr(error, "message", None)
    if code in {"timeout", "transport_error"}:
        return True
    if (
        code == "http_error"
        and isinstance(message, str)
        and message.strip().lower() == "request failed"
    ):
        return True
    return bool(event_data and event_data.get("retryable") is True)


def retry_wait_seconds(
    error: BaseException | object,
    retry_number: int,
    event_data: Mapping[str, Any] | None = None,
) -> float:
    retry_after = getattr(error, "retry_after", None)
    if retry_after is None and event_data is not None:
        retry_after = event_data.get("retry_after")
    if type(retry_after) in {int, float}:
        return max(0.0, min(MAX_PROVIDER_RETRY_DELAY_SECONDS, float(retry_after)))
    base = min(MAX_PROVIDER_RETRY_DELAY_SECONDS, float(2 ** (retry_number - 1)))
    return min(
        MAX_PROVIDER_RETRY_DELAY_SECONDS,
        random.uniform(base * 0.5, base * 1.5),
    )


def retry_reason(error: BaseException | object) -> str:
    explicit_reason = getattr(error, "retry_reason", None)
    if isinstance(explicit_reason, str) and explicit_reason:
        return explicit_reason
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    status_code = getattr(error, "status_code", None)
    if status_code is not None:
        return "http_error"
    if transient_network_error(error):
        return "transport_error"
    return "backend_error"


def retry_error_label(error: BaseException | object) -> str:
    status_code = getattr(error, "status_code", None)
    retry_reason_value = getattr(error, "retry_reason", None)
    if status_code == 429 or retry_reason_value == "rate_limit_error":
        return "429 rate limited"
    if status_code == 529 or retry_reason_value == "overloaded_error":
        return "529 overloaded"
    if status_code is not None:
        return f"{status_code} server error"
    if getattr(error, "retryable", False):
        return "provider overloaded"
    if transient_network_error(error):
        return "network error"
    return "provider error"
