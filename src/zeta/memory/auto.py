"""One coalescing worker for automatic project-memory reconciliation."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..context_eviction import estimated_text_tokens
from ..project_registry import ProjectRegistry, ProjectRegistryError
from ..providers.retry_policy import ProviderRetryBudget, use_retry_budget
from .reconciler import (
    ReconciliationError,
    ReconciliationResponse,
    Transcript,
    parse_proposal,
    prepare_request,
    project_transcript_row,
)
from .reconciliation_state import (
    ReconciliationFailure,
    ReconciliationState,
    TerminalReceipt,
)

InvokeResult = str | ReconciliationResponse
Invoke = Callable[[str], InvokeResult | Awaitable[InvokeResult]]
Notice = Callable[[str], None]
Clock = Callable[[], float]
IdleWait = Callable[[asyncio.Event, float], Awaitable[None]]


async def _wait_for_idle(wake: asyncio.Event, timeout: float) -> None:
    await asyncio.wait_for(wake.wait(), timeout=timeout)


_MAX_TRANSCRIPT_CHUNK_BYTES = 96 * 1024
_MAX_REQUEST_BYTES = 64 * 1024
_REPAIR_PROMPT_BYTES = 1024
_MAX_FAILURE_LOG_BYTES = 64 * 1024


class _ConcurrentMemoryUpdate(Exception):
    """Signal that a pending range must remain queued for a later retry."""


@dataclass(frozen=True, slots=True)
class AutoMemoryConfig:
    """Resolved automatic-memory settings for one session."""

    enabled: bool = True
    model: str = "gpt-5.6-luna"
    token_threshold: int = 50_000
    idle_seconds: float = 600.0
    cas_retries: int = 3
    minimum_interval: float = 1.0
    shutdown_grace_seconds: float = 2.5
    retry_backoff_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.token_threshold < 1:
            raise ValueError("memory token threshold must be positive")
        if self.idle_seconds <= 0:
            raise ValueError("memory idle seconds must be positive")
        if self.cas_retries < 1:
            raise ValueError("memory CAS retries must be positive")
        if self.minimum_interval < 0:
            raise ValueError("memory minimum interval cannot be negative")
        if self.shutdown_grace_seconds < 0:
            raise ValueError("memory shutdown grace cannot be negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("memory retry backoff cannot be negative")


@dataclass(slots=True)
class _PendingRange:
    start: int
    end: int
    reasons: set[str]
    key: str | None = None


class AutoMemoryReconciler:
    """Own all idle, growth, and eviction scheduling for one session."""

    def __init__(
        self,
        *,
        registry: ProjectRegistry,
        project_id: str,
        session_id: str,
        session_dir: Path,
        invoke: Invoke,
        config: AutoMemoryConfig | None = None,
        notice: Notice | None = None,
        clock: Clock = time.monotonic,
        idle_wait: IdleWait = _wait_for_idle,
        retry_clock: Clock = time.time,
    ) -> None:
        self.registry = registry
        self.project_id = project_id
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.invoke = invoke
        self.config = config or AutoMemoryConfig()
        self.notice = notice
        self._clock = clock
        self._idle_wait = idle_wait
        self._retry_clock = retry_clock
        self.position_path = self.session_dir / "memory-reconcile.json"
        self.failure_log_path = self.session_dir / "memory-reconcile-errors.jsonl"
        self.state = ReconciliationState(
            self.position_path,
            project_id=project_id,
            session_id=session_id,
            diagnostics_path=registry.root.parent / "logs" / "memory-reconciliation.jsonl",
        )
        self._pending: list[_PendingRange] = []
        self._wake = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        self._worker_task: asyncio.Task[None] | None = None
        self._closing = False
        self._busy = False
        self._activity_generation = 0
        self._seen_activity_generation = 0
        self._idle_deadline: float | None = None
        self._last_request_finished = 0.0
        self._conflict_retries = 0
        self._resume_catch_up = False

    @property
    def last_reconciled_seq(self) -> int:
        return self.state.seq

    @property
    def last_failure(self) -> ReconciliationFailure | None:
        return self.state.last_failure

    @property
    def _last_reconciled_bytes(self) -> int:
        return self.state.transcript_bytes

    @property
    def _last_reconciled_tokens(self) -> int:
        return self.state.transcript_tokens

    def terminal_receipts(self) -> tuple[TerminalReceipt, ...]:
        """Return durable terminal receipts for this session."""
        return self.state.terminal_receipts()

    def retry_terminal(self, key: str) -> bool:
        """Re-queue one terminal receipt for this session."""
        queued = self.state.retry_terminal(key, now=self._retry_clock())
        if queued and not self._closing:
            self._wake.set()
            self._drained.clear()
            self._ensure_worker()
        return queued

    def observe_tokens(self, _total_tokens: int) -> None:
        """Compatibility callback; durable transcript activity owns growth."""
        self.activity()

    def activity(self, seq: int | None = None) -> None:
        """Report a transcript append after it has been durably persisted."""
        del seq  # The worker reads the authoritative durable sequence itself.
        if not self.config.enabled or self._closing:
            return
        self._activity_generation += 1
        self._idle_deadline = self._clock() + self.config.idle_seconds
        self._ensure_worker()
        self._drained.clear()
        self._wake.set()

    def before_eviction(self, seq_start: int, seq_end: int) -> None:
        """Queue every sequence in the exact range that will leave context."""
        if not self.config.enabled or self._closing or seq_end < seq_start:
            return
        self._add_pending(seq_start, seq_end, "eviction")
        self._ensure_worker()
        self._drained.clear()
        self._wake.set()

    async def drain(self) -> None:
        """Wait until all work currently caused by activity has settled."""
        if not self.config.enabled:
            return
        self._ensure_worker()
        self._wake.set()
        while (
            self._busy
            or self._pending
            or self._resume_catch_up
            or bool(self.state.ready_retries(self._retry_clock()))
            or self._seen_activity_generation < self._activity_generation
        ):
            await self._drained.wait()
            if self._busy or self._pending:
                self._drained.clear()

    async def close(self) -> None:
        """Stop without starting provider work and bound any in-flight request."""
        self._closing = True
        self.notice = None
        self._wake.set()
        task = self._worker_task
        if task is None:
            return
        done, _ = await asyncio.wait(
            {task}, timeout=self.config.shutdown_grace_seconds
        )
        if done:
            await asyncio.gather(task, return_exceptions=True)
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def catch_up(self) -> None:
        """Queue durable transcript content left behind by an earlier shutdown."""
        if not self.config.enabled or self._closing:
            return
        self._resume_catch_up = True
        self._drained.clear()
        self._wake.set()

    def _ensure_worker(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    def _add_pending(self, start: int, end: int, *reasons: str) -> None:
        for uncovered_start, uncovered_end in self.state.uncovered_ranges(start, end):
            merged = _PendingRange(uncovered_start, uncovered_end, set(reasons))
            remaining: list[_PendingRange] = []
            for item in self._pending:
                if item.key is not None or (
                    item.end + 1 < merged.start or merged.end + 1 < item.start
                ):
                    remaining.append(item)
                    continue
                merged.start = min(merged.start, item.start)
                merged.end = max(merged.end, item.end)
                merged.reasons.update(item.reasons)
            remaining.append(merged)
            self._pending = sorted(remaining, key=lambda item: item.start)

    def _queue_ready_retries(self) -> None:
        queued = {item.key for item in self._pending if item.key is not None}
        for work in self.state.ready_retries(self._retry_clock()):
            if work.key in queued:
                continue
            self._pending.append(
                _PendingRange(
                    work.seq_start,
                    work.seq_end,
                    {work.reason, "scheduled-retry"},
                    key=work.key,
                )
            )
        self._pending.sort(key=lambda item: item.start)

    async def _worker(self) -> None:
        while True:
            if self._closing:
                self._drained.set()
                return
            self._queue_ready_retries()
            if self._resume_catch_up:
                self._resume_catch_up = False
                latest_seq, _, _ = await asyncio.to_thread(self._transcript_state)
                if latest_seq > self.last_reconciled_seq:
                    self._add_pending(
                        self.last_reconciled_seq + 1, latest_seq, "resume"
                    )
            generation = self._activity_generation
            if generation != self._seen_activity_generation:
                latest_seq, _transcript_bytes, transcript_tokens = await asyncio.to_thread(
                    self._transcript_state
                )
                self._seen_activity_generation = generation
                growth = max(0, transcript_tokens - self._last_reconciled_tokens)
                if growth >= self.config.token_threshold:
                    self._add_pending(self.last_reconciled_seq + 1, latest_seq, "tokens")
            if self._pending:
                item = self._pending.pop(0)
                delay = self.config.minimum_interval - (
                    self._clock() - self._last_request_finished
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                if self._closing:
                    self._add_pending(item.start, item.end, *item.reasons)
                    self._drained.set()
                    return
                self._busy = True
                try:
                    await self._reconcile_range(item)
                    self._conflict_retries = 0
                except _ConcurrentMemoryUpdate:
                    self._conflict_retries += 1
                    if not self._closing:
                        self._add_pending(item.start, item.end, *item.reasons)
                        await asyncio.sleep(
                            min(0.05 * (2 ** (self._conflict_retries - 1)), 1.0)
                        )
                except Exception as exc:  # noqa: BLE001 - isolate background work
                    failure = await asyncio.to_thread(
                        self.state.record_failure,
                        seq_start=item.start,
                        seq_end=item.end,
                        validation_summary=(
                            str(exc)
                            if isinstance(exc, ReconciliationError)
                            else type(exc).__name__
                        ),
                        reason="+".join(sorted(item.reasons)),
                        usage={},
                        retry_backoff_seconds=self.config.retry_backoff_seconds,
                        now=self._retry_clock(),
                        occurred_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    )
                    await asyncio.to_thread(self._append_failure_log, failure)
                    self._publish_failure_notice()
                finally:
                    self._busy = False
                    self._last_request_finished = self._clock()
                    self._idle_deadline = self._clock() + self.config.idle_seconds
                continue
            self._drained.set()
            if self._closing:
                return
            self._wake.clear()
            if self._activity_generation != self._seen_activity_generation:
                continue
            timeout = None
            if self._idle_deadline is not None:
                timeout = max(0.0, self._idle_deadline - self._clock())
            retry_after = self.state.next_retry_after()
            if retry_after is not None:
                retry_timeout = max(0.0, retry_after - self._retry_clock())
                timeout = retry_timeout if timeout is None else min(timeout, retry_timeout)
            try:
                await self._idle_wait(self._wake, timeout)
            except TimeoutError:
                latest_seq, _, _ = await asyncio.to_thread(self._transcript_state)
                if latest_seq > self.last_reconciled_seq:
                    self._add_pending(self.last_reconciled_seq + 1, latest_seq, "idle")
                    self._drained.clear()
                self._idle_deadline = self._clock() + self.config.idle_seconds

    async def _reconcile_range(self, item: _PendingRange) -> None:
        cursor = item.start
        reason = "+".join(sorted(item.reasons))
        while cursor <= item.end:
            if self._closing:
                return
            rows, end_offset, end_tokens = await asyncio.to_thread(
                self._transcript_chunk, cursor, item.end
            )
            if not rows:
                return
            raw_transcript = Transcript(self.session_id, tuple(rows))
            first_seq = int(raw_transcript.rows[0]["seq"])
            changed: tuple[str, ...] = ()
            selected_start = first_seq
            selected_end = first_seq
            failure: Exception | None = None
            terminal_failure = False
            usage: dict[str, int] = {}
            for attempt in range(self.config.cas_retries):
                try:
                    snapshot = await asyncio.to_thread(
                        self.registry.memory_snapshot, self.project_id
                    )
                    today = datetime.now(UTC).date()
                    request = prepare_request(
                        raw_transcript,
                        snapshot.contents,
                        as_of=today,
                        max_bytes=_MAX_REQUEST_BYTES - _REPAIR_PROMPT_BYTES,
                    )
                    if not request.transcript.rows:
                        return
                    selected_start = int(request.transcript.rows[0]["seq"])
                    selected_end = int(request.transcript.rows[-1]["seq"])
                    budget = ProviderRetryBudget()
                    try:
                        raw_text, first_usage = await self._invoke_request(
                            request.prompt, budget
                        )
                        usage = self._sum_usage(usage, first_usage)
                        proposal = parse_proposal(
                            raw_text,
                            expected_digest=snapshot.digest,
                            transcript=request.transcript,
                            as_of=today,
                        )
                    except ReconciliationError as first_error:
                        repair_prompt = self._repair_prompt(request.prompt, first_error)
                        raw_text, repair_usage = await self._invoke_request(
                            repair_prompt, budget
                        )
                        usage = self._sum_usage(usage, repair_usage)
                        proposal = parse_proposal(
                            raw_text,
                            expected_digest=snapshot.digest,
                            transcript=request.transcript,
                            as_of=today,
                        )
                    updates = {
                        replacement.name: replacement.content
                        for replacement in proposal.replacements
                    }
                    if not updates:
                        break
                    provenance = {
                        "session_id": self.session_id,
                        "seq_start": selected_start,
                        "seq_end": selected_end,
                        "model": self.config.model,
                        "usage": usage,
                    }
                    try:
                        result = await asyncio.to_thread(
                            self.registry.compare_and_swap_memory,
                            self.project_id,
                            expected_digest=proposal.base_digest,
                            updates=updates,
                            provenance=provenance,
                        )
                    except ProjectRegistryError as exc:
                        if "digest mismatch" in str(exc):
                            if attempt + 1 < self.config.cas_retries:
                                continue
                            raise _ConcurrentMemoryUpdate from exc
                        raise ReconciliationError(
                            "project memory update failed"
                        ) from exc
                    if result.published:
                        changed = tuple(updates)
                    break
                except _ConcurrentMemoryUpdate:
                    raise
                except Exception as unit_error:  # noqa: BLE001 - durable failure receipt
                    failure = unit_error
                    terminal_failure = (
                        isinstance(unit_error, ReconciliationError)
                        and str(unit_error) == "user row exceeds the request limit"
                    )
                    break
            else:
                raise _ConcurrentMemoryUpdate

            if failure is not None:
                summary = (
                    str(failure)
                    if isinstance(failure, ReconciliationError)
                    else type(failure).__name__
                )
                recorded = await asyncio.to_thread(
                    self.state.record_failure,
                    seq_start=selected_start,
                    seq_end=selected_end,
                    validation_summary=summary,
                    reason=reason,
                    usage=usage,
                    retry_backoff_seconds=self.config.retry_backoff_seconds,
                    now=self._retry_clock(),
                    occurred_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    end_offset=end_offset,
                    end_tokens=end_tokens,
                    terminal=terminal_failure,
                )
                await asyncio.to_thread(self._append_failure_log, recorded)
                self._publish_failure_notice()
            else:
                await asyncio.to_thread(
                    self.state.record_success,
                    seq_start=selected_start,
                    seq_end=selected_end,
                    reason=reason,
                    end_offset=end_offset,
                    end_tokens=end_tokens,
                )
                if changed and self.notice is not None:
                    details = ", ".join(f"{name} (+1)" for name in changed)
                    self.notice(f"memory updated: {details}")
            cursor = selected_end + 1

    async def _invoke_request(
        self, prompt: str, budget: ProviderRetryBudget
    ) -> tuple[str, dict[str, int]]:
        if self._closing:
            raise asyncio.CancelledError
        with use_retry_budget(budget):
            raw = self.invoke(prompt)
            if inspect.isawaitable(raw):
                raw = await raw
        if isinstance(raw, ReconciliationResponse):
            return raw.text, dict(raw.usage)
        return raw, {}

    @staticmethod
    def _sum_usage(
        total: dict[str, int], addition: dict[str, int]
    ) -> dict[str, int]:
        result = dict(total)
        for key, value in addition.items():
            if type(value) is int and value >= 0:
                result[key] = result.get(key, 0) + value
        return result

    @staticmethod
    def _repair_prompt(prompt: str, error: ReconciliationError) -> str:
        message = str(error).replace("\n", " ")[:240]
        return (
            f"Your prior response failed validation: {message}. "
            "Return a corrected JSON object. Cite only top-level seq values shown "
            f"in Completed transcript rows.\n{prompt}"
        )

    def _publish_failure_notice(self) -> None:
        if (
            self.notice is not None
            and not self._closing
            and self.last_failure is not None
        ):
            self.notice(f"memory update failed: {self.last_failure.status_line()}")

    def _append_failure_log(self, failure: ReconciliationFailure) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "occurred_at": failure.occurred_at,
                "message": failure.message,
                "seq_start": failure.seq_start,
                "seq_end": failure.seq_end,
                "terminal": failure.terminal,
            },
            sort_keys=True,
        ).encode() + b"\n"
        try:
            previous = self.failure_log_path.read_bytes()
        except FileNotFoundError:
            previous = b""
        records = (previous + line).splitlines(keepends=True)
        payload = b""
        for record in reversed(records):
            if payload and len(record) + len(payload) > _MAX_FAILURE_LOG_BYTES:
                break
            payload = record + payload
        fd, temporary = tempfile.mkstemp(
            prefix=".memory-reconcile-errors-", dir=self.session_dir
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, self.failure_log_path)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _transcript_state(self) -> tuple[int, int, int]:
        path = self.session_dir / "conversation.jsonl"
        latest = 0
        tokens = 0
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    seq = value.get("seq") if isinstance(value, dict) else None
                    if type(seq) is int:
                        latest = max(latest, seq)
                        tokens += estimated_text_tokens(raw.decode("utf-8"))
                return latest, handle.tell(), tokens
        except FileNotFoundError:
            return 0, 0, 0

    def _transcript_chunk(
        self, start: int, end: int
    ) -> tuple[list[dict[str, object]], int, int]:
        path = self.session_dir / "conversation.jsonl"
        rows: list[dict[str, object]] = []
        size = 0
        end_offset = self._last_reconciled_bytes
        tokens = 0
        end_tokens = self._last_reconciled_tokens
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    offset = handle.tell()
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    seq = value.get("seq") if isinstance(value, dict) else None
                    if type(seq) is not int:
                        continue
                    tokens += estimated_text_tokens(raw.decode("utf-8"))
                    if seq < start:
                        continue
                    if seq > end:
                        break
                    projected = project_transcript_row(value)
                    projected_size = len(
                        json.dumps(projected, ensure_ascii=False).encode("utf-8")
                    )
                    if rows and size + projected_size > _MAX_TRANSCRIPT_CHUNK_BYTES:
                        break
                    rows.append(projected)
                    size += projected_size
                    end_offset = offset
                    end_tokens = tokens
        except FileNotFoundError:
            pass
        return rows, end_offset, end_tokens
