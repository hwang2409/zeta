"""One coalescing worker for automatic project-memory reconciliation."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..context_eviction import estimated_text_tokens
from ..project_registry import ProjectRegistry, ProjectRegistryError
from .reconciler import (
    ReconciliationError,
    ReconciliationResponse,
    Transcript,
    parse_proposal,
    prepare_request,
)

InvokeResult = str | ReconciliationResponse
Invoke = Callable[[str], InvokeResult | Awaitable[InvokeResult]]
Notice = Callable[[str], None]
Clock = Callable[[], float]
IdleWait = Callable[[asyncio.Event, float], Awaitable[None]]
_LOG = logging.getLogger(__name__)


async def _wait_for_idle(wake: asyncio.Event, timeout: float) -> None:
    await asyncio.wait_for(wake.wait(), timeout=timeout)
_MAX_TRANSCRIPT_CHUNK_BYTES = 96 * 1024
_MAX_REQUEST_BYTES = 64 * 1024


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

    def __post_init__(self) -> None:
        if self.token_threshold < 1:
            raise ValueError("memory token threshold must be positive")
        if self.idle_seconds <= 0:
            raise ValueError("memory idle seconds must be positive")
        if self.cas_retries < 1:
            raise ValueError("memory CAS retries must be positive")
        if self.minimum_interval < 0:
            raise ValueError("memory minimum interval cannot be negative")


@dataclass(slots=True)
class _PendingRange:
    start: int
    end: int
    reasons: set[str]


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
        self.position_path = self.session_dir / "memory-reconcile.json"
        position = self._read_position()
        self.last_reconciled_seq = position["seq"]
        self._last_reconciled_bytes = position["transcript_bytes"]
        self._last_reconciled_tokens = position["transcript_tokens"]
        self._fragment_seq = position["fragment_seq"]
        self._fragment_offset = position["fragment_offset"]
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
        self.last_error: Exception | None = None
        self._conflict_retries = 0

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
        while self._busy or self._pending or self._seen_activity_generation < self._activity_generation:
            await self._drained.wait()
            if self._busy or self._pending:
                self._drained.clear()
        if self.last_error is not None:
            error, self.last_error = self.last_error, None
            raise error

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        task = self._worker_task
        if task is not None:
            try:
                await task
            except (OSError, RuntimeError, ValueError):
                # Reconciliation is best effort and never breaks shutdown.
                pass

    def _ensure_worker(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    def _add_pending(self, start: int, end: int, *reasons: str) -> None:
        merged = _PendingRange(start, end, set(reasons))
        remaining: list[_PendingRange] = []
        for item in self._pending:
            if item.end + 1 < merged.start or merged.end + 1 < item.start:
                remaining.append(item)
                continue
            merged.start = min(merged.start, item.start)
            merged.end = max(merged.end, item.end)
            merged.reasons.update(item.reasons)
        remaining.append(merged)
        self._pending = sorted(remaining, key=lambda item: item.start)

    async def _worker(self) -> None:
        while True:
            if self._closing and not self._pending and not self._busy:
                self._drained.set()
                return
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
                    self.last_error = exc
                    _LOG.warning(
                        "automatic memory reconciliation failed (%s)",
                        type(exc).__name__[:80],
                    )
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
        while cursor <= item.end:
            rows, end_offset, end_tokens = await asyncio.to_thread(
                self._transcript_chunk, cursor, item.end
            )
            if not rows:
                return
            raw_transcript = Transcript(self.session_id, tuple(rows))
            changed: tuple[str, ...] = ()
            selected_end = cursor - 1
            usage: dict[str, int] = {}
            for attempt in range(self.config.cas_retries):
                snapshot = await asyncio.to_thread(
                    self.registry.memory_snapshot, self.project_id
                )
                today = datetime.now(UTC).date()
                fragment_offset = (
                    self._fragment_offset
                    if self._fragment_seq == int(raw_transcript.rows[0]["seq"])
                    else 0
                )
                request = prepare_request(
                    raw_transcript,
                    snapshot.contents,
                    as_of=today,
                    max_bytes=_MAX_REQUEST_BYTES,
                    fragment_offset=fragment_offset,
                )
                if not request.transcript.rows:
                    return
                selected_end = int(request.transcript.rows[-1]["seq"])
                raw = self.invoke(request.prompt)
                if inspect.isawaitable(raw):
                    raw = await raw
                if isinstance(raw, ReconciliationResponse):
                    usage = dict(raw.usage)
                    raw_text = raw.text
                else:
                    raw_text = raw
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
                    "seq_start": int(request.transcript.rows[0]["seq"]),
                    "seq_end": selected_end,
                    "model": self.config.model,
                    "usage": usage,
                }
                if request.fragment is not None:
                    provenance.update(
                        fragment_start=request.fragment.start,
                        fragment_end=request.fragment.end,
                    )
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
            else:
                raise _ConcurrentMemoryUpdate
            if request.fragment is not None and not request.fragment.complete:
                self._fragment_seq = request.fragment.seq
                self._fragment_offset = request.fragment.end
            else:
                self.last_reconciled_seq = max(self.last_reconciled_seq, selected_end)
                self._fragment_seq = 0
                self._fragment_offset = 0
                # The offset was captured before the provider call. New appends during
                # that call therefore remain beyond the durable reconciled position.
                if selected_end >= int(rows[-1]["seq"]):
                    self._last_reconciled_bytes = max(
                        self._last_reconciled_bytes, end_offset
                    )
                    self._last_reconciled_tokens = max(
                        self._last_reconciled_tokens, end_tokens
                    )
            self._write_position("+".join(sorted(item.reasons)))
            if changed and self.notice is not None:
                details = ", ".join(f"{name} (+1)" for name in changed)
                self.notice(f"memory updated: {details}")
            if request.fragment is None or request.fragment.complete:
                cursor = selected_end + 1

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
                    if rows and size + len(raw) > _MAX_TRANSCRIPT_CHUNK_BYTES:
                        break
                    rows.append(value)
                    size += len(raw)
                    end_offset = offset
                    end_tokens = tokens
        except FileNotFoundError:
            pass
        return rows, end_offset, end_tokens

    def _read_position(self) -> dict[str, int]:
        empty = {
            "seq": 0,
            "transcript_bytes": 0,
            "transcript_tokens": 0,
            "fragment_seq": 0,
            "fragment_offset": 0,
        }
        try:
            value = json.loads(self.position_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return empty
        if not isinstance(value, Mapping):
            return empty
        position = {
            "seq": value.get("seq"),
            "transcript_bytes": value.get("transcript_bytes"),
            "transcript_tokens": value.get("transcript_tokens", 0),
            "fragment_seq": value.get("fragment_seq", 0),
            "fragment_offset": value.get("fragment_offset", 0),
        }
        if any(type(item) is not int or item < 0 for item in position.values()):
            return empty
        if bool(position["fragment_seq"]) != bool(position["fragment_offset"]):
            return empty
        return position

    def _write_position(self, reason: str) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "session_id": self.session_id,
                "seq": self.last_reconciled_seq,
                "transcript_bytes": self._last_reconciled_bytes,
                "transcript_tokens": self._last_reconciled_tokens,
                "fragment_seq": self._fragment_seq,
                "fragment_offset": self._fragment_offset,
                "reason": reason,
            },
            sort_keys=True,
        ).encode()
        fd, temporary = tempfile.mkstemp(prefix=".memory-reconcile-", dir=self.session_dir)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, self.position_path)
            directory_fd = os.open(self.session_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
