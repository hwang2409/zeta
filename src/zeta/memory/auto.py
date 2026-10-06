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
_LOG = logging.getLogger(__name__)
_MAX_TRANSCRIPT_CHUNK_BYTES = 96 * 1024
_MAX_REQUEST_BYTES = 64 * 1024


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
    ) -> None:
        self.registry = registry
        self.project_id = project_id
        self.session_id = session_id
        self.session_dir = Path(session_dir)
        self.invoke = invoke
        self.config = config or AutoMemoryConfig()
        self.notice = notice
        self.position_path = self.session_dir / "memory-reconcile.json"
        position = self._read_position()
        self.last_reconciled_seq = position["seq"]
        self._last_reconciled_bytes = position["transcript_bytes"]
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

    def observe_tokens(self, _total_tokens: int) -> None:
        """Compatibility callback; durable transcript activity owns growth."""
        self.activity()

    def activity(self, seq: int | None = None) -> None:
        """Report a transcript append after it has been durably persisted."""
        del seq  # The worker reads the authoritative durable sequence itself.
        if not self.config.enabled or self._closing:
            return
        self._activity_generation += 1
        self._idle_deadline = asyncio.get_running_loop().time() + self.config.idle_seconds
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

    def _add_pending(self, start: int, end: int, reason: str) -> None:
        merged = _PendingRange(start, end, {reason})
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
                latest_seq, transcript_bytes = await asyncio.to_thread(self._transcript_state)
                self._seen_activity_generation = generation
                growth = max(0, transcript_bytes - self._last_reconciled_bytes)
                if growth >= self.config.token_threshold * 4:
                    self._add_pending(self.last_reconciled_seq + 1, latest_seq, "tokens")
            if self._pending:
                item = self._pending.pop(0)
                delay = self.config.minimum_interval - (
                    time.monotonic() - self._last_request_finished
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                self._busy = True
                try:
                    await self._reconcile_range(item)
                except Exception as exc:  # noqa: BLE001 - isolate background work
                    self.last_error = exc
                    _LOG.warning(
                        "automatic memory reconciliation failed (%s)",
                        type(exc).__name__[:80],
                    )
                finally:
                    self._busy = False
                    self._last_request_finished = time.monotonic()
                    self._idle_deadline = (
                        asyncio.get_running_loop().time() + self.config.idle_seconds
                    )
                continue
            self._drained.set()
            if self._closing:
                return
            self._wake.clear()
            if self._activity_generation != self._seen_activity_generation:
                continue
            timeout = None
            if self._idle_deadline is not None:
                timeout = max(0.0, self._idle_deadline - asyncio.get_running_loop().time())
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=timeout)
            except TimeoutError:
                latest_seq, _ = await asyncio.to_thread(self._transcript_state)
                if latest_seq > self.last_reconciled_seq:
                    self._add_pending(self.last_reconciled_seq + 1, latest_seq, "idle")
                    self._drained.clear()
                self._idle_deadline = (
                    asyncio.get_running_loop().time() + self.config.idle_seconds
                )

    async def _reconcile_range(self, item: _PendingRange) -> None:
        cursor = item.start
        while cursor <= item.end:
            rows, end_offset = await asyncio.to_thread(
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
                request = prepare_request(
                    raw_transcript,
                    snapshot.contents,
                    as_of=today,
                    max_bytes=_MAX_REQUEST_BYTES,
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
                history_count = len(
                    await asyncio.to_thread(self.registry.memory_log, self.project_id)
                )
                try:
                    await asyncio.to_thread(
                        self.registry.compare_and_swap_memory,
                        self.project_id,
                        expected_digest=proposal.base_digest,
                        updates=updates,
                        provenance=provenance,
                    )
                except ProjectRegistryError as exc:
                    if "digest mismatch" in str(exc) and attempt + 1 < self.config.cas_retries:
                        continue
                    raise ReconciliationError(
                        "project memory kept changing during reconciliation"
                    ) from exc
                if len(await asyncio.to_thread(self.registry.memory_log, self.project_id)) > history_count:
                    changed = tuple(updates)
                break
            else:
                raise ReconciliationError("project memory kept changing during reconciliation")
            self.last_reconciled_seq = max(self.last_reconciled_seq, selected_end)
            # The offset was captured before the provider call. New appends during
            # that call therefore remain beyond the durable reconciled position.
            if selected_end >= int(rows[-1]["seq"]):
                self._last_reconciled_bytes = max(self._last_reconciled_bytes, end_offset)
            self._write_position("+".join(sorted(item.reasons)))
            if changed and self.notice is not None:
                details = ", ".join(f"{name} (+1)" for name in changed)
                self.notice(f"memory updated: {details}")
            cursor = selected_end + 1

    def _transcript_state(self) -> tuple[int, int]:
        path = self.session_dir / "conversation.jsonl"
        latest = 0
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
                return latest, handle.tell()
        except FileNotFoundError:
            return 0, 0

    def _transcript_chunk(self, start: int, end: int) -> tuple[list[dict[str, object]], int]:
        path = self.session_dir / "conversation.jsonl"
        rows: list[dict[str, object]] = []
        size = 0
        end_offset = self._last_reconciled_bytes
        try:
            with path.open("rb") as handle:
                for raw in handle:
                    offset = handle.tell()
                    try:
                        value = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    seq = value.get("seq") if isinstance(value, dict) else None
                    if type(seq) is not int or seq < start:
                        continue
                    if seq > end:
                        break
                    if rows and size + len(raw) > _MAX_TRANSCRIPT_CHUNK_BYTES:
                        break
                    rows.append(value)
                    size += len(raw)
                    end_offset = offset
        except FileNotFoundError:
            pass
        return rows, end_offset

    def _read_position(self) -> dict[str, int]:
        try:
            value = json.loads(self.position_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"seq": 0, "transcript_bytes": 0}
        if not isinstance(value, Mapping):
            return {"seq": 0, "transcript_bytes": 0}
        seq, transcript_bytes = value.get("seq"), value.get("transcript_bytes")
        if (
            type(seq) is not int
            or seq < 0
            or type(transcript_bytes) is not int
            or transcript_bytes < 0
        ):
            return {"seq": 0, "transcript_bytes": 0}
        return {"seq": seq, "transcript_bytes": transcript_bytes}

    def _write_position(self, reason: str) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "session_id": self.session_id,
                "seq": self.last_reconciled_seq,
                "transcript_bytes": self._last_reconciled_bytes,
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
