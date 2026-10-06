"""Background orchestration for automatic project-memory reconciliation.

``AutoMemoryReconciler`` is the small runtime seam. Callers report token growth,
activity, and an impending eviction range. This module owns transcript slicing,
serialization, compare-and-swap retries, durable progress, and notices.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..project_registry import ProjectRegistry, ProjectRegistryError
from .reconciler import (
    ReconciliationError,
    ReconciliationResponse,
    Transcript,
    build_prompt,
    memory_digest,
    parse_proposal,
)

InvokeResult = str | ReconciliationResponse
Invoke = Callable[[str], InvokeResult | Awaitable[InvokeResult]]
Notice = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AutoMemoryConfig:
    """Resolved automatic-memory settings for one session."""

    enabled: bool = True
    model: str = "gpt-5.6-luna"
    token_threshold: int = 50_000
    idle_seconds: float = 600.0
    cas_retries: int = 3

    def __post_init__(self) -> None:
        if self.token_threshold < 1:
            raise ValueError("memory token threshold must be positive")
        if self.idle_seconds <= 0:
            raise ValueError("memory idle seconds must be positive")
        if self.cas_retries < 1:
            raise ValueError("memory CAS retries must be positive")


class AutoMemoryReconciler:
    """Reconcile transcript ranges without blocking the active turn."""

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
        self._last_reconciled_tokens = position["tokens"]
        self._observed_tokens = self._last_reconciled_tokens
        self._tasks: set[asyncio.Task[None]] = set()
        self._idle_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self.last_error: Exception | None = None

    def observe_tokens(self, total_tokens: int) -> None:
        """Schedule reconciliation after configured new-token growth."""
        if not self.config.enabled or total_tokens < 0:
            return
        self._observed_tokens = max(self._observed_tokens, total_tokens)
        if total_tokens - self._last_reconciled_tokens >= self.config.token_threshold:
            self._schedule(self.last_reconciled_seq + 1, self._latest_seq(), "tokens")

    def activity(self) -> None:
        """Reset the non-blocking idle trigger after transcript activity."""
        if not self.config.enabled:
            return
        if self._idle_task is not None:
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._after_idle())

    def before_eviction(self, seq_start: int, seq_end: int) -> None:
        """Schedule exactly the transcript range an assembler will evict."""
        if not self.config.enabled:
            return
        self._schedule(seq_start, seq_end, "eviction")

    async def drain(self) -> None:
        """Wait for currently scheduled reconciliations; used by shutdown/tests."""
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        if self.last_error is not None:
            error, self.last_error = self.last_error, None
            raise error

    async def close(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        try:
            await self.drain()
        except (OSError, RuntimeError, ValueError):
            # Reconciliation is best effort and must not make session shutdown fail.
            pass

    async def _after_idle(self) -> None:
        try:
            await asyncio.sleep(self.config.idle_seconds)
            self._schedule(
                self.last_reconciled_seq + 1, self._latest_seq(), "idle"
            )
        except asyncio.CancelledError:
            return

    def _schedule(self, seq_start: int, seq_end: int, reason: str) -> None:
        if seq_end < seq_start:
            return
        task = asyncio.create_task(self._run(seq_start, seq_end, reason))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, seq_start: int, seq_end: int, reason: str) -> None:
        async with self._lock:
            try:
                rows = self._transcript_rows(seq_start, seq_end)
                if not rows:
                    return
                transcript = Transcript(self.session_id, tuple(rows))
                provenance = {
                    "session_id": self.session_id,
                    "seq_start": rows[0]["seq"],
                    "seq_end": rows[-1]["seq"],
                }
                changed: tuple[str, ...] = ()
                for attempt in range(self.config.cas_retries):
                    memory = dict(self.registry.load_memory(self.project_id))
                    today = datetime.now(UTC).date()
                    prompt = build_prompt(transcript, memory, as_of=today)
                    raw = self.invoke(prompt)
                    if inspect.isawaitable(raw):
                        raw = await raw
                    if isinstance(raw, ReconciliationResponse):
                        provenance["model"] = self.config.model
                        provenance["usage"] = dict(raw.usage)
                        raw_text = raw.text
                    else:
                        raw_text = raw
                    proposal = parse_proposal(
                        raw_text,
                        expected_digest=memory_digest(memory),
                        transcript=transcript,
                        as_of=today,
                    )
                    updates = {
                        replacement.name: replacement.content
                        for replacement in proposal.replacements
                    }
                    if not updates:
                        break
                    history_count = len(self.registry.memory_log(self.project_id))
                    try:
                        self.registry.compare_and_swap_memory(
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
                    if len(self.registry.memory_log(self.project_id)) > history_count:
                        changed = tuple(updates)
                    break
                else:
                    raise ReconciliationError(
                        "project memory kept changing during reconciliation"
                    )
                self.last_reconciled_seq = max(
                    self.last_reconciled_seq, rows[-1]["seq"]
                )
                self._last_reconciled_tokens = self._observed_tokens
                self._write_position(reason)
                if changed and self.notice is not None:
                    details = ", ".join(f"{name} (+1)" for name in changed)
                    self.notice(f"memory updated: {details}")
            except Exception as exc:  # noqa: BLE001 - background failures are retained
                self.last_error = exc

    def _latest_seq(self) -> int:
        rows = self._transcript_rows(1, 2**63 - 1)
        return rows[-1]["seq"] if rows else 0

    def _transcript_rows(self, start: int, end: int) -> list[dict[str, object]]:
        path = self.session_dir / "conversation.jsonl"
        rows: list[dict[str, object]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return rows
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = value.get("seq") if isinstance(value, dict) else None
            if type(seq) is int and start <= seq <= end:
                rows.append(value)
        return rows

    def _read_position(self) -> dict[str, int]:
        try:
            value = json.loads(self.position_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {"seq": 0, "tokens": 0}
        if not isinstance(value, Mapping):
            return {"seq": 0, "tokens": 0}
        seq, tokens = value.get("seq"), value.get("tokens")
        if type(seq) is not int or seq < 0 or type(tokens) is not int or tokens < 0:
            return {"seq": 0, "tokens": 0}
        return {"seq": seq, "tokens": tokens}

    def _write_position(self, reason: str) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                "session_id": self.session_id,
                "seq": self.last_reconciled_seq,
                "tokens": self._last_reconciled_tokens,
                "reason": reason,
            },
            sort_keys=True,
        ).encode()
        fd, temporary = tempfile.mkstemp(
            prefix=".memory-reconcile-", dir=self.session_dir
        )
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
