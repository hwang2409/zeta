"""Derived, asynchronous summaries for eviction receipts."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..memory.safety import contains_secret, redact_secrets

SUMMARY_INSTRUCTION = (
    "summarize this history for exact later factual recall. preserve pr numbers, "
    "branches, sha prefixes, verdicts, blockers, dependencies, user decisions, "
    "and promises. output at most 512 utf-8 bytes.\n\n"
)
_PROMPT_VERSION = 1
_RECORD_VERSION = 1


def sanitize_source(source: str) -> str:
    """Remove secrets and untrusted control markup before model input."""
    value = redact_secrets(source)
    value = re.sub(r"<\/?(?:tool|system|assistant|user)[^>]*>", " ", value, flags=re.IGNORECASE)
    return value[:65536]


def valid_summary(value: str) -> bool:
    return bool(value) and not contains_secret(value) and not re.search(
        r"<\/?(?:tool|system|assistant|user)[^>]*>|ignore previous|you are now",
        value,
        re.IGNORECASE,
    )


def trim_summary(value: str) -> str:
    data = value.encode("utf-8")[:512]
    return data.decode("utf-8", "ignore").strip()


@dataclass(frozen=True, slots=True)
class SummaryKey:
    kind: str
    seq_start: int
    seq_end: int
    source_sha256: str
    prompt_version: int = _PROMPT_VERSION

    def as_string(self) -> str:
        return f"{self.prompt_version}:{self.kind}:{self.seq_start}:{self.seq_end}:{self.source_sha256}"


class SummaryCache:
    """A tolerant JSONL cache. malformed records are independent misses."""
    def __init__(self, session_dir: str | Path):
        self.path = Path(session_dir) / "eviction-summaries.jsonl"
        self.records: dict[str, str] = {}
        self._load()
        self._lock = asyncio.Lock()

    def _load(self) -> None:
        try:
            lines = self.path.read_text().splitlines()
        except (FileNotFoundError, OSError):
            return
        for line in lines:
            try:
                row = json.loads(line)
                key = SummaryKey(str(row["kind"]), int(row["seq_start"]), int(row["seq_end"]), str(row["source_sha256"]), int(row["prompt_version"]))
                summary = str(row["summary"])
                if row.get("version") == _RECORD_VERSION and valid_summary(summary):
                    self.records[key.as_string()] = trim_summary(summary)
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue

    def get(self, key: SummaryKey) -> str | None:
        return self.records.get(key.as_string())

    async def put(self, key: SummaryKey, summary: str, *, input_tokens: int = 0) -> bool:
        summary = trim_summary(summary)
        if not valid_summary(summary):
            return False
        row = {"version": 1, "prompt_version": key.prompt_version, "kind": key.kind, "seq_start": key.seq_start, "seq_end": key.seq_end, "source_sha256": key.source_sha256, "summary": summary, "input_tokens": input_tokens, "output_tokens": len(summary), "created_at": time.time()}
        line = json.dumps(row, separators=(",", ":")) + "\n"
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            self.records[key.as_string()] = summary
        return True


class TokenBucket:
    def __init__(self, capacity: int = 150_000, refill_per_hour: int = 100_000, clock: Callable[[], float] = time.monotonic):
        self.capacity, self.rate, self.clock = capacity, refill_per_hour / 3600, clock
        self.tokens, self.updated = capacity, clock()
        self.lock = asyncio.Lock()

    async def reserve(self, amount: int) -> bool:
        async with self.lock:
            now = self.clock()
            self.tokens = min(self.capacity, self.tokens + max(0.0, now - self.updated) * self.rate)
            self.updated = now
            if amount > self.tokens:
                return False
            self.tokens -= amount
            return True


class EvictionSummaryWorker:
    """Coalescing bounded worker; callers provide already projected source."""
    def __init__(self, cache: SummaryCache, provider: Callable[[str], Awaitable[Any]], *, max_pending: int = 256, clock: Callable[[], float] = time.monotonic):
        self.cache, self.provider, self.max_pending, self.clock = cache, provider, max_pending, clock
        self.queue: OrderedDict[SummaryKey, str] = OrderedDict()
        self.running: set[asyncio.Task[Any]] = set()
        self.bucket = TokenBucket(clock=clock)
        self.closed = False
        self.drops = 0

    def admit(self, key: SummaryKey, source: str, *, exact_range: bool = False) -> bool:
        if self.closed or self.cache.get(key) is not None:
            return False
        if key in self.queue:
            self.queue[key] = source
            return True
        if len(self.queue) >= self.max_pending:
            self.drops += 1
            return False
        if exact_range:
            self.queue = OrderedDict([(key, source), *self.queue.items()])
        else:
            self.queue[key] = source
        return True

    async def run_once(self) -> None:
        if self.closed:
            return
        while self.queue and len(self.running) < 2:
            key, source = self.queue.popitem(last=False)
            task = asyncio.create_task(self._call(key, source))
            self.running.add(task)
            task.add_done_callback(self.running.discard)
        if self.running:
            await asyncio.gather(*tuple(self.running), return_exceptions=True)

    async def _call(self, key: SummaryKey, source: str) -> None:
        prompt = SUMMARY_INSTRUCTION + sanitize_source(source)
        estimate = max(1, len(prompt) // 4)
        if not await self.bucket.reserve(estimate):
            self.admit(key, source)
            return
        try:
            result = await self.provider(prompt)
            text = result if isinstance(result, str) else str(getattr(result, "text", result))
            await self.cache.put(key, text, input_tokens=estimate)
        except (OSError, RuntimeError, ValueError, TypeError):
            return

    async def close(self, grace: float = 2.5) -> None:
        self.closed = True
        if self.running:
            _, pending = await asyncio.wait(self.running, timeout=grace)
            for task in pending:
                task.cancel()


def source_hash(source: str) -> str:
    return hashlib.sha256((SUMMARY_INSTRUCTION + sanitize_source(source)).encode()).hexdigest()
