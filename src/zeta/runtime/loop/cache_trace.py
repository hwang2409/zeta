"""Opt-in, content-free prompt-cache traces for agent completions."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...protocol.types import Message, StreamEvent, StreamEventType, ToolSchema
from ...providers.stream_diagnostics import write_stream_diagnostic


def _message_fingerprint(message: Message) -> bytes:
    value = message.to_dict()
    replay = message.metadata.get("codex_output_items")
    if replay is None:
        value.pop("metadata", None)
    else:
        value["metadata"] = {"codex_output_items": replay}
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).digest()


class CacheTrace:
    @classmethod
    def from_environment(cls, session_id: str, agent_depth: int) -> CacheTrace | None:
        if os.environ.get("ZETA_CACHE_TRACE") != "1":
            return None
        path = (
            Path(os.environ.get("ZETA_HOME", Path.home() / ".zeta"))
            / "logs"
            / "cache-trace.jsonl"
        )
        return cls(path, session_id, agent_depth)

    def __init__(self, path: Path, session_id: str, agent_depth: int) -> None:
        self.path = path
        self.session_id = session_id
        self.agent_depth = agent_depth
        self.previous_messages: tuple[bytes, ...] | None = None
        self.previous_tools: bytes | None = None
        self.previous_at: float | None = None
        self.pending_messages: tuple[bytes, ...] = ()
        self.pending_tools: bytes = b""
        self.pending_at: float = 0

    def start(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema],
        backend: object,
        turn: int,
        plan_mode: bool,
        compacted: bool,
    ) -> dict[str, Any]:
        now = time.time()
        # ponytail: opt-in O(history) scan; incremental hashes only if tracing becomes hot.
        fingerprints = tuple(_message_fingerprint(message) for message in messages)
        tools_fingerprint = hashlib.sha256(
            json.dumps(
                sorted(tools, key=lambda tool: tool["name"]), sort_keys=True
            ).encode()
        ).digest()
        model = getattr(backend, "model", None)
        shared = None
        if self.previous_messages is not None:
            shared = 0
            for previous, current in zip(self.previous_messages, fingerprints):
                if previous != current:
                    break
                shared += 1
        record = {
            "timestamp": now,
            "session_id": self.session_id,
            "agent_depth": self.agent_depth,
            "provider": type(backend).__name__,
            "model": model if type(model) is str else None,
            "turn": turn,
            "plan_mode": plan_mode,
            "compacted": compacted,
            "message_count": len(messages),
            "tool_count": len(tools),
            "same_tools": (
                None
                if self.previous_tools is None
                else self.previous_tools == tools_fingerprint
            ),
            "shared_prefix_messages": shared,
            "gap_seconds": None
            if self.previous_at is None
            else max(0, now - self.previous_at),
        }
        self.pending_messages = fingerprints
        self.pending_tools = tools_fingerprint
        self.pending_at = now
        return record

    def finish(
        self,
        record: Mapping[str, Any],
        usage: Mapping[str, Any] | None,
        *,
        truncated: bool,
    ) -> None:
        usage = usage or {}
        counts = {
            "uncached_input_tokens": usage.get("input_tokens"),
            "cache_read_tokens": usage.get("cache_read_input_tokens"),
            "cache_write_tokens": usage.get("cache_creation_input_tokens"),
            "output_tokens": usage.get("output_tokens"),
        }
        counts = {
            key: value if type(value) is int and value >= 0 else None
            for key, value in counts.items()
        }
        total = sum(
            counts[key] or 0
            for key in (
                "uncached_input_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
            )
        )
        write_stream_diagnostic(
            self.path,
            {
                **record,
                **counts,
                "duration_seconds": max(0, time.time() - record["timestamp"]),
                "cache_hit_rate": (
                    100 * (counts["cache_read_tokens"] or 0) / total if total else None
                ),
                "truncated": truncated,
            },
        )

    def observe(self, record: Mapping[str, Any], event: StreamEvent) -> None:
        if event.type is not StreamEventType.MESSAGE_END:
            return
        usage = event.data.get("usage")
        self.finish(
            record,
            usage if isinstance(usage, Mapping) else None,
            truncated=bool(event.data.get("truncated")),
        )
        if not event.data.get("truncated"):
            self.previous_messages = self.pending_messages
            self.previous_tools = self.pending_tools
            self.previous_at = self.pending_at
