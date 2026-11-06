"""Bounded accumulation for MCP HTTP responses."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from typing import BinaryIO

from ..tools._spill import SpillStore


@dataclass(frozen=True, slots=True)
class MCPCallLimits:
    deadline_seconds: float = 300.0
    max_wire_bytes: int = 64 * 1024 * 1024
    spill_after_bytes: int = 256 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be finite and positive")
        if self.max_wire_bytes <= 0:
            raise ValueError("max_wire_bytes must be positive")
        if self.spill_after_bytes <= 0 or self.spill_after_bytes > self.max_wire_bytes:
            raise ValueError("spill_after_bytes must be between 1 and max_wire_bytes")


class MCPWireLimitError(RuntimeError):
    """The decoded response exceeded the call wire limit."""


class MCPTransientQuotaError(RuntimeError):
    """The session could not reserve transient spill space."""


class MCPCallEnvelope:
    """Charge and retain one decoded response, keeping blocking work off-loop."""

    def __init__(self, limits: MCPCallLimits, spill_store: SpillStore) -> None:
        self.limits = limits
        self.spill_store = spill_store
        self.total = 0
        self._parts: list[bytes] = []
        self._temporary = None
        self._handle: BinaryIO | None = None
        self._reserved = 0

    async def append(self, chunk: bytes) -> None:
        if self.total + len(chunk) > self.limits.max_wire_bytes:
            raise MCPWireLimitError
        self.total += len(chunk)
        if self._handle is None and self.total <= self.limits.spill_after_bytes:
            self._parts.append(chunk)
            return
        if self._handle is None:
            try:
                self.spill_store.reserve_transient(self.total)
            except RuntimeError as exc:
                raise MCPTransientQuotaError from exc
            self._reserved = self.total
            self._temporary = self.spill_store.temporary_file()
            self._handle = await asyncio.to_thread(self._temporary.__enter__)
            prefix = b"".join(self._parts)
            self._parts.clear()
            await asyncio.to_thread(self._handle.write, prefix)
        extra = self.total - self._reserved
        if extra:
            try:
                self.spill_store.reserve_transient(extra)
            except RuntimeError as exc:
                raise MCPTransientQuotaError from exc
            self._reserved += extra
        await asyncio.to_thread(self._handle.write, chunk)

    async def finish_json(self) -> object:
        try:
            if self._handle is None:
                return await asyncio.to_thread(json.loads, b"".join(self._parts))
            await asyncio.to_thread(self._handle.seek, 0)
            return await asyncio.to_thread(json.load, self._handle)
        finally:
            await self.close()

    async def close(self) -> None:
        temporary = self._temporary
        reserved = self._reserved
        self._temporary = None
        self._handle = None
        self._reserved = 0
        self._parts.clear()
        try:
            if temporary is not None:
                await asyncio.to_thread(temporary.__exit__, None, None, None)
        finally:
            if reserved:
                self.spill_store.release_transient(reserved)


__all__ = ["MCPCallEnvelope", "MCPCallLimits", "MCPTransientQuotaError", "MCPWireLimitError"]
