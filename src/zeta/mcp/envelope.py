"""Bounded accumulation for MCP responses."""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from functools import partial
from typing import BinaryIO

from ..core.thread_call import run_owned_thread_call
from ..tools._spill import SpillStore

_SPILL_WRITE_SIZE = 64 * 1024
_MEBIBYTE = 1024 * 1024


def _format_wire_limit(size: int) -> str:
    if size % _MEBIBYTE == 0:
        return f"{size // _MEBIBYTE} MiB"
    return f"{size} bytes"


def _format_deadline(seconds: float) -> str:
    return f"{seconds:g}s"


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
        self._spill_buffer = bytearray()
        self._reserved = 0

    async def append(self, chunk: bytes) -> None:
        if self.total + len(chunk) > self.limits.max_wire_bytes:
            raise MCPWireLimitError
        self.total += len(chunk)
        if self._handle is None and self.total <= self.limits.spill_after_bytes:
            self._parts.append(chunk)
            return
        if self._handle is None:
            await self._reserve(self.total)
            temporary = self.spill_store.temporary_file()
            self._handle = await run_owned_thread_call(
                temporary.__enter__,
                lambda _handle: temporary.__exit__(None, None, None),
                close_is_thread_safe=True,
            )
            self._temporary = temporary
            self._spill_buffer.extend(b"".join(self._parts))
            self._parts.clear()
        extra = self.total - self._reserved
        if extra:
            await self._reserve(extra)
        self._spill_buffer.extend(chunk)
        if len(self._spill_buffer) >= _SPILL_WRITE_SIZE:
            await self._flush_spill()

    async def _reserve(self, byte_count: int) -> None:
        """Reserve off-loop; a reservation that lands after cancellation is released."""

        try:
            await run_owned_thread_call(
                partial(self.spill_store.reserve_transient, byte_count),
                lambda _result: self.spill_store.release_transient(byte_count),
                close_is_thread_safe=True,
            )
        except RuntimeError as exc:
            raise MCPTransientQuotaError from exc
        self._reserved += byte_count

    async def _flush_spill(self) -> None:
        if self._handle is None or not self._spill_buffer:
            return
        chunk = bytes(self._spill_buffer)
        self._spill_buffer.clear()
        await asyncio.to_thread(self._handle.write, chunk)

    async def finish_json(self) -> object:
        try:
            if self._handle is None:
                return await asyncio.to_thread(json.loads, b"".join(self._parts))
            await self._flush_spill()
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
        self._spill_buffer.clear()
        try:
            if temporary is not None:
                await asyncio.to_thread(temporary.__exit__, None, None, None)
        finally:
            if reserved:
                self.spill_store.release_transient(reserved)


__all__ = [
    "MCPCallEnvelope",
    "MCPCallLimits",
    "MCPTransientQuotaError",
    "MCPWireLimitError",
]
