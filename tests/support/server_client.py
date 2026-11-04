"""Buffered JSON-RPC test client helpers."""

from __future__ import annotations

import asyncio
import json
import weakref
from collections import deque
from typing import Any

TIMEOUT = 30
_buffers: weakref.WeakKeyDictionary[asyncio.StreamReader, deque[dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def _buffer(reader: asyncio.StreamReader) -> deque[dict[str, Any]]:
    return _buffers.setdefault(reader, deque())


async def read(reader: asyncio.StreamReader) -> dict[str, Any]:
    buffered = _buffer(reader)
    if buffered:
        return buffered.popleft()
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line, "server closed the socket"
    return json.loads(line)


async def request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: str | int,
    method: str,
    params: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    writer.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            + "\n"
        ).encode()
    )
    await asyncio.wait_for(writer.drain(), TIMEOUT)
    frames: list[dict[str, Any]] = []
    buffered = _buffer(reader)
    prior = list(buffered)
    buffered.clear()
    new_frames: list[dict[str, Any]] = []
    while True:
        frame = await read(reader)
        frames.append(frame)
        new_frames.append(frame)
        if frame.get("id") == request_id:
            # Keep newly read notifications available to the next event helper.
            # Callers may inspect returned frames, but must not consume those
            # same notifications through an event helper.
            buffered.extend(prior)
            buffered.extend(
                candidate for candidate in new_frames if candidate.get("id") != request_id
            )
            return frames


async def frames_until_event(
    reader: asyncio.StreamReader, name: str
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        frame = await read(reader)
        frames.append(frame)
        if frame.get("params", {}).get("event") == name:
            return frames


async def event(reader: asyncio.StreamReader, name: str) -> dict[str, Any]:
    frames = await frames_until_event(reader, name)
    return frames[-1]["params"]


def discard_buffered(reader: asyncio.StreamReader, frames: list[dict[str, Any]]) -> None:
    """Drop notifications a caller already consumed from a request result."""
    buffered = _buffer(reader)
    for frame in (frame for frame in frames if frame.get("id") is None):
        try:
            buffered.remove(frame)
        except ValueError:
            pass
