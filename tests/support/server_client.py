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
    buffered = _buffer(reader)
    while True:
        line = await asyncio.wait_for(reader.readline(), TIMEOUT)
        assert line, "server closed the socket"
        frame = json.loads(line)
        if frame.get("id") == request_id:
            return [frame]
        buffered.append(frame)


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
