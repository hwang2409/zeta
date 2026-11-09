"""Real serve transports for replay protocol tests."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import secrets
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlencode

from tests.support.owned import await_or_fail
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer

Transport = Literal["unix", "websocket"]


def _lookup(params: dict[str, Any], dotted: str) -> Any:
    value: Any = params
    for part in dotted.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return value


def classify(frame: dict[str, Any]) -> tuple[str, object]:
    """Classify a wire frame by response id or replay stream."""
    if "id" in frame:
        return "response", frame["id"]
    params = frame.get("params", {})
    return "event", params.get("stream_id")


@dataclass
class ReplayClient:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    websocket: bool
    buffered: deque[dict[str, Any]] = field(default_factory=deque)

    async def send(self, request_id: int, method: str, params: dict[str, object]) -> None:
        value = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        payload = json.dumps(value, separators=(",", ":")).encode()
        self.writer.write(_masked_frame(payload) if self.websocket else payload + b"\n")
        await await_or_fail(self.writer.drain(), what=f"send {method}")

    async def read(self) -> dict[str, Any]:
        if self.buffered:
            return self.buffered.popleft()
        return await self._read_wire()

    async def _read_wire(self, *, what: str = "replay frame") -> dict[str, Any]:
        frame = await self._read_or_eof(what=what)
        assert frame is not None, "server closed replay connection"
        return frame

    async def _read_or_eof(
        self,
        *,
        what: str = "replay frame",
        received: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        received = list(self.buffered) if received is None else received
        describe = lambda: f"{what}; received={received!r}"
        # a write to a connection the server closed surfaces on the next read
        try:
            if not self.websocket:
                # test-contract: transport replay transport reads unix and websocket serve frames
                # test-contract: transport replay client reads raw unix frames, not standard JSON-RPC.
                line = await await_or_fail(
                    # test-contract: transport raw unix replay frame reader, not newline JSON-RPC client.
                    self.reader.readline(),
                    what=describe,
                )
                return json.loads(line) if line else None
            first, second = await await_or_fail(
                self.reader.readexactly(2),
                what=describe,
            )
        except (asyncio.IncompleteReadError, ConnectionError):
            return None
        if first & 0x0F == 8:
            return None
        assert first & 0x0F == 1 and second & 0x80 == 0
        size = second & 0x7F
        if size == 126:
            size = int.from_bytes(await self.reader.readexactly(2), "big")
        elif size == 127:
            size = int.from_bytes(await self.reader.readexactly(8), "big")
        return json.loads(await self.reader.readexactly(size))

    async def closed(self, request_id: int) -> list[dict[str, Any]]:
        """Send status; return the frames before the server closed, failing if it answers."""
        with contextlib.suppress(ConnectionError):
            await self.send(request_id, "status", {})
        frames = list(self.buffered)
        self.buffered.clear()
        while (frame := await self._read_or_eof()) is not None:
            assert frame.get("id") != request_id, "failed connection stayed open"
            frames.append(frame)
        return frames

    async def response(self, request_id: int) -> dict[str, Any]:
        skipped: list[dict[str, Any]] = []
        received = list(self.buffered)
        try:
            while True:
                frame = await self._read_or_eof(
                    what=f"response id {request_id}", received=received
                )
                assert frame is not None, "server closed replay connection"
                if frame.get("id") == request_id:
                    return frame
                skipped.append(frame)
                received.append(frame)
        finally:
            self.buffered.extend(skipped)

    async def request(self, request_id: int, method: str, params: dict[str, object]) -> dict[str, Any]:
        await self.send(request_id, method, params)
        return await self.response(request_id)

    async def event(
        self, name: str | None = None, match: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """The first matching event in arrival order; other frames stay buffered in order."""
        def matches(frame: dict[str, Any]) -> bool:
            params = frame.get("params")
            if not isinstance(params, dict) or (name is not None and params.get("event") != name):
                return False
            return all(_lookup(params, key) == value for key, value in (match or {}).items())

        # a match may sit behind other buffered frames (two events can arrive in either order)
        for index, frame in enumerate(self.buffered):
            if matches(frame):
                del self.buffered[index]
                return frame["params"]
        skipped: list[dict[str, Any]] = []
        received = list(self.buffered)
        try:
            while not matches(
                frame := await self._read_or_eof(
                    what=f"event {name or '<any>'}", received=received
                )
            ):
                skipped.append(frame)
                received.append(frame)
            return frame["params"]
        finally:
            self.buffered.extend(skipped)

    async def close(self) -> None:
        self.writer.close()
        # a connection the server closed reports its unsent bytes here
        with contextlib.suppress(ConnectionError):
            await self.writer.wait_closed()


async def connect_replay(server: ZetaServer, transport: Transport) -> ReplayClient:
    if transport == "unix":
        if server.connections is None:
            await server.start()
        reader, writer = await asyncio.open_unix_connection(server.socket_path)
        return ReplayClient(reader, writer, False)
    if server.connections is None:
        await server.start()
    assert server._websocket is not None
    reader, writer = await asyncio.open_connection("127.0.0.1", server._websocket.port)
    query = urlencode({"token": server._websocket.token})
    key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
    writer.write(
        (
            f"GET /?{query} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server._websocket.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        ).encode("ascii")
    )
    await await_or_fail(writer.drain(), what="websocket replay handshake")
    raw = await await_or_fail(reader.readuntil(b"\r\n\r\n"), what="websocket replay handshake response")
    lines = raw.decode("ascii").split("\r\n")
    assert lines[0].split()[1] == "101"
    headers = {
        name.lower(): value.strip()
        for line in lines[1:]
        if line
        for name, value in [line.split(":", 1)]
    }
    expected = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
    ).decode("ascii")
    assert headers["sec-websocket-accept"] == expected
    return ReplayClient(reader, writer, True)


def replay_server(home: Path, transport: Transport, backend_factory: Any) -> ZetaServer:
    options: dict[str, object] = {
        "home": home,
        "provider": "codex",
        "backend_factory": backend_factory,
    }
    if transport == "unix":
        options["socket_path"] = short_socket_path(home)
    else:
        options.update(port=0, ws_port=0)
    return ZetaServer(**options)


async def negotiate(
    client: ReplayClient,
    *,
    replay: bool = True,
    observer: bool = False,
    extra: tuple[str, ...] = (),
    protocol_version: str = "1.1",
) -> dict[str, Any]:
    features = (["serve_observers"] if observer else []) + (["session_replay_seq"] if replay else []) + list(extra)
    params: dict[str, object] = {"protocol_version": protocol_version, "features": features}
    if observer:
        params["role"] = "observer"
    return await client.request(1, "hello", params)


def _masked_frame(payload: bytes) -> bytes:
    size = len(payload)
    if size < 126:
        header = bytes((0x81, 0x80 | size))
    elif size <= 0xFFFF:
        header = bytes((0x81, 0x80 | 126)) + size.to_bytes(2, "big")
    else:
        header = bytes((0x81, 0x80 | 127)) + size.to_bytes(8, "big")
    mask = b"test"
    return header + mask + bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
