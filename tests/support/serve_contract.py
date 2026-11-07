"""Public-wire client for running one serve contract across real transports."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Literal

from websockets.asyncio.client import ClientConnection, connect

Transport = Literal["unix", "websocket"]
Role = Literal["controller", "observer"]


class ServeClient:
    def __init__(
        self,
        reader: asyncio.StreamReader | None = None,
        writer: asyncio.StreamWriter | None = None,
        websocket: ClientConnection | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._websocket = websocket
        self._events: deque[dict[str, Any]] = deque()

    async def request(
        self,
        request_id: str,
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        frame = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        await self._send(frame)
        while True:
            reply = await self._receive()
            if reply.get("id") == request_id:
                return reply
            self._events.append(reply)

    async def event(self, name: str) -> dict[str, Any]:
        while True:
            frame = self._events.popleft() if self._events else await self._receive()
            params = frame.get("params", {})
            if params.get("event") == name:
                return params

    async def close(self) -> None:
        if self._websocket is not None:
            await self._websocket.close()
        if self._writer is not None:
            self._writer.close()
            await self._writer.wait_closed()

    async def _send(self, frame: dict[str, object]) -> None:
        payload = json.dumps(frame, separators=(",", ":"))
        if self._websocket is not None:
            await self._websocket.send(payload)
            return
        assert self._writer is not None
        self._writer.write((payload + "\n").encode())
        await self._writer.drain()

    async def _receive(self) -> dict[str, Any]:
        if self._websocket is not None:
            payload = await self._websocket.recv()
            assert isinstance(payload, str), "serve sent a binary WebSocket frame"
        else:
            assert self._reader is not None
            line = await self._reader.readline()
            assert line, "serve closed the Unix socket"
            payload = line.decode()
        frame = json.loads(payload)
        assert isinstance(frame, dict)
        return frame


class ServeContract:
    def __init__(
        self,
        process: asyncio.subprocess.Process,
        home: Path,
        socket_path: Path,
        transport: Transport,
    ) -> None:
        self._process = process
        self._home = home
        self._socket_path = socket_path
        self._transport = transport
        self._clients: list[ServeClient] = []

    @classmethod
    async def start(cls, root: Path, transport: Transport) -> ServeContract:
        token = hashlib.sha256(str(root).encode()).hexdigest()[:16]
        socket_path = Path("/tmp") / f"zeta-contract-{token}.sock"
        home = root / "home"
        environment = os.environ.copy()
        environment["ZETA_HOME"] = str(home)
        environment["ZETA_TEST_SCRIPTED_PROVIDER"] = "1"
        entrypoint = Path(sys.executable).with_name("zeta")
        assert entrypoint.is_file(), (
            f"zeta entrypoint not found beside {sys.executable}"
        )
        process = await asyncio.create_subprocess_exec(
            entrypoint,
            "serve",
            "--socket",
            str(socket_path),
            "--ws",
            "0",
            "--provider",
            "codex",
            "--cwd",
            str(root),
            "--no-auto-memory",
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert process.stderr is not None
            output = bytearray()
            while True:
                line = await process.stderr.readline()
                output.extend(line)
                assert line, (
                    "zeta serve exited before readiness: "
                    f"{output.decode(errors='replace')}"
                )
                if line.startswith(b"zeta serve: websocket ready:"):
                    break
        except BaseException as error:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
            try:
                await process.communicate()
            except BaseException as cleanup_error:  # noqa: BLE001
                error.add_note(f"process reap failed during cleanup: {cleanup_error!r}")
            try:
                socket_path.unlink(missing_ok=True)
            except BaseException as cleanup_error:  # noqa: BLE001
                error.add_note(
                    f"socket removal failed during cleanup: {cleanup_error!r}"
                )
            raise
        return cls(process, home, socket_path, transport)

    async def connect(self, role: Role = "controller") -> ServeClient:
        if self._transport == "unix":
            reader, writer = await asyncio.open_unix_connection(str(self._socket_path))
            client = ServeClient(reader=reader, writer=writer)
        else:
            discovery = json.loads((self._home / "run" / "serve.ws.json").read_text())
            token = discovery[f"{role}_token"]
            websocket = await connect(
                f"ws://127.0.0.1:{discovery['port']}/",
                subprotocols=["zeta.v1", f"zeta.bearer.{token}"],
            )
            client = ServeClient(websocket=websocket)
        self._clients.append(client)
        return client

    async def close(self) -> None:
        try:
            for client in reversed(self._clients):
                await client.close()
        except BaseException as error:
            await self._cleanup_process(error)
            raise
        await self._cleanup_process()

    async def _cleanup_process(
        self, original_error: BaseException | None = None
    ) -> None:
        cleanup_errors: list[BaseException] = []
        try:
            if self._process.returncode is None:
                self._process.terminate()
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(error)

        stdout = stderr = b""
        try:
            stdout, stderr = await self._process.communicate()
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(error)

        try:
            assert self._process.returncode == 0, (stdout + stderr).decode(
                errors="replace"
            )
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(error)
        try:
            assert not self._socket_path.exists(), "serve left its Unix socket behind"
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(error)

        if original_error is not None:
            for error in cleanup_errors:
                original_error.add_note(f"server cleanup failed: {error!r}")
            return
        if cleanup_errors:
            error, *additional_errors = cleanup_errors
            for additional_error in additional_errors:
                error.add_note(f"additional cleanup failure: {additional_error!r}")
            raise error
