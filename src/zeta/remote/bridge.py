"""Hidden stdio transport for remote observer sessions."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import os
import socket
import stat
import sys
import time
from pathlib import Path
from typing import TextIO

from .descriptor import DescriptorError, descriptor_dir, validate

MAX_INPUT_BYTES = 64 * 1024
MAX_ROW_BYTES = 8192


def _machine_id(home: Path) -> str:
    path = descriptor_dir(home) / ".machine-id"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except FileNotFoundError:
        fd = None
    except OSError:
        return ""
    if fd is not None:
        try:
            info = os.fstat(fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or not stat.S_ISREG(info.st_mode):
                return ""
            value = os.read(fd, 257).decode("ascii").strip()
            return value if value and len(value) <= 256 else ""
        except (OSError, UnicodeDecodeError):
            return ""
        finally:
            os.close(fd)
    value = os.urandom(16).hex()
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            os.write(fd, value.encode("ascii"))
        finally:
            os.close(fd)
    except FileExistsError:
        return _machine_id(home)
    except OSError:
        return ""
    return value


def _rows(home: Path) -> list[dict[str, object]]:
    result = []
    directory = descriptor_dir(home)
    try:
        paths = sorted(directory.glob("*.json"))
    except OSError:
        return result
    for path in paths:
        try:
            descriptor = validate(path)
        except DescriptorError:
            continue
        row = descriptor.payload()
        row["kind"] = "session"
        row["state"] = _probe_state(descriptor)
        encoded = json.dumps(row, separators=(",", ":")).encode()
        if len(encoded) <= MAX_ROW_BYTES:
            result.append(row)
    return result


def _probe_state(descriptor) -> str | None:
    # The bridge is a separate process, so a short blocking probe cannot stall the TUI.
    sock = None
    deadline = time.monotonic() + 1.0
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(max(0.0, deadline - time.monotonic()))
        sock.connect(descriptor.socket)
        hello = {"jsonrpc": "2.0", "id": 1, "method": "hello", "params": {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers"]}}
        sock.sendall(json.dumps(hello, separators=(",", ":")).encode() + b"\n")

        data = b""

        def receive_response(request_id: int) -> dict[str, object]:
            nonlocal data
            while True:
                while b"\n" in data:
                    line, data = data.split(b"\n", 1)
                    if not line:
                        continue
                    response = json.loads(line)
                    if response.get("id") == request_id:
                        return response
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                sock.settimeout(remaining)
                chunk = sock.recv(4096)
                if not chunk:
                    raise ConnectionError("observer closed during probe")
                data += chunk
                if len(data) > MAX_INPUT_BYTES:
                    raise ValueError("observer response is too large")

        receive_response(1)
        sock.sendall(b'{"jsonrpc":"2.0","id":2,"method":"status","params":{}}\n')
        response = receive_response(2)
        result = response.get("result", {})
        if not isinstance(result, dict):
            return None
        if result.get("pending_approvals"):
            return "approval"
        state = result.get("state")
        if state == "tool":
            return "running"
        return state if state in {"idle", "running"} else None
    except (OSError, ValueError, TimeoutError, ConnectionError, json.JSONDecodeError):
        return None
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def list_descriptors(home: Path, out: TextIO = sys.stdout) -> int:
    header = {"kind": "host", "bridge_version": 1, "zeta_version": importlib.metadata.version("zeta"), "machine_id": _machine_id(home), "hostname": socket.gethostname()}
    print(json.dumps(header, separators=(",", ":")), file=out, flush=True)
    for row in _rows(home):
        print(json.dumps(row, separators=(",", ":")), file=out, flush=True)
    return 0


async def _copy(reader: asyncio.StreamReader, write) -> None:
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            break
        write(chunk)
        await asyncio.to_thread(sys.stdout.flush)


async def attach(home: Path, session_id: str) -> int:
    descriptor_file = descriptor_dir(home) / f"{session_id}.json"
    if Path(session_id).name != session_id or not session_id:
        print("zeta remote bridge: invalid session id", file=sys.stderr)
        return 2
    try:
        descriptor = await asyncio.to_thread(validate, descriptor_file, expected_session_id=session_id)
        reader, writer = await asyncio.open_unix_connection(descriptor.socket)
    except (DescriptorError, OSError, ValueError) as exc:
        print(f"zeta remote bridge: {exc}", file=sys.stderr)
        return 2

    async def stdin_to_socket() -> None:
        loop = asyncio.get_running_loop()
        fd = sys.stdin.fileno()
        pending: asyncio.Future[bytes] = loop.create_future()

        def ready() -> None:
            try:
                chunk = os.read(fd, MAX_INPUT_BYTES + 1)
            except OSError as exc:
                if not pending.done():
                    pending.set_exception(exc)
            else:
                if not pending.done():
                    pending.set_result(chunk)

        loop.add_reader(fd, ready)
        try:
            while True:
                chunk = await pending
                if not chunk:
                    return
                if len(chunk) > MAX_INPUT_BYTES:
                    raise ValueError("bridge input is too large")
                writer.write(chunk)
                await writer.drain()
                pending = loop.create_future()
        finally:
            loop.remove_reader(fd)
            if not pending.done():
                pending.cancel()

    async def socket_to_stdout() -> None:
        while True:
            chunk = await reader.read(MAX_INPUT_BYTES)
            if not chunk:
                return
            await asyncio.to_thread(sys.stdout.buffer.write, chunk)
            await asyncio.to_thread(sys.stdout.buffer.flush)

    try:
        tasks = [asyncio.create_task(stdin_to_socket()), asyncio.create_task(socket_to_stdout())]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except (BrokenPipeError, ConnectionError, ValueError) as exc:
        print(f"zeta remote bridge: {exc}", file=sys.stderr)
        return 2
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    return 0


def run(home: Path, *, session_id: str | None = None, list_mode: bool = False) -> int:
    if list_mode:
        return list_descriptors(home)
    if session_id is None:
        print("zeta remote bridge: attach requires a session id", file=sys.stderr)
        return 2
    return asyncio.run(attach(home, session_id))
