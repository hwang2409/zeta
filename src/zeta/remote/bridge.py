"""Hidden stdio transport for remote observer sessions."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import stat
import sys
from pathlib import Path
from typing import TextIO

from .descriptor import DescriptorError, descriptor_dir, validate

MAX_INPUT_BYTES = 64 * 1024
MAX_ROW_BYTES = 8192


def _machine_id(home: Path) -> str:
    path = descriptor_dir(home) / ".machine-id"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600 or not stat.S_ISREG(info.st_mode):
            raise OSError("unsafe machine id")
        value = os.read(fd, 257).decode("ascii").strip()
        os.close(fd)
    except OSError:
        value = ""
    if not value:
        value = os.urandom(16).hex()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(value, encoding="ascii")
        os.chmod(path, 0o600)
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
    import socket as _socket
    sock = None
    try:
        sock = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
        sock.settimeout(0.2)
        sock.connect(descriptor.socket)
        hello = {"id": 1, "method": "hello", "params": {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers"]}}
        sock.sendall(json.dumps(hello, separators=(",", ":")).encode() + b"\\n")
        data = b""
        while b"\\n" not in data and len(data) <= MAX_INPUT_BYTES:
            data += sock.recv(4096)
        data = data.split(b"\\n", 1)[1]
        sock.sendall(b'{"id":2,"method":"status","params":{}}\\n')
        while b"\\n" not in data and len(data) <= MAX_INPUT_BYTES:
            data += sock.recv(4096)
        response = json.loads(data.split(b"\\n", 1)[0])
        state = response.get("result", {}).get("state")
        return state if state in {"idle", "running", "approval"} else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    finally:
        if sock is not None:
            with contextlib.suppress(Exception):
                sock.close()


def list_descriptors(home: Path, out: TextIO = sys.stdout) -> int:
    header = {"kind": "host", "bridge_version": 1, "zeta_version": "unknown", "machine_id": _machine_id(home), "hostname": socket.gethostname()}
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

    loop = asyncio.get_running_loop()
    stdin_reader = asyncio.StreamReader(limit=MAX_INPUT_BYTES + 1)
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(stdin_reader), sys.stdin)

    async def stdin_to_socket() -> None:
        while True:
            try:
                chunk = await stdin_reader.readuntil(b"\\n")
            except asyncio.IncompleteReadError as exc:
                chunk = exc.partial
            except asyncio.LimitOverrunError as exc:
                raise ValueError("bridge input is too large") from exc
            if not chunk:
                return
            writer.write(chunk)
            await writer.drain()
            if not chunk.endswith(b"\\n"):
                return

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
