"""Hidden stdio transport for remote observer sessions."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import sys
from pathlib import Path
from typing import TextIO

from .descriptor import DescriptorError, descriptor_dir, validate

MAX_INPUT_BYTES = 64 * 1024
MAX_ROW_BYTES = 8192


def _machine_id(home: Path) -> str:
    path = descriptor_dir(home) / ".machine-id"
    try:
        value = path.read_text(encoding="ascii").strip()
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
        sock.sendall(b'{"id":2,"method":"status","params":{}}\\n')
        while b"\\n" not in data:
            data += sock.recv(4096)
        response = json.loads(data.split(b"\\n", 1)[-1])
        session = response.get("result", {}).get("session", {})
        state = session.get("status") or session.get("state")
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

    async def stdin_to_socket() -> None:
        while True:
            chunk = await asyncio.to_thread(sys.stdin.buffer.read, 65536)
            if not chunk:
                writer.close()
                return
            if len(chunk) > MAX_INPUT_BYTES:
                raise ValueError("bridge input is too large")
            writer.write(chunk)
            await writer.drain()

    async def socket_to_stdout() -> None:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return
            sys.stdout.buffer.write(chunk)
            await asyncio.to_thread(sys.stdout.buffer.flush)

    try:
        await asyncio.gather(stdin_to_socket(), socket_to_stdout())
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
