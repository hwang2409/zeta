"""Diagnostics for a contended serve controller slot.

Every controller ``hello`` refused with -32001 because another connection
already holds the slot writes one bounded stderr line that names both the
holder and the refused connection: transport, peer pid (unix) or remote
address (tcp/websocket) with the websocket User-Agent, the hello client
version, connection age, time since the last received frame, the running
request (or ``idle``), and whether the peer has closed. When the holder's
connection is already gone (EOF / transport lost) but its request task still
runs, the same line also carries the stuck request and the await chain its run
task is suspended on. This is diagnostic only; it changes no refusal behavior.

The line never carries tokens or secrets: only the User-Agent and remote
address, the hello client version, and the running method and id.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .connections import _OutputState
    from .server import _Client

logger = logging.getLogger("zeta.server.diagnostics")

# Bound the dump so repeated wedges cannot flood stderr.
_MAX_CHAIN_FRAMES = 40
_MAX_FRAME_CHARS = 200
_MAX_UA_CHARS = 80
_MAX_CLIENT_CHARS = 40


@dataclass(frozen=True, slots=True)
class PeerIdentity:
    """Peer facts captured once at accept; never fail if the OS withholds one."""

    transport: str  # "unix" | "tcp" | "websocket"
    pid: int | None = None
    remote: str | None = None
    user_agent: str | None = None


def _peer_pid(sock: socket.socket) -> int | None:
    """The connecting peer's pid for a unix socket, or None if unavailable."""
    try:
        if sys.platform == "darwin":
            # macOS LOCAL_PEERPID on SOL_LOCAL returns the peer pid as an int.
            return sock.getsockopt(0, 0x002)
        if sys.platform.startswith("linux"):
            creds = sock.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
            )
            pid, _uid, _gid = struct.unpack("3i", creds)
            return pid
    except OSError:
        return None
    return None


def capture_peer_identity(reader: object, writer: object) -> PeerIdentity:
    """Capture the peer's transport and address at accept, best effort."""
    if hasattr(reader, "remote_address") and hasattr(reader, "user_agent"):
        return PeerIdentity(
            "websocket",
            remote=getattr(reader, "remote_address", None),
            user_agent=getattr(reader, "user_agent", None),
        )
    sock = writer.get_extra_info("socket") if hasattr(writer, "get_extra_info") else None
    if sock is not None and sock.family == socket.AF_UNIX:
        return PeerIdentity("unix", pid=_peer_pid(sock))
    peer = writer.get_extra_info("peername") if hasattr(writer, "get_extra_info") else None
    remote = f"{peer[0]}:{peer[1]}" if isinstance(peer, tuple) and len(peer) >= 2 else None
    return PeerIdentity("tcp", remote=remote)


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "\u2026"


def _await_chain(task: asyncio.Task[object] | None) -> list[str]:
    """Walk the await chain of a suspended run task, following awaited coroutines.

    ``Task.get_stack()`` shows only the innermost frame of a suspended
    coroutine, so we follow ``cr_await`` (the coroutine a frame is awaiting)
    down the chain instead.
    """
    if task is None:
        return []
    lines: list[str] = []
    coro: object | None = task.get_coro()
    seen: set[int] = set()
    while coro is not None and len(lines) < _MAX_CHAIN_FRAMES:
        if id(coro) in seen:
            break
        seen.add(id(coro))
        frame = (
            getattr(coro, "cr_frame", None)
            or getattr(coro, "gi_frame", None)
            or getattr(coro, "ag_frame", None)
        )
        if frame is not None:
            name = frame.f_code.co_filename.rsplit("/src/", 1)[-1]
            lines.append(f"{name}:{frame.f_lineno} {frame.f_code.co_name}"[:_MAX_FRAME_CHARS])
        nxt = getattr(coro, "cr_await", None) or getattr(coro, "gi_yieldfrom", None)
        if nxt is coro:
            break
        coro = nxt
    return lines


def _output_summary(output: _OutputState | None) -> str:
    if output is None:
        return "output=none"
    writer = output.writer_task
    if writer is None:
        writer_state = "none"
    elif not writer.done():
        writer_state = "running"
    elif writer.cancelled():
        writer_state = "cancelled"
    else:
        writer_state = f"failed:{output.writer_task.exception()!r}"
    return (
        f"phase={output.phase} queue={len(output.queue)} "
        f"pending_callers={output.pending_callers} writer={writer_state}"
    )


def _identity(client: _Client, now: float) -> str:
    """One bounded, secret-free field list describing a connection."""
    peer = client._peer
    parts = [peer.transport]
    if peer.pid is not None:
        parts.append(f"pid={peer.pid}")
    if peer.remote is not None:
        parts.append(f"remote={peer.remote}")
    if peer.user_agent:
        parts.append(f'ua="{_clip(peer.user_agent, _MAX_UA_CHARS)}"')
    client_version = client._client_version
    parts.append(f"client={_clip(client_version, _MAX_CLIENT_CHARS) if client_version else '?'}")
    parts.append(f"age={now - client._connected_at:.1f}s")
    parts.append(f"last_frame={now - client._last_frame_at:.1f}s")
    request = client._request
    parts.append(f"req={request[0]}/{request[1]!r}" if request is not None else "req=idle")
    parts.append(f"peer_closed={'yes' if client.peer_closed() else 'no'}")
    return " ".join(parts)


def _lost_holder_dump(holder: _Client, output: _OutputState | None) -> str:
    """The stuck request and await chain when the holder's connection is gone."""
    request = holder._request
    if request is None:
        method, request_id, age = "<none>", None, 0.0
    else:
        method, request_id, started = request
        age = time.monotonic() - started
    chain = _await_chain(holder._run_task)
    return (
        f"holder request method={method} id={request_id!r} age={age:.1f}s; "
        f"{_output_summary(output)}; await chain: "
        f"{' <- '.join(chain) if chain else '<empty>'}"
    )


def log_controller_refusal(
    holder: _Client,
    output: _OutputState | None,
    refused: _Client,
    *,
    suppressed: int,
) -> None:
    """Write one bounded warning naming the holder and the refused connection.

    When the holder's connection is already closed while its request still
    runs, append the stuck request and its await chain.
    """
    now = time.monotonic()
    line = (
        f"serve: controller hello refused -32001 (another connection holds the slot); "
        f"holder[{_identity(holder, now)}]; refused[{_identity(refused, now)}]; "
        f"suppressed={suppressed}"
    )
    if holder.peer_closed() and holder._request is not None:
        line = f"{line}; {_lost_holder_dump(holder, output)}"
    logger.warning("%s", line)
