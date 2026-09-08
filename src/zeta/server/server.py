"""Newline-delimited JSON-RPC server for native zeta frontends."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import signal
import socket
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..core.approval import ApprovalDecision
from ..core.session import SessionError
from ..types import StreamEvent, StreamEventType, TextContent
from .protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    bounded,
    error_response,
    notification,
    parse_request,
    response,
)
from .runtime import BackendFactory, ServerRuntime


class ZetaServer:
    """Serve one local client over a Unix socket or localhost TCP."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        cwd: str | Path | None = None,
        socket_path: str | Path | None = None,
        port: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if socket_path is not None and port is not None:
            raise ValueError("choose socket_path or port, not both")
        if port is not None and not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.home = Path(home).expanduser().resolve() if home is not None else _home()
        self.socket_path = (
            Path(socket_path).expanduser().resolve()
            if socket_path is not None
            else self.home / "run" / "serve.sock"
        )
        self.port = port
        self.runtime = ServerRuntime(
            self.home,
            cwd=cwd,
            provider=provider,
            model=model,
            backend_factory=backend_factory,
        )
        self._server: asyncio.AbstractServer | None = None
        self._client_active = False
        self._client: _Client | None = None
        self._socket_created = False

    @property
    def address(self) -> str:
        if self.port is not None:
            return f"127.0.0.1:{self.port}"
        return str(self.socket_path)

    async def start(self) -> None:
        if self.port is None:
            self._prepare_socket_path()
            self._server = await asyncio.start_unix_server(
                self._accept, path=str(self.socket_path), limit=MAX_FRAME_BYTES + 1
            )
            os.chmod(self.socket_path, 0o600)
            self._socket_created = True
        else:
            self._server = await asyncio.start_server(
                self._accept, "127.0.0.1", self.port, limit=MAX_FRAME_BYTES + 1
            )
            bound = self._server.sockets
            if bound:
                self.port = int(bound[0].getsockname()[1])

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
        self._client_active = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self.runtime.close()
        if self.port is None and self._socket_created:
            with contextlib.suppress(FileNotFoundError):
                if stat.S_ISSOCK(self.socket_path.stat().st_mode):
                    self.socket_path.unlink()
            self._socket_created = False

    def _prepare_socket_path(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            parent_stat = self.socket_path.parent.stat()
            if parent_stat.st_uid == os.getuid():
                os.chmod(self.socket_path.parent, 0o700)
        except OSError:
            pass
        try:
            path_mode = self.socket_path.lstat().st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_mode):
            raise RuntimeError(f"refusing to replace non-socket path: {self.socket_path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(self.socket_path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise RuntimeError(f"could not inspect socket: {self.socket_path}") from exc
        else:
            raise RuntimeError(f"a server is already listening on {self.socket_path}")
        finally:
            probe.close()
        self.socket_path.unlink()

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._client_active:
            writer.write(error_response(None, -32001, "another client is already connected"))
            with contextlib.suppress(ConnectionError):
                await writer.drain()
            writer.close()
            await writer.wait_closed()
            return
        self._client_active = True
        self._client = _Client(self, reader, writer)
        self.runtime.set_background_event_sink(self._client._publish_background_event)
        try:
            await self._client.run()
        finally:
            self.runtime.set_background_event_sink(None)
            self._client = None
            self._client_active = False


class _Client:
    def __init__(
        self, server: ZetaServer, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self.server = server
        self.reader = reader
        self.writer = writer
        self.handshaken = False
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._turn_task: asyncio.Task[None] | None = None
        self.turn_state = "idle"
        self._approval_wires: dict[str | tuple[str, str], str] = {}
        self._approval_keys: dict[str, str | tuple[str, str]] = {}

    async def run(self) -> None:
        try:
            while True:
                try:
                    line = await self.reader.readline()
                except (asyncio.LimitOverrunError, ValueError) as exc:
                    await self._write(
                        error_response(None, -32600, "request frame exceeds 1048576 bytes")
                    )
                    if not self.handshaken:
                        break
                    separator_consumed = "Separator is found" in str(exc)
                    if not separator_consumed and not await self._discard_frame():
                        break
                    continue
                if not line:
                    break
                try:
                    request = parse_request(line)
                except ProtocolError as exc:
                    await self._write(error_response(None, exc.code, exc.message, exc.data))
                    if not self.handshaken:
                        break
                    continue
                request_id = request["id"]
                try:
                    result = await self._dispatch(request["method"], request["params"])
                except ProtocolError as exc:
                    await self._write(error_response(request_id, exc.code, exc.message, exc.data))
                except (SessionError, ValueError) as exc:
                    await self._write(error_response(request_id, -32602, str(exc)))
                except Exception as exc:  # noqa: BLE001 - keep the socket alive
                    await self._write(error_response(request_id, -32000, str(exc)))
                else:
                    await self._write(response(request_id, result), request_id=request_id)
                if not self.handshaken:
                    break
        finally:
            await self.close()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            policy = self.server.runtime.policy
            if policy is not None:
                for request in policy.pending_requests():
                    with contextlib.suppress(ValueError, RuntimeError):
                        policy.abort(request.key)
            if self._turn_task is not None and not self._turn_task.done():
                loop = self.server.runtime.loop
                if loop is not None:
                    loop.abort()
                self._turn_task.cancel()
                await asyncio.gather(self._turn_task, return_exceptions=True)
            self._turn_task = None
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()
            self._closed = True

    async def _discard_frame(self) -> bool:
        while True:
            byte = await self.reader.read(1)
            if not byte:
                return False
            if byte == b"\n":
                return True

    async def _dispatch(self, method: str, params: dict[str, Any]) -> object:
        if method == "hello":
            return self._hello(params)
        if not self.handshaken:
            raise ProtocolError(-32002, "hello must be the first request")
        if method == "list_sessions":
            return self._list_sessions()
        if method == "new_session":
            await self._require_idle()
            metadata = await self.server.runtime.create_session(
                provider=_optional_string(params, "provider"),
                model=_optional_string(params, "model"),
            )
            return {"session": metadata.to_dict()}
        if method == "resume":
            await self._require_idle()
            session_id = _required_string(params, "session_id")
            metadata = await self.server.runtime.resume_session(session_id)
            return {"session": metadata.to_dict()}
        if method == "send":
            return await self._send(_required_string(params, "text"))
        if method == "steer":
            return self._steer(_required_string(params, "text"))
        if method in {"approve", "deny"}:
            return await self._approval(method, _required_string(params, "request_id"))
        if method == "abort":
            return await self._abort()
        if method == "status":
            return self._status()
        raise ProtocolError(-32601, f"method not found: {method}")

    def _hello(self, params: dict[str, Any]) -> dict[str, object]:
        if self.handshaken:
            raise ProtocolError(-32600, "hello may only be sent once")
        version = params.get("protocol_version")
        if version != PROTOCOL_VERSION:
            raise ProtocolError(
                -32002,
                "unsupported protocol version",
                {"requested": version, "supported": [PROTOCOL_VERSION]},
            )
        self.handshaken = True
        return {
            "protocol_version": PROTOCOL_VERSION,
            "server": "zeta",
            "capabilities": {
                "requests": [
                    "list_sessions", "new_session", "resume", "send", "steer",
                    "approve", "deny", "abort", "status",
                ],
                "notifications": ["event"],
            },
        }

    async def _send(self, text: str) -> dict[str, object]:
        runtime = self.server.runtime
        if not text.strip():
            raise ProtocolError(-32602, "text must be a nonempty string")
        if runtime.loop is None:
            raise ProtocolError(-32003, "no active session")
        if self._turn_task is not None and not self._turn_task.done():
            raise ProtocolError(-32004, "a turn is already running")
        self._turn_task = asyncio.create_task(self._run_turn(text))
        return {"accepted": True, "session_id": runtime.session_id}

    def _steer(self, text: str) -> dict[str, object]:
        loop = self.server.runtime.loop
        if loop is None:
            raise ProtocolError(-32003, "no active session")
        if self._turn_task is None or self._turn_task.done():
            raise ProtocolError(-32005, "no turn is running")
        from ..types import Message, MessageRole

        loop.steer(Message(MessageRole.USER, [TextContent(text)]))
        return {"accepted": True}

    async def _approval(self, method: str, request_id: str) -> dict[str, object]:
        policy = self.server.runtime.policy
        loop = self.server.runtime.loop
        if policy is None or loop is None:
            raise ProtocolError(-32003, "no active session")
        core_key = self._approval_keys.get(request_id, request_id)
        resolved = policy.resolve(
            core_key,
            ApprovalDecision.ALLOW if method == "approve" else ApprovalDecision.DENY,
        )
        if not resolved:
            raise ProtocolError(-32006, f"approval request not found or already resolved: {request_id}")
        active = self._turn_task is not None and not self._turn_task.done()
        if not active and isinstance(core_key, str) and loop.prepare_resume_pending_tool(core_key):
            self.turn_state = "tool"
            task = asyncio.create_task(self._resume_tool(core_key))
            self._turn_task = task
        return {"accepted": True, "request_id": request_id, "decision": method}

    async def _abort(self) -> dict[str, object]:
        if self._turn_task is None or self._turn_task.done():
            return {"aborted": False}
        loop = self.server.runtime.loop
        if loop is not None:
            loop.abort()
        self._turn_task.cancel()
        await asyncio.gather(self._turn_task, return_exceptions=True)
        self._turn_task = None
        self.turn_state = "idle"
        await self._notify("turn_aborted")
        return {"aborted": True}

    def _status(self) -> dict[str, object]:
        runtime = self.server.runtime
        session = runtime.metadata.to_dict() if runtime.opened is not None else None
        pending = []
        if runtime.policy is not None:
            pending = [
                {
                    "request_id": self._wire_approval_key(item.key),
                    "tool_call": item.tool_call.to_dict(),
                }
                for item in runtime.policy.pending_requests()
            ]
        return {
            "session": session,
            "state": self.turn_state,
            "pending_approvals": pending,
            "usage": dict(runtime.usage),
            "compaction_markers": (
                runtime.opened.store.compaction_marker_count()
                if runtime.opened is not None
                else 0
            ),
        }

    async def _run_turn(self, text: str) -> None:
        loop = self.server.runtime.loop
        if loop is None:
            return
        self.turn_state = "running"
        try:
            async for event in loop.run_turn(text):
                await self._event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            await self._notify("error", error={"code": "server_error", "message": str(exc)})
        finally:
            self.turn_state = "idle"
            self._turn_task = None

    async def _resume_tool(self, request_id: str) -> None:
        loop = self.server.runtime.loop
        if loop is None:
            return
        try:
            await loop.resume_pending_tool(
                request_id,
                prepared=True,
                event_sink=lambda event: asyncio.create_task(self._event(event)),
            )
        finally:
            self.turn_state = "idle"
            self._turn_task = None

    async def _event(self, event: StreamEvent) -> None:
        kind = event.type
        if kind is StreamEventType.MESSAGE_UPDATE:
            delta = event.delta
            stream_kind = "assistant"
            if isinstance(event.content, TextContent):
                delta = event.content.text
            elif event.content is not None:
                delta = getattr(event.content, "text", None)
                stream_kind = str(getattr(event.content, "type", "content"))
            if delta:
                await self._notify("assistant_delta", delta=bounded(delta), kind=stream_kind)
            return
        if kind is StreamEventType.MESSAGE_END:
            usage = event.data.get("usage")
            if isinstance(usage, Mapping) and usage:
                self.server.runtime.usage.update(usage)
                await self._notify("usage", usage=dict(usage))
            if event.message is not None:
                await self._notify("assistant_message", message=event.message.to_dict())
            return
        if kind is StreamEventType.TOOL_EXECUTION_UPDATE:
            await self._notify(
                "tool_output",
                tool_call=_tool_call(event),
                output=bounded(event.delta or _data_text(event.data)),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_START:
            self.turn_state = "tool"
            await self._notify("tool_start", tool_call=_tool_call(event), data=dict(event.data))
            return
        if kind is StreamEventType.TOOL_EXECUTION_END:
            self.turn_state = "running"
            await self._notify(
                "tool_end",
                tool_call=_tool_call(event),
                tool_result=(event.tool_result.to_dict() if event.tool_result else None),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_END:
            await self._notify(
                "approval_end", tool_call=_tool_call(event), data=dict(event.data)
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_START:
            self.turn_state = "tool"
            raw_request_id = event.tool_call.id if event.tool_call else ""
            child_id = event.data.get("agent_instance_id")
            core_key: str | tuple[str, str] = (
                (child_id, raw_request_id)
                if isinstance(child_id, str) and raw_request_id
                else raw_request_id
            )
            await self._notify(
                "approval_request",
                request_id=self._wire_approval_key(core_key),
                tool_call=_tool_call(event),
            )
            return
        if kind is StreamEventType.AGENT_NOTIFICATION:
            await self._notify("sub_agent_receipt", data=dict(event.data))
            return
        if kind is StreamEventType.ERROR:
            await self._notify("error", error=event.error.to_dict() if event.error else {"code": "unknown", "message": "unknown error"}, data=dict(event.data))
            return
        if kind is StreamEventType.COMPACTION_START:
            await self._notify("compaction_start", data=dict(event.data))
            return
        if kind is StreamEventType.COMPACTION_END:
            await self._notify("compaction_end", data=dict(event.data))
            return
        await self._notify(kind.value, data=dict(event.data))

    def _publish_background_event(self, event: StreamEvent) -> None:
        """Schedule child events from the loop's synchronous sink."""

        if not self._closed:
            asyncio.create_task(self._event(event))

    async def _notify(self, event: str, **fields: object) -> None:
        session_id = self.server.runtime.opened.metadata.session_id if self.server.runtime.opened else None
        payload = notification(event, session_id, **fields)
        if len(payload) > MAX_FRAME_BYTES:
            payload = notification(
                "error",
                session_id,
                error={
                    "code": "frame_too_large",
                    "message": "outbound event exceeded 1048576 bytes",
                },
                data={"event": event},
            )
        await self._write(payload)

    def _list_sessions(self) -> dict[str, object]:
        sessions = [item.to_dict() for item in self.server.runtime.list_sessions()]
        page: list[dict[str, Any]] = []
        for offset, item in enumerate(sessions):
            candidate = [*page, item]
            if len(response(0, {"sessions": candidate})) > MAX_FRAME_BYTES - 128:
                return {
                    "sessions": page,
                    "truncated": True,
                    "next_offset": offset,
                }
            page = candidate
        return {"sessions": page}

    def _wire_approval_key(self, key: str | tuple[str, str]) -> str:
        if isinstance(key, str):
            self._approval_keys[key] = key
            return key
        wire = self._approval_wires.get(key)
        if wire is None:
            wire = f"approval-{uuid4().hex}"
            self._approval_wires[key] = wire
            self._approval_keys[wire] = key
        return wire

    async def _write(
        self, payload: bytes, *, request_id: str | int | None = None
    ) -> None:
        async with self._write_lock:
            try:
                if len(payload) > MAX_FRAME_BYTES:
                    payload = error_response(
                        request_id,
                        -32007,
                        "outbound frame exceeds 1048576 bytes",
                    )
                    if len(payload) > MAX_FRAME_BYTES:
                        payload = error_response(
                            None,
                            -32007,
                            "outbound frame exceeds 1048576 bytes",
                        )
                self.writer.write(payload)
                await self.writer.drain()
            except (ConnectionError, BrokenPipeError):
                pass

    async def _require_idle(self) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            raise ProtocolError(-32004, "a turn is already running")


def _home() -> Path:
    from ..core.session import env_home

    return env_home().expanduser().resolve()


def _required_string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


def _optional_string(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is not None and (not isinstance(value, str) or not value):
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


def _tool_call(event: StreamEvent) -> dict[str, object] | None:
    return event.tool_call.to_dict() if event.tool_call is not None else None


def _data_text(data: Mapping[str, object]) -> str:
    value = data.get("output", data.get("text", ""))
    return value if isinstance(value, str) else ""


async def run_server(server: ZetaServer) -> None:
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    installed: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except (NotImplementedError, RuntimeError):
            continue
        installed.append(signum)
    await server.start()
    serving = asyncio.create_task(server.serve_forever())
    try:
        await stopped.wait()
    finally:
        await server.close()
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        for signum in installed:
            loop.remove_signal_handler(signum)


__all__ = ["ZetaServer", "run_server"]
