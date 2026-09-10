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
from . import ergonomics, login, model_selection
from .protocol import (
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    FrameCodec,
    ProtocolError,
    bounded,
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
        try:
            if self._client is not None:
                await self._client.close()
        finally:
            self._client = None
            self._client_active = False
            try:
                if self._server is not None:
                    self._server.close()
                    await self._server.wait_closed()
            finally:
                self._server = None
                try:
                    await self.runtime.close()
                finally:
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
            raise RuntimeError(
                f"refusing to replace non-socket path: {self.socket_path}"
            )
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(self.socket_path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise RuntimeError(
                    f"could not inspect socket: {self.socket_path}"
                ) from exc
        else:
            raise RuntimeError(f"a server is already listening on {self.socket_path}")
        finally:
            probe.close()
        self.socket_path.unlink()

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._client_active:
            codec = FrameCodec()
            await codec.write(
                writer,
                codec.error_response(
                    None, -32001, "another client is already connected"
                ),
            )
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
        self,
        server: ZetaServer,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.server = server
        self.reader = reader
        self.writer = writer
        self.logins = login.Logins(server.home)
        self.handshaken = False
        self.protocol_version = "1.0"
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._turn_task: asyncio.Task[None] | None = None
        self._read_buffer = bytearray()
        self.codec = FrameCodec()
        self._approval_wires: dict[str | tuple[str, str], str] = {}
        self._approval_keys: dict[str, str | tuple[str, str]] = {}

    async def run(self) -> None:
        try:
            while True:
                line = await self._read_frame()
                if not line:
                    break
                try:
                    request = self.codec.parse_request(line)
                except ProtocolError as exc:
                    await self._write(
                        self.codec.error_response(
                            exc.request_id, exc.code, exc.message, exc.data
                        )
                    )
                    if not self.handshaken:
                        break
                    continue
                request_id = request["id"]
                try:
                    result = await self._dispatch(
                        request["id"], request["method"], request["params"]
                    )
                except ProtocolError as exc:
                    await self._write(
                        self.codec.error_response(
                            exc.request_id
                            if exc.request_id is not None
                            else request_id,
                            exc.code,
                            exc.message,
                            exc.data,
                        )
                    )
                except (SessionError, ValueError) as exc:
                    await self._write(
                        self.codec.error_response(request_id, -32602, str(exc))
                    )
                except Exception as exc:  # noqa: BLE001 - keep the socket alive
                    await self._write(
                        self.codec.error_response(request_id, -32000, str(exc))
                    )
                else:
                    await self._write(self.codec.response(request_id, result))
                if not self.handshaken:
                    break
        finally:
            await self.close()

    async def _read_frame(self) -> bytes | None:
        """Read one line while retaining a bounded prefix of oversized input."""

        while True:
            newline = self._read_buffer.find(b"\n")
            if newline >= 0:
                end = newline + 1
                if end <= MAX_FRAME_BYTES:
                    line = bytes(self._read_buffer[:end])
                    del self._read_buffer[:end]
                    return line
                prefix = bytes(self._read_buffer[:MAX_FRAME_BYTES])
                del self._read_buffer[:end]
                return prefix + b"\n"
            if len(self._read_buffer) > MAX_FRAME_BYTES:
                prefix = bytes(self._read_buffer[:MAX_FRAME_BYTES])
                self._read_buffer.clear()
                return await self._discard_oversized_frame(prefix)
            chunk = await self.reader.read(65_536)
            if not chunk:
                if not self._read_buffer:
                    return None
                line = bytes(self._read_buffer)
                self._read_buffer.clear()
                return line
            self._read_buffer.extend(chunk)

    async def _discard_oversized_frame(self, prefix: bytes) -> bytes:
        """Discard an oversized line and preserve any complete later frames."""

        while True:
            chunk = await self.reader.read(65_536)
            if not chunk:
                return prefix + b"\n"
            newline = chunk.find(b"\n")
            if newline < 0:
                continue
            self._read_buffer.extend(chunk[newline + 1 :])
            return prefix + b"\n"

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
            await self.logins.close()
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()
            self._closed = True

    async def _dispatch(
        self, request_id: str | int, method: str, params: dict[str, Any]
    ) -> object:
        if method == "hello":
            return self._hello(params)
        if not self.handshaken:
            raise ProtocolError(-32002, "hello must be the first request")
        if method in login.REQUESTS:
            if self.protocol_version != "1.1" or self.server.runtime.fake_catalog:
                raise ProtocolError(-32601, "login is unavailable on this server")
            return await self.logins.request(method, params)
        if method in ergonomics.EXTENSION_REQUESTS:
            if self.protocol_version != "1.1":
                raise ProtocolError(-32601, "request requires protocol 1.1")
            return await self._ergonomics(method, params)
        if method == "list_sessions":
            return self._list_sessions(request_id)
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
        if version not in ("1.0", PROTOCOL_VERSION):
            raise ProtocolError(
                -32002,
                "unsupported protocol version",
                {"requested": version, "supported": [PROTOCOL_VERSION]},
            )
        self.protocol_version = (
            PROTOCOL_VERSION if version == PROTOCOL_VERSION or params.get("client_version") == PROTOCOL_VERSION else "1.0"
        )
        self.handshaken = True
        return {
            "protocol_version": self.protocol_version,
            "server": "zeta",
            "capabilities": {
                "requests": [
                    "list_sessions",
                    "new_session",
                    "resume",
                    "send",
                    "steer",
                    "approve",
                    "deny",
                    "abort",
                    "status",
                ] + (ergonomics.EXTENSION_REQUESTS if self.protocol_version == "1.1" else [])
                + (login.REQUESTS if self.protocol_version == "1.1" and not self.server.runtime.fake_catalog else []),
                "notifications": ["event"],
            },
        }

    async def _ergonomics(self, method: str, params: dict[str, Any]) -> object:
        runtime = self.server.runtime
        if method in {"rename_session", "delete_session"}:
            session_id = runtime.manager.resolve_id(_required_string(params, "session_id"))
            if method == "delete_session":
                await self._require_idle()
                if runtime.opened is not None and session_id == runtime.session_id:
                    raise ProtocolError(-32005, "select another session before deleting this session", {"code": "active_session"})
                runtime.manager.delete(session_id)
                return {"session_id": session_id}
            name = params.get("name")
            if not isinstance(name, str):
                raise ProtocolError(-32602, "name must be a string")
            metadata = runtime.manager.rename(session_id, name)
            if runtime.opened is not None and session_id == runtime.session_id:
                runtime.manager._copy_metadata(runtime.metadata, metadata)
            return {"session": metadata.to_dict()}
        store = ergonomics.active(runtime, params)
        if method == "session_tree":
            return ergonomics.tree(runtime)
        if method == "session_history":
            return ergonomics.history(runtime, params)
        if method == "model_catalog":
            return ergonomics.catalog(runtime)
        if method == "session_settings":
            return ergonomics.settings(runtime)
        await self._require_idle()
        ergonomics.require_mutable(runtime)
        if method == "send_images":
            message = ergonomics.image_message(runtime, params)
            self._turn_task = asyncio.create_task(self._run_turn(params.get("text", ""), message))
            return {"accepted": True, "session_id": runtime.session_id}
        if method == "set_settings":
            model = _required_string(params, "model")
            mode = _required_string(params, "approval_mode")
            if model not in ergonomics.catalog(runtime)["models"] or mode not in {"ask", "allow", "deny"}:
                raise ProtocolError(-32602, "invalid model or approval mode")
            model_selection.apply(runtime, model, mode)
            return ergonomics.settings(runtime)
        if method == "fork_message":
            store.append_message_fork(_required_string(params, "message_id"))
        elif method == "switch_branch":
            head = _required_string(params, "head_id")
            current = store.replay()
            if not current or current[-1].id != head:
                store.switch_to_branch(head)
        return ergonomics.tree(runtime)

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
            raise ProtocolError(
                -32006, f"approval request not found or already resolved: {request_id}"
            )
        active = self._turn_task is not None and not self._turn_task.done()
        if (
            not active
            and isinstance(core_key, str)
            and loop.prepare_resume_pending_tool(core_key)
        ):
            if self.server.runtime.state is not None:
                self.server.runtime.state.tool_started()
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
        if self.server.runtime.state is not None:
            self.server.runtime.state.turn_finished()
        await self._notify("turn_aborted", self.server.runtime.session_id)
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
            "state": runtime.state.status if runtime.state is not None else "idle",
            "pending_approvals": pending,
            "usage": dict(runtime.usage),
            "compaction_markers": (
                runtime.opened.store.compaction_marker_count()
                if runtime.opened is not None
                else 0
            ),
        }

    async def _run_turn(self, text: str, user_message=None) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        state.turn_started()
        try:
            async for event in loop.run_turn(text, user_message=user_message):
                await self._event(event, session_id=session_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            await self._notify(
                "error",
                session_id,
                error={"code": "server_error", "message": str(exc)},
                data={},
            )
        finally:
            if self.server.runtime.state is state:
                state.turn_finished()
            if self._turn_task is asyncio.current_task():
                self._turn_task = None

    async def _resume_tool(self, request_id: str) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        event_tasks: list[asyncio.Task[None]] = []
        try:
            await loop.resume_pending_tool(
                request_id,
                prepared=True,
                event_sink=lambda event: event_tasks.append(
                    asyncio.create_task(self._event(event, session_id=session_id))
                ),
            )
        finally:
            if event_tasks:
                await asyncio.gather(*event_tasks, return_exceptions=True)
            if self.server.runtime.state is state:
                state.turn_finished()
            if self._turn_task is asyncio.current_task():
                self._turn_task = None

    async def _event(
        self,
        event: StreamEvent,
        *,
        session_id: str | None = None,
        background: bool = False,
    ) -> None:
        state = self.server.runtime.state
        if session_id is None and not background and state is not None:
            session_id = state.session_id
        foreground = (
            not background
            and state is not None
            and session_id is not None
            and state.session_id == session_id
        )
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
                await self._notify(
                    "assistant_delta",
                    session_id,
                    delta=bounded(delta),
                    kind=stream_kind,
                )
            return
        if kind is StreamEventType.MESSAGE_END:
            if foreground and not event.data.get("truncated"):
                model_selection.confirm(self.server.runtime)
            usage = event.data.get("usage")
            if foreground and isinstance(usage, Mapping) and usage:
                state.usage.update(usage)
                await self._notify("usage", session_id, usage=dict(usage))
            if event.message is not None:
                await self._notify(
                    "assistant_message", session_id, message=event.message.to_dict()
                )
            return
        if kind is StreamEventType.TOOL_EXECUTION_UPDATE:
            await self._notify(
                "tool_output",
                session_id,
                tool_call=_tool_call(event),
                output=bounded(event.delta or _data_text(event.data)),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_START:
            if foreground:
                state.tool_started()
            await self._notify(
                "tool_start",
                session_id,
                tool_call=_tool_call(event),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_END:
            if foreground:
                state.tool_finished()
            await self._notify(
                "tool_end",
                session_id,
                tool_call=_tool_call(event),
                tool_result=(
                    event.tool_result.to_dict() if event.tool_result else None
                ),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_END:
            await self._notify(
                "approval_end",
                session_id,
                tool_call=_tool_call(event),
                data=dict(event.data),
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_START:
            if foreground:
                state.tool_started()
            raw_request_id = event.tool_call.id if event.tool_call else ""
            child_id = event.data.get("agent_instance_id")
            core_key: str | tuple[str, str] = (
                (child_id, raw_request_id)
                if isinstance(child_id, str) and raw_request_id
                else raw_request_id
            )
            await self._notify(
                "approval_request",
                session_id,
                request_id=self._wire_approval_key(core_key),
                tool_call=_tool_call(event),
            )
            return
        if kind is StreamEventType.AGENT_NOTIFICATION:
            await self._notify("sub_agent_receipt", session_id, data=dict(event.data))
            return
        if kind is StreamEventType.ERROR:
            error = (
                {"code": event.error.code, "message": event.error.message}
                if event.error else {"code": "unknown", "message": "unknown error"}
            )
            login_provider = None
            if (foreground and event.error is not None and event.error.provider_error
                    and (event.error.code == "auth_error" or event.error.status_code == 401)
                    and self.protocol_version == "1.1" and not self.server.runtime.fake_catalog):
                login_provider = self.server.runtime.metadata.provider
            if foreground and event.error is not None:
                recovered = model_selection.recover(self.server.runtime, event.error)
                if self.protocol_version == "1.1":
                    error = recovered
            await self._notify(
                "error", session_id, error=error, data={**event.data, **({"login_provider": login_provider} if login_provider else {})},
            )
            return
        if kind is StreamEventType.COMPACTION_START:
            await self._notify("compaction_start", session_id, data=dict(event.data))
            return
        if kind is StreamEventType.COMPACTION_END:
            await self._notify("compaction_end", session_id, data=dict(event.data))
            return
        if foreground and kind is StreamEventType.AGENT_END:
            state.turn_finished()
        await self._notify(kind.value, session_id, data=dict(event.data))

    def _publish_background_event(self, session_id: str, event: StreamEvent) -> None:
        """Schedule child events from the loop's synchronous sink."""

        if not self._closed:
            asyncio.create_task(
                self._event(event, session_id=session_id, background=True)
            )

    async def _notify(
        self, event: str, session_id: str | None = None, **fields: object
    ) -> None:
        if session_id is None and self.server.runtime.opened is not None:
            session_id = self.server.runtime.session_id
        await self._write(self.codec.notification(event, session_id, **fields))

    def _list_sessions(self, request_id: str | int) -> dict[str, object]:
        metadata = self.server.runtime.list_sessions()
        previews = {
            item.session_id: item.preview
            for item in self.server.runtime.manager.list_session_previews(
                limit=len(metadata), sessions=metadata
            )
        }
        sessions = [
            {**item.to_dict(), "first_message_preview": previews.get(item.session_id, "")}
            for item in metadata
        ]
        if self.protocol_version != "1.1":
            for item in sessions:
                item.pop("name", None)
        page: list[dict[str, Any]] = []
        for offset, item in enumerate(sessions):
            candidate = [*page, item]
            if not self.codec.response_fits(request_id, {"sessions": candidate}):
                return {
                    "sessions": page,
                    "truncated": True,
                    "next_offset": offset,
                }
            if (
                len(self.codec.response(request_id, {"sessions": candidate}))
                > MAX_FRAME_BYTES - 128
            ):
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

    async def _write(self, payload: bytes) -> None:
        async with self._write_lock:
            await self.codec.write(self.writer, payload)

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
