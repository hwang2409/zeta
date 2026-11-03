from __future__ import annotations

import asyncio
import contextlib
import errno
import os
import socket
import stat
from pathlib import Path
from typing import Any

from ..core.session import SessionError, SessionNotFoundError
from ..project_errors import ProjectRegistryBusyError
from . import (
    abort_scope,
    accounts,
    ergonomics,
    login,
    model_selection,
    slash_commands,
)
from .connections import ConnectionRegistry
from .control import NotificationPresentation, ServeControl
from .project_requests import (
    PROJECT_REQUEST_EXCEPTIONS,
    PROJECT_REQUESTS,
    ProjectRequests,
    project_request_error,
)
from .protocol import (
    FEATURES,
    MAX_FRAME_BYTES,
    FrameCodec,
    ProtocolError,
    finish_frame_write,
    hello_result,
    negotiate_protocol_version,
)
from .runtime import BackendFactory, ServerRuntime
from .transport import WebSocketServer
from .turn_context import PendingTurnContexts


class ZetaServer:
    """Serve one local client over a Unix socket or localhost TCP."""

    def __init__(
        self,
        *,
        home: str | Path | None = None,
        cwd: str | Path | None = None,
        socket_path: str | Path | None = None,
        port: int | None = None,
        ws_port: int | None = None,
        ws_origins: tuple[str, ...] = (),
        provider: str | None = None,
        model: str | None = None,
        tools: str | None = None,
        disallowed_tools: str | None = None,
        require_tools: bool = False,
        allow_hooks: bool | None = None,
        auto_memory: bool | None = None,
        cli_yolo: bool | None = None,
        max_turns: int | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        if socket_path is not None and port is not None:
            raise ValueError("choose socket_path or port, not both")
        if port is not None and not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if ws_port is None and ws_origins:
            raise ValueError("ws_origins requires ws_port")
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
            tools=tools,
            disallowed_tools=disallowed_tools,
            require_tools=require_tools,
            allow_hooks=allow_hooks,
            auto_memory=auto_memory,
            cli_yolo=cli_yolo,
            max_turns=max_turns,
            backend_factory=backend_factory,
        )
        self._server: asyncio.AbstractServer | None = None
        self._websocket = (
            WebSocketServer(
                port=ws_port,
                origins=ws_origins,
                path=self.home / "run" / "serve.ws.json",
                accept=self._accept,
            )
            if ws_port is not None
            else None
        )
        self._client_active = False
        self._client: _Client | None = None
        self.connections: ConnectionRegistry | None = None
        self._account_warnings: tuple[str, ...] = ()
        self._socket_created = False
        self.notification_presentations = NotificationPresentation()
        self.turn_contexts = PendingTurnContexts()
        self.control = ServeControl(self)
        self.runtime.set_approval_control_binder(self.control.bind_policy)
        self.runtime.set_background_event_sink(self.control._publish_background_event)
        self.runtime.set_memory_notice_sink(self.control._publish_memory_notice)
        self.runtime.set_background_wake_sink(self.control._schedule_background_wake)

    @property
    def address(self) -> str:
        if self.port is not None:
            return f"127.0.0.1:{self.port}"
        return str(self.socket_path)

    async def start(self) -> None:
        await self.runtime.initialize_project_registry()
        self._account_warnings = await accounts.migrate(self.home)
        self.connections = ConnectionRegistry(asyncio.get_running_loop())
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
        if self._websocket is not None:
            try:
                await self._websocket.start()
            except BaseException:
                await self.close()
                raise

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
                await self.control.close()
                try:
                    await self.runtime.close()
                finally:
                    await self.control.wait_for_approval_settlement()
            finally:
                if self.connections is not None:
                    await self.connections.aclose()
                    self.connections = None
                try:
                    if self._websocket is not None:
                        await self._websocket.close()
                    if self._server is not None:
                        self._server.close()
                        await self._server.wait_closed()
                finally:
                    self._server = None
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
        registry = self.connections
        assert registry is not None
        client = _Client(self, reader, writer)
        try:
            registry.register(client)
        except RuntimeError:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            return
        try:
            await client.run()
        finally:
            if self._client is client:
                self._client = None
                self._client_active = False
            registry.release(client)


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
        if server.connections is None:
            server.connections = ConnectionRegistry(asyncio.get_running_loop())
            server.connections.register(self)
        self.connections = server.connections
        self.logins = login.Logins(server.home)
        self.handshaken = False
        self.role: str | None = None
        self.protocol_version = "1.0"
        self.features: frozenset[str] = frozenset()
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._read_buffer = bytearray()
        self.codec = FrameCodec()
        self.projects = ProjectRequests.for_runtime(
            home=server.home, runtime=server.runtime, codec=self.codec
        )

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
                        request_id, request["method"], request["params"]
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
                except PROJECT_REQUEST_EXCEPTIONS as exc:
                    code, message, data = project_request_error(exc)
                    await self._write(
                        self.codec.error_response(request_id, code, message, data)
                    )
                except ProjectRegistryBusyError as exc:
                    await self._write(
                        self.codec.error_response(
                            request_id,
                            -32000,
                            str(exc),
                            {"code": exc.code, "retryable": exc.retryable},
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
            controller = self.connections.current_controller()
            if (
                self.role == "controller"
                and controller is not None
                and controller.connection is self
            ):
                await self.server.control.controller_released()
            await accounts.cancel_owned_switch(self)
            await self.logins.close()
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()
            self._closed = True

    async def _route_request(
        self, method: str, params: dict[str, Any]
    ) -> tuple[bool, object]:
        if method == "hello":
            result = self._hello(params)
            if self.role == "controller":
                self.server._client_active = True
                self.server._client = self
                await self.server.control.controller_claimed()
            return True, result
        if not self.handshaken:
            raise ProtocolError(-32002, "hello must be the first request")
        assert self.server.connections is not None
        return await self.server.connections.route_request(self, method, params)

    async def _dispatch(
        self, request_id: str | int, method: str, params: dict[str, Any]
    ) -> object:
        handled, result = await self._route_request(method, params)
        if handled:
            return result
        if method in login.REQUESTS:
            if self.protocol_version == "1.0":
                raise ProtocolError(-32601, "login is unavailable on this server")
            return await self.logins.request(method, params)
        if method in ergonomics.EXTENSION_REQUESTS:
            if self.protocol_version == "1.0":
                raise ProtocolError(-32601, "request requires protocol 1.1")
            return await self._ergonomics(method, params)
        if method == "ping":
            if "ping" not in self.features:
                raise ProtocolError(-32601, "ping requires the negotiated ping feature")
            return {"pong": True}
        if method == "set_turn_context":
            contexts = self.server.turn_contexts
            return contexts.set_request(self.features, self.server.runtime, params)
        if method in PROJECT_REQUESTS:
            if "projects" not in self.features:
                raise ProtocolError(
                    -32601, f"{method} requires the negotiated projects feature"
                )
            assert self.projects is not None
            return await asyncio.to_thread(
                self.projects.dispatch,
                request_id,
                method,
                params,
                features=self.features,
            )
        if method == "list_sessions":
            return self._list_sessions(request_id, params)
        if method == "new_session":
            await self.server.control.require_idle()
            cwd = None
            if "cwd" in params:
                self._require_feature("session_cwd", "cwd")
                cwd = _directory(params, "cwd")
            await self.server.control._terminate_pending_approvals()
            self.server.control.session_replaced()
            await self.server.runtime.create_session(
                provider=_optional_string(params, "provider"),
                model=_optional_string(params, "model"),
                cwd=cwd,
            )
            self.server.control.clear_approvals()
            await self.server.control.controller_claimed()
            return {"session": self._session_snapshot()}
        if method == "resume":
            await self.server.control.require_idle()
            session_id = _required_string(params, "session_id")
            await self.server.control._terminate_pending_approvals()
            self.server.control.session_replaced()
            try:
                await self.server.runtime.resume_session(session_id)
            except SessionNotFoundError as exc:
                raise ProtocolError(
                    -32602,
                    str(exc),
                    {"code": "session_not_found", "session_id": session_id},
                ) from exc
            self.server.control.clear_approvals()
            await self.server.control.controller_claimed()
            return {"session": self._session_snapshot()}
        if method == "send":
            return await self.server.control.deliveries.send(params, self.features)
        if method == "steer":
            return await self.server.control.deliveries.steer(params, self.features)
        if method == "delivery_status":
            return self.server.control.deliveries.status(params, self.features)
        if method == "clear_steering":
            return abort_scope.clear_pending_steering(
                self.features, self.server.runtime.loop
            )
        if method in {"approve", "deny"}:
            scope = params.get("scope", "once")
            if not isinstance(scope, str) or scope not in {"once", "always_tool"}:
                raise ProtocolError(-32602, "scope must be 'once' or 'always_tool'")
            if scope == "always_tool" and method != "approve":
                raise ProtocolError(-32602, "scope 'always_tool' requires approve")
            return await self.server.control.resolve_approval(
                method, _required_string(params, "request_id"), scope
            )
        if method == "abort":
            return await self.server.control.abort(
                abort_scope.parse_abort_scope(params, self.features)
            )
        if method == "status":
            return await self._status()
        raise ProtocolError(-32601, f"method not found: {method}")

    def _hello(self, params: dict[str, Any]) -> dict[str, object]:
        if self.handshaken:
            raise ProtocolError(-32600, "hello may only be sent once")
        version = params.get("protocol_version")
        client_version = params.get("client_version")
        self.protocol_version = negotiate_protocol_version(version, client_version)
        requested = params.get("features")
        if requested is not None and (
            not isinstance(requested, list)
            or not all(isinstance(item, str) for item in requested)
        ):
            raise ProtocolError(-32602, "features must be an array of strings")
        extended = self.protocol_version != "1.0"
        accepted = (
            [item for item in FEATURES if item in (requested or ())] if extended else []
        )
        accepted = self.server.runtime.available_project_features(accepted)
        if "projects-memory-v2" in accepted and "projects" not in accepted:
            accepted.remove("projects-memory-v2")
        self.features = frozenset(accepted)
        assert self.server.connections is not None
        role = self.server.connections.negotiate_role(
            self,
            params.get("role", "controller"),
            self.protocol_version,
            self.features,
        )
        self.handshaken = True
        requests = [
            "list_sessions",
            "new_session",
            "resume",
            "send",
            "steer",
            "approve",
            "deny",
            "abort",
            "status",
        ]
        if extended:
            requests += ergonomics.EXTENSION_REQUESTS
            requests += login.REQUESTS
            if "ping" in self.features:
                requests.append("ping")
            self.server.turn_contexts.add_request(requests, self.features)
            if "projects" in self.features:
                requests += PROJECT_REQUESTS
            if "abort_scope" in self.features:
                requests.append("clear_steering")
            if "delivery_id" in self.features:
                requests.append("delivery_status")
        requests = self.server.connections.requests_for_role(role, requests)
        capabilities: dict[str, object] = {
            "requests": requests,
            "notifications": ["event"],
        }
        if extended and requested is not None:
            capabilities["features"] = accepted
        result = hello_result(
            self.protocol_version,
            capabilities,
            self.server.runtime.project_registry_warning,
        )
        if self.server._account_warnings:
            notices = result.setdefault("notices", [])
            assert isinstance(notices, list)
            notices.extend(self.server._account_warnings)
        return result

    def _require_feature(self, feature: str, param: str) -> None:
        if feature not in self.features:
            raise ProtocolError(
                -32602, f"{param} requires the negotiated {feature} feature"
            )

    async def _ergonomics(self, method: str, params: dict[str, Any]) -> object:
        runtime = self.server.runtime
        if method in {"rename_session", "delete_session"}:
            return await ergonomics.mutate_session(
                runtime,
                method,
                _required_string(params, "session_id"),
                params.get("name"),
                self.server.control.require_idle,
            )
        store = ergonomics.active(runtime, params)
        if method == "session_tree":
            return ergonomics.tree(runtime)
        if method == "session_history":
            return ergonomics.history(runtime, params)
        if method == "model_catalog":
            return ergonomics.catalog(runtime)
        if method == "session_settings":
            return ergonomics.settings(runtime)
        if method in slash_commands.READ_REQUESTS:
            return await slash_commands.read_request(
                method, runtime, self.server.control.require_idle, params, self
            )
        await self.server.control.require_idle()
        if method == "slash_run":
            register = self.server.control.model_input_registrar(self.features)
            return await slash_commands.run_command(
                runtime,
                _required_string(params, "text"),
                register_model_input=register,
                approval_control=self.server.control,
            )
        ergonomics.require_mutable(runtime, self.server.control.has_pending_approvals())
        if method == "send_images":
            message = ergonomics.image_message(runtime, params)
            await self.server.control.user_message(
                params.get("text", ""),
                "send",
                self.features,
                ergonomics.attachment_summary(message),
            )
            self.server.control.start_turn(params.get("text", ""), message)
            return {"accepted": True, "session_id": runtime.session_id}
        if method == "set_settings":
            model = _required_string(params, "model")
            mode = _required_string(params, "approval_mode")
            if model not in ergonomics.catalog(runtime)["models"] or mode not in {
                "ask",
                "allow",
                "deny",
            }:
                raise ProtocolError(-32602, "invalid model or approval mode")
            await model_selection.apply_rpc(runtime, model, mode)
            return ergonomics.settings(runtime)
        if method == "set_compaction":
            _required_string(params, "mode")
            raise ProtocolError(
                -32602,
                "compaction mode selection was removed; eviction is always used",
            )
        if method == "fork_message":
            store.append_message_fork(_required_string(params, "message_id"))
        elif method == "switch_branch":
            head = _required_string(params, "head_id")
            current = store.replay()
            if not current or current[-1].id != head:
                store.switch_to_branch(head)
        return ergonomics.tree(runtime)

    def _session_snapshot(self) -> dict[str, object] | None:
        runtime = self.server.runtime
        if runtime.opened is None:
            return None
        snapshot = runtime.metadata.to_dict()
        if runtime.policy is not None:
            snapshot["approval_mode"] = runtime.policy.default.value
        return snapshot

    async def _status(self) -> dict[str, object]:
        runtime = self.server.runtime
        session = self._session_snapshot()
        pending = (
            await self.server.control.pending_approvals() if self.role != "observer" else []
        )
        return {
            "session": session,
            "state": runtime.state.status if runtime.state is not None else "idle",
            "explicit_finish": (
                runtime.loop.explicit_finish
                if runtime.loop is not None
                else runtime._config(None, None).agents_explicit_finish
            ),
            "pending_approvals": pending,
            "usage": dict(runtime.usage),
            "compaction_markers": (
                runtime.opened.store.compaction_marker_count()
                if runtime.opened is not None
                else 0
            ),
        }

    def _list_sessions(
        self, request_id: str | int, params: dict[str, Any]
    ) -> dict[str, object]:
        paging = "list_sessions_paging" in self.features
        for name in ("offset", "limit"):
            if name in params:
                self._require_feature("list_sessions_paging", name)
        offset = _integer(params, "offset", 0, minimum=0)
        limit = _integer(params, "limit", None, minimum=1)
        if "project_id" in params:
            self._require_feature("projects", "project_id")
            assert self.projects is not None
            available = self.projects.project_sessions(params["project_id"])
        else:
            available = (
                self.server.runtime.list_sessions_read_only()
                if self.role == "observer"
                else self.server.runtime.list_sessions()
            )
        metadata = available[offset : None if limit is None else offset + limit]
        previews = {
            item.session_id: item.preview
            for item in self.server.runtime.manager.list_session_previews(
                limit=len(metadata), sessions=metadata
            )
        }
        sessions = [
            {
                **item.to_dict(),
                "first_message_preview": previews.get(item.session_id, ""),
            }
            for item in metadata
        ]
        if self.protocol_version == "1.0":
            for item in sessions:
                item.pop("name", None)
        page: list[dict[str, Any]] = []
        for index, item in enumerate(sessions):
            candidate = [*page, item]
            if not self.codec.response_fits(request_id, {"sessions": candidate}) or (
                len(self.codec.response(request_id, {"sessions": candidate}))
                > MAX_FRAME_BYTES - 128
            ):
                return {
                    "sessions": page,
                    "truncated": True,
                    "next_offset": offset + index,
                }
            page = candidate
        if not paging:
            return {"sessions": page}
        end = offset + len(page)
        return {"sessions": page, "next_offset": end if end < len(available) else None}

    async def _write(self, payload: bytes) -> bool:
        return await self.connections.send(self, payload)

    async def _write_physical(self, payload: bytes) -> bool:
        async with self._write_lock:
            return await finish_frame_write(self.codec, self.writer, payload)


def _home() -> Path:
    from ..core.session import env_home

    return env_home().expanduser().resolve()


def _required_string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


def _integer(
    params: dict[str, Any], name: str, default: int | None, *, minimum: int
) -> int | None:
    if name not in params:
        return default
    value = params[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ProtocolError(-32602, f"{name} must be an integer of at least {minimum}")
    return value


def _directory(params: dict[str, Any], name: str) -> Path:
    value = params.get(name)
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ProtocolError(-32602, f"{name} must be an absolute directory path")
    path = Path(value).resolve()
    if not path.is_dir():
        raise ProtocolError(-32602, f"{name} is not an existing directory: {value}")
    return path


def _optional_string(params: dict[str, Any], name: str) -> str | None:
    value = params.get(name)
    if value is not None and (not isinstance(value, str) or not value):
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


__all__ = ["ZetaServer"]
