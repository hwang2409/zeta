from __future__ import annotations

import asyncio
import contextlib
import errno
import fcntl
import logging
import os
import socket
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..core.session import SessionError
from ..core.store import ConversationIntegrityError, TranscriptLoadError
from ..core.transcript_recovery.runtime import TranscriptRepairRequiresUpgrade
from ..project_errors import ProjectRegistryBusyError
from ..storage_format import drain_releases, pending_releases
from . import (
    abort_scope,
    accounts,
    login,
    model_selection,
    slash_commands,
)
from . import control as session_grants
from .connections import ConnectionRegistry
from .control import NotificationPresentation, ServeControl
from .handlers import lifecycle, session, settings, validation
from .handlers.projects import (
    PROJECT_REQUEST_EXCEPTIONS,
    ProjectRequests,
    project_request_error,
)
from .protocol import (
    _ERGONOMICS_METHODS,
    _LOGIN_METHODS,
    _PROJECT_METHODS,
    FEATURES,
    MAX_FRAME_BYTES,
    ApprovalDecision,
    ErrorDescriptor,
    FrameCodec,
    FrameWriteOwner,
    MethodDescriptor,
    ProtocolError,
    _Frame,
    feature_error,
    hello_result,
    negotiate_protocol_version,
    protocol_error,
)
from .protocol import _dispatch as _dispatch_request
from .runtime import BackendFactory, ServerRuntime
from .session_replay import join_history_and_subscribe
from .transport import WebSocketServer
from .turn_context import PendingTurnContexts

logger = logging.getLogger(__name__)

# A GC finalizer only queues a lease release; an idle serve must still run it,
# or another process stays refused the session until serve's next lease call.
IDLE_DRAIN_INTERVAL = 1.0


async def _drain_released_leases() -> None:
    while True:
        await asyncio.sleep(IDLE_DRAIN_INTERVAL)
        if not pending_releases():
            continue
        try:
            # a release may wait on a registry lock another thread holds
            await asyncio.to_thread(drain_releases)
        except Exception:  # a failed drain must not stop serve
            logger.exception("draining released leases failed")


class _UnixSocketListener:
    def __init__(
        self,
        path: Path,
        server: asyncio.AbstractServer,
        lock_fd: int,
        identity: tuple[int, int],
    ) -> None:
        self.path = path
        self.server = server
        self._lock_fd = lock_fd
        self._identity = identity

    @classmethod
    async def start(
        cls,
        path: Path,
        accept: Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]],
    ) -> _UnixSocketListener:
        cls._prepare_parent(path)
        lock_fd = os.open(
            f"{path}.lock",
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        try:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"a server is already listening on {path}") from exc
            cls._prepare_path(path)
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            identity: tuple[int, int] | None = None
            try:
                unix_socket.bind(str(path))
                socket_stat = path.lstat()
                identity = (socket_stat.st_dev, socket_stat.st_ino)
                server = await asyncio.start_unix_server(
                    accept, sock=unix_socket, limit=MAX_FRAME_BYTES + 1
                )
                os.chmod(path, 0o600)
                return cls(path, server, lock_fd, identity)
            except BaseException:
                unix_socket.close()
                if identity is not None:
                    cls._unlink_if_owned(path, identity)
                raise
        except BaseException:
            os.close(lock_fd)
            raise

    async def close(self) -> None:
        try:
            self.server.close()
            await self.server.wait_closed()
            self._unlink_if_owned(self.path, self._identity)
        finally:
            os.close(self._lock_fd)

    @staticmethod
    def _prepare_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            parent_stat = path.parent.stat()
            if parent_stat.st_uid == os.getuid():
                os.chmod(path.parent, 0o700)
        except OSError:
            pass

    @classmethod
    def _prepare_path(cls, path: Path) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(path_stat.st_mode):
            raise RuntimeError(f"refusing to replace non-socket path: {path}")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(path))
        except OSError as exc:
            if exc.errno not in {errno.ECONNREFUSED, errno.ENOENT}:
                raise RuntimeError(f"could not inspect socket: {path}") from exc
        else:
            raise RuntimeError(f"a server is already listening on {path}")
        finally:
            probe.close()
        cls._unlink_if_owned(path, (path_stat.st_dev, path_stat.st_ino))

    @staticmethod
    def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError:
            return
        if (path_stat.st_dev, path_stat.st_ino) == identity:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()


class ZetaServer:
    """Serve one controller and negotiated observers over local transports."""

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
        approval_reconnect_grace: int = 90,
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
        self.approval_reconnect_grace = approval_reconnect_grace
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
        self._unix_listener: _UnixSocketListener | None = None
        self._frame_writes = FrameWriteOwner()
        self._idle_drain: asyncio.Task[None] | None = None
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
            self._unix_listener = await _UnixSocketListener.start(
                self.socket_path, self._accept
            )
            self._server = self._unix_listener.server
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
        self._idle_drain = asyncio.create_task(_drain_released_leases())

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._idle_drain is not None:
            self._idle_drain.cancel()
            await asyncio.wait([self._idle_drain])
            self._idle_drain = None
        try:
            if self._client is not None:
                await self._client.close(graceful=True)
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
                await self._frame_writes.aclose()
                try:
                    if self._websocket is not None:
                        await self._websocket.close()
                    if self._unix_listener is not None:
                        await self._unix_listener.close()
                    elif self._server is not None:
                        self._server.close()
                        await self._server.wait_closed()
                finally:
                    self._server = None
                    self._unix_listener = None

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
        self.max_role = getattr(reader, "max_role", "controller")
        self.protocol_version = "1.0"
        self.features: frozenset[str] = frozenset()
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._local_close = False
        self._controller_release: asyncio.Task[None] | None = None
        self._transport_failed = False
        self._read_buffer = bytearray()
        self.codec = FrameCodec()
        self.projects = ProjectRequests.for_runtime(
            home=server.home, runtime=server.runtime, codec=self.codec
        )

    async def run(self) -> None:
        graceful = True
        try:
            while True:
                line = await self._read_frame()
                if not line:
                    self._release_dead_connection()
                    graceful = False
                    break
                await self.connections.wait_until_settled(self)
                try:
                    request = self.codec.parse_request(line)
                except ProtocolError as exc:
                    await self._write(
                        self.codec.error_response(
                            exc.request_id, exc.descriptor, exc.message, exc.data
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
                            exc.descriptor,
                            exc.message,
                            exc.data,
                        )
                    )
                    if not self.handshaken:
                        self.connections.release(self)
                except PROJECT_REQUEST_EXCEPTIONS as exc:
                    descriptor, message, data = project_request_error(exc)
                    await self._write(
                        self.codec.error_response(request_id, descriptor, message, data)
                    )
                except ProjectRegistryBusyError as exc:
                    await self._write(
                        self.codec.error_response(
                            request_id,
                            ErrorDescriptor.PROJECT_REGISTRY_BUSY,
                            str(exc),
                            {"retryable": exc.retryable},
                        )
                    )
                except TranscriptRepairRequiresUpgrade as exc:
                    await self._write(
                        self.codec.error_response(
                            request_id,
                            ErrorDescriptor.TRANSCRIPT_REPAIR_REQUIRES_UPGRADE,
                            str(exc),
                            {
                                "home_epoch": exc.home_epoch,
                                "required_epoch": exc.required_epoch,
                            },
                        )
                    )
                except TranscriptLoadError as exc:
                    await self._write(
                        self.codec.error_response(
                            request_id,
                            ErrorDescriptor.CORRUPT_TRANSCRIPT,
                            str(exc),
                            exc.protocol_data(),
                        )
                    )
                except ConversationIntegrityError as exc:
                    await self._write(
                        self.codec.error_response(
                            request_id, ErrorDescriptor.INTERNAL_ERROR, str(exc)
                        )
                    )
                except (SessionError, ValueError) as exc:
                    await self._write(
                        self.codec.error_response(
                            request_id, ErrorDescriptor.INVALID_PARAMS, str(exc)
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep the socket alive
                    await self._write(
                        self.codec.error_response(
                            request_id, ErrorDescriptor.INTERNAL_ERROR, str(exc)
                        )
                    )
                else:
                    response = self.codec.response(request_id, result)
                    if request["method"] == "session_history_and_subscribe":
                        written = await self.connections.admit_join_response(self, response)
                    else:
                        written = await self._write(response)
                    if (
                        written
                        and request["method"] == "hello"
                        and self.role == "controller"
                        and "approval_reconnect" in self.features
                    ):
                        await self.server.control.controller_ready(reconnect=True)
                if not self.handshaken:
                    break
        finally:
            await self.close(graceful=graceful and not self._transport_failed)

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
            chunk = await self._read_chunk()
            if not chunk:
                if not self._read_buffer:
                    return None
                line = bytes(self._read_buffer)
                self._read_buffer.clear()
                return line
            self._read_buffer.extend(chunk)

    async def _read_chunk(self) -> bytes:
        try:
            return await self.reader.read(65_536)
        except ConnectionError:
            # A failed write to a gone peer stores its error in the reader; that is EOF.
            return b""

    async def _discard_oversized_frame(self, prefix: bytes) -> bytes:
        """Discard an oversized line and preserve any complete later frames."""
        while True:
            chunk = await self._read_chunk()
            if not chunk:
                return prefix + b"\n"
            newline = chunk.find(b"\n")
            if newline < 0:
                continue
            self._read_buffer.extend(chunk[newline + 1 :])
            return prefix + b"\n"

    def _begin_controller_release(self) -> asyncio.Task[None] | None:
        if self._controller_release is not None:
            return self._controller_release
        if self.role == "controller" and self.connections.is_controller(self):
            self._controller_release = self.server.control.begin_controller_release(
                reconnect="approval_reconnect" in self.features
            )
        return self._controller_release

    def _release_dead_connection(self) -> None:
        if self._local_close:
            return
        self._begin_controller_release()
        self.connections.release(self)

    async def close(self, *, graceful: bool = False) -> None:
        async with self._close_lock:
            if self._closed:
                return
            release = self._begin_controller_release()
            if release is not None:
                await asyncio.shield(release)
            await accounts.cancel_owned_switch(self)
            await self.logins.close()
            if graceful:
                await self.connections.flush(self)
            self._local_close = True
            self.writer.close()
            with contextlib.suppress(Exception):
                await self.writer.wait_closed()
            self.connections.release(self)
            self._closed = True

    async def _route_request(
        self, request_id: str | int, method: str, params: dict[str, Any]
    ) -> tuple[bool, object]:
        if method == "hello":
            result = self._hello(params)
            if self.role == "controller":
                reconnect = "approval_reconnect" in self.features
                claimed = await self.server.control.controller_claimed(
                    reconnect=reconnect,
                    require_host_binding="session_host_commands" in self.features,
                )
                if not claimed:
                    self.handshaken = False
                    raise ProtocolError(
                        ErrorDescriptor.CONNECTION_CAPACITY,
                        "another client is already connected",
                    )
                self.server._client_active = True
                self.server._client = self
                self.connections.activate_controller(self)
                if not reconnect:
                    await self.server.control.controller_ready()
            return True, result
        if not self.handshaken:
            raise ProtocolError(
                ErrorDescriptor.PROTOCOL_MISMATCH, "hello must be the first request"
            )
        assert self.server.connections is not None
        return await self.server.connections.route_request(
            self, request_id, method, params
        )

    async def _dispatch_session_grants(
        self, descriptor: MethodDescriptor, params: dict[str, Any]
    ) -> object:
        session_grants.require_feature(self.features)
        runtime = self.server.runtime
        if descriptor is MethodDescriptor.SESSION_GRANTS:
            return session_grants.read(runtime.policy, runtime.session_id, params)
        result, change = session_grants.revoke(
            runtime.policy, runtime.session_id, params
        )
        if change.changed:
            await self.server.control.publish_session_grant_change(change)
        return result

    _dispatch = _dispatch_request

    _descriptor_protocol_error = staticmethod(protocol_error)
    _descriptor_feature_error = staticmethod(feature_error)
    async def _dispatch_descriptor(
        self,
        request_id: str | int,
        descriptor: MethodDescriptor,
        params: dict[str, Any],
    ) -> object:
        method = descriptor.wire
        if descriptor in _LOGIN_METHODS:
            return await self.logins.request(method, params)
        if descriptor in _ERGONOMICS_METHODS:
            return await self._ergonomics(descriptor, params)
        if descriptor is MethodDescriptor.PING:
            if "ping" not in self.features:
                raise ProtocolError(
                    ErrorDescriptor.METHOD_NOT_FOUND,
                    "ping requires the negotiated ping feature",
                )
            return {"pong": True}
        if descriptor is MethodDescriptor.SET_TURN_CONTEXT:
            contexts = self.server.turn_contexts
            return contexts.set_request(self.features, self.server.runtime, params)
        if descriptor in _PROJECT_METHODS:
            assert self.projects is not None
            return await asyncio.to_thread(
                self.projects.dispatch,
                request_id,
                method,
                params,
                features=self.features,
            )
        if descriptor is MethodDescriptor.SESSION_HISTORY_AND_SUBSCRIBE:
            return await join_history_and_subscribe(self, params)
        if descriptor is MethodDescriptor.LIST_SESSIONS:
            result, _page_ids = await self.connections.run_read(
                self, lambda: lifecycle.list_sessions(self, request_id, params)
            )
            return result
        if descriptor is MethodDescriptor.NEW_SESSION:
            return await lifecycle.new_session(self, params)
        if descriptor is MethodDescriptor.RESUME:
            return await lifecycle.resume(self, params)
        if descriptor in {
            MethodDescriptor.SESSION_GRANTS,
            MethodDescriptor.REVOKE_SESSION_GRANTS,
        }:
            return await self._dispatch_session_grants(descriptor, params)
        if descriptor is MethodDescriptor.SEND:
            return await self.server.control.deliveries.send(params, self.features)
        if descriptor is MethodDescriptor.STEER:
            return await self.server.control.deliveries.steer(params, self.features)
        if descriptor is MethodDescriptor.DELIVERY_STATUS:
            return self.server.control.deliveries.status(params, self.features)
        if descriptor is MethodDescriptor.CLEAR_STEERING:
            return abort_scope.clear_pending_steering(
                self.features, self.server.runtime.loop
            )
        if descriptor in {MethodDescriptor.APPROVE, MethodDescriptor.DENY}:
            scope = params.get("scope", "once")
            if not isinstance(scope, str) or scope not in {"once", "always_tool"}:
                raise ProtocolError(
                    ErrorDescriptor.INVALID_PARAMS,
                    "scope must be 'once' or 'always_tool'",
                )
            if scope == "always_tool" and descriptor is not MethodDescriptor.APPROVE:
                raise ProtocolError(
                    ErrorDescriptor.INVALID_PARAMS,
                    "scope 'always_tool' requires approve",
                )
            decision = (
                ApprovalDecision.APPROVE
                if descriptor is MethodDescriptor.APPROVE
                else ApprovalDecision.DENY
            )
            return await self.server.control.resolve_approval(
                decision, validation.required_string(params, "request_id"), scope
            )
        if descriptor is MethodDescriptor.ABORT:
            return await self.server.control.abort(
                abort_scope.parse_abort_scope(params, self.features)
            )
        if descriptor is MethodDescriptor.STATUS:
            assert self.server.connections is not None
            session_id = (
                self.server.runtime.session_id
                if self.server.runtime.opened is not None
                else None
            )
            return await self.server.connections.read_status(
                self,
                await lifecycle.status(
                    self,
                    include_connection_setup_counts=(
                        params.get("include_connection_setup_counts") is True
                    )
                ),
                session_id,
            )
        raise ProtocolError(
            ErrorDescriptor.METHOD_NOT_FOUND, f"method not found: {method}"
        )

    def _hello(self, params: dict[str, Any]) -> dict[str, object]:
        if self.handshaken:
            raise ProtocolError(
                ErrorDescriptor.INVALID_REQUEST, "hello may only be sent once"
            )
        version = params.get("protocol_version")
        client_version = params.get("client_version")
        self.protocol_version = negotiate_protocol_version(version, client_version)
        requested = params.get("features")
        if requested is not None and (
            not isinstance(requested, list)
            or not all(isinstance(item, str) for item in requested)
        ):
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS, "features must be an array of strings"
            )
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
            max_role=self.max_role,
        )
        self.handshaken = True
        requests = MethodDescriptor.advertised(
            self.protocol_version, self.features, role
        )
        if extended:
            self.server.turn_contexts.add_request(requests, self.features)
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

    async def _ergonomics(
        self, descriptor: MethodDescriptor, params: dict[str, Any]
    ) -> object:
        if isinstance(descriptor, str):
            descriptor = MethodDescriptor.for_wire(descriptor)
            if descriptor is None:
                raise ProtocolError(
                    ErrorDescriptor.METHOD_NOT_FOUND, "method not found"
                )
        runtime = self.server.runtime
        method = descriptor.wire
        if descriptor is MethodDescriptor.REPAIR_TRANSCRIPT:
            return await session.repair_transcript(self, params)
        if descriptor in {
            MethodDescriptor.RENAME_SESSION,
            MethodDescriptor.DELETE_SESSION,
        }:

            return await session.mutate_session(
                runtime,
                method,
                validation.required_string(params, "session_id"),
                params.get("name"),
                self.server.control.require_idle,
            )
        if descriptor in {
            MethodDescriptor.SESSION_TREE,
            MethodDescriptor.SESSION_HISTORY,
        }:
            assert self.server.connections is not None
            return await self.server.connections.read_session(self, method, params)
        if descriptor is MethodDescriptor.CONTENT_READ:
            if "content_read" not in self.features:
                raise ProtocolError(
                    ErrorDescriptor.METHOD_NOT_FOUND,
                    "content_read requires the negotiated content_read feature",
                )
            assert self.server.connections is not None
            return await self.server.connections.read_session(self, method, params)
        if descriptor is MethodDescriptor.MODEL_CATALOG:
            return settings.catalog(runtime)
        store = validation.active(runtime, params)
        if descriptor is MethodDescriptor.SEND_IMAGES:
            return await self.server.control.deliveries.send_images(
                params, self.features
            )
        if descriptor is MethodDescriptor.SESSION_SETTINGS:
            return settings.settings(runtime, include_plan_mode=True)
        if descriptor in {
            MethodDescriptor.ACCOUNT_LIST,
            MethodDescriptor.ACCOUNT_LOGOUT,
            MethodDescriptor.ACCOUNT_REFRESH,
            MethodDescriptor.ACCOUNT_SWITCH,
            MethodDescriptor.SLASH_LIST,
        }:
            return await slash_commands.read_request(
                method, runtime, self.server.control.require_idle, params, self
            )
        if descriptor is MethodDescriptor.SLASH_RUN:
            text = validation.required_string(params, "text")
            registry = await slash_commands.registry_for_request(runtime)
            if self.server.control.turn_busy() and not slash_commands.can_run_while_busy(
                registry, text
            ):
                await self.server.control.require_idle()
            register = self.server.control.model_input_registrar(self.features)
            return await slash_commands.run_command(
                runtime,
                text,
                register_model_input=register,
                approval_control=self.server.control,
                slash_lock=self.server.control.slash_lock,
                busy=self.server.control.turn_busy(),
                registry=registry,
            )
        await self.server.control.require_idle()
        validation.require_mutable(runtime, self.server.control.has_pending_approvals())
        if descriptor is MethodDescriptor.SET_SETTINGS:
            model = validation.required_string(params, "model")
            mode = validation.required_string(params, "approval_mode")
            model_selection.validate(model, mode)
            await model_selection.apply_rpc(runtime, model, mode)
            return settings.settings(runtime)
        if descriptor is MethodDescriptor.SET_COMPACTION:
            validation.required_string(params, "mode")
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "compaction mode selection was removed; eviction is always used",
            )
        if descriptor is MethodDescriptor.FORK_MESSAGE:
            store.append_message_fork(validation.required_string(params, "message_id"))
        elif descriptor is MethodDescriptor.SWITCH_BRANCH:
            head = validation.required_string(params, "head_id")
            current = store.replay()
            if not current or current[-1].id != head:
                store.switch_to_branch(head)
        return session.tree(runtime)

    async def _write(self, payload: _Frame) -> bool:
        try:
            written = await self.connections.send(self, payload)
        except OSError:
            written = False
        if not written:
            self._transport_failed = True
        return written

    async def _write_physical(self, payload: _Frame) -> bool:
        return await self.server._frame_writes.write(
            self.codec, self.writer, payload, self._write_lock
        )


def _home() -> Path:
    from ..core.session import env_home

    return env_home().expanduser().resolve()


__all__ = ["ZetaServer"]
